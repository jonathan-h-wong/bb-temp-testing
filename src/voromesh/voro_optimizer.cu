// 2D VoroMesh optimisation and extraction in CUDA.
//
// Translation of the reference PyTorch code (github.com/nissmar/VoroMesh,
// src/utils/custom_voronoi.py: Voroloss_opt / train_voronoi, and
// src/single_voromesh.py) with the autograd replaced by an analytic gradient.
//
// Per iteration (captured once into a CUDA graph and replayed):
//   1. Bin generators into a uniform hash grid (count, exclusive scan, scatter).
//   2. One thread per (subsampled) target point: k nearest generators via ring
//      search, VoroLoss, and atomically accumulated gradients for the two
//      generators defining the closest bisector.
//   3. One thread per generator: Adam step with the MultiStepLR schedule
//      evaluated on the device, then clamp to the domain.
// Everything that changes between runs (schedule, active slot rectangle,
// target box) lives in a device-side RunParams, so the graph stays valid.
// Generators and targets are slot arrays; slots with x < 0 are empty and
// skipped by every kernel.
//
// Extraction: each thread clips the domain box by the bisectors of its
// generator's k nearest neighbours (in double precision, relative to the
// generator), certified by the security radius |cj - ci| > 2 max|v - ci|.
// Uncertified cells fall back to clipping against every generator. Each
// polygon edge remembers the neighbour that produced it, so every vertex can
// be keyed by the generator triple it is equidistant to; the boundary mesh is
// assembled on those keys, which makes it watertight by construction.

#include "voromesh/voro_optimizer.h"

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>

static_assert(sizeof(voromesh::Vec2) == sizeof(float2), "Vec2 must mirror float2");

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err__), __FILE__, __LINE__);             \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

namespace {

constexpr int kLossK        = 12; // neighbours per target point (paper: 16 in 3D)
constexpr int kCellK        = 32; // neighbours tried per cell before the brute-force fallback
constexpr int kMaxCellVerts = 24;
constexpr int kBlock        = 128;
// Edge labels: >= 0 is the neighbouring generator, -1..-4 the domain box sides.
constexpr int kBoxLabels    = 4;

__device__ __forceinline__ float2 operator+(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
__device__ __forceinline__ float2 operator-(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }
__device__ __forceinline__ float2 operator*(float s, float2 a) { return make_float2(s * a.x, s * a.y); }
__device__ __forceinline__ float  dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }

// ---------------------------------------------------------------------------
// Stateless RNG: PCG-style hash -> uniform float in [0,1)
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned int pcg_hash(unsigned int v) {
    unsigned int state = v * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ __forceinline__ float rand01(unsigned int index, unsigned int iter, unsigned int seed) {
    unsigned int h = pcg_hash(index ^ pcg_hash(iter * 2654435761u ^ pcg_hash(seed)));
    return (h & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

// Written by the device each iteration so the whole step can live in a graph.
struct IterState {
    int   iter;  // Adam step count, 1-based once a step has started
    int   count; // targets used by the loss this step
    float loss;  // sum of their VoroLoss
};

// Uploaded by BeginSchedule; read by the kernels through a device pointer.
struct RunParams {
    float lr, gamma, beta1, beta2, eps;
    int   milestones[4];
    float width, height;
    int   slotsW;
    int   cx0, cy0, cx1, cy1; // generator slots that move
    float bx0, by0, bx1, by1; // targets outside this box are ignored
};

// Canvas occupancy: row-major 8x8-voxel tiles, voxel (i, j) at bit (j & 7) * 8 + (i & 7).
struct Occupancy {
    const unsigned long long* masks;
    int                       tilesX, tilesY;
    float                     invVoxel;
};

__device__ bool occupied(const Occupancy& o, float2 p)
{
    if (!o.masks) return false;
    int i = int(floorf(p.x * o.invVoxel)), j = int(floorf(p.y * o.invVoxel));
    if (i < 0 || j < 0) return false;
    int tx = i >> 3, ty = j >> 3;
    if (tx >= o.tilesX || ty >= o.tilesY) return false;
    return (o.masks[ty * o.tilesX + tx] >> (((j & 7) << 3) | (i & 7))) & 1ull;
}

// Generators binned by cell; pos/idx are in sorted order, start has gridW*gridH+1 entries.
struct HashGrid {
    const int*    start;
    const int*    idx;
    const float2* pos;
    int           w, h;
    float         cellSize;
};

__device__ __forceinline__ int cellCoord(float v, float inv, int n) {
    return min(max(int(v * inv), 0), n - 1);
}

// K nearest generators of q, sorted by distance. Missing entries get idx -1.
// Searches square rings of cells outward until the K-th distance is within
// the distance from q to the edge of the searched block.
template <int K>
__device__ void knn(const HashGrid& g, float2 q, int* idx, float* d2)
{
    for (int k = 0; k < K; ++k) { idx[k] = -1; d2[k] = FLT_MAX; }
    float inv = 1.0f / g.cellSize;
    int cx = cellCoord(q.x, inv, g.w), cy = cellCoord(q.y, inv, g.h);

    for (int ring = 0;; ++ring) {
        int x0 = cx - ring, x1 = cx + ring, y0 = cy - ring, y1 = cy + ring;
        for (int y = max(y0, 0); y <= min(y1, g.h - 1); ++y) {
            // Interior rows of the ring only contribute their two end cells.
            int stride = (y == y0 || y == y1) ? 1 : max(x1 - x0, 1);
            for (int x = x0; x <= x1; x += stride) {
                if (x < 0 || x >= g.w) continue;
                int c = y * g.w + x;
                for (int s = g.start[c]; s < g.start[c + 1]; ++s) {
                    float2 p = g.pos[s];
                    float dx = p.x - q.x, dy = p.y - q.y;
                    float dd = dx * dx + dy * dy;
                    if (dd >= d2[K - 1]) continue;
                    int k = K - 1;
                    while (k > 0 && d2[k - 1] > dd) { d2[k] = d2[k - 1]; idx[k] = idx[k - 1]; --k; }
                    d2[k] = dd;
                    idx[k] = g.idx[s];
                }
            }
        }
        if (x0 <= 0 && y0 <= 0 && x1 >= g.w - 1 && y1 >= g.h - 1) break;
        float safe = fminf(fminf(q.x - x0 * g.cellSize, (x1 + 1) * g.cellSize - q.x),
                           fminf(q.y - y0 * g.cellSize, (y1 + 1) * g.cellSize - q.y));
        safe = fmaxf(safe, 0.0f);
        if (d2[K - 1] <= safe * safe) break;
    }
}

// ---------------------------------------------------------------------------
// Hash grid build
// ---------------------------------------------------------------------------
__global__ void binKernel(const float2* gen, int n, HashGrid g, int* counts, int* genCell, int* genRank)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (gen[i].x < 0.0f) { genCell[i] = -1; return; } // empty slot
    float inv = 1.0f / g.cellSize;
    int c = cellCoord(gen[i].y, inv, g.h) * g.w + cellCoord(gen[i].x, inv, g.w);
    genCell[i] = c;
    genRank[i] = atomicAdd(&counts[c], 1);
}

__global__ void scatterKernel(const float2* gen, int n, const int* genCell, const int* genRank,
                              const int* start, int* sortedIdx, float2* sortedPos)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || genCell[i] < 0) return;
    int s = start[genCell[i]] + genRank[i];
    sortedIdx[s] = i;
    sortedPos[s] = gen[i];
}

// ---------------------------------------------------------------------------
// Optimisation
// ---------------------------------------------------------------------------
__global__ void beginIterationKernel(IterState* st)
{
    st->iter += 1;
    st->count = 0;
    st->loss  = 0.0f;
}

// VoroLoss for one target point (Voroloss_opt.__call__) plus its gradient.
// With u = p - c0, e = cj - c0, l = |e|, n = e / l:
//   d        = <u, n> - l / 2
//   dd/dcj   = (u - <u, n> n) / l - n / 2
//   dd/dc0   = -n - dd/dcj
// and dL/dc = 2 d dd/dc. Gradients are summed here and divided by the number
// of contributing targets in the Adam kernel, matching loss.mean().
__global__ void vorolossKernel(const float2* __restrict__ targets, int numTargets, HashGrid g,
                               const float2* __restrict__ gen, float2* grad, IterState* st,
                               const RunParams* rp, float fraction, unsigned int seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float loss = 0.0f;
    int used = 0;

    float2 p = make_float2(-1.0f, -1.0f);
    if (i < numTargets) p = targets[i];
    bool take = p.x >= 0.0f; // empty slots have x < 0
    if (take && rp)
        take = p.x >= rp->bx0 && p.x <= rp->bx1 && p.y >= rp->by0 && p.y <= rp->by1;
    if (take && fraction < 1.0f)
        take = rand01(i, st->iter, seed) < fraction;

    int idx[kLossK];
    float d2[kLossK];
    if (take) knn<kLossK>(g, p, idx, d2);
    if (take && idx[0] >= 0) {
        int i0 = idx[0];
        float2 c0 = gen[i0];
        float2 u = p - c0;
        float best = FLT_MAX, bestD = 0.0f;
        int bestJ = -1;
        for (int k = 1; k < kLossK; ++k) {
            int j = idx[k];
            if (j < 0) break;
            float2 e = gen[j] - c0;
            float l = sqrtf(dot(e, e));
            if (l < 1e-12f) continue;
            float d = dot(u, e) / l - 0.5f * l;
            if (d * d < best) { best = d * d; bestD = d; bestJ = j; }
        }

        if (bestJ >= 0) {
            loss = best;
            used = 1;
            if (grad) {
                float2 e = gen[bestJ] - c0;
                float l = sqrtf(dot(e, e));
                float2 n = (1.0f / l) * e;
                float2 dcj = (1.0f / l) * (u - dot(u, n) * n) - 0.5f * n;
                float s = 2.0f * bestD;
                atomicAdd(&grad[bestJ].x, s * dcj.x);
                atomicAdd(&grad[bestJ].y, s * dcj.y);
                atomicAdd(&grad[i0].x, -s * (n.x + dcj.x));
                atomicAdd(&grad[i0].y, -s * (n.y + dcj.y));
            }
        }
    }

    for (int off = 16; off > 0; off >>= 1) {
        loss += __shfl_down_sync(0xFFFFFFFFu, loss, off);
        used += __shfl_down_sync(0xFFFFFFFFu, used, off);
    }
    if ((threadIdx.x & 31) == 0 && used > 0) {
        atomicAdd(&st->loss, loss);
        atomicAdd(&st->count, used);
    }
}

// torch.optim.Adam step with MultiStepLR evaluated from the device step count,
// followed by VoronoiValues.clamp(). Only slots in the active rectangle move.
// Also zeroes the gradient for the next step.
__global__ void adamKernel(float2* gen, float2* m, float2* v, float2* grad, int n,
                           const IterState* st, const RunParams* rp)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float2 gr = grad[i];
    grad[i] = make_float2(0.0f, 0.0f);

    float2 p = gen[i];
    int sx = i % rp->slotsW, sy = i / rp->slotsW;
    if (p.x < 0.0f || sx < rp->cx0 || sx >= rp->cx1 || sy < rp->cy0 || sy >= rp->cy1) return;

    int t = st->iter;
    float lr = rp->lr;
    for (int k = 0; k < 4; ++k)
        if (t - 1 >= rp->milestones[k]) lr *= rp->gamma;
    float bc1 = 1.0f - powf(rp->beta1, float(t));
    float bc2 = 1.0f - powf(rp->beta2, float(t));

    gr = (1.0f / float(max(st->count, 1))) * gr;
    float2 mi = rp->beta1 * m[i] + (1.0f - rp->beta1) * gr;
    float2 vi = rp->beta2 * v[i] + (1.0f - rp->beta2) * make_float2(gr.x * gr.x, gr.y * gr.y);
    m[i] = mi;
    v[i] = vi;

    p.x -= lr * (mi.x / bc1) / (sqrtf(vi.x / bc2) + rp->eps);
    p.y -= lr * (mi.y / bc1) / (sqrtf(vi.y / bc2) + rp->eps);
    p.x = fminf(fmaxf(p.x, 0.0f), rp->width);
    p.y = fminf(fmaxf(p.y, 0.0f), rp->height);
    gen[i] = p;
}

__global__ void scatterTargetsKernel(float2* targets, const int* slots, const float2* values, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) targets[slots[i]] = values[i];
}

// ---------------------------------------------------------------------------
// Extraction
// ---------------------------------------------------------------------------
struct CellPoly {
    double2 v[kMaxCellVerts];
    int     e[kMaxCellVerts]; // label of the edge v[k] -> v[k+1]
    int     n;
};

// Keeps the half-plane closer to the cell's generator (at the origin) than to
// the neighbour at r: <x, r> <= |r|^2 / 2. Returns false on vertex overflow.
__device__ bool clipByBisector(CellPoly& poly, double2 r, int label)
{
    double c = 0.5 * (r.x * r.x + r.y * r.y);
    bool anyOut = false;
    for (int k = 0; k < poly.n; ++k)
        anyOut |= poly.v[k].x * r.x + poly.v[k].y * r.y - c > 0.0;
    if (!anyOut) return true;

    CellPoly out;
    out.n = 0;
    bool ok = true;
    auto push = [&](double2 v, int e) {
        if (out.n < kMaxCellVerts) { out.v[out.n] = v; out.e[out.n] = e; ++out.n; }
        else ok = false;
    };
    for (int k = 0; k < poly.n; ++k) {
        double2 a = poly.v[k], b = poly.v[(k + 1) % poly.n];
        double fa = a.x * r.x + a.y * r.y - c;
        double fb = b.x * r.x + b.y * r.y - c;
        bool inA = fa <= 0.0, inB = fb <= 0.0;
        if (inA) push(a, poly.e[k]);
        if (inA != inB) {
            double t = fa / (fa - fb);
            double2 x = make_double2(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y));
            // Leaving the half-plane starts the new bisector edge; re-entering
            // resumes the old edge.
            push(x, inA ? label : poly.e[k]);
        }
    }
    poly = out;
    return ok;
}

__device__ double maxRadius2(const CellPoly& poly)
{
    double r2 = 0.0;
    for (int k = 0; k < poly.n; ++k)
        r2 = fmax(r2, poly.v[k].x * poly.v[k].x + poly.v[k].y * poly.v[k].y);
    return r2;
}

__global__ void voronoiCellKernel(HashGrid g, const float2* __restrict__ gen, int n, float width,
                                  float height, Occupancy occ, float2* outVerts, int* outEdges,
                                  int* outCount, uint8_t* outInside,
                                  int* flags /* [0] unsafe, [1] overflow */)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float2 ci = gen[i];
    if (ci.x < 0.0f) { // empty slot
        outCount[i] = 0;
        outInside[i] = 0;
        return;
    }
    int idx[kCellK];
    float d2[kCellK];
    knn<kCellK>(g, ci, idx, d2);

    // Domain box relative to ci, with positive signed area (clockwise on screen, y down).
    CellPoly poly;
    double x0 = -double(ci.x), y0 = -double(ci.y), x1 = width - double(ci.x), y1 = height - double(ci.y);
    poly.v[0] = make_double2(x0, y0); poly.e[0] = -1;
    poly.v[1] = make_double2(x1, y0); poly.e[1] = -2;
    poly.v[2] = make_double2(x1, y1); poly.e[2] = -3;
    poly.v[3] = make_double2(x0, y1); poly.e[3] = -4;
    poly.n = 4;

    bool ok = true;
    bool certified = false;
    double r2max = maxRadius2(poly);
    for (int k = 0; k < kCellK; ++k) {
        int j = idx[k];
        if (j < 0) { certified = true; break; } // every generator has been seen
        if (j == i) continue;
        if (double(d2[k]) > 4.0 * r2max) { certified = true; break; }
        float2 cj = gen[j];
        ok &= clipByBisector(poly, make_double2(double(cj.x) - ci.x, double(cj.y) - ci.y), j);
        r2max = maxRadius2(poly);
    }
    if (!certified && double(d2[kCellK - 1]) > 4.0 * r2max) certified = true;

    if (!certified) {
        atomicAdd(&flags[0], 1);
        for (int j = 0; j < n; ++j) {
            if (j == i || gen[j].x < 0.0f) continue;
            bool seen = false;
            for (int k = 0; k < kCellK; ++k) seen |= idx[k] == j;
            if (seen) continue;
            double2 r = make_double2(double(gen[j].x) - ci.x, double(gen[j].y) - ci.y);
            if (r.x * r.x + r.y * r.y > 4.0 * r2max) continue;
            ok &= clipByBisector(poly, r, j);
            r2max = maxRadius2(poly);
        }
    }
    if (!ok) atomicAdd(&flags[1], 1);

    // Centroid (shoelace) -> occupancy from the canvas voxel it falls in.
    double area = 0.0, cx = 0.0, cy = 0.0;
    for (int k = 0; k < poly.n; ++k) {
        double2 a = poly.v[k], b = poly.v[(k + 1) % poly.n];
        double cr = a.x * b.y - b.x * a.y;
        area += cr;
        cx += (a.x + b.x) * cr;
        cy += (a.y + b.y) * cr;
    }
    float2 centroid = ci;
    if (fabs(area) > 1e-12) {
        centroid.x += float(cx / (3.0 * area));
        centroid.y += float(cy / (3.0 * area));
    }
    outInside[i] = occupied(occ, centroid) ? 1 : 0;
    outCount[i] = poly.n;
    for (int k = 0; k < poly.n; ++k) {
        outVerts[i * kMaxCellVerts + k] = make_float2(float(ci.x + poly.v[k].x), float(ci.y + poly.v[k].y));
        outEdges[i * kMaxCellVerts + k] = poly.e[k];
    }
}

template <typename T>
void deviceAlloc(T*& p, size_t count)
{
    CUDA_CHECK(cudaFree(p));
    p = nullptr;
    if (count > 0) CUDA_CHECK(cudaMalloc(&p, count * sizeof(T)));
}

int blocksFor(int n) { return (n + kBlock - 1) / kBlock; }

} // namespace

namespace voromesh {

struct VoroOptimizer::Device
{
    cudaStream_t    stream = nullptr;
    cudaEvent_t     start = nullptr, stop = nullptr;
    cudaGraphExec_t graph = nullptr;

    // Hash grid (fixed size)
    int    gridW = 0, gridH = 0;
    int*   counts = nullptr;    // gridW*gridH + 1
    int*   cellStart = nullptr; // gridW*gridH + 1
    void*  scanTemp = nullptr;
    size_t scanTempBytes = 0;

    // Generator slots
    float2* gen = nullptr;
    float2* m = nullptr;
    float2* v = nullptr;
    float2* grad = nullptr;
    int*    genCell = nullptr;
    int*    genRank = nullptr;
    int*    sortedIdx = nullptr;
    float2* sortedPos = nullptr;
    float   slotCellSize = 0.0f;

    // Extraction outputs
    float2*  cellVerts = nullptr;
    int*     cellEdges = nullptr;
    int*     cellCount = nullptr;
    uint8_t* inside = nullptr;
    int*     flags = nullptr;

    // Target slots + staging for sparse updates
    float2* targets = nullptr;
    int*    stageSlots = nullptr;
    float2* stageValues = nullptr;
    int     stageCapacity = 0;

    // Canvas occupancy
    unsigned long long* occupancy = nullptr;
    int                 tilesX = 0, tilesY = 0;
    float               invVoxel = 0.0f;

    IterState* state = nullptr;
    RunParams* run = nullptr;

    HashGrid Grid(float spacing) const
    {
        return HashGrid{cellStart, sortedIdx, sortedPos, gridW, gridH, spacing};
    }

    Occupancy Occ() const { return Occupancy{occupancy, tilesX, tilesY, invVoxel}; }

    void ResetGraph()
    {
        if (graph) CUDA_CHECK(cudaGraphExecDestroy(graph));
        graph = nullptr;
    }
};

VoroOptimizer::VoroOptimizer(float width, float height, float spacing, const OptimizerParams &params)
    : mWidth(width), mHeight(height), mSpacing(spacing), mParams(params), mDev(std::make_unique<Device>())
{
    Device &d = *mDev;
    CUDA_CHECK(cudaStreamCreateWithFlags(&d.stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&d.start));
    CUDA_CHECK(cudaEventCreate(&d.stop));

    d.gridW = std::max(1, (int)std::ceil(width / spacing));
    d.gridH = std::max(1, (int)std::ceil(height / spacing));
    int cells = d.gridW * d.gridH + 1;
    deviceAlloc(d.counts, cells);
    deviceAlloc(d.cellStart, cells);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, d.scanTempBytes, d.counts, d.cellStart, cells));
    CUDA_CHECK(cudaMalloc(&d.scanTemp, d.scanTempBytes));

    deviceAlloc(d.state, 1);
    deviceAlloc(d.run, 1);
    deviceAlloc(d.flags, 2);
    CUDA_CHECK(cudaMemset(d.state, 0, sizeof(IterState)));
}

VoroOptimizer::~VoroOptimizer()
{
    Device &d = *mDev;
    d.ResetGraph();
    for (void *p : {(void *)d.counts, (void *)d.cellStart, d.scanTemp, (void *)d.gen, (void *)d.m, (void *)d.v,
                    (void *)d.grad, (void *)d.genCell, (void *)d.genRank, (void *)d.sortedIdx, (void *)d.sortedPos,
                    (void *)d.cellVerts, (void *)d.cellEdges, (void *)d.cellCount, (void *)d.inside,
                    (void *)d.flags, (void *)d.targets, (void *)d.stageSlots, (void *)d.stageValues,
                    (void *)d.occupancy, (void *)d.state, (void *)d.run})
        cudaFree(p);
    cudaEventDestroy(d.start);
    cudaEventDestroy(d.stop);
    cudaStreamDestroy(d.stream);
}

void VoroOptimizer::SetTargets(const std::vector<Vec2> &slots)
{
    Device &d = *mDev;
    int capacity = std::max((int)slots.size(), 1);
    if (capacity != mTargetCapacity) {
        d.ResetGraph();
        mTargetCapacity = capacity;
        deviceAlloc(d.targets, capacity);
    }
    CUDA_CHECK(cudaMemset(d.targets, 0xFF, capacity * sizeof(float2))); // NaN: fails every x >= 0 test
    CUDA_CHECK(cudaMemcpy(d.targets, slots.data(), slots.size() * sizeof(float2), cudaMemcpyHostToDevice));
}

void VoroOptimizer::UpdateTargets(const std::vector<int> &dirty, const std::vector<Vec2> &slots)
{
    if (dirty.empty()) return;
    Device &d = *mDev;
    if ((int)slots.size() > mTargetCapacity) {
        // Grow with slack (rare): re-upload everything.
        std::vector<Vec2> padded(slots);
        padded.resize(std::max(slots.size() * 3 / 2, slots.size() + 64), Vec2{-1.0f, -1.0f});
        SetTargets(padded);
        return;
    }

    int n = (int)dirty.size();
    if (n > d.stageCapacity) {
        d.stageCapacity = 2 * n;
        deviceAlloc(d.stageSlots, d.stageCapacity);
        deviceAlloc(d.stageValues, d.stageCapacity);
    }
    // Read values from `slots` (not the edit) so duplicate dirty entries agree.
    std::vector<Vec2> values(n);
    for (int k = 0; k < n; ++k) values[k] = slots[dirty[k]];
    CUDA_CHECK(cudaMemcpy(d.stageSlots, dirty.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.stageValues, values.data(), n * sizeof(float2), cudaMemcpyHostToDevice));
    scatterTargetsKernel<<<blocksFor(n), kBlock, 0, d.stream>>>(d.targets, d.stageSlots, d.stageValues, n);
    CUDA_CHECK(cudaGetLastError());
}

void VoroOptimizer::SetOccupancy(const std::vector<uint64_t> &masks, int tilesX, int tilesY, float voxelSize)
{
    Device &d = *mDev;
    if (tilesX != d.tilesX || tilesY != d.tilesY) {
        d.tilesX = tilesX;
        d.tilesY = tilesY;
        deviceAlloc(d.occupancy, size_t(tilesX) * tilesY);
    }
    d.invVoxel = 1.0f / voxelSize;
    CUDA_CHECK(cudaMemcpy(d.occupancy, masks.data(), masks.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
}

void VoroOptimizer::UpdateOccupancy(const std::vector<uint64_t> &masks, int tx0, int ty0, int tx1, int ty1)
{
    Device &d = *mDev;
    tx0 = std::max(tx0, 0), ty0 = std::max(ty0, 0);
    tx1 = std::min(tx1, d.tilesX), ty1 = std::min(ty1, d.tilesY);
    if (tx0 >= tx1 || ty0 >= ty1) return;
    size_t pitch = size_t(d.tilesX) * sizeof(uint64_t);
    size_t off   = size_t(ty0) * d.tilesX + tx0;
    CUDA_CHECK(cudaMemcpy2D(d.occupancy + off, pitch, masks.data() + off, pitch, (tx1 - tx0) * sizeof(uint64_t),
                            ty1 - ty0, cudaMemcpyHostToDevice));
}

void VoroOptimizer::SetGenerators(const Vec2 *slots, int slotsW, int slotsH, float slotCellSize)
{
    Device &d = *mDev;
    int n = slotsW * slotsH;
    if (slotsW != mSlotsW || slotsH != mSlotsH) {
        d.ResetGraph();
        mSlotsW = slotsW;
        mSlotsH = slotsH;
        deviceAlloc(d.gen, n);
        deviceAlloc(d.m, n);
        deviceAlloc(d.v, n);
        deviceAlloc(d.grad, n);
        deviceAlloc(d.genCell, n);
        deviceAlloc(d.genRank, n);
        deviceAlloc(d.sortedIdx, n);
        deviceAlloc(d.sortedPos, n);
        deviceAlloc(d.cellVerts, size_t(n) * kMaxCellVerts);
        deviceAlloc(d.cellEdges, size_t(n) * kMaxCellVerts);
        deviceAlloc(d.cellCount, n);
        deviceAlloc(d.inside, n);
    }
    d.slotCellSize = slotCellSize;
    CUDA_CHECK(cudaMemcpy(d.gen, slots, n * sizeof(float2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d.grad, 0, n * sizeof(float2)));
    BeginSchedule(mParams.iterations);
}

void VoroOptimizer::UpdateGenerators(const Vec2 *slots, int cx0, int cy0, int cx1, int cy1)
{
    Device &d = *mDev;
    cx0 = std::max(cx0, 0), cy0 = std::max(cy0, 0);
    cx1 = std::min(cx1, mSlotsW), cy1 = std::min(cy1, mSlotsH);
    if (cx0 >= cx1 || cy0 >= cy1) return;
    size_t pitch = size_t(mSlotsW) * sizeof(float2);
    size_t off   = size_t(cy0) * mSlotsW + cx0;
    CUDA_CHECK(cudaMemcpy2D(d.gen + off, pitch, slots + off, pitch, (cx1 - cx0) * sizeof(float2), cy1 - cy0,
                            cudaMemcpyHostToDevice));
}

void VoroOptimizer::BeginSchedule(int iterations, int cx0, int cy0, int cx1, int cy1)
{
    Device &d = *mDev;
    cx0 = std::max(cx0, 0), cy0 = std::max(cy0, 0);
    cx1 = std::min(cx1, mSlotsW), cy1 = std::min(cy1, mSlotsH);

    RunParams rp;
    rp.lr    = mParams.lrPerSpacing * mSpacing;
    rp.gamma = mParams.gamma;
    rp.beta1 = 0.9f;
    rp.beta2 = 0.999f;
    rp.eps   = 1e-8f;
    for (int k = 0; k < 4; ++k)
        rp.milestones[k] = (int)std::lround(double(mParams.milestones[k]) * iterations / mParams.iterations);
    rp.width  = mWidth;
    rp.height = mHeight;
    rp.slotsW = mSlotsW;
    rp.cx0 = cx0, rp.cy0 = cy0, rp.cx1 = cx1, rp.cy1 = cy1;
    float margin = 2.0f * mSpacing;
    rp.bx0 = cx0 * d.slotCellSize - margin;
    rp.by0 = cy0 * d.slotCellSize - margin;
    rp.bx1 = cx1 * d.slotCellSize + margin;
    rp.by1 = cy1 * d.slotCellSize + margin;
    CUDA_CHECK(cudaMemcpy(d.run, &rp, sizeof(rp), cudaMemcpyHostToDevice));

    if (cx0 < cx1 && cy0 < cy1) {
        size_t pitch = size_t(mSlotsW) * sizeof(float2);
        size_t off   = size_t(cy0) * mSlotsW + cx0;
        CUDA_CHECK(cudaMemset2D(d.m + off, pitch, 0, (cx1 - cx0) * sizeof(float2), cy1 - cy0));
        CUDA_CHECK(cudaMemset2D(d.v + off, pitch, 0, (cx1 - cx0) * sizeof(float2), cy1 - cy0));
    }
    CUDA_CHECK(cudaMemset(d.state, 0, sizeof(IterState)));
    mScheduleLength = iterations;
    mIteration      = 0;
    mLastBatchLoss  = 0.0f;
}

void VoroOptimizer::EnqueueBuildGrid()
{
    Device &d = *mDev;
    int cells = d.gridW * d.gridH + 1;
    int n = NumSlots();
    HashGrid g = d.Grid(mSpacing);
    CUDA_CHECK(cudaMemsetAsync(d.counts, 0, cells * sizeof(int), d.stream));
    binKernel<<<blocksFor(n), kBlock, 0, d.stream>>>(d.gen, n, g, d.counts, d.genCell, d.genRank);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d.scanTemp, d.scanTempBytes, d.counts, d.cellStart, cells, d.stream));
    scatterKernel<<<blocksFor(n), kBlock, 0, d.stream>>>(d.gen, n, d.genCell, d.genRank, d.cellStart, d.sortedIdx,
                                                          d.sortedPos);
}

void VoroOptimizer::EnqueueIteration()
{
    Device &d = *mDev;
    beginIterationKernel<<<1, 1, 0, d.stream>>>(d.state);
    EnqueueBuildGrid();
    vorolossKernel<<<blocksFor(mTargetCapacity), kBlock, 0, d.stream>>>(
        d.targets, mTargetCapacity, d.Grid(mSpacing), d.gen, d.grad, d.state, d.run, mParams.sampleFraction,
        mParams.seed);
    adamKernel<<<blocksFor(NumSlots()), kBlock, 0, d.stream>>>(d.gen, d.m, d.v, d.grad, NumSlots(), d.state, d.run);
}

void VoroOptimizer::BuildGraph()
{
    Device &d = *mDev;
    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamBeginCapture(d.stream, cudaStreamCaptureModeThreadLocal));
    EnqueueIteration();
    CUDA_CHECK(cudaStreamEndCapture(d.stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&d.graph, graph, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
}

float VoroOptimizer::Step(int iterations)
{
    if (iterations <= 0 || NumSlots() == 0 || mTargetCapacity == 0) return 0.0f;
    Device &d = *mDev;
    if (!d.graph) BuildGraph();

    CUDA_CHECK(cudaEventRecord(d.start, d.stream));
    for (int k = 0; k < iterations; ++k)
        CUDA_CHECK(cudaGraphLaunch(d.graph, d.stream));
    CUDA_CHECK(cudaEventRecord(d.stop, d.stream));

    IterState st;
    CUDA_CHECK(cudaMemcpyAsync(&st, d.state, sizeof(st), cudaMemcpyDeviceToHost, d.stream));
    CUDA_CHECK(cudaStreamSynchronize(d.stream));
    mIteration     = st.iter;
    mLastBatchLoss = st.count > 0 ? st.loss / st.count : 0.0f;

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, d.start, d.stop));
    return ms;
}

float VoroOptimizer::EvaluateLoss()
{
    if (NumSlots() == 0 || mTargetCapacity == 0) return 0.0f;
    Device &d = *mDev;
    EnqueueBuildGrid();
    // Clear count + loss but keep the step counter.
    CUDA_CHECK(cudaMemsetAsync(&d.state->count, 0, sizeof(int) + sizeof(float), d.stream));
    vorolossKernel<<<blocksFor(mTargetCapacity), kBlock, 0, d.stream>>>(
        d.targets, mTargetCapacity, d.Grid(mSpacing), d.gen, nullptr, d.state, nullptr, 1.0f, mParams.seed);
    CUDA_CHECK(cudaGetLastError());

    IterState st;
    CUDA_CHECK(cudaMemcpyAsync(&st, d.state, sizeof(st), cudaMemcpyDeviceToHost, d.stream));
    CUDA_CHECK(cudaStreamSynchronize(d.stream));
    return st.count > 0 ? st.loss / st.count : 0.0f;
}

VoroMesh2D VoroOptimizer::Extract()
{
    VoroMesh2D mesh;
    int n = NumSlots();
    if (n == 0) return mesh;
    Device &d = *mDev;

    CUDA_CHECK(cudaEventRecord(d.start, d.stream));
    EnqueueBuildGrid();
    CUDA_CHECK(cudaMemsetAsync(d.flags, 0, 2 * sizeof(int), d.stream));
    voronoiCellKernel<<<blocksFor(n), kBlock, 0, d.stream>>>(d.Grid(mSpacing), d.gen, n, mWidth, mHeight, d.Occ(),
                                                              d.cellVerts, d.cellEdges, d.cellCount, d.inside,
                                                              d.flags);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(d.stop, d.stream));
    CUDA_CHECK(cudaEventSynchronize(d.stop));
    CUDA_CHECK(cudaEventElapsedTime(&mesh.gpuMs, d.start, d.stop));

    auto t0 = std::chrono::steady_clock::now();

    std::vector<Vec2> verts(size_t(n) * kMaxCellVerts);
    std::vector<int>  labels(size_t(n) * kMaxCellVerts);
    std::vector<int>  counts(n);
    int               flags[2];
    mesh.generators.resize(n);
    mesh.inside.resize(n);
    CUDA_CHECK(cudaMemcpy(verts.data(), d.cellVerts, verts.size() * sizeof(float2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(labels.data(), d.cellEdges, labels.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(counts.data(), d.cellCount, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mesh.inside.data(), d.inside, n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mesh.generators.data(), d.gen, n * sizeof(float2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(flags, d.flags, sizeof(flags), cudaMemcpyDeviceToHost));
    mesh.unsafeCells   = flags[0];
    mesh.overflowCells = flags[1];

    // Compact the polygons into CSR.
    mesh.cellStart.resize(n + 1);
    mesh.cellStart[0] = 0;
    for (int i = 0; i < n; ++i) mesh.cellStart[i + 1] = mesh.cellStart[i] + counts[i];
    mesh.cellVerts.resize(mesh.cellStart[n]);
    for (int i = 0; i < n; ++i)
        std::copy_n(verts.begin() + size_t(i) * kMaxCellVerts, counts[i], mesh.cellVerts.begin() + mesh.cellStart[i]);

    // Boundary edges: every edge of an inside cell whose other side is an
    // outside cell or the domain box. A cell vertex sits between edges with
    // labels a and b, so it is equidistant to generators {i, a, b}.
    auto key = [](int a, int b, int c) {
        uint64_t v0 = uint64_t(a + kBoxLabels), v1 = uint64_t(b + kBoxLabels), v2 = uint64_t(c + kBoxLabels);
        if (v0 > v1) std::swap(v0, v1);
        if (v1 > v2) std::swap(v1, v2);
        if (v0 > v1) std::swap(v0, v1);
        return (v0 << 42) | (v1 << 21) | v2;
    };
    std::unordered_map<uint64_t, int> vertexOf;
    auto vertex = [&](uint64_t k, Vec2 p) {
        auto [it, inserted] = vertexOf.try_emplace(k, (int)mesh.vertices.size());
        if (inserted) mesh.vertices.push_back(p);
        return it->second;
    };
    for (int i = 0; i < n; ++i) {
        if (!mesh.inside[i]) continue;
        int c = counts[i];
        const int  *lab = labels.data() + size_t(i) * kMaxCellVerts;
        const Vec2 *pos = verts.data() + size_t(i) * kMaxCellVerts;
        for (int k = 0; k < c; ++k) {
            int l = lab[k];
            if (l >= 0 && mesh.inside[l]) continue;
            int prev = lab[(k + c - 1) % c], next = lab[(k + 1) % c];
            int a = vertex(key(i, prev, l), pos[k]);
            int b = vertex(key(i, l, next), pos[(k + 1) % c]);
            mesh.edges.push_back({a, b});
        }
    }

    // Watertight iff each vertex has exactly one outgoing and one incoming edge.
    std::vector<int> next(mesh.vertices.size(), -1), inDeg(mesh.vertices.size(), 0);
    mesh.watertight = true;
    for (auto [a, b] : mesh.edges) {
        if (next[a] != -1) mesh.watertight = false;
        next[a] = b;
        ++inDeg[b];
    }
    for (size_t v = 0; v < next.size(); ++v)
        if (next[v] == -1 || inDeg[v] != 1) mesh.watertight = false;
    if (mesh.watertight) {
        std::vector<uint8_t> seen(next.size(), 0);
        for (size_t v = 0; v < next.size(); ++v) {
            if (seen[v]) continue;
            ++mesh.loops;
            for (int w = (int)v; !seen[w]; w = next[w]) seen[w] = 1;
        }
    }

    mesh.hostMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return mesh;
}

} // namespace voromesh
