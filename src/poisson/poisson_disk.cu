// Parallel Poisson disk sampling via grid-based dart throwing (2D).
//
// Idea (after Wei, "Parallel Poisson Disk Sampling", SIGGRAPH 2008):
//   * Overlay a background grid with cell size r/sqrt(2). Each cell can then
//     hold at most one sample (cell diagonal == r).
//   * A dart in a cell can only conflict with samples in the surrounding
//     5x5 block of cells (|offset| <= 2).
//   * Split cells into 3x3 = 9 "phase groups" by (cx % 3, cy % 3). Two cells in
//     the same group are >= 3 cells apart, so their 5x5 neighborhoods never
//     contain each other's cell, and darts from the same group can never
//     conflict. Within one phase, every thread owns one cell, throws one dart,
//     tests it against the (read-only for this phase) neighbors, and writes
//     only to its own cell. No atomics or locks are needed.
//   * Repeat for `trials` rounds, each round running all 9 phases. Cells that
//     are already filled are skipped. The result is a (nearly) maximal set.
//
// Region resampling: phases can be restricted to a rectangle of cells. Phase
// groups are taken relative to the rectangle's origin, which keeps the
// ">= 3 cells apart" guarantee. Cells outside the rectangle are never written,
// so they act as fixed constraints and the combined set stays valid.

#include "poisson/poisson_disk.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static_assert(sizeof(poisson::Sample) == sizeof(float2), "Sample must mirror float2");

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

// ---------------------------------------------------------------------------
// Stateless RNG: PCG-style hash -> uniform float in [0,1)
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned int pcg_hash(unsigned int v) {
    unsigned int state = v * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ __forceinline__ float rand01(unsigned int cell, unsigned int round,
                                        unsigned int seed, unsigned int stream) {
    unsigned int h = pcg_hash(cell ^ pcg_hash(round * 2654435761u ^
                                              pcg_hash(seed + stream * 0x9E3779B9u)));
    return (h & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

struct Params {
    float width, height;   // domain [0,width) x [0,height)
    float radius;          // minimum distance between samples
    float cellSize;        // radius / sqrt(2)
    int   gridW, gridH;    // grid dimensions in cells
    unsigned int seed;
};

// Cell rectangle [x0,x1) x [y0,y1) that a launch is allowed to write.
struct Region {
    int x0, y0, x1, y1;
};

// Grid stores one float2 per cell. x < 0 means "empty".
__global__ void dartPhaseKernel(float2* grid, Params p, Region rg, int phaseX, int phaseY,
                                unsigned int round)
{
    // Thread (tx,ty) owns the cell (x0 + 3*tx + phaseX, y0 + 3*ty + phaseY).
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int cx = rg.x0 + 3 * tx + phaseX;
    int cy = rg.y0 + 3 * ty + phaseY;
    if (cx >= rg.x1 || cy >= rg.y1) return;

    int cellId = cy * p.gridW + cx;
    if (grid[cellId].x >= 0.0f) return;      // already has a sample

    // Throw a dart uniformly inside this cell.
    float x = (cx + rand01(cellId, round, p.seed, 0)) * p.cellSize;
    float y = (cy + rand01(cellId, round, p.seed, 1)) * p.cellSize;
    if (x >= p.width || y >= p.height) return;   // outside the domain

    // Check the 5x5 neighborhood for conflicts.
    float r2 = p.radius * p.radius;
    int x0 = max(cx - 2, 0), x1 = min(cx + 2, p.gridW - 1);
    int y0 = max(cy - 2, 0), y1 = min(cy + 2, p.gridH - 1);
    for (int ny = y0; ny <= y1; ++ny) {
        for (int nx = x0; nx <= x1; ++nx) {
            float2 q = grid[ny * p.gridW + nx];
            if (q.x < 0.0f) continue;
            float dx = q.x - x, dy = q.y - y;
            if (dx * dx + dy * dy < r2) return;      // too close -> reject
        }
    }

    // Accept. Only this thread writes this cell during this phase.
    grid[cellId] = make_float2(x, y);
}

__global__ void clearRegionKernel(float2* grid, int gridW, Region rg) {
    int cx = rg.x0 + blockIdx.x * blockDim.x + threadIdx.x;
    int cy = rg.y0 + blockIdx.y * blockDim.y + threadIdx.y;
    if (cx < rg.x1 && cy < rg.y1) grid[cy * gridW + cx] = make_float2(-1.0f, -1.0f);
}

} // namespace

namespace poisson {

Sampler::Sampler(float width, float height, float radius, int trials, unsigned seed)
    : mWidth(width), mHeight(height), mRadius(radius), mCellSize(radius / sqrtf(2.0f)),
      mTrials(trials), mSeed(seed)
{
    mGridW = (int)ceilf(width / mCellSize);
    mGridH = (int)ceilf(height / mCellSize);
    size_t nCells = (size_t)mGridW * mGridH;
    mCells.assign(nCells, Sample{-1.0f, -1.0f});

    CUDA_CHECK(cudaMalloc(&mDeviceGrid, nCells * sizeof(float2)));
    CUDA_CHECK(cudaEventCreate(reinterpret_cast<cudaEvent_t*>(&mStart)));
    CUDA_CHECK(cudaEventCreate(reinterpret_cast<cudaEvent_t*>(&mStop)));
    ResampleAll();
}

Sampler::~Sampler()
{
    cudaFree(mDeviceGrid);
    cudaEventDestroy(static_cast<cudaEvent_t>(mStart));
    cudaEventDestroy(static_cast<cudaEvent_t>(mStop));
}

float Sampler::Resample(int cx0, int cy0, int cx1, int cy1)
{
    Region rg{std::max(cx0, 0), std::max(cy0, 0), std::min(cx1, mGridW), std::min(cy1, mGridH)};
    if (rg.x0 >= rg.x1 || rg.y0 >= rg.y1) return 0.0f;
    int rw = rg.x1 - rg.x0, rh = rg.y1 - rg.y0;

    Params p{mWidth, mHeight, mRadius, mCellSize, mGridW, mGridH, mSeed};
    float2* dGrid = static_cast<float2*>(mDeviceGrid);
    auto start = static_cast<cudaEvent_t>(mStart), stop = static_cast<cudaEvent_t>(mStop);

    dim3 block(16, 16);
    CUDA_CHECK(cudaEventRecord(start));
    clearRegionKernel<<<dim3((rw + 15) / 16, (rh + 15) / 16), block>>>(dGrid, mGridW, rg);

    // Each phase launches one thread per cell in that phase group of the region.
    int threadsX = (rw + 2) / 3;
    int threadsY = (rh + 2) / 3;
    dim3 grid((threadsX + block.x - 1) / block.x, (threadsY + block.y - 1) / block.y);
    for (int round = 0; round < mTrials; ++round) {
        for (int py = 0; py < 3; ++py)
            for (int px = 0; px < 3; ++px)
                dartPhaseKernel<<<grid, block>>>(dGrid, p, rg, px, py, mRoundBase + (unsigned)round);
    }
    mRoundBase += (unsigned)mTrials;
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));

    // Copy just the region's rows back into the host mirror.
    size_t pitch = (size_t)mGridW * sizeof(float2);
    size_t offset = (size_t)rg.y0 * mGridW + rg.x0;
    CUDA_CHECK(cudaMemcpy2D(mCells.data() + offset, pitch, dGrid + offset, pitch,
                            rw * sizeof(float2), rh, cudaMemcpyDeviceToHost));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

bool Sampler::Verify(int cx0, int cy0, int cx1, int cy1) const
{
    float r2 = mRadius * mRadius;
    int vx0 = std::max(cx0 - 2, 0), vx1 = std::min(cx1 + 2, mGridW);
    int vy0 = std::max(cy0 - 2, 0), vy1 = std::min(cy1 + 2, mGridH);
    for (int cy = vy0; cy < vy1; ++cy) {
        for (int cx = vx0; cx < vx1; ++cx) {
            Sample a = mCells[cy * mGridW + cx];
            if (a.x < 0) continue;
            for (int ny = std::max(cy - 2, 0); ny <= std::min(cy + 2, mGridH - 1); ++ny)
                for (int nx = std::max(cx - 2, 0); nx <= std::min(cx + 2, mGridW - 1); ++nx) {
                    if (nx == cx && ny == cy) continue;
                    Sample b = mCells[ny * mGridW + nx];
                    if (b.x < 0) continue;
                    float dx = a.x - b.x, dy = a.y - b.y;
                    if (dx * dx + dy * dy < r2) return false;
                }
        }
    }
    return true;
}

} // namespace poisson
