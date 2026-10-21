// CPU twin of voro_optimizer.cu. The per-target loss/gradient, Adam step and
// cell clipping are line-for-line ports of the CUDA kernels; what differs is
// the bookkeeping that keeps a local edit local (see the header).

#include "voromesh/cpu_voro_optimizer.h"

#include "mesh_assembly.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>

namespace voromesh {

namespace {

constexpr int kLossK = 12; // neighbours per target point (paper: 16 in 3D)
constexpr int kCellK = 32; // neighbours tried per cell before the brute-force fallback

// kNN grid cell size in generator spacings. Ring search pays per cell visited,
// so cells holding a few generators each beat one-generator cells.
constexpr float kGridCellScale = 2.0f;

// Same stateless hash as the GPU subsampling, so both paths pick the same targets.
unsigned PcgHash(unsigned v)
{
    unsigned state = v * 747796405u + 2891336453u;
    unsigned word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float Rand01(unsigned index, unsigned iter, unsigned seed)
{
    unsigned h = PcgHash(index ^ PcgHash(iter * 2654435761u ^ PcgHash(seed)));
    return (h & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

float Dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }

struct DVec2
{
    double x, y;
};

struct Poly
{
    DVec2 v[CpuVoroOptimizer::kMaxCellVerts];
    int   e[CpuVoroOptimizer::kMaxCellVerts];
    int   n;
};

// Keeps the half-plane closer to the cell's generator (at the origin) than to
// the neighbour at r: <x, r> <= |r|^2 / 2. Returns false on vertex overflow.
bool ClipByBisector(Poly &poly, DVec2 r, int label)
{
    double c = 0.5 * (r.x * r.x + r.y * r.y);
    bool anyOut = false;
    for (int k = 0; k < poly.n; ++k)
        anyOut |= poly.v[k].x * r.x + poly.v[k].y * r.y - c > 0.0;
    if (!anyOut)
        return true;

    Poly out;
    out.n   = 0;
    bool ok = true;
    auto push = [&](DVec2 v, int e) {
        if (out.n < CpuVoroOptimizer::kMaxCellVerts) {
            out.v[out.n] = v;
            out.e[out.n] = e;
            ++out.n;
        } else {
            ok = false;
        }
    };
    for (int k = 0; k < poly.n; ++k) {
        DVec2  a = poly.v[k], b = poly.v[(k + 1) % poly.n];
        double fa = a.x * r.x + a.y * r.y - c;
        double fb = b.x * r.x + b.y * r.y - c;
        bool   inA = fa <= 0.0, inB = fb <= 0.0;
        if (inA)
            push(a, poly.e[k]);
        if (inA != inB) {
            double t = fa / (fa - fb);
            // Leaving the half-plane starts the new bisector edge; re-entering
            // resumes the old edge.
            push({a.x + t * (b.x - a.x), a.y + t * (b.y - a.y)}, inA ? label : poly.e[k]);
        }
    }
    poly = out;
    return ok;
}

double MaxRadius2(const Poly &poly)
{
    double r2 = 0.0;
    for (int k = 0; k < poly.n; ++k)
        r2 = std::max(r2, poly.v[k].x * poly.v[k].x + poly.v[k].y * poly.v[k].y);
    return r2;
}

} // namespace

// K nearest generators, sorted by distance. Missing entries get idx -1.
template <int K>
struct CpuVoroOptimizer::Knn
{
    int   idx[K];
    float d2[K];

    Knn()
    {
        for (int k = 0; k < K; ++k) {
            idx[k] = -1;
            d2[k]  = FLT_MAX;
        }
    }
    float Worst() const { return d2[K - 1]; }
    void  Insert(int j, float dd)
    {
        if (dd >= d2[K - 1])
            return;
        int k = K - 1;
        while (k > 0 && d2[k - 1] > dd) {
            d2[k]  = d2[k - 1];
            idx[k] = idx[k - 1];
            --k;
        }
        d2[k]  = dd;
        idx[k] = j;
    }
};

void CpuVoroOptimizer::PointGrid::Build(const std::vector<int> &ids, const std::vector<Vec2> &gen, float cs)
{
    cellSize = cs;
    idx.resize(ids.size());
    pos.resize(ids.size());
    if (ids.empty()) {
        w = h = 0;
        start.assign(1, 0);
        return;
    }
    float minX = FLT_MAX, minY = FLT_MAX, maxX = -FLT_MAX, maxY = -FLT_MAX;
    for (int i : ids) {
        minX = std::min(minX, gen[i].x), maxX = std::max(maxX, gen[i].x);
        minY = std::min(minY, gen[i].y), maxY = std::max(maxY, gen[i].y);
    }
    x0 = minX;
    y0 = minY;
    w  = int((maxX - minX) / cs) + 1;
    h  = int((maxY - minY) / cs) + 1;

    // Counting sort by cell.
    float inv = 1.0f / cs;
    start.assign(size_t(w) * h + 1, 0);
    cellOf.resize(ids.size());
    for (size_t k = 0; k < ids.size(); ++k) {
        Vec2 p = gen[ids[k]];
        int  c = std::min(int((p.y - y0) * inv), h - 1) * w + std::min(int((p.x - x0) * inv), w - 1);
        cellOf[k] = c;
        ++start[c + 1];
    }
    for (size_t c = 0; c + 1 < start.size(); ++c)
        start[c + 1] += start[c];
    cursor.assign(start.begin(), start.end() - 1);
    for (size_t k = 0; k < ids.size(); ++k) {
        int s  = cursor[cellOf[k]]++;
        idx[s] = ids[k];
        pos[s] = gen[ids[k]];
    }
}

// Adds the generators of `g` to `knn`. Searches square rings of cells outward
// until the K-th distance is within the distance from q to the edge of the
// searched block. Can be called on several grids in turn: the K-th distance
// only shrinks, so earlier searches stay valid.
template <int K>
void CpuVoroOptimizer::Search(const PointGrid &g, Vec2 q, Knn<K> &knn) const
{
    if (g.idx.empty())
        return;
    const float cs = g.cellSize;
    float gx1 = g.x0 + g.w * cs, gy1 = g.y0 + g.h * cs;
    float bx = std::max({g.x0 - q.x, 0.0f, q.x - gx1}), by = std::max({g.y0 - q.y, 0.0f, q.y - gy1});
    if (bx * bx + by * by >= knn.Worst())
        return; // the whole grid is farther than the current K-th neighbour

    float inv = 1.0f / cs;
    int   cx  = std::clamp(int(std::floor((q.x - g.x0) * inv)), 0, g.w - 1);
    int   cy  = std::clamp(int(std::floor((q.y - g.y0) * inv)), 0, g.h - 1);
    for (int ring = 0;; ++ring) {
        int x0 = cx - ring, x1 = cx + ring, y0 = cy - ring, y1 = cy + ring;
        for (int y = std::max(y0, 0); y <= std::min(y1, g.h - 1); ++y) {
            // Interior rows of the ring only contribute their two end cells.
            int stride = (y == y0 || y == y1) ? 1 : std::max(x1 - x0, 1);
            for (int x = x0; x <= x1; x += stride) {
                if (x < 0 || x >= g.w)
                    continue;
                int c = y * g.w + x;
                for (int s = g.start[c]; s < g.start[c + 1]; ++s) {
                    float dx = g.pos[s].x - q.x, dy = g.pos[s].y - q.y;
                    knn.Insert(g.idx[s], dx * dx + dy * dy);
                }
            }
        }
        if (x0 <= 0 && y0 <= 0 && x1 >= g.w - 1 && y1 >= g.h - 1)
            break;
        float safe = std::min(std::min(q.x - (g.x0 + x0 * cs), (g.x0 + (x1 + 1) * cs) - q.x),
                              std::min(q.y - (g.y0 + y0 * cs), (g.y0 + (y1 + 1) * cs) - q.y));
        safe = std::max(safe, 0.0f);
        if (knn.Worst() <= safe * safe)
            break;
    }
}

CpuVoroOptimizer::CpuVoroOptimizer(float width, float height, float spacing, const OptimizerParams &params)
    : mWidth(width), mHeight(height), mSpacing(spacing), mParams(params)
{
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------
void CpuVoroOptimizer::SetTargets(const std::vector<Vec2> &slots) { mTargets = slots; }

void CpuVoroOptimizer::UpdateTargets(const std::vector<int> &dirty, const std::vector<Vec2> &slots)
{
    if (slots.size() > mTargets.size())
        mTargets.resize(slots.size(), Vec2{-1.0f, -1.0f});
    for (int s : dirty)
        mTargets[s] = slots[s];
}

void CpuVoroOptimizer::SetOccupancy(const std::vector<uint64_t> &masks, int tilesX, int tilesY, float voxelSize)
{
    mOccupancy  = masks;
    mTilesX     = tilesX;
    mTilesY     = tilesY;
    mVoxelSize  = voxelSize;
    mInvVoxel   = 1.0f / voxelSize;
    mCellsValid = false;
}

void CpuVoroOptimizer::UpdateOccupancy(const std::vector<uint64_t> &masks, int tx0, int ty0, int tx1, int ty1)
{
    tx0 = std::max(tx0, 0), ty0 = std::max(ty0, 0);
    tx1 = std::min(tx1, mTilesX), ty1 = std::min(ty1, mTilesY);
    for (int ty = ty0; ty < ty1; ++ty)
        for (int tx = tx0; tx < tx1; ++tx)
            mOccupancy[size_t(ty) * mTilesX + tx] = masks[size_t(ty) * mTilesX + tx];
    float tile = 8.0f * mVoxelSize;
    MarkDirty(tx0 * tile, ty0 * tile, tx1 * tile, ty1 * tile);
}

bool CpuVoroOptimizer::Occupied(Vec2 p) const
{
    int i = int(std::floor(p.x * mInvVoxel)), j = int(std::floor(p.y * mInvVoxel));
    if (i < 0 || j < 0)
        return false;
    int tx = i >> 3, ty = j >> 3;
    if (tx >= mTilesX || ty >= mTilesY)
        return false;
    return (mOccupancy[size_t(ty) * mTilesX + tx] >> (((j & 7) << 3) | (i & 7))) & 1;
}

void CpuVoroOptimizer::SetGenerators(const Vec2 *slots, int slotsW, int slotsH, float slotCellSize)
{
    int n         = slotsW * slotsH;
    mSlotsW       = slotsW;
    mSlotsH       = slotsH;
    mSlotCellSize = slotCellSize;
    mGen.assign(slots, slots + n);
    mM.assign(n, Vec2{0, 0});
    mV.assign(n, Vec2{0, 0});
    mGrad.assign(n, Vec2{0, 0});
    mCellsValid = false;
    BeginSchedule(mParams.iterations);
}

void CpuVoroOptimizer::UpdateGenerators(const Vec2 *slots, int cx0, int cy0, int cx1, int cy1)
{
    cx0 = std::max(cx0, 0), cy0 = std::max(cy0, 0);
    cx1 = std::min(cx1, mSlotsW), cy1 = std::min(cy1, mSlotsH);
    for (int cy = cy0; cy < cy1; ++cy)
        for (int cx = cx0; cx < cx1; ++cx)
            mGen[size_t(cy) * mSlotsW + cx] = slots[size_t(cy) * mSlotsW + cx];
    MarkDirty(cx0 * mSlotCellSize, cy0 * mSlotCellSize, cx1 * mSlotCellSize, cy1 * mSlotCellSize);
}

void CpuVoroOptimizer::MarkDirty(float x0, float y0, float x1, float y1)
{
    if (!mDirty) {
        mDirtyBox[0] = x0, mDirtyBox[1] = y0, mDirtyBox[2] = x1, mDirtyBox[3] = y1;
        mDirty       = true;
        return;
    }
    mDirtyBox[0] = std::min(mDirtyBox[0], x0), mDirtyBox[1] = std::min(mDirtyBox[1], y0);
    mDirtyBox[2] = std::max(mDirtyBox[2], x1), mDirtyBox[3] = std::max(mDirtyBox[3], y1);
}

// ---------------------------------------------------------------------------
// Optimisation
// ---------------------------------------------------------------------------
void CpuVoroOptimizer::BeginSchedule(int iterations, int cx0, int cy0, int cx1, int cy1)
{
    cx0 = std::max(cx0, 0), cy0 = std::max(cy0, 0);
    cx1 = std::min(cx1, mSlotsW), cy1 = std::min(cy1, mSlotsH);

    mLr = mParams.lrPerSpacing * mSpacing;
    for (int k = 0; k < 4; ++k)
        mMilestones[k] = (int)std::lround(double(mParams.milestones[k]) * iterations / mParams.iterations);

    // Split live generators into active (reset Adam) and frozen (binned once).
    int              n = NumSlots();
    std::vector<int> frozen;
    mActive.assign(n, 0);
    mActiveSlots.clear();
    for (int i = 0; i < n; ++i) {
        if (mGen[i].x < 0.0f)
            continue;
        int sx = i % mSlotsW, sy = i / mSlotsW;
        if (sx >= cx0 && sx < cx1 && sy >= cy0 && sy < cy1) {
            mActive[i] = 1;
            mActiveSlots.push_back(i);
            mM[i] = mV[i] = mGrad[i] = Vec2{0, 0};
        } else {
            frozen.push_back(i);
        }
    }
    mFrozenGrid.Build(frozen, mGen, kGridCellScale * mSpacing);

    // Targets within two spacings of the active rectangle.
    float margin = 2.0f * mSpacing;
    float bx0 = cx0 * mSlotCellSize - margin, by0 = cy0 * mSlotCellSize - margin;
    float bx1 = cx1 * mSlotCellSize + margin, by1 = cy1 * mSlotCellSize + margin;
    mActiveTargets.clear();
    for (int t = 0; t < (int)mTargets.size(); ++t) {
        Vec2 p = mTargets[t];
        if (p.x >= 0.0f && p.x >= bx0 && p.x <= bx1 && p.y >= by0 && p.y <= by1)
            mActiveTargets.push_back(t);
    }

    MarkDirty(cx0 * mSlotCellSize, cy0 * mSlotCellSize, cx1 * mSlotCellSize, cy1 * mSlotCellSize);
    mScheduleLength = iterations;
    mIteration      = 0;
    mLastBatchLoss  = 0.0f;
}

// One VoroLoss + Adam step: the ports of vorolossKernel and adamKernel.
void CpuVoroOptimizer::StepOnce()
{
    int t = ++mIteration;
    mActiveGrid.Build(mActiveSlots, mGen, kGridCellScale * mSpacing);

    const float frac = mParams.sampleFraction;
    float       lossSum = 0.0f;
    int         count   = 0;
    for (int ti : mActiveTargets) {
        if (frac < 1.0f && Rand01(unsigned(ti), unsigned(t), mParams.seed) >= frac)
            continue;
        Vec2        p = mTargets[ti];
        Knn<kLossK> knn;
        Search(mActiveGrid, p, knn);
        Search(mFrozenGrid, p, knn);
        int i0 = knn.idx[0];
        if (i0 < 0)
            continue;

        Vec2  c0 = mGen[i0];
        Vec2  u{p.x - c0.x, p.y - c0.y};
        float best = FLT_MAX, bestD = 0.0f;
        int   bestJ = -1;
        for (int k = 1; k < kLossK; ++k) {
            int j = knn.idx[k];
            if (j < 0)
                break;
            Vec2  e{mGen[j].x - c0.x, mGen[j].y - c0.y};
            float l = std::sqrt(Dot(e, e));
            if (l < 1e-12f)
                continue;
            float d = Dot(u, e) / l - 0.5f * l;
            if (d * d < best) {
                best  = d * d;
                bestD = d;
                bestJ = j;
            }
        }
        if (bestJ < 0)
            continue;
        lossSum += best;
        ++count;

        // Gradient (see vorolossKernel); frozen generators ignore theirs.
        Vec2  e{mGen[bestJ].x - c0.x, mGen[bestJ].y - c0.y};
        float l = std::sqrt(Dot(e, e));
        Vec2  n{e.x / l, e.y / l};
        float un = Dot(u, n);
        Vec2  dcj{(u.x - un * n.x) / l - 0.5f * n.x, (u.y - un * n.y) / l - 0.5f * n.y};
        float s = 2.0f * bestD;
        if (mActive[bestJ]) {
            mGrad[bestJ].x += s * dcj.x;
            mGrad[bestJ].y += s * dcj.y;
        }
        if (mActive[i0]) {
            mGrad[i0].x -= s * (n.x + dcj.x);
            mGrad[i0].y -= s * (n.y + dcj.y);
        }
    }
    mLastBatchLoss = count > 0 ? lossSum / count : 0.0f;

    // Adam + MultiStepLR + clamp.
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float       lr = mLr;
    for (int k = 0; k < 4; ++k)
        if (t - 1 >= mMilestones[k])
            lr *= mParams.gamma;
    float bc1 = 1.0f - std::pow(b1, float(t)), bc2 = 1.0f - std::pow(b2, float(t));
    float inv = 1.0f / float(std::max(count, 1));
    for (int i : mActiveSlots) {
        Vec2 g{mGrad[i].x * inv, mGrad[i].y * inv};
        mGrad[i] = Vec2{0, 0};
        Vec2 &m = mM[i], &v = mV[i], &p = mGen[i];
        m.x = b1 * m.x + (1.0f - b1) * g.x;
        m.y = b1 * m.y + (1.0f - b1) * g.y;
        v.x = b2 * v.x + (1.0f - b2) * g.x * g.x;
        v.y = b2 * v.y + (1.0f - b2) * g.y * g.y;
        p.x -= lr * (m.x / bc1) / (std::sqrt(v.x / bc2) + eps);
        p.y -= lr * (m.y / bc1) / (std::sqrt(v.y / bc2) + eps);
        p.x = std::clamp(p.x, 0.0f, mWidth);
        p.y = std::clamp(p.y, 0.0f, mHeight);
    }
}

float CpuVoroOptimizer::Step(int iterations)
{
    if (iterations <= 0 || NumSlots() == 0)
        return 0.0f;
    auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < iterations; ++k)
        StepOnce();
    return std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

float CpuVoroOptimizer::EvaluateLoss()
{
    mLive.clear();
    for (int i = 0; i < NumSlots(); ++i)
        if (mGen[i].x >= 0.0f)
            mLive.push_back(i);
    mAllGrid.Build(mLive, mGen, kGridCellScale * mSpacing);

    double sum   = 0.0;
    int    count = 0;
    for (Vec2 p : mTargets) {
        if (!(p.x >= 0.0f))
            continue;
        Knn<kLossK> knn;
        Search(mAllGrid, p, knn);
        int i0 = knn.idx[0];
        if (i0 < 0)
            continue;
        Vec2  c0 = mGen[i0];
        Vec2  u{p.x - c0.x, p.y - c0.y};
        float best = FLT_MAX;
        for (int k = 1; k < kLossK && knn.idx[k] >= 0; ++k) {
            Vec2  e{mGen[knn.idx[k]].x - c0.x, mGen[knn.idx[k]].y - c0.y};
            float l = std::sqrt(Dot(e, e));
            if (l < 1e-12f)
                continue;
            float d = Dot(u, e) / l - 0.5f * l;
            best    = std::min(best, d * d);
        }
        if (best < FLT_MAX) {
            sum += best;
            ++count;
        }
    }
    return count > 0 ? float(sum / count) : 0.0f;
}

// ---------------------------------------------------------------------------
// Extraction
// ---------------------------------------------------------------------------

// Port of voronoiCellKernel: clip the domain box by neighbour bisectors,
// certified by the security radius, then occupancy from the centroid.
void CpuVoroOptimizer::ClipCell(int i, const PointGrid &all, int &unsafe, int &overflow)
{
    Cell &cell = mCells[i];
    Vec2  ci   = mGen[i];
    if (ci.x < 0.0f) {
        cell.n      = 0;
        cell.inside = 0;
        return;
    }

    Knn<kCellK> knn;
    Search(all, ci, knn);

    Poly   poly;
    double x0 = -double(ci.x), y0 = -double(ci.y), x1 = mWidth - double(ci.x), y1 = mHeight - double(ci.y);
    poly.v[0] = {x0, y0}, poly.e[0] = -1;
    poly.v[1] = {x1, y0}, poly.e[1] = -2;
    poly.v[2] = {x1, y1}, poly.e[2] = -3;
    poly.v[3] = {x0, y1}, poly.e[3] = -4;
    poly.n    = 4;

    bool   ok = true, certified = false;
    double r2max = MaxRadius2(poly);
    for (int k = 0; k < kCellK; ++k) {
        int j = knn.idx[k];
        if (j < 0) {
            certified = true; // every generator has been seen
            break;
        }
        if (j == i)
            continue;
        if (double(knn.d2[k]) > 4.0 * r2max) {
            certified = true;
            break;
        }
        ok &= ClipByBisector(poly, {double(mGen[j].x) - ci.x, double(mGen[j].y) - ci.y}, j);
        r2max = MaxRadius2(poly);
    }
    if (!certified && double(knn.d2[kCellK - 1]) > 4.0 * r2max)
        certified = true;

    if (!certified) {
        ++unsafe;
        for (int j : mLive) {
            if (j == i || std::find(knn.idx, knn.idx + kCellK, j) != knn.idx + kCellK)
                continue;
            DVec2 r{double(mGen[j].x) - ci.x, double(mGen[j].y) - ci.y};
            if (r.x * r.x + r.y * r.y > 4.0 * r2max)
                continue;
            ok &= ClipByBisector(poly, r, j);
            r2max = MaxRadius2(poly);
        }
    }
    if (!ok)
        ++overflow;

    double area = 0.0, cx = 0.0, cy = 0.0;
    for (int k = 0; k < poly.n; ++k) {
        DVec2  a = poly.v[k], b = poly.v[(k + 1) % poly.n];
        double cr = a.x * b.y - b.x * a.y;
        area += cr;
        cx += (a.x + b.x) * cr;
        cy += (a.y + b.y) * cr;
    }
    Vec2 centroid = ci;
    if (std::fabs(area) > 1e-12) {
        centroid.x += float(cx / (3.0 * area));
        centroid.y += float(cy / (3.0 * area));
    }

    cell.inside = Occupied(centroid) ? 1 : 0;
    cell.n      = poly.n;
    for (int k = 0; k < poly.n; ++k) {
        cell.v[k] = Vec2{float(ci.x + poly.v[k].x), float(ci.y + poly.v[k].y)};
        cell.e[k] = poly.e[k];
    }
}

VoroMesh2D CpuVoroOptimizer::Extract()
{
    VoroMesh2D mesh;
    int        n = NumSlots();
    if (n == 0)
        return mesh;
    auto t0 = std::chrono::steady_clock::now();

    mLive.clear();
    for (int i = 0; i < n; ++i)
        if (mGen[i].x >= 0.0f)
            mLive.push_back(i);
    mAllGrid.Build(mLive, mGen, kGridCellScale * mSpacing);

    // A cell only depends on generators within twice its radius, and cells
    // here are about one spacing across, so re-clipping everything within
    // four spacings of the changes is conservative.
    int unsafe = 0, overflow = 0, clipped = 0;
    if (!mCellsValid) {
        mCells.assign(n, Cell{});
        for (int i = 0; i < n; ++i)
            ClipCell(i, mAllGrid, unsafe, overflow);
        clipped = n;
    } else if (mDirty) {
        float m  = 4.0f * mSpacing;
        float x0 = mDirtyBox[0] - m, y0 = mDirtyBox[1] - m, x1 = mDirtyBox[2] + m, y1 = mDirtyBox[3] + m;
        auto  in = [&](float x, float y) { return x >= x0 && x <= x1 && y >= y0 && y <= y1; };
        for (int i = 0; i < n; ++i) {
            float ox = (i % mSlotsW + 0.5f) * mSlotCellSize, oy = (i / mSlotsW + 0.5f) * mSlotCellSize;
            if (in(ox, oy) || (mGen[i].x >= 0.0f && in(mGen[i].x, mGen[i].y))) {
                ClipCell(i, mAllGrid, unsafe, overflow);
                ++clipped;
            }
        }
    }
    mCellsValid        = true;
    mDirty             = false;
    mesh.unsafeCells   = unsafe;
    mesh.overflowCells = overflow;
    mesh.clippedCells  = clipped;
    auto t1            = std::chrono::steady_clock::now();

    mesh.generators = mGen;
    mesh.inside.resize(n);
    for (int i = 0; i < n; ++i)
        mesh.inside[i] = mCells[i].inside;
    detail::AssembleMesh(mesh, n, [&](int i, int &c, const Vec2 *&v, const int *&e) {
        c = mCells[i].n;
        v = mCells[i].v;
        e = mCells[i].e;
    });

    auto t2         = std::chrono::steady_clock::now();
    mesh.cellsMs    = std::chrono::duration<float, std::milli>(t1 - t0).count();
    mesh.assemblyMs = std::chrono::duration<float, std::milli>(t2 - t1).count();
    return mesh;
}

} // namespace voromesh
