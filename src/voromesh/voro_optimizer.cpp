// VoroMesh optimisation and extraction; see voro_optimizer.h for the design.

#include "voromesh/voro_optimizer.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <unordered_map>

namespace voromesh {

namespace {

constexpr int kLossK = 12; // neighbours per target point (paper: 16 in 3D)
constexpr int kCellK = 32; // neighbours tried per cell before the brute-force fallback

// Parallel grain. Chunking depends only on the work size, never on the thread
// count, so results are identical for any executor. Loss chunks hold at least
// kMinTargetsPerChunk targets (~6 sampled, a few us), and there are at most
// kMaxLossChunks of them because each one adds a gradient buffer to the serial
// reduction. An extraction chunk is up to 64 cells to classify and maybe clip.
constexpr size_t kMinTargetsPerChunk = 32;
constexpr size_t kMaxLossChunks      = 32;
constexpr size_t kSlotsPerChunk      = 64;

// Edge labels: >= 0 is the neighbouring generator slot, -1..-4 the domain box
// sides. Boundary vertex keys shift them up by kBoxLabels to stay unsigned.
constexpr int kBoxLabels = 4;

// kNN grid cell size in generator spacings. Ring search pays per cell visited,
// so cells holding a few generators each beat one-generator cells.
constexpr float kGridCellScale = 2.0f;

// Stateless hash: which targets a step subsamples depends only on (target,
// iteration, seed), never on which thread evaluates them.
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
    DVec2 v[VoroOptimizer::kMaxCellVerts];
    int   e[VoroOptimizer::kMaxCellVerts];
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
        if (out.n < VoroOptimizer::kMaxCellVerts) {
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
struct VoroOptimizer::Knn
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

void VoroOptimizer::PointGrid::Build(const std::vector<int> &ids, const std::vector<Vec2> &gen, float cs)
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
void VoroOptimizer::Search(const PointGrid &g, Vec2 q, Knn<K> &knn) const
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

VoroOptimizer::VoroOptimizer(float width, float height, float spacing, const OptimizerParams &params)
    : mWidth(width), mHeight(height), mSpacing(spacing), mParams(params)
{
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------
void VoroOptimizer::SetTargets(const std::vector<Vec2> &slots)
{
    mTargets     = slots;
    mOwnersValid = false;
}

void VoroOptimizer::UpdateTargets(const std::vector<int> &dirty, const std::vector<Vec2> &slots)
{
    if (slots.size() > mTargets.size())
        mTargets.resize(slots.size(), Vec2{-1.0f, -1.0f});
    for (int s : dirty)
        mTargets[s] = slots[s];
}

void VoroOptimizer::SetOccupancy(const std::vector<uint64_t> &masks, int tilesX, int tilesY, float voxelSize)
{
    mOccupancy  = masks;
    mTilesX     = tilesX;
    mTilesY     = tilesY;
    mVoxelSize  = voxelSize;
    mInvVoxel   = 1.0f / voxelSize;
    mCellsValid = false;
    mTileClass.resize(size_t(tilesX) * tilesY);
    UpdateTileClass(0, 0, tilesX, tilesY);
}

void VoroOptimizer::UpdateOccupancy(const std::vector<uint64_t> &masks, int tx0, int ty0, int tx1, int ty1)
{
    tx0 = std::max(tx0, 0), ty0 = std::max(ty0, 0);
    tx1 = std::min(tx1, mTilesX), ty1 = std::min(ty1, mTilesY);
    for (int ty = ty0; ty < ty1; ++ty)
        for (int tx = tx0; tx < tx1; ++tx)
            mOccupancy[size_t(ty) * mTilesX + tx] = masks[size_t(ty) * mTilesX + tx];
    UpdateTileClass(tx0, ty0, tx1, ty1);
    float tile = 8.0f * mVoxelSize;
    MarkDirty(tx0 * tile, ty0 * tile, tx1 * tile, ty1 * tile);
}

void VoroOptimizer::UpdateTileClass(int tx0, int ty0, int tx1, int ty1)
{
    for (int ty = ty0; ty < ty1; ++ty)
        for (int tx = tx0; tx < tx1; ++tx) {
            uint64_t w = mOccupancy[size_t(ty) * mTilesX + tx];
            mTileClass[size_t(ty) * mTilesX + tx] = w == 0 ? 0 : w == ~uint64_t(0) ? 1 : 2;
        }
}

// Deep if every canvas tile overlapping the square of half-width
// kBandSpacings * spacing around g has the same uniform label (tiles outside
// the domain count as outside). Whole tiles make the window slightly larger,
// which only errs towards the band.
int VoroOptimizer::Classify(Vec2 g) const
{
    float d   = kBandSpacings * mSpacing;
    int   tx0 = int(std::floor((g.x - d) * mInvVoxel)) >> 3, tx1 = int(std::floor((g.x + d) * mInvVoxel)) >> 3;
    int   ty0 = int(std::floor((g.y - d) * mInvVoxel)) >> 3, ty1 = int(std::floor((g.y + d) * mInvVoxel)) >> 3;
    int   cls = -1;
    for (int ty = ty0; ty <= ty1; ++ty)
        for (int tx = tx0; tx <= tx1; ++tx) {
            int c = (tx < 0 || ty < 0 || tx >= mTilesX || ty >= mTilesY) ? 0 : mTileClass[size_t(ty) * mTilesX + tx];
            if (c == 2)
                return -1;
            if (cls < 0)
                cls = c;
            else if (c != cls)
                return -1;
        }
    return cls;
}

bool VoroOptimizer::Occupied(Vec2 p) const
{
    int i = int(std::floor(p.x * mInvVoxel)), j = int(std::floor(p.y * mInvVoxel));
    if (i < 0 || j < 0)
        return false;
    int tx = i >> 3, ty = j >> 3;
    if (tx >= mTilesX || ty >= mTilesY)
        return false;
    return (mOccupancy[size_t(ty) * mTilesX + tx] >> (((j & 7) << 3) | (i & 7))) & 1;
}

void VoroOptimizer::SetGenerators(const Vec2 *slots, int slotsW, int slotsH, float slotCellSize)
{
    int n         = slotsW * slotsH;
    mSlotsW       = slotsW;
    mSlotsH       = slotsH;
    mSlotCellSize = slotCellSize;
    mGen.assign(slots, slots + n);
    mM.assign(n, Vec2{0, 0});
    mV.assign(n, Vec2{0, 0});
    mCellsValid = false;
    BeginSchedule(mParams.iterations);
}

void VoroOptimizer::UpdateGenerators(const Vec2 *slots, int cx0, int cy0, int cx1, int cy1)
{
    cx0 = std::max(cx0, 0), cy0 = std::max(cy0, 0);
    cx1 = std::min(cx1, mSlotsW), cy1 = std::min(cy1, mSlotsH);
    for (int cy = cy0; cy < cy1; ++cy)
        for (int cx = cx0; cx < cx1; ++cx)
            mGen[size_t(cy) * mSlotsW + cx] = slots[size_t(cy) * mSlotsW + cx];
    MarkDirty(cx0 * mSlotCellSize, cy0 * mSlotCellSize, cx1 * mSlotCellSize, cy1 * mSlotCellSize);
}

void VoroOptimizer::MarkDirty(float x0, float y0, float x1, float y1)
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
void VoroOptimizer::BeginSchedule(int iterations, int cx0, int cy0, int cx1, int cy1)
{
    cx0 = std::max(cx0, 0), cy0 = std::max(cy0, 0);
    cx1 = std::min(cx1, mSlotsW), cy1 = std::min(cy1, mSlotsH);

    mLr = mParams.lrPerSpacing * mSpacing;
    for (int k = 0; k < 4; ++k)
        mMilestones[k] = (int)std::lround(double(mParams.milestones[k]) * iterations / mParams.iterations);

    // Split live generators into active (reset Adam) and frozen (binned once).
    int              n = NumSlots();
    std::vector<int> frozen;
    mActiveLocal.assign(n, -1);
    mActiveSlots.clear();
    for (int i = 0; i < n; ++i) {
        if (mGen[i].x < 0.0f)
            continue;
        int sx = i % mSlotsW, sy = i / mSlotsW;
        if (sx >= cx0 && sx < cx1 && sy >= cy0 && sy < cy1) {
            mActiveLocal[i] = (int)mActiveSlots.size();
            mActiveSlots.push_back(i);
            mM[i] = mV[i] = Vec2{0, 0};
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
    mTargetsPerChunk = std::max(kMinTargetsPerChunk, (mActiveTargets.size() + kMaxLossChunks - 1) / kMaxLossChunks);
    int chunks       = int((mActiveTargets.size() + mTargetsPerChunk - 1) / mTargetsPerChunk);
    mChunkGrad.assign(size_t(chunks) * mActiveSlots.size(), Vec2{0, 0});
    mChunkLoss.assign(chunks, 0.0f);
    mChunkCount.assign(chunks, 0);

    MarkDirty(cx0 * mSlotCellSize, cy0 * mSlotCellSize, cx1 * mSlotCellSize, cy1 * mSlotCellSize);
    mScheduleLength = iterations;
    mIteration      = 0;
    mLastBatchLoss  = 0.0f;
}

// VoroLoss and its gradient for one chunk of the active targets. With
// u = p - c0, e = cj - c0, l = |e|, n = e / l:
//   d        = <u, n> - l / 2
//   dd/dcj   = (u - <u, n> n) / l - n / 2
//   dd/dc0   = -n - dd/dcj
// and dL/dc = 2 d dd/dc, summed here and divided by the number of contributing
// targets in FinishIteration (loss.mean() in the reference). Gradients go to the chunk's own buffer, so chunks can run
// concurrently; frozen generators ignore theirs.
void VoroOptimizer::LossChunk(int chunk, int iteration)
{
    const size_t A     = mActiveSlots.size();
    Vec2        *grad  = mChunkGrad.data() + size_t(chunk) * A;
    const float  frac  = mParams.sampleFraction;
    const size_t begin = size_t(chunk) * mTargetsPerChunk;
    const size_t end   = std::min(begin + mTargetsPerChunk, mActiveTargets.size());
    float        lossSum = 0.0f;
    int          count   = 0;
    for (size_t a = begin; a < end; ++a) {
        int ti = mActiveTargets[a];
        if (frac < 1.0f && Rand01(unsigned(ti), unsigned(iteration), mParams.seed) >= frac)
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

        Vec2  e{mGen[bestJ].x - c0.x, mGen[bestJ].y - c0.y};
        float l = std::sqrt(Dot(e, e));
        Vec2  n{e.x / l, e.y / l};
        float un = Dot(u, n);
        Vec2  dcj{(u.x - un * n.x) / l - 0.5f * n.x, (u.y - un * n.y) / l - 0.5f * n.y};
        float s = 2.0f * bestD;
        if (int lj = mActiveLocal[bestJ]; lj >= 0) {
            grad[lj].x += s * dcj.x;
            grad[lj].y += s * dcj.y;
        }
        if (int l0 = mActiveLocal[i0]; l0 >= 0) {
            grad[l0].x -= s * (n.x + dcj.x);
            grad[l0].y -= s * (n.y + dcj.y);
        }
    }
    mChunkLoss[chunk]  = lossSum;
    mChunkCount[chunk] = count;
}

// Serial start of an iteration: re-bin the active generators.
void VoroOptimizer::BeginIteration()
{
    ++mIteration;
    mActiveGrid.Build(mActiveSlots, mGen, kGridCellScale * mSpacing);
}

// Serial end of an iteration: sum the loss chunks in chunk order, then Adam
// with the MultiStepLR schedule, then clamp to the domain.
void VoroOptimizer::FinishIteration()
{
    const int t      = mIteration;
    const int chunks = (int)mChunkLoss.size();
    float lossSum = 0.0f;
    int   count   = 0;
    for (int c = 0; c < chunks; ++c) {
        lossSum += mChunkLoss[c];
        count += mChunkCount[c];
    }
    mLastBatchLoss = count > 0 ? lossSum / count : 0.0f;

    // Adam + MultiStepLR + clamp.
    const float  b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    const size_t A  = mActiveSlots.size();
    float        lr = mLr;
    for (int k = 0; k < 4; ++k)
        if (t - 1 >= mMilestones[k])
            lr *= mParams.gamma;
    float bc1 = 1.0f - std::pow(b1, float(t)), bc2 = 1.0f - std::pow(b2, float(t));
    float inv = 1.0f / float(std::max(count, 1));
    for (size_t a = 0; a < A; ++a) {
        Vec2 g{0.0f, 0.0f};
        for (int c = 0; c < chunks; ++c) {
            Vec2 &cg = mChunkGrad[size_t(c) * A + a];
            g.x += cg.x;
            g.y += cg.y;
            cg = Vec2{0.0f, 0.0f};
        }
        g.x *= inv;
        g.y *= inv;
        int   i = mActiveSlots[a];
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

float VoroOptimizer::Step(int iterations)
{
    if (iterations <= 0 || NumSlots() == 0)
        return 0.0f;
    auto t0 = std::chrono::steady_clock::now();
    // Each iteration is one round: re-bin, loss chunks in parallel, Adam.
    RunRounds(
        mExec, iterations, (int)mChunkLoss.size(), [this](int) { BeginIteration(); },
        [this](int, int c) { LossChunk(c, mIteration); }, [this](int) { FinishIteration(); });
    return std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

float VoroOptimizer::EvaluateLoss()
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

// Clip the domain box by the bisectors of the nearest neighbours (in double
// precision, relative to the generator), certified by the security radius
// |cj - ci| > 2 max|v - ci|, with a brute-force fallback; then occupancy from
// the canvas at the centroid.
void VoroOptimizer::ClipCell(int i, const PointGrid &all, int &unsafe, int &overflow)
{
    Cell &cell = mCells[i];
    Vec2  ci   = mGen[i];
    if (ci.x < 0.0f) {
        cell.n      = 0;
        cell.inside = 0;
        cell.band   = 0;
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
    cell.band   = 1;
    cell.n      = poly.n;
    for (int k = 0; k < poly.n; ++k) {
        cell.v[k] = Vec2{float(ci.x + poly.v[k].x), float(ci.y + poly.v[k].y)};
        cell.e[k] = poly.e[k];
    }
}

// The inside cell that owns surface point p: its nearest generator if that
// cell is inside, else the inside neighbour whose bisector with it is closest
// to p (the boundary edge p was fitted to, as in VoroLoss).
int VoroOptimizer::OwnerOf(Vec2 p) const
{
    Knn<kLossK> knn;
    Search(mAllGrid, p, knn);
    int i0 = knn.idx[0];
    if (i0 < 0 || mCells[i0].inside)
        return i0;
    Vec2  c0 = mGen[i0];
    Vec2  u{p.x - c0.x, p.y - c0.y};
    float best  = FLT_MAX;
    int   owner = -1;
    for (int k = 1; k < kLossK && knn.idx[k] >= 0; ++k) {
        int j = knn.idx[k];
        if (!mCells[j].inside)
            continue;
        Vec2  e{mGen[j].x - c0.x, mGen[j].y - c0.y};
        float l = std::sqrt(Dot(e, e));
        if (l < 1e-12f)
            continue;
        float d = std::fabs(Dot(u, e) / l - 0.5f * l);
        if (d < best) {
            best  = d;
            owner = j;
        }
    }
    return owner;
}

// CSR cells and the boundary mesh from the persistent cell polygons.
// `mesh.inside` must already be set.
void VoroOptimizer::Assemble(VoroMesh2D &mesh) const
{
    const int n = NumSlots();

    mesh.cellStart.assign(n + 1, 0);
    mesh.bandCells = 0;
    for (int i = 0; i < n; ++i) {
        mesh.cellStart[i + 1] = mesh.cellStart[i] + mCells[i].n;
        mesh.bandCells += mCells[i].n > 0;
    }
    mesh.cellVerts.resize(mesh.cellStart[n]);
    mesh.cellNeighbors.resize(mesh.cellStart[n]);
    for (int i = 0; i < n; ++i) {
        std::copy_n(mCells[i].v, mCells[i].n, mesh.cellVerts.begin() + mesh.cellStart[i]);
        std::copy_n(mCells[i].e, mCells[i].n, mesh.cellNeighbors.begin() + mesh.cellStart[i]);
    }

    // Boundary edges: every edge of an inside cell whose other side is an
    // outside cell or the domain box. A cell vertex sits between edges with
    // labels a and b, so it is equidistant to generators {i, a, b}; that
    // triple is its key, shared exactly by every cell that has the vertex.
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
    mesh.shell.assign(n, 0);
    mesh.shellCells = 0;
    for (int i = 0; i < n; ++i) {
        if (!mesh.inside[i]) continue;
        const Cell &cell = mCells[i];
        const int   c    = cell.n;
        for (int k = 0; k < c; ++k) {
            int l = cell.e[k];
            if (l >= 0 && mesh.inside[l]) continue;
            int prev = cell.e[(k + c - 1) % c], next = cell.e[(k + 1) % c];
            int a = vertex(key(i, prev, l), cell.v[k]);
            int b = vertex(key(i, l, next), cell.v[(k + 1) % c]);
            mesh.edges.push_back({a, b});
            mesh.edgeCell.push_back(i);
            if (!mesh.shell[i]) {
                mesh.shell[i] = 1;
                ++mesh.shellCells;
            }
        }
    }

    // Watertight iff each vertex has exactly one outgoing and one incoming edge.
    std::vector<int> next(mesh.vertices.size(), -1), out(mesh.vertices.size(), -1), inDeg(mesh.vertices.size(), 0);
    mesh.watertight = true;
    for (size_t e = 0; e < mesh.edges.size(); ++e) {
        auto [a, b] = mesh.edges[e];
        if (next[a] != -1) mesh.watertight = false;
        next[a] = b;
        out[a]  = (int)e;
        ++inDeg[b];
    }
    for (size_t v = 0; v < next.size(); ++v)
        if (next[v] == -1 || inDeg[v] != 1) mesh.watertight = false;
    mesh.edgeLoop.assign(mesh.edges.size(), -1);
    if (mesh.watertight) {
        std::vector<uint8_t> seen(next.size(), 0);
        for (size_t v = 0; v < next.size(); ++v) {
            if (seen[v]) continue;
            for (int w = (int)v; !seen[w]; w = next[w]) {
                seen[w]               = 1;
                mesh.edgeLoop[out[w]] = mesh.loops;
            }
            ++mesh.loops;
        }
    }
}

VoroMesh2D VoroOptimizer::Extract()
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

    // A cell only depends on generators within twice its radius (about a
    // spacing here), and the band test looks kBandSpacings around each
    // generator, so re-processing everything within four spacings of the
    // changes is conservative.
    // The same box decides which target owners are re-assigned.
    const float m  = 4.0f * mSpacing;
    const float x0 = mDirtyBox[0] - m, y0 = mDirtyBox[1] - m, x1 = mDirtyBox[2] + m, y1 = mDirtyBox[3] + m;
    auto        in = [&](float x, float y) { return mDirty && x >= x0 && x <= x1 && y >= y0 && y <= y1; };
    const bool  allCells = !mCellsValid;
    std::vector<int> todo;
    if (allCells) {
        mCells.assign(n, Cell{});
        todo.resize(n);
        for (int i = 0; i < n; ++i)
            todo[i] = i;
    } else if (mDirty) {
        for (int i = 0; i < n; ++i) {
            float ox = (i % mSlotsW + 0.5f) * mSlotCellSize, oy = (i / mSlotsW + 0.5f) * mSlotCellSize;
            if (in(ox, oy) || (mGen[i].x >= 0.0f && in(mGen[i].x, mGen[i].y)))
                todo.push_back(i);
        }
    }

    // Band cells are clipped; deep cells only take their canvas label. Each
    // chunk writes only its own cells and counters; results merge in order.
    struct ChunkOut
    {
        int              unsafe = 0, overflow = 0, clipped = 0;
        std::vector<int> band;
    };
    int                   chunks = int((todo.size() + kSlotsPerChunk - 1) / kSlotsPerChunk);
    std::vector<ChunkOut> outs(chunks);
    RunTasks(mExec, chunks, [&](int c) {
        ChunkOut    &out = outs[c];
        const size_t end = std::min(todo.size(), size_t(c + 1) * kSlotsPerChunk);
        for (size_t k = size_t(c) * kSlotsPerChunk; k < end; ++k) {
            int   i    = todo[k];
            Cell &cell = mCells[i];
            int   cls  = mGen[i].x < 0.0f ? 0 : Classify(mGen[i]);
            if (cls < 0) {
                ClipCell(i, mAllGrid, out.unsafe, out.overflow);
                ++out.clipped;
                out.band.push_back(i);
            } else {
                cell.n      = 0;
                cell.band   = 0;
                cell.inside = uint8_t(cls);
            }
        }
    });
    int              unsafe = 0, overflow = 0, clipped = 0, promoted = 0;
    std::vector<int> band; // cells clipped in this pass
    for (const ChunkOut &out : outs) {
        unsafe += out.unsafe;
        overflow += out.overflow;
        clipped += out.clipped;
        band.insert(band.end(), out.band.begin(), out.band.end());
    }

    // A deep neighbour with the opposite label touches the surface after all
    // (its cell is wider than the band test assumed): clip it too, and check
    // its neighbours in turn.
    for (size_t w = 0; w < band.size(); ++w) {
        int i = band[w];
        for (int k = 0; k < mCells[i].n; ++k) {
            int j = mCells[i].e[k];
            if (j < 0 || mCells[j].band || mCells[j].inside == mCells[i].inside)
                continue;
            ClipCell(j, mAllGrid, unsafe, overflow);
            ++clipped;
            ++promoted;
            band.push_back(j);
        }
    }

    mesh.unsafeCells   = unsafe;
    mesh.overflowCells = overflow;
    mesh.clippedCells  = clipped;
    mesh.promotedCells = promoted;
    auto t1            = std::chrono::steady_clock::now();

    // Owners of the target points near the changes (all of them after a reset).
    // Edits only add targets inside the changed occupancy tiles, so new slots
    // fall in the box too.
    const bool allTargets = allCells || !mOwnersValid;
    mTargetOwner.resize(mTargets.size(), -1);
    std::vector<int> ownerTodo;
    for (int t = 0; t < (int)mTargets.size(); ++t) {
        Vec2 p = mTargets[t];
        if (!(p.x >= 0.0f))
            mTargetOwner[t] = -1;
        else if (allTargets || in(p.x, p.y))
            ownerTodo.push_back(t);
    }
    RunTasks(mExec, int((ownerTodo.size() + kSlotsPerChunk - 1) / kSlotsPerChunk), [&](int c) {
        const size_t end = std::min(ownerTodo.size(), size_t(c + 1) * kSlotsPerChunk);
        for (size_t k = size_t(c) * kSlotsPerChunk; k < end; ++k)
            mTargetOwner[ownerTodo[k]] = OwnerOf(mTargets[ownerTodo[k]]);
    });
    mCellsValid  = true;
    mOwnersValid = true;
    mDirty       = false;
    auto t2      = std::chrono::steady_clock::now();

    mesh.generators = mGen;
    mesh.inside.resize(n);
    for (int i = 0; i < n; ++i)
        mesh.inside[i] = mCells[i].inside;
    mesh.targetOwner = mTargetOwner;
    Assemble(mesh);

    auto t3         = std::chrono::steady_clock::now();
    mesh.cellsMs    = std::chrono::duration<float, std::milli>(t1 - t0).count();
    mesh.ownersMs   = std::chrono::duration<float, std::milli>(t2 - t1).count();
    mesh.assemblyMs = std::chrono::duration<float, std::milli>(t3 - t2).count();
    return mesh;
}

} // namespace voromesh
