#include "voromesh/canvas.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <tuple>

namespace voromesh {

namespace {

// Mean position of the curve samples that land in one voxel.
struct Accum
{
    double sx = 0.0, sy = 0.0;
    int    n  = 0;

    void Add(float x, float y)
    {
        sx += x;
        sy += y;
        ++n;
    }
    Vec2 Mean() const { return {float(sx / n), float(sy / n)}; }
};

uint64_t VoxelKey(int i, int j) { return (uint64_t(uint32_t(j)) << 32) | uint32_t(i); }
int      VoxelI(uint64_t k) { return int32_t(uint32_t(k)); }
int      VoxelJ(uint64_t k) { return int32_t(uint32_t(k >> 32)); }

} // namespace

Canvas::Canvas(float width, float height, float voxelSize)
    : mVoxelSize(voxelSize), mVoxelsX(int(std::ceil(width / voxelSize))), mVoxelsY(int(std::ceil(height / voxelSize))),
      mTilesX((mVoxelsX + kTileSize - 1) / kTileSize), mTilesY((mVoxelsY + kTileSize - 1) / kTileSize),
      mInside(size_t(mTilesX) * mTilesY, 0)
{
}

bool Canvas::IsInside(int i, int j) const
{
    return InDomain(i, j) && (mInside[size_t(j >> kTileLog2) * mTilesX + (i >> kTileLog2)] >> Bit(i, j) & 1);
}

bool Canvas::IsSurface(int i, int j) const
{
    auto it = mSurface.find(TileKey(i >> kTileLog2, j >> kTileLog2));
    return it != mSurface.end() && (it->second.mask >> Bit(i, j) & 1);
}

int Canvas::AllocSlot(Vec2 p)
{
    int slot;
    if (!mFreeSlots.empty()) {
        slot = mFreeSlots.back();
        mFreeSlots.pop_back();
    } else {
        slot = (int)mSlots.size();
        mSlots.push_back({});
    }
    mSlots[slot] = p;
    return slot;
}

void Canvas::FreeSlot(int slot)
{
    mSlots[slot] = {-1.0f, -1.0f};
    mFreeSlots.push_back(slot);
}

void Canvas::RasterizeShape(const std::vector<Vec2> &polygon)
{
    mSurface.clear();
    mSurfaceCount = 0;
    mSlots.clear();
    mFreeSlots.clear();
    std::fill(mInside.begin(), mInside.end(), 0);

    // Surface: walk every segment in steps of a quarter voxel.
    const float inv  = 1.0f / mVoxelSize;
    const float step = 0.25f * mVoxelSize;
    std::unordered_map<uint64_t, Accum> accum;
    for (size_t s = 0; s < polygon.size(); ++s) {
        Vec2  a = polygon[s], b = polygon[(s + 1) % polygon.size()];
        float len = std::hypot(b.x - a.x, b.y - a.y);
        int   n   = std::max(1, int(std::ceil(len / step)));
        for (int k = 0; k < n; ++k) { // b is the first sample of the next segment
            float t = float(k) / n;
            float x = a.x + t * (b.x - a.x), y = a.y + t * (b.y - a.y);
            int   i = int(std::floor(x * inv)), j = int(std::floor(y * inv));
            if (InDomain(i, j))
                accum[VoxelKey(i, j)].Add(x, y);
        }
    }

    // Assign target slots in row-major tile order so the GPU array starts out
    // spatially coherent.
    std::vector<uint64_t> voxels;
    voxels.reserve(accum.size());
    for (const auto &entry : accum)
        voxels.push_back(entry.first);
    auto order = [](uint64_t k) {
        int i = VoxelI(k), j = VoxelJ(k);
        return std::tuple(j >> kTileLog2, i >> kTileLog2, Bit(i, j));
    };
    std::sort(voxels.begin(), voxels.end(), [&](uint64_t a, uint64_t b) { return order(a) < order(b); });
    for (uint64_t k : voxels) {
        int          i = VoxelI(k), j = VoxelJ(k), bit = Bit(i, j);
        SurfaceTile &tile = mSurface[TileKey(i >> kTileLog2, j >> kTileLog2)];
        tile.mask |= uint64_t(1) << bit;
        tile.points[bit] = accum[k].Mean();
        tile.slots[bit]  = AllocSlot(tile.points[bit]);
        ++mSurfaceCount;
    }

    // Occupancy: scanline fill through voxel centres (even-odd rule).
    std::vector<float> xs;
    for (int j = 0; j < mVoxelsY; ++j) {
        float y = (j + 0.5f) * mVoxelSize;
        xs.clear();
        for (size_t s = 0, t = polygon.size() - 1; s < polygon.size(); t = s++) {
            Vec2 a = polygon[s], b = polygon[t];
            if ((a.y > y) != (b.y > y))
                xs.push_back(a.x + (y - a.y) * (b.x - a.x) / (b.y - a.y));
        }
        std::sort(xs.begin(), xs.end());
        for (size_t k = 0; k + 1 < xs.size(); k += 2) {
            int i0 = std::max(int(std::ceil(xs[k] * inv - 0.5f)), 0);
            int i1 = std::min(int(std::floor(xs[k + 1] * inv - 0.5f)), mVoxelsX - 1);
            for (int i = i0; i <= i1; ++i)
                mInside[size_t(j >> kTileLog2) * mTilesX + (i >> kTileLog2)] |= uint64_t(1) << Bit(i, j);
        }
    }
}

Canvas::Edit Canvas::StampDisc(Vec2 c, float radius, bool inside)
{
    Edit        e;
    const float inv = 1.0f / mVoxelSize;
    const float r2  = radius * radius;
    int i0 = std::max(int(std::floor((c.x - radius) * inv)), 0);
    int j0 = std::max(int(std::floor((c.y - radius) * inv)), 0);
    int i1 = std::min(int(std::floor((c.x + radius) * inv)), mVoxelsX - 1);
    int j1 = std::min(int(std::floor((c.y + radius) * inv)), mVoxelsY - 1);
    if (i0 > i1 || j0 > j1)
        return e;

    // 1. Rim samples that lie on the other side of the current occupancy
    //    (decided before the occupancy changes).
    std::unordered_map<uint64_t, Accum> rim;
    int n = std::max(16, int(std::ceil(2.0f * 3.14159265f * radius / (0.25f * mVoxelSize))));
    for (int k = 0; k < n; ++k) {
        float a = 2.0f * 3.14159265f * k / n;
        float x = c.x + radius * std::cos(a), y = c.y + radius * std::sin(a);
        int   i = int(std::floor(x * inv)), j = int(std::floor(y * inv));
        if (InDomain(i, j) && IsInside(i, j) != inside)
            rim[VoxelKey(i, j)].Add(x, y);
    }

    // 2. Delete surface samples strictly inside the disc.
    for (int ty = j0 >> kTileLog2; ty <= j1 >> kTileLog2; ++ty) {
        for (int tx = i0 >> kTileLog2; tx <= i1 >> kTileLog2; ++tx) {
            auto it = mSurface.find(TileKey(tx, ty));
            if (it == mSurface.end())
                continue;
            SurfaceTile &tile = it->second;
            for (uint64_t m = tile.mask; m; m &= m - 1) {
                int  bit = std::countr_zero(m);
                Vec2 p   = tile.points[bit];
                if ((p.x - c.x) * (p.x - c.x) + (p.y - c.y) * (p.y - c.y) >= r2)
                    continue;
                tile.mask &= ~(uint64_t(1) << bit);
                FreeSlot(tile.slots[bit]);
                e.dirtySlots.push_back(tile.slots[bit]);
                --mSurfaceCount;
                ++e.removed;
            }
            if (!tile.mask)
                mSurface.erase(it);
        }
    }

    // 3. Occupancy of voxel centres inside the disc.
    for (int j = j0; j <= j1; ++j) {
        float dy = (j + 0.5f) * mVoxelSize - c.y;
        for (int i = i0; i <= i1; ++i) {
            float dx = (i + 0.5f) * mVoxelSize - c.x;
            if (dx * dx + dy * dy >= r2)
                continue;
            uint64_t &word = mInside[size_t(j >> kTileLog2) * mTilesX + (i >> kTileLog2)];
            uint64_t  bit  = uint64_t(1) << Bit(i, j);
            if (bool(word & bit) != inside) {
                word ^= bit;
                ++e.flipped;
            }
        }
    }

    // 4. The kept rim becomes surface (replacing any old sample in its voxel).
    for (const auto &[k, acc] : rim) {
        int          i = VoxelI(k), j = VoxelJ(k), bit = Bit(i, j);
        SurfaceTile &tile = mSurface[TileKey(i >> kTileLog2, j >> kTileLog2)];
        Vec2         p    = acc.Mean();
        if (tile.mask >> bit & 1) {
            mSlots[tile.slots[bit]] = p;
        } else {
            tile.mask |= uint64_t(1) << bit;
            tile.slots[bit] = AllocSlot(p);
            ++mSurfaceCount;
            ++e.added;
        }
        tile.points[bit] = p;
        e.dirtySlots.push_back(tile.slots[bit]);
    }

    e.tx0 = i0 >> kTileLog2;
    e.ty0 = j0 >> kTileLog2;
    e.tx1 = (i1 >> kTileLog2) + 1;
    e.ty1 = (j1 >> kTileLog2) + 1;
    return e;
}

} // namespace voromesh
