#pragma once

// ---------------------------------------------------------------------------
// canvas.h – the authoritative 2D shape: surface samples + inside/outside.
//
// Voxel (i, j) covers [i, i+1) x [j, j+1) * voxelSize. Voxels are grouped in
// 8x8 tiles; within a tile, voxel (i, j) is bit (j & 7) * 8 + (i & 7).
//   * Occupancy: a dense row-major array of tile masks over the domain, bit
//     set = voxel centre inside the object (128 KB at 1024^2).
//   * Surface: a sparse hash map of the tiles the boundary passes through.
//     Each surface voxel holds the mean of the boundary samples inside it (a
//     sub-voxel surface point) and its target slot: a stable index into the
//     flat target array the optimiser fits to.
// Both are addressed the same way: voxel (i, j) is tile (tx, ty) =
// (i >> 3, j >> 3), bit Bit(i, j), and
//   occupancy  mInside[ty * mTilesX + tx] >> bit & 1
//   surface    mSurface[TileKey(tx, ty)]: .mask >> bit & 1, .points[bit], .slots[bit]
// They hold independent facts (a surface voxel can be inside or outside), and
// each edit pass updates one of them.
// A target slot is a surface point's persistent ID: the point keeps it until
// it is deleted (a rim sample landing in an existing point's voxel moves that
// point in place), and freed slots are reused by later points.
// Edits report the target slots and the occupancy tiles they changed, so only
// those need to be copied to the optimiser.
// ---------------------------------------------------------------------------
#include "voromesh/shape2d.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace voromesh {

class Canvas
{
public:
    static constexpr int kTileLog2 = 3;
    static constexpr int kTileSize = 1 << kTileLog2; // voxels per tile side

    struct SurfaceTile
    {
        uint64_t mask = 0;
        Vec2     points[kTileSize * kTileSize];
        int      slots[kTileSize * kTileSize];
    };

    struct Edit
    {
        std::vector<int> dirtySlots;              // target slots whose value changed
        int tx0 = 0, ty0 = 0, tx1 = 0, ty1 = 0;   // occupancy tiles [tx0,tx1) x [ty0,ty1) rewritten
        int removed = 0, added = 0, flipped = 0;  // surface samples, occupancy voxels

        bool Changed() const { return removed + added + flipped > 0; }
    };

    Canvas(float width, float height, float voxelSize);

    /// Replaces the canvas with a closed polygon: its outline becomes the
    /// surface and its interior (by voxel centre) the occupancy.
    void RasterizeShape(const std::vector<Vec2> &polygon);

    /// Stamps a disc. Its interior becomes `inside` (false = carve, true =
    /// add), surface samples strictly inside it are deleted, and its rim
    /// becomes surface wherever it previously lay on the other side (inside
    /// the object when carving, outside when adding).
    Edit StampDisc(Vec2 center, float radius, bool inside);

    bool InDomain(int i, int j) const { return i >= 0 && j >= 0 && i < mVoxelsX && j < mVoxelsY; }
    bool IsInside(int i, int j) const;
    bool IsSurface(int i, int j) const;

    /// The flat target array, indexed by target slot; {-1, -1} marks a free slot.
    const std::vector<Vec2>     &TargetSlots() const { return mSlots; }
    /// Connected-component label per target slot, -1 for free slots and for
    /// points not labelled yet (every new point starts at -1).
    const std::vector<int>      &SlotLabels() const { return mSlotLabels; }
    void                         SetSlotLabel(int slot, int label) { mSlotLabels[slot] = label; }
    const std::vector<uint64_t> &InsideMasks() const { return mInside; }

    int    TilesX() const { return mTilesX; }
    int    TilesY() const { return mTilesY; }
    float  VoxelSize() const { return mVoxelSize; }
    size_t SurfaceCount() const { return mSurfaceCount; }
    size_t SurfaceTileCount() const { return mSurface.size(); }

private:
    static uint64_t TileKey(int tx, int ty) { return (uint64_t(uint32_t(ty)) << 32) | uint32_t(tx); }
    static int      Bit(int i, int j) { return ((j & (kTileSize - 1)) << kTileLog2) | (i & (kTileSize - 1)); }

    int  AllocSlot(Vec2 p);
    void FreeSlot(int slot);

    float mVoxelSize;
    int   mVoxelsX, mVoxelsY, mTilesX, mTilesY;

    std::vector<uint64_t>                     mInside;
    std::unordered_map<uint64_t, SurfaceTile> mSurface;
    size_t                                    mSurfaceCount = 0;

    std::vector<Vec2> mSlots;
    std::vector<int>  mSlotLabels; // parallel to mSlots
    std::vector<int>  mFreeSlots;
};

} // namespace voromesh
