#pragma once

// ---------------------------------------------------------------------------
// cpu_voro_optimizer.h – single-threaded CPU twin of VoroOptimizer.
//
// Same algorithm, data layout and API (slots, x < 0 = empty; VoroLoss + Adam
// with the MultiStepLR schedule; bisector clipping with generator-triple
// vertex keys), but organised for small local edits instead of throughput:
//   * BeginSchedule() collects the targets near the active rectangle and the
//     active generators once; an iteration only visits those.
//   * Frozen generators are binned into a kNN grid once per schedule; only
//     the active ones are re-binned each iteration.
//   * Extract() keeps every cell polygon between calls and only re-clips the
//     cells near what changed since the previous extraction.
//   * Extract() only clips the surface band. A generator whose surroundings
//     (kBandSpacings around it) are uniformly inside or outside on the canvas
//     is "deep": its cell lies on one side of the surface, so it takes that
//     label and gets no polygon. Every clipped polygon then checks its
//     neighbours; a deep neighbour with the opposite label touches the
//     surface after all and is clipped too, so the boundary stays exact.
// "Uploads" are plain copies into the optimiser's own arrays.
// ---------------------------------------------------------------------------
#include "voromesh/voro_optimizer.h" // OptimizerParams, VoroMesh2D

#include <cstdint>
#include <vector>

namespace voromesh {

class CpuVoroOptimizer
{
public:
    CpuVoroOptimizer(float width, float height, float spacing, const OptimizerParams &params = {});

    void SetTargets(const std::vector<Vec2> &slots);
    void UpdateTargets(const std::vector<int> &dirty, const std::vector<Vec2> &slots);

    void SetOccupancy(const std::vector<uint64_t> &masks, int tilesX, int tilesY, float voxelSize);
    void UpdateOccupancy(const std::vector<uint64_t> &masks, int tx0, int ty0, int tx1, int ty1);

    void SetGenerators(const Vec2 *slots, int slotsW, int slotsH, float slotCellSize);
    void UpdateGenerators(const Vec2 *slots, int cx0, int cy0, int cx1, int cy1);

    void BeginSchedule(int iterations, int cx0, int cy0, int cx1, int cy1);
    void BeginSchedule(int iterations) { BeginSchedule(iterations, 0, 0, mSlotsW, mSlotsH); }

    /// Runs `iterations` steps of the current schedule. Returns CPU time in ms.
    float Step(int iterations);
    float Optimize() { return Step(mScheduleLength - mIteration); }

    float LastBatchLoss() const { return mLastBatchLoss; }
    float EvaluateLoss();

    /// Re-clips only the cells near changes since the last call (all cells the
    /// first time, or after SetGenerators / SetOccupancy).
    VoroMesh2D Extract();

    /// Half-width of the canvas window that decides deep vs. band, in spacings.
    static constexpr float kBandSpacings = 2.0f;

    int                    Iteration() const { return mIteration; }
    int                    ScheduleLength() const { return mScheduleLength; }
    int                    NumSlots() const { return mSlotsW * mSlotsH; }
    const OptimizerParams &Params() const { return mParams; }

    static constexpr int kMaxCellVerts = 24;

private:
    // Uniform grid over a subset of generators, built by counting sort.
    struct PointGrid
    {
        float             x0 = 0, y0 = 0, cellSize = 1;
        int               w = 0, h = 0;
        std::vector<int>  start; // w*h + 1
        std::vector<int>  idx;   // generator slot, in cell order
        std::vector<Vec2> pos;
        std::vector<int>  cellOf, cursor; // build scratch, kept to avoid reallocation

        void Build(const std::vector<int> &ids, const std::vector<Vec2> &gen, float cs);
    };

    struct Cell
    {
        int     n      = 0;
        uint8_t inside = 0;
        uint8_t band   = 0; // has a polygon
        Vec2    v[kMaxCellVerts];
        int     e[kMaxCellVerts]; // label of edge v[k] -> v[k+1]
    };

    template <int K> struct Knn;
    template <int K> void Search(const PointGrid &g, Vec2 q, Knn<K> &knn) const;

    void StepOnce();
    void ClipCell(int i, const PointGrid &all, int &unsafe, int &overflow);
    bool Occupied(Vec2 p) const;
    int  Classify(Vec2 g) const; // 0 deep outside, 1 deep inside, -1 surface band
    void UpdateTileClass(int tx0, int ty0, int tx1, int ty1);
    void MarkDirty(float x0, float y0, float x1, float y1);

    float           mWidth, mHeight, mSpacing;
    OptimizerParams mParams;

    std::vector<Vec2>     mTargets;
    std::vector<uint64_t> mOccupancy;
    std::vector<uint8_t>  mTileClass; // per tile: 0 all outside, 1 all inside, 2 mixed
    int                   mTilesX = 0, mTilesY = 0;
    float                 mInvVoxel = 0.0f, mVoxelSize = 1.0f;

    int               mSlotsW = 0, mSlotsH = 0;
    float             mSlotCellSize = 1.0f;
    std::vector<Vec2> mGen, mM, mV, mGrad;

    // Current schedule
    int                  mIteration = 0, mScheduleLength = 0;
    float                mLr = 0.0f;
    int                  mMilestones[4] = {};
    std::vector<uint8_t> mActive;        // per slot
    std::vector<int>     mActiveSlots;   // live slots that move
    std::vector<int>     mActiveTargets; // targets near the active rectangle
    PointGrid            mFrozenGrid, mActiveGrid;
    float                mLastBatchLoss = 0.0f;

    // Persistent extraction state
    std::vector<Cell> mCells;
    std::vector<int>  mLive; // scratch: live slots
    PointGrid         mAllGrid;
    bool              mCellsValid = false;
    bool              mDirty      = false;
    float             mDirtyBox[4] = {};
};

} // namespace voromesh
