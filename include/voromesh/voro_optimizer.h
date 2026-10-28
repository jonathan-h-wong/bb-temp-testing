#pragma once

// ---------------------------------------------------------------------------
// voro_optimizer.h – 2D VoroMesh (Maruani et al., ICCV 2023) for local edits.
//
// A shape is represented by Voronoi generators with a binary occupancy; its
// boundary is the set of Voronoi edges between cells of opposite occupancy,
// which is watertight by construction. Generator positions are fitted to a
// target point cloud by minimising VoroLoss: for each target point p with
// nearest generator c0, the squared distance to the closest bisector of
// (c0, cj) over its k nearest generators cj,
//
//     d_j = <p - c0, cj - c0> / |cj - c0| - |cj - c0| / 2,   L(p) = min_j d_j^2,
//
// averaged over a random subset of the targets each iteration and minimised
// with Adam + MultiStepLR (the reference single_voromesh.py settings).
// Occupancy is set afterwards from each cell's centroid, as in the paper.
//
// Data layout: targets and generators live in fixed slots (x < 0 marks an
// empty slot), so an edit only rewrites the slots it touched. Generators use
// one slot per cell of the Poisson sampler's grid, so a local Poisson
// resample maps to a rectangle of slots. The optimiser keeps its own copies
// of the targets and the canvas occupancy; Update*() refresh parts of them.
//
// Organised for small local edits:
//   * BeginSchedule() restricts optimisation to a rectangle of slots (the
//     rest stay frozen) and collects the targets near it; an iteration only
//     visits those. Frozen generators are binned into a kNN grid once per
//     schedule; only the active ones are re-binned each iteration.
//   * Extract() keeps every cell polygon between calls and only re-clips the
//     cells near what changed since the previous extraction.
//   * Extract() only clips the surface band. A generator whose surroundings
//     (kBandSpacings around it) are uniformly inside or outside on the canvas
//     is "deep": its cell lies on one side of the surface, so it takes that
//     label and gets no polygon. Every clipped polygon then checks its
//     neighbours; a deep neighbour with the opposite label touches the
//     surface after all and is clipped too, so the boundary stays exact.
//   * With an Executor (parallel.h), each iteration's loss/gradient pass and
//     the band classification + clipping of Extract() run as fixed-size
//     chunks on it. Each loss chunk accumulates its own gradients; chunks are
//     summed in order, so results do not depend on the thread count.
// ---------------------------------------------------------------------------
#include "voromesh/parallel.h"
#include "voromesh/shape2d.h"

#include <array>
#include <cstdint>
#include <vector>

namespace voromesh {

struct OptimizerParams
{
    int                iterations      = 400;  // paper: 400 epochs
    int                localIterations = 100;  // after a local edit; milestones are scaled to fit
    float              lrPerSpacing    = 0.09f; // Adam lr in units of the seed radius. The paper's
                                                // 5e-3 on a 2/31 grid spacing is ~0.08 spacings.
    float              sampleFraction  = 0.2f;  // paper: random_mask
    float              gamma           = 0.5f;  // MultiStepLR decay
    std::array<int, 4> milestones      = {80, 150, 200, 250}; // for `iterations` steps
    unsigned           seed            = 1;     // target subsampling
};

struct VoroMesh2D
{
    // Voronoi cells clipped to the domain: one convex polygon per generator
    // slot. Empty slots (x < 0) have no polygon, and neither do cells outside
    // the surface band, whose occupancy comes from the canvas.
    std::vector<Vec2>    generators;
    std::vector<uint8_t> inside;    // occupancy: cell centroid inside the canvas shape
    std::vector<uint8_t> shell;     // inside cells with an edge on the boundary (the collision shell)
    std::vector<int>     cellStart; // CSR offsets into cellVerts, size generators + 1
    std::vector<Vec2>    cellVerts;
    std::vector<int>     cellNeighbors; // parallel to cellVerts: label of edge v[k] -> v[k+1],
                                        // the neighbouring slot or -1..-4 for the domain box

    // Boundary mesh: Voronoi edges between cells of opposite occupancy. Vertices
    // are shared by key (the generator triple they are equidistant to), so the
    // connectivity is exact, not welded by distance. Outer loops have positive
    // signed area in world coordinates (clockwise on screen, since y points
    // down) and holes negative.
    std::vector<Vec2>               vertices;
    std::vector<std::array<int, 2>> edges;
    std::vector<int>                edgeCell; // per edge: the inside (shell) cell it bounds
    std::vector<int>                edgeLoop; // per edge: its loop, or -1 if not watertight

    // Per target slot: the inside cell that owns the surface point (its nearest
    // generator if inside, else the inside neighbour whose bisector is closest),
    // or -1 for free slots and points with no inside cell nearby.
    std::vector<int> targetOwner;

    bool  watertight    = false; // every boundary vertex has exactly one in- and one out-edge
    int   loops         = 0;     // closed boundary curves
    int   unsafeCells   = 0;     // cells whose kNN clip could not be certified (brute-force fallback)
    int   overflowCells = 0;     // cells that exceeded the fixed vertex budget (should stay 0)
    int   clippedCells  = 0;     // cells (re)computed by this extraction
    int   bandCells     = 0;     // cells with a polygon
    int   shellCells    = 0;
    int   promotedCells = 0;     // band cells found by the neighbour-label check, not the canvas test
    float cellsMs       = 0.0f;  // classification + cell clipping + occupancy
    float ownersMs      = 0.0f;  // target owners near the changes
    float assemblyMs    = 0.0f;  // CSR cells + boundary assembly
};

class VoroOptimizer
{
public:
    /// Domain is [0,width] x [0,height]. `spacing` is the expected generator
    /// spacing (the Poisson radius): it sizes the kNN grids and the lr.
    VoroOptimizer(float width, float height, float spacing, const OptimizerParams &params = {});

    /// Runs the parallel loops on `exec` (not owned; nullptr = serial). It is
    /// only used inside Step(), Optimize() and Extract().
    void      SetExecutor(Executor *exec) { mExec = exec; }
    Executor *GetExecutor() const { return mExec; }

    // ---- Targets: the points the Voronoi boundary should pass through -------
    void SetTargets(const std::vector<Vec2> &slots);
    /// Copies slots[s] for every s in `dirty` (grows the target array if needed).
    void UpdateTargets(const std::vector<int> &dirty, const std::vector<Vec2> &slots);

    // ---- Occupancy: dense row-major 8x8-voxel tile masks (see Canvas) --------
    void SetOccupancy(const std::vector<uint64_t> &masks, int tilesX, int tilesY, float voxelSize);
    /// Copies tiles [tx0,tx1) x [ty0,ty1) of `masks`.
    void UpdateOccupancy(const std::vector<uint64_t> &masks, int tx0, int ty0, int tx1, int ty1);

    // ---- Generators: row-major slotsW x slotsH slots --------------------------
    /// Copies all slots and begins a global schedule of `iterations` steps.
    /// `slotCellSize` is the world size of one slot cell (the Poisson grid cell).
    void SetGenerators(const Vec2 *slots, int slotsW, int slotsH, float slotCellSize);
    /// Copies slots [cx0,cx1) x [cy0,cy1) of the full row-major `slots`.
    void UpdateGenerators(const Vec2 *slots, int cx0, int cy0, int cx1, int cy1);

    /// Restarts Adam and the lr schedule, with the milestones scaled to
    /// `iterations` steps. Only generators whose slot lies in
    /// [cx0,cx1) x [cy0,cy1) move; targets farther than two spacings outside
    /// that rectangle are ignored.
    void BeginSchedule(int iterations, int cx0, int cy0, int cx1, int cy1);
    void BeginSchedule(int iterations) { BeginSchedule(iterations, 0, 0, mSlotsW, mSlotsH); }

    /// Runs `iterations` steps of the current schedule. Returns CPU time in ms.
    float Step(int iterations);
    /// Runs the remaining steps of the current schedule.
    float Optimize() { return Step(mScheduleLength - mIteration); }

    /// Mean VoroLoss over the subsampled targets of the last step.
    float LastBatchLoss() const { return mLastBatchLoss; }
    /// Mean VoroLoss over all targets for the current generators.
    float EvaluateLoss();

    /// Re-clips only the cells near changes since the last call (all cells the
    /// first time, or after SetGenerators / SetOccupancy), and re-assigns the
    /// owners of the target points near them.
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
        int     e[kMaxCellVerts]; // label of edge v[k] -> v[k+1]: neighbour slot, or -1..-4 for the box
    };

    template <int K> struct Knn;
    template <int K> void Search(const PointGrid &g, Vec2 q, Knn<K> &knn) const;

    void BeginIteration();
    void FinishIteration();
    void LossChunk(int chunk, int iteration);
    void ClipCell(int i, const PointGrid &all, int &unsafe, int &overflow);
    int  OwnerOf(Vec2 p) const;
    void Assemble(VoroMesh2D &mesh) const;
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
    std::vector<Vec2> mGen, mM, mV;
    Executor         *mExec = nullptr;

    // Current schedule
    int                  mIteration = 0, mScheduleLength = 0;
    float                mLr = 0.0f;
    int                  mMilestones[4] = {};
    std::vector<int>     mActiveLocal;   // per slot: index into mActiveSlots, -1 if frozen
    std::vector<int>     mActiveSlots;   // live slots that move
    std::vector<int>     mActiveTargets; // targets near the active rectangle
    PointGrid            mFrozenGrid, mActiveGrid;
    float                mLastBatchLoss = 0.0f;

    // Loss chunks: chunk c covers mActiveTargets[c * mTargetsPerChunk, ...).
    size_t             mTargetsPerChunk = 1;
    std::vector<Vec2>  mChunkGrad;  // numChunks x mActiveSlots.size()
    std::vector<float> mChunkLoss;
    std::vector<int>   mChunkCount;

    // Persistent extraction state
    std::vector<Cell> mCells;
    std::vector<int>  mTargetOwner;
    std::vector<int>  mLive; // scratch: live slots
    PointGrid         mAllGrid;
    bool              mCellsValid  = false;
    bool              mOwnersValid = false;
    bool              mDirty      = false;
    float             mDirtyBox[4] = {};
};

} // namespace voromesh
