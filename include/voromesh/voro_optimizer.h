#pragma once

// ---------------------------------------------------------------------------
// voro_optimizer.h – 2D VoroMesh (Maruani et al., ICCV 2023) on the GPU.
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
// Local editing: targets and generators live in fixed slots (x < 0 marks an
// empty slot), so an edit only uploads the slots it touched. Generators use
// one slot per cell of the Poisson sampler's grid, so a local Poisson
// resample maps to a rectangle of slots. BeginSchedule() can restrict the
// optimisation to such a rectangle; everything else stays frozen.
// ---------------------------------------------------------------------------
#include "voromesh/shape2d.h"

#include <array>
#include <cstdint>
#include <memory>
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
    // Full Voronoi diagram clipped to the domain: one convex polygon per
    // generator slot (empty slots have an empty polygon and x < 0).
    std::vector<Vec2>    generators;
    std::vector<uint8_t> inside;    // occupancy: cell centroid inside the canvas shape
    std::vector<int>     cellStart; // CSR offsets into cellVerts, size generators + 1
    std::vector<Vec2>    cellVerts;

    // Boundary mesh: Voronoi edges between cells of opposite occupancy. Vertices
    // are shared by key (the generator triple they are equidistant to), so the
    // connectivity is exact, not welded by distance. Each loop has positive
    // signed area in world coordinates (clockwise on screen, since y points down).
    std::vector<Vec2>               vertices;
    std::vector<std::array<int, 2>> edges;

    bool  watertight    = false; // every boundary vertex has exactly one in- and one out-edge
    int   loops         = 0;     // closed boundary curves
    int   unsafeCells   = 0;     // cells whose kNN clip could not be certified (brute-force fallback)
    int   overflowCells = 0;     // cells that exceeded the fixed vertex budget (should stay 0)
    float gpuMs         = 0.0f;  // hash grid + cell clipping + occupancy
    float hostMs        = 0.0f;  // download + boundary assembly
};

class VoroOptimizer
{
public:
    /// Domain is [0,width] x [0,height]. `spacing` is the expected generator
    /// spacing (the Poisson radius): it sizes the kNN hash grid and the lr.
    VoroOptimizer(float width, float height, float spacing, const OptimizerParams &params = {});
    ~VoroOptimizer();

    VoroOptimizer(const VoroOptimizer &) = delete;
    VoroOptimizer &operator=(const VoroOptimizer &) = delete;

    // ---- Targets: the points the Voronoi boundary should pass through -------
    void SetTargets(const std::vector<Vec2> &slots);
    /// Uploads slots[s] for every s in `dirty` (grows the GPU array if needed).
    void UpdateTargets(const std::vector<int> &dirty, const std::vector<Vec2> &slots);

    // ---- Occupancy: dense row-major 8x8-voxel tile masks (see Canvas) --------
    void SetOccupancy(const std::vector<uint64_t> &masks, int tilesX, int tilesY, float voxelSize);
    /// Uploads tiles [tx0,tx1) x [ty0,ty1) of `masks`.
    void UpdateOccupancy(const std::vector<uint64_t> &masks, int tx0, int ty0, int tx1, int ty1);

    // ---- Generators: row-major slotsW x slotsH slots --------------------------
    /// Uploads all slots and begins a global schedule of `iterations` steps.
    /// `slotCellSize` is the world size of one slot cell (the Poisson grid cell).
    void SetGenerators(const Vec2 *slots, int slotsW, int slotsH, float slotCellSize);
    /// Uploads slots [cx0,cx1) x [cy0,cy1) of the full row-major `slots`.
    void UpdateGenerators(const Vec2 *slots, int cx0, int cy0, int cx1, int cy1);

    /// Restarts Adam and the lr schedule, with the milestones scaled to
    /// `iterations` steps. Only generators whose slot lies in
    /// [cx0,cx1) x [cy0,cy1) move; targets farther than two spacings outside
    /// that rectangle are ignored.
    void BeginSchedule(int iterations, int cx0, int cy0, int cx1, int cy1);
    void BeginSchedule(int iterations) { BeginSchedule(iterations, 0, 0, mSlotsW, mSlotsH); }

    /// Runs `iterations` steps of the current schedule. Returns GPU time in ms.
    float Step(int iterations);
    /// Runs the remaining steps of the current schedule.
    float Optimize() { return Step(mScheduleLength - mIteration); }

    /// Mean VoroLoss over the subsampled targets of the last step.
    float LastBatchLoss() const { return mLastBatchLoss; }
    /// Mean VoroLoss over all targets for the current generators.
    float EvaluateLoss();

    /// Builds the clipped Voronoi diagram, occupancy and boundary mesh.
    VoroMesh2D Extract();

    int                    Iteration() const { return mIteration; }
    int                    ScheduleLength() const { return mScheduleLength; }
    int                    NumSlots() const { return mSlotsW * mSlotsH; }
    const OptimizerParams &Params() const { return mParams; }

private:
    struct Device;

    void BuildGraph();
    void EnqueueBuildGrid();
    void EnqueueIteration();

    float           mWidth, mHeight, mSpacing;
    OptimizerParams mParams;
    int             mSlotsW = 0, mSlotsH = 0;
    int             mTargetCapacity = 0;
    int             mIteration = 0, mScheduleLength = 0;
    float           mLastBatchLoss = 0.0f;

    std::unique_ptr<Device> mDev;
};

} // namespace voromesh
