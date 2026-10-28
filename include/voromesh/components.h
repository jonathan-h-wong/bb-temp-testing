#pragma once

// ---------------------------------------------------------------------------
// components.h – connected components of a VoroMesh2D, with persistent IDs.
//
// Only shell cells (inside cells with a boundary edge) are nodes; deep cells
// are never needed:
//   * Connectivity: two shell cells sharing a Voronoi edge are connected.
//     Walking a boundary loop never leaves its shell cells (at each boundary
//     vertex the next boundary edge belongs to the same cell or to the inside
//     neighbour across the shared edge), so every loop's cells end up in one
//     group. Union-find over these edges runs in parallel chunks; links always
//     point to the smaller slot, so roots are deterministic.
//   * Holes: a hole connects to its outer boundary only through the interior,
//     so a group of hole loops alone (negative signed area) is attached to the
//     smallest outer loop that contains it. A connected region has exactly one
//     outer loop, so each component is one outer group plus its holes.
//   * Mass properties: area, centre of mass and polar inertia from the
//     boundary edges (Green's theorem), per unit density.
//
// Identity. Labels persist on generator slots between updates:
//   1. Shell cells keep the label they had if they were shell cells before.
//      A cell that just became a shell cell (newly exposed or added material)
//      takes the label of the previous component whose region contains its
//      generator, if any.
//   2. The components are recomputed from scratch, and old labels are matched
//      to new components through those per-cell labels:
//        no old label                      Created (new ID)
//        one old label, one component      same ID (Changed if its shape did)
//        one old label, several components Split: the largest keeps the ID
//        several old labels, one component Merged: the largest old one survives
//        old label in no component         Destroyed
//      A merge and a split in the same update produce both events.
// Target points take the label of their owner cell (VoroMesh2D::targetOwner);
// an owner that is not a shell cell is located by region instead, and points
// with no owner stay unlabelled (-1).
// ---------------------------------------------------------------------------
#include "voromesh/parallel.h"
#include "voromesh/voro_optimizer.h"

#include <array>
#include <atomic>
#include <memory>
#include <vector>

namespace voromesh {

/// Rigid-body mass properties per unit density: area, centre of mass, and the
/// polar moment of inertia about the centre of mass.
struct MassProps
{
    float area    = 0.0f;
    Vec2  com     = {0.0f, 0.0f};
    float inertia = 0.0f;
};

/// Mass properties of disjoint parts taken as one body (parallel axis theorem).
MassProps Combine(const MassProps *parts, int count);

/// 2D rigid-body velocity: centre-of-mass velocity and angular velocity.
struct Kinematics
{
    Vec2  v = {0.0f, 0.0f};
    float w = 0.0f;
};

/// A piece split from a body, or the body after its shape (and centre of mass)
/// changed: it moves with the body's velocity field at its own centre of mass,
/// v + w x (pieceCom - bodyCom), and keeps the angular velocity.
Kinematics InheritKinematics(const Kinematics &body, Vec2 bodyCom, Vec2 pieceCom);

/// Bodies merged into one: linear and angular momentum of the parts are
/// conserved, the angular one about the merged centre of mass. `parts` are the
/// parts' mass properties before the merge, `merged` the merged body's.
Kinematics MergeKinematics(const MassProps *parts, const Kinematics *kin, int count, const MassProps &merged);

struct ComponentEvent
{
    enum class Type
    {
        Created,
        Destroyed,
        Changed,
        Split,
        Merged
    };
    Type type;
    int  id; // created / destroyed / changed component, split parent, merge survivor
    /// `id` before this update: for Split it is the parent as a whole (after
    /// any merge in the same update), which the pieces' velocities come from.
    MassProps        before;
    std::vector<int> others; // Split: the new pieces (id keeps the largest); Merged: the absorbed IDs
    std::vector<MassProps> othersBefore; // Merged: the absorbed components before this update
};

class ComponentTracker
{
public:
    struct Component
    {
        int       id;
        MassProps mass;
        int       loops; // boundary loops: 1 + holes
    };

    /// Parallel union-find on `exec` (not owned; nullptr = serial).
    void SetExecutor(Executor *exec) { mExec = exec; }

    /// Forgets every component and label; IDs restart at 0.
    void Clear();

    /// Recomputes components for `mesh`, carrying IDs over from the previous
    /// update. `targets` are the target slots `mesh.targetOwner` refers to.
    /// Returns the CPU time in ms.
    float Update(const VoroMesh2D &mesh, const std::vector<Vec2> &targets);

    const std::vector<ComponentEvent> &Events() const { return mEvents; }
    const std::vector<Component>      &Components() const { return mComponents; } // sorted by ID
    const Component                   *Find(int id) const;
    /// Per generator slot: component ID of shell cells, -1 for every other cell.
    const std::vector<int> &CellLabels() const { return mCellLabel; }
    /// Per target slot: component ID, -1 for free slots and unlabelled points.
    const std::vector<int> &TargetLabels() const { return mTargetLabel; }
    /// The component whose region contains p, or -1.
    int Locate(Vec2 p) const { return Locate(mRegions, p); }

    // Statistics of the last update
    int   InheritedCells() const { return mInheritedCells; } // new shell cells labelled by region
    int   AttachedHoles() const { return mAttachedHoles; }
    int   UnattachedHoles() const { return mUnattachedHoles; } // hole groups with no outer loop (should stay 0)
    int   OrphanTargets() const { return mOrphanTargets; }
    float UnionFindMs() const { return mUnionFindMs; }

private:
    // Boundary of one component, for point-in-region queries.
    struct Region
    {
        int                             id;
        Vec2                            lo, hi;
        std::vector<std::array<Vec2, 2>> edges;
    };

    static int Locate(const std::vector<Region> &regions, Vec2 p);
    int        Find(int x);
    void       Union(int a, int b);

    Executor *mExec = nullptr;
    int       mNextId = 0;

    std::vector<int>            mCellLabel, mTargetLabel;
    std::vector<Component>      mComponents;
    std::vector<Region>         mRegions;
    std::vector<ComponentEvent> mEvents;

    std::unique_ptr<std::atomic<int>[]> mParent; // union-find over generator slots
    int                                 mParentSize = 0;
    std::vector<int>                    mRoot;

    int   mInheritedCells = 0, mAttachedHoles = 0, mUnattachedHoles = 0, mOrphanTargets = 0;
    float mUnionFindMs = 0.0f;
};

} // namespace voromesh
