// ---------------------------------------------------------------------------
// voromesh_demo – interactive local editing of a 2D VoroMesh.
//
//   * Canvas (CPU): 1024^2 voxels holding the surface samples (sparse 8x8
//     tiles) and the inside/outside occupancy (dense bitmask). Starts as a heart.
//   * Targets: the canvas surface samples, one per target slot.
//   * Generators: Poisson disk samples, one slot per Poisson grid cell. The
//     Poisson radius (24 voxels) is deliberately much coarser than the
//     canvas: the Voronoi mesh is a lightweight physics proxy.
//   * VoroLoss + Adam, then watertight extraction of the surface band. The
//     parallel loops run on a Jolt JobSystemThreadPool, the pool physics uses.
//
// Each brush stamp runs the whole local pipeline:
//   1. Canvas: the disc interior becomes outside (carve) or inside (add),
//      surface samples inside the disc are deleted, and the disc rim becomes
//      surface where it lay on the other side (inside the object when carving).
//   2. Copy the changed target slots and occupancy tiles to the optimiser.
//   3. Poisson-resample the generator slots around the disc (the slots
//      outside keep their optimised positions).
//   4. Re-optimise only those generators.
//   5. Extract.
//   6. Connected components: persistent IDs on the shell cells and the canvas
//      surface points, with Created / Split / Merged / Destroyed events.
//   7. Re-render.
//
// Drawn: the interior bounded by the Voronoi boundary (stencil fill, so holes
// work), the shell cells (inside cells on the boundary: the future convex
// collision pieces) and their generators, the surface-band cell edges, the
// boundary itself, and the canvas surface points. With component colouring,
// each component gets a colour from its ID and a centre-of-mass marker, and
// unlabelled surface points are grey.
//
// Controls
//   Left drag     carve (disc -> outside)
//   Right drag    add (disc -> inside)
//   Wheel / [ ]   brush radius
//   - / =         fewer / more optimisation steps per stamp (0 shows the raw reseed)
//   R             reset to the heart (global optimisation)
//   F / V         toggle fill (interior + shell cells) / surface-band cell edges
//   G / T         toggle shell generators / canvas surface points
//   C             toggle component colouring
//   P             toggle parallel optimisation, extraction and components
//   Esc           quit (prints edit timing statistics)
//
// Usage: voromesh_demo [radius=24] [raster=1024] [localIterations=100] [--threads N] [--shot prefix]
//   radius            Poisson radius of the generators (domain is 1024 x 1024)
//   raster            canvas resolution across the domain (voxel = 1024 / raster)
//   localIterations   optimisation steps after each stamp
//   --threads         threads for the parallel loops (default: all hardware
//                     threads; 1 = serial)
//   --shot            run a scripted edit sequence (including a split, a chip,
//                     a merge and a removal), print timings, component events
//                     and checks, save <prefix>_before.png, <prefix>_step<k>.png
//                     and <prefix>_after.png, then exit
// ---------------------------------------------------------------------------
#include "poisson/poisson_disk_cpu.h"
#include "voromesh/canvas.h"
#include "voromesh/components.h"
#include "voromesh/jolt_executor.h"
#include "voromesh/shape2d.h"
#include "voromesh/voro_optimizer.h"

#include <Jolt/Jolt.h>

#include <Jolt/Core/Factory.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/PhysicsSettings.h> // cMaxPhysicsJobs / cMaxPhysicsBarriers
#include <Jolt/RegisterTypes.h>

#include <GLFW/glfw3.h> // also pulls in the system GL 1.1 header

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifndef GL_MULTISAMPLE
#define GL_MULTISAMPLE 0x809D // GL 1.3; Windows' gl.h only goes up to 1.1
#endif

static_assert(sizeof(poisson::Sample) == sizeof(voromesh::Vec2), "Poisson cells are copied as generator slots");

namespace {

using voromesh::Vec2;
using Clock = std::chrono::steady_clock;

constexpr float kDomain       = 1024.0f;
constexpr float kFlashSeconds = 0.5f; // how long a re-optimised region stays highlighted

double Ms(Clock::time_point a, Clock::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// ---------------------------------------------------------------------------
// Editor: canvas + Poisson sampler + optimiser, and the local edit pipeline
// ---------------------------------------------------------------------------
struct EditTiming
{
    double canvasMs = 0, inputsMs = 0, poissonMs = 0, optimizeMs = 0, extractMs = 0, componentsMs = 0, totalMs = 0;
    float  poissonInnerMs = 0, optimizeInnerMs = 0;    // as reported by the sampler / optimiser
    float  cellsMs = 0, ownersMs = 0, assemblyMs = 0;  // extraction split, as reported by the optimiser
    float  unionFindMs = 0;                            // as reported by the tracker
    int    removed = 0, added = 0, slots = 0, clipped = 0;
    int    labelUploads = 0;                           // surface points whose component label changed
};

struct Region
{
    int cx0, cy0, cx1, cy1; // Poisson cells / generator slots
};

class Editor
{
public:
    Editor(float radius, float voxel, const voromesh::OptimizerParams &params)
        : mRadius(radius), mLocalIterations(params.localIterations), mCanvas(kDomain, kDomain, voxel),
          mOpt(kDomain, kDomain, radius, params)
    {
    }

    /// Back to the heart: full rasterization, fresh Poisson seeds, global optimisation.
    float Reset(unsigned seed)
    {
        mCanvas.RasterizeShape(voromesh::MakeHeart(0.5f * kDomain, 0.5f * kDomain, 0.8f * kDomain, 4096));
        mOpt.SetTargets(mCanvas.TargetSlots());
        mOpt.SetOccupancy(mCanvas.InsideMasks(), mCanvas.TilesX(), mCanvas.TilesY(), mCanvas.VoxelSize());
        mSampler = std::make_unique<poisson::CpuSampler>(kDomain, kDomain, mRadius, 30, seed);
        mOpt.SetGenerators(Slots(), mSampler->GridW(), mSampler->GridH(), mSampler->CellSize());
        float ms = mOpt.Optimize();
        mMesh    = mOpt.Extract();
        mTracker.Clear();
        UpdateComponents();
        return ms;
    }

    /// One brush stamp through the whole local pipeline. Returns false (and
    /// does nothing else) if the stamp did not change the canvas.
    bool Stamp(Vec2 c, float brush, bool add, EditTiming &t, Region &region)
    {
        auto t0 = Clock::now();
        voromesh::Canvas::Edit e = mCanvas.StampDisc(c, brush, add);
        auto t1 = Clock::now();
        if (!e.Changed())
            return false;

        mOpt.UpdateTargets(e.dirtySlots, mCanvas.TargetSlots());
        mOpt.UpdateOccupancy(mCanvas.InsideMasks(), e.tx0, e.ty0, e.tx1, e.ty1);
        auto t2 = Clock::now();

        // Generators within two Poisson radii of the disc are replaced.
        float margin = brush + 2.0f * mRadius, cs = mSampler->CellSize();
        region.cx0 = std::max(int(std::floor((c.x - margin) / cs)), 0);
        region.cy0 = std::max(int(std::floor((c.y - margin) / cs)), 0);
        region.cx1 = std::min(int(std::ceil((c.x + margin) / cs)), mSampler->GridW());
        region.cy1 = std::min(int(std::ceil((c.y + margin) / cs)), mSampler->GridH());
        t.poissonInnerMs = mSampler->Resample(region.cx0, region.cy0, region.cx1, region.cy1);
        mOpt.UpdateGenerators(Slots(), region.cx0, region.cy0, region.cx1, region.cy1);
        auto t3 = Clock::now();

        mOpt.BeginSchedule(mLocalIterations, region.cx0, region.cy0, region.cx1, region.cy1);
        t.optimizeInnerMs = mOpt.Optimize();
        auto t4 = Clock::now();

        mMesh   = mOpt.Extract();
        auto t5 = Clock::now();

        t.labelUploads = UpdateComponents();
        auto t6        = Clock::now();

        t.canvasMs     = Ms(t0, t1);
        t.inputsMs     = Ms(t1, t2);
        t.poissonMs    = Ms(t2, t3);
        t.optimizeMs   = Ms(t3, t4);
        t.extractMs    = Ms(t4, t5);
        t.componentsMs = Ms(t5, t6);
        t.totalMs      = Ms(t0, t6);
        t.unionFindMs  = mTracker.UnionFindMs();
        t.ownersMs     = mMesh.ownersMs;
        t.removed    = e.removed;
        t.added      = e.added;
        t.slots      = (region.cx1 - region.cx0) * (region.cy1 - region.cy0);
        t.clipped    = mMesh.clippedCells;
        t.cellsMs    = mMesh.cellsMs;
        t.assemblyMs = mMesh.assemblyMs;
        return true;
    }

    std::vector<Vec2> SurfacePoints() const
    {
        std::vector<Vec2> pts;
        for (Vec2 p : mCanvas.TargetSlots())
            if (p.x >= 0.0f)
                pts.push_back(p);
        return pts;
    }

    const voromesh::Canvas           &Canvas() const { return mCanvas; }
    const voromesh::VoroMesh2D       &Mesh() const { return mMesh; }
    const voromesh::ComponentTracker &Tracker() const { return mTracker; }
    voromesh::VoroOptimizer          &Optimizer() { return mOpt; }
    const poisson::CpuSampler        &Sampler() const { return *mSampler; }
    float                             Radius() const { return mRadius; }
    int                               LocalIterations() const { return mLocalIterations; }
    void                              SetLocalIterations(int n) { mLocalIterations = n; }
    void                              SetExecutor(voromesh::Executor *exec)
    {
        mOpt.SetExecutor(exec);
        mTracker.SetExecutor(exec);
    }

private:
    const Vec2 *Slots() const { return reinterpret_cast<const Vec2 *>(mSampler->Cells().data()); }

    // Components of the new mesh, then the surface points' labels into the
    // canvas. Returns how many changed: the splat buffer entries to re-upload.
    int UpdateComponents()
    {
        mTracker.Update(mMesh, mCanvas.TargetSlots());
        const std::vector<int> &labels  = mTracker.TargetLabels();
        int                     changed = 0;
        for (int t = 0; t < (int)labels.size(); ++t)
            if (mCanvas.SlotLabels()[t] != labels[t]) {
                mCanvas.SetSlotLabel(t, labels[t]);
                ++changed;
            }
        return changed;
    }

    float                                mRadius;
    int                                  mLocalIterations; // optimisation steps per stamp
    voromesh::Canvas                     mCanvas;
    voromesh::VoroOptimizer              mOpt;
    voromesh::ComponentTracker           mTracker;
    std::unique_ptr<poisson::CpuSampler> mSampler;
    voromesh::VoroMesh2D                 mMesh;
};

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------
float SegmentDistance(Vec2 p, Vec2 a, Vec2 b)
{
    float ex = b.x - a.x, ey = b.y - a.y;
    float l2 = ex * ex + ey * ey;
    float t  = l2 > 0.0f ? std::clamp(((p.x - a.x) * ex + (p.y - a.y) * ey) / l2, 0.0f, 1.0f) : 0.0f;
    return std::hypot(p.x - a.x - t * ex, p.y - a.y - t * ey);
}

// Geometric accuracy of the extracted boundary against the canvas surface:
// surface point -> nearest boundary edge, and boundary vertex -> nearest point.
struct Accuracy
{
    float meanToMesh = 0, maxToMesh = 0, meanToTargets = 0, maxToTargets = 0;
};

Accuracy MeasureAccuracy(const voromesh::VoroMesh2D &mesh, const std::vector<Vec2> &targets)
{
    Accuracy acc;
    if (mesh.edges.empty() || targets.empty())
        return acc;
    for (Vec2 p : targets) {
        float best = 1e30f;
        for (auto [a, b] : mesh.edges)
            best = std::min(best, SegmentDistance(p, mesh.vertices[a], mesh.vertices[b]));
        acc.meanToMesh += best;
        acc.maxToMesh = std::max(acc.maxToMesh, best);
    }
    acc.meanToMesh /= targets.size();
    for (Vec2 v : mesh.vertices) {
        float best = 1e30f;
        for (Vec2 p : targets)
            best = std::min(best, std::hypot(p.x - v.x, p.y - v.y));
        acc.meanToTargets += best;
        acc.maxToTargets = std::max(acc.maxToTargets, best);
    }
    acc.meanToTargets /= mesh.vertices.size();
    return acc;
}

void PrintMesh(const char *label, Editor &editor)
{
    const voromesh::VoroMesh2D &mesh  = editor.Mesh();
    float                       voxel = editor.Canvas().VoxelSize();
    float                       loss  = editor.Optimizer().EvaluateLoss();
    Accuracy acc  = MeasureAccuracy(mesh, editor.SurfacePoints());
    int      live = 0, inside = 0;
    for (size_t i = 0; i < mesh.generators.size(); ++i) {
        live += mesh.generators[i].x >= 0.0f;
        inside += mesh.inside[i];
    }
    std::printf("[%s]\n", label);
    std::printf("  canvas                   %zu surface points in %zu tiles\n", editor.Canvas().SurfaceCount(),
                editor.Canvas().SurfaceTileCount());
    std::printf("  VoroLoss (all targets)   %.4f  (RMS %.3f voxels)\n", loss, std::sqrt(loss) / voxel);
    std::printf("  cells                    %d live of %zu slots (%d inside), unsafe %d, overflow %d\n", live,
                mesh.generators.size(), inside, mesh.unsafeCells, mesh.overflowCells);
    std::printf("  surface band             %d cells with polygons, %d shell cells, %d promoted last pass\n",
                mesh.bandCells, mesh.shellCells, mesh.promotedCells);
    std::printf("  boundary                 %zu vertices, %zu edges, %d loop(s), %s\n", mesh.vertices.size(),
                mesh.edges.size(), mesh.loops, mesh.watertight ? "WATERTIGHT" : "NOT WATERTIGHT");
    std::printf("  surface -> mesh          mean %.3f  max %.3f voxels\n", acc.meanToMesh / voxel,
                acc.maxToMesh / voxel);
    std::printf("  mesh vertex -> surface   mean %.3f  max %.3f voxels\n", acc.meanToTargets / voxel,
                acc.maxToTargets / voxel);
    const voromesh::ComponentTracker &tr = editor.Tracker();
    std::printf("  components               %zu (holes attached %d, unattached %d), %d unlabelled surface points\n",
                tr.Components().size(), tr.AttachedHoles(), tr.UnattachedHoles(), tr.OrphanTargets());
    for (const auto &c : tr.Components())
        std::printf("    #%-3d area %9.1f  com (%6.1f, %6.1f)  inertia %.4g  loops %d\n", c.id, c.mass.area,
                    c.mass.com.x, c.mass.com.y, c.mass.inertia, c.loops);
}

// Component events of the last update, except plain shape changes (counted).
void PrintEvents(const voromesh::ComponentTracker &tr)
{
    using Type   = voromesh::ComponentEvent::Type;
    int changed  = 0;
    auto list    = [](const std::vector<int> &ids) {
        std::string s;
        for (int id : ids)
            s += (s.empty() ? "#" : ", #") + std::to_string(id);
        return s;
    };
    for (const voromesh::ComponentEvent &ev : tr.Events()) {
        switch (ev.type) {
        case Type::Changed: ++changed; break;
        case Type::Created: std::printf("      created   #%d\n", ev.id); break;
        case Type::Destroyed:
            std::printf("      destroyed #%d (area %.1f)\n", ev.id, ev.before.area);
            break;
        case Type::Split:
            std::printf("      split     #%d (area %.1f) -> keeps #%d, new %s\n", ev.id, ev.before.area, ev.id,
                        list(ev.others).c_str());
            break;
        case Type::Merged:
            std::printf("      merged    %s into #%d\n", list(ev.others).c_str(), ev.id);
            break;
        }
    }
    if (changed > 0 && changed != (int)tr.Events().size())
        std::printf("      (+%d changed)\n", changed);
}

// Split pieces inherit the parent's velocity field; merging them back must
// return the parent's motion (about the pieces' combined centre of mass).
// Returns the largest error over the split events of the last update.
float CheckSplitKinematics(const voromesh::ComponentTracker &tr)
{
    const voromesh::Kinematics parent{{30.0f, -12.0f}, 0.4f};
    float                      worst = 0.0f;
    for (const voromesh::ComponentEvent &ev : tr.Events()) {
        if (ev.type != voromesh::ComponentEvent::Type::Split)
            continue;
        std::vector<int> ids = ev.others;
        ids.push_back(ev.id);
        std::vector<voromesh::MassProps>  parts;
        std::vector<voromesh::Kinematics> kin;
        for (int id : ids) {
            parts.push_back(tr.Find(id)->mass);
            kin.push_back(voromesh::InheritKinematics(parent, ev.before.com, parts.back().com));
        }
        voromesh::MassProps  whole  = voromesh::Combine(parts.data(), (int)parts.size());
        voromesh::Kinematics merged = voromesh::MergeKinematics(parts.data(), kin.data(), (int)parts.size(), whole);
        voromesh::Kinematics expect = voromesh::InheritKinematics(parent, ev.before.com, whole.com);
        worst = std::max({worst, std::fabs(merged.v.x - expect.v.x), std::fabs(merged.v.y - expect.v.y),
                          std::fabs(merged.w - expect.w)});
    }
    return worst;
}

// Cross-check against the canvas: 4-connected components of the inside voxels.
// Every cell of a mesh component must sit in one canvas component, and no two
// mesh components may share one. Canvas components the mesh misses entirely
// (features thinner than the generator spacing) are counted, not errors.
bool VerifyComponents(Editor &editor)
{
    const voromesh::Canvas     &canvas = editor.Canvas();
    const voromesh::VoroMesh2D &mesh   = editor.Mesh();
    const voromesh::ComponentTracker &tr = editor.Tracker();
    const int W = canvas.TilesX() * voromesh::Canvas::kTileSize, H = canvas.TilesY() * voromesh::Canvas::kTileSize;
    std::vector<int> label(size_t(W) * H, -1);
    std::vector<int> area, stack;
    for (int j0 = 0; j0 < H; ++j0)
        for (int i0 = 0; i0 < W; ++i0) {
            if (label[size_t(j0) * W + i0] >= 0 || !canvas.IsInside(i0, j0))
                continue;
            int id = (int)area.size();
            area.push_back(0);
            label[size_t(j0) * W + i0] = id;
            stack.push_back(j0 * W + i0);
            while (!stack.empty()) {
                int v = stack.back(), i = v % W, j = v / W;
                stack.pop_back();
                ++area[id];
                const int ni[4] = {i - 1, i + 1, i, i}, nj[4] = {j, j, j - 1, j + 1};
                for (int k = 0; k < 4; ++k)
                    if (canvas.IsInside(ni[k], nj[k]) && label[size_t(nj[k]) * W + ni[k]] < 0) {
                        label[size_t(nj[k]) * W + ni[k]] = id;
                        stack.push_back(nj[k] * W + ni[k]);
                    }
            }
        }

    // Each shell cell's centroid (inside the canvas shape by construction of
    // the occupancy) votes for the canvas component it lies in.
    float voxel = canvas.VoxelSize();
    std::unordered_map<int, int> canvasOf; // mesh component ID -> canvas component
    int split = 0, shared = 0, offCanvas = 0;
    for (size_t c = 0; c < mesh.generators.size(); ++c) {
        int id = tr.CellLabels()[c];
        if (id < 0)
            continue;
        double a = 0, cx = 0, cy = 0;
        for (int s = mesh.cellStart[c], e = mesh.cellStart[c + 1], k = s; k < e; ++k) {
            Vec2   p = mesh.cellVerts[k], q = mesh.cellVerts[k + 1 < e ? k + 1 : s];
            double cr = double(p.x) * q.y - double(q.x) * p.y;
            a += cr, cx += (p.x + q.x) * cr, cy += (p.y + q.y) * cr;
        }
        int i = int(cx / (3 * a) / voxel), j = int(cy / (3 * a) / voxel);
        int cc = (i >= 0 && j >= 0 && i < W && j < H) ? label[size_t(j) * W + i] : -1;
        if (cc < 0) {
            ++offCanvas;
            continue;
        }
        auto [it, inserted] = canvasOf.try_emplace(id, cc);
        split += !inserted && it->second != cc;
    }
    std::vector<int> owners(area.size(), 0);
    for (auto [id, cc] : canvasOf)
        shared += ++owners[cc] == 2;
    int   missed = 0;
    float r      = editor.Radius() / voxel;
    for (size_t cc = 0; cc < area.size(); ++cc)
        missed += owners[cc] == 0 && area[cc] >= 3.14159f * r * r; // at least a generator disc
    bool ok = split == 0 && shared == 0 && offCanvas == 0;
    std::printf("    verify: %zu mesh / %zu canvas components | %s (split %d, shared %d, off-canvas cells %d), "
                "%d canvas components >= one generator disc missed\n",
                tr.Components().size(), area.size(), ok ? "CONSISTENT" : "MISMATCH", split, shared, offCanvas,
                missed);
    return ok;
}

// Order-sensitive hash of every label, to compare runs with different thread counts.
uint32_t LabelChecksum(const Editor &editor)
{
    uint32_t h    = 2166136261u;
    auto     feed = [&](int v) { h = (h ^ uint32_t(v)) * 16777619u; };
    for (int v : editor.Tracker().CellLabels()) feed(v);
    for (int v : editor.Canvas().SlotLabels()) feed(v);
    for (const auto &c : editor.Tracker().Components()) feed(c.id), feed(int(c.mass.area));
    return h;
}

struct EditStats
{
    std::vector<EditTiming> edits;

    void Print() const
    {
        if (edits.empty())
            return;
        auto stat = [&](const char *name, double EditTiming::*field) {
            std::vector<double> v;
            for (const EditTiming &e : edits)
                v.push_back(e.*field);
            std::sort(v.begin(), v.end());
            double mean = 0;
            for (double x : v)
                mean += x;
            mean /= v.size();
            std::printf("  %-22s mean %7.3f  median %7.3f  p95 %7.3f  max %7.3f ms\n", name, mean, v[v.size() / 2],
                        v[std::min(v.size() - 1, size_t(v.size() * 0.95))], v.back());
        };
        std::printf("Edit timings over %zu stamps (wall clock, incl. sync):\n", edits.size());
        stat("canvas stamp (CPU)", &EditTiming::canvasMs);
        stat("targets+occupancy in", &EditTiming::inputsMs);
        stat("poisson resample", &EditTiming::poissonMs);
        stat("local optimisation", &EditTiming::optimizeMs);
        stat("extract", &EditTiming::extractMs);
        stat("components", &EditTiming::componentsMs);
        stat("TOTAL", &EditTiming::totalMs);
    }
};

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
struct Input
{
    bool   reset       = false;
    bool   toggleFill  = false;
    bool   toggleEdges = false;
    bool   toggleGens  = false;
    bool   toggleTgts  = false;
    bool   toggleComps = false;
    bool   toggleParallel = false;
    int    brushStep   = 0;
    int    iterStep    = 0;
    double scroll      = 0.0;
};

void KeyCallback(GLFWwindow *window, int key, int, int action, int)
{
    if (action != GLFW_PRESS && action != GLFW_REPEAT)
        return;
    auto *in = static_cast<Input *>(glfwGetWindowUserPointer(window));
    switch (key) {
    case GLFW_KEY_ESCAPE:        glfwSetWindowShouldClose(window, GLFW_TRUE); break;
    case GLFW_KEY_R:             in->reset = true; break;
    case GLFW_KEY_F:             in->toggleFill = true; break;
    case GLFW_KEY_V:             in->toggleEdges = true; break;
    case GLFW_KEY_G:             in->toggleGens = true; break;
    case GLFW_KEY_T:             in->toggleTgts = true; break;
    case GLFW_KEY_C:             in->toggleComps = true; break;
    case GLFW_KEY_P:             in->toggleParallel = true; break;
    case GLFW_KEY_RIGHT_BRACKET: in->brushStep = +1; break;
    case GLFW_KEY_LEFT_BRACKET:  in->brushStep = -1; break;
    case GLFW_KEY_EQUAL:         in->iterStep = +1; break;
    case GLFW_KEY_MINUS:         in->iterStep = -1; break;
    default: break;
    }
}

void ScrollCallback(GLFWwindow *window, double, double yoff)
{
    static_cast<Input *>(glfwGetWindowUserPointer(window))->scroll += yoff;
}

// A distinct, stable colour per component ID (golden-ratio hue steps).
std::array<float, 3> ComponentColor(int id)
{
    float h = std::fmod(0.08f + 0.618034f * float(id), 1.0f) * 6.0f, s = 0.6f, v = 0.95f;
    float c = v * s, x = c * (1.0f - std::fabs(std::fmod(h, 2.0f) - 1.0f)), m = v - c;
    float r = 0, g = 0, b = 0;
    switch (int(h)) {
    case 0: r = c, g = x; break;
    case 1: r = x, g = c; break;
    case 2: g = c, b = x; break;
    case 3: g = x, b = c; break;
    case 4: r = x, b = c; break;
    default: r = c, b = x; break;
    }
    return {r + m, g + m, b + m};
}

// Triangles/lines for the legacy-GL renderer, rebuilt whenever the mesh
// changes. Everything that belongs to a component is batched per component.
struct DrawBuffers
{
    struct Batch
    {
        int                  id;
        std::array<float, 3> rgb;
        Vec2                 com;
        std::vector<float>   interiorFan, shellTris, generators, surface;
    };
    std::vector<Batch> batches;
    std::vector<float> cellLines, boundaryLines, unlabelled;

    void Build(const voromesh::VoroMesh2D &mesh, const voromesh::ComponentTracker &tracker,
               const voromesh::Canvas &canvas)
    {
        batches.clear();
        cellLines.clear();
        boundaryLines.clear();
        unlabelled.clear();
        std::unordered_map<int, size_t> batchOf;
        for (const auto &c : tracker.Components()) {
            batchOf[c.id] = batches.size();
            batches.push_back({c.id, ComponentColor(c.id), c.mass.com, {}, {}, {}, {}});
        }
        auto batch = [&](int id) -> Batch & { return batches[batchOf.at(id)]; };

        for (size_t i = 0; i + 1 < mesh.cellStart.size(); ++i) {
            int         s = mesh.cellStart[i], n = mesh.cellStart[i + 1] - s;
            const Vec2 *v = mesh.cellVerts.data() + s;
            if (mesh.shell[i]) {
                Batch &b = batch(tracker.CellLabels()[i]);
                for (int k = 1; k + 1 < n; ++k)
                    b.shellTris.insert(b.shellTris.end(), {v[0].x, v[0].y, v[k].x, v[k].y, v[k + 1].x, v[k + 1].y});
                b.generators.insert(b.generators.end(), {mesh.generators[i].x, mesh.generators[i].y});
            }
            for (int k = 0; k < n; ++k)
                cellLines.insert(cellLines.end(), {v[k].x, v[k].y, v[(k + 1) % n].x, v[(k + 1) % n].y});
        }
        // Each interior is drawn with an even-odd stencil: one triangle from
        // the origin to every boundary edge of the component covers each of its
        // interior points an odd number of times, holes included.
        for (size_t e = 0; e < mesh.edges.size(); ++e) {
            Vec2   p = mesh.vertices[mesh.edges[e][0]], q = mesh.vertices[mesh.edges[e][1]];
            Batch &b = batch(tracker.CellLabels()[mesh.edgeCell[e]]);
            boundaryLines.insert(boundaryLines.end(), {p.x, p.y, q.x, q.y});
            b.interiorFan.insert(b.interiorFan.end(), {0.0f, 0.0f, p.x, p.y, q.x, q.y});
        }
        // Surface points by the label stored in the canvas.
        const std::vector<Vec2> &pts = canvas.TargetSlots();
        for (size_t t = 0; t < pts.size(); ++t) {
            if (!(pts[t].x >= 0.0f))
                continue;
            int                 id  = canvas.SlotLabels()[t];
            std::vector<float> &dst = id >= 0 ? batch(id).surface : unlabelled;
            dst.insert(dst.end(), {pts[t].x, pts[t].y});
        }
    }
};

void DrawArray(const std::vector<float> &xy, GLenum mode)
{
    if (xy.empty())
        return;
    glVertexPointer(2, GL_FLOAT, 0, xy.data());
    glDrawArrays(mode, 0, GLsizei(xy.size() / 2));
}

void DrawCircle(Vec2 c, float r)
{
    std::vector<float> xy;
    for (int k = 0; k < 64; ++k) {
        float a = 2.0f * 3.14159265f * k / 64;
        xy.insert(xy.end(), {c.x + r * std::cos(a), c.y + r * std::sin(a)});
    }
    DrawArray(xy, GL_LINE_LOOP);
}

bool SavePng(const std::string &path, int w, int h)
{
    std::vector<unsigned char> px(size_t(w) * h * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_BACK);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, px.data());
    stbi_flip_vertically_on_write(1);
    return stbi_write_png(path.c_str(), w, h, 3, px.data(), w * 3) != 0;
}

struct Flash
{
    float x0, y0, x1, y1, age;
};

void PrintEdit(const char *what, Vec2 c, const EditTiming &t, int iterations)
{
    std::printf("  %-5s (%4.0f,%4.0f): -%3d +%3d pts, %3d slots | canvas %.3f  in %.3f  poisson %.3f  "
                "opt %.3f (%d it)  extract %.3f (%4d cells %.3f + owners %.3f + assembly %.3f)  "
                "comps %.3f (uf %.3f, %d labels) | total %.3f ms\n",
                what, c.x, c.y, t.removed, t.added, t.slots, t.canvasMs, t.inputsMs, t.poissonMs, t.optimizeMs,
                iterations, t.extractMs, t.clipped, t.cellsMs, t.ownersMs, t.assemblyMs, t.componentsMs,
                t.unionFindMs, t.labelUploads, t.totalMs);
}

struct Options
{
    float                     radius = 24.0f;
    int                       raster = 1024;
    voromesh::OptimizerParams params;
    std::string               shot;
    int                       threads = 0; // 0 = all hardware threads
};

// Jolt's global state. Only the job system is used so far; the factory and
// type registration are what shape creation (convex hulls) will need.
struct JoltRuntime
{
    JoltRuntime()
    {
        JPH::RegisterDefaultAllocator();
        JPH::Factory::sInstance = new JPH::Factory();
        JPH::RegisterTypes();
    }
    ~JoltRuntime()
    {
        JPH::UnregisterTypes();
        delete JPH::Factory::sInstance;
        JPH::Factory::sInstance = nullptr;
    }
};

int Run(const Options &opt)
{
    float radius = opt.radius;
    int   raster = opt.raster;
    const voromesh::OptimizerParams &params = opt.params;
    const std::string               &shot   = opt.shot;
    float voxel = kDomain / raster;

    // One pool for everything: topology runs between physics updates, so the
    // workers physics uses are idle then. Workers + the calling thread = threads.
    JoltRuntime jolt;
    int         threads = opt.threads > 0 ? opt.threads : int(std::max(1u, std::thread::hardware_concurrency()));
    std::unique_ptr<JPH::JobSystemThreadPool> pool;
    std::unique_ptr<voromesh::JoltExecutor>   exec;
    if (threads > 1) {
        pool = std::make_unique<JPH::JobSystemThreadPool>(JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, threads - 1);
        exec = std::make_unique<voromesh::JoltExecutor>(*pool);
    }
    bool parallel = exec != nullptr;

    Editor editor(radius, voxel, params);
    editor.SetExecutor(exec.get());
    unsigned seed     = 1;
    auto     r0       = Clock::now();
    float    globalMs = editor.Reset(seed);
    std::printf("Threads: %s\n",
                parallel ? (std::to_string(threads) + " (Jolt JobSystemThreadPool)").c_str() : "1 (serial)");
    std::printf("Canvas %d x %d (voxel %.3f), Poisson r = %.1f -> %d x %d generator slots\n", raster, raster, voxel,
                radius, editor.Sampler().GridW(), editor.Sampler().GridH());
    std::printf("Reset + global optimisation (%d it): %.3f ms optimise, %.3f ms wall\n", params.iterations,
                globalMs, Ms(r0, Clock::now()));
    PrintMesh("heart", editor);

    // ---- Window ------------------------------------------------------------
    glfwSetErrorCallback([](int code, const char *msg) { std::fprintf(stderr, "GLFW error %d: %s\n", code, msg); });
    if (!glfwInit())
        throw std::runtime_error("glfwInit failed");
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_STENCIL_BITS, 8); // interior fill
    GLFWwindow *window = glfwCreateWindow(900, 900, "VoroMesh 2D", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("glfwCreateWindow failed");
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    Input input;
    glfwSetWindowUserPointer(window, &input);
    glfwSetKeyCallback(window, &KeyCallback);
    glfwSetScrollCallback(window, &ScrollCallback);

    bool               showFill = true, showEdges = true, showGens = true, showSurface = true, showComps = true;
    float              brush = 2.0f * radius;
    DrawBuffers        draw;
    std::vector<Flash> flashes;
    EditStats          stats;
    EditTiming         last;
    draw.Build(editor.Mesh(), editor.Tracker(), editor.Canvas());

    // Runs one stamp, records stats, and refreshes the draw buffers.
    auto stamp = [&](Vec2 c, float r, bool add) {
        Region region;
        if (!editor.Stamp(c, r, add, last, region))
            return false;
        stats.edits.push_back(last);
        float cs = editor.Sampler().CellSize();
        flashes.push_back({region.cx0 * cs, region.cy0 * cs, region.cx1 * cs, region.cy1 * cs, 0.0f});
        draw.Build(editor.Mesh(), editor.Tracker(), editor.Canvas());
        return true;
    };

    auto render = [&](int fbW, int fbH, float dt, const Vec2 *cursor, bool add) {
        glViewport(0, 0, fbW, fbH);
        glClearColor(0.07f, 0.07f, 0.09f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        int side = std::min(fbW, fbH);
        glViewport((fbW - side) / 2, (fbH - side) / 2, side, side);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.0, kDomain, kDomain, 0.0, -1.0, 1.0); // y down
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnableClientState(GL_VERTEX_ARRAY);

        // Colour of a component's parts: its own, or the single-object palette.
        auto tint = [&](const DrawBuffers::Batch &b, float scale, float alpha, std::array<float, 3> plain) {
            if (showComps)
                glColor4f(scale * b.rgb[0], scale * b.rgb[1], scale * b.rgb[2], alpha);
            else
                glColor4f(plain[0], plain[1], plain[2], alpha);
        };

        if (showFill) {
            glEnable(GL_STENCIL_TEST);
            const float domain[] = {0, 0, kDomain, 0, kDomain, kDomain, 0, kDomain};
            for (const DrawBuffers::Batch &b : draw.batches) {
                glClear(GL_STENCIL_BUFFER_BIT);
                glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
                glStencilFunc(GL_ALWAYS, 0, 1);
                glStencilOp(GL_KEEP, GL_KEEP, GL_INVERT);
                DrawArray(b.interiorFan, GL_TRIANGLES);
                glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
                glStencilFunc(GL_EQUAL, 1, 1);
                glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
                tint(b, 0.32f, 1.0f, {0.30f, 0.10f, 0.15f});
                glVertexPointer(2, GL_FLOAT, 0, domain);
                glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
            }
            glDisable(GL_STENCIL_TEST);

            for (const DrawBuffers::Batch &b : draw.batches) {
                tint(b, 0.62f, 1.0f, {0.62f, 0.18f, 0.27f});
                DrawArray(b.shellTris, GL_TRIANGLES);
            }
        }

        // Recently re-optimised regions fade out.
        for (Flash &f : flashes) {
            f.age += dt;
            float a = 0.25f * (1.0f - f.age / kFlashSeconds);
            glColor4f(0.2f, 0.75f, 1.0f, std::max(a, 0.0f));
            const float rect[] = {f.x0, f.y0, f.x1, f.y0, f.x1, f.y1, f.x0, f.y1};
            glVertexPointer(2, GL_FLOAT, 0, rect);
            glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
        }
        flashes.erase(std::remove_if(flashes.begin(), flashes.end(),
                                     [](const Flash &f) { return f.age >= kFlashSeconds; }),
                      flashes.end());

        if (showEdges) {
            glLineWidth(1.0f);
            glColor4f(1.0f, 1.0f, 1.0f, 0.13f);
            DrawArray(draw.cellLines, GL_LINES);
        }
        float pxPerUnit = side / kDomain;
        glEnable(GL_POINT_SMOOTH);
        if (showSurface) {
            glPointSize(std::clamp(1.5f * pxPerUnit, 1.5f, 3.0f));
            for (const DrawBuffers::Batch &b : draw.batches) {
                tint(b, 1.0f, 0.9f, {0.35f, 0.85f, 1.0f});
                DrawArray(b.surface, GL_POINTS);
            }
            if (showComps)
                glColor4f(0.55f, 0.55f, 0.55f, 0.9f); // unlabelled
            else
                glColor4f(0.35f, 0.85f, 1.0f, 0.9f);
            DrawArray(draw.unlabelled, GL_POINTS);
        }
        glLineWidth(2.5f);
        glColor4f(1.0f, 0.85f, 0.4f, 1.0f);
        DrawArray(draw.boundaryLines, GL_LINES);
        if (showGens) {
            glPointSize(std::clamp(radius * pxPerUnit * 0.18f, 2.0f, 6.0f));
            for (const DrawBuffers::Batch &b : draw.batches) {
                if (showComps)
                    glColor4f(0.5f + 0.5f * b.rgb[0], 0.5f + 0.5f * b.rgb[1], 0.5f + 0.5f * b.rgb[2], 0.9f);
                else
                    glColor4f(1.0f, 1.0f, 1.0f, 0.85f);
                DrawArray(b.generators, GL_POINTS);
            }
        }
        glDisable(GL_POINT_SMOOTH);
        if (showComps) { // centre-of-mass markers
            glLineWidth(2.0f);
            for (const DrawBuffers::Batch &b : draw.batches) {
                const float s = 10.0f, x[] = {b.com.x - s, b.com.y, b.com.x + s, b.com.y,
                                              b.com.x, b.com.y - s, b.com.x, b.com.y + s};
                glColor4f(1.0f, 1.0f, 1.0f, 0.95f);
                glVertexPointer(2, GL_FLOAT, 0, x);
                glDrawArrays(GL_LINES, 0, 4);
            }
        }

        if (cursor) {
            glLineWidth(1.5f);
            if (add)
                glColor4f(0.5f, 1.0f, 0.5f, 0.9f);
            else
                glColor4f(1.0f, 1.0f, 1.0f, 0.9f);
            DrawCircle(*cursor, brush);
        }
        glDisableClientState(GL_VERTEX_ARRAY);
    };

    // ---- Scripted edit sequence (--shot) -----------------------------------
    if (!shot.empty()) {
        int fbW, fbH;
        glfwGetFramebufferSize(window, &fbW, &fbH);
        render(fbW, fbH, 0.0f, nullptr, false);
        std::string before = shot + "_before.png";
        std::printf("%s %s\n", SavePng(before, fbW, fbH) ? "Saved" : "FAILED to save", before.c_str());
        glfwSwapBuffers(window);

        // Each drag exercises part of the pipeline; the component events are
        // printed under the stamp that caused them.
        struct Drag
        {
            const char *what;
            Vec2        from, to;
            float       r;
            bool        add;
        };
        const Drag drags[] = {
            {"bite the right lobe", {640, 120}, {830, 230}, 48, false},
            {"punch a hole (one component, two loops)", {512, 560}, {512, 560}, 70, false},
            {"add a bump on the left", {120, 300}, {80, 440}, 40, true},
            {"cut off the tip (chip: newly exposed cells)", {300, 740}, {724, 740}, 20, false},
            {"cut top to bottom (split)", {512, 60}, {512, 730}, 24, false},
            {"bridge the halves (merge)", {380, 420}, {644, 420}, 36, true},
            {"remove the chip (destroy)", {512, 830}, {512, 830}, 80, false},
        };
        std::printf("Scripted edits (%d local iterations each):\n", params.localIterations);
        bool  consistent = true;
        float kinError   = 0.0f;
        int   step       = 0;
        for (const Drag &d : drags) {
            std::printf(" %d. %s\n", ++step, d.what);
            float len = std::hypot(d.to.x - d.from.x, d.to.y - d.from.y);
            int   n   = std::max(1, int(std::ceil(len / (0.25f * d.r))));
            for (int k = 0; k <= n; ++k) {
                float t = float(k) / n;
                Vec2  c{d.from.x + t * (d.to.x - d.from.x), d.from.y + t * (d.to.y - d.from.y)};
                if (stamp(c, d.r, d.add)) {
                    PrintEdit(d.add ? "add" : "carve", c, last, params.localIterations);
                    PrintEvents(editor.Tracker());
                    kinError = std::max(kinError, CheckSplitKinematics(editor.Tracker()));
                }
                if (len == 0.0f)
                    break;
            }
            const voromesh::ComponentTracker &tr = editor.Tracker();
            std::printf("    -> %zu component(s), %d loop(s), holes attached %d, cells labelled by region %d, "
                        "unlabelled points %d, checksum %08x\n",
                        tr.Components().size(), editor.Mesh().loops, tr.AttachedHoles(), tr.InheritedCells(),
                        tr.OrphanTargets(), LabelChecksum(editor));
            consistent &= VerifyComponents(editor);
            render(fbW, fbH, 1.0f, nullptr, false);
            std::string png = shot + "_step" + std::to_string(step) + ".png";
            if (!SavePng(png, fbW, fbH))
                std::printf("FAILED to save %s\n", png.c_str());
            glfwSwapBuffers(window);
        }
        stats.Print();
        PrintMesh("after edits", editor);
        std::printf("Components consistent with the canvas after every drag: %s\n", consistent ? "yes" : "NO");
        std::printf("Split kinematics round trip (pieces merged back vs parent), max error: %.3g\n", kinError);

        render(fbW, fbH, 1.0f, nullptr, false); // dt > flash time: no highlight
        std::string after = shot + "_after.png";
        std::printf("%s %s\n", SavePng(after, fbW, fbH) ? "Saved" : "FAILED to save", after.c_str());
        glfwSwapBuffers(window);
        glfwDestroyWindow(window);
        glfwTerminate();
        return editor.Mesh().watertight && consistent ? 0 : 1;
    }

    // ---- Interactive loop --------------------------------------------------
    auto  prevTime  = Clock::now();
    float fpsAccum  = 0.0f, fps = 0.0f, titleTimer = 0.0f;
    int   fpsFrames = 0;
    bool  stroking  = false;
    Vec2  lastStamp{0, 0};

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        auto  now = Clock::now();
        float dt  = std::chrono::duration<float>(now - prevTime).count();
        prevTime  = now;

        if (input.reset) {
            editor.Reset(++seed);
            draw.Build(editor.Mesh(), editor.Tracker(), editor.Canvas());
            input.reset = false;
        }
        if (input.brushStep != 0 || input.scroll != 0.0) {
            float steps = float(input.brushStep) + float(input.scroll);
            brush       = std::clamp(brush * std::pow(1.15f, steps), 8.0f, 250.0f);
            input.brushStep = 0;
            input.scroll    = 0.0;
        }
        if (input.iterStep != 0) {
            // Step along a fixed ladder, starting from the nearest rung.
            static constexpr int kLadder[] = {0, 10, 25, 50, 100, 200, 400};
            constexpr int        kRungs    = int(std::size(kLadder));
            int rung = 0;
            while (rung + 1 < kRungs && kLadder[rung + 1] <= editor.LocalIterations())
                ++rung;
            rung = std::clamp(rung + input.iterStep, 0, kRungs - 1);
            editor.SetLocalIterations(kLadder[rung]);
            std::printf("Local optimisation: %d iterations per stamp\n", kLadder[rung]);
            input.iterStep = 0;
        }
        if (input.toggleFill)  { showFill = !showFill;       input.toggleFill = false; }
        if (input.toggleEdges) { showEdges = !showEdges;     input.toggleEdges = false; }
        if (input.toggleGens)  { showGens = !showGens;       input.toggleGens = false; }
        if (input.toggleTgts)  { showSurface = !showSurface; input.toggleTgts = false; }
        if (input.toggleComps) { showComps = !showComps;     input.toggleComps = false; }
        if (input.toggleParallel) {
            parallel = exec && !parallel;
            editor.SetExecutor(parallel ? exec.get() : nullptr);
            std::printf("Parallel loops: %s\n", parallel ? "on" : "off");
            input.toggleParallel = false;
        }

        int winW, winH, fbW, fbH;
        glfwGetWindowSize(window, &winW, &winH);
        glfwGetFramebufferSize(window, &fbW, &fbH);
        if (winW <= 0 || winH <= 0 || fbW <= 0 || fbH <= 0) { // minimized
            glfwWaitEvents();
            continue;
        }

        // Cursor -> world, through the centred square viewport.
        double mx, my;
        glfwGetCursorPos(window, &mx, &my);
        float fbScale = float(fbW) / float(winW);
        int   side    = std::min(fbW, fbH);
        Vec2  cursor{float((mx * fbScale - (fbW - side) / 2) / side * kDomain),
                     float((my * fbScale - (fbH - side) / 2) / side * kDomain)};

        bool carve = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        bool add   = !carve && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
        if (carve || add) {
            // Stamp on press, then every quarter brush radius of travel.
            if (!stroking || std::hypot(cursor.x - lastStamp.x, cursor.y - lastStamp.y) >= 0.25f * brush) {
                if (stamp(cursor, brush, add))
                    PrintEvents(editor.Tracker());
                lastStamp = cursor;
            }
            stroking = true;
        } else {
            stroking = false;
        }

        bool addHint = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
        render(fbW, fbH, dt, &cursor, addHint);
        glfwSwapBuffers(window);

        fpsAccum += dt;
        ++fpsFrames;
        titleTimer += dt;
        if (fpsAccum >= 0.5f) {
            fps       = fpsFrames / fpsAccum;
            fpsAccum  = 0.0f;
            fpsFrames = 0;
        }
        if (titleTimer >= 0.1f) {
            titleTimer = 0.0f;
            const voromesh::VoroMesh2D &mesh = editor.Mesh();
            char title[448];
            std::snprintf(title, sizeof(title),
                          "VoroMesh 2D [%s] | brush %.0f | opt %d it | last edit %.2f ms (canvas %.2f, in %.2f, "
                          "poisson %.2f, opt %.2f, extract %.2f, comps %.2f) | %zu edits | %d shell / %d band cells | "
                          "%s, %d loop(s), %zu component(s) | %.0f fps",
                          parallel ? (std::to_string(threads) + " threads").c_str() : "serial", brush,
                          editor.LocalIterations(), last.totalMs, last.canvasMs, last.inputsMs,
                          last.poissonMs, last.optimizeMs, last.extractMs, last.componentsMs, stats.edits.size(),
                          mesh.shellCells, mesh.bandCells, mesh.watertight ? "watertight" : "NOT WATERTIGHT",
                          mesh.loops, editor.Tracker().Components().size(), fps);
            glfwSetWindowTitle(window, title);
        }
    }

    stats.Print();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

constexpr const char *kUsage =
    "usage: voromesh_demo [radius=24] [raster=1024] [localIterations=100] [--threads N] [--shot prefix]";

Options ParseOptions(int argc, char **argv)
{
    Options                   opt;
    std::vector<const char *> pos;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--shot") && i + 1 < argc)
            opt.shot = argv[++i];
        else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc)
            opt.threads = std::atoi(argv[++i]);
        else if (argv[i][0] == '-' && argv[i][1] == '-')
            throw std::runtime_error(std::string("unknown option ") + argv[i] + "\n" + kUsage);
        else
            pos.push_back(argv[i]);
    }
    if (pos.size() > 3)
        throw std::runtime_error(std::string("too many arguments\n") + kUsage);
    if (pos.size() > 0) opt.radius = (float)std::atof(pos[0]);
    if (pos.size() > 1) opt.raster = std::atoi(pos[1]);
    if (pos.size() > 2) opt.params.localIterations = std::atoi(pos[2]);
    if (!(opt.radius > 0.0f) || opt.raster <= 0 || opt.params.localIterations < 0 || opt.threads < 0)
        throw std::runtime_error(std::string("invalid argument\n") + kUsage);
    return opt;
}

} // namespace

int main(int argc, char **argv)
{
    try {
        return Run(ParseOptions(argc, argv));
    } catch (const std::exception &e) {
        std::fprintf(stderr, "Fatal: %s\n", e.what());
        return 1;
    }
}
