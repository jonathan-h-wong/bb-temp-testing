// ---------------------------------------------------------------------------
// voromesh_demo – interactive local editing of a 2D VoroMesh.
//
//   * Canvas (CPU): 1024^2 voxels holding the surface samples (sparse 8x8
//     tiles) and the inside/outside occupancy (dense bitmask). Starts as a heart.
//   * Targets: the canvas surface samples, mirrored in GPU slots.
//   * Generators: GPU Poisson disk samples, one slot per Poisson grid cell.
//     The Poisson radius (24 voxels) is deliberately much coarser than the
//     canvas: the Voronoi mesh is a lightweight physics proxy.
//   * VoroLoss + Adam on the GPU, then watertight boundary extraction.
//
// Each brush stamp runs the whole local pipeline:
//   1. Canvas: the disc interior becomes outside (carve) or inside (add),
//      surface samples inside the disc are deleted, and the disc rim becomes
//      surface where it lay on the other side (inside the object when carving).
//   2. Stream the changed target slots and occupancy tiles to the GPU.
//   3. Poisson-resample the generator slots around the disc (the slots
//      outside keep their optimised positions).
//   4. Re-optimise only those generators.
//   5. Extract and re-render.
//
// Controls
//   Left drag     carve (disc -> outside)
//   Right drag    add (disc -> inside)
//   Wheel / [ ]   brush radius
//   - / =         fewer / more optimisation steps per stamp (0 shows the raw reseed)
//   R             reset to the heart (global optimisation)
//   F / V         toggle cell fill / Voronoi edges
//   G / T         toggle generators / canvas surface points
//   Esc           quit (prints edit timing statistics)
//
// Usage: voromesh_demo [radius=24] [raster=1024] [localIterations=100] [--cpu] [--shot prefix]
//   radius            Poisson radius of the generators (domain is 1024 x 1024)
//   raster            canvas resolution across the domain (voxel = 1024 / raster)
//   localIterations   optimisation steps after each stamp
//   --cpu             run Poisson sampling, optimisation and extraction on the
//                     CPU (CpuSampler + CpuVoroOptimizer) instead of CUDA
//   --shot            run a scripted edit sequence, print timings, save
//                     <prefix>_before.png and <prefix>_after.png, then exit
// ---------------------------------------------------------------------------
#include "poisson/poisson_disk.h"
#include "poisson/poisson_disk_cpu.h"
#include "voromesh/canvas.h"
#include "voromesh/cpu_voro_optimizer.h"
#include "voromesh/shape2d.h"
#include "voromesh/voro_optimizer.h"

#include <GLFW/glfw3.h> // also pulls in the system GL 1.1 header

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
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
#include <vector>

#ifndef GL_MULTISAMPLE
#define GL_MULTISAMPLE 0x809D // GL 1.3; Windows' gl.h only goes up to 1.1
#endif

static_assert(sizeof(poisson::Sample) == sizeof(voromesh::Vec2), "Poisson cells are uploaded as generator slots");

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
// Backends: where Poisson sampling, optimisation and extraction run
// ---------------------------------------------------------------------------
struct GpuBackend
{
    using Sampler   = poisson::Sampler;
    using Optimizer = voromesh::VoroOptimizer;
    static constexpr const char *kName = "GPU";
};

struct CpuBackend
{
    using Sampler   = poisson::CpuSampler;
    using Optimizer = voromesh::CpuVoroOptimizer;
    static constexpr const char *kName = "CPU";
};

// ---------------------------------------------------------------------------
// Editor: canvas + Poisson sampler + optimiser, and the local edit pipeline
// ---------------------------------------------------------------------------
struct EditTiming
{
    double canvasMs = 0, uploadMs = 0, poissonMs = 0, optimizeMs = 0, extractMs = 0, totalMs = 0;
    float  poissonInnerMs = 0, optimizeInnerMs = 0; // as reported by the sampler / optimiser
    float  cellsMs = 0, assemblyMs = 0;                // extraction split, as reported by the optimiser
    int    removed = 0, added = 0, slots = 0, clipped = 0;
};

struct Region
{
    int cx0, cy0, cx1, cy1; // Poisson cells / generator slots
};

template <class Backend>
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
        mSampler = std::make_unique<typename Backend::Sampler>(kDomain, kDomain, mRadius, 30, seed);
        mOpt.SetGenerators(Slots(), mSampler->GridW(), mSampler->GridH(), mSampler->CellSize());
        float ms = mOpt.Optimize();
        mMesh    = mOpt.Extract();
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

        t.canvasMs   = Ms(t0, t1);
        t.uploadMs   = Ms(t1, t2);
        t.poissonMs  = Ms(t2, t3);
        t.optimizeMs = Ms(t3, t4);
        t.extractMs  = Ms(t4, t5);
        t.totalMs    = Ms(t0, t5);
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

    const voromesh::Canvas          &Canvas() const { return mCanvas; }
    const voromesh::VoroMesh2D      &Mesh() const { return mMesh; }
    typename Backend::Optimizer     &Optimizer() { return mOpt; }
    const typename Backend::Sampler &Sampler() const { return *mSampler; }
    float                            Radius() const { return mRadius; }
    int                              LocalIterations() const { return mLocalIterations; }
    void                             SetLocalIterations(int n) { mLocalIterations = n; }

private:
    const Vec2 *Slots() const { return reinterpret_cast<const Vec2 *>(mSampler->Cells().data()); }

    float                                      mRadius;
    int                                        mLocalIterations; // optimisation steps per stamp
    voromesh::Canvas                           mCanvas;
    typename Backend::Optimizer                mOpt;
    std::unique_ptr<typename Backend::Sampler> mSampler;
    voromesh::VoroMesh2D                       mMesh;
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

template <class EditorT>
void PrintMesh(const char *label, EditorT &editor)
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
    std::printf("  boundary                 %zu vertices, %zu edges, %d loop(s), %s\n", mesh.vertices.size(),
                mesh.edges.size(), mesh.loops, mesh.watertight ? "WATERTIGHT" : "NOT WATERTIGHT");
    std::printf("  surface -> mesh          mean %.3f  max %.3f voxels\n", acc.meanToMesh / voxel,
                acc.maxToMesh / voxel);
    std::printf("  mesh vertex -> surface   mean %.3f  max %.3f voxels\n", acc.meanToTargets / voxel,
                acc.maxToTargets / voxel);
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
        stat("targets+occupancy in", &EditTiming::uploadMs);
        stat("poisson resample", &EditTiming::poissonMs);
        stat("local optimisation", &EditTiming::optimizeMs);
        stat("extract", &EditTiming::extractMs);
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

// Triangles/lines for the legacy-GL renderer, rebuilt whenever the mesh changes.
struct DrawBuffers
{
    std::vector<float> insideTris, outsideTris, cellLines, boundaryLines, generators, surface;

    void Build(const voromesh::VoroMesh2D &mesh, const std::vector<Vec2> &surfacePoints)
    {
        insideTris.clear();
        outsideTris.clear();
        cellLines.clear();
        boundaryLines.clear();
        generators.clear();
        surface.clear();
        for (size_t i = 0; i + 1 < mesh.cellStart.size(); ++i) {
            int                 s = mesh.cellStart[i], n = mesh.cellStart[i + 1] - s;
            const Vec2         *v = mesh.cellVerts.data() + s;
            std::vector<float> &tris = mesh.inside[i] ? insideTris : outsideTris;
            for (int k = 1; k + 1 < n; ++k)
                tris.insert(tris.end(), {v[0].x, v[0].y, v[k].x, v[k].y, v[k + 1].x, v[k + 1].y});
            for (int k = 0; k < n; ++k)
                cellLines.insert(cellLines.end(), {v[k].x, v[k].y, v[(k + 1) % n].x, v[(k + 1) % n].y});
        }
        for (auto [a, b] : mesh.edges) {
            Vec2 p = mesh.vertices[a], q = mesh.vertices[b];
            boundaryLines.insert(boundaryLines.end(), {p.x, p.y, q.x, q.y});
        }
        for (Vec2 g : mesh.generators)
            if (g.x >= 0.0f)
                generators.insert(generators.end(), {g.x, g.y});
        for (Vec2 p : surfacePoints)
            surface.insert(surface.end(), {p.x, p.y});
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
                "opt %.3f (%d it)  extract %.3f (%4d cells %.3f + assembly %.3f) | total %.3f ms\n",
                what, c.x, c.y, t.removed, t.added, t.slots, t.canvasMs, t.uploadMs, t.poissonMs, t.optimizeMs,
                iterations, t.extractMs, t.clipped, t.cellsMs, t.assemblyMs, t.totalMs);
}

struct Options
{
    float                     radius = 24.0f;
    int                       raster = 1024;
    voromesh::OptimizerParams params;
    std::string               shot;
    bool                      cpu = false;
};

template <class Backend>
int Run(const Options &opt)
{
    float radius = opt.radius;
    int   raster = opt.raster;
    const voromesh::OptimizerParams &params = opt.params;
    const std::string               &shot   = opt.shot;
    float voxel = kDomain / raster;

    Editor<Backend> editor(radius, voxel, params);
    unsigned        seed     = 1;
    auto            r0       = Clock::now();
    float           globalMs = editor.Reset(seed);
    std::printf("Backend: %s\n", Backend::kName);
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

    bool               showFill = true, showEdges = true, showGens = true, showSurface = true;
    float              brush = 2.0f * radius;
    DrawBuffers        draw;
    std::vector<Flash> flashes;
    EditStats          stats;
    EditTiming         last;
    draw.Build(editor.Mesh(), editor.SurfacePoints());

    // Runs one stamp, records stats, and refreshes the draw buffers.
    auto stamp = [&](Vec2 c, float r, bool add) {
        Region region;
        if (!editor.Stamp(c, r, add, last, region))
            return false;
        stats.edits.push_back(last);
        float cs = editor.Sampler().CellSize();
        flashes.push_back({region.cx0 * cs, region.cy0 * cs, region.cx1 * cs, region.cy1 * cs, 0.0f});
        draw.Build(editor.Mesh(), editor.SurfacePoints());
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

        if (showFill) {
            glColor4f(0.12f, 0.13f, 0.16f, 1.0f);
            DrawArray(draw.outsideTris, GL_TRIANGLES);
            glColor4f(0.55f, 0.16f, 0.24f, 1.0f);
            DrawArray(draw.insideTris, GL_TRIANGLES);
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
            glColor4f(0.35f, 0.85f, 1.0f, 0.9f);
            DrawArray(draw.surface, GL_POINTS);
        }
        glLineWidth(2.5f);
        glColor4f(1.0f, 0.85f, 0.4f, 1.0f);
        DrawArray(draw.boundaryLines, GL_LINES);
        if (showGens) {
            glPointSize(std::clamp(radius * pxPerUnit * 0.18f, 2.0f, 6.0f));
            glColor4f(1.0f, 1.0f, 1.0f, 0.85f);
            DrawArray(draw.generators, GL_POINTS);
        }
        glDisable(GL_POINT_SMOOTH);

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

        // A drag that bites into the right lobe, a hole punched in the middle,
        // and a drag that adds a bump on the left.
        struct Drag
        {
            Vec2  from, to;
            float r;
            bool  add;
        };
        const Drag drags[] = {
            {{640, 120}, {830, 230}, 48, false},
            {{512, 560}, {512, 560}, 70, false},
            {{120, 300}, {80, 440}, 40, true},
        };
        std::printf("Scripted edits (%d local iterations each):\n", params.localIterations);
        for (const Drag &d : drags) {
            float len = std::hypot(d.to.x - d.from.x, d.to.y - d.from.y);
            int   n   = std::max(1, int(std::ceil(len / (0.25f * d.r))));
            for (int k = 0; k <= n; ++k) {
                float t = float(k) / n;
                Vec2  c{d.from.x + t * (d.to.x - d.from.x), d.from.y + t * (d.to.y - d.from.y)};
                if (stamp(c, d.r, d.add))
                    PrintEdit(d.add ? "add" : "carve", c, last, params.localIterations);
                if (len == 0.0f)
                    break;
            }
        }
        stats.Print();
        PrintMesh("after edits", editor);

        render(fbW, fbH, 1.0f, nullptr, false); // dt > flash time: no highlight
        std::string after = shot + "_after.png";
        std::printf("%s %s\n", SavePng(after, fbW, fbH) ? "Saved" : "FAILED to save", after.c_str());
        glfwSwapBuffers(window);
        glfwDestroyWindow(window);
        glfwTerminate();
        return editor.Mesh().watertight ? 0 : 1;
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
            draw.Build(editor.Mesh(), editor.SurfacePoints());
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
                stamp(cursor, brush, add);
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
            char title[384];
            std::snprintf(title, sizeof(title),
                          "VoroMesh 2D [%s] | brush %.0f | opt %d it | last edit %.2f ms (canvas %.2f, in %.2f, "
                          "poisson %.2f, opt %.2f, extract %.2f) | %zu edits | %s, %d loop(s) | %.0f fps",
                          Backend::kName, brush, editor.LocalIterations(), last.totalMs, last.canvasMs, last.uploadMs, last.poissonMs, last.optimizeMs,
                          last.extractMs, stats.edits.size(), mesh.watertight ? "watertight" : "NOT WATERTIGHT",
                          mesh.loops, fps);
            glfwSetWindowTitle(window, title);
        }
    }

    stats.Print();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

Options ParseOptions(int argc, char **argv)
{
    Options                   opt;
    std::vector<const char *> pos;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--shot") && i + 1 < argc)
            opt.shot = argv[++i];
        else if (!std::strcmp(argv[i], "--cpu"))
            opt.cpu = true;
        else
            pos.push_back(argv[i]);
    }
    if (pos.size() > 0) opt.radius = (float)std::atof(pos[0]);
    if (pos.size() > 1) opt.raster = std::atoi(pos[1]);
    if (pos.size() > 2) opt.params.localIterations = std::atoi(pos[2]);
    return opt;
}

} // namespace

int main(int argc, char **argv)
{
    try {
        Options opt = ParseOptions(argc, argv);
        return opt.cpu ? Run<CpuBackend>(opt) : Run<GpuBackend>(opt);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "Fatal: %s\n", e.what());
        return 1;
    }
}
