// ---------------------------------------------------------------------------
// sphere_complex_demo – tests three Jolt features in realtime:
//   1. Composite "sphere complexes": rigid bodies whose shape is a
//      StaticCompoundShape made of several overlapping SphereShapes.
//   2. Dynamically adding / removing bodies while the simulation runs.
//   3. Per-body initial properties: every spawn gets its own mass, position,
//      linear velocity and angular velocity. Right after creation the values
//      are read back from Jolt and compared to what was requested; the
//      pass/fail tally is shown in the title bar, failures go to stderr.
//
// Controls
//   Space  spawn 1 complex (params logged)  B  spawn burst of 20
//   L      launch a heavy, fast, spinning complex into the pile (logged)
//   D      delete 1 random complex          X  delete 20 random complexes
//   C      clear all complexes              T  toggle auto-churn (continuous add+delete)
//   S      cycle color: mass (blue = light, red = heavy) / per-body / sleep state
//   V      toggle velocity arrows           W  toggle wireframe
//   Esc    quit
//   Left-drag orbits the camera, scroll zooms.
// ---------------------------------------------------------------------------
#include "viz/viewer.h"

#include <Jolt/Jolt.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayerInterfaceTable.h>
#include <Jolt/Physics/Collision/BroadPhase/ObjectVsBroadPhaseLayerFilterTable.h>
#include <Jolt/Physics/Collision/ObjectLayerPairFilterTable.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/RegisterTypes.h>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <exception>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace JPH;

namespace {

// ---------------------------------------------------------------------------
// Scene parameters
// ---------------------------------------------------------------------------
constexpr float kBoxHalfWidth  = 6.0f;  // inner half-extent of the container in X/Z
constexpr float kBoxHeight     = 8.0f;
constexpr float kWallThickness = 0.5f;
constexpr float kSpawnHeight   = 14.0f;
constexpr float kPhysicsDt     = 1.0f / 60.0f;
constexpr int   kBurstCount    = 20;
constexpr int   kChurnTarget   = 150;   // auto-churn keeps roughly this many complexes alive
constexpr float kChurnInterval = 0.05f; // seconds between auto-churn add/delete events

// Randomized initial properties per spawn. Mass is log-uniform so light and
// heavy bodies are equally common; the range is kept moderate (40:1) because
// very large mass ratios in a resting stack make any solver jitter.
constexpr float kMinMass           = 0.5f;  // kg
constexpr float kMaxMass           = 20.0f; // kg
constexpr float kMaxHorizontalVel  = 4.0f;  // m/s
constexpr float kMaxAngularSpeed   = 10.0f; // rad/s (Jolt's default cap is ~47)
constexpr float kLaunchMass        = 100.0f;
constexpr float kLaunchSpeed       = 14.0f;
constexpr float kLaunchAngularSpeed = 20.0f;

enum class ColorMode { Mass, Instance, Sleep };

/// Blue (light) -> red (heavy) on a log scale over [kMinMass, kMaxMass].
Color MassColor(float mass)
{
    const float t = std::clamp(std::log(mass / kMinMass) / std::log(kMaxMass / kMinMass), 0.0f, 1.0f);
    return Color(uint8(40 + 215 * t), uint8(110 - 60 * t), uint8(255 - 215 * t));
}

/// Everything about a body's initial state that this demo varies per spawn.
struct SpawnParams
{
    RVec3 position;
    Quat  rotation;
    float mass;            // kg; inertia is computed from the shape, scaled to this mass
    Vec3  linearVelocity;  // m/s, world space
    Vec3  angularVelocity; // rad/s, world space
};

namespace Layers {
constexpr ObjectLayer kNonMoving = 0;
constexpr ObjectLayer kMoving    = 1;
constexpr uint        kCount     = 2;
} // namespace Layers

void TraceImpl(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    std::vprintf(fmt, args);
    va_end(args);
    std::printf("\n");
}

#ifdef JPH_ENABLE_ASSERTS
bool AssertFailedImpl(const char *expr, const char *msg, const char *file, uint line)
{
    std::fprintf(stderr, "%s:%u: (%s) %s\n", file, line, expr, msg ? msg : "");
    return true; // break into debugger
}
#endif

/// Builds a random connected cluster of overlapping spheres as one compound shape.
RefConst<Shape> MakeSphereComplex(std::mt19937 &rng)
{
    std::uniform_int_distribution<int>    countDist(3, 7);
    std::uniform_real_distribution<float> radiusDist(0.25f, 0.5f);
    std::uniform_real_distribution<float> unitDist(-1.0f, 1.0f);

    struct Sphere { Vec3 center; float radius; };
    std::vector<Sphere> spheres{{Vec3::sZero(), radiusDist(rng)}};

    const int count = countDist(rng);
    while (int(spheres.size()) < count) {
        // Attach each new sphere to a random existing one so the cluster stays connected.
        const Sphere parent = spheres[std::uniform_int_distribution<size_t>(0, spheres.size() - 1)(rng)];
        Vec3 dir;
        do { dir = Vec3(unitDist(rng), unitDist(rng), unitDist(rng)); } while (dir.LengthSq() < 1e-4f || dir.LengthSq() > 1.0f);
        const float r = radiusDist(rng);
        spheres.push_back({parent.center + dir.Normalized() * (parent.radius + r) * 0.75f, r});
    }

    StaticCompoundShapeSettings settings;
    for (const Sphere &s : spheres)
        settings.AddShape(s.center, Quat::sIdentity(), new SphereShape(s.radius));

    ShapeSettings::ShapeResult result = settings.Create();
    if (result.HasError())
        throw std::runtime_error(result.GetError().c_str());
    return result.Get();
}

/// Hides the static container; Scene::Draw renders it itself (grey floor + wire
/// outline) so the walls don't block the view of the complexes inside.
class HideStatic final : public BodyDrawFilter
{
public:
    bool ShouldDraw(const Body &body) const override { return !body.IsStatic(); }
};

/// Rising-edge key detection on top of GLFW polling.
class KeyEdges
{
public:
    explicit KeyEdges(GLFWwindow *window) : mWindow(window) {}
    bool Pressed(int key)
    {
        const bool down = glfwGetKey(mWindow, key) == GLFW_PRESS;
        const bool wasDown = mDown[key];
        mDown[key] = down;
        return down && !wasDown;
    }

private:
    GLFWwindow *mWindow;
    bool        mDown[GLFW_KEY_LAST + 1] = {};
};

class Scene
{
public:
    Scene()
        : mTempAllocator(32 * 1024 * 1024),
          mJobSystem(cMaxPhysicsJobs, cMaxPhysicsBarriers, std::max(1, int(std::thread::hardware_concurrency()) - 1)),
          mObjectPairFilter(Layers::kCount),
          mBroadPhaseLayers(Layers::kCount, 2)
    {
        mObjectPairFilter.EnableCollision(Layers::kMoving, Layers::kNonMoving);
        mObjectPairFilter.EnableCollision(Layers::kMoving, Layers::kMoving);
        mBroadPhaseLayers.MapObjectToBroadPhaseLayer(Layers::kNonMoving, BroadPhaseLayer(0));
        mBroadPhaseLayers.MapObjectToBroadPhaseLayer(Layers::kMoving, BroadPhaseLayer(1));

        // Must be built after the two tables above are configured: it snapshots them at construction.
        mObjectVsBroadPhase = std::make_unique<ObjectVsBroadPhaseLayerFilterTable>(mBroadPhaseLayers, 2, mObjectPairFilter,
                                                                                  Layers::kCount);

        mPhysics.Init(/*maxBodies*/ 10240, /*numBodyMutexes*/ 0, /*maxBodyPairs*/ 65536,
                      /*maxContactConstraints*/ 20480, mBroadPhaseLayers, *mObjectVsBroadPhase, mObjectPairFilter);

        BuildContainer();
        mPhysics.OptimizeBroadPhase();
    }

    ~Scene() { Clear(); }

    /// Random mass / position / velocity / spin, dropped from above the box.
    SpawnParams RandomParams()
    {
        std::uniform_real_distribution<float> xz(-kBoxHalfWidth + 1.0f, kBoxHalfWidth - 1.0f);
        std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
        std::uniform_real_distribution<float> logMass(std::log(kMinMass), std::log(kMaxMass));

        SpawnParams p;
        p.position        = RVec3(xz(mRng), kSpawnHeight + 1.5f * (unit(mRng) + 1.0f), xz(mRng));
        p.rotation        = Quat::sRandom(mRng);
        p.mass            = std::exp(logMass(mRng));
        p.linearVelocity  = Vec3(kMaxHorizontalVel * unit(mRng), -2.0f + 2.0f * unit(mRng), kMaxHorizontalVel * unit(mRng));
        p.angularVelocity = Vec3::sRandom(mRng) * (kMaxAngularSpeed * 0.5f * (unit(mRng) + 1.0f));
        return p;
    }

    /// A heavy complex fired horizontally from the -X wall toward the pile, spinning hard.
    SpawnParams LaunchParams()
    {
        std::uniform_real_distribution<float> unit(-1.0f, 1.0f);

        SpawnParams p;
        p.position        = RVec3(-kBoxHalfWidth + 1.5f, 0.6f * kBoxHeight, 2.0f * unit(mRng));
        p.rotation        = Quat::sRandom(mRng);
        p.mass            = kLaunchMass;
        p.linearVelocity  = Vec3(kLaunchSpeed, 2.0f, 1.5f * unit(mRng));
        p.angularVelocity = Vec3::sRandom(mRng) * kLaunchAngularSpeed;
        return p;
    }

    void Spawn(bool log = false) { Spawn(RandomParams(), log); }

    void Spawn(const SpawnParams &p, bool log)
    {
        BodyCreationSettings settings(MakeSphereComplex(mRng), p.position, p.rotation, EMotionType::Dynamic,
                                      Layers::kMoving);
        settings.mFriction = 0.6f;
        settings.mRestitution = 0.2f;
        settings.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia; // use our mass, shape's inertia
        settings.mMassPropertiesOverride.mMass = p.mass;
        settings.mLinearVelocity = p.linearVelocity;
        settings.mAngularVelocity = p.angularVelocity;

        const BodyID id = Bodies().CreateAndAddBody(settings, EActivation::Activate);
        mComplexes.push_back(id);
        VerifyInitialState(id, p, log);
    }

    void DeleteRandom()
    {
        if (mComplexes.empty())
            return;
        const size_t i = std::uniform_int_distribution<size_t>(0, mComplexes.size() - 1)(mRng);
        const BodyID id = mComplexes[i];
        mComplexes[i] = mComplexes.back();
        mComplexes.pop_back();

        BodyInterface &bodies = Bodies();
        AABox neighborhood = bodies.GetTransformedShape(id).GetWorldSpaceBounds();
        neighborhood.ExpandBy(Vec3::sReplicate(0.5f));

        bodies.RemoveBody(id);
        bodies.DestroyBody(id);

        // Jolt does not wake bodies resting on a removed body; do it explicitly
        // so anything that was supported by it falls into the gap.
        bodies.ActivateBodiesInAABox(neighborhood, BroadPhaseLayerFilter{}, ObjectLayerFilter{});
    }

    void Clear()
    {
        if (mComplexes.empty())
            return;
        Bodies().RemoveBodies(mComplexes.data(), int(mComplexes.size()));
        Bodies().DestroyBodies(mComplexes.data(), int(mComplexes.size()));
        mComplexes.clear();
    }

    void Step() { mPhysics.Update(kPhysicsDt, /*collisionSteps*/ 1, &mTempAllocator, &mJobSystem); }

    void Draw(DebugRenderer &renderer, BodyManager::DrawSettings settings, ColorMode colorMode)
    {
        HideStatic filter;
        if (colorMode == ColorMode::Mass) {
            // Jolt has no mass coloring, so draw the shapes ourselves; DrawBodies
            // still handles the rest (e.g. velocity arrows).
            const bool wireframe = settings.mDrawShapeWireframe;
            settings.mDrawShape = false;
            mPhysics.DrawBodies(settings, &renderer, &filter);
            for (const BodyID id : mComplexes) {
                BodyLockRead lock(mPhysics.GetBodyLockInterface(), id);
                if (!lock.Succeeded())
                    continue;
                const Body &body = lock.GetBody();
                const float mass = 1.0f / body.GetMotionProperties()->GetInverseMass();
                body.GetShape()->Draw(&renderer, body.GetCenterOfMassTransform(), Vec3::sOne(), MassColor(mass), false,
                                      wireframe);
            }
        } else {
            settings.mDrawShapeColor = colorMode == ColorMode::Sleep ? BodyManager::EShapeColor::SleepColor
                                                                     : BodyManager::EShapeColor::InstanceColor;
            mPhysics.DrawBodies(settings, &renderer, &filter);
        }
        renderer.DrawBox(AABox(Vec3(-kBoxHalfWidth, -kWallThickness, -kBoxHalfWidth), Vec3(kBoxHalfWidth, 0, kBoxHalfWidth)),
                         Color(70, 72, 80));
        renderer.DrawWireBox(AABox(Vec3(-kBoxHalfWidth, 0, -kBoxHalfWidth), Vec3(kBoxHalfWidth, kBoxHeight, kBoxHalfWidth)),
                             Color::sGrey);
    }

    size_t NumComplexes() const { return mComplexes.size(); }
    uint   NumActive() const { return mPhysics.GetNumActiveBodies(EBodyType::RigidBody); }
    uint   NumInitChecks() const { return mInitChecks; }
    uint   NumInitFailures() const { return mInitFailures; }

private:
    BodyInterface &Bodies() { return mPhysics.GetBodyInterface(); }

    /// Reads the freshly created body's state back from Jolt (before any
    /// simulation step) and checks it matches what was requested.
    void VerifyInitialState(BodyID id, const SpawnParams &p, bool log)
    {
        BodyInterface &bodies = Bodies();
        const RVec3 position = bodies.GetPosition(id);
        const Vec3  linearVelocity = bodies.GetLinearVelocity(id);
        const Vec3  angularVelocity = bodies.GetAngularVelocity(id);
        float mass;
        {
            BodyLockRead lock(mPhysics.GetBodyLockInterface(), id);
            mass = 1.0f / lock.GetBody().GetMotionProperties()->GetInverseMass();
        }

        // Position goes through a center-of-mass round trip, so allow float noise.
        const bool posOk  = Vec3(position - p.position).LengthSq() < 1e-6f;
        const bool massOk = std::abs(mass - p.mass) <= 1e-4f * p.mass;
        const bool velOk  = linearVelocity.IsClose(p.linearVelocity, 1e-8f);
        const bool angOk  = angularVelocity.IsClose(p.angularVelocity, 1e-8f);
        const bool ok = posOk && massOk && velOk && angOk;

        ++mInitChecks;
        if (!ok)
            ++mInitFailures;

        if (log || !ok) {
            auto v3 = [](Vec3 v) {
                char buf[64];
                std::snprintf(buf, sizeof(buf), "(%6.2f %6.2f %6.2f)", v.GetX(), v.GetY(), v.GetZ());
                return std::string(buf);
            };
            std::fprintf(ok ? stdout : stderr,
                         "[%s] body %u\n"
                         "  mass     req %8.3f        got %8.3f        %s\n"
                         "  position req %s got %s %s\n"
                         "  lin vel  req %s got %s %s\n"
                         "  ang vel  req %s got %s %s\n",
                         ok ? "init ok" : "INIT MISMATCH", id.GetIndex(), p.mass, mass, massOk ? "ok" : "BAD",
                         v3(Vec3(p.position)).c_str(), v3(Vec3(position)).c_str(), posOk ? "ok" : "BAD",
                         v3(p.linearVelocity).c_str(), v3(linearVelocity).c_str(), velOk ? "ok" : "BAD",
                         v3(p.angularVelocity).c_str(), v3(angularVelocity).c_str(), angOk ? "ok" : "BAD");
            std::fflush(ok ? stdout : stderr);
        }
    }

    void BuildContainer()
    {
        const float h = kBoxHalfWidth, t = kWallThickness, H = kBoxHeight;
        auto addStatic = [&](Vec3 halfExtent, RVec3 pos) {
            return Bodies().CreateAndAddBody(BodyCreationSettings(new BoxShape(halfExtent), pos, Quat::sIdentity(),
                                                                  EMotionType::Static, Layers::kNonMoving),
                                             EActivation::DontActivate);
        };
        addStatic(Vec3(h + t, t, h + t), RVec3(0, -t, 0));
        addStatic(Vec3(t, H / 2, h + t), RVec3(-h - t, H / 2, 0));
        addStatic(Vec3(t, H / 2, h + t), RVec3(h + t, H / 2, 0));
        addStatic(Vec3(h + t, H / 2, t), RVec3(0, H / 2, -h - t));
        addStatic(Vec3(h + t, H / 2, t), RVec3(0, H / 2, h + t));
    }

    TempAllocatorImpl                  mTempAllocator;
    JobSystemThreadPool                mJobSystem;
    ObjectLayerPairFilterTable         mObjectPairFilter;
    BroadPhaseLayerInterfaceTable      mBroadPhaseLayers;
    std::unique_ptr<ObjectVsBroadPhaseLayerFilterTable> mObjectVsBroadPhase;
    PhysicsSystem                      mPhysics;

    std::vector<BodyID> mComplexes;
    std::mt19937        mRng{1234};
    uint                mInitChecks = 0;
    uint                mInitFailures = 0;
};

void Run()
{
    viz::Viewer viewer(1280, 800, "Sphere complexes");
    viewer.SetOrbitTarget(Vec3(0, kBoxHeight * 0.4f, 0));
    viewer.SetOrbitDistance(24.0f);

    Scene scene;
    for (int i = 0; i < 40; ++i)
        scene.Spawn();

    BodyManager::DrawSettings drawSettings;
    ColorMode colorMode = ColorMode::Mass;

    KeyEdges keys(viewer.Window());
    bool  churn = false;
    float churnTimer = 0.0f, accumulator = 0.0f, titleTimer = 0.0f;
    double stepMs = 0.0;
    int   frames = 0;
    auto  last = std::chrono::steady_clock::now();

    while (viewer.BeginFrame()) {
        const auto now = std::chrono::steady_clock::now();
        const float frameDt = std::chrono::duration<float>(now - last).count();
        last = now;

        // --- Input -----------------------------------------------------------
        if (keys.Pressed(GLFW_KEY_ESCAPE)) glfwSetWindowShouldClose(viewer.Window(), GLFW_TRUE);
        if (keys.Pressed(GLFW_KEY_SPACE))  scene.Spawn(/*log*/ true);
        if (keys.Pressed(GLFW_KEY_L))      scene.Spawn(scene.LaunchParams(), /*log*/ true);
        if (keys.Pressed(GLFW_KEY_B))      for (int i = 0; i < kBurstCount; ++i) scene.Spawn();
        if (keys.Pressed(GLFW_KEY_D))      scene.DeleteRandom();
        if (keys.Pressed(GLFW_KEY_X))      for (int i = 0; i < kBurstCount; ++i) scene.DeleteRandom();
        if (keys.Pressed(GLFW_KEY_C))      scene.Clear();
        if (keys.Pressed(GLFW_KEY_T))      churn = !churn;
        if (keys.Pressed(GLFW_KEY_W))      drawSettings.mDrawShapeWireframe = !drawSettings.mDrawShapeWireframe;
        if (keys.Pressed(GLFW_KEY_V))      drawSettings.mDrawVelocity = !drawSettings.mDrawVelocity;
        if (keys.Pressed(GLFW_KEY_S))      colorMode = ColorMode((int(colorMode) + 1) % 3);

        // Auto-churn: keep adding, and delete random complexes once above the target count.
        if (churn) {
            for (churnTimer += frameDt; churnTimer >= kChurnInterval; churnTimer -= kChurnInterval) {
                scene.Spawn();
                if (scene.NumComplexes() > size_t(kChurnTarget))
                    scene.DeleteRandom();
            }
        }

        // --- Physics (fixed timestep, capped to avoid spiral of death) --------
        accumulator = std::min(accumulator + frameDt, 4 * kPhysicsDt);
        while (accumulator >= kPhysicsDt) {
            const auto t0 = std::chrono::steady_clock::now();
            scene.Step();
            stepMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            accumulator -= kPhysicsDt;
        }

        // --- Render ----------------------------------------------------------
        scene.Draw(viewer.Renderer(), drawSettings, colorMode);
        viewer.EndFrame();

        ++frames;
        if ((titleTimer += frameDt) >= 0.25f) {
            static const char *kColorNames[] = {"mass", "per-body", "sleep"};
            char title[320];
            std::snprintf(title, sizeof(title),
                          "Sphere complexes | %zu bodies (%u active) | init check %u/%u ok | step %.2f ms | %.0f fps | "
                          "churn %s | color: %s  [Space/B add, L launch, D/X delete, C clear, T churn, S color, "
                          "V velocity, W wire]",
                          scene.NumComplexes(), scene.NumActive(), scene.NumInitChecks() - scene.NumInitFailures(),
                          scene.NumInitChecks(), stepMs, frames / titleTimer, churn ? "ON" : "off",
                          kColorNames[int(colorMode)]);
            viewer.SetTitle(title);
            titleTimer = 0.0f;
            frames = 0;
        }
    }
}

} // namespace

int main()
{
    RegisterDefaultAllocator();
    Trace = TraceImpl;
    JPH_IF_ENABLE_ASSERTS(AssertFailed = AssertFailedImpl;)
    Factory::sInstance = new Factory();
    RegisterTypes();

    int rc = 0;
    try {
        Run();
    } catch (const std::exception &e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        rc = 1;
    }

    UnregisterTypes();
    delete Factory::sInstance;
    Factory::sInstance = nullptr;
    return rc;
}
