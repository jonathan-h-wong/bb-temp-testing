#pragma once

// ---------------------------------------------------------------------------
// viewer.h – minimal realtime visualizer for Jolt.
//
// GLRenderer implements Jolt's DebugRendererSimple (lines + triangles) on top
// of legacy OpenGL vertex arrays, so anything Jolt can debug-draw
// (PhysicsSystem::DrawBodies, DrawWireBox, ...) shows up in the window.
// Viewer owns the GLFW window, an orbit camera, and the renderer.
// ---------------------------------------------------------------------------
#include <Jolt/Jolt.h>
#include <Jolt/Renderer/DebugRendererSimple.h>

#include <string>
#include <vector>

struct GLFWwindow;

namespace viz {

class GLRenderer final : public JPH::DebugRendererSimple
{
public:
    GLRenderer();

    void DrawLine(JPH::RVec3Arg from, JPH::RVec3Arg to, JPH::ColorArg color) override;
    void DrawTriangle(JPH::RVec3Arg v1, JPH::RVec3Arg v2, JPH::RVec3Arg v3, JPH::ColorArg color,
                      ECastShadow castShadow) override;
    void DrawText3D(JPH::RVec3Arg, const std::string_view &, JPH::ColorArg, float) override {}

    /// Submit everything queued since the last Flush() and clear the queues.
    void Flush();

private:
    struct Vertex
    {
        float x, y, z;
        unsigned char r, g, b, a;
    };

    std::vector<Vertex> mLines;
    std::vector<Vertex> mTriangles;
};

class Viewer
{
public:
    Viewer(int width, int height, const char *title);
    ~Viewer();

    Viewer(const Viewer &) = delete;
    Viewer &operator=(const Viewer &) = delete;

    /// Polls events; returns false once the window has been closed.
    bool BeginFrame();
    void EndFrame();

    GLRenderer &Renderer() { return mRenderer; }
    GLFWwindow *Window() { return mWindow; }

    void SetTitle(const std::string &title);
    void SetOrbitTarget(JPH::Vec3 target) { mTarget = target; }
    void SetOrbitDistance(float distance) { mDistance = distance; }

private:
    static void sScrollCallback(GLFWwindow *window, double dx, double dy);

    GLFWwindow *mWindow = nullptr;
    GLRenderer  mRenderer;

    JPH::Vec3 mTarget   = JPH::Vec3::sZero();
    float     mYaw      = 0.6f;
    float     mPitch    = 0.45f;
    float     mDistance = 20.0f;
    double    mLastMouseX = 0.0, mLastMouseY = 0.0;
};

} // namespace viz
