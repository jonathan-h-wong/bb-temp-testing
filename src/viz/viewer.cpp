#include "viz/viewer.h"

#include <GLFW/glfw3.h> // also pulls in the system GL 1.1 header

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#ifndef GL_MULTISAMPLE
#define GL_MULTISAMPLE 0x809D // GL 1.3; Windows' gl.h only goes up to 1.1
#endif

using namespace JPH;

namespace viz {

// ---------------------------------------------------------------------------
// GLRenderer
// ---------------------------------------------------------------------------
GLRenderer::GLRenderer()
{
    // DebugRendererSimple's constructor already calls DebugRenderer::Initialize().
    mLines.reserve(1 << 16);
    mTriangles.reserve(1 << 20);
}

void GLRenderer::DrawLine(RVec3Arg from, RVec3Arg to, ColorArg color)
{
    mLines.push_back({float(from.GetX()), float(from.GetY()), float(from.GetZ()), color.r, color.g, color.b, color.a});
    mLines.push_back({float(to.GetX()), float(to.GetY()), float(to.GetZ()), color.r, color.g, color.b, color.a});
}

void GLRenderer::DrawTriangle(RVec3Arg v1, RVec3Arg v2, RVec3Arg v3, ColorArg color, ECastShadow)
{
    // Cheap two-sided directional lighting baked into the vertex color, so the
    // shapes read as 3D without needing shaders.
    static const Vec3 kLightDir = Vec3(0.4f, 1.0f, 0.3f).Normalized();
    Vec3 a(v1), b(v2), c(v3);
    Vec3 n = (b - a).Cross(c - a).NormalizedOr(Vec3::sAxisY());
    float shade = 0.35f + 0.65f * std::abs(n.Dot(kLightDir));

    auto lit = [shade](uint8 ch) { return uint8(std::min(255.0f, ch * shade)); };
    uint8 r = lit(color.r), g = lit(color.g), bl = lit(color.b);

    for (const Vec3 &p : {a, b, c})
        mTriangles.push_back({p.GetX(), p.GetY(), p.GetZ(), r, g, bl, color.a});
}

void GLRenderer::Flush()
{
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    if (!mTriangles.empty()) {
        // Push filled geometry back slightly so wireframe lines on top don't z-fight.
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(1.0f, 1.0f);
        glVertexPointer(3, GL_FLOAT, sizeof(Vertex), &mTriangles[0].x);
        glColorPointer(4, GL_UNSIGNED_BYTE, sizeof(Vertex), &mTriangles[0].r);
        glDrawArrays(GL_TRIANGLES, 0, GLsizei(mTriangles.size()));
        glDisable(GL_POLYGON_OFFSET_FILL);
    }
    if (!mLines.empty()) {
        glVertexPointer(3, GL_FLOAT, sizeof(Vertex), &mLines[0].x);
        glColorPointer(4, GL_UNSIGNED_BYTE, sizeof(Vertex), &mLines[0].r);
        glDrawArrays(GL_LINES, 0, GLsizei(mLines.size()));
    }

    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    mLines.clear();
    mTriangles.clear();
    NextFrame();
}

// ---------------------------------------------------------------------------
// Viewer
// ---------------------------------------------------------------------------
Viewer::Viewer(int width, int height, const char *title)
{
    glfwSetErrorCallback([](int code, const char *msg) { std::fprintf(stderr, "GLFW error %d: %s\n", code, msg); });
    if (!glfwInit())
        throw std::runtime_error("glfwInit failed");

    glfwWindowHint(GLFW_SAMPLES, 4);
    mWindow = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!mWindow) {
        glfwTerminate();
        throw std::runtime_error("glfwCreateWindow failed");
    }
    glfwMakeContextCurrent(mWindow);
    glfwSwapInterval(1);

    glfwSetWindowUserPointer(mWindow, this);
    glfwSetScrollCallback(mWindow, &Viewer::sScrollCallback);
    glfwGetCursorPos(mWindow, &mLastMouseX, &mLastMouseY);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glDisable(GL_CULL_FACE);
}

Viewer::~Viewer()
{
    glfwDestroyWindow(mWindow);
    glfwTerminate();
}

void Viewer::sScrollCallback(GLFWwindow *window, double, double dy)
{
    auto *self = static_cast<Viewer *>(glfwGetWindowUserPointer(window));
    self->mDistance = std::clamp(self->mDistance * std::pow(0.9f, float(dy)), 2.0f, 200.0f);
}

bool Viewer::BeginFrame()
{
    glfwPollEvents();
    if (glfwWindowShouldClose(mWindow))
        return false;

    // Left-drag orbits the camera around the target.
    double mx, my;
    glfwGetCursorPos(mWindow, &mx, &my);
    if (glfwGetMouseButton(mWindow, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        mYaw   -= float(mx - mLastMouseX) * 0.005f;
        mPitch  = std::clamp(mPitch + float(my - mLastMouseY) * 0.005f, -1.5f, 1.5f);
    }
    mLastMouseX = mx;
    mLastMouseY = my;

    int w, h;
    glfwGetFramebufferSize(mWindow, &w, &h);
    h = std::max(h, 1);
    glViewport(0, 0, w, h);
    glClearColor(0.12f, 0.13f, 0.16f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    const float nearZ = 0.1f, farZ = 500.0f;
    const float halfH = nearZ * std::tan(0.5f * DegreesToRadians(60.0f));
    const float halfW = halfH * float(w) / float(h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-halfW, halfW, -halfH, halfH, nearZ, farZ);

    Vec3 eye = mTarget + mDistance * Vec3(std::cos(mPitch) * std::sin(mYaw), std::sin(mPitch),
                                          std::cos(mPitch) * std::cos(mYaw));
    Float4 view[4];
    Mat44::sLookAt(eye, mTarget, Vec3::sAxisY()).StoreFloat4x4(view); // column-major, as GL expects
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(&view[0].x);

    mRenderer.SetCameraPos(RVec3(eye)); // lets Jolt pick geometry LODs
    return true;
}

void Viewer::EndFrame()
{
    mRenderer.Flush();
    glfwSwapBuffers(mWindow);
}

void Viewer::SetTitle(const std::string &title)
{
    glfwSetWindowTitle(mWindow, title.c_str());
}

} // namespace viz
