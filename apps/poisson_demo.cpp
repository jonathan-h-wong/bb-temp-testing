// ---------------------------------------------------------------------------
// poisson_demo – interactive test of the GPU Poisson disk sampler.
//
// The domain is split into square chunks of N x N background-grid cells.
// Clicking a chunk clears every sample inside it and refills it on the GPU,
// with the surrounding samples held fixed as constraints. Each resample is
// checked on the host for distance violations (shown in the title bar).
//
// Controls
//   Left click / drag   resample the chunk(s) under the cursor
//   Right hold          resample the hovered chunk every frame (stress test)
//   [ / ]               halve / double chunk size (1 = a single grid cell)
//   - / =               shrink / grow the sample radius (rebuilds the domain)
//   R                   resample the whole domain
//   D                   toggle dots / radius-r/2 disks (disks never overlap)
//   G                   toggle chunk grid lines
//   Esc                 quit
//
// Usage: poisson_demo [width=1024] [height=1024] [radius=8] [trials=30] [seed=1]
// ---------------------------------------------------------------------------
#include "poisson/poisson_disk.h"

#include <GLFW/glfw3.h> // also pulls in the system GL 1.1 header

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <stdexcept>
#include <vector>

#ifndef GL_MULTISAMPLE
#define GL_MULTISAMPLE 0x809D // GL 1.3; Windows' gl.h only goes up to 1.1
#endif

namespace {

constexpr float kFlashSeconds = 0.4f; // how long a resampled chunk stays highlighted

struct Flash
{
    int   cx0, cy0, cx1, cy1;
    float age;
};

struct Input
{
    bool resampleAll   = false;
    int  chunkSizeStep = 0; // +1 doubles, -1 halves
    int  radiusStep    = 0;
    bool toggleDisks   = false;
    bool toggleGrid    = false;
};

void KeyCallback(GLFWwindow *window, int key, int, int action, int)
{
    if (action != GLFW_PRESS && action != GLFW_REPEAT)
        return;
    auto *in = static_cast<Input *>(glfwGetWindowUserPointer(window));
    switch (key) {
    case GLFW_KEY_ESCAPE:        glfwSetWindowShouldClose(window, GLFW_TRUE); break;
    case GLFW_KEY_R:             in->resampleAll = true; break;
    case GLFW_KEY_RIGHT_BRACKET: in->chunkSizeStep = +1; break;
    case GLFW_KEY_LEFT_BRACKET:  in->chunkSizeStep = -1; break;
    case GLFW_KEY_EQUAL:         in->radiusStep = +1; break;
    case GLFW_KEY_MINUS:         in->radiusStep = -1; break;
    case GLFW_KEY_D:             in->toggleDisks = true; break;
    case GLFW_KEY_G:             in->toggleGrid = true; break;
    default: break;
    }
}

// Letterboxed mapping between domain units and window pixels.
struct View
{
    float scale = 1.0f; // window pixels per domain unit
    float offX = 0.0f, offY = 0.0f;

    void Fit(int winW, int winH, float domW, float domH)
    {
        scale = std::min(winW / domW, winH / domH);
        offX  = 0.5f * (winW - domW * scale);
        offY  = 0.5f * (winH - domH * scale);
    }
};

void DrawRect(float x0, float y0, float x1, float y1, GLenum mode)
{
    const float v[] = {x0, y0, x1, y0, x1, y1, x0, y1};
    glVertexPointer(2, GL_FLOAT, 0, v);
    glDrawArrays(mode, 0, 4);
}

int Run(int argc, char **argv)
{
    float    width  = argc > 1 ? (float)std::atof(argv[1]) : 1024.0f;
    float    height = argc > 2 ? (float)std::atof(argv[2]) : 1024.0f;
    float    radius = argc > 3 ? (float)std::atof(argv[3]) : 8.0f;
    int      trials = argc > 4 ? std::atoi(argv[4]) : 30;
    unsigned seed   = argc > 5 ? (unsigned)std::atoi(argv[5]) : 1u;

    glfwSetErrorCallback([](int code, const char *msg) { std::fprintf(stderr, "GLFW error %d: %s\n", code, msg); });
    if (!glfwInit())
        throw std::runtime_error("glfwInit failed");

    glfwWindowHint(GLFW_SAMPLES, 4);
    GLFWwindow *window = glfwCreateWindow(900, int(900 * height / width), "Poisson disk demo", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("glfwCreateWindow failed");
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    Input input;
    glfwSetWindowUserPointer(window, &input);
    glfwSetKeyCallback(window, &KeyCallback);

    auto sampler = std::make_unique<poisson::Sampler>(width, height, radius, trials, seed);
    std::printf("Domain %.0f x %.0f, r=%.3f, grid %d x %d, %d trials/cell\n", width, height, radius,
                sampler->GridW(), sampler->GridH(), trials);
    std::printf("Initial fill: %s\n", sampler->VerifyAll() ? "PASS" : "FAIL");

    int   chunkCells = 16;
    bool  showDisks  = false;
    bool  showGrid   = true;
    bool  allValid   = sampler->VerifyAll();
    int   resamples  = 0;
    float lastGpuMs  = 0.0f;
    int   lastChunkX = -1, lastChunkY = -1; // chunk last resampled by the current left drag

    std::vector<Flash>        flashes;
    std::vector<float>        points;
    bool                      pointsDirty = true;
    View                      view;

    auto resampleChunk = [&](int chunkX, int chunkY) {
        int cx0 = chunkX * chunkCells, cy0 = chunkY * chunkCells;
        int cx1 = cx0 + chunkCells, cy1 = cy0 + chunkCells;
        lastGpuMs = sampler->Resample(cx0, cy0, cx1, cy1);
        allValid  = allValid && sampler->Verify(cx0, cy0, cx1, cy1);
        ++resamples;
        pointsDirty = true;
        flashes.push_back({cx0, cy0, cx1, cy1, 0.0f});
    };

    auto   prevTime   = std::chrono::steady_clock::now();
    float  titleTimer = 0.0f, fpsAccum = 0.0f;
    int    fpsFrames  = 0;
    float  fps        = 0.0f;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        auto  now = std::chrono::steady_clock::now();
        float dt  = std::chrono::duration<float>(now - prevTime).count();
        prevTime  = now;

        // ---- Keyboard ----------------------------------------------------
        if (input.chunkSizeStep != 0) {
            int maxCells = std::max(sampler->GridW(), sampler->GridH());
            chunkCells   = input.chunkSizeStep > 0 ? std::min(chunkCells * 2, maxCells) : std::max(chunkCells / 2, 1);
            input.chunkSizeStep = 0;
            lastChunkX = lastChunkY = -1;
        }
        if (input.radiusStep != 0) {
            radius  = std::clamp(radius * (input.radiusStep > 0 ? 1.25f : 0.8f), 1.0f, 128.0f);
            sampler = std::make_unique<poisson::Sampler>(width, height, radius, trials, seed);
            allValid    = sampler->VerifyAll();
            pointsDirty = true;
            flashes.clear();
            input.radiusStep = 0;
            std::printf("r=%.3f, grid %d x %d (%d cells), %s\n", radius, sampler->GridW(), sampler->GridH(),
                        sampler->GridW() * sampler->GridH(), allValid ? "PASS" : "FAIL");
        }
        if (input.resampleAll) {
            lastGpuMs = sampler->ResampleAll();
            allValid  = sampler->VerifyAll();
            ++resamples;
            pointsDirty = true;
            flashes.push_back({0, 0, sampler->GridW(), sampler->GridH(), 0.0f});
            input.resampleAll = false;
        }
        if (input.toggleDisks) { showDisks = !showDisks; input.toggleDisks = false; }
        if (input.toggleGrid)  { showGrid = !showGrid;   input.toggleGrid = false; }

        // ---- Mouse -------------------------------------------------------
        int winW, winH, fbW, fbH;
        glfwGetWindowSize(window, &winW, &winH);
        glfwGetFramebufferSize(window, &fbW, &fbH);
        if (winW <= 0 || winH <= 0) { // minimized
            glfwWaitEvents();
            continue;
        }
        view.Fit(winW, winH, width, height);

        double mx, my;
        glfwGetCursorPos(window, &mx, &my);
        float domX = float(mx - view.offX) / view.scale;
        float domY = float(my - view.offY) / view.scale;
        float chunkSize = chunkCells * sampler->CellSize();
        bool  inDomain  = domX >= 0 && domY >= 0 && domX < width && domY < height;
        int   hoverX    = inDomain ? int(domX / chunkSize) : -1;
        int   hoverY    = inDomain ? int(domY / chunkSize) : -1;

        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            // Resample once per chunk entered, so dragging paints across chunks.
            if (inDomain && (hoverX != lastChunkX || hoverY != lastChunkY)) {
                resampleChunk(hoverX, hoverY);
                lastChunkX = hoverX;
                lastChunkY = hoverY;
            }
        } else {
            lastChunkX = lastChunkY = -1;
        }
        if (inDomain && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
            resampleChunk(hoverX, hoverY);

        // ---- Rebuild point list ------------------------------------------
        if (pointsDirty) {
            points.clear();
            for (const poisson::Sample &s : sampler->Cells())
                if (s.x >= 0.0f) {
                    points.push_back(s.x);
                    points.push_back(s.y);
                }
            pointsDirty = false;
        }

        // ---- Draw --------------------------------------------------------
        glViewport(0, 0, fbW, fbH);
        glClearColor(0.07f, 0.07f, 0.09f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Letterboxed viewport in framebuffer pixels (handles DPI scaling).
        float fbScale = float(fbW) / float(winW);
        glViewport(int(view.offX * fbScale), int((winH - view.offY - height * view.scale) * fbScale),
                   int(width * view.scale * fbScale), int(height * view.scale * fbScale));
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.0, width, height, 0.0, -1.0, 1.0); // y down, like the mouse
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnableClientState(GL_VERTEX_ARRAY);

        glColor4f(0.12f, 0.13f, 0.16f, 1.0f);
        DrawRect(0, 0, width, height, GL_TRIANGLE_FAN);

        // Recently resampled chunks fade out.
        float cs = sampler->CellSize();
        for (Flash &f : flashes) {
            f.age += dt;
            float a = 0.35f * (1.0f - f.age / kFlashSeconds);
            glColor4f(0.2f, 0.75f, 1.0f, std::max(a, 0.0f));
            DrawRect(f.cx0 * cs, f.cy0 * cs, f.cx1 * cs, f.cy1 * cs, GL_TRIANGLE_FAN);
        }
        flashes.erase(std::remove_if(flashes.begin(), flashes.end(), [](const Flash &f) { return f.age >= kFlashSeconds; }),
                      flashes.end());

        if (showGrid) {
            std::vector<float> lines;
            for (float x = chunkSize; x < width; x += chunkSize)
                lines.insert(lines.end(), {x, 0.0f, x, height});
            for (float y = chunkSize; y < height; y += chunkSize)
                lines.insert(lines.end(), {0.0f, y, width, y});
            glColor4f(1.0f, 1.0f, 1.0f, chunkCells == 1 ? 0.05f : 0.12f);
            glLineWidth(1.0f);
            glVertexPointer(2, GL_FLOAT, 0, lines.data());
            glDrawArrays(GL_LINES, 0, GLsizei(lines.size() / 2));
        }

        // Samples. As disks of diameter r they may touch but never overlap.
        float pxPerUnit = view.scale * fbScale;
        glEnable(GL_POINT_SMOOTH);
        glPointSize(showDisks ? std::max(radius * pxPerUnit, 1.0f) : std::clamp(radius * pxPerUnit * 0.3f, 2.0f, 6.0f));
        glColor4f(1.0f, 0.82f, 0.35f, showDisks ? 0.85f : 1.0f);
        if (!points.empty()) {
            glVertexPointer(2, GL_FLOAT, 0, points.data());
            glDrawArrays(GL_POINTS, 0, GLsizei(points.size() / 2));
        }
        glDisable(GL_POINT_SMOOTH);

        if (inDomain) {
            glColor4f(1.0f, 1.0f, 1.0f, 0.8f);
            glLineWidth(2.0f);
            DrawRect(hoverX * chunkSize, hoverY * chunkSize, std::min((hoverX + 1) * chunkSize, width),
                     std::min((hoverY + 1) * chunkSize, height), GL_LINE_LOOP);
        }

        glDisableClientState(GL_VERTEX_ARRAY);
        glfwSwapBuffers(window);

        // ---- Title stats -------------------------------------------------
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
            char title[256];
            std::snprintf(title, sizeof(title),
                          "Poisson disk | %zu samples | r=%.2f | chunk %dx%d cells | last resample %.3f ms GPU | "
                          "%d resamples | %s | %.0f fps",
                          points.size() / 2, radius, chunkCells, chunkCells, lastGpuMs, resamples,
                          allValid ? "valid" : "CONFLICT DETECTED", fps);
            glfwSetWindowTitle(window, title);
        }
    }

    sampler.reset();
    glfwDestroyWindow(window);
    glfwTerminate();
    return allValid ? 0 : 1;
}

} // namespace

int main(int argc, char **argv)
{
    try {
        return Run(argc, argv);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "Fatal: %s\n", e.what());
        return 1;
    }
}
