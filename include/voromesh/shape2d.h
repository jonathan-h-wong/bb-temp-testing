#pragma once

// ---------------------------------------------------------------------------
// shape2d.h – 2D target shapes for the VoroMesh experiments.
// ---------------------------------------------------------------------------
#include <vector>

namespace voromesh {

/// Same layout as poisson::Sample, so Poisson cells can be used as generator slots.
struct Vec2
{
    float x, y;
};

/// Closed polygon (last vertex is not repeated) tracing the classic parametric
/// heart x = 16 sin^3 t, y = 13 cos t - 5 cos 2t - 2 cos 3t - cos 4t.
/// `width` is the heart's total width in world units, centred on (cx, cy).
/// World y points down (screen convention), so the heart renders upright.
std::vector<Vec2> MakeHeart(float cx, float cy, float width, int segments);

} // namespace voromesh
