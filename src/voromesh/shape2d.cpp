#include "voromesh/shape2d.h"

#include <cmath>

namespace voromesh {

std::vector<Vec2> MakeHeart(float cx, float cy, float width, int segments)
{
    // Curve spans x in [-16, 16] and y in roughly [-17, 12]; centre it vertically.
    constexpr float kPi     = 3.14159265358979f;
    constexpr float kMidY   = -2.5f;
    const float     scale   = width / 32.0f;

    std::vector<Vec2> pts;
    pts.reserve(segments);
    for (int i = 0; i < segments; ++i) {
        float t = 2.0f * kPi * i / segments;
        float s = std::sin(t);
        float x = 16.0f * s * s * s;
        float y = 13.0f * std::cos(t) - 5.0f * std::cos(2 * t) - 2.0f * std::cos(3 * t) - std::cos(4 * t);
        pts.push_back({cx + scale * x, cy - scale * (y - kMidY)});
    }
    return pts;
}

} // namespace voromesh
