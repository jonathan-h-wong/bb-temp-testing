#include "poisson/poisson_disk_cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace poisson {

namespace {

// Same stateless PCG-style hash as the GPU sampler.
unsigned PcgHash(unsigned v)
{
    unsigned state = v * 747796405u + 2891336453u;
    unsigned word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float Rand01(unsigned cell, unsigned round, unsigned seed, unsigned stream)
{
    unsigned h = PcgHash(cell ^ PcgHash(round * 2654435761u ^ PcgHash(seed + stream * 0x9E3779B9u)));
    return (h & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

} // namespace

CpuSampler::CpuSampler(float width, float height, float radius, int trials, unsigned seed)
    : mWidth(width), mHeight(height), mRadius(radius), mCellSize(radius / std::sqrt(2.0f)), mTrials(trials),
      mSeed(seed)
{
    mGridW = (int)std::ceil(width / mCellSize);
    mGridH = (int)std::ceil(height / mCellSize);
    mCells.assign(size_t(mGridW) * mGridH, Sample{-1.0f, -1.0f});
    ResampleAll();
}

float CpuSampler::Resample(int cx0, int cy0, int cx1, int cy1)
{
    auto t0 = std::chrono::steady_clock::now();
    cx0 = std::max(cx0, 0), cy0 = std::max(cy0, 0);
    cx1 = std::min(cx1, mGridW), cy1 = std::min(cy1, mGridH);
    if (cx0 >= cx1 || cy0 >= cy1)
        return 0.0f;

    mOrder.clear();
    for (int cy = cy0; cy < cy1; ++cy)
        for (int cx = cx0; cx < cx1; ++cx) {
            mCells[size_t(cy) * mGridW + cx] = Sample{-1.0f, -1.0f};
            mOrder.push_back(cy * mGridW + cx);
        }
    // Fisher-Yates with the hash RNG, so scan order does not bias the fill.
    for (size_t k = mOrder.size() - 1; k > 0; --k) {
        size_t j = size_t(Rand01(unsigned(k), mRoundBase, mSeed, 2) * (k + 1));
        std::swap(mOrder[k], mOrder[std::min(j, k)]);
    }

    const float r2 = mRadius * mRadius;
    for (int round = 0; round < mTrials; ++round) {
        unsigned rnd = mRoundBase + unsigned(round);
        for (int cell : mOrder) {
            if (mCells[cell].x >= 0.0f)
                continue;
            int   cx = cell % mGridW, cy = cell / mGridW;
            float x  = (cx + Rand01(cell, rnd, mSeed, 0)) * mCellSize;
            float y  = (cy + Rand01(cell, rnd, mSeed, 1)) * mCellSize;
            if (x >= mWidth || y >= mHeight)
                continue;

            bool ok = true;
            for (int ny = std::max(cy - 2, 0); ok && ny <= std::min(cy + 2, mGridH - 1); ++ny)
                for (int nx = std::max(cx - 2, 0); nx <= std::min(cx + 2, mGridW - 1); ++nx) {
                    Sample q = mCells[size_t(ny) * mGridW + nx];
                    if (q.x < 0.0f)
                        continue;
                    float dx = q.x - x, dy = q.y - y;
                    if (dx * dx + dy * dy < r2) {
                        ok = false;
                        break;
                    }
                }
            if (ok)
                mCells[cell] = Sample{x, y};
        }
    }
    mRoundBase += unsigned(mTrials);
    return std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

bool CpuSampler::Verify(int cx0, int cy0, int cx1, int cy1) const
{
    float r2 = mRadius * mRadius;
    int vx0 = std::max(cx0 - 2, 0), vx1 = std::min(cx1 + 2, mGridW);
    int vy0 = std::max(cy0 - 2, 0), vy1 = std::min(cy1 + 2, mGridH);
    for (int cy = vy0; cy < vy1; ++cy) {
        for (int cx = vx0; cx < vx1; ++cx) {
            Sample a = mCells[size_t(cy) * mGridW + cx];
            if (a.x < 0)
                continue;
            for (int ny = std::max(cy - 2, 0); ny <= std::min(cy + 2, mGridH - 1); ++ny)
                for (int nx = std::max(cx - 2, 0); nx <= std::min(cx + 2, mGridW - 1); ++nx) {
                    if (nx == cx && ny == cy)
                        continue;
                    Sample b = mCells[size_t(ny) * mGridW + nx];
                    if (b.x < 0)
                        continue;
                    float dx = a.x - b.x, dy = a.y - b.y;
                    if (dx * dx + dy * dy < r2)
                        return false;
                }
        }
    }
    return true;
}

} // namespace poisson
