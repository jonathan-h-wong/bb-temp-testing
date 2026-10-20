#pragma once

// ---------------------------------------------------------------------------
// poisson_disk.h – GPU Poisson disk sampling via grid-based dart throwing.
//
// Wraps the parallel phase-group dart thrower (after Wei 2008) so it can be
// re-run on any rectangular block of background-grid cells: the block is
// cleared and refilled while the samples around it stay fixed and act as
// constraints, so the full sample set remains a valid Poisson disk set.
// ---------------------------------------------------------------------------
#include <vector>

namespace poisson {

/// Same layout as CUDA's float2. x < 0 marks an empty cell.
struct Sample
{
    float x, y;
};

class Sampler
{
public:
    Sampler(float width, float height, float radius, int trials = 30, unsigned seed = 1);
    ~Sampler();

    Sampler(const Sampler &) = delete;
    Sampler &operator=(const Sampler &) = delete;

    /// Clears cells [cx0,cx1) x [cy0,cy1) (clamped to the grid), throws
    /// `trials` rounds of darts into them, and copies the region back into
    /// Cells(). Every call uses fresh random numbers. Returns GPU time in ms.
    float Resample(int cx0, int cy0, int cx1, int cy1);
    float ResampleAll() { return Resample(0, 0, mGridW, mGridH); }

    /// Host-side check that no two samples touching the given cell region
    /// (expanded by the 2-cell conflict radius) are closer than `radius`.
    bool Verify(int cx0, int cy0, int cx1, int cy1) const;
    bool VerifyAll() const { return Verify(0, 0, mGridW, mGridH); }

    /// Host mirror of the grid, row-major, GridW() * GridH() entries.
    const std::vector<Sample> &Cells() const { return mCells; }

    float Width() const { return mWidth; }
    float Height() const { return mHeight; }
    float Radius() const { return mRadius; }
    float CellSize() const { return mCellSize; }
    int   GridW() const { return mGridW; }
    int   GridH() const { return mGridH; }
    int   Trials() const { return mTrials; }

private:
    float    mWidth, mHeight, mRadius, mCellSize;
    int      mGridW, mGridH, mTrials;
    unsigned mSeed;
    unsigned mRoundBase = 0; // advanced by `trials` per Resample so darts never repeat

    void *mDeviceGrid = nullptr; // float2[gridW * gridH]
    void *mStart = nullptr, *mStop = nullptr; // cudaEvent_t

    std::vector<Sample> mCells;
};

} // namespace poisson
