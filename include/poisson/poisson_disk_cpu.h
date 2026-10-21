#pragma once

// ---------------------------------------------------------------------------
// poisson_disk_cpu.h – single-threaded CPU twin of poisson::Sampler.
//
// Same background grid (cell size r/sqrt(2), at most one sample per cell,
// x < 0 marks an empty cell), same region semantics: Resample() clears a
// rectangle of cells and refills it by dart throwing, with the samples
// around it held fixed. Being serial, it needs no phase groups: cells are
// visited in a shuffled order and each dart is checked against its 5x5
// neighbourhood, which already contains every earlier acceptance.
// ---------------------------------------------------------------------------
#include "poisson/poisson_disk.h" // Sample

#include <vector>

namespace poisson {

class CpuSampler
{
public:
    CpuSampler(float width, float height, float radius, int trials = 30, unsigned seed = 1);

    /// Clears cells [cx0,cx1) x [cy0,cy1) (clamped to the grid) and refills
    /// them with `trials` rounds of darts. Returns CPU time in ms.
    float Resample(int cx0, int cy0, int cx1, int cy1);
    float ResampleAll() { return Resample(0, 0, mGridW, mGridH); }

    /// No two samples touching the cell region (plus the 2-cell conflict
    /// radius) are closer than `radius`.
    bool Verify(int cx0, int cy0, int cx1, int cy1) const;
    bool VerifyAll() const { return Verify(0, 0, mGridW, mGridH); }

    /// Row-major, GridW() * GridH() entries.
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

    std::vector<Sample> mCells;
    std::vector<int>    mOrder; // scratch: shuffled cell order
};

} // namespace poisson
