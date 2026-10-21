#pragma once

// ---------------------------------------------------------------------------
// jolt_executor.h – voromesh::Executor on top of a Jolt JobSystem.
//
// Header-only and outside the voromesh library, so only code that already
// links Jolt includes it. Topology runs between PhysicsSystem::Update calls
// (Update waits for all of its jobs before returning), so the same pool's
// workers are idle then and free to take this work.
//
// Creating a Jolt job costs a couple of microseconds on the calling thread
// and waking a sleeping worker costs more, so:
//   * Run() starts at most one job per pool worker, and every participant
//     (the caller included) claims task indices from a shared counter.
//   * RunRounds() starts those jobs once for all rounds; between rounds the
//     helpers spin until the caller publishes the next one.
// The calling thread works on every round and only ever waits for tasks to
// finish, never for a particular helper, so a helper that starts late (or not
// at all, if the pool is busy) cannot stall anything. Which thread runs a task
// never affects the result, since tasks write only their own outputs.
//
// One barrier is reused for every call, and Jolt allows only one thread to
// wait on a barrier, so an executor must only be used from one thread at a time.
// ---------------------------------------------------------------------------
#include "voromesh/parallel.h"

#include <Jolt/Jolt.h>

#include <Jolt/Core/JobSystem.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace voromesh {

class JoltExecutor final : public Executor
{
public:
    explicit JoltExecutor(JPH::JobSystem &jobs) : mJobs(jobs), mBarrier(jobs.CreateBarrier()) {}
    ~JoltExecutor() override { mJobs.DestroyBarrier(mBarrier); }

    JoltExecutor(const JoltExecutor &) = delete;
    JoltExecutor &operator=(const JoltExecutor &) = delete;

    /// Threads that can run tasks at once (pool workers + the caller).
    int Concurrency() const { return mJobs.GetMaxConcurrency(); }

    void Run(int numTasks, const TaskFn &fn) override
    {
        std::atomic<int> next{0};
        auto drain = [&] {
            for (int t = next.fetch_add(1, std::memory_order_relaxed); t < numTasks;
                 t = next.fetch_add(1, std::memory_order_relaxed))
                fn(t);
        };
        Launch(std::min(numTasks, Concurrency()) - 1, drain);
        drain();
        mJobs.WaitForJobs(mBarrier);
        mHandles.clear(); // release the job references
    }

    void RunRounds(int rounds, int numTasks, const RoundFn &before, const RoundTaskFn &task,
                   const RoundFn &after) override
    {
        // Per-round counters, so a helper that lags a round behind can never
        // claim a task of the wrong round.
        std::unique_ptr<std::atomic<int>[]> next(new std::atomic<int>[rounds]);
        std::unique_ptr<std::atomic<int>[]> done(new std::atomic<int>[rounds]);
        for (int r = 0; r < rounds; ++r) {
            next[r].store(0, std::memory_order_relaxed);
            done[r].store(0, std::memory_order_relaxed);
        }
        std::atomic<int> published{-1}; // last round whose before() has run

        auto work = [&](int r) {
            int n = 0;
            for (int t = next[r].fetch_add(1, std::memory_order_relaxed); t < numTasks;
                 t = next[r].fetch_add(1, std::memory_order_relaxed), ++n)
                task(r, t);
            if (n > 0)
                done[r].fetch_add(n, std::memory_order_release);
        };
        auto helper = [&] {
            for (int r = 0; r < rounds; ++r) {
                SpinUntil([&] { return published.load(std::memory_order_acquire) >= r; });
                work(r);
            }
        };

        Launch(std::min(numTasks, Concurrency()) - 1, helper);
        for (int r = 0; r < rounds; ++r) {
            before(r);
            published.store(r, std::memory_order_release);
            work(r);
            SpinUntil([&] { return done[r].load(std::memory_order_acquire) == numTasks; });
            after(r);
        }
        mJobs.WaitForJobs(mBarrier);
        mHandles.clear();
    }

private:
    template <class Fn>
    void Launch(int helpers, const Fn &fn)
    {
        mHandles.clear();
        for (int h = 0; h < helpers; ++h)
            mHandles.push_back(mJobs.CreateJob("voromesh", JPH::Color::sCyan, fn));
        if (!mHandles.empty())
            mBarrier->AddJobs(mHandles.data(), JPH::uint(mHandles.size()));
    }

    template <class Pred>
    static void SpinUntil(const Pred &ready)
    {
        for (int i = 0; !ready(); ++i) {
#if defined(_M_X64) || defined(__x86_64__)
            if (i < 1024) {
                _mm_pause();
                continue;
            }
#endif
            std::this_thread::yield();
        }
    }

    JPH::JobSystem                        &mJobs;
    JPH::JobSystem::Barrier               *mBarrier;
    std::vector<JPH::JobSystem::JobHandle> mHandles;
};

} // namespace voromesh
