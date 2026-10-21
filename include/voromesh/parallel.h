#pragma once

// ---------------------------------------------------------------------------
// parallel.h – the fork/join hook the CPU optimiser uses for parallel loops.
//
// The library has no threads of its own: the application adapts whatever job
// system it already runs (see jolt_executor.h for Jolt's), so topology work
// shares cores with physics instead of competing with a second pool.
// Without an executor every loop runs inline on the calling thread.
//
// Loops are split into fixed-size chunks that never depend on the thread
// count, and per-chunk results are combined in chunk order, so results are
// bit-identical however many threads run them.
// ---------------------------------------------------------------------------
#include <functional>

namespace voromesh {

class Executor
{
public:
    using TaskFn      = std::function<void(int task)>;
    using RoundFn     = std::function<void(int round)>;
    using RoundTaskFn = std::function<void(int round, int task)>;

    virtual ~Executor() = default;

    /// Runs fn(task) once for every task in [0, numTasks) and returns when
    /// all have finished. Tasks may run on the calling thread. Calls are not
    /// re-entrant: a task must not call back into the executor.
    virtual void Run(int numTasks, const TaskFn &fn) = 0;

    /// `rounds` fork/joins back to back: before(r) on the calling thread, then
    /// task(r, t) for every t in [0, numTasks), then after(r) on the calling
    /// thread. This default runs one Run() per round; an executor can instead
    /// keep its workers through all rounds to skip the per-round wake-ups.
    virtual void RunRounds(int rounds, int numTasks, const RoundFn &before, const RoundTaskFn &task,
                           const RoundFn &after)
    {
        for (int r = 0; r < rounds; ++r) {
            before(r);
            Run(numTasks, [&](int t) { task(r, t); });
            after(r);
        }
    }
};

/// Runs the tasks on `exec`, or inline when there is no executor or only one task.
inline void RunTasks(Executor *exec, int numTasks, const Executor::TaskFn &fn)
{
    if (numTasks <= 0)
        return;
    if (!exec || numTasks == 1) {
        for (int t = 0; t < numTasks; ++t)
            fn(t);
        return;
    }
    exec->Run(numTasks, fn);
}

/// Executor::RunRounds on `exec`, or inline when there is no executor or only one task.
inline void RunRounds(Executor *exec, int rounds, int numTasks, const Executor::RoundFn &before,
                      const Executor::RoundTaskFn &task, const Executor::RoundFn &after)
{
    if (exec && numTasks > 1) {
        exec->RunRounds(rounds, numTasks, before, task, after);
        return;
    }
    for (int r = 0; r < rounds; ++r) {
        before(r);
        for (int t = 0; t < numTasks; ++t)
            task(r, t);
        after(r);
    }
}

} // namespace voromesh
