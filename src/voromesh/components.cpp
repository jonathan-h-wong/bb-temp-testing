// Connected components with persistent IDs; see components.h for the design.

#include "voromesh/components.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <unordered_map>

namespace voromesh {

namespace {

constexpr int kCellsPerChunk = 64; // shell cells per union-find task

// Green's theorem over directed boundary edges: the integrals of 1, x, y, x^2
// and y^2 over the region the edges bound (holes subtract themselves).
struct Moments
{
    double a = 0.0, sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0;

    void AddEdge(Vec2 p, Vec2 q)
    {
        double px = p.x, py = p.y, qx = q.x, qy = q.y;
        double cr = px * qy - qx * py;
        a += cr / 2.0;
        sx += (px + qx) * cr / 6.0;
        sy += (py + qy) * cr / 6.0;
        sxx += (px * px + px * qx + qx * qx) * cr / 12.0;
        syy += (py * py + py * qy + qy * qy) * cr / 12.0;
    }
    void Add(const Moments &o)
    {
        a += o.a, sx += o.sx, sy += o.sy, sxx += o.sxx, syy += o.syy;
    }
    MassProps Props() const
    {
        MassProps m;
        if (std::fabs(a) < 1e-12)
            return m;
        double cx = sx / a, cy = sy / a;
        m.area    = float(a);
        m.com     = {float(cx), float(cy)};
        m.inertia = float(sxx + syy - a * (cx * cx + cy * cy));
        return m;
    }
};

bool SameProps(const MassProps &a, const MassProps &b)
{
    return a.area == b.area && a.com.x == b.com.x && a.com.y == b.com.y && a.inertia == b.inertia;
}

// Even-odd crossing test of p against a set of closed loops.
template <class EdgeFn>
bool Contains(Vec2 p, int count, const EdgeFn &edge)
{
    bool in = false;
    for (int k = 0; k < count; ++k) {
        auto [a, b] = edge(k);
        if ((a.y > p.y) != (b.y > p.y) && p.x < a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y))
            in = !in;
    }
    return in;
}

} // namespace

// ---------------------------------------------------------------------------
// Mass properties and kinematics
// ---------------------------------------------------------------------------
MassProps Combine(const MassProps *parts, int count)
{
    double a = 0.0, sx = 0.0, sy = 0.0;
    for (int k = 0; k < count; ++k) {
        a += parts[k].area;
        sx += double(parts[k].area) * parts[k].com.x;
        sy += double(parts[k].area) * parts[k].com.y;
    }
    MassProps m;
    if (a <= 0.0)
        return m;
    double cx = sx / a, cy = sy / a, inertia = 0.0;
    for (int k = 0; k < count; ++k) {
        double dx = parts[k].com.x - cx, dy = parts[k].com.y - cy;
        inertia += parts[k].inertia + parts[k].area * (dx * dx + dy * dy);
    }
    m.area    = float(a);
    m.com     = {float(cx), float(cy)};
    m.inertia = float(inertia);
    return m;
}

Kinematics InheritKinematics(const Kinematics &body, Vec2 bodyCom, Vec2 pieceCom)
{
    float rx = pieceCom.x - bodyCom.x, ry = pieceCom.y - bodyCom.y;
    return {{body.v.x - body.w * ry, body.v.y + body.w * rx}, body.w};
}

Kinematics MergeKinematics(const MassProps *parts, const Kinematics *kin, int count, const MassProps &merged)
{
    // The rigid motion of the parts taken together, about their combined
    // centre of mass, then moved to the merged body's centre of mass.
    MassProps whole = Combine(parts, count);
    if (whole.area <= 0.0f)
        return {};
    double px = 0.0, py = 0.0, l = 0.0;
    for (int k = 0; k < count; ++k) {
        double m = parts[k].area;
        px += m * kin[k].v.x;
        py += m * kin[k].v.y;
        double rx = parts[k].com.x - whole.com.x, ry = parts[k].com.y - whole.com.y;
        l += double(parts[k].inertia) * kin[k].w + m * (rx * kin[k].v.y - ry * kin[k].v.x);
    }
    Kinematics k;
    k.v = {float(px / whole.area), float(py / whole.area)};
    k.w = whole.inertia > 0.0f ? float(l / whole.inertia) : 0.0f;
    return InheritKinematics(k, whole.com, merged.com);
}

// ---------------------------------------------------------------------------
// Tracker
// ---------------------------------------------------------------------------
void ComponentTracker::Clear()
{
    mNextId = 0;
    mCellLabel.clear();
    mTargetLabel.clear();
    mComponents.clear();
    mRegions.clear();
    mEvents.clear();
}

const ComponentTracker::Component *ComponentTracker::Find(int id) const
{
    auto it = std::lower_bound(mComponents.begin(), mComponents.end(), id,
                               [](const Component &c, int v) { return c.id < v; });
    return it != mComponents.end() && it->id == id ? &*it : nullptr;
}

int ComponentTracker::Locate(const std::vector<Region> &regions, Vec2 p)
{
    for (const Region &r : regions) {
        if (p.x < r.lo.x || p.y < r.lo.y || p.x > r.hi.x || p.y > r.hi.y)
            continue;
        if (Contains(p, (int)r.edges.size(), [&](int k) { return std::pair(r.edges[k][0], r.edges[k][1]); }))
            return r.id;
    }
    return -1;
}

// Lock-free union-find. Parents only ever point to smaller slots, so a root is
// the smallest slot of its set whatever order the unions ran in. Find halves
// paths with a CAS that only shortcuts to an ancestor.
int ComponentTracker::Find(int x)
{
    for (;;) {
        int p = mParent[x].load(std::memory_order_acquire);
        if (p == x)
            return x;
        int g = mParent[p].load(std::memory_order_acquire);
        if (g != p)
            mParent[x].compare_exchange_weak(p, g, std::memory_order_acq_rel);
        x = g;
    }
}

void ComponentTracker::Union(int a, int b)
{
    for (;;) {
        a = Find(a);
        b = Find(b);
        if (a == b)
            return;
        if (a < b)
            std::swap(a, b);
        int expected = a; // link the larger root under the smaller, if it is still a root
        if (mParent[a].compare_exchange_strong(expected, b, std::memory_order_acq_rel))
            return;
    }
}

float ComponentTracker::Update(const VoroMesh2D &mesh, const std::vector<Vec2> &targets)
{
    auto      t0 = std::chrono::steady_clock::now();
    const int n  = (int)mesh.generators.size();
    mEvents.clear();
    mInheritedCells = mAttachedHoles = mUnattachedHoles = mOrphanTargets = 0;
    if ((int)mCellLabel.size() != n)
        mCellLabel.assign(n, -1);
    if (mParentSize != n) {
        mParent.reset(new std::atomic<int>[n]);
        mParentSize = n;
    }
    mRoot.resize(n);

    // ---- 1. Union-find over shell cells sharing an edge ---------------------
    std::vector<int> shell;
    for (int i = 0; i < n; ++i)
        if (mesh.shell[i]) {
            shell.push_back(i);
            mParent[i].store(i, std::memory_order_relaxed);
        }
    const int chunks = int((shell.size() + kCellsPerChunk - 1) / kCellsPerChunk);
    auto      chunk  = [&](int c, auto &&fn) {
        const size_t end = std::min(shell.size(), size_t(c + 1) * kCellsPerChunk);
        for (size_t k = size_t(c) * kCellsPerChunk; k < end; ++k)
            fn(shell[k]);
    };
    RunTasks(mExec, chunks, [&](int c) {
        chunk(c, [&](int i) {
            for (int k = mesh.cellStart[i]; k < mesh.cellStart[i + 1]; ++k) {
                int l = mesh.cellNeighbors[k];
                if (l >= 0 && l != i && mesh.shell[l])
                    Union(i, l);
            }
        });
    });
    RunTasks(mExec, chunks, [&](int c) { chunk(c, [&](int i) { mRoot[i] = Find(i); }); });
    mUnionFindMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();

    // ---- 2. Groups, their moments and loops ----------------------------------
    // Shell cells are visited in slot order and a root is its set's smallest
    // slot, so groups come out ordered by root.
    std::vector<int> groupOfRoot(n, -1), groupRoot;
    for (int i : shell)
        if (groupOfRoot[mRoot[i]] < 0) {
            groupOfRoot[mRoot[i]] = (int)groupRoot.size();
            groupRoot.push_back(mRoot[i]);
        }
    const int G      = (int)groupRoot.size();
    auto      groupOf = [&](int cell) { return groupOfRoot[mRoot[cell]]; };

    const int            L = mesh.watertight ? mesh.loops : 0;
    std::vector<Moments> groupMom(G), loopMom(L);
    std::vector<int>     loopGroup(L, -1), groupLoops(G, 0);
    std::vector<std::vector<int>> loopEdges(L);
    for (size_t e = 0; e < mesh.edges.size(); ++e) {
        Vec2 p = mesh.vertices[mesh.edges[e][0]], q = mesh.vertices[mesh.edges[e][1]];
        int  g = groupOf(mesh.edgeCell[e]);
        groupMom[g].AddEdge(p, q);
        if (int l = mesh.edgeLoop[e]; l >= 0 && l < L) {
            loopMom[l].AddEdge(p, q);
            loopGroup[l] = g;
            loopEdges[l].push_back((int)e);
        }
    }
    std::vector<int> outerLoop(G, -1);
    for (int l = 0; l < L; ++l) {
        int g = loopGroup[l];
        ++groupLoops[g];
        if (loopMom[l].a > 0.0 && (outerLoop[g] < 0 || loopMom[l].a > loopMom[outerLoop[g]].a))
            outerLoop[g] = l;
    }

    // ---- 3. Attach hole-only groups to the smallest outer loop around them --
    std::vector<int> parentGroup(G);
    for (int g = 0; g < G; ++g)
        parentGroup[g] = g;
    if (mesh.watertight) {
        for (int g = 0; g < G; ++g) {
            if (outerLoop[g] >= 0)
                continue;
            Vec2 p    = mesh.generators[groupRoot[g]]; // inside the material next to the hole
            int  best = -1;
            for (int h = 0; h < G; ++h) {
                int l = outerLoop[h];
                if (l < 0 || (best >= 0 && loopMom[l].a >= loopMom[outerLoop[best]].a))
                    continue;
                const std::vector<int> &le = loopEdges[l];
                if (Contains(p, (int)le.size(), [&](int k) {
                        auto [a, b] = mesh.edges[le[k]];
                        return std::pair(mesh.vertices[a], mesh.vertices[b]);
                    }))
                    best = h;
            }
            if (best >= 0) {
                parentGroup[g] = best;
                ++mAttachedHoles;
            } else {
                ++mUnattachedHoles;
            }
        }
    }

    // ---- 4. Components: outer groups with their holes ------------------------
    std::vector<int> compOfGroup(G, -1);
    int              K = 0;
    for (int g = 0; g < G; ++g)
        if (parentGroup[g] == g)
            compOfGroup[g] = K++;
    std::vector<Moments> compMom(K);
    std::vector<int>     compLoops(K, 0);
    for (int g = 0; g < G; ++g) {
        compOfGroup[g] = compOfGroup[parentGroup[g]];
        compMom[compOfGroup[g]].Add(groupMom[g]);
        compLoops[compOfGroup[g]] += groupLoops[g];
    }
    std::vector<MassProps> compProps(K);
    for (int k = 0; k < K; ++k)
        compProps[k] = compMom[k].Props();
    auto compOf = [&](int cell) { return compOfGroup[groupOf(cell)]; };

    // ---- 5. Old labels per component -----------------------------------------
    // A cell that was a shell cell keeps its label; one that just became a shell
    // cell takes the old component whose region contains its generator.
    std::vector<std::vector<int>> compOld(K);
    for (int i : shell) {
        int lab = mCellLabel[i];
        if (lab < 0) {
            lab = Locate(mRegions, mesh.generators[i]);
            mInheritedCells += lab >= 0;
        }
        if (lab >= 0)
            compOld[compOf(i)].push_back(lab);
    }
    std::unordered_map<int, int> prevIndex; // old ID -> index in mComponents
    for (int p = 0; p < (int)mComponents.size(); ++p)
        prevIndex[mComponents[p].id] = p;
    const int P = (int)mComponents.size();

    // Clusters of components and old IDs linked by a shared label (serial: a
    // handful of nodes). Nodes 0..K-1 are components, K..K+P-1 old IDs.
    std::vector<int> cl(K + P);
    for (int k = 0; k < K + P; ++k)
        cl[k] = k;
    auto clFind = [&](int x) {
        while (cl[x] != x)
            x = cl[x] = cl[cl[x]];
        return x;
    };
    for (int k = 0; k < K; ++k) {
        std::sort(compOld[k].begin(), compOld[k].end());
        compOld[k].erase(std::unique(compOld[k].begin(), compOld[k].end()), compOld[k].end());
        for (int id : compOld[k]) {
            auto it = prevIndex.find(id);
            if (it == prevIndex.end())
                continue;
            int a = clFind(k), b = clFind(K + it->second);
            if (a != b)
                cl[std::max(a, b)] = std::min(a, b);
        }
    }
    std::vector<std::vector<int>> clComps(K + P), clOld(K + P);
    for (int k = 0; k < K; ++k)
        clComps[clFind(k)].push_back(k);
    for (int p = 0; p < P; ++p)
        clOld[clFind(K + p)].push_back(p);

    // ---- 6. IDs and events ----------------------------------------------------
    std::vector<int> newId(K, -1);
    for (int c = 0; c < K + P; ++c) {
        const std::vector<int> &N = clComps[c], &O = clOld[c];
        if (N.empty())
            continue; // destroyed old IDs are reported below
        if (O.empty()) {
            for (int k : N) {
                newId[k] = mNextId++;
                mEvents.push_back({ComponentEvent::Type::Created, newId[k], {}, {}, {}});
            }
            continue;
        }
        // The largest old component survives a merge (ties: lowest ID, as O is
        // in ID order) ...
        int survivor = O[0];
        for (int p : O)
            if (mComponents[p].mass.area > mComponents[survivor].mass.area)
                survivor = p;
        const int sid    = mComponents[survivor].id;
        MassProps parent = mComponents[survivor].mass;
        if (O.size() > 1) {
            ComponentEvent ev{ComponentEvent::Type::Merged, sid, parent, {}, {}};
            std::vector<MassProps> all;
            for (int p : O) {
                all.push_back(mComponents[p].mass);
                if (p == survivor)
                    continue;
                ev.others.push_back(mComponents[p].id);
                ev.othersBefore.push_back(mComponents[p].mass);
            }
            mEvents.push_back(std::move(ev));
            parent = Combine(all.data(), (int)all.size());
        }
        // ... and the largest new piece keeps its ID.
        int largest = N[0];
        for (int k : N)
            if (compProps[k].area > compProps[largest].area)
                largest = k;
        newId[largest] = sid;
        if (N.size() > 1) {
            ComponentEvent ev{ComponentEvent::Type::Split, sid, parent, {}, {}};
            for (int k : N)
                if (k != largest) {
                    newId[k] = mNextId++;
                    ev.others.push_back(newId[k]);
                }
            mEvents.push_back(std::move(ev));
        } else if (O.size() == 1 && (!SameProps(compProps[largest], parent) ||
                                     compLoops[largest] != mComponents[survivor].loops)) {
            mEvents.push_back({ComponentEvent::Type::Changed, sid, parent, {}, {}});
        }
    }
    for (int c = 0; c < K + P; ++c)
        if (clComps[c].empty())
            for (int p : clOld[c])
                mEvents.push_back({ComponentEvent::Type::Destroyed, mComponents[p].id, mComponents[p].mass, {}, {}});

    // ---- 7. New state ---------------------------------------------------------
    mComponents.resize(K);
    for (int k = 0; k < K; ++k)
        mComponents[k] = {newId[k], compProps[k], compLoops[k]};
    std::sort(mComponents.begin(), mComponents.end(), [](const Component &a, const Component &b) { return a.id < b.id; });

    std::fill(mCellLabel.begin(), mCellLabel.end(), -1);
    for (int i : shell)
        mCellLabel[i] = newId[compOf(i)];

    mRegions.assign(K, Region{});
    for (int k = 0; k < K; ++k) {
        mRegions[k].id = newId[k];
        mRegions[k].lo = {FLT_MAX, FLT_MAX};
        mRegions[k].hi = {-FLT_MAX, -FLT_MAX};
    }
    for (size_t e = 0; e < mesh.edges.size(); ++e) {
        Region &r = mRegions[compOf(mesh.edgeCell[e])];
        Vec2    p = mesh.vertices[mesh.edges[e][0]], q = mesh.vertices[mesh.edges[e][1]];
        r.edges.push_back({p, q});
        r.lo = {std::min({r.lo.x, p.x, q.x}), std::min({r.lo.y, p.y, q.y})};
        r.hi = {std::max({r.hi.x, p.x, q.x}), std::max({r.hi.y, p.y, q.y})};
    }

    // Target points: their owner's label, or the region around a non-shell owner.
    mTargetLabel.assign(targets.size(), -1);
    for (size_t t = 0; t < targets.size(); ++t) {
        if (!(targets[t].x >= 0.0f))
            continue;
        int owner = t < mesh.targetOwner.size() ? mesh.targetOwner[t] : -1;
        int lab   = owner < 0 ? -1 : mCellLabel[owner] >= 0 ? mCellLabel[owner] : Locate(mesh.generators[owner]);
        mTargetLabel[t] = lab;
        mOrphanTargets += lab < 0;
    }
    return std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace voromesh
