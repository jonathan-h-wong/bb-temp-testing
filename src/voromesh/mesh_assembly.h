#pragma once

// ---------------------------------------------------------------------------
// mesh_assembly.h (internal) – shared by the GPU and CPU optimisers.
//
// Turns per-slot cell polygons into VoroMesh2D's CSR cells and the boundary
// mesh. Edge labels: >= 0 is the neighbouring generator slot, -1..-4 the
// domain box sides. `mesh.generators` and `mesh.inside` must already be set.
// `poly(i, count, verts, labels)` fills in slot i's polygon.
// ---------------------------------------------------------------------------
#include "voromesh/voro_optimizer.h"

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace voromesh::detail {

constexpr int kBoxLabels = 4;

template <class PolyFn>
void AssembleMesh(VoroMesh2D &mesh, int n, PolyFn poly)
{
    // Compact the polygons into CSR.
    mesh.cellStart.assign(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        int c; const Vec2 *v; const int *e;
        poly(i, c, v, e);
        mesh.cellStart[i + 1] = mesh.cellStart[i] + c;
    }
    mesh.cellVerts.resize(mesh.cellStart[n]);
    for (int i = 0; i < n; ++i) {
        int c; const Vec2 *v; const int *e;
        poly(i, c, v, e);
        std::copy_n(v, c, mesh.cellVerts.begin() + mesh.cellStart[i]);
    }

    // Boundary edges: every edge of an inside cell whose other side is an
    // outside cell or the domain box. A cell vertex sits between edges with
    // labels a and b, so it is equidistant to generators {i, a, b}.
    auto key = [](int a, int b, int c) {
        uint64_t v0 = uint64_t(a + kBoxLabels), v1 = uint64_t(b + kBoxLabels), v2 = uint64_t(c + kBoxLabels);
        if (v0 > v1) std::swap(v0, v1);
        if (v1 > v2) std::swap(v1, v2);
        if (v0 > v1) std::swap(v0, v1);
        return (v0 << 42) | (v1 << 21) | v2;
    };
    std::unordered_map<uint64_t, int> vertexOf;
    auto vertex = [&](uint64_t k, Vec2 p) {
        auto [it, inserted] = vertexOf.try_emplace(k, (int)mesh.vertices.size());
        if (inserted) mesh.vertices.push_back(p);
        return it->second;
    };
    for (int i = 0; i < n; ++i) {
        if (!mesh.inside[i]) continue;
        int c; const Vec2 *pos; const int *lab;
        poly(i, c, pos, lab);
        for (int k = 0; k < c; ++k) {
            int l = lab[k];
            if (l >= 0 && mesh.inside[l]) continue;
            int prev = lab[(k + c - 1) % c], next = lab[(k + 1) % c];
            int a = vertex(key(i, prev, l), pos[k]);
            int b = vertex(key(i, l, next), pos[(k + 1) % c]);
            mesh.edges.push_back({a, b});
        }
    }

    // Watertight iff each vertex has exactly one outgoing and one incoming edge.
    std::vector<int> next(mesh.vertices.size(), -1), inDeg(mesh.vertices.size(), 0);
    mesh.watertight = true;
    for (auto [a, b] : mesh.edges) {
        if (next[a] != -1) mesh.watertight = false;
        next[a] = b;
        ++inDeg[b];
    }
    for (size_t v = 0; v < next.size(); ++v)
        if (next[v] == -1 || inDeg[v] != 1) mesh.watertight = false;
    if (mesh.watertight) {
        std::vector<uint8_t> seen(next.size(), 0);
        for (size_t v = 0; v < next.size(); ++v) {
            if (seen[v]) continue;
            ++mesh.loops;
            for (int w = (int)v; !seen[w]; w = next[w]) seen[w] = 1;
        }
    }
}

} // namespace voromesh::detail
