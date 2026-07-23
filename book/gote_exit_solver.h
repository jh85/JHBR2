/*
  Retrograde solver for the Gote book-exit objective.

  Gote (WHITE) controls OR nodes and minimizes plies to an exit edge.
  Sente (BLACK) controls AND nodes and maximizes them. Nodes left unresolved
  are cycles from which Gote cannot force an exit.
*/

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace jhbr2 {

constexpr uint32_t kExitLeaf = std::numeric_limits<uint32_t>::max();
constexpr uint32_t kExitDisallowed =
    std::numeric_limits<uint32_t>::max() - 1;
constexpr uint32_t kExitDistanceInfinite =
    std::numeric_limits<uint32_t>::max();

struct GoteExitSolution {
  std::vector<uint32_t> distance;
  uint64_t allowed_internal_edges = 0;
  uint64_t finite_nodes = 0;
  uint64_t infinite_nodes = 0;
};

// Graph requirements:
//   uint64_t NodeCount() const
//   bool IsGote(uint32_t node) const
//   uint32_t EdgeCount(uint32_t node) const
//   uint32_t EdgeChild(uint32_t node, uint32_t edge) const
//
// EdgeChild returns kExitLeaf for an allowed move leaving the source book and
// kExitDisallowed for a move rejected by the evaluation margin.
template <typename Graph>
GoteExitSolution SolveGoteExitGraph(const Graph& graph) {
  const uint64_t node_count64 = graph.NodeCount();
  if (node_count64 >= kExitDisallowed) {
    throw std::runtime_error("Gote exit graph has too many nodes");
  }
  const uint32_t node_count = static_cast<uint32_t>(node_count64);

  std::vector<uint32_t> reverse_counts(node_count, 0);
  std::vector<uint32_t> unresolved(node_count, 0);
  std::vector<uint8_t> has_allowed_edge(node_count, 0);
  std::vector<uint8_t> has_exit_edge(node_count, 0);
  uint64_t internal_edges = 0;

  for (uint32_t node = 0; node < node_count; ++node) {
    const uint32_t edge_count = graph.EdgeCount(node);
    for (uint32_t edge = 0; edge < edge_count; ++edge) {
      const uint32_t child = graph.EdgeChild(node, edge);
      if (child == kExitDisallowed) continue;
      has_allowed_edge[node] = 1;
      if (child == kExitLeaf) {
        has_exit_edge[node] = 1;
        continue;
      }
      if (child >= node_count) {
        throw std::runtime_error("Gote exit graph child is out of range");
      }
      if (reverse_counts[child] == std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("Gote exit reverse degree overflow");
      }
      ++reverse_counts[child];
      ++internal_edges;
      if (!graph.IsGote(node)) {
        if (unresolved[node] == std::numeric_limits<uint32_t>::max()) {
          throw std::runtime_error("Gote exit out-degree overflow");
        }
        ++unresolved[node];
      }
    }
  }

  std::vector<uint64_t> reverse_offsets(uint64_t(node_count) + 1, 0);
  for (uint32_t node = 0; node < node_count; ++node) {
    reverse_offsets[node + 1] =
        reverse_offsets[node] + reverse_counts[node];
  }
  if (reverse_offsets.back() != internal_edges ||
      internal_edges > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("Gote exit reverse graph size overflow");
  }

  std::fill(reverse_counts.begin(), reverse_counts.end(), 0);
  std::vector<uint32_t> reverse_parents(static_cast<size_t>(internal_edges));
  for (uint32_t parent = 0; parent < node_count; ++parent) {
    const uint32_t edge_count = graph.EdgeCount(parent);
    for (uint32_t edge = 0; edge < edge_count; ++edge) {
      const uint32_t child = graph.EdgeChild(parent, edge);
      if (child >= kExitDisallowed) continue;
      const uint64_t position =
          reverse_offsets[child] + reverse_counts[child]++;
      reverse_parents[static_cast<size_t>(position)] = parent;
    }
  }

  GoteExitSolution solution;
  solution.allowed_internal_edges = internal_edges;
  solution.distance.assign(node_count, kExitDistanceInfinite);
  std::vector<uint32_t> max_child_distance(node_count, 0);
  std::vector<uint32_t> queue;
  queue.reserve(node_count);

  for (uint32_t node = 0; node < node_count; ++node) {
    bool immediately_finite = false;
    if (graph.IsGote(node)) {
      immediately_finite = has_exit_edge[node] != 0;
    } else {
      // If Sente has no in-book successor, every known move exits. An empty
      // move list is not treated as an exit because it cannot yield a policy.
      immediately_finite =
          has_allowed_edge[node] != 0 && unresolved[node] == 0;
    }
    if (immediately_finite) {
      solution.distance[node] = 1;
      queue.push_back(node);
    }
  }

  size_t head = 0;
  while (head < queue.size()) {
    const uint32_t child = queue[head++];
    const uint32_t child_distance = solution.distance[child];
    for (uint64_t i = reverse_offsets[child];
         i < reverse_offsets[child + 1]; ++i) {
      const uint32_t parent = reverse_parents[static_cast<size_t>(i)];
      if (solution.distance[parent] != kExitDistanceInfinite) continue;

      if (graph.IsGote(parent)) {
        // Queue order is nondecreasing in distance, so the first finite child
        // is the shortest one.
        solution.distance[parent] =
            child_distance == kExitDistanceInfinite
                ? kExitDistanceInfinite
                : child_distance + 1;
        queue.push_back(parent);
      } else {
        max_child_distance[parent] =
            std::max(max_child_distance[parent], child_distance);
        if (unresolved[parent] == 0) {
          throw std::runtime_error("Gote exit unresolved underflow");
        }
        --unresolved[parent];
        if (unresolved[parent] == 0) {
          solution.distance[parent] = max_child_distance[parent] + 1;
          queue.push_back(parent);
        }
      }
    }
  }

  solution.finite_nodes = queue.size();
  solution.infinite_nodes = node_count64 - solution.finite_nodes;
  return solution;
}

}  // namespace jhbr2
