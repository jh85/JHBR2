#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "book/gote_exit_solver.h"

namespace {

struct Graph {
  struct Node {
    bool gote;
    std::vector<uint32_t> children;
  };
  std::vector<Node> nodes;

  uint64_t NodeCount() const { return nodes.size(); }
  bool IsGote(uint32_t node) const { return nodes[node].gote; }
  uint32_t EdgeCount(uint32_t node) const {
    return nodes[node].children.size();
  }
  uint32_t EdgeChild(uint32_t node, uint32_t edge) const {
    return nodes[node].children[edge];
  }
};

int failures = 0;

void Check(const std::string& name, bool condition) {
  if (condition) {
    std::cout << "  OK    " << name << '\n';
  } else {
    std::cout << "  FAIL  " << name << '\n';
    ++failures;
  }
}

}  // namespace

int main() {
  using namespace jhbr2;

  {
    // Gote can exit now or enter a longer line.
    Graph graph{{{true, {kExitLeaf, 1}},
                 {false, {kExitLeaf}}}};
    auto result = SolveGoteExitGraph(graph);
    Check("Gote chooses immediate exit", result.distance[0] == 1);
  }

  {
    // Sente can exit immediately or prolong through node 1.
    Graph graph{{{false, {kExitLeaf, 1}},
                 {true, {kExitLeaf}}}};
    auto result = SolveGoteExitGraph(graph);
    Check("Sente maximizes exit distance", result.distance[0] == 2);
    Check("child distance is one", result.distance[1] == 1);
  }

  {
    // Sente can maintain a self-cycle, so Gote cannot force an exit.
    Graph graph{{{false, {kExitLeaf, 0}}}};
    auto result = SolveGoteExitGraph(graph);
    Check("Sente-maintained cycle is infinite",
          result.distance[0] == kExitDistanceInfinite);
  }

  {
    // Gote can leave a cycle even though another edge loops.
    Graph graph{{{true, {0, kExitLeaf}}}};
    auto result = SolveGoteExitGraph(graph);
    Check("Gote escapes cycle", result.distance[0] == 1);
  }

  {
    // The only exit was rejected by the evaluation margin.
    Graph graph{{{true, {kExitDisallowed, 0}}}};
    auto result = SolveGoteExitGraph(graph);
    Check("disallowed exit does not resolve cycle",
          result.distance[0] == kExitDistanceInfinite);
  }

  std::cout << "\n=== Summary: " << failures << " failed ===\n";
  return failures == 0 ? 0 : 1;
}
