#pragma once

#include <vector>

#include "dlshogi_mcts/uct_node.h"

namespace dlshogi_mcts {

// One selected edge in a root-to-leaf MCTS playout.
struct trajectory_t {
  uct_node_t* parent = nullptr;
  unsigned child_idx = 0;
};

// Parameters needed by dlshogi's PUCT/FPU selection rule. Root and non-root
// constants are selected by the caller before invoking SelectPuctChild().
struct PuctParameters {
  float c_init = 1.25f;
  float c_base = 19652.0f;
  float fpu_reduction = 0.0f;
  float moves_left_weight = 0.0f;
  float moves_left_threshold = 0.0f;
  float moves_left_cap = 20.0f;
};

// Outcome from the perspective of the player who traversed an edge. The
// ChildNode flag names follow dlshogi's opponent-result convention, so an edge
// win is represented by ChildNode::SetLose() and vice versa.
enum class EdgeOutcome { kWin, kLoss, kDraw };

// Mark a proven terminal edge and return its value from the edge mover's
// perspective. draw_value must also be from that player's perspective.
float ResolveTerminalEdge(child_node_t* edge, EdgeOutcome outcome,
                          float draw_value = 0.5f);

// Return a previously proven edge value. The caller supplies the draw value
// from the edge mover's perspective.
bool TryGetProvenEdgeValue(const child_node_t* edge, float draw_value,
                           float* value);

// Select a child using dlshogi's PUCT/FPU rule. This also propagates proven
// win/loss/draw states to parent and accumulates the selected prior for FPU.
unsigned SelectPuctChild(child_node_t* parent, uct_node_t* current,
                         const PuctParameters& params);

// Back up a value that is already expressed from the perspective of the
// player who traversed the final edge. Perspective is alternated exactly once
// per edge while walking toward the root.
// Returns false without updating the tree if the value, moves-left estimate,
// or an existing accumulator is non-finite.
bool BackupTrajectory(const std::vector<trajectory_t>& trajectory,
                      float leaf_parent_value, float leaf_moves_left = 0.0f);

}  // namespace dlshogi_mcts
