/*
  JHBR2 Shogi Engine — df-pn Checkmate Solver Implementation

  Best-first proof-number search ported from YaneuraOu's mate_dfpn.hpp.
  Adapted to use JHBR2's ShogiBoard instead of YaneuraOu's Position.

  References:
    - YaneuraOu/source/mate/mate_dfpn.hpp (ParallelSearch, ExpandNode, SummarizeNode)
    - YaneuraOu/source/mate/mate_move_picker.h (check/evasion generation)
*/

#include "mate/dfpn.h"

#include <algorithm>
#include <cassert>
#include <utility>

namespace jhbr2 {

using namespace lczero;

// =====================================================================
// Helpers
// =====================================================================

// Saturating add for proof/disproof numbers.
static uint32_t SatAdd(uint32_t a, uint32_t b) {
  uint64_t sum = static_cast<uint64_t>(a) + b;
  return sum >= DfpnNode::INF ? DfpnNode::INF : static_cast<uint32_t>(sum);
}

// =====================================================================
// Constructor
// =====================================================================

MateDfpnSolver::MateDfpnSolver(size_t default_nodes_limit)
    : default_nodes_limit_(default_nodes_limit) {}

// =====================================================================
// Main search entry point
// =====================================================================

Move MateDfpnSolver::search(ShogiBoard board, size_t nodes_limit) {
  return search(std::move(board), nodes_limit, Deadline::max());
}

Move MateDfpnSolver::search(ShogiBoard board, size_t nodes_limit,
                            Deadline deadline) {
  deadline_ = deadline;
  nodes_searched_ = 0;
  nodes_limit_ = nodes_limit;
  mate_ply_ = 0;
  pv_.clear();
  path_hashes_.clear();

  // Do not clear stop_ here. The caller can launch search() on a worker and
  // legitimately request cancellation before that worker gets scheduled.
  // Clearing it here loses the request and makes a subsequent join unbounded.
  if (ShouldStop()) return Move();

  // An expansion can allocate many move children. Small root budgets need
  // more than four child slots per expanded node on tactical positions.
  // Cap the pool at 2M nodes to bound allocation time and memory use.
  constexpr size_t kMaxPoolNodes = 2000000;
  const size_t scaled_pool =
      nodes_limit >= kMaxPoolNodes / 8 ? kMaxPoolNodes : nodes_limit * 8;
  const size_t pool_size =
      std::clamp(scaled_pool, size_t{1024}, kMaxPoolNodes);
  pool_.Alloc(pool_size);

  // Create root node (OR node — attacker).
  DfpnNode root;
  root.pn = 1;
  root.dn = 1;

  ShogiBoard& pos = board;

  // Push initial hash for repetition detection.
  path_hashes_.push_back(pos.Hash());

  // Search iteratively until solved or out of resources.
  while (root.pn != 0 && root.dn != 0 &&
         !pool_.OutOfMemory() &&
         !ShouldStop() &&
         nodes_searched_ < nodes_limit) {
    Search<true>(pos, root, DfpnNode::INF, DfpnNode::INF, 0);
  }

  path_hashes_.pop_back();

  // A stopped or expired search is always unresolved. In particular, do not
  // return a proof assembled from partially expanded nodes.
  if (ShouldStop()) return Move();

  // Interpret result.
  if (root.pn == 0) {
    // Mate proven.
    ExtractPV(root);
    if (ShouldStop()) {
      pv_.clear();
      mate_ply_ = 0;
      return Move();
    }
    if (!pv_.empty()) {
      return pv_[0];
    }
    return Move();  // Shouldn't happen, but fallback.
  } else if (root.dn == 0) {
    // No mate proven.
    return NoMateMove();
  } else {
    // Unsolved (out of nodes/memory or stopped).
    return Move();
  }
}

bool MateDfpnSolver::ShouldStop() {
  if (stop_.load(std::memory_order_acquire)) return true;
  if (deadline_ != Deadline::max() && Clock::now() >= deadline_) return true;
  return false;
}

// =====================================================================
// Core recursive search
// =====================================================================
// Reference: YaneuraOu mate_dfpn.hpp ParallelSearch

template<bool or_node>
void MateDfpnSolver::Search(ShogiBoard& board, DfpnNode& node,
                             uint32_t second_pn, uint32_t second_dn,
                             int ply) {
  // Early exit if stopped.
  if (ShouldStop()) return;

  // Expand if not yet expanded.
  if (!node.is_expanded()) {
    ExpandNode<or_node>(board, node, ply);
    if (ShouldStop()) return;
    nodes_searched_++;
    SummarizeNode<or_node>(node);
    return;
  }

  // Loop: search the best child until threshold exceeded or solved.
  while (node.pn < second_pn &&
         node.dn < second_dn &&
         node.pn != 0 && node.dn != 0 &&
         !pool_.OutOfMemory() &&
         !ShouldStop() &&
         nodes_searched_ < nodes_limit_) {

    // Find best child and 2nd-best thresholds.
    uint32_t child_second_pn = second_pn;
    uint32_t child_second_dn = second_dn;
    DfpnNode* best = SelectBestChild<or_node>(node, child_second_pn, child_second_dn);

    if (!best) break;

    // Apply move.
    UndoInfo undo = board.DoMove(best->last_move);

    // Rule-aware repetition check over both the game history inherited at the
    // root and the moves made by this df-pn search. CheckRepetition() reports
    // from the child side-to-move's perspective. The child is an AND
    // (defender) node after an OR move, and an OR (attacker) node after an AND
    // move, so the same kWin/kLoss result has opposite proof meaning.
    const auto repetition =
        board.CheckRepetition(kRepetitionLookbackPly);
    if (repetition != ShogiBoard::RepetitionResult::kNone) {
      bool attacker_wins = false;
      if constexpr (or_node) {
        // The child is the defender: their repetition loss proves the
        // attacker's win; their win or a draw disproves mate.
        attacker_wins =
            repetition == ShogiBoard::RepetitionResult::kLoss;
      } else {
        // The child is the attacker: their repetition win proves a win; their
        // loss or a draw disproves mate.
        attacker_wins =
            repetition == ShogiBoard::RepetitionResult::kWin;
      }

      if (attacker_wins) {
        best->set_mate(ply + 1);
      } else {
        best->set_nomate(ply + 1);
      }
      best->repeated = true;
      board.UndoMove(best->last_move, undo);
      SummarizeNode<or_node>(node);
      continue;
    }

    // Retain the path-only hash guard for cycles beyond the bounded
    // rule-aware window. In a checking/evasion df-pn tree, such a cycle cannot
    // establish a checkmate, so the conservative result is no-mate.
    uint64_t hash = board.Hash();
    if (IsRepetition(hash)) {
      best->set_nomate(ply + 1);
      best->repeated = true;
      board.UndoMove(best->last_move, undo);
      SummarizeNode<or_node>(node);
      continue;
    }

    path_hashes_.push_back(hash);

    // Recurse with alternating OR/AND.
    Search<!or_node>(board, *best, child_second_pn, child_second_dn, ply + 1);

    path_hashes_.pop_back();
    board.UndoMove(best->last_move, undo);

    // Once cancellation is observed, only unwind board/path state. Avoid
    // repeatedly summarizing every ancestor on a potentially deep path.
    if (ShouldStop()) return;

    // Update node's pn/dn from children.
    SummarizeNode<or_node>(node);
  }
}

// =====================================================================
// Expand node: generate moves and create children
// =====================================================================
// Reference: YaneuraOu mate_dfpn.hpp ExpandNode

template<bool or_node>
void MateDfpnSolver::ExpandNode(ShogiBoard& board, DfpnNode& node, int ply) {
  if (ShouldStop()) return;

  // Quick 1-ply mate check for OR nodes.
  if constexpr (or_node) {
    Move mate1 = Mate1Ply(board);
    if (ShouldStop()) return;
    if (!mate1.is_null()) {
      // Found a 1-ply mate, including a mating countercheck when the
      // attacker starts in check.
      node.set_mate(ply);
      node.mate_distance = 1;
      node.child_num = 1;

      DfpnNode* children = pool_.NewNodes(1);
      if (children) {
        node.children = children;
        children[0].last_move = mate1;
        children[0].set_mate(ply + 1);
        children[0].child_num = 0;
      }
      return;
    }
  }

  // Generate moves.
  MoveList moves;
  if constexpr (or_node) {
    // Attacker: generate checking moves.
    moves = GenerateChecks(board);
  } else {
    // Defender: generate all legal evasions (we must be in check).
    moves = board.GenerateLegalMoves();
  }
  if (ShouldStop()) return;

  if (moves.empty()) {
    if constexpr (or_node) {
      // No checking moves = cannot deliver check = no mate from here.
      node.set_nomate(ply);
    } else {
      // No evasion moves = checkmate! (Defender has no moves while in check.)
      node.set_mate(ply);
    }
    node.child_num = 0;
    return;
  }

  // Allocate children.
  DfpnNode* children = pool_.NewNodes(moves.size());
  if (!children) {
    // Out of memory — mark as unsolved.
    return;
  }

  node.children = children;
  node.child_num = static_cast<uint8_t>(std::min((int)moves.size(), 254));

  for (size_t i = 0; i < moves.size() && i < 254; i++) {
    children[i].last_move = moves[i];
    children[i].pn = 1;
    children[i].dn = 1;
    children[i].children = nullptr;
    children[i].child_num = DfpnNode::NOT_EXPANDED;
    children[i].repeated = false;
    children[i].mate_distance = 0;
  }

  // Summarize from children.
  SummarizeNode<or_node>(node);
}

// =====================================================================
// Summarize: update pn/dn from children
// =====================================================================
// OR node:  pn = min(children.pn), dn = sum(children.dn)
// AND node: pn = sum(children.pn), dn = min(children.dn)

template<bool or_node>
void MateDfpnSolver::SummarizeNode(DfpnNode& node) {
  if (node.child_num == 0 || node.child_num == DfpnNode::NOT_EXPANDED) return;

  uint32_t min_val = DfpnNode::INF;
  uint32_t sum_val = 0;
  bool any_repeated = false;

  for (int i = 0; i < node.child_num; i++) {
    DfpnNode& c = node.children[i];

    if constexpr (or_node) {
      // OR: pn = min(children.pn), dn = sum(children.dn)
      min_val = std::min(min_val, c.pn);
      sum_val = SatAdd(sum_val, c.dn);
    } else {
      // AND: pn = sum(children.pn), dn = min(children.dn)
      sum_val = SatAdd(sum_val, c.pn);
      min_val = std::min(min_val, c.dn);
    }
    if (c.repeated) any_repeated = true;
  }

  if constexpr (or_node) {
    node.pn = min_val;
    node.dn = sum_val;
  } else {
    node.pn = sum_val;
    node.dn = min_val;
  }

  node.repeated = any_repeated;
  if (node.pn == 0) {
    uint16_t distance = or_node ? UINT16_MAX : 0;
    for (int i = 0; i < node.child_num; ++i) {
      const DfpnNode& child = node.children[i];
      if (child.pn != 0) continue;
      if constexpr (or_node) {
        distance = std::min(distance, child.mate_distance);
      } else {
        distance = std::max(distance, child.mate_distance);
      }
    }
    node.mate_distance =
        distance == UINT16_MAX ? 0 : static_cast<uint16_t>(distance + 1);
  }
}

// =====================================================================
// Select best child with 2nd-best thresholds
// =====================================================================
// Reference: YaneuraOu mate_dfpn.hpp select_the_best_child

template<bool or_node>
DfpnNode* MateDfpnSolver::SelectBestChild(DfpnNode& node,
                                            uint32_t& second_pn,
                                            uint32_t& second_dn) {
  if (node.child_num == 0) return nullptr;

  int best_idx = -1;
  uint32_t best_val = DfpnNode::INF;
  uint32_t second_best_val = DfpnNode::INF;

  // For OR node: select child with minimum pn (easiest to prove mate).
  // For AND node: select child with minimum dn (hardest to disprove mate).
  for (int i = 0; i < node.child_num; i++) {
    uint32_t val;
    if constexpr (or_node) {
      val = node.children[i].pn;
    } else {
      val = node.children[i].dn;
    }

    if (val < best_val) {
      second_best_val = best_val;
      best_val = val;
      best_idx = i;
    } else if (val < second_best_val) {
      second_best_val = val;
    }
  }

  if (best_idx < 0) return nullptr;

  DfpnNode& best = node.children[best_idx];

  // Compute thresholds for the child.
  // The child should stop searching if its pn exceeds the 2nd-best sibling's pn
  // (for OR node), because then we'd switch to searching that sibling instead.
  if constexpr (or_node) {
    // Child threshold: min(second_pn_from_parent, second_best_sibling_pn + 1)
    second_pn = std::min(second_pn, SatAdd(second_best_val, 1));
    // dn threshold: second_dn from parent minus sum of other children's dn, plus this child's dn
    // Simplified: just pass second_dn as-is (conservative but correct).
    // More precise: second_dn = min(second_dn, node.dn - best.dn + best.dn_threshold)
    // For simplicity, we use the parent's threshold directly.
    // The child's dn threshold = second_dn (the dn budget from parent).
  } else {
    // AND node: select child with min dn.
    second_dn = std::min(second_dn, SatAdd(second_best_val, 1));
  }

  return &best;
}

// =====================================================================
// Generate checking moves (for OR node)
// =====================================================================

MoveList MateDfpnSolver::GenerateChecks(ShogiBoard& board) {
  // Generate all legal moves, then filter for those that give check.
  MoveList all_moves = board.GenerateLegalMoves();
  MoveList checks;

  for (const Move& m : all_moves) {
    if (ShouldStop()) return checks;
    UndoInfo undo = board.DoMove(m);
    if (board.InCheck()) {
      checks.push_back(m);
    }
    board.UndoMove(m, undo);
  }

  return checks;
}

// =====================================================================
// 1-ply mate check
// =====================================================================

Move MateDfpnSolver::Mate1Ply(ShogiBoard& board) {
  return board.FindMateInOne();
}

// =====================================================================
// Repetition detection
// =====================================================================

bool MateDfpnSolver::IsRepetition(uint64_t hash) const {
  for (auto h : path_hashes_) {
    if (h == hash) return true;
  }
  return false;
}

// =====================================================================
// Extract PV from solved tree
// =====================================================================

void MateDfpnSolver::ExtractPV(DfpnNode& root) {
  pv_.clear();
  mate_ply_ = 0;

  DfpnNode* node = &root;
  bool or_node = true;

  while (node->child_num > 0 && node->children) {
    // The attacker chooses the shortest proven line; the defender chooses
    // the longest. mate_distance is maintained when proof numbers are
    // summarized, so this produces the principal mate line without walking
    // the full solved tree again.
    DfpnNode* best = nullptr;
    uint16_t best_distance = or_node ? UINT16_MAX : 0;
    for (int i = 0; i < node->child_num; i++) {
      DfpnNode& child = node->children[i];
      if (child.pn != 0) continue;
      const bool better = or_node ? child.mate_distance < best_distance
                                  : !best || child.mate_distance > best_distance;
      if (!better) continue;
      best = &child;
      best_distance = child.mate_distance;
    }

    if (!best) break;

    pv_.push_back(best->last_move);
    mate_ply_++;
    node = best;
    or_node = !or_node;
  }
}

// Explicit template instantiations.
template void MateDfpnSolver::Search<true>(ShogiBoard&, DfpnNode&, uint32_t, uint32_t, int);
template void MateDfpnSolver::Search<false>(ShogiBoard&, DfpnNode&, uint32_t, uint32_t, int);
template void MateDfpnSolver::ExpandNode<true>(ShogiBoard&, DfpnNode&, int);
template void MateDfpnSolver::ExpandNode<false>(ShogiBoard&, DfpnNode&, int);
template void MateDfpnSolver::SummarizeNode<true>(DfpnNode&);
template void MateDfpnSolver::SummarizeNode<false>(DfpnNode&);
template DfpnNode* MateDfpnSolver::SelectBestChild<true>(DfpnNode&, uint32_t&, uint32_t&);
template DfpnNode* MateDfpnSolver::SelectBestChild<false>(DfpnNode&, uint32_t&, uint32_t&);

}  // namespace jhbr2
