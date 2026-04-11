/*
  JHBR2 Shogi Engine — df-pn Checkmate Solver

  Best-first proof-number search for tsume (checkmate) problems.
  Ported from YaneuraOu's mate_dfpn.hpp, adapted to JHBR2's ShogiBoard.

  The algorithm:
    - OR nodes (attacker): tries to prove a forced checkmate exists.
      Generates checking moves. pn = min(children.pn), dn = sum(children.dn).
    - AND nodes (defender): tries to disprove checkmate.
      Generates evasion moves. pn = sum(children.pn), dn = min(children.dn).
    - A position is proved mate when pn = 0 (at root OR node).
    - A position is proved no-mate when dn = 0 (at root OR node).

  Memory model: linear allocation from a pre-allocated buffer, no GC.
  No hash table (following YaneuraOu's modern approach).

  References:
    - YaneuraOu/source/mate/mate_dfpn.hpp
    - YaneuraOu/source/mate/mate_move_picker.h
*/

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "shogi/board.h"
#include "shogi/types.h"

namespace jhbr2 {

using lczero::ShogiBoard;
using lczero::Move;
using lczero::MoveList;
using lczero::Color;
using lczero::Square;

// =====================================================================
// df-pn Node
// =====================================================================

struct DfpnNode {
  static constexpr uint32_t INF  = 0xFFFFFFF0u;
  static constexpr uint32_t MATE = INF - 2000;
  static constexpr uint8_t NOT_EXPANDED = 255;

  DfpnNode* children = nullptr;
  uint32_t pn = 1;
  uint32_t dn = 1;
  Move last_move;
  uint8_t child_num = NOT_EXPANDED;
  bool repeated = false;

  bool is_expanded() const { return child_num != NOT_EXPANDED; }

  // OR node: pn=0 means mate proven. dn=0 means no mate.
  // AND node: pn=0 means no mate. dn=0 means mate proven.
  bool is_mate_or() const { return pn == 0 && dn >= MATE; }
  bool is_nomate_or() const { return dn == 0 && pn >= MATE; }

  template<bool or_node>
  void set_mate(int ply = 0) {
    if constexpr (or_node) {
      pn = 0;
      dn = INF - ply;
    } else {
      pn = INF - ply;
      dn = 0;
    }
  }

  template<bool or_node>
  void set_mated(int ply = 0) {
    if constexpr (or_node) {
      pn = INF - ply;
      dn = 0;
    } else {
      pn = 0;
      dn = INF - ply;
    }
  }

  void set_nomate(int ply = 0) {
    pn = INF - ply;
    dn = 0;
  }
};

// =====================================================================
// Node allocator (linear, no GC)
// =====================================================================

class DfpnNodePool {
 public:
  DfpnNodePool() = default;

  void Alloc(size_t num_nodes) {
    pool_ = std::make_unique<DfpnNode[]>(num_nodes);
    capacity_ = num_nodes;
    used_ = 0;
  }

  DfpnNode* NewNodes(size_t count) {
    if (used_ + count > capacity_) return nullptr;
    DfpnNode* ptr = &pool_[used_];
    used_ += count;
    return ptr;
  }

  void Reset() { used_ = 0; }
  bool OutOfMemory() const { return used_ >= capacity_; }
  size_t Used() const { return used_; }
  size_t Capacity() const { return capacity_; }

 private:
  std::unique_ptr<DfpnNode[]> pool_;
  size_t capacity_ = 0;
  size_t used_ = 0;
};

// =====================================================================
// MateDfpnSolver
// =====================================================================

class MateDfpnSolver {
 public:
  explicit MateDfpnSolver(size_t default_nodes_limit = 100000);

  // Search for a forced checkmate from the given position.
  //
  // Returns:
  //   Valid Move = mate found (the first mating move)
  //   Move() with raw == 0 = unsolved (out of nodes/memory)
  //   kNoMateMove = proved that no forced mate exists
  //
  // The side_to_move of the board is the attacker (OR node).
  Move search(ShogiBoard board, size_t nodes_limit);
  Move search(ShogiBoard board) {
    return search(board, default_nodes_limit_);
  }

  // Special "no mate" sentinel move.
  static Move NoMateMove() {
    // Use a move value that can't be a real move.
    // We'll use from=127, to=127 (both out of range).
    return Move::Normal(lczero::Square::FromIdx(127),
                        lczero::Square::FromIdx(127));
  }

  static bool IsNoMate(Move m) {
    return m.raw() == NoMateMove().raw();
  }

  // After a successful search:
  int get_mate_ply() const { return mate_ply_; }
  std::vector<Move> get_pv() const { return pv_; }
  size_t get_nodes_searched() const { return nodes_searched_; }

  void stop() { stop_ = true; }

 private:
  // Core recursive search.
  // or_node: true = attacker (generate checks), false = defender (generate evasions).
  // second_pn/second_dn: threshold from parent (stop if pn/dn exceeds 2nd-best sibling).
  template<bool or_node>
  void Search(ShogiBoard& board, DfpnNode& node,
              uint32_t second_pn, uint32_t second_dn, int ply);

  // Expand a node: generate moves and create children.
  template<bool or_node>
  void ExpandNode(ShogiBoard& board, DfpnNode& node, int ply);

  // Update pn/dn from children.
  template<bool or_node>
  void SummarizeNode(DfpnNode& node);

  // Select the best child and compute 2nd-best thresholds.
  template<bool or_node>
  DfpnNode* SelectBestChild(DfpnNode& node,
                             uint32_t& second_pn, uint32_t& second_dn);

  // Generate checking moves (for OR node / attacker).
  MoveList GenerateChecks(ShogiBoard& board);

  // 1-ply mate check (fast shortcut).
  Move Mate1Ply(ShogiBoard& board);

  // Extract PV from the solved tree.
  void ExtractPV(DfpnNode& root);

  // --- Members ---
  size_t default_nodes_limit_;
  size_t nodes_limit_ = 0;
  DfpnNodePool pool_;
  size_t nodes_searched_ = 0;
  int mate_ply_ = 0;
  std::vector<Move> pv_;
  bool stop_ = false;

  // Repetition detection: track hashes on the current search path.
  std::vector<uint64_t> path_hashes_;
  bool IsRepetition(uint64_t hash) const;
};

}  // namespace jhbr2
