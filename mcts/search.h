/*
  JHBR2 Shogi Engine — MCTS Search

  PUCT-based Monte Carlo Tree Search with:
    - Dynamic cpuct (lc0-style)
    - FPU (First Play Urgency) with relative reduction
    - Dirichlet noise at root
    - df-pn mate detection at leaf nodes
    - Mate-aware UCB selection (dlshogi-style)
    - PV mate search after main loop

  References:
    - lc0 src/mcts/search.cc
    - dlshogi UctSearch.cpp, PvMateSearch.cpp
    - JHBR2/shogi_mcts.py (Python prototype)
*/

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mcts/node.h"
#ifdef USE_TENSORRT
#include "mcts/nn_tensorrt.h"
#else
#include "mcts/nn_eval.h"
#endif
#include "mcts/search_mt.h"
#include "shogi/board.h"

namespace jhbr2 {

// Forward declaration.
class MateDfpnSolver;

// =====================================================================
// Search result
// =====================================================================

struct SearchResult {
  Move best_move;

  int nodes = 0;
  float time_sec = 0.0f;
  float nps = 0.0f;
  float root_q = 0.0f;
  float root_d = 0.0f;
  int score_cp = 0;
  int8_t mate_status = 0;  // 0=unknown, 1=win, -1=loss

  std::vector<Move> pv;

  // Top children info: (move, visits, Q, policy)
  struct ChildInfo {
    Move move;
    int n;
    float q;
    float p;
  };
  std::vector<ChildInfo> top_children;
};

// =====================================================================
// MCTSSearch — main search class
// =====================================================================

class MCTSSearch {
 public:
  MCTSSearch(NNEvaluator& evaluator, const MCTSConfig& config);
  ~MCTSSearch();

  // Run search from the given board position.
  // game_ply: current game move number (for temperature scheduling).
  SearchResult Search(ShogiBoard board, int game_ply = 1);

  // Set stop flag (can be called from another thread).
  void Stop() { stop_ = true; }

 private:
  // --- Core MCTS operations ---

  // Select: walk from root to a leaf using PUCT.
  // Returns the leaf node and the board state at that node.
  struct SelectResult {
    Node* node;
    ShogiBoard board;
    int edge_idx;  // Edge index in parent that leads to this node (-1 for root)
  };
  SelectResult Select(Node* root, const ShogiBoard& root_board);

  // Select the best child by PUCT score.
  // Returns edge index. May set parent's mate_status if all children are wins.
  int SelectBestChild(Node* node, bool is_root);

  // Backpropagate value from a leaf to the root.
  void Backpropagate(Node* node, float value, float draw);

  // Extract the principal variation.
  std::vector<Move> GetPV(Node* root);

  // Select the final move (by visit count, with temperature).
  Move SelectMove(Node* root, int game_ply);

  // --- Tree warmup ---

  // Build initial tree via policy-guided selection + batch GPU eval.
  // Expands warmup_nodes before multi-threaded search starts.
  void WarmupTree(Node* root, const ShogiBoard& root_board, int num_nodes);

  // Expand a node using NN output (shared by warmup and MCTS).
  static void ExpandNodeWithNN(Node* node, const NNOutput& eval,
                               const MoveList& legal_moves);

  // --- Mate search ---

  // Check for 1-ply mate (essentially free).
  Move Mate1Ply(ShogiBoard& board);

  // PV mate search: deep df-pn along principal variation.
  void PvMateSearch(Node* root, const ShogiBoard& root_board);

  // Propagate mate status up the tree after a mate is detected.
  void PropagateMateUp(Node* node);

  // --- Helpers ---

  // Dynamic cpuct.
  float Cpuct(int parent_n) const;

  // Convert root Q to centipawn score.
  static int QToCentipawns(float q);

  // Add Dirichlet noise to root edges.
  void AddDirichletNoise(Node* root);

  // --- Multi-threaded search ---

  // Barrier-based multi-threaded search (used when num_search_threads > 1).
  SearchResult SearchMT(ShogiBoard board, int game_ply,
                        std::unique_ptr<Node>& root,
                        const MoveList& legal_moves);

  // Select one leaf for a simulation context.
  void MTSelectOne(SimContext& sim, Node* start_node, const ShogiBoard& start_board);

  // Expand one simulation context with NN output.
  void MTExpandOne(SimContext& sim);

  // --- Members ---
  NNEvaluator& evaluator_;
  MCTSConfig config_;
  std::unique_ptr<MateDfpnSolver> dfpn_leaf_;
  std::unique_ptr<MateDfpnSolver> dfpn_pv_;
  std::atomic<bool> stop_{false};
};

}  // namespace jhbr2
