/*
  JHBR2 Shogi Engine — lc0-style MCTS Search (simplified port)

  Preserves lc0's proven algorithms:
    - PUCT selection with dynamic cpuct
    - FPU (First Play Urgency) with relative reduction
    - Virtual loss via n_in_flight (separate from N)
    - Running-average Q (not W/N) via FinalizeScoreUpdate
    - Multi-threaded gathering with collision handling
    - Batch NN evaluation
    - Dirichlet noise at root

  Simplified from lc0:
    - No Syzygy tablebase
    - No WDL rescaling / contempt
    - No out-of-order evaluation
    - No task parallelization within gathering
    - No OptionsDict — simple struct config
    - Uses our NNEvaluator and ShogiBoard directly
*/

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

#include "lc0_mcts/backend.h"
#include "lc0_mcts/node.h"
#include "lc0_mcts/types.h"

namespace lc0_shogi {

using jhbr2::NNEvaluator;
using jhbr2::NNOutput;

// =====================================================================
// Search parameters — simple struct, no OptionsDict
// =====================================================================

struct SearchConfig {
  // PUCT (lc0 defaults)
  float cpuct = 1.745f;
  float cpuct_base = 38739.0f;
  float cpuct_factor = 3.894f;

  // FPU
  float fpu_value = 0.330f;
  float fpu_value_at_root = 1.0f;

  // Noise
  float noise_epsilon = 0.0f;
  float noise_alpha = 0.15f;

  // Collision limits
  int max_collision_events = 32;
  int max_collision_visits = 9999;

  // Search limits
  int max_nodes = 800;
  float max_time = 0.0f;  // seconds, 0 = unlimited

  // Threading
  int num_threads = 1;
  int minibatch_size = 32;  // Target leaves per worker iteration

  // Draw score (positive = prefer draws)
  float draw_score = 0.0f;

  // Solid tree threshold
  int solid_threshold = 512;

  // Temperature
  float temperature = 0.0f;
  int temp_cutoff_move = 0;

  // Sticky endgames (propagate bounds from terminal children)
  bool sticky_endgames = true;

  // Use per-leaf gathering instead of bulk distribution.
  // Better NPS with static-batch TensorRT engines.
  bool per_leaf_gathering = false;

  // Policy softmax temperature (1.0 = use NN output directly)
  float policy_softmax_temp = 1.0f;

  // Leaf df-pn: inline mate detection at leaf nodes.
  // Budget in nodes (0 = disabled). Typical: 10-100.
  int leaf_dfpn_nodes = 0;
};

// =====================================================================
// Search result (returned to USI handler)
// =====================================================================

struct SearchResult {
  Move best_move;
  Move ponder_move;
  int nodes = 0;
  float time_sec = 0.0f;
  float nps = 0.0f;
  float root_q = 0.0f;
  float root_d = 0.0f;
  int score_cp = 0;
  std::vector<Move> pv;
};

// =====================================================================
// Search — manages the search tree and worker threads
// =====================================================================

class Search {
 public:
  Search(NNEvaluator& evaluator, const SearchConfig& config);
  ~Search();

  // Run search from position. game_ply is for temperature scheduling.
  SearchResult Run(ShogiBoard board, int game_ply = 1);

  // Stop search (can be called from another thread).
  void Stop() { stop_.store(true, std::memory_order_release); }

 private:
  friend class SearchWorker;

  bool IsSearchActive() const;

  // Get best child without temperature.
  EdgeAndNode GetBestChildNoTemperature(Node* parent) const;

  // Get draw score (flips sign for odd depth).
  float GetDrawScore(bool is_odd_depth) const;

  // Cancel all shared collisions.
  void CancelSharedCollisions();

  // Extract PV from tree.
  std::vector<Move> GetPV(Node* root) const;

  // Convert Q to centipawns.
  static int QToCentipawns(float q);

  Backend backend_;
  SearchConfig config_;

  // Tree.
  NodeTree tree_;
  Node* root_node_ = nullptr;

  // Threading.
  std::atomic<bool> stop_{false};
  mutable std::shared_mutex nodes_mutex_;

  // Stats (guarded by nodes_mutex_).
  EdgeAndNode current_best_edge_;
  int64_t total_playouts_ = 0;
  int64_t total_batches_ = 0;
  uint16_t max_depth_ = 0;

  // Collision tracking.
  std::vector<std::pair<Node*, int>> shared_collisions_;

  // Timing.
  std::chrono::steady_clock::time_point start_time_;

  // Board at root (for move generation during ExtendNode).
  ShogiBoard root_board_;
  bool root_is_black_to_move_ = false;
};

// =====================================================================
// SearchWorker — one per thread, does the actual MCTS work
// =====================================================================

class SearchWorker {
 public:
  SearchWorker(Search* search, const SearchConfig& config);

  // Main loop: execute iterations until search stops.
  void RunBlocking();

 private:
  // One complete MCTS iteration.
  void ExecuteOneIteration();

  // Stages of one iteration.
  void GatherMinibatch();
  void RunNNComputation();
  void FetchMinibatchResults();
  void DoBackupUpdate();

  struct NodeToProcess {
    Node* node;
    int multivisit = 1;
    int maxvisit = 0;
    uint16_t depth = 0;
    bool is_collision = false;
    bool nn_queried = false;

    float eval_q = 0.0f;
    float eval_d = 0.0f;
    float eval_m = 0.0f;

    std::vector<Move> moves_to_node;

    bool IsExtendable() const { return !is_collision && !node->IsTerminal(); }
    bool IsCollision() const { return is_collision; }

    static NodeToProcess Visit(Node* node, uint16_t depth,
                                std::vector<Move> moves = {}) {
      NodeToProcess ntp;
      ntp.node = node;
      ntp.depth = depth;
      ntp.multivisit = 1;
      ntp.moves_to_node = std::move(moves);
      return ntp;
    }
    static NodeToProcess Collision(Node* node, uint16_t depth,
                                   int count, int max_count = 0) {
      NodeToProcess ntp;
      ntp.node = node;
      ntp.depth = depth;
      ntp.multivisit = count;
      ntp.maxvisit = max_count;
      ntp.is_collision = true;
      return ntp;
    }
  };

  // --- Gathering strategies ---

  // Per-leaf: one PickNodeToExtend call per leaf (better for static-batch TRT).
  void GatherMinibatchPerLeaf();
  NodeToProcess PickNodeToExtend();
  void ExtendNodeInPlace(NodeToProcess& ntp);

  // Bulk: lc0-style one recursive tree walk (better for dynamic-batch).
  void GatherMinibatchBulk();
  void PickNodesToExtend(int collision_limit);
  void PickNodesToExtendTask(Node* node, int base_depth, int collision_limit,
                             const std::vector<Move>& moves_to_base,
                             std::vector<NodeToProcess>* receiver);
  void ProcessPickedNodes();
  void ExtendNode(Node* node, int depth, const std::vector<Move>& moves);

  void DoBackupUpdateSingleNode(const NodeToProcess& ntp);
  bool MaybeSetBounds(Node* p, float m, int* n_to_fix,
                      float* v_delta, float* d_delta, float* m_delta) const;

  Search* const search_;
  const SearchConfig& config_;
  std::vector<NodeToProcess> minibatch_;

  // Workspace for bulk visit distribution (reused across iterations).
  std::vector<std::unique_ptr<std::array<int, 256>>> vtp_buffer_;
  std::array<Node::Iterator, 256> cur_iters_;

  std::unique_ptr<Computation> computation_;
  std::vector<int> nn_batch_indices_;
};

}  // namespace lc0_shogi
