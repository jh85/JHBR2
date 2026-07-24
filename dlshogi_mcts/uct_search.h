#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "dlshogi_mcts/uct_node.h"
#include "inference/nn_cache.h"
#ifdef USE_TENSORRT
#include "inference/nn_tensorrt.h"
#else
#include "inference/nn_eval.h"
#endif

namespace dlshogi_mcts {

struct SearchInfo {
  int depth = 0;
  int score_cp = 0;
  int nodes = 0;
  int nps = 0;
  int time_ms = 0;
  std::vector<lczero::Move> pv;
  jhbr2::NNCacheStats nn_cache;
};
using InfoCallback = std::function<void(const SearchInfo&)>;

struct SearchConfig {
  float c_init = 1.25f;
  float c_base = 19652.0f;
  float c_fpu_reduction = 0.27f;
  float c_init_root = 1.25f;
  float c_base_root = 19652.0f;
  float c_fpu_reduction_root = 0.0f;
  float draw_value_black = 0.5f;
  float draw_value_white = 0.5f;
  float resign_threshold = 0.01f;
  int max_nodes = 800;
  float max_time = 0.0f;
  int workers_per_gpu = 2;
  int minibatch_size = 128;
  int max_moves_to_draw = 100000;
  int leaf_mate_depth = 5;

  // Before returning a move, reject root candidates that let the opponent
  // force mate within this many plies. This makes deeper defensive coverage
  // affordable without paying for it at every MCTS leaf.
  int root_mate_depth = 7;

  // Moves-left (MLH) effect in selection. Disabled by default (weight 0):
  // when > 0, nudges selection toward shorter lines when winning / longer when
  // losing. Needs a model trained with the MLH head; tune with real games.
  float moves_left_weight = 0.0f;     // master switch / strength
  float moves_left_threshold = 0.0f;  // only apply when |q-0.5| exceeds this
  float moves_left_cap = 20.0f;       // clamp |child_M - parent_M| (plies)

  size_t nn_cache_size = 0;
  int info_interval_ms = 1000;
  InfoCallback info_callback = nullptr;
};

struct SearchResult {
  lczero::Move best_move;
  bool tree_reused = false;
  int root_visits_before = 0;
  int nodes = 0;
  float time_sec = 0.0f;
  float nps = 0.0f;
  int score_cp = 0;
  std::vector<lczero::Move> pv;
  jhbr2::NNCacheStats nn_cache;
};

class Search;

class UCTSearcherGroup {
 public:
  UCTSearcherGroup(Search* owner, jhbr2::NNEvaluator* nn, int gpu_id,
                   int threads, int batch_max);
  UCTSearcherGroup(UCTSearcherGroup&&) noexcept;
  UCTSearcherGroup& operator=(UCTSearcherGroup&&) noexcept;
  ~UCTSearcherGroup();

  void Run();
  void Join();

  Search* owner = nullptr;
  jhbr2::NNEvaluator* nn = nullptr;
  int gpu_id = 0;

 private:
  std::vector<std::unique_ptr<class UCTSearcher>> searchers_;
};

class Search {
 public:
  Search(std::vector<jhbr2::NNEvaluator*> evaluators, const SearchConfig& config);
  ~Search();

  SearchResult Run(lczero::ShogiBoard board, uint64_t starting_pos_key,
                   const std::vector<lczero::Move>& moves);
  void Stop() { stop_.store(true, std::memory_order_release); }
  void SetMaxTime(float seconds) { config_.max_time = seconds; }
  void SetMaxNodes(size_t n) { config_.max_nodes = static_cast<int>(n); }

 private:
  friend class UCTSearcher;
  friend class UCTSearcherGroup;

  bool IsSearchActive() const;
  void ExpandRoot();
  void RejectRootMates();
  unsigned SelectBestChild(const uct_node_t* node) const;
  SearchResult BuildResult() const;
  std::vector<lczero::Move> GetPV() const;
  int QToCentipawns(float win_rate) const;
  void MaybeOutputInfo();

  SearchConfig config_;
  std::vector<jhbr2::NNEvaluator*> evaluators_;
  std::vector<UCTSearcherGroup> groups_;
  NodeTree tree_;
  jhbr2::NNCache nn_cache_;
  lczero::ShogiBoard root_board_;
  uct_node_t* root_ = nullptr;
  bool tree_reused_ = false;
  int root_visits_before_ = 0;
  std::atomic<bool> stop_{false};
  std::atomic<int> playout_count_{0};
  Timer timer_;
  mutable std::mutex info_mutex_;
  int last_info_ms_ = 0;
};

}  // namespace dlshogi_mcts
