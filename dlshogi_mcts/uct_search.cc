#include "dlshogi_mcts/uct_search.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#include "dlshogi_mcts/search_primitives.h"
#include "dlshogi_mcts/search_repetition.h"
#include "inference/nn_diagnostics.h"
#include "mate/shallow_mate.h"

namespace dlshogi_mcts {

using lczero::BLACK;
using lczero::Move;
using lczero::MoveList;
using lczero::ShogiBoard;
using lczero::WHITE;

namespace {

constexpr uint64_t kMutexNum = 65536;
std::array<std::mutex, kMutexNum> g_position_mutexes;
std::mutex g_root_expand_mutex;

std::mutex& GetPositionMutex(const ShogiBoard* board) {
  return g_position_mutexes[board->Hash() & (kMutexNum - 1)];
}

void AddVirtualLoss(child_node_t* child, uct_node_t* current) {
  current->move_count.fetch_add(kVirtualLoss, std::memory_order_acq_rel);
  child->move_count.fetch_add(kVirtualLoss, std::memory_order_acq_rel);
}

void SubVirtualLoss(child_node_t* child, uct_node_t* current) {
  current->move_count.fetch_sub(kVirtualLoss, std::memory_order_acq_rel);
  child->move_count.fetch_sub(kVirtualLoss, std::memory_order_acq_rel);
}

float DrawValue(const SearchConfig& cfg, lczero::Color color) {
  return color == BLACK ? cfg.draw_value_black : cfg.draw_value_white;
}

struct BatchCacheKey {
  uint64_t hash = 0;
  uint16_t num_moves = 0;

  bool operator==(const BatchCacheKey& other) const {
    return hash == other.hash && num_moves == other.num_moves;
  }
};

struct BatchCacheKeyHash {
  size_t operator()(const BatchCacheKey& key) const {
    size_t h = std::hash<uint64_t>{}(key.hash);
    h ^= std::hash<uint16_t>{}(key.num_moves) + 0x9e3779b97f4a7c15ULL +
         (h << 6) + (h >> 2);
    return h;
  }
};

jhbr2::CachedNNValue ToCachedNNValue(jhbr2::NNOutput&& out,
                                     uint16_t num_legal_moves) {
  jhbr2::CachedNNValue cached;
  cached.wdl[0] = out.wdl[0];
  cached.wdl[1] = out.wdl[1];
  cached.wdl[2] = out.wdl[2];
  cached.moves_left = out.moves_left;
  cached.policy = std::move(out.policy);
  cached.num_legal_moves = num_legal_moves;
  return cached;
}

bool ValidateProbabilities(const float wdl[3],
                           const std::vector<float>& policy,
                           size_t expected_policy_size,
                           float moves_left, std::string* reason) {
  float wdl_sum = 0.0f;
  for (int i = 0; i < 3; ++i) {
    if (!std::isfinite(wdl[i]) || wdl[i] < 0.0f || wdl[i] > 1.0f) {
      if (reason) *reason = "non-finite or out-of-range WDL probability";
      return false;
    }
    wdl_sum += wdl[i];
  }
  if (!std::isfinite(wdl_sum) || std::abs(wdl_sum - 1.0f) > 1.0e-3f) {
    if (reason) *reason = "WDL probabilities do not sum to one";
    return false;
  }
  if (!std::isfinite(moves_left)) {
    if (reason) *reason = "non-finite moves-left value";
    return false;
  }
  if (policy.size() != expected_policy_size || policy.empty()) {
    if (reason) *reason = "policy size does not match legal move count";
    return false;
  }
  float policy_sum = 0.0f;
  for (float probability : policy) {
    if (!std::isfinite(probability) || probability < 0.0f ||
        probability > 1.0f) {
      if (reason) *reason = "non-finite or out-of-range policy probability";
      return false;
    }
    policy_sum += probability;
  }
  if (!std::isfinite(policy_sum) ||
      std::abs(policy_sum - 1.0f) > 1.0e-3f) {
    if (reason) *reason = "policy probabilities do not sum to one";
    return false;
  }
  return true;
}

bool ValidateNNOutput(const jhbr2::NNOutput& output,
                      size_t expected_policy_size, std::string* reason) {
  if (!output.valid) {
    if (reason) *reason = "inference backend rejected the batch";
    return false;
  }
  if (!std::isfinite(output.value) || output.value < -1.001f ||
      output.value > 1.001f || !std::isfinite(output.draw)) {
    if (reason) *reason = "non-finite or out-of-range processed value";
    return false;
  }
  return ValidateProbabilities(output.wdl, output.policy,
                               expected_policy_size, output.moves_left,
                               reason);
}

bool ValidateCachedValue(const jhbr2::CachedNNValue& value,
                         size_t expected_policy_size, std::string* reason) {
  if (value.num_legal_moves != expected_policy_size) {
    if (reason) *reason = "cached legal move count mismatch";
    return false;
  }
  return ValidateProbabilities(value.wdl, value.policy,
                               expected_policy_size, value.moves_left,
                               reason);
}

}  // namespace

struct visitor_t {
  std::vector<trajectory_t> trajectories;
  float value_win = 0.5f;
  float value_m = 0.0f;  // leaf moves-left, filled by EvalNode
  float terminal_value = 0.5f;  // perspective of the final edge's mover
  bool eval_valid = true;
  visitor_t() { trajectories.reserve(128); }
};

enum class PlayoutStatus { kTerminal, kQueuing, kDiscarded };

struct batch_element_t {
  batch_element_t(uct_node_t* node_in, const ShogiBoard& board_in,
                  float* value_win_in, float* value_m_in, bool* valid_in)
      : node(node_in),
        board(board_in),
        value_win(value_win_in),
        value_m(value_m_in),
        valid(valid_in) {
    for (int i = 0; i < node->child_num; ++i) {
      legal_moves.push_back(node->child[i].move);
    }
  }

  uct_node_t* node;
  ShogiBoard board;
  MoveList legal_moves;
  float* value_win;
  float* value_m;
  bool* valid;
};

void ApplyEvaluation(batch_element_t& elem, float value, float moves_left,
                     const std::vector<float>& policy) {
  for (int i = 0; i < elem.node->child_num; ++i) {
    const float probability =
        i < static_cast<int>(policy.size())
            ? policy[i]
            : 1.0f / std::max<int>(1, elem.node->child_num);
    elem.node->child[i].nnrate = probability;
  }
  // No child can be selected before SetEvaled() below. From this point on,
  // SelectPuctChild() accumulates the selected priors exactly as dlshogi does.
  elem.node->visited_nnrate.store(0.0f, std::memory_order_release);
  elem.node->eval_m = moves_left;
  if (elem.value_win) *elem.value_win = (value + 1.0f) * 0.5f;
  if (elem.value_m) *elem.value_m = moves_left;
  elem.node->SetEvaled();
}

void ApplyEvaluation(batch_element_t& elem,
                     const jhbr2::CachedNNValue& cached) {
  ApplyEvaluation(elem, cached.wdl[0] - cached.wdl[2], cached.moves_left,
                  cached.policy);
}

void ApplyEvaluation(batch_element_t& elem, const jhbr2::NNOutput& output) {
  ApplyEvaluation(elem, output.value, output.moves_left, output.policy);
}

struct LocalCacheProbe {
  jhbr2::NNCache::Handle hit;
  int miss_index = -1;
  int wait_index = -1;
  bool invalid = false;
};

class UCTSearcher {
 public:
  UCTSearcher(UCTSearcherGroup* grp, int thread_id, int batch_max)
      : grp_(grp), thread_id_(thread_id), batch_max_(batch_max) {
    batch_.reserve(batch_max_);
  }

  void Run() { handle_ = std::thread([this] { ParallelUctSearch(); }); }
  void Join() {
    if (handle_.joinable()) handle_.join();
  }

 private:
  void ParallelUctSearch();
  PlayoutStatus UctSearch(ShogiBoard* board, child_node_t* parent,
                          uct_node_t* current, visitor_t& visitor);
  unsigned SelectMaxUcbChild(child_node_t* parent, uct_node_t* current);
  void QueuingNode(const ShogiBoard* board, uct_node_t* node,
                   float* value_win, float* value_m, bool* valid);
  void EvalNode();

  UCTSearcherGroup* grp_;
  int thread_id_;
  int batch_max_;
  std::vector<batch_element_t> batch_;
  std::thread handle_;
};

UCTSearcherGroup::UCTSearcherGroup(Search* owner_in, jhbr2::NNEvaluator* nn_in,
                                   int gpu_id_in, int threads_in,
                                   int batch_max_in)
    : owner(owner_in),
      nn(nn_in),
      gpu_id(gpu_id_in) {
  searchers_.reserve(threads_in);
  for (int i = 0; i < threads_in; ++i) {
    searchers_.push_back(
        std::make_unique<UCTSearcher>(this, i, batch_max_in));
  }
}

UCTSearcherGroup::UCTSearcherGroup(UCTSearcherGroup&&) noexcept = default;
UCTSearcherGroup& UCTSearcherGroup::operator=(UCTSearcherGroup&&) noexcept =
    default;
UCTSearcherGroup::~UCTSearcherGroup() { Join(); }

void UCTSearcherGroup::Run() {
  for (auto& searcher : searchers_) searcher->Run();
}

void UCTSearcherGroup::Join() {
  for (auto& searcher : searchers_) searcher->Join();
}

Search::Search(std::vector<jhbr2::NNEvaluator*> evaluators,
               const SearchConfig& config)
    : config_(config),
      evaluators_(std::move(evaluators)),
      nn_cache_(config.nn_cache_size) {
  groups_.reserve(evaluators_.size());
  for (int g = 0; g < static_cast<int>(evaluators_.size()); ++g) {
    groups_.emplace_back(this, evaluators_[g], g, config_.workers_per_gpu,
                         config_.minibatch_size);
  }
}

Search::~Search() {
  Stop();
  for (auto& group : groups_) group.Join();
}

void Search::ResetForNewGame() {
  // USI command processing is synchronous, so searches should already be
  // idle here. Join defensively before invalidating pointers into the tree.
  Stop();
  for (auto& group : groups_) group.Join();
  root_ = nullptr;
  tree_.DeallocateTree();
  tree_reused_ = false;
  root_visits_before_ = 0;
  playout_count_.store(0, std::memory_order_release);

  // Neural evaluations depend only on the position and model, not on the
  // game. Preserve the cache across games: destroying millions of cached
  // policy vectors synchronously can consume the next game's clock.
  nn_cache_.ResetStats();
}

bool Search::IsSearchActive() const {
  if (stop_.load(std::memory_order_acquire)) return false;
  if (config_.max_nodes > 0 &&
      playout_count_.load(std::memory_order_acquire) >= config_.max_nodes) {
    return false;
  }
  if (config_.max_time > 0.0f &&
      timer_.ElapsedMs() >= static_cast<int>(config_.max_time * 1000.0f)) {
    return false;
  }
  return true;
}

void Search::ExpandRoot() {
  std::lock_guard<std::mutex> lk(g_root_expand_mutex);
  if (root_->child_num == 0) root_->ExpandNode(&root_board_);
}

void Search::RejectRootMates() {
  if (config_.root_mate_depth <= 0 || !root_ || root_->child_num == 0) {
    return;
  }

  // Usually this checks only the selected move. If it permits a forced
  // mate, mark it as winning for the opponent and try the next-best root
  // candidate using the visits already gathered by MCTS.
  for (int attempt = 0; attempt < root_->child_num; ++attempt) {
    const unsigned idx = SelectBestChild(root_);
    auto& child = root_->child[idx];
    if (child.IsLose() || child.IsWin()) return;

    ShogiBoard reply = root_board_;
    reply.DoMove(child.move);
    if (!jhbr2::shallow_mate::HasMateWithin(
            reply, config_.root_mate_depth)) {
      return;
    }
    child.SetWin();
  }
}

unsigned Search::SelectBestChild(const uct_node_t* node) const {
  unsigned best = 0;
  int best_visits = std::numeric_limits<int>::min();
  float best_prior = -1.0f;
  for (int i = 0; i < node->child_num; ++i) {
    const auto& child = node->child[i];
    const int visits = child.move_count.load(std::memory_order_acquire);
    if (child.IsLose()) return i;
    if (child.IsWin()) continue;
    if (visits > best_visits ||
        (visits == best_visits && child.nnrate > best_prior)) {
      best_visits = visits;
      best_prior = child.nnrate;
      best = static_cast<unsigned>(i);
    }
  }
  return best;
}

SearchResult Search::Run(ShogiBoard board, uint64_t starting_pos_key,
                         const std::vector<Move>& moves) {
  stop_.store(false, std::memory_order_release);
  playout_count_.store(0, std::memory_order_release);
  nn_cache_.ResetStats();
  timer_.Restart();
  last_info_ms_ = 0;
  root_board_ = std::move(board);
  tree_reused_ = tree_.ResetToPosition(starting_pos_key, moves);
  root_ = tree_.GetCurrentHead();
  root_visits_before_ =
      std::max(0, root_->move_count.load(std::memory_order_acquire));

  auto root_legal = root_board_.GenerateLegalMoves();
  if (root_legal.empty()) return BuildResult();
  if (root_legal.size() == 1) {
    SearchResult result;
    result.best_move = root_legal[0];
    result.tree_reused = tree_reused_;
    result.root_visits_before = root_visits_before_;
    result.nn_cache = nn_cache_.GetStats();
    return result;
  }

  ExpandRoot();
  for (auto& group : groups_) group.Run();
  for (auto& group : groups_) group.Join();
  RejectRootMates();
  MaybeOutputInfo();
  return BuildResult();
}

unsigned UCTSearcher::SelectMaxUcbChild(child_node_t* parent,
                                        uct_node_t* current) {
  const auto& cfg = grp_->owner->config_;
  const bool is_root = parent == nullptr;
  PuctParameters params;
  params.c_init = is_root ? cfg.c_init_root : cfg.c_init;
  params.c_base = is_root ? cfg.c_base_root : cfg.c_base;
  params.fpu_reduction =
      is_root ? cfg.c_fpu_reduction_root : cfg.c_fpu_reduction;
  params.moves_left_weight = cfg.moves_left_weight;
  params.moves_left_threshold = cfg.moves_left_threshold;
  params.moves_left_cap = cfg.moves_left_cap;
  return SelectPuctChild(parent, current, params);
}

PlayoutStatus UCTSearcher::UctSearch(ShogiBoard* board, child_node_t* parent,
                                     uct_node_t* current,
                                     visitor_t& visitor) {
  const auto& cfg = grp_->owner->config_;
  const float parent_draw_value =
      DrawValue(cfg, ~board->side_to_move());

  if (TryGetProvenEdgeValue(parent, parent_draw_value,
                            &visitor.terminal_value)) {
    return PlayoutStatus::kTerminal;
  }
  if (board->CanDeclareWin()) {
    visitor.terminal_value =
        ResolveTerminalEdge(parent, EdgeOutcome::kLoss);
    return PlayoutStatus::kTerminal;
  }

  switch (GetSearchRepetitionResult(*board, parent == nullptr)) {
    case ShogiBoard::RepetitionResult::kLoss:
      visitor.terminal_value =
          ResolveTerminalEdge(parent, EdgeOutcome::kWin);
      return PlayoutStatus::kTerminal;
    case ShogiBoard::RepetitionResult::kWin:
      visitor.terminal_value =
          ResolveTerminalEdge(parent, EdgeOutcome::kLoss);
      return PlayoutStatus::kTerminal;
    case ShogiBoard::RepetitionResult::kDraw:
      visitor.terminal_value = ResolveTerminalEdge(
          parent, EdgeOutcome::kDraw, parent_draw_value);
      return PlayoutStatus::kTerminal;
    case ShogiBoard::RepetitionResult::kNone:
      break;
  }

  if (board->ply() > cfg.max_moves_to_draw) {
    const EdgeOutcome outcome = board->GenerateLegalMoves().empty()
                                    ? EdgeOutcome::kWin
                                    : EdgeOutcome::kDraw;
    visitor.terminal_value =
        ResolveTerminalEdge(parent, outcome, parent_draw_value);
    return PlayoutStatus::kTerminal;
  }

  unsigned next = 0;
  Move next_move;
  uct_node_t* next_node = nullptr;
  {
    std::lock_guard<std::mutex> lk(GetPositionMutex(board));
    if (!current->IsEvaled()) {
      if (current->child_num != 0) return PlayoutStatus::kDiscarded;
      if (cfg.leaf_mate_depth > 0) {
        ShogiBoard tmp = *board;
        if (jhbr2::shallow_mate::HasMateWithin(tmp, cfg.leaf_mate_depth)) {
          visitor.terminal_value =
              ResolveTerminalEdge(parent, EdgeOutcome::kLoss);
          return PlayoutStatus::kTerminal;
        }
      }
      current->ExpandNode(board);
      if (current->child_num == 0) {
        current->SetEvaled();
        visitor.terminal_value =
            ResolveTerminalEdge(parent, EdgeOutcome::kWin);
        return PlayoutStatus::kTerminal;
      }
      QueuingNode(board, current, &visitor.value_win, &visitor.value_m,
                  &visitor.eval_valid);
      return PlayoutStatus::kQueuing;
    }

    if (current->child_num == 0) {
      visitor.terminal_value =
          ResolveTerminalEdge(parent, EdgeOutcome::kWin);
      return PlayoutStatus::kTerminal;
    }
    next = SelectMaxUcbChild(parent, current);
    AddVirtualLoss(&current->child[next], current);
    visitor.trajectories.push_back({current, next});
    next_move = current->child[next].move;
    next_node = current->child_nodes[next].get();
    if (next_node == nullptr) next_node = current->CreateChildNode(next);
  }

  board->DoMove(next_move);
  return UctSearch(board, &current->child[next], next_node, visitor);
}

void UCTSearcher::QueuingNode(const ShogiBoard* board, uct_node_t* node,
                              float* value_win, float* value_m, bool* valid) {
  batch_.emplace_back(node, *board, value_win, value_m, valid);
}

void UCTSearcher::EvalNode() {
  if (batch_.empty()) return;

  const size_t batch_size = batch_.size();
  std::vector<int> result_to_miss(batch_size, -1);
  std::vector<std::pair<ShogiBoard, MoveList>> miss_batch;
  std::vector<uint64_t> miss_keys;
  std::vector<uint16_t> miss_num_moves;
  std::vector<jhbr2::NNCache::Probe> miss_reservations;
  std::vector<jhbr2::NNCache::Probe> wait_probes;
  std::vector<int> result_to_wait(batch_size, -1);
  std::unordered_map<BatchCacheKey, LocalCacheProbe, BatchCacheKeyHash>
      local_probes;

  miss_batch.reserve(batch_size);
  miss_keys.reserve(batch_size);
  miss_num_moves.reserve(batch_size);
  miss_reservations.reserve(batch_size);
  wait_probes.reserve(batch_size);
  local_probes.reserve(batch_size);

  auto& nn_cache = grp_->owner->nn_cache_;
  auto reject = [&](batch_element_t& elem, const char* stage,
                    const std::string& reason) {
    // Expansion happens before inference.  Leave a rejected node in a safe,
    // evaluated state so a reused tree cannot permanently discard this branch.
    // The synthetic neutral value is not backed up or cached.
    if (!elem.node->IsEvaled()) {
      const float uniform =
          elem.node->child_num > 0 ? 1.0f / elem.node->child_num : 0.0f;
      std::vector<float> neutral_policy(elem.node->child_num, uniform);
      ApplyEvaluation(elem, 0.0f, 0.0f, neutral_policy);
    }
    if (elem.valid) *elem.valid = false;
    grp_->owner->Stop();
    std::ostringstream details;
    details << "backend=mcts reason=\"" << reason << "\""
            << " gpu=" << grp_->gpu_id << " slot=" << thread_id_
            << " position_hash=0x" << std::hex << elem.board.Hash()
            << std::dec << " legal_moves=" << elem.legal_moves.size()
            << " sfen=\"" << elem.board.ToSfen() << "\"";
    jhbr2::nn_diagnostics::LogOnce(stage, details.str());
  };

  for (size_t i = 0; i < batch_size; ++i) {
    auto& elem = batch_[i];
    const uint64_t key = elem.board.Hash();
    const uint16_t num_moves =
        static_cast<uint16_t>(elem.legal_moves.size());

    const BatchCacheKey batch_key{key, num_moves};
    auto [probe_it, inserted] = local_probes.try_emplace(batch_key);
    if (!inserted) {
      if (probe_it->second.invalid) {
        reject(elem, "local_batch", "duplicate of an invalid evaluation");
      } else if (probe_it->second.hit) {
        ApplyEvaluation(elem, *probe_it->second.hit);
      } else if (probe_it->second.miss_index >= 0) {
        result_to_miss[i] = probe_it->second.miss_index;
      } else {
        result_to_wait[i] = probe_it->second.wait_index;
      }
      continue;
    }

    jhbr2::NNCache::Probe cache_probe;
    if (nn_cache.Enabled()) {
      cache_probe = nn_cache.LookupOrReserve(key, num_moves);
      if (cache_probe.IsHit()) {
        probe_it->second.hit = cache_probe.Hit();
        std::string reason;
        if (!ValidateCachedValue(*probe_it->second.hit, num_moves, &reason)) {
          probe_it->second.invalid = true;
          reject(elem, "nncache_hit", reason);
          continue;
        }
        ApplyEvaluation(elem, *probe_it->second.hit);
        continue;
      }
      if (cache_probe.IsWaiter()) {
        const int wait_idx = static_cast<int>(wait_probes.size());
        probe_it->second.wait_index = wait_idx;
        result_to_wait[i] = wait_idx;
        wait_probes.push_back(std::move(cache_probe));
        continue;
      }
    }

    const int miss_idx = static_cast<int>(miss_batch.size());
    probe_it->second.miss_index = miss_idx;
    result_to_miss[i] = miss_idx;
    miss_batch.emplace_back(elem.board, elem.legal_moves);
    miss_keys.push_back(key);
    miss_num_moves.push_back(num_moves);
    if (nn_cache.Enabled()) {
      miss_reservations.push_back(std::move(cache_probe));
    }
  }

  std::vector<jhbr2::NNOutput> miss_results;
  if (!miss_batch.empty()) {
    miss_results = grp_->nn->EvaluateBatchSlot(thread_id_, miss_batch);
  }

  if (miss_results.size() != miss_batch.size()) {
    std::ostringstream reason;
    reason << "inference result count " << miss_results.size()
           << " does not match request count " << miss_batch.size();
    miss_results.resize(miss_batch.size());
    for (auto& output : miss_results) output.valid = false;
    if (!batch_.empty()) reject(batch_.front(), "inference_count", reason.str());
  }

  std::vector<bool> miss_valid(miss_results.size(), false);
  for (size_t i = 0; i < miss_results.size(); ++i) {
    std::string reason;
    miss_valid[i] =
        ValidateNNOutput(miss_results[i], miss_num_moves[i], &reason);
    if (!miss_valid[i]) {
      auto it = std::find(result_to_miss.begin(), result_to_miss.end(),
                          static_cast<int>(i));
      if (it != result_to_miss.end()) {
        reject(batch_[std::distance(result_to_miss.begin(), it)],
               "inference_result", reason);
      }
    }
  }

  for (size_t i = 0; i < batch_size; ++i) {
    const int miss_idx = result_to_miss[i];
    if (miss_idx >= 0 && miss_valid[miss_idx]) {
      ApplyEvaluation(batch_[i], miss_results[miss_idx]);
    } else if (miss_idx >= 0) {
      reject(batch_[i], "inference_result",
             "inference result rejected before cache/backup");
    }
  }

  if (nn_cache.Enabled()) {
    for (size_t i = 0; i < miss_results.size(); ++i) {
      if (miss_valid[i]) {
        nn_cache.Publish(std::move(miss_reservations[i]),
                         ToCachedNNValue(std::move(miss_results[i]),
                                         miss_num_moves[i]));
      } else {
        nn_cache.Cancel(std::move(miss_reservations[i]));
      }
    }
  }

  std::vector<jhbr2::NNCache::Handle> waited_values(wait_probes.size());
  for (size_t i = 0; i < wait_probes.size(); ++i) {
    waited_values[i] = wait_probes[i].Wait();
  }
  for (size_t i = 0; i < batch_size; ++i) {
    const int wait_idx = result_to_wait[i];
    if (wait_idx < 0) continue;
    if (!waited_values[wait_idx]) {
      reject(batch_[i], "nncache_wait", "owner cancelled invalid inference");
      continue;
    }
    std::string reason;
    if (!ValidateCachedValue(*waited_values[wait_idx],
                             batch_[i].legal_moves.size(), &reason)) {
      reject(batch_[i], "nncache_wait", reason);
      continue;
    }
    ApplyEvaluation(batch_[i], *waited_values[wait_idx]);
  }
  batch_.clear();
}

void UCTSearcher::ParallelUctSearch() {
  auto* current_root = grp_->owner->root_;
  if (!grp_->owner->IsSearchActive()) return;
  {
    std::lock_guard<std::mutex> lk(g_root_expand_mutex);
    if (grp_->owner->IsSearchActive() && !current_root->IsEvaled()) {
      batch_.clear();
      float value_win = 0.5f;
      float value_m = 0.0f;
      bool eval_valid = true;
      QueuingNode(&grp_->owner->root_board_, current_root, &value_win, &value_m,
                  &eval_valid);
      EvalNode();
      if (!eval_valid) return;
    }
  }
  if (!grp_->owner->IsSearchActive()) return;

  std::vector<visitor_t> visitor_pool(batch_max_);
  std::vector<visitor_t*> visitor_batch;
  std::vector<std::vector<trajectory_t>*> discarded;
  visitor_batch.reserve(batch_max_);
  discarded.reserve(batch_max_);
  auto unwind = [](const std::vector<trajectory_t>& path) {
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
      SubVirtualLoss(&it->parent->child[it->child_idx], it->parent);
    }
  };

  while (grp_->owner->IsSearchActive()) {
    visitor_batch.clear();
    discarded.clear();
    batch_.clear();

    for (int i = 0; i < batch_max_ && grp_->owner->IsSearchActive(); ++i) {
      ShogiBoard board = grp_->owner->root_board_;
      visitor_pool[i].trajectories.clear();
      visitor_pool[i].value_win = 0.5f;
      visitor_pool[i].value_m = 0.0f;
      visitor_pool[i].terminal_value = 0.5f;
      visitor_pool[i].eval_valid = true;
      const PlayoutStatus status =
          UctSearch(&board, nullptr, current_root, visitor_pool[i]);
      if (status != PlayoutStatus::kDiscarded) {
        grp_->owner->playout_count_.fetch_add(1, std::memory_order_acq_rel);
      }
      if (status == PlayoutStatus::kDiscarded) {
        discarded.push_back(&visitor_pool[i].trajectories);
      } else if (status == PlayoutStatus::kQueuing) {
        visitor_batch.push_back(&visitor_pool[i]);
      } else {
        if (!BackupTrajectory(visitor_pool[i].trajectories,
                              visitor_pool[i].terminal_value)) {
          unwind(visitor_pool[i].trajectories);
          grp_->owner->Stop();
        }
      }
    }

    EvalNode();

    for (auto* path : discarded) {
      unwind(*path);
    }

    for (auto* visitor : visitor_batch) {
      if (!visitor->eval_valid) {
        unwind(visitor->trajectories);
        continue;
      }
      if (!BackupTrajectory(visitor->trajectories,
                            1.0f - visitor->value_win, visitor->value_m)) {
        unwind(visitor->trajectories);
        grp_->owner->Stop();
      }
    }

    grp_->owner->MaybeOutputInfo();
  }
}

int Search::QToCentipawns(float win_rate) const {
  if (!std::isfinite(win_rate)) {
    std::ostringstream details;
    details << "backend=mcts reason=\"non-finite root win rate\""
            << " win_rate=" << win_rate;
    jhbr2::nn_diagnostics::LogOnce("cp_conversion", details.str());
    return 0;
  }
  win_rate = std::clamp(win_rate, 0.001f, 0.999f);
  return static_cast<int>(-std::log(1.0f / win_rate - 1.0f) * 756.0f);
}

std::vector<Move> Search::GetPV() const {
  std::vector<Move> pv;
  const uct_node_t* node = root_;
  while (node && node->child_num > 0 && node->child) {
    const unsigned idx = SelectBestChild(node);
    pv.push_back(node->child[idx].move);
    if (!node->child_nodes || !node->child_nodes[idx]) break;
    node = node->child_nodes[idx].get();
    if (pv.size() > 256) break;
  }
  return pv;
}

void Search::MaybeOutputInfo() {
  if (!config_.info_callback) return;
  const int elapsed = timer_.ElapsedMs();
  std::lock_guard<std::mutex> lk(info_mutex_);
  if (elapsed - last_info_ms_ < config_.info_interval_ms) {
    return;
  }
  last_info_ms_ = elapsed;
  const int nodes = playout_count_.load(std::memory_order_acquire);
  SearchInfo info;
  info.nodes = nodes;
  info.time_ms = elapsed;
  info.nps = elapsed > 0 ? static_cast<int>(nodes * 1000LL / elapsed) : 0;
  info.pv = GetPV();
  info.depth = static_cast<int>(info.pv.size());
  info.nn_cache = nn_cache_.GetStats();
  if (root_ && root_->child_num > 0) {
    const unsigned best = SelectBestChild(root_);
    const auto& ch = root_->child[best];
    const int n = ch.move_count.load(std::memory_order_acquire);
    const float wp = ch.IsLose() ? 1.0f
                     : ch.IsWin() ? 0.0f
                     : n > 0 ? ch.win.load(std::memory_order_acquire) / n
                             : 0.5f;
    info.score_cp = QToCentipawns(wp);
  }
  config_.info_callback(info);
}

SearchResult Search::BuildResult() const {
  SearchResult result;
  result.tree_reused = tree_reused_;
  result.root_visits_before = root_visits_before_;
  result.nodes = playout_count_.load(std::memory_order_acquire);
  result.time_sec = timer_.ElapsedMs() / 1000.0f;
  result.nps = result.time_sec > 0.001f ? result.nodes / result.time_sec : 0.0f;
  result.nn_cache = nn_cache_.GetStats();
  if (!root_ || root_->child_num == 0) return result;

  const unsigned best = SelectBestChild(root_);
  const auto& child = root_->child[best];
  result.best_move = child.move;
  const int n = child.move_count.load(std::memory_order_acquire);
  const float wp = child.IsLose() ? 1.0f
                   : child.IsWin() ? 0.0f
                   : n > 0 ? child.win.load(std::memory_order_acquire) / n
                           : 0.5f;
  const float safe_wp = std::isfinite(wp) ? wp : 0.5f;
  if (!std::isfinite(wp)) {
    std::ostringstream details;
    details << "backend=mcts reason=\"non-finite final root win rate\""
            << " visits=" << n << " win="
            << child.win.load(std::memory_order_acquire);
    jhbr2::nn_diagnostics::LogOnce("build_result", details.str());
  }
  if (safe_wp < config_.resign_threshold) result.best_move = Move();
  result.score_cp = QToCentipawns(safe_wp);
  result.pv = GetPV();
  return result;
}

}  // namespace dlshogi_mcts
