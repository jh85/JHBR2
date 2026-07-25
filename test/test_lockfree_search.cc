#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dlshogi_mcts/uct_search.h"
#include "shogi/bitboard.h"
#include "shogi/board.h"
#include "shogi/types.h"

// A deterministic, thread-safe evaluator lets this test exercise the complete
// multi-worker MCTS path without requiring ONNX Runtime, TensorRT, or a GPU.
namespace jhbr2 {

struct NNEvaluator::Impl {};

NNEvaluator::NNEvaluator(const std::string& onnx_path, bool use_gpu,
                         int device_id, int num_slots,
                         ModelFormat model_format)
    : impl_(std::make_unique<Impl>()) {
  (void)onnx_path;
  (void)use_gpu;
  (void)device_id;
  (void)num_slots;
  (void)model_format;
}

NNEvaluator::~NNEvaluator() = default;

NNOutput NNEvaluator::Evaluate(const ShogiBoard& board,
                               const MoveList& legal_moves) {
  std::vector<std::pair<ShogiBoard, MoveList>> batch;
  batch.emplace_back(board, legal_moves);
  return std::move(EvaluateBatch(batch).front());
}

std::vector<NNOutput> NNEvaluator::EvaluateBatch(
    const std::vector<std::pair<ShogiBoard, MoveList>>& batch) {
  std::vector<NNOutput> outputs(batch.size());
  for (size_t i = 0; i < batch.size(); ++i) {
    auto& output = outputs[i];
    const size_t move_count =
        static_cast<size_t>(std::max(batch[i].second.size(), 0));
    output.value = 0.0f;
    output.draw = 0.1f;
    output.wdl[0] = 0.45f;
    output.wdl[1] = 0.1f;
    output.wdl[2] = 0.45f;
    output.moves_left = 80.0f;
    output.valid = true;
    output.policy.assign(move_count,
                         move_count == 0
                             ? 0.0f
                             : 1.0f / static_cast<float>(move_count));
  }
  return outputs;
}

}  // namespace jhbr2

namespace {

using dlshogi_mcts::Search;
using dlshogi_mcts::SearchConfig;
using dlshogi_mcts::SearchResult;
using lczero::Move;
using lczero::ShogiBoard;

int failures = 0;

void Check(const char* name, bool condition) {
  if (!condition) {
    std::printf("  FAIL  %s\n", name);
    ++failures;
  }
}

bool IsLegalMove(ShogiBoard board, Move move) {
  const auto legal = board.GenerateLegalMoves();
  return std::find(legal.begin(), legal.end(), move) != legal.end();
}

bool IsLegalPv(ShogiBoard board, const std::vector<Move>& pv) {
  for (Move move : pv) {
    if (!IsLegalMove(board, move)) return false;
    board.DoMove(move);
  }
  return true;
}

void CheckResult(const char* prefix, const SearchResult& result,
                 const ShogiBoard& board, int minimum_nodes) {
  const std::string best_name = std::string(prefix) + " returns a legal move";
  const std::string nodes_name =
      std::string(prefix) + " completes the requested search";
  const std::string cp_name = std::string(prefix) + " returns a bounded CP";
  const std::string pv_name = std::string(prefix) + " returns a legal PV";
  Check(best_name.c_str(),
        !result.best_move.is_null() && IsLegalMove(board, result.best_move));
  Check(nodes_name.c_str(), result.nodes >= minimum_nodes);
  Check(cp_name.c_str(),
        result.score_cp >= -100000 && result.score_cp <= 100000);
  Check(pv_name.c_str(), IsLegalPv(board, result.pv));
}

void TestConcurrentSearchAndReuse() {
  ShogiBoard start;
  start.SetStartPos();
  const uint64_t starting_key = start.Hash();

  jhbr2::NNEvaluator evaluator("", false, 0, 32);
  SearchConfig config;
  config.workers_per_gpu = 32;
  config.minibatch_size = 32;
  config.max_nodes = 12000;
  config.max_time = 0.0f;
  config.leaf_mate_depth = 0;
  config.root_mate_depth = 0;
  config.nn_cache_size = 20000;
  config.info_interval_ms = 0;
  config.info_callback = [](const dlshogi_mcts::SearchInfo&) {};

  Search search({&evaluator}, config);
  const SearchResult first = search.Run(start, starting_key, {});
  CheckResult("lock-free search", first, start, config.max_nodes);
  Check("first search starts a fresh tree", !first.tree_reused);
  if (first.best_move.is_null() || !IsLegalMove(start, first.best_move)) return;

  ShogiBoard continued = start;
  continued.DoMove(first.best_move);
  const SearchResult reused =
      search.Run(continued, starting_key, {first.best_move});
  CheckResult("reused lock-free search", reused, continued, config.max_nodes);
  Check("continued search reuses the selected subtree", reused.tree_reused);

  search.PrepareForNewGame();
  const SearchResult fresh = search.Run(start, starting_key, {});
  CheckResult("new-game lock-free search", fresh, start, config.max_nodes);
  Check("new-game search does not reuse the old tree", !fresh.tree_reused);
  Check("new-game search begins without retained visits",
        fresh.root_visits_before == 0);
}

}  // namespace

int main() {
  lczero::ShogiTables::Init();
  TestConcurrentSearchAndReuse();
  std::printf("\n=== Lock-free MCTS search: %d failed ===\n", failures);
  return failures == 0 ? 0 : 1;
}
