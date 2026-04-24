/*
  JHBR2 Shogi Engine — lc0-style MCTS Search (simplified port)

  Algorithms ported from lc0/src/search/classic/search.cc
  Copyright (C) 2018-2023 The LCZero Authors (GPL v3)

  Key algorithms preserved:
    - PUCT selection with dynamic cpuct (ComputeCpuct)
    - FPU with relative reduction
    - FinalizeScoreUpdate with running-average Q
    - TryStartScoreUpdate / CancelScoreUpdate for collision handling
    - n_in_flight_ separate from n_ (proper virtual loss)
    - Batch NN evaluation with multi-threaded gathering
    - Dirichlet noise at root
    - Sticky endgames (bounds propagation)
    - Solid children optimization

  Simplified:
    - No Syzygy, no WDL rescaling, no OOO eval, no task parallelization
    - Uses our NNEvaluator and ShogiBoard directly
*/

#include "lc0_mcts/search.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>

#include "shogi/encoder.h"

namespace lc0_shogi {

using namespace lczero;

// =====================================================================
// Helpers (from lc0)
// =====================================================================

namespace {

inline float ComputeCpuct(const SearchConfig& config, uint32_t N,
                          bool is_root) {
  const float init = config.cpuct;
  const float k = config.cpuct_factor;
  const float base = config.cpuct_base;
  return init + (k ? k * std::log((N + base) / base) : 0.0f);
}

inline float GetFpu(const SearchConfig& config, const Node* node,
                    bool is_root, float draw_score, float visited_pol) {
  const float value = is_root ? config.fpu_value_at_root : config.fpu_value;
  return -node->GetQ(-draw_score) - value * std::sqrt(visited_pol);
}

void ApplyDirichletNoise(Node* node, float epsilon, float alpha) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::gamma_distribution<float> gamma(alpha, 1.0f);

  std::vector<float> noise;
  float noise_sum = 0.0f;
  for (int i = 0; i < node->GetNumEdges(); i++) {
    float n = gamma(gen);
    noise.push_back(n);
    noise_sum += n;
  }
  if (noise_sum < 1e-10f) return;
  for (auto& n : noise) n /= noise_sum;

  int i = 0;
  for (auto& edge : node->Edges()) {
    float p = edge.GetP();
    edge.edge()->SetP(p * (1.0f - epsilon) + noise[i] * epsilon);
    i++;
  }
}

}  // namespace

// =====================================================================
// Search
// =====================================================================

Search::Search(NNEvaluator& evaluator, const SearchConfig& config)
    : backend_(evaluator), config_(config) {}

Search::~Search() = default;

SearchResult Search::Run(ShogiBoard board, int game_ply) {
  stop_.store(false, std::memory_order_release);
  start_time_ = std::chrono::steady_clock::now();

  root_board_ = board;
  root_is_black_to_move_ = (board.side_to_move() == lczero::WHITE);

  // Reset tree to position (no tree reuse for now).
  tree_.ResetToPosition(board, {});
  root_node_ = tree_.GetCurrentHead();

  // Expand root if not already expanded.
  if (!root_node_->HasChildren()) {
    MoveList legal_moves = board.GenerateLegalMoves();
    if (legal_moves.empty()) {
      SearchResult result;
      result.best_move = Move();
      return result;
    }

    // Evaluate root position.
    auto root_comp = backend_.CreateComputation();
    root_comp->AddInput(board, legal_moves);
    root_comp->ComputeBlocking();

    NNOutput root_eval;
    root_eval.value = root_comp->GetQ(0);
    root_eval.draw = root_comp->GetD(0);
    root_eval.policy = root_comp->GetPolicy(0);

    // Create edges with policy priors.
    root_node_->CreateEdges(legal_moves);
    int i = 0;
    for (auto& edge : root_node_->Edges()) {
      edge.edge()->SetP(root_eval.policy[i]);
      i++;
    }
    root_node_->SortEdges();

    // Initialize root with NN value.
    root_node_->FinalizeScoreUpdate(root_eval.value, root_eval.draw, 0.0f, 1);

    // Add Dirichlet noise if configured.
    if (config_.noise_epsilon > 0.0f) {
      ApplyDirichletNoise(root_node_, config_.noise_epsilon,
                          config_.noise_alpha);
    }
  }

  // Initialize stats.
  total_playouts_ = 0;
  total_batches_ = 0;
  max_depth_ = 0;
  shared_collisions_.clear();
  current_best_edge_ = GetBestChildNoTemperature(root_node_);

  // Launch worker threads.
  std::vector<std::thread> threads;
  for (int t = 0; t < config_.num_threads; t++) {
    threads.emplace_back([this]() {
      SearchWorker worker(this, config_);
      worker.RunBlocking();
    });
  }

  // Wait for all threads.
  for (auto& t : threads) t.join();

  // Collect result.
  auto t1 = std::chrono::steady_clock::now();
  float elapsed = std::chrono::duration<float>(t1 - start_time_).count();

  SearchResult result;
  auto best = GetBestChildNoTemperature(root_node_);
  result.best_move = best.GetMove();
  result.nodes = total_playouts_;
  result.time_sec = elapsed;
  result.nps = elapsed > 0.001f ? total_playouts_ / elapsed : 0.0f;
  result.root_q = root_node_->GetWL();
  result.root_d = root_node_->GetD();
  result.score_cp = QToCentipawns(root_node_->GetWL());
  result.pv = GetPV(root_node_);

  return result;
}

bool Search::IsSearchActive() const {
  if (stop_.load(std::memory_order_acquire)) return false;

  {
    std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
    if (total_playouts_ >= config_.max_nodes) return false;
  }

  if (config_.max_time > 0.0f) {
    float elapsed = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - start_time_).count();
    if (elapsed >= config_.max_time) return false;
  }

  return true;
}

EdgeAndNode Search::GetBestChildNoTemperature(Node* parent) const {
  EdgeAndNode best;
  for (auto& edge : parent->Edges()) {
    if (edge.GetN() > best.GetN()) {
      best = edge;
    }
  }
  return best;
}

float Search::GetDrawScore(bool is_odd_depth) const {
  return (is_odd_depth == root_is_black_to_move_)
             ? config_.draw_score
             : -config_.draw_score;
}

void Search::CancelSharedCollisions() {
  for (auto& [node, count] : shared_collisions_) {
    for (Node* n = node; n != root_node_->GetParent(); n = n->GetParent()) {
      n->CancelScoreUpdate(count);
    }
  }
  shared_collisions_.clear();
}

std::vector<Move> Search::GetPV(Node* root) const {
  std::vector<Move> pv;
  Node* node = root;
  while (node->HasChildren()) {
    EdgeAndNode best;
    for (auto& edge : node->Edges()) {
      if (edge.GetN() > best.GetN()) best = edge;
    }
    if (!best || best.GetN() == 0) break;
    pv.push_back(best.GetMove());
    node = best.node();
    if (!node) break;
  }
  return pv;
}

int Search::QToCentipawns(float q) {
  if (q >= 1.0f) return 10000;
  if (q <= -1.0f) return -10000;
  return static_cast<int>(290.680623072 * std::tan(1.5620688421 * q));
}

// =====================================================================
// SearchWorker
// =====================================================================

SearchWorker::SearchWorker(Search* search, const SearchConfig& config)
    : search_(search), config_(config) {}

void SearchWorker::RunBlocking() {
  do {
    ExecuteOneIteration();
  } while (search_->IsSearchActive());
}

void SearchWorker::ExecuteOneIteration() {
  minibatch_.clear();
  nn_batch_indices_.clear();
  computation_ = search_->backend_.CreateComputation();

  // 1. Gather minibatch.
  GatherMinibatch();

  // 2. Run NN computation.
  RunNNComputation();

  // 3. Fetch results.
  FetchMinibatchResults();

  // 4. Backup.
  DoBackupUpdate();
}

// =====================================================================
// GatherMinibatch
// =====================================================================

void SearchWorker::GatherMinibatch() {
  int target = config_.minibatch_size;
  int collisions_left = config_.max_collision_visits;
  int gathered = 0;

  while (gathered < target && collisions_left > 0) {
    if (search_->stop_.load(std::memory_order_acquire)) break;

    auto ntp = PickNodeToExtend();

    if (ntp.is_collision) {
      collisions_left -= ntp.multivisit;
      minibatch_.push_back(std::move(ntp));
      continue;
    }

    // Extend the node (generate legal moves, detect terminal).
    ExtendNode(ntp);

    if (!ntp.node->IsTerminal()) {
      // Needs NN evaluation — add to computation batch.
      ShogiBoard board = search_->root_board_;
      for (const auto& m : ntp.moves_to_node) {
        board.DoMove(m);
      }
      MoveList legal_moves = board.GenerateLegalMoves();

      nn_batch_indices_.push_back(static_cast<int>(minibatch_.size()));
      computation_->AddInput(board, legal_moves);
    }

    minibatch_.push_back(std::move(ntp));
    gathered++;
  }
}

// =====================================================================
// PickNodeToExtend — PUCT selection (from lc0)
// =====================================================================

SearchWorker::NodeToProcess SearchWorker::PickNodeToExtend() {
  Node* node = search_->root_node_;
  uint16_t depth = 0;
  std::vector<Move> moves;
  bool is_root = true;

  // Lock tree for reading/writing.
  std::unique_lock<std::shared_mutex> lock(search_->nodes_mutex_);

  while (true) {
    // If node is not expanded or is terminal, we've reached a leaf.
    if (!node->HasChildren() || node->IsTerminal()) {
      // Try to claim this node for expansion.
      if (node->TryStartScoreUpdate()) {
        return NodeToProcess::Visit(node, depth, std::move(moves));
      } else {
        // Collision — another thread is expanding this node.
        return NodeToProcess::Collision(node, depth, 1);
      }
    }

    // PUCT selection among children.
    const float draw_score = search_->GetDrawScore(depth % 2 == 1);
    const float cpuct = ComputeCpuct(config_, node->GetN(), is_root);
    const float puct_mult =
        cpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));

    // Compute FPU.
    float visited_pol = 0.0f;
    for (auto* child : node->VisitedNodes()) {
      visited_pol += node->GetEdgeToNode(child)->GetP();
    }
    const float fpu = GetFpu(config_, node, is_root, draw_score, visited_pol);

    // Find best child by PUCT score.
    float best_score = -1e10f;
    Edge_Iterator<false> best_edge;

    for (auto edge = node->Edges(); edge != edge.end(); ++edge) {
      float q, u;
      if (edge.HasNode() && edge.GetN() > 0) {
        q = edge.GetQ(fpu, draw_score);
        u = edge.GetP() * puct_mult / (1 + edge.GetNStarted());
      } else {
        q = fpu;
        int n_started = edge.HasNode() ? edge.GetNStarted() : 0;
        u = edge.GetP() * puct_mult / (1 + n_started);
      }

      float score = q + u;
      if (score > best_score) {
        best_score = score;
        best_edge = edge;
      }
    }

    if (!best_edge) {
      // No valid edge found — shouldn't happen.
      return NodeToProcess::Collision(node, depth, 1);
    }

    // Add virtual loss to this node for other threads.
    node->IncrementNInFlight(1);

    // Move to the selected child.
    moves.push_back(best_edge.GetMove());
    Node* child = best_edge.GetOrSpawnNode(node);
    node = child;
    depth++;
    is_root = false;
  }
}

// =====================================================================
// ExtendNode — terminal detection and edge creation (from lc0)
// =====================================================================

void SearchWorker::ExtendNode(NodeToProcess& ntp) {
  Node* node = ntp.node;

  // Reconstruct board position.
  ShogiBoard board = search_->root_board_;
  for (const auto& m : ntp.moves_to_node) {
    board.DoMove(m);
  }

  auto legal_moves = board.GenerateLegalMoves();

  // No legal moves = checkmate (shogi has no stalemate).
  if (legal_moves.empty()) {
    // Side to move is checkmated. From lc0 convention:
    // WHITE_WON = the side that "just moved" wins = current side loses.
    // So checkmate = side to move loses.
    node->MakeTerminal(GameResult::WHITE_WON);
    return;
  }

  // Declare win (entering king).
  if (board.CanDeclareWin()) {
    node->MakeTerminal(GameResult::BLACK_WON);
    return;
  }

  // Repetition check.
  if (node != search_->root_node_) {
    auto rep = board.CheckRepetition();
    if (rep == ShogiBoard::RepetitionResult::kDraw) {
      node->MakeTerminal(GameResult::DRAW);
      return;
    } else if (rep == ShogiBoard::RepetitionResult::kWin) {
      // Current side wins (opponent made perpetual check).
      node->MakeTerminal(GameResult::BLACK_WON);
      return;
    } else if (rep == ShogiBoard::RepetitionResult::kLoss) {
      // Current side loses (we made perpetual check).
      node->MakeTerminal(GameResult::WHITE_WON);
      return;
    }
  }

  // Not terminal — create edges for legal moves.
  node->CreateEdges(legal_moves);
}

// =====================================================================
// RunNNComputation — batch evaluation
// =====================================================================

void SearchWorker::RunNNComputation() {
  if (computation_->UsedBatchSize() == 0) return;
  computation_->ComputeBlocking();  // Internally serialized via Backend mutex.
}

// =====================================================================
// FetchMinibatchResults — populate nodes with NN output
// =====================================================================

void SearchWorker::FetchMinibatchResults() {
  // Map NN results to minibatch entries.
  for (int b = 0; b < static_cast<int>(nn_batch_indices_.size()); b++) {
    int mb_idx = nn_batch_indices_[b];
    auto& ntp = minibatch_[mb_idx];
    const auto& policy = computation_->GetPolicy(b);

    // Set policy priors on edges.
    int p_idx = 0;
    for (auto& edge : ntp.node->Edges()) {
      if (p_idx < static_cast<int>(policy.size())) {
        edge.edge()->SetP(policy[p_idx]);
      }
      p_idx++;
    }

    // Sort edges by policy (lc0 does this).
    ntp.node->SortEdges();

    // Add Dirichlet noise at root.
    if (config_.noise_epsilon > 0.0f && ntp.node == search_->root_node_) {
      ApplyDirichletNoise(ntp.node, config_.noise_epsilon, config_.noise_alpha);
    }

    // Store eval for backup. Negate Q (NN returns from side-to-move perspective,
    // but backup expects from "just moved" perspective like lc0).
    ntp.eval_q = -computation_->GetQ(b);
    ntp.eval_d = computation_->GetD(b);
    ntp.eval_m = computation_->GetM(b);
    ntp.nn_queried = true;
  }

  // For terminal nodes, use stored values.
  for (auto& ntp : minibatch_) {
    if (ntp.is_collision) continue;
    if (!ntp.nn_queried) {
      ntp.eval_q = ntp.node->GetWL();
      ntp.eval_d = ntp.node->GetD();
      ntp.eval_m = ntp.node->GetM();
    }
  }
}

// =====================================================================
// DoBackupUpdate — propagate values up the tree (from lc0)
// =====================================================================

void SearchWorker::DoBackupUpdate() {
  std::unique_lock<std::shared_mutex> lock(search_->nodes_mutex_);

  bool work_done = false;
  for (const auto& ntp : minibatch_) {
    DoBackupUpdateSingleNode(ntp);
    if (!ntp.is_collision) work_done = true;
  }
  if (!work_done) return;

  search_->CancelSharedCollisions();
  search_->total_batches_++;
}

void SearchWorker::DoBackupUpdateSingleNode(const NodeToProcess& ntp) {
  Node* node = ntp.node;

  if (ntp.is_collision) {
    // Collisions are handled via shared_collisions.
    search_->shared_collisions_.emplace_back(node, ntp.multivisit);
    return;
  }

  auto update_parent_bounds =
      config_.sticky_endgames && node->IsTerminal() && !node->GetN();

  float v = ntp.eval_q;
  float d = ntp.eval_d;
  float m = ntp.eval_m;
  int n_to_fix = 0;
  float v_delta = 0.0f, d_delta = 0.0f, m_delta = 0.0f;
  uint32_t solid_threshold = static_cast<uint32_t>(config_.solid_threshold);

  // Walk from leaf up to root (lc0's backup loop).
  for (Node* n = node; n != search_->root_node_->GetParent();
       n = n->GetParent()) {
    Node* p = n->GetParent();

    // If node became terminal from another path, use its values.
    if (n->IsTerminal()) {
      v = n->GetWL();
      d = n->GetD();
      m = n->GetM();
    }

    // Update this node with running-average Q.
    n->FinalizeScoreUpdate(v, d, m, ntp.multivisit);

    // Adjust for terminal bounds discovered by other paths.
    if (n_to_fix > 0 && !n->IsTerminal()) {
      n->AdjustForTerminal(v_delta, d_delta, m_delta, n_to_fix);
    }

    // Solidify if enough visits.
    if (n->GetN() >= solid_threshold) {
      if (n->MakeSolid() && n == search_->root_node_) {
        search_->current_best_edge_ =
            search_->GetBestChildNoTemperature(search_->root_node_);
      }
    }

    if (!p) break;

    // Try setting parent bounds for sticky endgames.
    if (p->IsTerminal()) n_to_fix = 0;
    bool old_update = update_parent_bounds;
    update_parent_bounds =
        update_parent_bounds && p != search_->root_node_ && !p->IsTerminal() &&
        MaybeSetBounds(p, m, &n_to_fix, &v_delta, &d_delta, &m_delta);

    // Flip for opponent.
    v = -v;
    v_delta = -v_delta;
    m++;

    // Update best edge at root.
    if (p == search_->root_node_ &&
        ((old_update && n->IsTerminal()) ||
         (n != search_->current_best_edge_.node() &&
          search_->current_best_edge_.GetN() <= n->GetN()))) {
      search_->current_best_edge_ =
          search_->GetBestChildNoTemperature(search_->root_node_);
    }
  }

  search_->total_playouts_ += ntp.multivisit;
  search_->max_depth_ = std::max(search_->max_depth_, ntp.depth);
}

bool SearchWorker::MaybeSetBounds(Node* p, float m, int* n_to_fix,
                                  float* v_delta, float* d_delta,
                                  float* m_delta) const {
  auto losing_m = 0.0f;
  auto lower = GameResult::BLACK_WON;
  auto upper = GameResult::BLACK_WON;

  for (const auto& edge : p->Edges()) {
    const auto [edge_lower, edge_upper] = edge.GetBounds();
    lower = std::max(edge_lower, lower);
    upper = std::max(edge_upper, upper);

    if (edge_lower == GameResult::WHITE_WON) break;
    if (edge_upper == GameResult::BLACK_WON) {
      losing_m = std::max(losing_m, edge.GetM(0.0f));
    }
  }

  // Flip bounds for parent perspective.
  GameResult parent_lower =
      (upper == GameResult::WHITE_WON)  ? GameResult::BLACK_WON
      : (upper == GameResult::DRAW)     ? GameResult::DRAW
                                        : GameResult::WHITE_WON;
  GameResult parent_upper =
      (lower == GameResult::WHITE_WON)  ? GameResult::BLACK_WON
      : (lower == GameResult::DRAW)     ? GameResult::DRAW
                                        : GameResult::WHITE_WON;

  if (parent_lower == p->GetBounds().first &&
      parent_upper == p->GetBounds().second) {
    return false;  // No change.
  }

  p->SetBounds(parent_lower, parent_upper);

  // If terminal, adjust visits.
  if (parent_lower == parent_upper) {
    float v, d;
    if (parent_lower == GameResult::WHITE_WON) {
      v = 1.0f;
      d = 0.0f;
    } else if (parent_lower == GameResult::BLACK_WON) {
      v = -1.0f;
      d = 0.0f;
      m = losing_m;
    } else {
      v = 0.0f;
      d = 1.0f;
    }
    p->MakeTerminal(parent_lower, m + 1);
    *n_to_fix = p->GetN();
    *v_delta = v - p->GetWL();
    *d_delta = d - p->GetD();
    *m_delta = (m + 1) - p->GetM();
    return true;
  }

  return false;
}

}  // namespace lc0_shogi
