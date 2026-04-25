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

#include "mate/dfpn.h"
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

Search::Search(NNEvaluator& evaluator, const SearchConfig& config,
               NNEvaluator* evaluator2)
    : backend_(evaluator2 ? Backend(evaluator, *evaluator2, config.num_threads)
                          : Backend(evaluator, config.num_threads)),
      config_(config) {}

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

    // Single legal move — no need to search.
    if (legal_moves.size() == 1) {
      SearchResult result;
      result.best_move = legal_moves[0];
      result.nodes = 0;
      return result;
    }

    // Evaluate root position (direct call, before GPU thread starts).
    auto root_comp = backend_.CreateComputation();
    root_comp->AddInput(board, legal_moves);
    backend_.EvalDirect(root_comp.get());

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

  // Start GPU thread for shared batching (multi-worker only).
  backend_.StartGPUThread();

  // Launch worker threads.
  std::vector<std::thread> threads;
  for (int t = 0; t < config_.num_threads; t++) {
    threads.emplace_back([this, t]() {
      SearchWorker worker(this, config_, t);
      worker.RunBlocking();
    });
  }

  // Wait for all workers, then stop GPU thread.
  for (auto& t : threads) t.join();
  backend_.StopGPUThread();

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

SearchWorker::SearchWorker(Search* search, const SearchConfig& config,
                           int worker_id)
    : search_(search), config_(config), worker_id_(worker_id) {}

void SearchWorker::RunBlocking() {
  do {
    ExecuteOneIteration();
  } while (search_->IsSearchActive());
}

void SearchWorker::ExecuteOneIteration() {
  minibatch_.clear();
  nn_batch_indices_.clear();
  computation_ = search_->backend_.CreateComputation(worker_id_);

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
// GatherMinibatch — lc0-style bulk visit distribution
// =====================================================================

void SearchWorker::GatherMinibatch() {
  if (config_.per_leaf_gathering) {
    GatherMinibatchPerLeaf();
  } else {
    GatherMinibatchBulk();
  }
}

// =====================================================================
// Per-leaf gathering — one PickNodeToExtend per leaf
// Better NPS with static-batch TensorRT engines.
// =====================================================================

void SearchWorker::GatherMinibatchPerLeaf() {
  int target = config_.minibatch_size;
  int max_collisions = target * 2;
  int collisions = 0;
  int gathered = 0;

  while (gathered < target && collisions < max_collisions) {
    if (search_->stop_.load(std::memory_order_acquire)) break;

    auto ntp = PickNodeToExtend();

    if (ntp.is_collision) {
      collisions += ntp.multivisit;
      minibatch_.push_back(std::move(ntp));
      continue;
    }

    ExtendNodeInPlace(ntp);

    if (!ntp.node->IsTerminal()) {
      ShogiBoard board = search_->root_board_;
      for (const auto& m : ntp.moves_to_node) board.DoMove(m);
      MoveList legal_moves = board.GenerateLegalMoves();
      nn_batch_indices_.push_back(static_cast<int>(minibatch_.size()));
      computation_->AddInput(board, legal_moves);
    }

    minibatch_.push_back(std::move(ntp));
    gathered++;
  }
}

SearchWorker::NodeToProcess SearchWorker::PickNodeToExtend() {
  Node* node = search_->root_node_;
  uint16_t depth = 0;
  std::vector<Move> moves;
  bool is_root = true;

  std::unique_lock<std::shared_mutex> lock(search_->nodes_mutex_);

  while (true) {
    if (node->GetN() == 0 || !node->HasChildren() || node->IsTerminal()) {
      if (node->TryStartScoreUpdate()) {
        return NodeToProcess::Visit(node, depth, std::move(moves));
      } else {
        node->IncrementNInFlight(1);
        return NodeToProcess::Collision(node, depth, 1);
      }
    }

    const float draw_score = search_->GetDrawScore(depth % 2 == 1);
    const float cpuct = ComputeCpuct(config_, node->GetN(), is_root);
    const float puct_mult =
        cpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));

    float visited_pol = 0.0f;
    for (auto* child : node->VisitedNodes()) {
      visited_pol += node->GetEdgeToNode(child)->GetP();
    }
    const float fpu = GetFpu(config_, node, is_root, draw_score, visited_pol);

    float best_score = -1e10f;
    Edge_Iterator<false> best_edge;

    for (auto edge = node->Edges(); edge != edge.end(); ++edge) {
      float q = (edge.HasNode() && edge.GetN() > 0)
                    ? edge.GetQ(fpu, draw_score) : fpu;
      if (!std::isfinite(q)) q = 0.0f;
      int n_started = (edge.HasNode()) ? edge.GetNStarted() : 0;
      float u = edge.GetP() * puct_mult / (1 + n_started);
      float score = q + u;
      if (!std::isfinite(score)) score = fpu;
      if (best_edge.edge() == nullptr || score > best_score) {
        best_score = score;
        best_edge = edge;
      }
    }

    if (!best_edge) {
      // No edges at all — shouldn't happen for expanded nodes.
      node->IncrementNInFlight(1);
      return NodeToProcess::Collision(node, depth, 1);
    }

    node->IncrementNInFlight(1);
    moves.push_back(best_edge.GetMove());
    Node* child = best_edge.GetOrSpawnNode(node);
    node = child;
    depth++;
    is_root = false;
  }
}

void SearchWorker::ExtendNodeInPlace(NodeToProcess& ntp) {
  Node* node = ntp.node;
  ShogiBoard board = search_->root_board_;
  for (const auto& m : ntp.moves_to_node) board.DoMove(m);

  auto legal_moves = board.GenerateLegalMoves();

  if (legal_moves.empty()) {
    node->MakeTerminal(GameResult::WHITE_WON);
    return;
  }
  if (board.CanDeclareWin()) {
    node->MakeTerminal(GameResult::BLACK_WON);
    return;
  }
  if (node != search_->root_node_) {
    auto rep = board.CheckRepetition();
    if (rep == ShogiBoard::RepetitionResult::kDraw) {
      node->MakeTerminal(GameResult::DRAW); return;
    } else if (rep == ShogiBoard::RepetitionResult::kWin) {
      node->MakeTerminal(GameResult::BLACK_WON); return;
    } else if (rep == ShogiBoard::RepetitionResult::kLoss) {
      node->MakeTerminal(GameResult::WHITE_WON); return;
    }
  }

  // Leaf df-pn: inline mate detection with tiny budget.
  if (config_.leaf_dfpn_nodes > 0) {
    jhbr2::MateDfpnSolver solver(config_.leaf_dfpn_nodes);
    Move mate_move = solver.search(board, config_.leaf_dfpn_nodes);
    if (!mate_move.is_null() && !jhbr2::MateDfpnSolver::IsNoMate(mate_move)) {
      // Side to move can force mate → this position is winning.
      node->MakeTerminal(GameResult::BLACK_WON);
      return;
    }
  }

  node->CreateEdges(legal_moves);
}

// =====================================================================
// Bulk gathering — lc0-style one recursive tree walk
// Better for dynamic-batch backends.
// =====================================================================

void SearchWorker::GatherMinibatchBulk() {
  int collision_limit = config_.minibatch_size;
  PickNodesToExtend(collision_limit);
  ProcessPickedNodes();
}

void SearchWorker::PickNodesToExtend(int collision_limit) {
  std::unique_lock<std::shared_mutex> lock(search_->nodes_mutex_);
  std::vector<Move> empty;
  PickNodesToExtendTask(search_->root_node_, 0, collision_limit,
                        empty, &minibatch_);
}

// =====================================================================
// PickNodesToExtendTask — ported from lc0 with minimal changes
// =====================================================================
// Distributes collision_limit visits down the tree in ONE recursive walk.
// At each internal node, PUCT determines how many visits each child gets.
// Leaves become Visit entries; nodes being expanded become Collisions.

void SearchWorker::PickNodesToExtendTask(
    Node* node, int base_depth, int collision_limit,
    const std::vector<Move>& moves_to_base,
    std::vector<NodeToProcess>* receiver) {

  // Workspace arrays (reused via vtp_buffer_).
  std::vector<std::unique_ptr<std::array<int, 256>>> visits_to_perform;
  std::vector<int> vtp_last_filled;
  std::vector<int> current_path;
  std::vector<Move> moves_to_path = moves_to_base;

  std::array<float, 256> current_pol;
  std::array<float, 256> current_util;
  std::array<float, 256> current_score;
  std::array<int, 256> current_nstarted;

  Node::Iterator best_edge;
  Node::Iterator second_best_edge;

  bool is_root_node = (node == search_->root_node_);
  const float even_draw_score = search_->GetDrawScore(false);
  const float odd_draw_score = search_->GetDrawScore(true);

  int max_limit = std::numeric_limits<int>::max();

  current_path.push_back(-1);
  while (current_path.size() > 0) {
    if (current_path.back() == -1) {
      // Determine how many visits this node should receive.
      int cur_limit = collision_limit;
      if (current_path.size() > 1) {
        cur_limit =
            (*visits_to_perform.back())[current_path[current_path.size() - 2]];
      }

      // Terminal or unexpanded node → collision.
      if (node->GetN() == 0 || node->IsTerminal()) {
        if (is_root_node) {
          if (node->TryStartScoreUpdate()) {
            cur_limit -= 1;
            auto ntp = NodeToProcess::Visit(
                node, static_cast<uint16_t>(current_path.size() + base_depth));
            ntp.moves_to_node = moves_to_path;
            minibatch_.push_back(std::move(ntp));
          }
        }
        if (cur_limit > 0) {
          if (is_root_node) node->IncrementNInFlight(cur_limit);
          receiver->push_back(NodeToProcess::Collision(
              node, static_cast<uint16_t>(current_path.size() + base_depth),
              cur_limit, 0));
        }
        node = node->GetParent();
        current_path.pop_back();
        continue;
      }

      // Root VL: increment n_in_flight for all visits being distributed.
      if (is_root_node) {
        node->IncrementNInFlight(cur_limit);
      }

      // Allocate visits_to_perform array for this depth level.
      if (vtp_buffer_.size() > 0) {
        visits_to_perform.push_back(std::move(vtp_buffer_.back()));
        vtp_buffer_.pop_back();
      } else {
        visits_to_perform.push_back(std::make_unique<std::array<int, 256>>());
      }
      vtp_last_filled.push_back(-1);

      // Cache PUCT parameters.
      int max_needed = node->GetNumEdges();
      max_needed =
          std::min(max_needed, std::max(0, node->GetNStarted()) + cur_limit + 2);
      if (max_needed <= 0) {
        receiver->push_back(NodeToProcess::Collision(
            node, static_cast<uint16_t>(current_path.size() + base_depth),
            cur_limit, 0));
        node = node->GetParent();
        current_path.pop_back();
        continue;
      }
      node->CopyPolicy(max_needed, current_pol.data());
      for (int i = 0; i < max_needed; i++) {
        current_util[i] = std::numeric_limits<float>::lowest();
      }

      const float draw_score = ((current_path.size() + base_depth) % 2 == 0)
                                   ? odd_draw_score : even_draw_score;
      float visited_pol = 0.0f;
      for (Node* child : node->VisitedNodes()) {
        int index = child->Index();
        visited_pol += current_pol[index];
        float child_q = child->GetQ(draw_score);
        current_util[index] = std::isfinite(child_q) ? child_q : 0.0f;
      }
      float fpu = GetFpu(config_, node, is_root_node, draw_score, visited_pol);
      if (!std::isfinite(fpu)) fpu = 0.0f;
      for (int i = 0; i < max_needed; i++) {
        if (current_util[i] == std::numeric_limits<float>::lowest()) {
          current_util[i] = fpu;
        }
      }

      const float cpuct = ComputeCpuct(config_, node->GetN(), is_root_node);
      const float puct_mult =
          cpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));
      int cache_filled_idx = -1;

      // Distribute visits to children via PUCT.
      while (cur_limit > 0) {
        float best = std::numeric_limits<float>::lowest();
        int best_idx = -1;
        float best_without_u = std::numeric_limits<float>::lowest();
        float second_best = std::numeric_limits<float>::lowest();
        bool can_exit = false;
        best_edge.Reset();

        for (int idx = 0; idx < max_needed; ++idx) {
          if (idx > cache_filled_idx) {
            if (idx == 0) {
              cur_iters_[idx] = node->Edges();
            } else {
              cur_iters_[idx] = cur_iters_[idx - 1];
              ++cur_iters_[idx];
            }
            current_nstarted[idx] = cur_iters_[idx].GetNStarted();
          }
          int nstarted = current_nstarted[idx];
          const float util = current_util[idx];
          if (idx > cache_filled_idx) {
            current_score[idx] =
                current_pol[idx] * puct_mult / (1 + nstarted) + util;
            cache_filled_idx++;
          }

          float score = current_score[idx];
          if (!std::isfinite(score)) {
            score = std::numeric_limits<float>::lowest();
          }
          if (best_idx < 0 || score > best) {
            second_best = best;
            second_best_edge = best_edge;
            best = score;
            best_idx = idx;
            best_without_u = util;
            best_edge = cur_iters_[idx];
          } else if (score > second_best) {
            second_best = score;
            second_best_edge = cur_iters_[idx];
          }
          if (can_exit) break;
          if (nstarted == 0) can_exit = true;
        }

        if (best_idx < 0 || !best_edge) {
          receiver->push_back(NodeToProcess::Collision(
              node, static_cast<uint16_t>(current_path.size() + base_depth),
              cur_limit, 0));
          cur_limit = 0;
          break;
        }

        // Determine how many visits go to best child before second-best
        // becomes better.
        int new_visits = 0;
        if (second_best_edge) {
          int estimated_visits_to_change_best = std::numeric_limits<int>::max();
          if (best_without_u < second_best) {
            const auto n1 = current_nstarted[best_idx] + 1;
            estimated_visits_to_change_best = static_cast<int>(
                std::max(1.0f, std::min(current_pol[best_idx] * puct_mult /
                                                (second_best - best_without_u) -
                                            n1 + 1,
                                        1e9f)));
          }
          second_best_edge.Reset();
          max_limit = std::min(max_limit, estimated_visits_to_change_best);
          new_visits = std::min(cur_limit, estimated_visits_to_change_best);
        } else {
          new_visits = cur_limit;
        }

        // Record visits for this child.
        if (best_idx >= vtp_last_filled.back()) {
          auto* vtp_array = visits_to_perform.back().get()->data();
          std::fill(vtp_array + (vtp_last_filled.back() + 1),
                    vtp_array + best_idx + 1, 0);
        }
        (*visits_to_perform.back())[best_idx] += new_visits;
        cur_limit -= new_visits;

        Node* child_node = best_edge.GetOrSpawnNode(node);

        // Try to claim this child for expansion.
        bool claimed = false;
        if (child_node->TryStartScoreUpdate()) {
          current_nstarted[best_idx]++;
          new_visits -= 1;
          claimed = true;
        }
        // Collision cancellation subtracts from the collision node too, so
        // every non-claimed visit must be represented in the child VL count.
        if (new_visits > 0) {
          child_node->IncrementNInFlight(new_visits);
          current_nstarted[best_idx] += new_visits;
        }
        current_score[best_idx] = current_pol[best_idx] * puct_mult /
                                      (1 + current_nstarted[best_idx]) +
                                  current_util[best_idx];

        // If claimed and it's a new leaf or terminal, create a Visit entry.
        if (claimed &&
            (child_node->GetN() == 0 || child_node->IsTerminal())) {
          (*visits_to_perform.back())[best_idx] -= 1;
          auto ntp = NodeToProcess::Visit(
              child_node,
              static_cast<uint16_t>(current_path.size() + 1 + base_depth));
          ntp.moves_to_node = moves_to_path;
          ntp.moves_to_node.push_back(best_edge.GetMove());
          receiver->push_back(std::move(ntp));
        }

        if (best_idx > vtp_last_filled.back() &&
            (*visits_to_perform.back())[best_idx] > 0) {
          vtp_last_filled.back() = best_idx;
        }
      }
      is_root_node = false;
      // Fall through to select the first child to recurse into.
    }

    // Find next child with visits_to_perform > 0 to recurse into.
    int min_idx = current_path.back();
    bool found_child = false;
    if (vtp_last_filled.back() > min_idx) {
      int idx = -1;
      for (auto& child : node->Edges()) {
        idx++;
        if (idx > min_idx && (*visits_to_perform.back())[idx] > 0) {
          if (moves_to_path.size() !=
              current_path.size() + static_cast<size_t>(base_depth)) {
            moves_to_path.push_back(child.GetMove());
          } else {
            moves_to_path.back() = child.GetMove();
          }
          current_path.back() = idx;
          current_path.push_back(-1);
          node = child.GetOrSpawnNode(node);
          found_child = true;
          break;
        }
        if (idx >= vtp_last_filled.back()) break;
      }
    }
    if (!found_child) {
      node = node->GetParent();
      if (!moves_to_path.empty()) moves_to_path.pop_back();
      current_path.pop_back();
      vtp_buffer_.push_back(std::move(visits_to_perform.back()));
      visits_to_perform.pop_back();
      vtp_last_filled.pop_back();
    }
  }
}

// =====================================================================
// ProcessPickedNodes — extend nodes + add to NN batch
// =====================================================================

void SearchWorker::ProcessPickedNodes() {
  for (int i = 0; i < static_cast<int>(minibatch_.size()); i++) {
    auto& ntp = minibatch_[i];
    if (ntp.IsCollision()) continue;
    if (!ntp.IsExtendable()) continue;

    // Extend the node.
    ExtendNode(ntp.node, ntp.depth, ntp.moves_to_node);

    if (!ntp.node->IsTerminal()) {
      // Add to NN batch.
      ntp.nn_queried = true;
      ShogiBoard board = search_->root_board_;
      for (const auto& m : ntp.moves_to_node) {
        board.DoMove(m);
      }
      MoveList legal_moves = board.GenerateLegalMoves();
      nn_batch_indices_.push_back(i);
      computation_->AddInput(board, legal_moves);
    }
  }
}

// =====================================================================
// ExtendNode — terminal detection and edge creation (from lc0)
// =====================================================================

void SearchWorker::ExtendNode(Node* node, int depth,
                              const std::vector<Move>& moves) {
  ShogiBoard board = search_->root_board_;
  for (const auto& m : moves) board.DoMove(m);

  auto legal_moves = board.GenerateLegalMoves();

  if (legal_moves.empty()) {
    node->MakeTerminal(GameResult::WHITE_WON);
    return;
  }

  if (board.CanDeclareWin()) {
    node->MakeTerminal(GameResult::BLACK_WON);
    return;
  }

  if (node != search_->root_node_) {
    auto rep = board.CheckRepetition();
    if (rep == ShogiBoard::RepetitionResult::kDraw) {
      node->MakeTerminal(GameResult::DRAW);
      return;
    } else if (rep == ShogiBoard::RepetitionResult::kWin) {
      node->MakeTerminal(GameResult::BLACK_WON);
      return;
    } else if (rep == ShogiBoard::RepetitionResult::kLoss) {
      node->MakeTerminal(GameResult::WHITE_WON);
      return;
    }
  }

  // Leaf df-pn: inline mate detection with tiny budget.
  if (config_.leaf_dfpn_nodes > 0) {
    jhbr2::MateDfpnSolver solver(config_.leaf_dfpn_nodes);
    Move mate_move = solver.search(board, config_.leaf_dfpn_nodes);
    if (!mate_move.is_null() && !jhbr2::MateDfpnSolver::IsNoMate(mate_move)) {
      node->MakeTerminal(GameResult::BLACK_WON);
      return;
    }
  }

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
