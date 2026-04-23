/*
  JHBR2 Shogi Engine — MCTS Search Implementation

  References:
    - lc0 src/mcts/search.cc (PUCT formula, FPU)
    - dlshogi UctSearch.cpp:649-683 (leaf df-pn)
    - dlshogi UctSearch.cpp:772-906 (UCB with mate)
    - dlshogi PvMateSearch.cpp:48-181 (PV mate search)
    - JHBR2/shogi_mcts.py (Python prototype)
*/

#include "mcts/search.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <thread>
#include <chrono>
#include <cmath>
#include <random>

#include "mate/dfpn.h"

namespace jhbr2 {

using namespace lczero;

// =====================================================================
// Constructor / Destructor
// =====================================================================

MCTSSearch::MCTSSearch(NNEvaluator& evaluator, const MCTSConfig& config,
                       NNEvaluator* warmup_evaluator)
    : evaluator_(evaluator), warmup_evaluator_(warmup_evaluator), config_(config) {
  if (config_.leaf_dfpn_nodes > 0) {
    dfpn_leaf_ = std::make_unique<MateDfpnSolver>(config_.leaf_dfpn_nodes);
  }
  if (config_.pv_dfpn_nodes > 0) {
    dfpn_pv_ = std::make_unique<MateDfpnSolver>(config_.pv_dfpn_nodes);
  }
}

MCTSSearch::~MCTSSearch() = default;

// =====================================================================
// Main search loop
// =====================================================================

SearchResult MCTSSearch::Search(ShogiBoard board, int game_ply) {
  stop_ = false;

  MoveList legal_moves = board.GenerateLegalMoves();
  SearchResult result;

  // No legal moves = checkmate.
  if (legal_moves.empty()) {
    result.best_move = Move();
    result.mate_status = -1;
    return result;
  }

  // Single legal move = forced.
  if (legal_moves.size() == 1) {
    result.best_move = legal_moves[0];
    result.nodes = 0;
    return result;
  }

  // Check for mate at root before spending time on NN eval.
  // Tier 1: Mate-in-1 (essentially free).
  {
    Move mate1 = Mate1Ply(board);
    if (!mate1.is_null()) {
      result.best_move = mate1;
      result.mate_status = 1;
      result.nodes = 0;
      return result;
    }
  }
  // Tier 2: df-pn mate search at root (finds deeper forced mates).
  if (dfpn_leaf_) {
    Move mate_move = dfpn_leaf_->search(board, config_.leaf_dfpn_nodes);
    if (!mate_move.is_null() && !MateDfpnSolver::IsNoMate(mate_move)) {
      result.best_move = mate_move;
      result.mate_status = 1;
      result.nodes = 0;
      return result;
    }
  }

  // Evaluate root position.
  NNOutput root_eval = evaluator_.Evaluate(board, legal_moves);

  // Create root node.
  auto root = std::make_unique<Node>();
  std::vector<std::pair<Move, float>> move_priors;
  move_priors.reserve(legal_moves.size());
  for (size_t i = 0; i < legal_moves.size(); i++) {
    move_priors.emplace_back(legal_moves[i], root_eval.policy[i]);
  }
  root->Expand(move_priors);
  root->SetFirstEval(root_eval.value, root_eval.draw);

  // Add Dirichlet noise at root.
  if (config_.noise_epsilon > 0.0f) {
    AddDirichletNoise(root.get());
  }

  // Tree warmup: build initial tree with batch GPU eval before multi-threaded search.
  auto pre_search_start = std::chrono::steady_clock::now();
  if (config_.warmup_nodes > 0) {
    WarmupTree(root.get(), board, config_.warmup_nodes);
  }
  // Subtract warmup + root eval time from the MCTS time budget.
  if (config_.max_time > 0.0f) {
    float pre_search_time = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - pre_search_start).count();
    config_.max_time = std::max(config_.max_time - pre_search_time, 0.1f);
  }

  // Route to multi-threaded search if configured.
  if (config_.num_search_threads > 1) {
    return SearchAsync(board, game_ply, root, legal_moves);
  }

  // --- Single-threaded batch search ---
  // One thread selects multiple leaves, sends batch to GPU, expands all.
  // No concurrency = no data races. GPU fully utilized via batching.
  auto t0 = std::chrono::steady_clock::now();
  int nodes_expanded = 0;
  const int batch_target = config_.minibatch_size;

  while (nodes_expanded < config_.max_nodes && !stop_) {
    // Check time limit.
    if (config_.max_time > 0.0f) {
      float elapsed = std::chrono::duration<float>(
          std::chrono::steady_clock::now() - t0).count();
      if (elapsed >= config_.max_time) break;
    }

    // --- Gather phase: select up to batch_target leaves ---
    struct PendingLeaf {
      Node* node;
      ShogiBoard board;
      MoveList legal_moves;
    };
    std::vector<PendingLeaf> pending;

    for (int g = 0; g < batch_target && nodes_expanded + (int)pending.size() < config_.max_nodes; g++) {
      auto sel = Select(root.get(), board);
      Node* leaf = sel.node;

      // Terminal: backpropagate immediately.
      if (leaf->is_terminal()) {
        Backpropagate(leaf, leaf->terminal_v(), leaf->terminal_d());
        continue;
      }

      ShogiBoard& leaf_board = sel.board;
      MoveList leaf_legal = leaf_board.GenerateLegalMoves();

      // Checkmate.
      if (leaf_legal.empty()) {
        leaf->SetTerminal(-1.0f);
        leaf->set_mate_status(-1);
        if (leaf->parent() && leaf->parent_edge_idx() >= 0)
          leaf->parent()->edge(leaf->parent_edge_idx()).SetWin();
        Backpropagate(leaf, -1.0f, 0.0f);
        PropagateMateUp(leaf);
        continue;
      }

      // Mate-in-1.
      Move mate1 = Mate1Ply(leaf_board);
      if (!mate1.is_null()) {
        leaf->SetTerminal(1.0f);
        leaf->set_mate_status(1);
        if (leaf->parent() && leaf->parent_edge_idx() >= 0)
          leaf->parent()->edge(leaf->parent_edge_idx()).SetLose();
        Backpropagate(leaf, 1.0f, 0.0f);
        PropagateMateUp(leaf);
        continue;
      }

      // Leaf df-pn.
      if (dfpn_leaf_ && !leaf->dfpn_checked()) {
        leaf->set_dfpn_checked(true);
        Move mate_move = dfpn_leaf_->search(leaf_board, config_.leaf_dfpn_nodes);
        if (MateDfpnSolver::IsNoMate(mate_move)) {
          leaf->set_dfpn_proven_no_mate(true);
        } else if (!mate_move.is_null()) {
          leaf->SetTerminal(1.0f);
          leaf->set_mate_status(1);
          if (leaf->parent() && leaf->parent_edge_idx() >= 0)
            leaf->parent()->edge(leaf->parent_edge_idx()).SetLose();
          Backpropagate(leaf, 1.0f, 0.0f);
          PropagateMateUp(leaf);
          continue;
        }
      }

      // Declare win.
      if (leaf_board.CanDeclareWin()) {
        leaf->SetTerminal(1.0f);
        leaf->set_mate_status(1);
        if (leaf->parent() && leaf->parent_edge_idx() >= 0)
          leaf->parent()->edge(leaf->parent_edge_idx()).SetLose();
        Backpropagate(leaf, 1.0f, 0.0f);
        continue;
      }

      // Repetition.
      auto rep = leaf_board.CheckRepetition();
      if (rep != ShogiBoard::RepetitionResult::kNone) {
        float rep_v = 0.0f, rep_d = 0.0f;
        Edge::MateFlag rep_flag = Edge::kDraw;
        switch (rep) {
          case ShogiBoard::RepetitionResult::kDraw: rep_v = config_.draw_score; rep_d = 1.0f; rep_flag = Edge::kDraw; break;
          case ShogiBoard::RepetitionResult::kWin: rep_v = 1.0f; rep_flag = Edge::kWin; break;
          case ShogiBoard::RepetitionResult::kLoss: rep_v = -1.0f; rep_flag = Edge::kLose; break;
          default: break;
        }
        leaf->SetTerminal(rep_v, rep_d);
        if (leaf->parent() && leaf->parent_edge_idx() >= 0)
          leaf->parent()->edge(leaf->parent_edge_idx()).mate_flag = rep_flag;
        Backpropagate(leaf, rep_v, rep_d);
        continue;
      }

      // This leaf needs NN eval — add to batch.
      pending.push_back({leaf, leaf_board, std::move(leaf_legal)});
    }

    if (pending.empty()) continue;

    // --- GPU batch eval ---
    std::vector<std::pair<ShogiBoard, MoveList>> nn_batch;
    nn_batch.reserve(pending.size());
    for (auto& p : pending) {
      nn_batch.emplace_back(p.board, p.legal_moves);
    }
    auto results = evaluator_.EvaluateBatch(nn_batch);

    // --- Expand and backpropagate all ---
    for (int i = 0; i < (int)pending.size(); i++) {
      ExpandNodeWithNN(pending[i].node, results[i], pending[i].legal_moves);
      Backpropagate(pending[i].node, results[i].value, results[i].draw);
    }
    nodes_expanded += (int)pending.size();
  }

  // --- PV mate search (Tier 3) ---
  if (dfpn_pv_ && config_.pv_dfpn_nodes > 0) {
    PvMateSearch(root.get(), board);
  }

  // --- Collect result ---
  auto t1 = std::chrono::steady_clock::now();
  float elapsed = std::chrono::duration<float>(t1 - t0).count();

  result.best_move = SelectMove(root.get(), game_ply);
  result.nodes = nodes_expanded;
  result.time_sec = elapsed;
  result.nps = elapsed > 0.001f ? nodes_expanded / elapsed : 0.0f;
  result.root_q = root->q();
  result.root_d = root->d_avg();
  result.score_cp = QToCentipawns(root->q());
  result.mate_status = root->mate_status();
  result.pv = GetPV(root.get());

  // Top children.
  std::vector<std::pair<int, int>> child_visits;
  for (int i = 0; i < root->num_edges(); i++) {
    Node* c = root->child(i);
    int n = c ? c->n() : 0;
    child_visits.push_back({n, i});
  }
  std::sort(child_visits.begin(), child_visits.end(),
            [](auto& a, auto& b) { return a.first > b.first; });
  for (int k = 0; k < std::min(5, (int)child_visits.size()); k++) {
    int i = child_visits[k].second;
    Node* c = root->child(i);
    SearchResult::ChildInfo ci;
    ci.move = root->edge(i).move;
    ci.n = c ? c->n() : 0;
    ci.q = c ? c->q() : 0.0f;
    ci.p = root->edge(i).policy;
    result.top_children.push_back(ci);
  }

  return result;
}

// =====================================================================
// Select: PUCT walk from root to leaf
// =====================================================================

MCTSSearch::SelectResult MCTSSearch::Select(Node* root,
                                             const ShogiBoard& root_board) {
  Node* node = root;
  ShogiBoard board = root_board;

  while (node->is_expanded() && !node->is_terminal()) {
    int best = SelectBestChild(node, node->parent() == nullptr);
    if (best < 0) {
      // All children are terminal wins (we lose). Mark this node.
      node->SetTerminal(-1.0f);
      node->set_mate_status(-1);
      break;
    }

    // Apply the move.
    Move m = node->edge(best).move;
    board.DoMove(m);

    // Get or create child node.
    Node* child = node->GetOrCreateChild(best);
    node = child;

    if (!node->is_expanded()) break;  // Leaf reached
  }

  SelectResult sel;
  sel.node = node;
  sel.board = board;
  sel.edge_idx = node->parent_edge_idx();
  return sel;
}

// =====================================================================
// SelectBestChild: PUCT with mate awareness
// =====================================================================
// Reference: dlshogi UctSearch.cpp:772-906

int MCTSSearch::SelectBestChild(Node* node, bool is_root) {
  // Use n_started (N + in-flight) for parent visit count in exploration.
  const uint32_t parent_n = node->n_started();
  const float cpuct = Cpuct(parent_n);
  const float sqrt_parent = std::sqrt(std::max(parent_n, 1u));

  // FPU (First Play Urgency) with relative reduction.
  // Use real Q (unaffected by virtual loss) for FPU baseline.
  float visited_policy = 0.0f;
  for (int i = 0; i < node->num_edges(); i++) {
    Node* c = node->child(i);
    if (c && c->n() > 0) visited_policy += node->edge(i).policy;
  }
  float fpu_reduction = is_root ? config_.fpu_root : config_.fpu_value;
  float fpu = -node->q() - fpu_reduction * std::sqrt(std::max(visited_policy, 0.0f));

  float best_score = -1e10f;
  int best_idx = -1;
  bool all_children_win = true;  // Track if all edges are proven wins (= we lose)

  for (int i = 0; i < node->num_edges(); i++) {
    const Edge& edge = node->edge(i);

    // Skip edges where opponent wins (our loss).
    // dlshogi UctSearch.cpp:823-824
    if (edge.IsLose()) continue;

    // If this edge leads to opponent being mated (our win),
    // immediately select it. dlshogi UctSearch.cpp:826-830
    if (edge.IsWin()) {
      // Parent can choose this winning edge.
      if (node->parent()) {
        node->parent()->edge(node->parent_edge_idx()).SetLose();
      }
      return i;
    }

    all_children_win = false;

    // Standard PUCT score.
    // Q uses real visits only (never corrupted by virtual loss).
    // U uses n_started (N + in-flight) to reduce exploration for in-flight nodes.
    Node* c = node->child(i);
    float q;
    float u;
    if (c && c->n() > 0) {
      q = -c->q();  // Negate: child's value is from child's perspective
      u = cpuct * edge.policy * sqrt_parent / (1.0f + c->n_started());
    } else {
      q = fpu;
      // Unvisited but possibly in-flight: use n_started in denominator.
      uint32_t child_started = c ? c->n_started() : 0;
      u = cpuct * edge.policy * sqrt_parent / (1.0f + child_started);
    }

    float score = q + u;
    if (score > best_score) {
      best_score = score;
      best_idx = i;
    }
  }

  // If all children are wins for opponent (all edges are Lose), this node loses.
  // dlshogi UctSearch.cpp:886-890
  if (all_children_win && node->num_edges() > 0) {
    node->set_mate_status(-1);
    node->SetTerminal(-1.0f);
    if (node->parent()) {
      node->parent()->edge(node->parent_edge_idx()).SetWin();
    }
    return -1;
  }

  return best_idx;
}

// =====================================================================
// Backpropagate
// =====================================================================

void MCTSSearch::Backpropagate(Node* node, float value, float draw) {
  float v = value;
  float d = draw;
  while (node != nullptr) {
    node->AddVisit(v, d);
    v = -v;  // Flip perspective
    node = node->parent();
  }
}

// =====================================================================
// Mate propagation
// =====================================================================
// After marking a leaf node with mate_status, propagate upward.
// Reference: dlshogi UctSearch.cpp:826-891

void MCTSSearch::PropagateMateUp(Node* node) {
  Node* current = node->parent();
  while (current != nullptr) {
    // Check: does the current node have a winning edge?
    // (An edge that is Lose = opponent mated = we win)
    bool has_winning_edge = false;
    bool all_losing = true;  // All edges are Lose (opponent wins)

    for (int i = 0; i < current->num_edges(); i++) {
      const Edge& e = current->edge(i);
      if (e.IsWin()) {
        // This edge leads to opponent being mated. We can choose it!
        has_winning_edge = true;
        break;
      }
      if (!e.IsLose()) {
        all_losing = false;
      }
    }

    if (has_winning_edge) {
      // Current node can force a win.
      // But we need to check: is the edge to current (from grandparent) updated?
      // The edge in SelectBestChild already sets it. But if we discover it here:
      if (current->parent() && current->parent_edge_idx() >= 0) {
        current->parent()->edge(current->parent_edge_idx()).SetLose();
      }
      current->set_mate_status(1);
      current = current->parent();
    } else if (all_losing && current->num_edges() > 0) {
      // All children are losses for us (opponent wins everywhere).
      current->set_mate_status(-1);
      current->SetTerminal(-1.0f);
      if (current->parent() && current->parent_edge_idx() >= 0) {
        current->parent()->edge(current->parent_edge_idx()).SetWin();
      }
      current = current->parent();
    } else {
      break;  // Not enough info to propagate further.
    }
  }
}

// =====================================================================
// PV Mate Search (Tier 3)
// =====================================================================
// Reference: dlshogi PvMateSearch.cpp:48-181

void MCTSSearch::PvMateSearch(Node* root, const ShogiBoard& root_board) {
  if (!dfpn_pv_) return;

  Node* node = root;
  ShogiBoard board = root_board;

  while (node->is_expanded()) {
    // Find most-visited child (PV move).
    int best_idx = -1;
    uint32_t best_n = 0;
    for (int i = 0; i < node->num_edges(); i++) {
      Node* c = node->child(i);
      if (c && c->n() > best_n) {
        best_n = c->n();
        best_idx = i;
      }
    }
    if (best_idx < 0) break;

    // Skip if edge already resolved.
    const Edge& edge = node->edge(best_idx);
    if (edge.IsWin() || edge.IsLose()) break;

    // Apply move.
    board.DoMove(edge.move);
    Node* child = node->child(best_idx);
    if (!child) break;

    // Skip if already deeply checked.
    if (child->dfpn_proven_no_mate() || child->mate_status() != 0) {
      node = child;
      continue;
    }

    // Run deep df-pn.
    if (!child->dfpn_checked()) {
      child->set_dfpn_checked(true);
      Move mate_move = dfpn_pv_->search(board, config_.pv_dfpn_nodes);
      if (MateDfpnSolver::IsNoMate(mate_move)) {
        // Proved no mate.
        child->set_dfpn_proven_no_mate(true);
      } else if (!mate_move.is_null()) {
        // Mate found at PV node.
        child->set_mate_status(1);
        child->SetTerminal(1.0f);
        // The edge from parent to this child: child's side wins = parent loses.
        node->edge(best_idx).SetLose();
        PropagateMateUp(child);

        // If root is now resolved, stop.
        if (root->mate_status() != 0) return;
      }
      // else: unsolved/timeout — skip.
    }

    node = child;
    if (!child->is_expanded()) break;
  }
}

// =====================================================================
// Move selection
// =====================================================================

Move MCTSSearch::SelectMove(Node* root, int game_ply) {
  // If root has a proven mate, pick the winning move.
  if (root->mate_status() == 1) {
    for (int i = 0; i < root->num_edges(); i++) {
      if (root->edge(i).IsWin()) {
        return root->edge(i).move;
      }
    }
  }

  // Temperature-based selection.
  if (config_.temperature > 0.0f && game_ply <= config_.temp_moves) {
    // Proportional to N^(1/T).
    std::vector<double> weights;
    for (int i = 0; i < root->num_edges(); i++) {
      Node* c = root->child(i);
      double n = c ? c->n() : 0;
      if (config_.temperature != 1.0f) {
        n = std::pow(n, 1.0 / config_.temperature);
      }
      weights.push_back(n);
    }
    double total = 0;
    for (auto w : weights) total += w;
    if (total > 0) {
      std::mt19937 rng(std::random_device{}());
      std::discrete_distribution<int> dist(weights.begin(), weights.end());
      int idx = dist(rng);
      return root->edge(idx).move;
    }
  }

  // Argmax by visit count.
  int best_idx = 0;
  uint32_t best_n = 0;
  for (int i = 0; i < root->num_edges(); i++) {
    Node* c = root->child(i);
    uint32_t n = c ? c->n() : 0;
    if (n > best_n) {
      best_n = n;
      best_idx = i;
    }
  }
  return root->edge(best_idx).move;
}

// =====================================================================
// Helpers
// =====================================================================

float MCTSSearch::Cpuct(int parent_n) const {
  return config_.cpuct_init +
         config_.cpuct_factor *
             std::log((parent_n + config_.cpuct_base) / config_.cpuct_base);
}

int MCTSSearch::QToCentipawns(float q) {
  // Clamp to avoid tan() blowup.
  q = std::max(-0.999f, std::min(0.999f, q));
  return static_cast<int>(90.0f * std::tan(1.5637541897f * q));
}

std::vector<Move> MCTSSearch::GetPV(Node* root) {
  std::vector<Move> pv;
  Node* node = root;
  while (node->is_expanded()) {
    int best_idx = -1;
    uint32_t best_n = 0;
    for (int i = 0; i < node->num_edges(); i++) {
      Node* c = node->child(i);
      if (c && c->n() > best_n) {
        best_n = c->n();
        best_idx = i;
      }
    }
    if (best_idx < 0) break;
    pv.push_back(node->edge(best_idx).move);
    node = node->child(best_idx);
    if (!node) break;
  }
  return pv;
}

void MCTSSearch::AddDirichletNoise(Node* root) {
  int n = root->num_edges();
  if (n == 0) return;

  std::mt19937 rng(std::random_device{}());
  std::gamma_distribution<float> gamma(config_.noise_alpha, 1.0f);

  std::vector<float> noise(n);
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    noise[i] = gamma(rng);
    sum += noise[i];
  }
  if (sum > 0.0f) {
    for (int i = 0; i < n; i++) noise[i] /= sum;
  }

  float eps = config_.noise_epsilon;
  for (int i = 0; i < n; i++) {
    root->edge(i).policy =
        root->edge(i).policy * (1.0f - eps) + eps * noise[i];
  }
}

Move MCTSSearch::Mate1Ply(ShogiBoard& board) {
  // Try each legal move; if it results in checkmate, return it.
  // This is extremely fast for typical positions because we bail
  // at the first mate found.
  MoveList moves = board.GenerateLegalMoves();
  for (const Move& m : moves) {
    UndoInfo undo = board.DoMove(m);
    bool is_mate = board.GenerateLegalMoves().empty();
    board.UndoMove(m, undo);
    if (is_mate) return m;
  }
  return Move();  // No 1-ply mate
}

// =====================================================================
// Shared helper: expand a node with NN output
// =====================================================================

void MCTSSearch::ExpandNodeWithNN(Node* node, const NNOutput& eval,
                                   const MoveList& legal_moves) {
  std::vector<std::pair<Move, float>> priors;
  priors.reserve(legal_moves.size());
  for (int i = 0; i < legal_moves.size(); i++) {
    priors.emplace_back(legal_moves[i], eval.policy[i]);
  }
  node->Expand(priors);
  node->set_evaluated(true);
}

// =====================================================================
// Tree warmup: single-threaded BFS with batch GPU evaluation
// =====================================================================

void MCTSSearch::WarmupTree(Node* root, const ShogiBoard& root_board,
                             int num_nodes) {
  int expanded = 0;
  int batch_size = config_.warmup_batch;

  while (expanded < num_nodes && !stop_) {
    // Collect a batch of unexpanded leaves via policy-guided selection.
    struct LeafInfo {
      Node* node;
      ShogiBoard board;
      MoveList legal_moves;
      std::vector<Node*> path;  // for backpropagation
    };
    std::vector<LeafInfo> leaves;

    int consecutive_failures = 0;
    for (int b = 0; b < batch_size && expanded + (int)leaves.size() < num_nodes; b++) {
      // Walk from root to an unexpanded leaf using PUCT.
      Node* node = root;
      ShogiBoard board = root_board;
      std::vector<Node*> path;

      while (node->is_expanded() && !node->is_terminal()) {
        path.push_back(node);
        // Add virtual loss so subsequent selections in this batch diverge.
        node->AddVirtualLoss(1);

        int best = SelectBestChild(node, node == root);
        if (best < 0) break;

        board.DoMove(node->edge(best).move);
        Node* child = node->GetOrCreateChild(best);
        node = child;
        if (!node->is_expanded()) break;
      }

      if (node->is_terminal() || node->is_expanded()) {
        // Can't expand — remove virtual loss and skip.
        for (auto* n : path) n->RemoveVirtualLoss(1);
        consecutive_failures++;
        // Stop collecting if too many consecutive failures
        // (tree doesn't have enough unexpanded leaves for this batch).
        if (consecutive_failures > 100) break;
        continue;
      }
      consecutive_failures = 0;

      if (!node->TryStartExpansion()) {
        // Collision — another selection in this batch claimed this node.
        for (auto* n : path) n->RemoveVirtualLoss(1);
        consecutive_failures++;
        if (consecutive_failures > 100) break;
        continue;
      }
      consecutive_failures = 0;

      path.push_back(node);
      node->AddVirtualLoss(1);

      MoveList legal = board.GenerateLegalMoves();

      // Terminal checks.
      if (legal.empty()) {
        node->SetTerminal(-1.0f);
        node->set_mate_status(-1);
        if (node->parent() && node->parent_edge_idx() >= 0)
          node->parent()->edge(node->parent_edge_idx()).SetWin();
        node->FinishExpansion();
        // Backprop terminal value.
        float v = -1.0f;
        for (int i = (int)path.size() - 1; i >= 0; i--) {
          path[i]->RemoveVirtualLoss(1);
          path[i]->AddVisit(v, 0.0f);
          v = -v;
        }
        expanded++;
        continue;
      }

      // Skip Mate1Ply during warmup — too expensive per leaf.
      // MCTS will catch mates later via leaf Mate1Ply and df-pn.
      // Move mate1 = Mate1Ply(board);
      if (false) {
        node->FinishExpansion();
        float v = 1.0f;
        for (int i = (int)path.size() - 1; i >= 0; i--) {
          path[i]->RemoveVirtualLoss(1);
          path[i]->AddVisit(v, 0.0f);
          v = -v;
        }
        expanded++;
        continue;
      }

      leaves.push_back({node, board, legal, path});
    }

    if (leaves.empty()) break;

    // Batch evaluate all collected leaves.
    std::vector<std::pair<ShogiBoard, MoveList>> nn_batch;
    nn_batch.reserve(leaves.size());
    for (auto& leaf : leaves) {
      nn_batch.emplace_back(leaf.board, leaf.legal_moves);
    }

    // Use warmup evaluator if available (smaller batch engine).
    NNEvaluator& warmup_eval = warmup_evaluator_ ? *warmup_evaluator_ : evaluator_;
    auto results = warmup_eval.EvaluateBatch(nn_batch);

    // Expand and backpropagate each leaf.
    for (int i = 0; i < (int)leaves.size(); i++) {
      auto& leaf = leaves[i];
      auto& eval = results[i];

      ExpandNodeWithNN(leaf.node, eval, leaf.legal_moves);
      leaf.node->FinishExpansion();

      // Backpropagate NN value through the path.
      float v = eval.value, d = eval.draw;
      for (int j = (int)leaf.path.size() - 1; j >= 0; j--) {
        leaf.path[j]->RemoveVirtualLoss(1);
        leaf.path[j]->AddVisit(v, d);
        v = -v;
      }
      expanded++;
    }
  }
}

// =====================================================================
// Async queue search (lc0-style)
// =====================================================================

MCTSSearch::LeafSelectResult MCTSSearch::SelectLeaf(
    Node* root, const ShogiBoard& root_board) {
  LeafSelectResult res;
  res.needs_nn = false;
  res.value = 0.0f;
  res.draw = 0.0f;

  Node* node = root;
  ShogiBoard board = root_board;
  int vl = config_.virtual_loss_count;

  while (node->is_expanded() && !node->is_terminal()) {
    res.path.push_back(node);
    node->AddVirtualLoss(vl);

    int best = SelectBestChild(node, node == root);
    if (best < 0) {
      res.leaf = node;
      res.value = -1.0f;
      return res;
    }

    board.DoMove(node->edge(best).move);
    Node* child = node->GetOrCreateChild(best);
    node = child;
    if (!node->is_expanded()) break;
  }

  res.path.push_back(node);
  node->AddVirtualLoss(vl);
  res.leaf = node;
  res.board = board;

  if (node->is_terminal()) {
    res.value = node->terminal_v();
    res.draw = node->terminal_d();
    return res;
  }

  if (!node->TryStartExpansion()) {
    // Collision — use parent Q as estimate.
    res.value = node->parent() ? -node->parent()->q() : 0.0f;
    return res;
  }

  res.legal_moves = board.GenerateLegalMoves();

  // Terminal checks.
  if (res.legal_moves.empty()) {
    node->SetTerminal(-1.0f);
    node->set_mate_status(-1);
    if (node->parent() && node->parent_edge_idx() >= 0)
      node->parent()->edge(node->parent_edge_idx()).SetWin();
    node->FinishExpansion();
    res.value = -1.0f;
    return res;
  }

  if (res.board.CanDeclareWin()) {
    node->SetTerminal(1.0f);
    if (node->parent() && node->parent_edge_idx() >= 0)
      node->parent()->edge(node->parent_edge_idx()).SetLose();
    node->FinishExpansion();
    res.value = 1.0f;
    return res;
  }

  auto rep = res.board.CheckRepetition();
  if (rep != ShogiBoard::RepetitionResult::kNone) {
    float rep_v = config_.draw_score, rep_d = 0.0f;
    switch (rep) {
      case ShogiBoard::RepetitionResult::kDraw: rep_v = config_.draw_score; rep_d = 1.0f; break;
      case ShogiBoard::RepetitionResult::kWin: rep_v = 1.0f; break;
      case ShogiBoard::RepetitionResult::kLoss: rep_v = -1.0f; break;
      default: break;
    }
    node->SetTerminal(rep_v, rep_d);
    node->FinishExpansion();
    res.value = rep_v;
    res.draw = rep_d;
    return res;
  }

  // Leaf df-pn with tiny budget.
  if (config_.leaf_dfpn_nodes > 0 && !node->dfpn_checked()) {
    node->set_dfpn_checked(true);
    MateDfpnSolver leaf_dfpn(config_.leaf_dfpn_nodes);
    Move mate_move = leaf_dfpn.search(res.board, config_.leaf_dfpn_nodes);
    if (MateDfpnSolver::IsNoMate(mate_move)) {
      node->set_dfpn_proven_no_mate(true);
    } else if (!mate_move.is_null()) {
      node->SetTerminal(1.0f);
      node->set_mate_status(1);
      if (node->parent() && node->parent_edge_idx() >= 0)
        node->parent()->edge(node->parent_edge_idx()).SetLose();
      node->FinishExpansion();
      res.value = 1.0f;
      return res;
    }
  }

  // This leaf needs NN evaluation.
  res.needs_nn = true;
  return res;
}

void MCTSSearch::AsyncWorker(int worker_id, Node* root,
                              const ShogiBoard& root_board,
                              AsyncBatchQueue& queue,
                              std::atomic<int>& total_nodes,
                              std::atomic<bool>& search_done) {
  int vl = config_.virtual_loss_count;
  int gather_target = config_.minibatch_size;

  while (!search_done.load(std::memory_order_relaxed)) {
    if (total_nodes.load(std::memory_order_relaxed) >= config_.max_nodes) break;

    // --- Phase 1: Gather multiple leaves ---
    // Leaves needing NN eval are collected; terminals are backpropagated immediately.
    WorkerBatch batch;
    batch.worker_id = worker_id;

    int collisions = 0;
    for (int g = 0; g < gather_target; g++) {
      if (search_done.load(std::memory_order_relaxed)) break;
      if (total_nodes.load(std::memory_order_relaxed) >= config_.max_nodes) break;

      auto sel = SelectLeaf(root, root_board);

      if (sel.needs_nn) {
        LeafRequest req;
        req.board = std::move(sel.board);
        req.legal_moves = std::move(sel.legal_moves);
        req.leaf = sel.leaf;
        req.path = std::move(sel.path);
        batch.leaves.push_back(std::move(req));
      } else if (sel.leaf && sel.leaf->is_terminal()) {
        // Terminal — backpropagate real value.
        float v = sel.value, d = sel.draw;
        for (int i = (int)sel.path.size() - 1; i >= 0; i--) {
          sel.path[i]->RemoveVirtualLoss(vl);
          sel.path[i]->AddVisit(v, d);
          v = -v;
        }
      } else {
        // Collision — only remove virtual loss, don't add fake visits.
        collisions++;
        for (int i = (int)sel.path.size() - 1; i >= 0; i--) {
          sel.path[i]->RemoveVirtualLoss(vl);
        }
      }

      // If too many collisions, stop gathering early — tree is congested.
      if (collisions > gather_target / 2) break;
    }

    if (batch.leaves.empty()) continue;

    // Check search_done before submitting — GPU thread may have already exited.
    if (search_done.load(std::memory_order_relaxed)) {
      // Remove virtual loss from gathered leaves and exit.
      for (auto& req : batch.leaves) {
        for (auto* n : req.path) n->RemoveVirtualLoss(vl);
      }
      break;
    }

    // Save leaf/path data locally — SubmitBatchAndWait moves the batch away.
    struct LocalLeaf {
      Node* leaf;
      MoveList legal_moves;
      std::vector<Node*> path;
    };
    std::vector<LocalLeaf> local;
    local.reserve(batch.leaves.size());
    for (auto& req : batch.leaves) {
      local.push_back({req.leaf, req.legal_moves, req.path});
    }

    // --- Phase 2: Submit batch to GPU and wait ---
    auto results = queue.SubmitBatchAndWait(batch);

    if (search_done.load(std::memory_order_relaxed)) {
      // Remove virtual loss from all pending leaves and exit.
      for (auto& ll : local) {
        for (auto* n : ll.path) n->RemoveVirtualLoss(vl);
      }
      break;
    }

    // --- Phase 3: Expand and backpropagate all leaves ---
    for (int i = 0; i < (int)local.size(); i++) {
      auto& ll = local[i];
      auto& eval = results[i];

      ExpandNodeWithNN(ll.leaf, eval, ll.legal_moves);
      ll.leaf->FinishExpansion();

      float v = eval.value, d = eval.draw;
      for (int j = (int)ll.path.size() - 1; j >= 0; j--) {
        ll.path[j]->RemoveVirtualLoss(vl);
        ll.path[j]->AddVisit(v, d);
        v = -v;
      }
    }
    total_nodes.fetch_add((int)local.size(), std::memory_order_relaxed);
  }
}

SearchResult MCTSSearch::SearchAsync(ShogiBoard board, int game_ply,
                                      std::unique_ptr<Node>& root,
                                      const MoveList& legal_moves) {
  const int N = config_.num_search_threads;
  // Max GPU batch = sum of all workers' minibatches.
  // warmup_batch acts as the TensorRT engine batch size limit.
  const int max_gpu_batch = config_.warmup_batch > 0 ? config_.warmup_batch
                            : N * config_.minibatch_size;

  std::atomic<int> total_nodes{0};
  std::atomic<bool> search_done{false};
  auto t0 = std::chrono::steady_clock::now();

  // Create batch queue.
  AsyncBatchQueue queue(evaluator_, max_gpu_batch, N);

  // Launch GPU thread.
  auto gpu_thread = std::thread([&]() {
    queue.GPULoop();
  });

  // Launch worker threads.
  std::vector<std::thread> workers;
  for (int w = 0; w < N; w++) {
    workers.emplace_back([&, w]() {
      AsyncWorker(w, root.get(), board, queue, total_nodes, search_done);
    });
  }

  // Monitor: stop when time or node limit reached.
  while (!search_done.load(std::memory_order_relaxed)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    int n = total_nodes.load(std::memory_order_relaxed);
    if (n >= config_.max_nodes || stop_.load(std::memory_order_relaxed)) {
      search_done.store(true, std::memory_order_relaxed);
      break;
    }
    if (config_.max_time > 0.0f) {
      float elapsed = std::chrono::duration<float>(
          std::chrono::steady_clock::now() - t0).count();
      if (elapsed >= config_.max_time) {
        search_done.store(true, std::memory_order_relaxed);
        break;
      }
    }
  }

  search_done.store(true, std::memory_order_relaxed);

  // Wait for workers first — they'll exit due to search_done.
  for (auto& w : workers) w.join();

  // Now stop the GPU thread — workers are done, no more submissions.
  queue.NotifyStop();
  gpu_thread.join();

  // --- PV mate search ---
  if (dfpn_pv_ && config_.pv_dfpn_nodes > 0) {
    PvMateSearch(root.get(), board);
  }

  // --- Collect result ---
  auto t1 = std::chrono::steady_clock::now();
  float elapsed = std::chrono::duration<float>(t1 - t0).count();

  SearchResult result;
  result.best_move = SelectMove(root.get(), game_ply);
  result.nodes = total_nodes.load();
  result.time_sec = elapsed;
  result.nps = elapsed > 0.001f ? total_nodes.load() / elapsed : 0.0f;
  result.root_q = root->q();
  result.root_d = root->d_avg();
  result.score_cp = QToCentipawns(root->q());
  result.mate_status = root->mate_status();
  result.pv = GetPV(root.get());

  // Top children.
  std::vector<std::pair<int, int>> child_visits;
  for (int i = 0; i < root->num_edges(); i++) {
    Node* c = root->child(i);
    int n = c ? c->n() : 0;
    child_visits.push_back({n, i});
  }
  std::sort(child_visits.begin(), child_visits.end(),
            [](auto& a, auto& b) { return a.first > b.first; });
  for (int k = 0; k < std::min(5, (int)child_visits.size()); k++) {
    int i = child_visits[k].second;
    Node* c = root->child(i);
    SearchResult::ChildInfo ci;
    ci.move = root->edge(i).move;
    ci.n = c ? c->n() : 0;
    ci.q = c ? c->q() : 0.0f;
    ci.p = root->edge(i).policy;
    result.top_children.push_back(ci);
  }

  return result;
}

// =====================================================================
// Multi-threaded search with multi-leaf and multi-expand (legacy barrier)
// =====================================================================

void MCTSSearch::MTSelectOne(SimContext& sim, Node* start_node,
                              const ShogiBoard& start_board) {
  Node* node = start_node;
  ShogiBoard board = start_board;
  int vl = config_.virtual_loss_count;

  while (node->is_expanded() && !node->is_terminal()) {
    sim.path.push_back(node);
    node->AddVirtualLoss(vl);

    int best = SelectBestChild(node, node == start_node && sim.expand_records.empty());
    if (best < 0) {
      sim.leaf = node;
      sim.leaf_type = SimContext::kTerminal;
      sim.value = -1.0f;
      sim.draw = 0.0f;
      return;
    }

    Move m = node->edge(best).move;
    board.DoMove(m);
    Node* child = node->GetOrCreateChild(best);
    node = child;
    if (!node->is_expanded()) break;
  }

  sim.path.push_back(node);
  node->AddVirtualLoss(vl);
  sim.leaf = node;
  sim.board = board;

  if (node->is_terminal()) {
    sim.leaf_type = SimContext::kTerminal;
    sim.value = node->terminal_v();
    sim.draw = node->terminal_d();
    return;
  }

  if (!node->TryStartExpansion()) {
    sim.leaf_type = SimContext::kCollision;
    sim.value = node->parent() ? -node->parent()->q() : 0.0f;
    sim.draw = 0.0f;
    return;
  }

  sim.legal_moves = board.GenerateLegalMoves();

  if (sim.legal_moves.empty()) {
    node->SetTerminal(-1.0f);
    node->set_mate_status(-1);
    if (node->parent() && node->parent_edge_idx() >= 0)
      node->parent()->edge(node->parent_edge_idx()).SetWin();
    node->FinishExpansion();
    sim.leaf_type = SimContext::kTerminal;
    sim.value = -1.0f;
    sim.draw = 0.0f;
    return;
  }

  // Mate1Ply skipped in MT path — too expensive (~2.5ms per leaf).
  // Mates are caught by: root df-pn, leaf df-pn, and terminal detection
  // (child with no legal moves → checkmate on next visit).

  if (sim.board.CanDeclareWin()) {
    node->SetTerminal(1.0f);
    if (node->parent() && node->parent_edge_idx() >= 0)
      node->parent()->edge(node->parent_edge_idx()).SetLose();
    node->FinishExpansion();
    sim.leaf_type = SimContext::kTerminal;
    sim.value = 1.0f;
    sim.draw = 0.0f;
    return;
  }

  auto rep = sim.board.CheckRepetition();
  if (rep != ShogiBoard::RepetitionResult::kNone) {
    float rep_v = config_.draw_score, rep_d = 0.0f;
    switch (rep) {
      case ShogiBoard::RepetitionResult::kDraw: rep_v = config_.draw_score; rep_d = 1.0f; break;
      case ShogiBoard::RepetitionResult::kWin: rep_v = 1.0f; break;
      case ShogiBoard::RepetitionResult::kLoss: rep_v = -1.0f; break;
      default: break;
    }
    node->SetTerminal(rep_v, rep_d);
    node->FinishExpansion();
    sim.leaf_type = SimContext::kTerminal;
    sim.value = rep_v;
    sim.draw = rep_d;
    return;
  }

  if (config_.leaf_dfpn_nodes > 0 && !node->dfpn_checked()) {
    node->set_dfpn_checked(true);
    MateDfpnSolver leaf_dfpn(config_.leaf_dfpn_nodes);
    Move mate_move = leaf_dfpn.search(sim.board, config_.leaf_dfpn_nodes);
    if (MateDfpnSolver::IsNoMate(mate_move)) {
      node->set_dfpn_proven_no_mate(true);
    } else if (!mate_move.is_null()) {
      node->SetTerminal(1.0f);
      node->set_mate_status(1);
      if (node->parent() && node->parent_edge_idx() >= 0)
        node->parent()->edge(node->parent_edge_idx()).SetLose();
      node->FinishExpansion();
      sim.leaf_type = SimContext::kTerminal;
      sim.value = 1.0f;
      sim.draw = 0.0f;
      return;
    }
  }

  sim.leaf_type = SimContext::kNeedsNN;
}

void MCTSSearch::MTExpandOne(SimContext& sim) {
  float v = 0.0f, d = 0.0f;

  if (sim.leaf_type == SimContext::kNeedsNN) {
    Node* leaf = sim.leaf;
    const NNOutput& eval = sim.nn_output;
    ExpandNodeWithNN(leaf, eval, sim.legal_moves);
    leaf->FinishExpansion();
    v = eval.value;
    d = eval.draw;
    sim.can_continue = true;
  } else if (sim.leaf_type == SimContext::kTerminal) {
    v = sim.value;
    d = sim.draw;
    sim.can_continue = false;
  } else {
    v = sim.value;
    d = sim.draw;
    sim.can_continue = false;
  }

  sim.expand_records.push_back({
    (int)sim.path.size(), v, d, sim.leaf_type
  });
}

SearchResult MCTSSearch::SearchMT(ShogiBoard board, int game_ply,
                                   std::unique_ptr<Node>& root,
                                   const MoveList& legal_moves) {
  const int N = config_.num_search_threads;
  const int K = config_.sims_per_thread;
  const int D = config_.expand_depth;
  SearchBarrier barrier1(N);
  SearchBarrier barrier2(N);
  SearchBarrier barrier3(N);

  std::vector<ThreadContext> contexts(N);
  BatchQueue batch_queue;
  batch_queue.Init(N);
  for (int i = 0; i < N; i++) {
    contexts[i].thread_id = i;
    contexts[i].Init(K);
    batch_queue.SetContext(i, &contexts[i]);
  }

  std::atomic<int> total_nodes{0};
  std::atomic<bool> search_done{false};
  auto t0 = std::chrono::steady_clock::now();

  std::vector<std::thread> threads;
  for (int t = 0; t < N; t++) {
    threads.emplace_back([&, t]() {
      ThreadContext& ctx = contexts[t];

      while (!search_done.load(std::memory_order_relaxed)) {
        ctx.ResetAll();

        for (int depth = 0; depth < D; depth++) {

          // ===== PHASE 1: SELECT K leaves per thread =====
          for (int k = 0; k < K; k++) {
            SimContext& sim = ctx.sims[k];
            if (depth == 0) {
              MTSelectOne(sim, root.get(), board);
            } else if (sim.can_continue) {
              Node* cont_node = sim.leaf;
              ShogiBoard cont_board = sim.board;
              sim.ResetForNextDepth();
              MTSelectOne(sim, cont_node, cont_board);
            }
          }

          barrier1.Wait();

          // ===== PHASE 2: GPU EVAL (thread 0) =====
          if (t == 0) {
            auto nn_batch = batch_queue.BuildBatch();
            if (!nn_batch.empty()) {
              auto results = evaluator_.EvaluateBatch(nn_batch);
              batch_queue.DistributeResults(results);
            }

            int n = total_nodes.load(std::memory_order_relaxed);
            if (n >= config_.max_nodes || stop_.load(std::memory_order_relaxed)) {
              search_done.store(true, std::memory_order_relaxed);
            }
            if (config_.max_time > 0.0f) {
              float elapsed = std::chrono::duration<float>(
                  std::chrono::steady_clock::now() - t0).count();
              if (elapsed >= config_.max_time)
                search_done.store(true, std::memory_order_relaxed);
            }
          }

          barrier2.Wait();

          // ===== PHASE 3: EXPAND K leaves per thread =====
          if (search_done.load(std::memory_order_relaxed)) {
            for (auto& sim : ctx.sims) sim.can_continue = false;
          } else {
            for (int k = 0; k < K; k++) {
              SimContext& sim = ctx.sims[k];
              if (sim.can_continue || (depth == 0 && sim.leaf_type == SimContext::kNeedsNN)) {
                MTExpandOne(sim);
                if (sim.leaf_type == SimContext::kNeedsNN)
                  total_nodes.fetch_add(1, std::memory_order_relaxed);
              }
            }
          }

          barrier3.Wait();
          if (search_done.load(std::memory_order_relaxed)) break;
        }

        // ===== BACKPROPAGATE all K sims =====
        int vl = config_.virtual_loss_count;
        for (int k = 0; k < K; k++) {
          SimContext& sim = ctx.sims[k];
          if (sim.path.empty()) continue;

          if (sim.expand_records.empty()) {
            for (int i = (int)sim.path.size() - 1; i >= 0; i--)
              sim.path[i]->RemoveVirtualLoss(vl);
          } else {
            auto& deepest = sim.expand_records.back();
            float v = deepest.value, d = deepest.draw;
            for (int i = (int)sim.path.size() - 1; i >= 0; i--) {
              sim.path[i]->RemoveVirtualLoss(vl);
              sim.path[i]->AddVisit(v, d);
              v = -v;
            }
          }
        }
      }
    });
  }

  for (auto& thread : threads) thread.join();

  // --- Post-search: PV mate search ---
  if (dfpn_pv_ && config_.pv_dfpn_nodes > 0) {
    PvMateSearch(root.get(), board);
  }

  // --- Collect result ---
  auto t1 = std::chrono::steady_clock::now();
  float elapsed = std::chrono::duration<float>(t1 - t0).count();

  SearchResult result;
  result.best_move = SelectMove(root.get(), game_ply);
  result.nodes = total_nodes.load();
  result.time_sec = elapsed;
  result.nps = elapsed > 0.001f ? total_nodes.load() / elapsed : 0.0f;
  result.root_q = root->q();
  result.root_d = root->d_avg();
  result.score_cp = QToCentipawns(root->q());
  result.mate_status = root->mate_status();
  result.pv = GetPV(root.get());

  // Top children.
  std::vector<std::pair<int, int>> child_visits;
  for (int i = 0; i < root->num_edges(); i++) {
    Node* c = root->child(i);
    int n = c ? c->n() : 0;
    child_visits.push_back({n, i});
  }
  std::sort(child_visits.begin(), child_visits.end(),
            [](auto& a, auto& b) { return a.first > b.first; });
  for (int k = 0; k < std::min(5, (int)child_visits.size()); k++) {
    int i = child_visits[k].second;
    Node* c = root->child(i);
    SearchResult::ChildInfo ci;
    ci.move = root->edge(i).move;
    ci.n = c ? c->n() : 0;
    ci.q = c ? c->q() : 0.0f;
    ci.p = root->edge(i).policy;
    result.top_children.push_back(ci);
  }

  return result;
}

}  // namespace jhbr2
