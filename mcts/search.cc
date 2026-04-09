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
#include <cassert>
#include <chrono>
#include <cmath>
#include <random>

#include "mate/dfpn.h"

namespace jhbr2 {

using namespace lczero;

// =====================================================================
// Constructor / Destructor
// =====================================================================

MCTSSearch::MCTSSearch(NNEvaluator& evaluator, const MCTSConfig& config)
    : evaluator_(evaluator), config_(config) {
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

SearchResult MCTSSearch::Search(const ShogiBoard& board, int game_ply) {
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

  // --- Main search loop ---
  auto t0 = std::chrono::steady_clock::now();
  int nodes_expanded = 0;

  while (nodes_expanded < config_.max_nodes && !stop_) {
    // Check time limit.
    if (config_.max_time > 0.0f) {
      auto now = std::chrono::steady_clock::now();
      float elapsed = std::chrono::duration<float>(now - t0).count();
      if (elapsed >= config_.max_time) break;
    }

    // 1. Select: walk from root to a leaf.
    auto sel = Select(root.get(), board);
    Node* leaf = sel.node;

    // 2. If terminal, backpropagate known value.
    if (leaf->is_terminal()) {
      Backpropagate(leaf, leaf->terminal_v(), leaf->terminal_d());
      continue;
    }

    // 3. Generate legal moves at the leaf.
    ShogiBoard& leaf_board = sel.board;
    MoveList leaf_legal = leaf_board.GenerateLegalMoves();

    // --- Mate detection (before NN eval, following dlshogi) ---

    // No legal moves = checkmate.
    if (leaf_legal.empty()) {
      leaf->SetTerminal(-1.0f);
      leaf->set_mate_status(-1);
      // The parent's edge that led here: opponent is mated = win for parent.
      if (leaf->parent() && leaf->parent_edge_idx() >= 0) {
        leaf->parent()->edge(leaf->parent_edge_idx()).SetLose();
      }
      Backpropagate(leaf, -1.0f, 0.0f);
      PropagateMateUp(leaf);
      continue;
    }

    // Tier 1: 1-ply mate check (essentially free).
    Move mate1 = Mate1Ply(leaf_board);
    if (!mate1.is_null()) {
      leaf->SetTerminal(1.0f);
      leaf->set_mate_status(1);
      if (leaf->parent() && leaf->parent_edge_idx() >= 0) {
        leaf->parent()->edge(leaf->parent_edge_idx()).SetWin();
      }
      Backpropagate(leaf, 1.0f, 0.0f);
      PropagateMateUp(leaf);
      continue;
    }

    // Tier 2: Shallow df-pn mate search.
    if (dfpn_leaf_ && !leaf->dfpn_checked()) {
      leaf->set_dfpn_checked(true);
      Move mate_move = dfpn_leaf_->search(leaf_board, config_.leaf_dfpn_nodes);
      if (MateDfpnSolver::IsNoMate(mate_move)) {
        // Proved no mate exists.
        leaf->set_dfpn_proven_no_mate(true);
      } else if (!mate_move.is_null()) {
        // Mate found (valid mating move returned).
        leaf->SetTerminal(1.0f);
        leaf->set_mate_status(1);
        if (leaf->parent() && leaf->parent_edge_idx() >= 0) {
          leaf->parent()->edge(leaf->parent_edge_idx()).SetWin();
        }
        Backpropagate(leaf, 1.0f, 0.0f);
        PropagateMateUp(leaf);
        continue;
      }
      // else: mate_move.is_null() = unsolved/timeout, proceed normally.
    }

    // Check entering-king declaration.
    if (leaf_board.CanDeclareWin()) {
      leaf->SetTerminal(1.0f);
      leaf->set_mate_status(1);
      if (leaf->parent() && leaf->parent_edge_idx() >= 0) {
        leaf->parent()->edge(leaf->parent_edge_idx()).SetWin();
      }
      Backpropagate(leaf, 1.0f, 0.0f);
      continue;
    }

    // Check repetition.
    auto rep = leaf_board.CheckRepetition();
    if (rep != ShogiBoard::RepetitionResult::kNone) {
      float rep_v = 0.0f;
      float rep_d = 0.0f;
      Edge::MateFlag rep_flag = Edge::kDraw;
      switch (rep) {
        case ShogiBoard::RepetitionResult::kDraw:
          rep_v = config_.draw_score;
          rep_d = 1.0f;
          rep_flag = Edge::kDraw;
          break;
        case ShogiBoard::RepetitionResult::kWin:
          rep_v = 1.0f;
          rep_flag = Edge::kWin;
          break;
        case ShogiBoard::RepetitionResult::kLoss:
          rep_v = -1.0f;
          rep_flag = Edge::kLose;
          break;
        default:
          break;
      }
      leaf->SetTerminal(rep_v, rep_d);
      if (leaf->parent() && leaf->parent_edge_idx() >= 0) {
        leaf->parent()->edge(leaf->parent_edge_idx()).mate_flag = rep_flag;
      }
      Backpropagate(leaf, rep_v, rep_d);
      continue;
    }

    // 4. Evaluate with NN.
    NNOutput eval = evaluator_.Evaluate(leaf_board, leaf_legal);

    // 5. Expand leaf.
    std::vector<std::pair<Move, float>> leaf_priors;
    leaf_priors.reserve(leaf_legal.size());
    for (size_t i = 0; i < leaf_legal.size(); i++) {
      leaf_priors.emplace_back(leaf_legal[i], eval.policy[i]);
    }
    leaf->Expand(leaf_priors);
    leaf->set_evaluated(true);

    // 6. Backpropagate.
    Backpropagate(leaf, eval.value, eval.draw);
    nodes_expanded++;
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
  const float cpuct = Cpuct(node->n());
  const float sqrt_parent = std::sqrt(std::max(node->n(), 1u));

  // FPU (First Play Urgency) with relative reduction.
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
    Node* c = node->child(i);
    float q;
    float u;
    if (c && c->n() > 0) {
      q = -c->q();  // Negate: child's value is from child's perspective
      u = cpuct * edge.policy * sqrt_parent / (1.0f + c->n());
    } else {
      q = fpu;
      u = cpuct * edge.policy * sqrt_parent;  // denominator is 1
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
        node->edge(best_idx).SetWin();
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
      if (root->edge(i).IsLose()) {
        // This edge leads to opponent being mated = our win.
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

Move MCTSSearch::Mate1Ply(const ShogiBoard& board) {
  // Try each legal move; if it results in checkmate, return it.
  // This is extremely fast for typical positions because we bail
  // at the first mate found.
  MoveList moves = board.GenerateLegalMoves();
  for (const Move& m : moves) {
    ShogiBoard copy = board;
    copy.DoMove(m);
    if (copy.GenerateLegalMoves().empty()) {
      return m;  // Opponent has no legal moves = checkmate
    }
  }
  return Move();  // No 1-ply mate
}

}  // namespace jhbr2
