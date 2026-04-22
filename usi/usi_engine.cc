/*
  JHBR2 Shogi Engine — USI Protocol Implementation

  Reference: JHBR2/shogi_usi.py (Python prototype)
*/

#include "usi/usi_engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "mate/dfpn.h"
#include "shogi/encoder.h"

namespace jhbr2 {

using namespace lczero;

// =====================================================================
// Helpers
// =====================================================================

static std::vector<std::string> Split(const std::string& s) {
  std::vector<std::string> parts;
  std::istringstream iss(s);
  std::string token;
  while (iss >> token) parts.push_back(token);
  return parts;
}

// =====================================================================
// Constructor
// =====================================================================

USIEngine::USIEngine() {
  board_.SetStartPos();
}

// =====================================================================
// Main loop
// =====================================================================

void USIEngine::Run() {
  std::string line;
  while (std::getline(std::cin, line)) {
    // Trim whitespace.
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
      line.pop_back();
    if (line.empty()) continue;

    auto parts = Split(line);
    if (parts.empty()) continue;

    const auto& cmd = parts[0];

    if (cmd == "usi")         CmdUsi();
    else if (cmd == "isready")    CmdIsReady();
    else if (cmd == "setoption")  CmdSetOption(parts);
    else if (cmd == "usinewgame") CmdUsiNewGame();
    else if (cmd == "position")   CmdPosition(parts);
    else if (cmd == "go")         CmdGo(parts);
    else if (cmd == "stop")       CmdStop();
    else if (cmd == "quit")       break;
    else if (cmd == "gameover")   CmdGameOver(parts);
    else if (cmd == "d")          CmdDebug();
  }
}

// =====================================================================
// USI command handlers
// =====================================================================

void USIEngine::Send(const std::string& msg) {
  std::cout << msg << std::endl;
}

void USIEngine::Log(const std::string& msg) {
  std::cout << "info string " << msg << std::endl;
}

void USIEngine::CmdUsi() {
  Send(std::string("id name ") + ENGINE_NAME);
  Send(std::string("id author ") + ENGINE_AUTHOR);

  Send("option name MaxNodes type spin default 800 min 1 max 1000000");
  Send("option name OnnxModel type string default shogi_bt4.onnx");
  Send("option name NoiseEpsilon type string default 0.0");
  Send("option name LeafDfpnNodes type spin default 100 min 0 max 100000");
  Send("option name PvDfpnNodes type spin default 100000 min 0 max 10000000");
  Send("option name UseGPU type check default true");
  Send("option name Threads type spin default 1 min 1 max 4096");
  Send("option name ExpandDepth type spin default 1 min 1 max 8");
  Send("option name SimsPerThread type spin default 1 min 1 max 10000");
  Send("option name DfPnMaxTime type spin default 4000 min 100 max 60000");
  Send("option name MaxMoveTime type spin default 0 min 0 max 300000");

  Send("usiok");
}

void USIEngine::CmdIsReady() {
  if (!evaluator_) {
    Log("Loading model: " + onnx_path_);

    // Initialize encoder tables.
    ShogiEncoderTables::Init();

    evaluator_ = std::make_unique<NNEvaluator>(onnx_path_, use_gpu_);

    config_.max_nodes = max_nodes_;
    config_.noise_epsilon = noise_epsilon_;
    config_.leaf_dfpn_nodes = leaf_dfpn_nodes_;
    config_.pv_dfpn_nodes = pv_dfpn_nodes_;

    search_ = std::make_unique<MCTSSearch>(*evaluator_, config_);

    Log("Model loaded, max_nodes=" + std::to_string(config_.max_nodes));
  }
  Send("readyok");
}

static std::string ToLower(const std::string& s) {
  std::string r = s;
  std::transform(r.begin(), r.end(), r.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return r;
}

void USIEngine::CmdSetOption(const std::vector<std::string>& parts) {
  // Parse: setoption name <NAME> value <VALUE>
  std::string name, value;
  for (size_t i = 1; i < parts.size(); i++) {
    if (parts[i] == "name" && i + 1 < parts.size()) {
      name = parts[i + 1];
    } else if (parts[i] == "value" && i + 1 < parts.size()) {
      value = parts[i + 1];
    }
  }

  std::string name_lower = ToLower(name);

  if (name_lower == "maxnodes") {
    max_nodes_ = std::stoi(value);
    config_.max_nodes = max_nodes_;
  } else if (name_lower == "onnxmodel") {
    onnx_path_ = value;
  } else if (name_lower == "noiseepsilon") {
    noise_epsilon_ = std::stof(value);
    config_.noise_epsilon = noise_epsilon_;
  } else if (name_lower == "leafdfpnnodes") {
    leaf_dfpn_nodes_ = std::stoi(value);
    config_.leaf_dfpn_nodes = leaf_dfpn_nodes_;
  } else if (name_lower == "pvdfpnnodes") {
    pv_dfpn_nodes_ = std::stoi(value);
    config_.pv_dfpn_nodes = pv_dfpn_nodes_;
  } else if (name_lower == "usegpu") {
    use_gpu_ = (value == "true");
  } else if (name_lower == "threads") {
    config_.num_search_threads = std::stoi(value);
  } else if (name_lower == "expanddepth") {
    config_.expand_depth = std::stoi(value);
  } else if (name_lower == "dfpnmaxtime") {
    dfpn_max_time_ms_ = std::stoi(value);
  } else if (name_lower == "maxmovetime") {
    max_move_time_ms_ = std::stoi(value);
  } else if (name_lower == "simsperthread") {
    config_.sims_per_thread = std::stoi(value);
  }

  Log("Set " + name + " = " + value);
}

void USIEngine::CmdUsiNewGame() {
  board_.SetStartPos();
  board_.ClearHistory();
  game_ply_ = 0;
  // Recreate search with current config.
  if (evaluator_) {
    search_ = std::make_unique<MCTSSearch>(*evaluator_, config_);
  }
}

void USIEngine::CmdPosition(const std::vector<std::string>& parts) {
  board_ = ShogiBoard();
  size_t idx = 1;

  if (idx >= parts.size()) return;

  if (parts[idx] == "startpos") {
    board_.SetStartPos();
    idx++;
  } else if (parts[idx] == "sfen") {
    idx++;
    // Collect SFEN parts until "moves" or end.
    std::string sfen;
    while (idx < parts.size() && parts[idx] != "moves") {
      if (!sfen.empty()) sfen += " ";
      sfen += parts[idx];
      idx++;
    }
    board_.SetFromSfen(sfen);
  }

  // Apply moves.
  if (idx < parts.size() && parts[idx] == "moves") {
    idx++;
    while (idx < parts.size()) {
      Move m = Move::Parse(parts[idx]);
      board_.DoMove(m);
      idx++;
    }
  }

  game_ply_ = board_.RepetitionCount();  // Approximate ply from history
}

void USIEngine::CmdGo(const std::vector<std::string>& parts) {
  if (!search_) {
    Send("bestmove resign");
    return;
  }

  // Parse time controls.
  int btime = 0, wtime = 0, byoyomi = 0, binc = 0, winc = 0;
  int nodes_limit = max_nodes_;
  float max_time = 0.0f;

  size_t i = 1;
  while (i < parts.size()) {
    if (parts[i] == "btime" && i + 1 < parts.size()) {
      btime = std::stoi(parts[i + 1]); i += 2;
    } else if (parts[i] == "wtime" && i + 1 < parts.size()) {
      wtime = std::stoi(parts[i + 1]); i += 2;
    } else if (parts[i] == "byoyomi" && i + 1 < parts.size()) {
      byoyomi = std::stoi(parts[i + 1]); i += 2;
    } else if (parts[i] == "binc" && i + 1 < parts.size()) {
      binc = std::stoi(parts[i + 1]); i += 2;
    } else if (parts[i] == "winc" && i + 1 < parts.size()) {
      winc = std::stoi(parts[i + 1]); i += 2;
    } else if (parts[i] == "nodes" && i + 1 < parts.size()) {
      nodes_limit = std::stoi(parts[i + 1]); i += 2;
    } else if (parts[i] == "infinite") {
      max_time = 0; nodes_limit = 1000000; i++;
    } else if (parts[i] == "mate") {
      // Delegate to go mate handler.
      CmdGoMate(parts);
      return;
    } else if (parts[i] == "ponder") {
      i++;  // Ignore ponder
    } else {
      i++;
    }
  }

  // Time management.
  if (byoyomi > 0) {
    max_time = byoyomi / 1000.0f * 0.9f;
  } else if (btime > 0 || wtime > 0) {
    int my_time = (board_.side_to_move() == BLACK) ? btime : wtime;
    int my_inc = (board_.side_to_move() == BLACK) ? binc : winc;
    max_time = (my_time * 0.05f + my_inc * 0.8f) / 1000.0f;
    max_time = std::max(max_time, 0.1f);
  }

  // Apply per-move time cap (MaxMoveTime option).
  // Reserve 2 seconds for overhead (root eval, df-pn, thread join).
  if (max_move_time_ms_ > 0) {
    float cap = std::max(max_move_time_ms_ / 1000.0f - 2.0f, 0.5f);
    if (max_time <= 0.0f || cap < max_time) {
      max_time = cap;
    }
  }

  // Check entering-king declaration.
  if (board_.CanDeclareWin()) {
    Send("bestmove win");
    return;
  }

  // Set search limits.
  config_.max_nodes = nodes_limit;
  config_.max_time = max_time;

  // --- Launch root df-pn in parallel with MCTS (dlshogi-style) ---
  // Scale df-pn budget and wait time with available time.
  // More time remaining → larger df-pn budget → find deeper mates.
  int my_time_ms = (board_.side_to_move() == BLACK) ? btime : wtime;
  int my_inc_ms = (board_.side_to_move() == BLACK) ? binc : winc;
  int available_ms = my_time_ms + my_inc_ms + byoyomi;

  int dfpn_min_wait_ms;
  size_t root_dfpn_nodes;
  if (available_ms <= 0) {
    // No time info (go nodes, go infinite) — use defaults.
    dfpn_min_wait_ms = 300;
    root_dfpn_nodes = 100000;
  } else if (available_ms < 10000) {
    // Under 10 seconds — minimal df-pn.
    dfpn_min_wait_ms = 100;
    root_dfpn_nodes = 10000;
  } else if (available_ms < 60000) {
    // 10-60 seconds — moderate df-pn.
    dfpn_min_wait_ms = 300;
    root_dfpn_nodes = 100000;
  } else if (available_ms < 300000) {
    // 1-5 minutes — generous df-pn.
    dfpn_min_wait_ms = 500;
    root_dfpn_nodes = 500000;
  } else {
    // 5+ minutes — deep df-pn search.
    dfpn_min_wait_ms = 1000;
    root_dfpn_nodes = 2000000;
  }

  // Hard wall-clock deadline for the entire move (MCTS + df-pn + everything).
  auto move_start_time = std::chrono::steady_clock::now();
  int hard_deadline_ms = (max_move_time_ms_ > 0)
      ? max_move_time_ms_
      : static_cast<int>(max_time * 1000) + 2000;  // clock time + 2s margin

  MateDfpnSolver root_dfpn(root_dfpn_nodes);
  std::atomic<bool> dfpn_done{false};
  Move dfpn_mate_move;
  ShogiBoard dfpn_board = board_;

  auto dfpn_thread = std::thread([&]() {
    dfpn_mate_move = root_dfpn.search(dfpn_board, root_dfpn_nodes);
    dfpn_done = true;
  });

  // --- Run MCTS concurrently ---
  search_ = std::make_unique<MCTSSearch>(*evaluator_, config_);
  SearchResult result = search_->Search(board_, game_ply_);

  // --- Stop df-pn and wait for it to finish ---
  root_dfpn.stop();

  // Brief wait if we have time remaining before the hard deadline.
  auto total_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - move_start_time).count();
  int remaining_ms = hard_deadline_ms - (int)total_elapsed_ms;
  int wait_ms = std::min(dfpn_min_wait_ms, std::max(remaining_ms - 500, 0));

  if (!dfpn_done && wait_ms > 0) {
    auto wait_start = std::chrono::steady_clock::now();
    while (!dfpn_done) {
      auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - wait_start).count();
      if (elapsed_ms >= wait_ms) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    root_dfpn.stop();  // ensure stopped if wait timed out
  }
  dfpn_thread.join();

  // --- Choose result: prefer mate if found ---
  bool use_mate = dfpn_done &&
                  !dfpn_mate_move.is_null() &&
                  !MateDfpnSolver::IsNoMate(dfpn_mate_move);

  if (use_mate) {
    auto pv = root_dfpn.get_pv();
    std::string pv_str;
    for (const auto& m : pv) {
      if (!pv_str.empty()) pv_str += " ";
      pv_str += m.ToString();
    }
    if (pv_str.empty()) pv_str = dfpn_mate_move.ToString();

    int mate_ply = (int)pv.size();
    Log("Root df-pn found mate in " + std::to_string(mate_ply) + " ply (" +
        std::to_string(root_dfpn.get_nodes_searched()) + " nodes)");

    Send("info depth 1 score mate " + std::to_string((mate_ply + 1) / 2) +
         " nodes " + std::to_string(root_dfpn.get_nodes_searched()) +
         " pv " + pv_str);
    Send("bestmove " + dfpn_mate_move.ToString());
    return;
  }

  // --- Use MCTS result ---
  if (result.best_move.is_null()) {
    Send("bestmove resign");
    return;
  }

  std::string pv_str;
  for (const auto& m : result.pv) {
    if (!pv_str.empty()) pv_str += " ";
    pv_str += m.ToString();
  }
  if (pv_str.empty()) pv_str = result.best_move.ToString();

  std::string score_str;
  if (result.mate_status == 1) {
    score_str = "score mate +";
  } else if (result.mate_status == -1) {
    score_str = "score mate -";
  } else {
    score_str = "score cp " + std::to_string(result.score_cp);
  }

  Send("info depth 1 " + score_str +
       " nodes " + std::to_string(result.nodes) +
       " time " + std::to_string(static_cast<int>(result.time_sec * 1000)) +
       " nps " + std::to_string(static_cast<int>(result.nps)) +
       " pv " + pv_str);

  Send("bestmove " + result.best_move.ToString());
}

void USIEngine::CmdGoMate(const std::vector<std::string>& parts) {
  // Parse: go mate <time_ms> | go mate infinite
  int time_limit_ms = 0;  // 0 = infinite
  for (size_t i = 1; i < parts.size(); i++) {
    if (parts[i] == "mate") {
      if (i + 1 < parts.size() && parts[i + 1] != "infinite") {
        time_limit_ms = std::stoi(parts[i + 1]);
      }
      break;
    }
  }

  // Scale node budget with time limit. The df-pn can search ~50K-200K nodes/sec.
  size_t max_nodes;
  if (time_limit_ms <= 0) {
    max_nodes = 10000000;  // infinite: 10M nodes
  } else {
    max_nodes = std::max((size_t)(time_limit_ms * 200), (size_t)100000);  // ~200K nodes/sec
  }
  MateDfpnSolver solver(max_nodes);

  // Run df-pn in a separate thread so we can enforce the time limit.
  std::atomic<bool> search_done{false};
  Move mate_move;

  auto search_thread = std::thread([&]() {
    mate_move = solver.search(board_, max_nodes);
    search_done = true;
  });

  // Wait for completion or time limit.
  auto t0 = std::chrono::steady_clock::now();
  while (!search_done) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (time_limit_ms > 0) {
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - t0).count();
      if (elapsed >= time_limit_ms) {
        solver.stop();
        break;
      }
    }
  }

  search_thread.join();

  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - t0).count();

  // Report result.
  if (!search_done && time_limit_ms > 0) {
    // Time's up before search completed.
    Log("Mate search timeout after " + std::to_string(elapsed) + " ms, " +
        std::to_string(solver.get_nodes_searched()) + " nodes");
    Send("checkmate timeout");
  } else if (!mate_move.is_null() && !MateDfpnSolver::IsNoMate(mate_move)) {
    // Mate found.
    auto pv = solver.get_pv();
    std::string pv_str;
    for (const auto& m : pv) {
      if (!pv_str.empty()) pv_str += " ";
      pv_str += m.ToString();
    }
    Log("Mate found in " + std::to_string(pv.size()) + " ply, " +
        std::to_string(solver.get_nodes_searched()) + " nodes, " +
        std::to_string(elapsed) + " ms");
    Send("checkmate " + pv_str);
  } else if (MateDfpnSolver::IsNoMate(mate_move)) {
    Log("No mate proven (" + std::to_string(solver.get_nodes_searched()) +
        " nodes, " + std::to_string(elapsed) + " ms)");
    Send("checkmate nomate");
  } else {
    // Unsolved (out of memory or nodes).
    Log("Mate search unsolved (" + std::to_string(solver.get_nodes_searched()) +
        " nodes, " + std::to_string(elapsed) + " ms)");
    Send("checkmate timeout");
  }
}

void USIEngine::CmdStop() {
  if (search_) search_->Stop();
}

void USIEngine::CmdGameOver(const std::vector<std::string>& parts) {
  if (parts.size() > 1) {
    Log("Game over: " + parts[1]);
  }
}

void USIEngine::CmdDebug() {
  Log("Position: " + board_.ToSfen());
  auto moves = board_.GenerateLegalMoves();
  Log("Legal moves: " + std::to_string(moves.size()));
}

}  // namespace jhbr2
