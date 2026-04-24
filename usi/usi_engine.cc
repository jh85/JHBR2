/*
  JHBR2 Shogi Engine — USI Protocol Implementation
  Now using lc0-style MCTS search.
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

static std::string ToLower(const std::string& s) {
  std::string r = s;
  std::transform(r.begin(), r.end(), r.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return r;
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

  Send("option name MaxNodes type spin default 800 min 1 max 10000000");
  Send("option name OnnxModel type string default shogi_bt4.onnx");
  Send("option name NoiseEpsilon type string default 0.0");
  Send("option name UseGPU type check default true");
  Send("option name Threads type spin default 1 min 1 max 128");
  Send("option name MinibatchSize type spin default 32 min 1 max 256");
  Send("option name PerLeafGathering type check default true");
  Send("option name LeafDfpnNodes type spin default 10 min 0 max 10000");
  Send("option name DfPnMaxTime type spin default 4000 min 100 max 60000");
  Send("option name MaxMoveTime type spin default 0 min 0 max 300000");

  Send("usiok");
}

void USIEngine::CmdIsReady() {
  if (!evaluator_) {
    Log("Loading model: " + onnx_path_);

    ShogiEncoderTables::Init();

    evaluator_ = std::make_unique<NNEvaluator>(onnx_path_, use_gpu_);

    Log("Model loaded, max_nodes=" + std::to_string(max_nodes_));
  }
  Send("readyok");
}

void USIEngine::CmdSetOption(const std::vector<std::string>& parts) {
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
  } else if (name_lower == "onnxmodel") {
    onnx_path_ = value;
  } else if (name_lower == "noiseepsilon") {
    noise_epsilon_ = std::stof(value);
    lc0_config_.noise_epsilon = noise_epsilon_;
  } else if (name_lower == "usegpu") {
    use_gpu_ = (value == "true");
  } else if (name_lower == "threads") {
    lc0_config_.num_threads = std::stoi(value);
  } else if (name_lower == "minibatchsize") {
    lc0_config_.minibatch_size = std::stoi(value);
  } else if (name_lower == "perleafgathering") {
    lc0_config_.per_leaf_gathering = (value == "true");
  } else if (name_lower == "leafdfpnnodes") {
    lc0_config_.leaf_dfpn_nodes = std::stoi(value);
  } else if (name_lower == "dfpnmaxtime") {
    dfpn_max_time_ms_ = std::stoi(value);
  } else if (name_lower == "maxmovetime") {
    max_move_time_ms_ = std::stoi(value);
  }

  Log("Set " + name + " = " + value);
}

void USIEngine::CmdUsiNewGame() {
  board_.SetStartPos();
  board_.ClearHistory();
  game_ply_ = 0;
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
    std::string sfen;
    while (idx < parts.size() && parts[idx] != "moves") {
      if (!sfen.empty()) sfen += " ";
      sfen += parts[idx];
      idx++;
    }
    board_.SetFromSfen(sfen);
  }

  if (idx < parts.size() && parts[idx] == "moves") {
    idx++;
    while (idx < parts.size()) {
      Move m = Move::Parse(parts[idx]);
      board_.DoMove(m);
      idx++;
    }
  }

  game_ply_ = board_.RepetitionCount();
}

void USIEngine::CmdGo(const std::vector<std::string>& parts) {
  if (!evaluator_) {
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
      max_time = 0; nodes_limit = 10000000; i++;
    } else if (parts[i] == "mate") {
      CmdGoMate(parts);
      return;
    } else if (parts[i] == "ponder") {
      i++;
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

  if (max_move_time_ms_ > 0) {
    float cap = std::max(max_move_time_ms_ / 1000.0f - 3.0f, 0.5f);
    if (max_time <= 0.0f || cap < max_time) max_time = cap;
  }

  // Check entering-king declaration.
  if (board_.CanDeclareWin()) {
    Send("bestmove win");
    return;
  }

  // Configure lc0 MCTS search.
  lc0_config_.max_nodes = nodes_limit;
  lc0_config_.max_time = max_time;

  // --- Launch root df-pn in parallel ---
  int my_time_ms = (board_.side_to_move() == BLACK) ? btime : wtime;
  int my_inc_ms = (board_.side_to_move() == BLACK) ? binc : winc;
  int available_ms = my_time_ms + my_inc_ms + byoyomi;

  int dfpn_min_wait_ms;
  size_t root_dfpn_nodes;
  if (available_ms <= 0) {
    dfpn_min_wait_ms = 300; root_dfpn_nodes = 100000;
  } else if (available_ms < 10000) {
    dfpn_min_wait_ms = 100; root_dfpn_nodes = 10000;
  } else if (available_ms < 60000) {
    dfpn_min_wait_ms = 300; root_dfpn_nodes = 100000;
  } else if (available_ms < 300000) {
    dfpn_min_wait_ms = 500; root_dfpn_nodes = 500000;
  } else {
    dfpn_min_wait_ms = 1000; root_dfpn_nodes = 2000000;
  }

  auto move_start_time = std::chrono::steady_clock::now();
  int hard_deadline_ms = (max_move_time_ms_ > 0)
      ? max_move_time_ms_
      : static_cast<int>(max_time * 1000) + 2000;

  MateDfpnSolver root_dfpn(root_dfpn_nodes);
  std::atomic<bool> dfpn_done{false};
  Move dfpn_mate_move;
  ShogiBoard dfpn_board = board_;

  auto dfpn_thread = std::thread([&]() {
    dfpn_mate_move = root_dfpn.search(dfpn_board, root_dfpn_nodes);
    dfpn_done = true;
  });

  // --- Run lc0-style MCTS ---
  lc0_search_ = std::make_unique<lc0_shogi::Search>(*evaluator_, lc0_config_);
  auto result = lc0_search_->Run(board_, game_ply_);

  // --- Stop df-pn and wait ---
  root_dfpn.stop();

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
    root_dfpn.stop();
  }
  dfpn_thread.join();

  // --- Choose result ---
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
    Log("Root df-pn found mate in " + std::to_string(mate_ply) + " ply");

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

  Send("info depth 1 score cp " + std::to_string(result.score_cp) +
       " nodes " + std::to_string(result.nodes) +
       " time " + std::to_string(static_cast<int>(result.time_sec * 1000)) +
       " nps " + std::to_string(static_cast<int>(result.nps)) +
       " pv " + pv_str);

  Send("bestmove " + result.best_move.ToString());
}

void USIEngine::CmdGoMate(const std::vector<std::string>& parts) {
  int time_limit_ms = 0;
  for (size_t i = 1; i < parts.size(); i++) {
    if (parts[i] == "mate") {
      if (i + 1 < parts.size() && parts[i + 1] != "infinite") {
        time_limit_ms = std::stoi(parts[i + 1]);
      }
      break;
    }
  }

  size_t max_nodes;
  if (time_limit_ms <= 0) {
    max_nodes = 10000000;
  } else {
    max_nodes = std::max((size_t)(time_limit_ms * 200), (size_t)100000);
  }
  MateDfpnSolver solver(max_nodes);

  std::atomic<bool> search_done{false};
  Move mate_move;

  auto search_thread = std::thread([&]() {
    mate_move = solver.search(board_, max_nodes);
    search_done = true;
  });

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

  if (!search_done && time_limit_ms > 0) {
    Log("Mate search timeout after " + std::to_string(elapsed) + " ms");
    Send("checkmate timeout");
  } else if (!mate_move.is_null() && !MateDfpnSolver::IsNoMate(mate_move)) {
    auto pv = solver.get_pv();
    std::string pv_str;
    for (const auto& m : pv) {
      if (!pv_str.empty()) pv_str += " ";
      pv_str += m.ToString();
    }
    Log("Mate found in " + std::to_string(pv.size()) + " ply");
    Send("checkmate " + pv_str);
  } else if (MateDfpnSolver::IsNoMate(mate_move)) {
    Send("checkmate nomate");
  } else {
    Send("checkmate timeout");
  }
}

void USIEngine::CmdStop() {
  if (lc0_search_) lc0_search_->Stop();
}

void USIEngine::CmdGameOver(const std::vector<std::string>& parts) {
  if (parts.size() > 1) Log("Game over: " + parts[1]);
}

void USIEngine::CmdDebug() {
  Log("Position: " + board_.ToSfen());
  auto moves = board_.GenerateLegalMoves();
  Log("Legal moves: " + std::to_string(moves.size()));
}

}  // namespace jhbr2
