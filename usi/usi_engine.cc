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

static ModelFormat ParseModelFormat(const std::string& s) {
  std::string v = ToLower(s);
  if (v == "dlshogi" || v == "dlshogimodel") return ModelFormat::kDlshogi;
  if (v == "jhbr2" || v == "default") return ModelFormat::kJHBR2;
  return ModelFormat::kAuto;
}

static std::string ModelFormatToString(ModelFormat format) {
  switch (format) {
    case ModelFormat::kAuto:
      return "auto";
    case ModelFormat::kJHBR2:
      return "jhbr2";
    case ModelFormat::kDlshogi:
      return "dlshogi";
  }
  return "auto";
}

// =====================================================================
// Constructor
// =====================================================================

USIEngine::USIEngine() {
  board_.SetStartPos();
  position_start_key_ = board_.Hash();
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
  Send("option name ModelFormat type combo default auto var auto var jhbr2 var dlshogi");
  Send("option name DlshogiModel type check default false");
  Send("option name NoiseEpsilon type string default 0.0");
  Send("option name UseGPU type check default true");
  // Threads is kept as an alias for WorkersPerGpu (backward compat).
  Send("option name Threads type spin default 2 min 1 max 8");
  Send("option name WorkersPerGpu type spin default 2 min 1 max 8");
  Send("option name MinibatchSize type spin default 256 min 1 max 4096");
  Send("option name PerLeafGathering type check default true");
  Send("option name LeafDfpnNodes type spin default 10 min 0 max 10000");
  Send("option name LeafMateMode type combo default dfpn var off var dfpn var shallow");
  Send("option name LeafMateDepth type spin default 3 min 1 max 7");
  Send("option name NNCacheSize type spin default 0 min 0 max 100000000");
  Send("option name NumGPUs type spin default 1 min 1 max 8");
  // MaxGpuBatch sets the TRT engine's max profile shape (and the
  // per-slot buffer size). With per-worker submission there is no
  // combining, so this should match MinibatchSize.
  Send("option name MaxGpuBatch type spin default 1024 min 64 max 16384");
  Send("option name VirtualLossWeight type string default 1.0");
  Send("option name MaxMovesToDraw type spin default 100000 min 1 max 100000");
  Send("option name DfPnMaxTime type spin default 4000 min 100 max 60000");
  Send("option name MaxMoveTime type spin default 0 min 0 max 300000");
  Send("option name MaxMoveTime1m type spin default 0 min 0 max 60000");
  Send("option name BookFile type string default ");
  Send("option name BookOnTheFly type check default false");

  Send("usiok");
}

void USIEngine::CmdIsReady() {
  if (evaluators_.empty()) {
    ShogiEncoderTables::Init();

    for (int g = 0; g < num_gpus_; g++) {
      Log("Loading model on GPU " + std::to_string(g) + ": " + onnx_path_);
      evaluators_.push_back(
          std::make_unique<NNEvaluator>(onnx_path_, use_gpu_, g,
                                        max_gpu_batch_,
                                        lc0_config_.workers_per_gpu,
                                        model_format_));
    }

    Log("Model loaded, GPUs=" + std::to_string(num_gpus_) +
        ", format=" + ModelFormatToString(model_format_) +
        ", max_nodes=" + std::to_string(max_nodes_));

    // Load opening book if specified.
    if (!book_path_.empty()) {
      int book_count = book_.Load(book_path_, book_on_the_fly_);
      if (book_on_the_fly_) {
        Log("Book on-the-fly: " + book_path_);
      } else {
        Log("Book loaded: " + std::to_string(book_count) + " positions from " + book_path_);
      }
    }
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
    evaluators_.clear();
    lc0_search_.reset();
  } else if (name_lower == "modelformat") {
    model_format_ = ParseModelFormat(value);
    evaluators_.clear();
    lc0_search_.reset();
  } else if (name_lower == "dlshogimodel") {
    model_format_ = (value == "true") ? ModelFormat::kDlshogi
                                      : ModelFormat::kAuto;
    evaluators_.clear();
    lc0_search_.reset();
  } else if (name_lower == "noiseepsilon") {
    noise_epsilon_ = std::stof(value);
    // dlshogi-style search does not inject root noise in USI play.
  } else if (name_lower == "usegpu") {
    use_gpu_ = (value == "true");
  } else if (name_lower == "threads") {
    // Backward-compat alias for WorkersPerGpu.
    int n = std::stoi(value);
    lc0_config_.workers_per_gpu = n;
    lc0_search_.reset();
    evaluators_.clear();
  } else if (name_lower == "workerspergpu") {
    int n = std::stoi(value);
    if (n < 1) n = 1;
    lc0_config_.workers_per_gpu = n;
    // Each evaluator must be (re-)constructed with this many TRT
    // execution slots, so drop both Search and evaluators.
    lc0_search_.reset();
    evaluators_.clear();
  } else if (name_lower == "minibatchsize") {
    lc0_config_.minibatch_size = std::stoi(value);
    lc0_search_.reset();
  } else if (name_lower == "perleafgathering") {
    // Per-leaf gathering is always enabled by the dlshogi-style worker loop.
  } else if (name_lower == "leafdfpnnodes") {
    // df-pn at MCTS leaves is not part of this backend; use shallow mate.
  } else if (name_lower == "leafmatemode") {
    std::string v = value;
    for (auto& c : v) c = std::tolower(c);
    if (v == "shallow") {
      if (lc0_config_.leaf_mate_depth <= 0 ||
          lc0_config_.leaf_mate_depth % 2 == 0) {
        lc0_config_.leaf_mate_depth = 5;
      }
    } else {
      lc0_config_.leaf_mate_depth = 0;
    }
  } else if (name_lower == "leafmatedepth") {
    int d = std::stoi(value);
    // Clamp to supported odd values: 1, 3, 5, 7.
    if (d < 1) d = 1;
    if (d > 7) d = 7;
    if (d % 2 == 0) d -= 1;          // round even down to odd
    lc0_config_.leaf_mate_depth = d;
  } else if (name_lower == "nncachesize") {
    lc0_config_.nn_cache_size = static_cast<size_t>(std::stoull(value));
    // The cache is owned by the persistent Search object, so rebuild it when
    // capacity changes. Evaluators can stay loaded.
    lc0_search_.reset();
  } else if (name_lower == "numgpus") {
    num_gpus_ = std::stoi(value);
    lc0_config_.num_gpus = num_gpus_;
    evaluators_.clear();
    lc0_search_.reset();
  } else if (name_lower == "maxmovestodraw") {
    int n = std::stoi(value);
    if (n < 1) n = 1;
    lc0_config_.max_moves_to_draw = n;
  } else if (name_lower == "virtuallossweight") {
    float w = std::stof(value);
    if (w < 0.1f) w = 0.1f;
    if (w > 100.0f) w = 100.0f;
    (void)w;
  } else if (name_lower == "maxgpubatch") {
    max_gpu_batch_ = std::stoi(value);
    // Evaluators bake the TRT max_shapes profile at construction time, so
    // a change here only takes effect after isready rebuilds them.
    evaluators_.clear();
    lc0_search_.reset();
  } else if (name_lower == "dfpnmaxtime") {
    dfpn_max_time_ms_ = std::stoi(value);
  } else if (name_lower == "maxmovetime") {
    max_move_time_ms_ = std::stoi(value);
  } else if (name_lower == "maxmovetime1m") {
    max_move_time_1m_ms_ = std::stoi(value);
  } else if (name_lower == "bookfile") {
    book_path_ = value;
  } else if (name_lower == "bookonthefly") {
    book_on_the_fly_ = (value == "true");
  }

  Log("Set " + name + " = " + value);
}

void USIEngine::CmdUsiNewGame() {
  board_.SetStartPos();
  board_.ClearHistory();
  game_ply_ = 0;
  position_start_key_ = board_.Hash();
  position_moves_.clear();
  // Drop the Search object so the next `go` rebuilds it with a fresh
  // tree. Otherwise tree reuse would carry over visit counts from the
  // previous game's positions, which is incorrect.
  lc0_search_.reset();
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

  position_start_key_ = board_.Hash();
  position_moves_.clear();

  if (idx < parts.size() && parts[idx] == "moves") {
    idx++;
    while (idx < parts.size()) {
      Move m = Move::Parse(parts[idx]);
      board_.DoMove(m);
      position_moves_.push_back(m);
      idx++;
    }
  }

  game_ply_ = board_.ply();
}

void USIEngine::CmdGo(const std::vector<std::string>& parts) {
  if (evaluators_.empty()) {
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

  {
    int my_time = (board_.side_to_move() == BLACK) ? btime : wtime;
    int cap_ms = max_move_time_ms_;
    if (max_move_time_1m_ms_ > 0 && my_time > 0 && my_time < 60000)
      cap_ms = max_move_time_1m_ms_;
    if (cap_ms > 0) {
      float cap = std::max(cap_ms / 1000.0f - 0.5f, 0.5f);
      if (max_time <= 0.0f || cap < max_time) max_time = cap;
    }
  }

  // Check entering-king declaration.
  if (board_.CanDeclareWin()) {
    Send("bestmove win");
    return;
  }

  // Probe opening book.
  if (book_.is_loaded()) {
    auto* entry = book_.Probe(board_.ToSfen());
    if (entry) {
      Log("Book hit: " + entry->move_usi + " (eval=" +
          std::to_string(entry->eval) + ", depth=" +
          std::to_string(entry->depth) + ")");
      std::string response = "bestmove " + entry->move_usi;
      if (entry->ponder_usi != "none" && !entry->ponder_usi.empty()) {
        response += " ponder " + entry->ponder_usi;
      }
      Send(response);
      return;
    }
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
  int hard_deadline_ms = 0;
  if (max_move_time_ms_ > 0) {
    hard_deadline_ms = max_move_time_ms_;
  } else if (max_time > 0.0f) {
    hard_deadline_ms = static_cast<int>(max_time * 1000) + 2000;
  }

  // Use shared_ptr so detached thread doesn't access destroyed locals.
  struct DfpnState {
    MateDfpnSolver solver;
    std::atomic<bool> done{false};
    Move mate_move;
    ShogiBoard board;
    DfpnState(size_t nodes, const ShogiBoard& b) : solver(nodes), board(b) {}
  };
  auto dfpn = std::make_shared<DfpnState>(root_dfpn_nodes, board_);

  auto dfpn_thread = std::thread([dfpn, root_dfpn_nodes]() {
    dfpn->mate_move = dfpn->solver.search(dfpn->board, root_dfpn_nodes);
    dfpn->done = true;
  });

  // --- Run dlshogi-style MCTS ---
  // Set info callback for periodic GUI output during search.
  lc0_config_.info_callback = [this](const dlshogi_mcts::SearchInfo& info) {
    std::string pv_str;
    for (const auto& m : info.pv) {
      if (!pv_str.empty()) pv_str += " ";
      pv_str += m.ToString();
    }
    Send("info depth " + std::to_string(info.depth) +
         " score cp " + std::to_string(info.score_cp) +
         " nodes " + std::to_string(info.nodes) +
         " nps " + std::to_string(info.nps) +
         " time " + std::to_string(info.time_ms) +
         (pv_str.empty() ? "" : " pv " + pv_str));
  };

  // Persistent Search object across `go` commands.
  if (!lc0_search_) {
    std::vector<jhbr2::NNEvaluator*> eval_ptrs;
    for (auto& e : evaluators_) eval_ptrs.push_back(e.get());
    lc0_search_ = std::make_unique<dlshogi_mcts::Search>(eval_ptrs, lc0_config_);
  }
  // Search holds its own config snapshot — push per-move
  // updates so max_time / max_nodes reflect THIS go command, not the
  // first one ever issued.
  lc0_search_->SetMaxTime(lc0_config_.max_time);
  lc0_search_->SetMaxNodes(lc0_config_.max_nodes);

  // Watchdog: hard deadline enforcement for timed searches. Pure node-limited
  // searches intentionally have no implicit time cap.
  std::atomic<bool> search_done{false};
  std::thread watchdog;
  if (hard_deadline_ms > 0) {
    watchdog = std::thread([this, &search_done, hard_deadline_ms,
                            move_start_time]() {
      while (!search_done.load(std::memory_order_acquire)) {
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - move_start_time).count();
        if (elapsed_ms >= hard_deadline_ms) {
          if (lc0_search_) lc0_search_->Stop();
          return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    });
  }

  auto result = lc0_search_->Run(board_, position_start_key_, position_moves_,
                                 game_ply_);
  search_done.store(true, std::memory_order_release);
  if (watchdog.joinable()) watchdog.join();

  // --- Stop df-pn and wait ---
  dfpn->solver.stop();

  if (hard_deadline_ms <= 0) {
    if (dfpn_thread.joinable()) dfpn_thread.join();
  } else {
    // Wait for df-pn with hard deadline — never exceed MaxMoveTime.
    auto wait_start = std::chrono::steady_clock::now();
    int max_wait_ms = std::max(hard_deadline_ms - (int)std::chrono::duration_cast<
        std::chrono::milliseconds>(wait_start - move_start_time).count(), 100);
    while (!dfpn->done) {
      auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - wait_start).count();
      if (elapsed_ms >= max_wait_ms) {
        // df-pn still running past deadline — detach and abandon.
        // shared_ptr keeps DfpnState alive until the thread finishes.
        dfpn_thread.detach();
        dfpn->done = true;  // Pretend it's done — don't use its result.
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (dfpn_thread.joinable()) dfpn_thread.join();
  }

  // --- Choose result ---
  bool use_mate = dfpn->done &&
                  !dfpn->mate_move.is_null() &&
                  !MateDfpnSolver::IsNoMate(dfpn->mate_move);

  if (use_mate) {
    auto pv = dfpn->solver.get_pv();
    std::string pv_str;
    for (const auto& m : pv) {
      if (!pv_str.empty()) pv_str += " ";
      pv_str += m.ToString();
    }
    if (pv_str.empty()) pv_str = dfpn->mate_move.ToString();

    int mate_ply = (int)pv.size();
    Log("Root df-pn found mate in " + std::to_string(mate_ply) + " ply");

    Send("info depth 1 score mate " + std::to_string((mate_ply + 1) / 2) +
         " nodes " + std::to_string(dfpn->solver.get_nodes_searched()) +
         " pv " + pv_str);
    Send("bestmove " + dfpn->mate_move.ToString());
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

  int pv_depth = static_cast<int>(result.pv.size());
  Send("info depth " + std::to_string(std::max(pv_depth, 1)) +
       " score cp " + std::to_string(result.score_cp) +
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
