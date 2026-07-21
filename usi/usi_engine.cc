/*
  JHBR2 Shogi Engine — USI Protocol Implementation
  Uses the dlshogi-style MCTS search.
*/

#include "usi/usi_engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "mate/dfpn.h"
#include "shogi/encoder.h"
#include "usi/time_manager.h"

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

static std::string FormatNNCacheStats(const NNCacheStats& stats) {
  const double hit_rate = stats.lookups == 0
                              ? 0.0
                              : 100.0 * static_cast<double>(stats.hits) /
                                    static_cast<double>(stats.lookups);
  const double reuse_rate =
      stats.lookups == 0
          ? 0.0
          : 100.0 * static_cast<double>(stats.hits + stats.in_flight_waits) /
                static_cast<double>(stats.lookups);
  std::ostringstream out;
  out << "nncache size " << stats.size << "/" << stats.capacity
      << " probes " << stats.lookups << " hits " << stats.hits
      << " hitrate " << std::fixed << std::setprecision(1) << hit_rate << "%"
      << " reuse_rate " << reuse_rate << "%"
      << " inserts " << stats.inserts
      << " duplicate_inserts " << stats.duplicate_inserts
      << " evictions " << stats.evictions
      << " in_flight_owners " << stats.in_flight_owners
      << " in_flight_waits " << stats.in_flight_waits
      << " lock_contentions " << stats.lock_contentions
      << " lock_wait_us " << stats.lock_wait_ns / 1000;
  return out.str();
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
  Send("option name MinibatchSize type spin default 128 min 1 max 4096");
  Send("option name PerLeafGathering type check default true");
  Send("option name LeafDfpnNodes type spin default 10 min 0 max 10000");
  Send("option name LeafMateMode type combo default shallow var off var dfpn var shallow");
  Send("option name LeafMateDepth type spin default 5 min 1 max 7");
  Send("option name RootMateDepth type spin default 7 min 0 max 7");
  Send("option name NNCacheSize type spin default 0 min 0 max 100000000");
  Send("option name NumGPUs type spin default 1 min 1 max 8");
  Send("option name VirtualLossWeight type string default 1.0");
  Send("option name MaxMovesToDraw type spin default 100000 min 1 max 100000");
  Send("option name MovesLeftWeight type string default 0.0");
  Send("option name MovesLeftThreshold type string default 0.0");
  Send("option name MovesLeftCap type string default 20.0");
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
                                        search_config_.workers_per_gpu,
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
    search_.reset();
  } else if (name_lower == "modelformat") {
    model_format_ = ParseModelFormat(value);
    evaluators_.clear();
    search_.reset();
  } else if (name_lower == "dlshogimodel") {
    model_format_ = (value == "true") ? ModelFormat::kDlshogi
                                      : ModelFormat::kAuto;
    evaluators_.clear();
    search_.reset();
  } else if (name_lower == "noiseepsilon") {
    noise_epsilon_ = std::stof(value);
    // dlshogi-style search does not inject root noise in USI play.
  } else if (name_lower == "usegpu") {
    use_gpu_ = (value == "true");
  } else if (name_lower == "threads") {
    // Backward-compat alias for WorkersPerGpu.
    int n = std::stoi(value);
    search_config_.workers_per_gpu = n;
    search_.reset();
    evaluators_.clear();
  } else if (name_lower == "workerspergpu") {
    int n = std::stoi(value);
    if (n < 1) n = 1;
    search_config_.workers_per_gpu = n;
    // Each evaluator must be (re-)constructed with this many TRT
    // execution slots, so drop both Search and evaluators.
    search_.reset();
    evaluators_.clear();
  } else if (name_lower == "minibatchsize") {
    search_config_.minibatch_size =
        std::clamp(std::stoi(value), 1, 4096);
    search_.reset();
  } else if (name_lower == "perleafgathering") {
    // Per-leaf gathering is always enabled by the dlshogi-style worker loop.
  } else if (name_lower == "leafdfpnnodes") {
    // df-pn at MCTS leaves is not part of this backend; use shallow mate.
  } else if (name_lower == "leafmatemode") {
    std::string v = value;
    for (auto& c : v) c = std::tolower(c);
    if (v == "shallow") {
      if (search_config_.leaf_mate_depth <= 0 ||
          search_config_.leaf_mate_depth % 2 == 0) {
        search_config_.leaf_mate_depth = 5;
      }
    } else {
      search_config_.leaf_mate_depth = 0;
    }
    search_.reset();
  } else if (name_lower == "leafmatedepth") {
    int d = std::stoi(value);
    // Clamp to supported odd values: 1, 3, 5, 7.
    if (d < 1) d = 1;
    if (d > 7) d = 7;
    if (d % 2 == 0) d -= 1;          // round even down to odd
    search_config_.leaf_mate_depth = d;
    search_.reset();
  } else if (name_lower == "rootmatedepth") {
    int d = std::stoi(value);
    if (d < 0) d = 0;
    if (d > 7) d = 7;
    if (d > 0 && d % 2 == 0) d -= 1;
    search_config_.root_mate_depth = d;
    search_.reset();
  } else if (name_lower == "nncachesize") {
    search_config_.nn_cache_size = static_cast<size_t>(std::stoull(value));
    // The cache is owned by the persistent Search object, so rebuild it when
    // capacity changes. Evaluators can stay loaded.
    search_.reset();
  } else if (name_lower == "numgpus") {
    num_gpus_ = std::stoi(value);
    search_config_.num_gpus = num_gpus_;
    evaluators_.clear();
    search_.reset();
  } else if (name_lower == "maxmovestodraw") {
    int n = std::stoi(value);
    if (n < 1) n = 1;
    search_config_.max_moves_to_draw = n;
  } else if (name_lower == "movesleftweight") {
    search_config_.moves_left_weight = std::max(0.0f, std::stof(value));
  } else if (name_lower == "movesleftthreshold") {
    search_config_.moves_left_threshold =
        std::clamp(std::stof(value), 0.0f, 0.5f);
  } else if (name_lower == "movesleftcap") {
    search_config_.moves_left_cap = std::max(0.0f, std::stof(value));
  } else if (name_lower == "virtuallossweight") {
    float w = std::stof(value);
    if (w < 0.1f) w = 0.1f;
    if (w > 100.0f) w = 100.0f;
    (void)w;
  } else if (name_lower == "maxgpubatch") {
    Log("MaxGpuBatch is deprecated and ignored");
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
  search_.reset();
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
      nodes_limit = 10000000; i++;
    } else if (parts[i] == "mate") {
      CmdGoMate(parts);
      return;
    } else if (parts[i] == "ponder") {
      i++;
    } else {
      i++;
    }
  }

  TimeControl time_control;
  time_control.main_time_ms =
      board_.side_to_move() == BLACK ? btime : wtime;
  time_control.increment_ms =
      board_.side_to_move() == BLACK ? binc : winc;
  time_control.byoyomi_ms = byoyomi;
  time_control.has_main_time = btime > 0 || wtime > 0;

  TimeOptions time_options;
  time_options.max_move_time_ms = max_move_time_ms_;
  time_options.max_move_time_1m_ms = max_move_time_1m_ms_;
  time_options.dfpn_max_time_ms = dfpn_max_time_ms_;
  const TimeBudget time_budget =
      TimeManager::Compute(time_control, time_options);

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

  // Configure the dlshogi-style MCTS search.
  search_config_.max_nodes = nodes_limit;
  search_config_.max_time = time_budget.mcts_time_seconds;

  // --- Launch root df-pn in parallel ---
  auto move_start_time = std::chrono::steady_clock::now();

  // Keep the solver and its result in one state shared by the worker and
  // deadline timer.
  struct DfpnState {
    MateDfpnSolver solver;
    std::atomic<bool> done{false};
    Move mate_move;
    ShogiBoard board;
    DfpnState(size_t nodes, const ShogiBoard& b) : solver(nodes), board(b) {}
  };
  auto dfpn =
      std::make_shared<DfpnState>(time_budget.root_dfpn_nodes, board_);

  auto dfpn_thread =
      std::thread([dfpn, root_dfpn_nodes = time_budget.root_dfpn_nodes]() {
        dfpn->mate_move = dfpn->solver.search(dfpn->board, root_dfpn_nodes);
        dfpn->done.store(true, std::memory_order_release);
      });

  // Enforce DfPnMaxTime independently of MCTS duration. Cancellation is
  // atomic and checked throughout df-pn, so the worker can always be joined
  // safely instead of being detached while its result is still shared.
  auto dfpn_timer = std::thread(
      [dfpn, root_dfpn_time_ms = time_budget.root_dfpn_time_ms,
       move_start_time]() {
        while (!dfpn->done.load(std::memory_order_acquire)) {
          const auto elapsed_ms =
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - move_start_time)
                  .count();
          if (elapsed_ms >= root_dfpn_time_ms) {
            dfpn->solver.stop();
            return;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
      });

  // --- Run dlshogi-style MCTS ---
  // Set info callback for periodic GUI output during search.
  search_config_.info_callback = [this](const dlshogi_mcts::SearchInfo& info) {
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
    if (info.nn_cache.capacity > 0) {
      Log(FormatNNCacheStats(info.nn_cache));
    }
  };

  // Persistent Search object across `go` commands.
  if (!search_) {
    std::vector<jhbr2::NNEvaluator*> eval_ptrs;
    for (auto& e : evaluators_) eval_ptrs.push_back(e.get());
    search_ =
        std::make_unique<dlshogi_mcts::Search>(eval_ptrs, search_config_);
  }
  // Search holds its own config snapshot — push per-move
  // updates so max_time / max_nodes reflect THIS go command, not the
  // first one ever issued.
  search_->SetMaxTime(search_config_.max_time);
  search_->SetMaxNodes(search_config_.max_nodes);

  // Watchdog: hard deadline enforcement for timed searches. Pure node-limited
  // searches intentionally have no implicit time cap.
  std::atomic<bool> search_done{false};
  std::thread watchdog;
  if (time_budget.hard_deadline_ms > 0) {
    watchdog = std::thread(
        [this, &search_done,
         hard_deadline_ms = time_budget.hard_deadline_ms, move_start_time]() {
          while (!search_done.load(std::memory_order_acquire)) {
            auto elapsed_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - move_start_time)
                    .count();
            if (elapsed_ms >= hard_deadline_ms) {
              if (search_) search_->Stop();
              return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
          }
        });
  }

  auto result =
      search_->Run(board_, position_start_key_, position_moves_, game_ply_);
  search_done.store(true, std::memory_order_release);
  if (watchdog.joinable()) watchdog.join();

  // If MCTS finishes first, preserve the original clock-scaled grace period
  // for root df-pn. The grace period is capped by DfPnMaxTime and by the move
  // deadline, so it cannot recreate the old tournament time overruns.
  auto dfpn_wait_deadline =
      std::chrono::steady_clock::now() +
      std::chrono::milliseconds(time_budget.root_dfpn_grace_ms);
  const auto dfpn_time_deadline =
      move_start_time +
      std::chrono::milliseconds(time_budget.root_dfpn_time_ms);
  dfpn_wait_deadline = std::min(dfpn_wait_deadline, dfpn_time_deadline);
  if (time_budget.hard_deadline_ms > 0) {
    const auto move_deadline =
        move_start_time +
        std::chrono::milliseconds(time_budget.hard_deadline_ms);
    dfpn_wait_deadline = std::min(dfpn_wait_deadline, move_deadline);
  }

  while (!dfpn->done.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < dfpn_wait_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  if (!dfpn->done.load(std::memory_order_acquire)) dfpn->solver.stop();
  if (dfpn_thread.joinable()) dfpn_thread.join();
  if (dfpn_timer.joinable()) dfpn_timer.join();

  // --- Choose result ---
  bool use_mate = dfpn->done.load(std::memory_order_acquire) &&
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
  if (result.nn_cache.capacity > 0) {
    Log(FormatNNCacheStats(result.nn_cache));
  }

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
  if (search_) search_->Stop();
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
