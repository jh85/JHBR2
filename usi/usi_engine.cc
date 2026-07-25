/*
  JHBR2 Shogi Engine — USI Protocol Implementation
  Uses the dlshogi-style MCTS search.
*/

#include "usi/usi_engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "book/book_selection.h"
#include "mate/dfpn.h"
#include "shogi/encoder.h"
#include "usi/search_info.h"
#include "usi/time_manager.h"

namespace jhbr2 {

using namespace lczero;

namespace {

// TensorRT creates one execution context and one set of buffers per worker.
// Production has successfully used 16 workers/GPU on RTX 5090. Keep a
// generous USI safety ceiling so higher-end hardware can be benchmarked,
// while still preventing an accidental unbounded allocation.
constexpr int kMaxWorkersPerGpu = 64;

const char* RepetitionResultName(ShogiBoard::RepetitionResult result) {
  switch (result) {
    case ShogiBoard::RepetitionResult::kNone:
      return "none";
    case ShogiBoard::RepetitionResult::kDraw:
      return "draw";
    case ShogiBoard::RepetitionResult::kWin:
      return "opponent-win";
    case ShogiBoard::RepetitionResult::kLoss:
      return "opponent-loss";
  }
  return "unknown";
}

ShogiBoard::RepetitionResult RootMoveRepetitionResult(
    const ShogiBoard& board, Move move) {
  ShogiBoard child = board;
  child.DoMove(move);
  return child.CheckRepetition(MateDfpnSolver::kRepetitionLookbackPly);
}

}  // namespace

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

static int ParseInt(const std::string& value) {
  std::size_t consumed = 0;
  const long long parsed = std::stoll(value, &consumed);
  if (consumed != value.size() ||
      parsed < std::numeric_limits<int>::min() ||
      parsed > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("expected an integer");
  }
  return static_cast<int>(parsed);
}

static std::size_t ParseSize(const std::string& value) {
  if (value.empty() || value.front() == '-') {
    throw std::invalid_argument("expected a non-negative integer");
  }
  std::size_t consumed = 0;
  const unsigned long long parsed = std::stoull(value, &consumed);
  if (consumed != value.size() ||
      parsed > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument("expected a non-negative integer");
  }
  return static_cast<std::size_t>(parsed);
}

static float ParseFiniteFloat(const std::string& value) {
  std::size_t consumed = 0;
  const float parsed = std::stof(value, &consumed);
  if (consumed != value.size() || !std::isfinite(parsed)) {
    throw std::invalid_argument("expected a finite number");
  }
  return parsed;
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

// USI hashfull is expressed in permill.  JHBR2 does not have a fixed-size
// MCTS node arena like dlshogi, so its bounded NN position cache is the only
// meaningful hash-table occupancy to report.
static int NNCacheHashfull(const NNCacheStats& stats) {
  if (stats.capacity == 0) return 0;
  const size_t used = std::min(stats.size, stats.capacity);
  return static_cast<int>(used * 1000 / stats.capacity);
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

void USIEngine::EnsureSearch() {
  if (search_ || evaluators_.empty()) return;

  std::vector<jhbr2::NNEvaluator*> eval_ptrs;
  eval_ptrs.reserve(evaluators_.size());
  for (auto& evaluator : evaluators_) eval_ptrs.push_back(evaluator.get());
  search_ =
      std::make_unique<dlshogi_mcts::Search>(eval_ptrs, search_config_);
}

void USIEngine::CmdUsi() {
  Send(std::string("id name ") + ENGINE_NAME);
  Send(std::string("id author ") + ENGINE_AUTHOR);

  Send("option name MaxNodes type spin default 800 min 1 max 10000000");
  Send("option name OnnxModel type string default shogi_bt4.onnx");
  Send("option name ModelFormat type combo default auto var auto var jhbr2 var dlshogi");
  Send("option name DlshogiModel type check default false");
  Send("option name UseGPU type check default true");
  // Threads is kept as an alias for WorkersPerGpu (backward compat).
  Send("option name Threads type spin default 2 min 1 max 64");
  Send("option name WorkersPerGpu type spin default 2 min 1 max 64");
  Send("option name MinibatchSize type spin default 128 min 1 max 4096");
  Send("option name CInit type string default 1.25");
  Send("option name CBase type string default 19652.0");
  Send("option name FpuReduction type string default 0.27");
  Send("option name CInitRoot type string default 1.25");
  Send("option name CBaseRoot type string default 19652.0");
  Send("option name FpuReductionRoot type string default 0.0");
  Send("option name DrawValueBlack type string default 0.5");
  Send("option name DrawValueWhite type string default 0.5");
  Send("option name ResignThreshold type string default 0.01");
  Send("option name InfoIntervalMs type spin default 1000 min 100 max 10000");
  Send("option name LeafMateMode type combo default shallow var off var shallow");
  Send("option name LeafMateDepth type spin default 5 min 1 max 7");
  Send("option name RootMateDepth type spin default 7 min 0 max 7");
  Send("option name NNCacheSize type spin default 0 min 0 max 100000000");
  Send("option name NumGPUs type spin default 1 min 1 max 8");
  Send("option name MaxMovesToDraw type spin default 100000 min 1 max 100000");
  Send("option name MovesLeftWeight type string default 0.0");
  Send("option name MovesLeftThreshold type string default 0.0");
  Send("option name MovesLeftCap type string default 20.0");
  Send("option name DfPnMaxTime type spin default 4000 min 100 max 60000");
  Send("option name MaxMoveTime type spin default 0 min 0 max 300000");
  Send("option name MaxMoveTime1m type spin default 0 min 0 max 60000");
  Send("option name BookFile type string default ");
  Send("option name UseGoteExitBook type check default false");
  Send("option name GoteExitBookFile type string default "
       "user_book1_gote_exit.ybb");

  Send("usiok");
}

void USIEngine::CmdIsReady() {
  if (evaluators_.empty()) {
    ShogiEncoderTables::Init();

    try {
      for (int g = 0; g < num_gpus_; g++) {
        Log("Loading model on GPU " + std::to_string(g) + ": " + onnx_path_);
        auto evaluator =
            std::make_unique<NNEvaluator>(onnx_path_, use_gpu_, g,
                                          search_config_.workers_per_gpu,
                                          model_format_);
        if (evaluator->num_slots() == 0) {
          throw std::runtime_error("inference backend created no worker slots");
        }
        evaluators_.push_back(std::move(evaluator));
      }
    } catch (const std::exception& error) {
      evaluators_.clear();
      Log("Model load failed: " + std::string(error.what()));
      Send("readyok");
      return;
    }

    Log("Model loaded, GPUs=" + std::to_string(num_gpus_) +
        ", format=" + ModelFormatToString(model_format_) +
        ", max_nodes=" + std::to_string(max_nodes_));

  }

  // isready is the acknowledged per-game preparation barrier. Reuse the
  // Search allocation and GPU workers, but clear the previous game's tree and
  // NN entries before readyok so cleanup is never charged to a timed move.
  if (search_) {
    search_->PrepareForNewGame();
  } else {
    EnsureSearch();
  }
  new_game_prepared_ = true;

  // Book loading is independent of model loading so BookFile changes followed
  // by isready take effect without rebuilding the TensorRT evaluators.
  if (books_dirty_) {
    book_.Close();
    gote_exit_book_.Close();

    if (!book_path_.empty()) {
      const uint64_t book_count = book_.Load(book_path_);
      if (book_.is_loaded()) {
        Log("YBB book ready: " + std::to_string(book_count) +
            " positions from " + book_path_);
      } else {
        Log("YBB book error: " + book_.last_error());
      }
    }

    if (use_gote_exit_book_ && !gote_exit_book_path_.empty()) {
      const uint64_t book_count =
          gote_exit_book_.Load(gote_exit_book_path_);
      if (gote_exit_book_.is_loaded()) {
        Log("Gote exit YBB ready: " + std::to_string(book_count) +
            " positions from " + gote_exit_book_path_);
      } else if (use_gote_exit_book_) {
        Log("Gote exit YBB error: " + gote_exit_book_.last_error());
      }
    }
    books_dirty_ = false;
  }
  Send("readyok");
}

void USIEngine::CmdSetOption(const std::vector<std::string>& parts) {
  std::string name, value;
  std::size_t value_marker = parts.size();
  for (std::size_t i = 1; i < parts.size(); ++i) {
    if (parts[i] == "value") {
      value_marker = i;
      break;
    }
  }
  const std::size_t name_begin =
      parts.size() > 1 && parts[1] == "name" ? 2 : parts.size();
  for (std::size_t i = name_begin; i < value_marker; ++i) {
    if (!name.empty()) name += ' ';
    name += parts[i];
  }
  if (value_marker < parts.size()) {
    for (std::size_t i = value_marker + 1; i < parts.size(); ++i) {
      if (!value.empty()) value += ' ';
      value += parts[i];
    }
  }

  if (name.empty()) {
    Log("Ignored malformed setoption without a name");
    return;
  }

  const std::string name_lower = ToLower(name);
  const auto reset_search = [this]() { search_.reset(); };
  const auto reset_inference = [this]() {
    search_.reset();
    evaluators_.clear();
  };

  try {
    if (name_lower == "maxnodes") {
      max_nodes_ = std::clamp(ParseInt(value), 1, 10000000);
    } else if (name_lower == "onnxmodel") {
      onnx_path_ = value;
      reset_inference();
    } else if (name_lower == "modelformat") {
      model_format_ = ParseModelFormat(value);
      reset_inference();
    } else if (name_lower == "dlshogimodel") {
      model_format_ = ToLower(value) == "true" ? ModelFormat::kDlshogi
                                                : ModelFormat::kAuto;
      reset_inference();
    } else if (name_lower == "usegpu") {
      use_gpu_ = ToLower(value) == "true" || value == "1";
      reset_inference();
    } else if (name_lower == "threads" ||
               name_lower == "workerspergpu") {
      search_config_.workers_per_gpu =
          std::clamp(ParseInt(value), 1, kMaxWorkersPerGpu);
      // TensorRT allocates one execution slot per worker.
      reset_inference();
    } else if (name_lower == "minibatchsize") {
      search_config_.minibatch_size =
          std::clamp(ParseInt(value), 1, 4096);
      reset_search();
    } else if (name_lower == "cinit" || name_lower == "c_init") {
      search_config_.c_init =
          std::clamp(ParseFiniteFloat(value), 0.0f, 100.0f);
      reset_search();
    } else if (name_lower == "cbase" || name_lower == "c_base") {
      search_config_.c_base =
          std::clamp(ParseFiniteFloat(value), 1.0f, 1.0e9f);
      reset_search();
    } else if (name_lower == "fpureduction" ||
               name_lower == "c_fpu_reduction") {
      search_config_.c_fpu_reduction =
          std::clamp(ParseFiniteFloat(value), 0.0f, 100.0f);
      reset_search();
    } else if (name_lower == "cinitroot" ||
               name_lower == "c_init_root") {
      search_config_.c_init_root =
          std::clamp(ParseFiniteFloat(value), 0.0f, 100.0f);
      reset_search();
    } else if (name_lower == "cbaseroot" ||
               name_lower == "c_base_root") {
      search_config_.c_base_root =
          std::clamp(ParseFiniteFloat(value), 1.0f, 1.0e9f);
      reset_search();
    } else if (name_lower == "fpureductionroot" ||
               name_lower == "c_fpu_reduction_root") {
      search_config_.c_fpu_reduction_root =
          std::clamp(ParseFiniteFloat(value), 0.0f, 100.0f);
      reset_search();
    } else if (name_lower == "drawvalueblack") {
      search_config_.draw_value_black =
          std::clamp(ParseFiniteFloat(value), 0.0f, 1.0f);
      reset_search();
    } else if (name_lower == "drawvaluewhite") {
      search_config_.draw_value_white =
          std::clamp(ParseFiniteFloat(value), 0.0f, 1.0f);
      reset_search();
    } else if (name_lower == "resignthreshold") {
      search_config_.resign_threshold =
          std::clamp(ParseFiniteFloat(value), 0.0f, 0.5f);
      reset_search();
    } else if (name_lower == "infointervalms") {
      search_config_.info_interval_ms =
          std::clamp(ParseInt(value), 100, 10000);
      reset_search();
    } else if (name_lower == "leafmatemode") {
      const std::string mode = ToLower(value);
      if (mode == "shallow") {
        if (search_config_.leaf_mate_depth <= 0 ||
            search_config_.leaf_mate_depth % 2 == 0) {
          search_config_.leaf_mate_depth = 5;
        }
      } else if (mode == "off") {
        search_config_.leaf_mate_depth = 0;
      } else if (mode == "dfpn") {
        // Historical behavior treated the unimplemented df-pn mode as off.
        search_config_.leaf_mate_depth = 0;
        Log("LeafMateMode=dfpn is retired; treating it as off");
      } else {
        Log("Ignored unsupported LeafMateMode value: " + value);
        return;
      }
      reset_search();
    } else if (name_lower == "leafmatedepth") {
      int depth = std::clamp(ParseInt(value), 1, 7);
      if (depth % 2 == 0) --depth;
      search_config_.leaf_mate_depth = depth;
      reset_search();
    } else if (name_lower == "rootmatedepth") {
      int depth = std::clamp(ParseInt(value), 0, 7);
      if (depth > 0 && depth % 2 == 0) --depth;
      search_config_.root_mate_depth = depth;
      reset_search();
    } else if (name_lower == "nncachesize") {
      search_config_.nn_cache_size =
          std::min<std::size_t>(ParseSize(value), 100000000);
      reset_search();
    } else if (name_lower == "numgpus") {
      num_gpus_ = std::clamp(ParseInt(value), 1, 8);
      reset_inference();
    } else if (name_lower == "maxmovestodraw") {
      search_config_.max_moves_to_draw =
          std::clamp(ParseInt(value), 1, 100000);
      reset_search();
    } else if (name_lower == "movesleftweight") {
      search_config_.moves_left_weight =
          std::clamp(ParseFiniteFloat(value), 0.0f, 100.0f);
      reset_search();
    } else if (name_lower == "movesleftthreshold") {
      search_config_.moves_left_threshold =
          std::clamp(ParseFiniteFloat(value), 0.0f, 0.5f);
      reset_search();
    } else if (name_lower == "movesleftcap") {
      search_config_.moves_left_cap =
          std::clamp(ParseFiniteFloat(value), 0.0f, 10000.0f);
      reset_search();
    } else if (name_lower == "dfpnmaxtime") {
      dfpn_max_time_ms_ = std::clamp(ParseInt(value), 100, 60000);
    } else if (name_lower == "maxmovetime") {
      max_move_time_ms_ = std::clamp(ParseInt(value), 0, 300000);
    } else if (name_lower == "maxmovetime1m") {
      max_move_time_1m_ms_ = std::clamp(ParseInt(value), 0, 60000);
    } else if (name_lower == "bookfile") {
      book_path_ = value;
      books_dirty_ = true;
    } else if (name_lower == "goteexitbookfile") {
      gote_exit_book_path_ = value;
      books_dirty_ = true;
    } else if (name_lower == "usegoteexitbook") {
      const bool enabled = ToLower(value) == "true" || value == "1";
      if (enabled != use_gote_exit_book_) books_dirty_ = true;
      use_gote_exit_book_ = enabled;
    } else if (name_lower == "noiseepsilon" ||
               name_lower == "perleafgathering" ||
               name_lower == "leafdfpnnodes" ||
               name_lower == "virtuallossweight" ||
               name_lower == "maxgpubatch" ||
               name_lower == "bookonthefly") {
      Log("Option " + name + " is retired and ignored");
      return;
    } else {
      Log("Unknown option ignored: " + name);
      return;
    }
  } catch (const std::exception& error) {
    Log("Invalid value for " + name + ": " + error.what());
    return;
  }

  Log("Set " + name + " = " + value);
}

void USIEngine::CmdUsiNewGame() {
  board_.SetStartPos();
  board_.ClearHistory();
  position_start_key_ = board_.Hash();
  position_moves_.clear();
  if (search_ && !new_game_prepared_) {
    // USI clients should issue isready/readyok before each game. Preserve
    // strict per-game cache isolation for older clients, but make the
    // unsynchronised slow path visible because it can delay the next go.
    Log("WARNING usinewgame received without per-game isready; "
        "clearing search state now");
    search_->PrepareForNewGame();
  }
  new_game_prepared_ = false;
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

  // Probe the specialized policy only on Gote turns. A miss deliberately
  // falls through to MCTS instead of the normal book: it means this line has
  // left the generated Gote policy.
  OpeningBook* active_book = nullptr;
  const OpeningBookChoice book_choice = ChooseOpeningBook(
      board_.side_to_move(), use_gote_exit_book_, book_.is_loaded(),
      gote_exit_book_.is_loaded());
  if (book_choice == OpeningBookChoice::kGoteExit) {
    active_book = &gote_exit_book_;
  } else if (book_choice == OpeningBookChoice::kNormal) {
    active_book = &book_;
  }

  if (active_book != nullptr) {
    auto* entry = active_book->Probe(board_);
    if (entry) {
      const std::string move_usi = entry->move.ToString();
      Log(std::string(book_choice == OpeningBookChoice::kGoteExit
                          ? "Gote exit book hit: "
                          : "Book hit: ") +
          move_usi + " (eval=" +
          std::to_string(entry->eval) + ", depth=" +
          std::to_string(entry->depth) + ")");
      Send("bestmove " + move_usi);
      return;
    }
  }

  // Configure the dlshogi-style MCTS search.
  search_config_.max_nodes = nodes_limit;
  search_config_.max_time = time_budget.mcts_time_seconds;

  // --- Launch root df-pn in parallel ---
  auto move_start_time = std::chrono::steady_clock::now();

  // Keep the solver and its result in one state shared with the worker.
  struct DfpnState {
    MateDfpnSolver solver;
    std::atomic<bool> done{false};
    Move mate_move;
    ShogiBoard board;
    DfpnState(size_t nodes, const ShogiBoard& b) : solver(nodes), board(b) {}
  };
  auto dfpn =
      std::make_shared<DfpnState>(time_budget.root_dfpn_nodes, board_);

  const auto dfpn_time_deadline =
      move_start_time +
      std::chrono::milliseconds(time_budget.root_dfpn_time_ms);
  auto dfpn_thread =
      std::thread([dfpn, root_dfpn_nodes = time_budget.root_dfpn_nodes,
                   dfpn_time_deadline]() {
        dfpn->mate_move = dfpn->solver.search(
            dfpn->board, root_dfpn_nodes, dfpn_time_deadline);
        dfpn->done.store(true, std::memory_order_release);
      });

  // --- Run dlshogi-style MCTS ---
  // Set info callback for periodic GUI output during search.
  search_config_.info_callback = [this](const dlshogi_mcts::SearchInfo& info) {
    // Keep free-form diagnostics before the structured record.  Some GUIs
    // incorrectly treat `info string` as a new empty analysis record, so the
    // last line in each update must be the complete depth/score/PV record.
    if (info.nn_cache.capacity > 0) {
      Log(FormatNNCacheStats(info.nn_cache));
    }

    USISearchInfo usi_info;
    usi_info.depth = info.depth;
    usi_info.seldepth = info.depth;
    usi_info.score_cp = info.score_cp;
    usi_info.nodes = std::max(info.nodes, 0);
    usi_info.nps = std::max(info.nps, 0);
    usi_info.hashfull = NNCacheHashfull(info.nn_cache);
    usi_info.time_ms = std::max(info.time_ms, 0);
    usi_info.pv = info.pv;
    Send(FormatUSISearchInfo(usi_info));
  };

  // Persistent Search object across `go` commands and games.
  EnsureSearch();
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
      search_->Run(board_, position_start_key_, position_moves_);
  search_done.store(true, std::memory_order_release);
  if (watchdog.joinable()) watchdog.join();
  Log(std::string("tree_reused ") + (result.tree_reused ? "true" : "false") +
      " root_visits_before " +
      std::to_string(result.root_visits_before));

  // If MCTS finishes first, preserve the original clock-scaled grace period
  // for root df-pn. The grace period is capped by DfPnMaxTime and by the move
  // deadline, so it cannot recreate the old tournament time overruns.
  auto dfpn_wait_deadline =
      std::chrono::steady_clock::now() +
      std::chrono::milliseconds(time_budget.root_dfpn_grace_ms);
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

  // --- Choose result ---
  bool use_mate = dfpn->done.load(std::memory_order_acquire) &&
                  !dfpn->mate_move.is_null() &&
                  !MateDfpnSolver::IsNoMate(dfpn->mate_move);

  // Defense in depth: a root df-pn result must not replace MCTS when its very
  // first move enters a repetition. The solver now adjudicates these nodes
  // itself, but keeping the final boundary check prevents a future df-pn
  // regression from turning an OUTE_SENNICHITE loss into bestmove.
  if (use_mate) {
    const auto repetition =
        RootMoveRepetitionResult(board_, dfpn->mate_move);
    if (repetition != ShogiBoard::RepetitionResult::kNone) {
      Log("Rejected root df-pn move " + dfpn->mate_move.ToString() +
          ": repetition=" + RepetitionResultName(repetition));
      use_mate = false;
    }
  }

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

  if (result.nn_cache.capacity > 0) {
    Log(FormatNNCacheStats(result.nn_cache));
  }

  USISearchInfo usi_info;
  usi_info.pv = result.pv;
  if (usi_info.pv.empty()) usi_info.pv.push_back(result.best_move);
  usi_info.depth = static_cast<int>(usi_info.pv.size());
  usi_info.seldepth = usi_info.depth;
  usi_info.score_cp = result.score_cp;
  usi_info.nodes = std::max(result.nodes, 0);
  usi_info.nps = static_cast<std::uint64_t>(std::max(result.nps, 0.0f));
  usi_info.hashfull = NNCacheHashfull(result.nn_cache);
  usi_info.time_ms = static_cast<std::uint64_t>(
      std::max(result.time_sec, 0.0f) * 1000.0f);
  Send(FormatUSISearchInfo(usi_info));

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
  const auto t0 = std::chrono::steady_clock::now();
  const auto deadline =
      time_limit_ms > 0
          ? t0 + std::chrono::milliseconds(time_limit_ms)
          : MateDfpnSolver::Deadline::max();

  auto search_thread = std::thread([&, deadline]() {
    mate_move = solver.search(board_, max_nodes, deadline);
    search_done.store(true, std::memory_order_release);
  });

  while (!search_done.load(std::memory_order_acquire)) {
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

  if (!search_done.load(std::memory_order_acquire) && time_limit_ms > 0) {
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
