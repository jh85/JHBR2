/*
  JHBR2 Shogi Engine — USI Protocol Handler

  Implements the Universal Shogi Interface protocol for communication
  with Shogi GUIs (ShogiGUI, Shogidokoro, etc.) and tournament software.

  Reference: http://shogidokoro.starfree.jp/usi.html
*/

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#ifdef USE_TENSORRT
#include "inference/nn_tensorrt.h"
#else
#include "inference/nn_eval.h"
#endif
#include "book/opening_book.h"
#include "dlshogi_mcts/uct_search.h"
#include "shogi/board.h"

namespace jhbr2 {

class USIEngine {
 public:
  static constexpr const char* ENGINE_NAME = "JHBR2";
  static constexpr const char* ENGINE_AUTHOR = "JHBR2 Team";

  USIEngine();

  // Main loop: read USI commands from stdin, write responses to stdout.
  void Run();

 private:
  // --- Command handlers ---
  void CmdUsi();
  void CmdIsReady();
  void CmdSetOption(const std::vector<std::string>& parts);
  void CmdUsiNewGame();
  void CmdPosition(const std::vector<std::string>& parts);
  void CmdGo(const std::vector<std::string>& parts);
  void CmdGoMate(const std::vector<std::string>& parts);
  void CmdStop();
  void CmdGameOver(const std::vector<std::string>& parts);
  void CmdDebug();

  // --- Helpers ---
  void Send(const std::string& msg);
  void Log(const std::string& msg);

  // --- Members ---
  lczero::ShogiBoard board_;
  std::vector<std::unique_ptr<NNEvaluator>> evaluators_;
  std::unique_ptr<dlshogi_mcts::Search> search_;
  dlshogi_mcts::SearchConfig search_config_;
  uint64_t position_start_key_ = 0;
  std::vector<lczero::Move> position_moves_;

  // Options
  std::string onnx_path_ = "shogi_bt4.onnx";
  ModelFormat model_format_ = ModelFormat::kAuto;
  int max_nodes_ = 800;
  int num_gpus_ = 1;
  bool use_gpu_ = true;
  int dfpn_max_time_ms_ = 4000;
  int max_move_time_ms_ = 0;
  int max_move_time_1m_ms_ = 0;
  std::string book_path_;
  std::string gote_exit_book_path_ = "user_book1_gote_exit.ybb";
  bool use_gote_exit_book_ = false;
  bool books_dirty_ = true;
  OpeningBook book_;
  OpeningBook gote_exit_book_;
};

}  // namespace jhbr2
