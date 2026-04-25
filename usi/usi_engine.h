/*
  JHBR2 Shogi Engine — USI Protocol Handler

  Implements the Universal Shogi Interface protocol for communication
  with Shogi GUIs (ShogiGUI, Shogidokoro, etc.) and tournament software.

  Reference: http://shogidokoro.starfree.jp/usi.html
*/

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#ifdef USE_TENSORRT
#include "mcts/nn_tensorrt.h"
#else
#include "mcts/nn_eval.h"
#endif
#include "lc0_mcts/search.h"
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
  std::unique_ptr<NNEvaluator> evaluator_;
  std::unique_ptr<NNEvaluator> evaluator2_;  // Second GPU
  std::unique_ptr<lc0_shogi::Search> lc0_search_;
  lc0_shogi::SearchConfig lc0_config_;
  int game_ply_ = 0;

  // Options
  std::string onnx_path_ = "shogi_bt4.onnx";
  int max_nodes_ = 800;
  int num_gpus_ = 1;
  float noise_epsilon_ = 0.0f;
  bool use_gpu_ = true;
  int dfpn_max_time_ms_ = 4000;
  int max_move_time_ms_ = 0;
};

}  // namespace jhbr2
