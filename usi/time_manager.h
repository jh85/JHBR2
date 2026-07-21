#pragma once

#include <cstddef>

namespace jhbr2 {

struct TimeControl {
  int main_time_ms = 0;
  int increment_ms = 0;
  int byoyomi_ms = 0;

  // CmdGo historically enables main-time allocation when either player's
  // main-time field is positive, even if main_time_ms is zero for this side.
  bool has_main_time = false;
};

struct TimeOptions {
  int max_move_time_ms = 0;
  int max_move_time_1m_ms = 0;
  int dfpn_max_time_ms = 4000;
};

struct TimeBudget {
  // Nominal MCTS duration. Zero leaves MCTS node-limited.
  float mcts_time_seconds = 0.0f;

  // Selected USI option cap and the absolute watchdog deadline. When no
  // option cap applies, the watchdog allows two seconds beyond nominal MCTS.
  int active_move_cap_ms = 0;
  int hard_deadline_ms = 0;

  // Concurrent root DFPN limits and allowed wait after MCTS completes.
  int root_dfpn_time_ms = 0;
  int root_dfpn_grace_ms = 0;
  std::size_t root_dfpn_nodes = 0;
};

class TimeManager {
 public:
  static TimeBudget Compute(const TimeControl& control,
                            const TimeOptions& options);
};

}  // namespace jhbr2
