#include "usi/time_manager.h"

#include <algorithm>
#include <cstdint>

namespace jhbr2 {

TimeBudget TimeManager::Compute(const TimeControl& control,
                                const TimeOptions& options) {
  TimeBudget budget;

  if (control.byoyomi_ms > 0) {
    budget.mcts_time_seconds = control.byoyomi_ms / 1000.0f * 0.9f;
  } else if (control.has_main_time) {
    budget.mcts_time_seconds =
        (control.main_time_ms * 0.05f + control.increment_ms * 0.8f) /
        1000.0f;
    budget.mcts_time_seconds =
        std::max(budget.mcts_time_seconds, 0.1f);
  }

  budget.active_move_cap_ms = options.max_move_time_ms;
  if (options.max_move_time_1m_ms > 0 && control.main_time_ms > 0 &&
      control.main_time_ms < 60000) {
    budget.active_move_cap_ms = options.max_move_time_1m_ms;
  }

  if (budget.active_move_cap_ms > 0) {
    const float cap_seconds =
        std::max(budget.active_move_cap_ms / 1000.0f - 0.5f, 0.5f);
    if (budget.mcts_time_seconds <= 0.0f ||
        cap_seconds < budget.mcts_time_seconds) {
      budget.mcts_time_seconds = cap_seconds;
    }
  }

  const int64_t available_ms = static_cast<int64_t>(control.main_time_ms) +
                               control.increment_ms + control.byoyomi_ms;
  if (available_ms <= 0) {
    budget.root_dfpn_grace_ms = 300;
    budget.root_dfpn_nodes = 100000;
  } else if (available_ms < 10000) {
    budget.root_dfpn_grace_ms = 100;
    budget.root_dfpn_nodes = 10000;
  } else if (available_ms < 60000) {
    budget.root_dfpn_grace_ms = 300;
    budget.root_dfpn_nodes = 100000;
  } else if (available_ms < 300000) {
    budget.root_dfpn_grace_ms = 500;
    budget.root_dfpn_nodes = 500000;
  } else {
    budget.root_dfpn_grace_ms = 1000;
    budget.root_dfpn_nodes = 2000000;
  }

  if (budget.active_move_cap_ms > 0) {
    budget.hard_deadline_ms = budget.active_move_cap_ms;
  } else if (budget.mcts_time_seconds > 0.0f) {
    budget.hard_deadline_ms =
        static_cast<int>(budget.mcts_time_seconds * 1000.0f) + 2000;
  }

  budget.root_dfpn_time_ms = options.dfpn_max_time_ms;
  if (budget.hard_deadline_ms > 0) {
    budget.root_dfpn_time_ms =
        std::min(budget.root_dfpn_time_ms,
                 std::max(budget.hard_deadline_ms - 50, 1));
  }

  return budget;
}

}  // namespace jhbr2
