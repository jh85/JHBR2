#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>

#include "usi/time_manager.h"

namespace {

int failures = 0;

template <typename T, typename U>
void CheckEqual(const std::string& name, T actual, U expected) {
  if (actual == static_cast<T>(expected)) {
    std::cout << "  OK    " << name << '\n';
  } else {
    std::cout << "  FAIL  " << name << ": expected " << expected
              << ", got " << actual << '\n';
    ++failures;
  }
}

void CheckNear(const std::string& name, float actual, float expected) {
  if (std::fabs(actual - expected) < 0.0001f) {
    std::cout << "  OK    " << name << '\n';
  } else {
    std::cout << "  FAIL  " << name << ": expected " << expected
              << ", got " << actual << '\n';
    ++failures;
  }
}

void CheckDfpnSchedule(const std::string& name, int available_ms,
                       int expected_grace_ms, std::size_t expected_nodes) {
  jhbr2::TimeControl control;
  control.main_time_ms = available_ms;
  const auto budget =
      jhbr2::TimeManager::Compute(control, jhbr2::TimeOptions{});
  CheckEqual(name + " grace", budget.root_dfpn_grace_ms,
             expected_grace_ms);
  CheckEqual(name + " nodes", budget.root_dfpn_nodes, expected_nodes);
}

}  // namespace

int main() {
  using jhbr2::TimeControl;
  using jhbr2::TimeManager;
  using jhbr2::TimeOptions;

  {
    const auto budget = TimeManager::Compute(TimeControl{}, TimeOptions{});
    CheckNear("node-only MCTS is unlimited", budget.mcts_time_seconds, 0.0f);
    CheckEqual("node-only has no move cap", budget.active_move_cap_ms, 0);
    CheckEqual("node-only has no hard deadline", budget.hard_deadline_ms, 0);
    CheckEqual("node-only DFPN max", budget.root_dfpn_time_ms, 4000);
  }

  {
    TimeControl control;
    control.byoyomi_ms = 10000;
    const auto budget = TimeManager::Compute(control, TimeOptions{});
    CheckNear("byoyomi uses 90 percent", budget.mcts_time_seconds, 9.0f);
    CheckEqual("byoyomi hard deadline includes watchdog margin",
               budget.hard_deadline_ms, 11000);
    CheckEqual("DFPN remains below byoyomi deadline",
               budget.root_dfpn_time_ms, 4000);
  }

  {
    TimeControl control;
    control.main_time_ms = 100000;
    control.has_main_time = true;
    const auto budget = TimeManager::Compute(control, TimeOptions{});
    CheckNear("main time uses five percent", budget.mcts_time_seconds, 5.0f);
    CheckEqual("main-time watchdog margin", budget.hard_deadline_ms, 7000);
  }

  {
    TimeControl control;
    control.main_time_ms = 100000;
    control.increment_ms = 1000;
    control.has_main_time = true;
    const auto budget = TimeManager::Compute(control, TimeOptions{});
    CheckNear("increment contributes 80 percent", budget.mcts_time_seconds,
              5.8f);
  }

  TimeOptions tournament_options;
  tournament_options.max_move_time_ms = 15000;
  tournament_options.max_move_time_1m_ms = 9000;
  tournament_options.dfpn_max_time_ms = 8000;

  {
    TimeControl control;
    control.main_time_ms = 300000;
    control.has_main_time = true;
    const auto budget = TimeManager::Compute(control, tournament_options);
    CheckEqual("normal clock selects MaxMoveTime", budget.active_move_cap_ms,
               15000);
    CheckNear("normal cap reserves 500 ms", budget.mcts_time_seconds, 14.5f);
    CheckEqual("normal hard deadline", budget.hard_deadline_ms, 15000);
    CheckEqual("normal DFPN maximum", budget.root_dfpn_time_ms, 8000);
    CheckEqual("normal DFPN grace", budget.root_dfpn_grace_ms, 1000);
    CheckEqual("normal DFPN nodes", budget.root_dfpn_nodes, 2000000);
  }

  {
    TimeControl control;
    control.main_time_ms = 59999;
    control.byoyomi_ms = 20000;
    control.has_main_time = true;
    const auto budget = TimeManager::Compute(control, tournament_options);
    CheckEqual("below 60 seconds selects MaxMoveTime1m",
               budget.active_move_cap_ms, 9000);
    CheckNear("low-time cap reserves 500 ms", budget.mcts_time_seconds, 8.5f);
    CheckEqual("low-time hard deadline uses MaxMoveTime1m",
               budget.hard_deadline_ms, 9000);
    CheckEqual("low-time DFPN maximum", budget.root_dfpn_time_ms, 8000);
    CheckEqual("low-time DFPN grace", budget.root_dfpn_grace_ms, 500);
    CheckEqual("low-time DFPN nodes", budget.root_dfpn_nodes, 500000);
  }

  {
    TimeControl control;
    control.main_time_ms = 60000;
    control.byoyomi_ms = 20000;
    control.has_main_time = true;
    const auto budget = TimeManager::Compute(control, tournament_options);
    CheckEqual("60-second boundary uses MaxMoveTime",
               budget.active_move_cap_ms, 15000);
    CheckNear("60-second MCTS cap", budget.mcts_time_seconds, 14.5f);
    CheckEqual("60-second hard deadline", budget.hard_deadline_ms, 15000);
  }

  {
    TimeControl control;
    control.main_time_ms = 40000;
    control.has_main_time = true;
    const auto budget = TimeManager::Compute(control, tournament_options);
    CheckNear("base allocation can finish before cap",
              budget.mcts_time_seconds, 2.0f);
    CheckEqual("low-time cap still bounds watchdog",
               budget.hard_deadline_ms, 9000);
  }

  {
    TimeControl control;
    control.main_time_ms = 40000;
    control.has_main_time = true;
    TimeOptions options;
    options.max_move_time_1m_ms = 9000;
    options.dfpn_max_time_ms = 8000;
    const auto budget = TimeManager::Compute(control, options);
    CheckEqual("low-time cap works without general cap",
               budget.active_move_cap_ms, 9000);
    CheckEqual("low-time-only cap controls watchdog",
               budget.hard_deadline_ms, 9000);
  }

  {
    TimeControl control;
    control.byoyomi_ms = 20000;
    const auto budget = TimeManager::Compute(control, tournament_options);
    CheckEqual("zero main time keeps general cap", budget.active_move_cap_ms,
               15000);
    CheckNear("zero-main byoyomi is generally capped",
              budget.mcts_time_seconds, 14.5f);
  }

  {
    TimeOptions options;
    options.max_move_time_ms = 500;
    const auto budget = TimeManager::Compute(TimeControl{}, options);
    CheckNear("small cap keeps 500 ms MCTS minimum",
              budget.mcts_time_seconds, 0.5f);
    CheckEqual("small cap is the hard deadline", budget.hard_deadline_ms, 500);
    CheckEqual("DFPN leaves 50 ms at small cap", budget.root_dfpn_time_ms,
               450);
  }

  {
    TimeControl control;
    control.main_time_ms = 0;
    control.increment_ms = 1000;
    control.has_main_time = true;
    const auto budget = TimeManager::Compute(control, TimeOptions{});
    CheckNear("clock presence is independent of own main time",
              budget.mcts_time_seconds, 0.8f);
  }

  CheckDfpnSchedule("no clock", 0, 300, 100000);
  CheckDfpnSchedule("below 10 seconds", 9999, 100, 10000);
  CheckDfpnSchedule("at 10 seconds", 10000, 300, 100000);
  CheckDfpnSchedule("below 60 seconds", 59999, 300, 100000);
  CheckDfpnSchedule("at 60 seconds", 60000, 500, 500000);
  CheckDfpnSchedule("below 300 seconds", 299999, 500, 500000);
  CheckDfpnSchedule("at 300 seconds", 300000, 1000, 2000000);

  std::cout << "\n=== Summary: " << failures << " failed ===\n";
  return failures == 0 ? 0 : 1;
}
