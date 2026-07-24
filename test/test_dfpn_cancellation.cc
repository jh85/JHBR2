#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include "mate/dfpn.h"
#include "shogi/board.h"

namespace {

int failures = 0;

void Check(const std::string& name, bool condition) {
  if (condition) {
    std::cout << "  OK    " << name << '\n';
  } else {
    std::cout << "  FAIL  " << name << '\n';
    ++failures;
  }
}

lczero::ShogiBoard StartPosition() {
  lczero::ShogiBoard board;
  const bool parsed = board.SetFromSfen(
      "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/"
      "LNSGKGSNL b - 1");
  Check("start position parses", parsed);
  return board;
}

}  // namespace

int main() {
  using jhbr2::MateDfpnSolver;

  lczero::ShogiTables::Init();

  // This is the race that used to make the USI thread's join unbounded:
  // stop() ran before the worker entered search(), then search() cleared it.
  {
    auto board = StartPosition();
    MateDfpnSolver solver(1000000);
    lczero::Move result;
    solver.stop();
    std::thread worker(
        [&]() { result = solver.search(board, 1000000); });
    worker.join();

    Check("pre-start cancellation returns unresolved", result.is_null());
    Check("pre-start cancellation expands no nodes",
          solver.get_nodes_searched() == 0);
  }

  // An absolute deadline must be honored before any allocation or expansion.
  {
    auto board = StartPosition();
    MateDfpnSolver solver(1000000);
    const auto expired =
        MateDfpnSolver::Clock::now() - std::chrono::milliseconds(1);
    const lczero::Move result = solver.search(board, 1000000, expired);

    Check("expired deadline returns unresolved", result.is_null());
    Check("expired deadline expands no nodes",
          solver.get_nodes_searched() == 0);
  }

  // Normal, completed searches must still return their proven result.
  {
    lczero::ShogiBoard board;
    const bool parsed =
        board.SetFromSfen("3pkp3/3p1p3/9/9/9/9/4R4/4r4/4K4 b - 1");
    Check("mate position parses", parsed);

    MateDfpnSolver solver(10000);
    const lczero::Move result = solver.search(
        board, 10000,
        MateDfpnSolver::Clock::now() + std::chrono::seconds(10));
    Check("deadline-aware search preserves mate result",
          !result.is_null() && !MateDfpnSolver::IsNoMate(result));
    Check("deadline-aware search preserves mate PV",
          result.ToString() == "5g5h" && solver.get_pv().size() == 1);
  }

  if (failures != 0) {
    std::cout << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << "All df-pn cancellation tests passed\n";
  return 0;
}
