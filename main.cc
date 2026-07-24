/*
  JHBR2 Shogi Engine — Main Entry Point

  Runs the USI protocol handler. The engine reads commands from stdin
  and writes responses to stdout.

  Usage:
    ./jhbr2
    (then type USI commands, or connect via a Shogi GUI)
*/

#include "usi/usi_engine.h"
#include "shogi/bitboard.h"
#include "shogi/encoder.h"

#include <csignal>
#include <cstdio>
#include <execinfo.h>
#include <unistd.h>

static void crash_handler(int sig) {
  fprintf(stderr, "\n=== JHBR2 CRASH: signal %d ===\n", sig);
  void* frames[32];
  int n = backtrace(frames, 32);
  backtrace_symbols_fd(frames, n, 2);  // write to stderr
  fflush(stderr);
  _exit(1);
}

int main(int /*argc*/, char* /*argv*/[]) {
  signal(SIGSEGV, crash_handler);
  signal(SIGABRT, crash_handler);
  signal(SIGFPE, crash_handler);
  // Initialize static tables.
  lczero::ShogiTables::Init();
  lczero::ShogiEncoderTables::Init();

  // Run USI engine.
  jhbr2::USIEngine engine;
  engine.Run();

  return 0;
}
