/*
  JHBR2 Shogi Engine — Main Entry Point

  Runs the USI protocol handler. The engine reads commands from stdin
  and writes responses to stdout.

  Usage:
    ./jhbr2
    (then type USI commands, or connect via a Shogi GUI)
*/

#include "usi/usi_engine.h"
#include "shogi/encoder.h"

int main(int /*argc*/, char* /*argv*/[]) {
  // Initialize static tables.
  lczero::ShogiEncoderTables::Init();

  // Run USI engine.
  jhbr2::USIEngine engine;
  engine.Run();

  return 0;
}
