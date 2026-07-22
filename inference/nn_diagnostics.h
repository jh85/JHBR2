#pragma once

#include <atomic>
#include <cstdio>
#include <string>

// Temporary one-shot NN diagnostics.  The containment checks that call this
// helper remain active when logging is disabled; only the diagnostic message
// is compiled out.  Configure with -DENABLE_NN_DIAGNOSTICS=OFF after the
// incident has been identified.
#ifndef JHBR2_ENABLE_NN_DIAGNOSTICS
#define JHBR2_ENABLE_NN_DIAGNOSTICS 0
#endif

namespace jhbr2::nn_diagnostics {

#if JHBR2_ENABLE_NN_DIAGNOSTICS

inline std::atomic<bool> g_first_failure_logged{false};

inline void LogOnce(const char* stage, const std::string& details) {
  bool expected = false;
  if (!g_first_failure_logged.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel,
          std::memory_order_acquire)) {
    return;
  }
  std::fprintf(stderr, "[NN_DIAG] first_failure stage=%s %s\n", stage,
               details.c_str());
  std::fflush(stderr);
}

#else

inline void LogOnce(const char*, const std::string&) {}

#endif

}  // namespace jhbr2::nn_diagnostics
