#include "usi/search_info.h"

#include <algorithm>
#include <sstream>

namespace jhbr2 {

std::string FormatUSISearchInfo(const USISearchInfo& info) {
  std::ostringstream out;
  out << "info depth " << std::max(info.depth, 0);
  if (info.seldepth > 0) {
    out << " seldepth " << info.seldepth;
  }
  out << " multipv " << std::max(info.multipv, 1)
      << " score cp " << info.score_cp
      << " nodes " << info.nodes
      << " nps " << info.nps
      << " hashfull " << std::clamp(info.hashfull, 0, 1000)
      << " time " << info.time_ms;
  if (!info.pv.empty()) {
    out << " pv";
    for (const auto& move : info.pv) {
      out << ' ' << move.ToString();
    }
  }
  return out.str();
}

}  // namespace jhbr2
