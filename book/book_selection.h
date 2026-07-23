/*
  Side-aware choice between JHBR2's normal and Gote early-exit books.
*/

#pragma once

#include "shogi/types.h"

namespace jhbr2 {

enum class OpeningBookChoice {
  kNone,
  kNormal,
  kGoteExit,
};

constexpr OpeningBookChoice ChooseOpeningBook(
    lczero::Color side_to_move, bool use_gote_exit_book,
    bool normal_book_loaded, bool gote_exit_book_loaded) {
  if (side_to_move == lczero::WHITE && use_gote_exit_book) {
    // A specialized-book miss means the generated line has left the source
    // book. Do not silently re-enter it through the normal book.
    return gote_exit_book_loaded ? OpeningBookChoice::kGoteExit
                                 : OpeningBookChoice::kNone;
  }
  return normal_book_loaded ? OpeningBookChoice::kNormal
                            : OpeningBookChoice::kNone;
}

}  // namespace jhbr2
