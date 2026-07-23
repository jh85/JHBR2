#include <iostream>
#include <string>

#include "book/book_selection.h"

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

}  // namespace

int main() {
  using jhbr2::ChooseOpeningBook;
  using jhbr2::OpeningBookChoice;
  using lczero::BLACK;
  using lczero::WHITE;

  Check("Sente always uses normal book",
        ChooseOpeningBook(BLACK, true, true, true) ==
            OpeningBookChoice::kNormal);
  Check("Gote uses normal book when feature is disabled",
        ChooseOpeningBook(WHITE, false, true, true) ==
            OpeningBookChoice::kNormal);
  Check("Gote uses specialized book when enabled",
        ChooseOpeningBook(WHITE, true, true, true) ==
            OpeningBookChoice::kGoteExit);
  Check("Gote does not fall back to normal book",
        ChooseOpeningBook(WHITE, true, true, false) ==
            OpeningBookChoice::kNone);
  Check("no book when neither is loaded",
        ChooseOpeningBook(BLACK, false, false, false) ==
            OpeningBookChoice::kNone);

  std::cout << "\n=== Summary: " << failures << " failed ===\n";
  return failures == 0 ? 0 : 1;
}
