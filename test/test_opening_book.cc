#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "book/opening_book.h"

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
  const auto unique = std::chrono::steady_clock::now()
                          .time_since_epoch()
                          .count();
  const auto path = std::filesystem::temp_directory_path() /
                    ("jhbr2_opening_book_" + std::to_string(unique) + ".db");

  {
    std::ofstream book_file(path);
    book_file << "#YANEURAOU-DB2016 1.00\n"
              << "sfen 4k4/9/9/9/9/9/9/9/4K4 b - 1\n"
              << "5i5h none 123 10 4\n"
              << "5i4h none 100 8 2\n"
              << "sfen 4k4/9/9/9/9/9/9/9/4K4 w - 1\n"
              << "5a5b none -45 7 3\n";
  }

  const std::string black_key = "4k4/9/9/9/9/9/9/9/4K4 b - 99";
  const std::string white_key = "4k4/9/9/9/9/9/9/9/4K4 w - 42";

  {
    jhbr2::OpeningBook book;
    Check("preload position count", book.Load(path.string()) == 2);
    Check("preload reports loaded", book.is_loaded());

    const auto* black = book.Probe(black_key);
    Check("preload normalizes ply number",
          black && black->move_usi == "5i5h" && black->eval == 123 &&
              black->depth == 10);

    const auto* white = book.Probe(white_key);
    Check("preload finds second position",
          white && white->move_usi == "5a5b" && white->eval == -45);
    Check("preload misses unknown position",
          book.Probe("9/9/9/9/9/9/9/9/9 b - 1") == nullptr);
  }

  {
    jhbr2::OpeningBook book;
    Check("on-the-fly opens book", book.Load(path.string(), true) == 0 &&
                                           book.is_loaded());

    const auto* black = book.Probe(black_key);
    Check("on-the-fly finds first position",
          black && black->move_usi == "5i5h" && black->eval == 123);

    const auto* white = book.Probe(white_key);
    Check("on-the-fly finds second position",
          white && white->move_usi == "5a5b" && white->depth == 7);
    Check("on-the-fly misses unknown position",
          book.Probe("9/9/9/9/9/9/9/9/9 b - 1") == nullptr);
  }

  std::error_code error;
  std::filesystem::remove(path, error);
  Check("temporary book removed", !error);

  std::cout << "\n=== Summary: " << failures << " failed ===\n";
  return failures == 0 ? 0 : 1;
}
