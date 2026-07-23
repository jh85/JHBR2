/*
  Build a compact Gote-only YBB policy that tries to leave a source YBB as
  early as possible without choosing a move outside an evaluation margin.

  The input is expected to be a distributed/PetaShock-style YBB: every
  in-book transposition that should be considered must be present in its move
  lists. The generator follows stored moves, resolves their successor
  PackedSfen keys, solves the asymmetric reachability game, and writes one
  selected move for each Gote position.
*/

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "book/gote_exit_solver.h"
#include "book/ybb_format.h"
#include "shogi/board.h"

namespace {

using jhbr2::DecodeYbbIndexEntry;
using jhbr2::DecodeYbbMoveRecord;
using jhbr2::EncodeYbbIndexEntry;
using jhbr2::EncodeYbbMoveRecord;
using jhbr2::GoteExitSolution;
using jhbr2::YbbIndexEntry;
using jhbr2::YbbMoveRecord;
using lczero::BLACK;
using lczero::ComparePackedSfen;
using lczero::Move;
using lczero::PackedSfen;
using lczero::ShogiBoard;
using lczero::WHITE;

struct Options {
  std::filesystem::path input;
  std::filesystem::path output;
  int eval_margin = 30;
  unsigned threads = std::max(1u, std::thread::hardware_concurrency());
  uint64_t max_positions = 0;
  bool force = false;
  bool validate_moves = false;
};

std::string Usage(const char* program) {
  return std::string("Usage: ") + program +
         " --input SOURCE.ybb --output user_book1_gote_exit.ybb"
         " [--eval-margin CP] [--threads N] [--max-positions N]"
         " [--validate-moves] [--force]";
}

uint64_t ParseUnsigned(const std::string& text, const char* option) {
  size_t used = 0;
  uint64_t value = 0;
  try {
    value = std::stoull(text, &used);
  } catch (const std::exception&) {
    throw std::runtime_error(std::string("invalid ") + option + ": " + text);
  }
  if (used != text.size()) {
    throw std::runtime_error(std::string("invalid ") + option + ": " + text);
  }
  return value;
}

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[++i];
    };

    if (arg == "--input") {
      options.input = value("--input");
    } else if (arg == "--output") {
      options.output = value("--output");
    } else if (arg == "--eval-margin") {
      const uint64_t parsed =
          ParseUnsigned(value("--eval-margin"), "--eval-margin");
      if (parsed > 32000) {
        throw std::runtime_error("--eval-margin must be at most 32000");
      }
      options.eval_margin = static_cast<int>(parsed);
    } else if (arg == "--threads") {
      const uint64_t parsed = ParseUnsigned(value("--threads"), "--threads");
      if (parsed == 0 || parsed > 1024) {
        throw std::runtime_error("--threads must be between 1 and 1024");
      }
      options.threads = static_cast<unsigned>(parsed);
    } else if (arg == "--max-positions") {
      options.max_positions =
          ParseUnsigned(value("--max-positions"), "--max-positions");
    } else if (arg == "--validate-moves") {
      options.validate_moves = true;
    } else if (arg == "--force") {
      options.force = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << Usage(argv[0]) << '\n';
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + arg);
    }
  }
  if (options.input.empty() || options.output.empty()) {
    throw std::runtime_error(Usage(argv[0]));
  }
  if (std::filesystem::absolute(options.input) ==
      std::filesystem::absolute(options.output)) {
    throw std::runtime_error("input and output paths must differ");
  }
  return options;
}

class MappedFile {
 public:
  explicit MappedFile(const std::filesystem::path& path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("cannot open input: " + path.string());

    struct stat status {};
    if (fstat(fd_, &status) != 0 || status.st_size < 0) {
      close(fd_);
      fd_ = -1;
      throw std::runtime_error("cannot stat input: " + path.string());
    }
    size_ = static_cast<uint64_t>(status.st_size);
    if (size_ == 0) {
      close(fd_);
      fd_ = -1;
      throw std::runtime_error("input file is empty");
    }

    void* mapping =
        mmap(nullptr, static_cast<size_t>(size_), PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapping == MAP_FAILED) {
      close(fd_);
      fd_ = -1;
      throw std::runtime_error("cannot mmap input: " + path.string());
    }
    data_ = static_cast<const uint8_t*>(mapping);
  }

  ~MappedFile() {
    if (data_ != nullptr) {
      munmap(const_cast<uint8_t*>(data_), static_cast<size_t>(size_));
    }
    if (fd_ >= 0) close(fd_);
  }

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  const uint8_t* data() const { return data_; }
  uint64_t size() const { return size_; }

 private:
  int fd_ = -1;
  const uint8_t* data_ = nullptr;
  uint64_t size_ = 0;
};

struct PackedSfenHash {
  size_t operator()(const PackedSfen& packed) const noexcept {
    // FNV-1a is inexpensive and deterministic; equality still checks all
    // 32 bytes, so hash collisions cannot merge positions.
    uint64_t hash = 1469598103934665603ULL;
    for (uint8_t byte : packed.data) {
      hash ^= byte;
      hash *= 1099511628211ULL;
    }
    return static_cast<size_t>(hash);
  }
};

class MappedYbb {
 public:
  explicit MappedYbb(const std::filesystem::path& path) : file_(path) {
    if (file_.size() < jhbr2::kYbbHeaderSize ||
        std::memcmp(file_.data(), jhbr2::kYbbMagic.data(),
                    jhbr2::kYbbMagic.size()) != 0) {
      throw std::runtime_error("invalid YBB V1 header");
    }
    record_count_ = jhbr2::ReadLe64(file_.data() + 16);
    flags_ = jhbr2::ReadLe64(file_.data() + 24);
    if ((flags_ & ~jhbr2::kYbbKnownFlags) != 0) {
      throw std::runtime_error("unsupported YBB flags: " +
                               std::to_string(flags_));
    }
    if (!jhbr2::YbbIndexSize(record_count_, &moves_base_) ||
        moves_base_ > file_.size()) {
      throw std::runtime_error("invalid YBB index size");
    }
    move_record_size_ = jhbr2::YbbMoveRecordSize(flags_);
  }

  uint64_t record_count() const { return record_count_; }
  uint64_t flags() const { return flags_; }
  uint64_t move_record_size() const { return move_record_size_; }
  uint64_t moves_base() const { return moves_base_; }
  uint64_t file_size() const { return file_.size(); }

  YbbIndexEntry Entry(uint64_t index) const {
    if (index >= record_count_) {
      throw std::runtime_error("YBB index out of range");
    }
    YbbIndexEntry entry;
    DecodeYbbIndexEntry(
        file_.data() + jhbr2::kYbbHeaderSize +
            index * jhbr2::kYbbIndexRecordSize,
        &entry);
    return entry;
  }

  YbbMoveRecord MoveAt(const YbbIndexEntry& entry,
                       uint32_t local_move) const {
    if (local_move >= entry.move_count) {
      throw std::runtime_error("YBB move index out of range");
    }
    const uint64_t relative =
        entry.moves_offset + uint64_t(local_move) * move_record_size_;
    if (relative > file_.size() - moves_base_ ||
        move_record_size_ > file_.size() - moves_base_ - relative) {
      throw std::runtime_error("YBB move area out of range");
    }
    YbbMoveRecord move;
    DecodeYbbMoveRecord(file_.data() + moves_base_ + relative, flags_, &move);
    return move;
  }

 private:
  MappedFile file_;
  uint64_t record_count_ = 0;
  uint64_t flags_ = 0;
  uint64_t moves_base_ = 0;
  uint64_t move_record_size_ = 0;
};

bool PlausibleMove(uint16_t raw) {
  if (raw == 0) return false;
  const Move move = Move::FromRaw(raw);
  if (!move.to().IsValid()) return false;
  if (move.is_drop()) {
    return !move.is_promotion() && move.drop_piece().IsHandPiece();
  }
  return move.from().IsValid() && move.from() != move.to();
}

class ExitGraph {
 public:
  ExitGraph(const MappedYbb* book, uint32_t node_count,
            const std::vector<uint32_t>* child_index)
      : book_(book), node_count_(node_count), child_index_(child_index) {}

  uint64_t NodeCount() const { return node_count_; }

  bool IsGote(uint32_t node) const {
    return (book_->Entry(node).packed_sfen.data[0] & 1U) != 0;
  }

  uint32_t EdgeCount(uint32_t node) const {
    return book_->Entry(node).move_count;
  }

  uint32_t EdgeChild(uint32_t node, uint32_t edge) const {
    const auto entry = book_->Entry(node);
    const uint64_t global =
        entry.moves_offset / book_->move_record_size() + edge;
    return child_index_->at(static_cast<size_t>(global));
  }

 private:
  const MappedYbb* book_;
  uint32_t node_count_;
  const std::vector<uint32_t>* child_index_;
};

struct BuildStats {
  uint64_t processed_positions = 0;
  uint64_t allowed_moves = 0;
  uint64_t disallowed_moves = 0;
  uint64_t leaf_moves = 0;
  uint64_t internal_moves = 0;
  uint64_t invalid_moves = 0;
};

BuildStats& operator+=(BuildStats& lhs, const BuildStats& rhs) {
  lhs.processed_positions += rhs.processed_positions;
  lhs.allowed_moves += rhs.allowed_moves;
  lhs.disallowed_moves += rhs.disallowed_moves;
  lhs.leaf_moves += rhs.leaf_moves;
  lhs.internal_moves += rhs.internal_moves;
  lhs.invalid_moves += rhs.invalid_moves;
  return lhs;
}

std::string FormatDuration(std::chrono::steady_clock::duration duration) {
  const auto seconds =
      std::chrono::duration_cast<std::chrono::seconds>(duration).count();
  const int hours = static_cast<int>(seconds / 3600);
  const int minutes = static_cast<int>((seconds % 3600) / 60);
  const int secs = static_cast<int>(seconds % 60);
  std::ostringstream output;
  output << std::setfill('0') << std::setw(2) << hours << ':'
         << std::setw(2) << minutes << ':' << std::setw(2) << secs;
  return output.str();
}

struct PreparedBook {
  uint32_t node_count = 0;
  uint64_t move_count = 0;
  std::unordered_map<PackedSfen, uint32_t, PackedSfenHash> positions;
};

PreparedBook ValidateAndIndex(const MappedYbb& book, uint64_t max_positions) {
  if (book.record_count() >= jhbr2::kExitDisallowed) {
    throw std::runtime_error("YBB has too many positions for 32-bit graph IDs");
  }
  const uint64_t selected_count =
      max_positions == 0
          ? book.record_count()
          : std::min(max_positions, book.record_count());

  PreparedBook prepared;
  prepared.node_count = static_cast<uint32_t>(selected_count);
  prepared.positions.max_load_factor(0.80f);
  prepared.positions.reserve(static_cast<size_t>(selected_count));

  uint64_t expected_offset = 0;
  std::optional<PackedSfen> previous;
  for (uint64_t i = 0; i < book.record_count(); ++i) {
    const auto entry = book.Entry(i);
    if (previous && ComparePackedSfen(*previous, entry.packed_sfen) >= 0) {
      throw std::runtime_error("YBB index is not strictly sorted at record " +
                               std::to_string(i));
    }
    previous = entry.packed_sfen;
    if (entry.moves_offset != expected_offset) {
      throw std::runtime_error("YBB move offsets are not contiguous at record " +
                               std::to_string(i));
    }
    const uint64_t move_bytes =
        uint64_t(entry.move_count) * book.move_record_size();
    if (expected_offset > book.file_size() - book.moves_base() ||
        move_bytes >
            book.file_size() - book.moves_base() - expected_offset) {
      throw std::runtime_error("YBB moves area is truncated at record " +
                               std::to_string(i));
    }
    expected_offset += move_bytes;

    if (i < selected_count) {
      auto [unused, inserted] =
          prepared.positions.emplace(entry.packed_sfen,
                                     static_cast<uint32_t>(i));
      if (!inserted) {
        throw std::runtime_error("duplicate PackedSfen in YBB index");
      }
      prepared.move_count += entry.move_count;
    }
  }
  if (book.moves_base() + expected_offset != book.file_size()) {
    throw std::runtime_error("YBB has trailing or missing moves data");
  }
  if (prepared.move_count > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("YBB move count exceeds address space");
  }
  return prepared;
}

std::vector<uint32_t> ResolveChildren(
    const MappedYbb& book, const PreparedBook& prepared,
    const Options& options, BuildStats* combined_stats) {
  std::vector<uint32_t> children(
      static_cast<size_t>(prepared.move_count), jhbr2::kExitDisallowed);
  std::atomic<uint64_t> next_node{0};
  std::atomic<uint64_t> completed{0};
  std::atomic<bool> failed{false};
  std::mutex error_mutex;
  std::string error_message;
  std::vector<BuildStats> thread_stats(options.threads);
  const auto start = std::chrono::steady_clock::now();

  auto fail = [&](std::string message) {
    bool expected = false;
    if (failed.compare_exchange_strong(expected, true)) {
      std::lock_guard<std::mutex> lock(error_mutex);
      error_message = std::move(message);
    }
  };

  auto worker = [&](unsigned worker_id) {
    BuildStats& stats = thread_stats[worker_id];
    while (!failed.load(std::memory_order_relaxed)) {
      const uint64_t node = next_node.fetch_add(1, std::memory_order_relaxed);
      if (node >= prepared.node_count) break;

      try {
        const auto entry = book.Entry(node);
        const bool gote = (entry.packed_sfen.data[0] & 1U) != 0;
        int best_eval = std::numeric_limits<int>::min();
        for (uint32_t local = 0; local < entry.move_count; ++local) {
          const auto record = book.MoveAt(entry, local);
          if (PlausibleMove(record.move)) {
            best_eval = std::max(best_eval, int(record.eval));
          }
        }

        ShogiBoard board;
        if (!board.SetFromPackedSfen(entry.packed_sfen, entry.ply)) {
          throw std::runtime_error("cannot decode PackedSfen");
        }
        lczero::MoveList legal_moves;
        if (options.validate_moves) {
          legal_moves = board.GenerateLegalMoves();
        }

        const uint64_t first_move =
            entry.moves_offset / book.move_record_size();
        for (uint32_t local = 0; local < entry.move_count; ++local) {
          const uint64_t global = first_move + local;
          if (global >= children.size()) {
            throw std::runtime_error("move index exceeds selected graph");
          }
          const auto record = book.MoveAt(entry, local);
          if (!PlausibleMove(record.move)) {
            ++stats.invalid_moves;
            children[static_cast<size_t>(global)] =
                jhbr2::kExitDisallowed;
            continue;
          }
          if (gote && int(record.eval) < best_eval - options.eval_margin) {
            ++stats.disallowed_moves;
            children[static_cast<size_t>(global)] =
                jhbr2::kExitDisallowed;
            continue;
          }

          const Move move = Move::FromRaw(record.move);
          if (options.validate_moves &&
              std::find(legal_moves.begin(), legal_moves.end(), move) ==
                  legal_moves.end()) {
            throw std::runtime_error("illegal stored move " + move.ToString());
          }

          const auto undo = board.DoMove(move);
          PackedSfen child_key;
          const bool encoded = board.ToPackedSfen(&child_key);
          board.UndoMove(move, undo);
          if (!encoded) {
            throw std::runtime_error("cannot encode child PackedSfen");
          }

          const auto child = prepared.positions.find(child_key);
          if (child == prepared.positions.end()) {
            children[static_cast<size_t>(global)] = jhbr2::kExitLeaf;
            ++stats.leaf_moves;
          } else {
            children[static_cast<size_t>(global)] = child->second;
            ++stats.internal_moves;
          }
          ++stats.allowed_moves;
        }
        ++stats.processed_positions;
        completed.fetch_add(1, std::memory_order_relaxed);
      } catch (const std::exception& error) {
        fail("record " + std::to_string(node) + ": " + error.what());
      }
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(options.threads);
  for (unsigned i = 0; i < options.threads; ++i) {
    workers.emplace_back(worker, i);
  }

  uint64_t last_report = 0;
  while (!failed.load(std::memory_order_relaxed)) {
    const uint64_t done = completed.load(std::memory_order_relaxed);
    if (done >= prepared.node_count) break;
    if (done >= last_report + 250000) {
      last_report = done;
      const double percent =
          prepared.node_count == 0
              ? 100.0
              : 100.0 * double(done) / double(prepared.node_count);
      std::cout << "Resolve children: " << done << "/"
                << prepared.node_count << " (" << std::fixed
                << std::setprecision(1) << percent << "%), elapsed "
                << FormatDuration(std::chrono::steady_clock::now() - start)
                << std::endl;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }
  for (auto& thread : workers) thread.join();

  if (failed.load()) {
    throw std::runtime_error(error_message);
  }
  for (const auto& stats : thread_stats) *combined_stats += stats;
  return children;
}

std::optional<uint32_t> SelectGoteMove(
    const MappedYbb& book, const ExitGraph& graph,
    const GoteExitSolution& solution, uint32_t node) {
  const auto entry = book.Entry(node);
  if ((entry.packed_sfen.data[0] & 1U) == 0 || entry.move_count == 0) {
    return std::nullopt;
  }

  std::optional<uint32_t> best_normal;
  std::optional<uint32_t> best_exit;
  uint32_t best_distance = jhbr2::kExitDistanceInfinite;
  int best_exit_eval = std::numeric_limits<int>::min();

  for (uint32_t local = 0; local < entry.move_count; ++local) {
    const auto record = book.MoveAt(entry, local);
    if (!PlausibleMove(record.move)) continue;

    if (!best_normal) {
      best_normal = local;
    } else {
      const auto current = book.MoveAt(entry, *best_normal);
      if (record.eval > current.eval ||
          (record.eval == current.eval && record.move < current.move)) {
        best_normal = local;
      }
    }

    const uint32_t child = graph.EdgeChild(node, local);
    if (child == jhbr2::kExitDisallowed) continue;
    uint32_t distance = jhbr2::kExitDistanceInfinite;
    if (child == jhbr2::kExitLeaf) {
      distance = 1;
    } else if (solution.distance[child] != jhbr2::kExitDistanceInfinite) {
      distance = solution.distance[child] + 1;
    }

    if (distance < best_distance ||
        (distance == best_distance &&
         (int(record.eval) > best_exit_eval ||
          (int(record.eval) == best_exit_eval && best_exit &&
           record.move < book.MoveAt(entry, *best_exit).move)))) {
      best_exit = local;
      best_distance = distance;
      best_exit_eval = record.eval;
    }
  }

  if (solution.distance[node] != jhbr2::kExitDistanceInfinite && best_exit) {
    return best_exit;
  }
  return best_normal;
}

std::filesystem::path TemporaryOutputPath(
    const std::filesystem::path& output) {
  return output.string() + ".tmp." + std::to_string(getpid());
}

std::filesystem::path BackupPath(const std::filesystem::path& output) {
  const auto ticks = std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
  std::filesystem::path candidate =
      output.string() + ".previous." + std::to_string(ticks);
  for (uint32_t suffix = 1; std::filesystem::exists(candidate); ++suffix) {
    candidate = output.string() + ".previous." + std::to_string(ticks) +
                "." + std::to_string(suffix);
  }
  return candidate;
}

struct OutputStats {
  uint64_t positions = 0;
  uint64_t forced = 0;
  uint64_t fallback = 0;
  uint64_t immediate = 0;
  uint32_t max_distance = 0;
};

OutputStats WriteOutput(const Options& options, const MappedYbb& book,
                        const ExitGraph& graph,
                        const GoteExitSolution& solution) {
  OutputStats stats;
  for (uint32_t node = 0; node < graph.NodeCount(); ++node) {
    const auto entry = book.Entry(node);
    if ((entry.packed_sfen.data[0] & 1U) == 0) continue;
    if (!SelectGoteMove(book, graph, solution, node)) continue;
    ++stats.positions;
    if (solution.distance[node] == jhbr2::kExitDistanceInfinite) {
      ++stats.fallback;
    } else {
      ++stats.forced;
      stats.max_distance =
          std::max(stats.max_distance, solution.distance[node]);
      if (solution.distance[node] == 1) ++stats.immediate;
    }
  }

  const auto temporary = TemporaryOutputPath(options.output);
  if (std::filesystem::exists(temporary)) {
    throw std::runtime_error("temporary output already exists: " +
                             temporary.string());
  }
  std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot create output: " + temporary.string());
  }
  if (!jhbr2::WriteYbbHeader(output, stats.positions,
                             jhbr2::kYbbFlagMoveDepth)) {
    throw std::runtime_error("cannot write output header");
  }

  uint64_t output_move_offset = 0;
  for (uint32_t node = 0; node < graph.NodeCount(); ++node) {
    const auto source = book.Entry(node);
    if ((source.packed_sfen.data[0] & 1U) == 0) continue;
    const auto selected =
        SelectGoteMove(book, graph, solution, node);
    if (!selected) continue;

    YbbIndexEntry output_entry;
    output_entry.packed_sfen = source.packed_sfen;
    output_entry.moves_offset = output_move_offset;
    output_entry.ply = source.ply;
    output_entry.move_count = 1;
    std::array<uint8_t, 44> bytes{};
    EncodeYbbIndexEntry(output_entry, &bytes);
    output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    output_move_offset += 6;
  }

  for (uint32_t node = 0; node < graph.NodeCount(); ++node) {
    const auto source = book.Entry(node);
    if ((source.packed_sfen.data[0] & 1U) == 0) continue;
    const auto selected =
        SelectGoteMove(book, graph, solution, node);
    if (!selected) continue;
    const auto move = book.MoveAt(source, *selected);
    std::array<uint8_t, 6> bytes{};
    EncodeYbbMoveRecord(move, &bytes);
    output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }
  output.close();
  if (!output) {
    throw std::runtime_error("failed while writing output: " +
                             temporary.string());
  }

  std::optional<std::filesystem::path> backup;
  if (std::filesystem::exists(options.output)) {
    if (!options.force) {
      std::filesystem::remove(temporary);
      throw std::runtime_error(
          "output exists; use --force to preserve it as a timestamped backup");
    }
    backup = BackupPath(options.output);
    std::filesystem::rename(options.output, *backup);
  }
  try {
    std::filesystem::rename(temporary, options.output);
  } catch (...) {
    if (backup && !std::filesystem::exists(options.output)) {
      std::filesystem::rename(*backup, options.output);
    }
    throw;
  }
  if (backup) {
    std::cout << "Previous output preserved as: " << backup->string() << '\n';
  }
  return stats;
}

int Run(const Options& options) {
  lczero::ShogiTables::Init();
  const auto started = std::chrono::steady_clock::now();

  if (std::filesystem::exists(options.output) && !options.force) {
    throw std::runtime_error(
        "output exists; pass --force to preserve it as a timestamped backup");
  }

  std::cout << "Input:       " << options.input << '\n'
            << "Output:      " << options.output << '\n'
            << "Eval margin: " << options.eval_margin << " cp\n"
            << "Threads:     " << options.threads << '\n';
  if (options.max_positions != 0) {
    std::cout << "TEST LIMIT:  first " << options.max_positions
              << " positions (not suitable for play)\n";
  }

  MappedYbb book(options.input);
  std::cout << "YBB records: " << book.record_count()
            << ", flags=" << book.flags() << std::endl;
  PreparedBook prepared =
      ValidateAndIndex(book, options.max_positions);
  std::cout << "Indexed " << prepared.node_count << " positions and "
            << prepared.move_count << " moves" << std::endl;

  BuildStats build_stats;
  auto children =
      ResolveChildren(book, prepared, options, &build_stats);
  std::cout << "Resolved children: internal=" << build_stats.internal_moves
            << " leaf=" << build_stats.leaf_moves
            << " disallowed=" << build_stats.disallowed_moves
            << " invalid=" << build_stats.invalid_moves << std::endl;

  ExitGraph graph(&book, prepared.node_count, &children);
  std::cout << "Solving asymmetric exit distances..." << std::endl;
  GoteExitSolution solution = jhbr2::SolveGoteExitGraph(graph);
  std::cout << "Solved: finite=" << solution.finite_nodes
            << " infinite=" << solution.infinite_nodes
            << " internal_edges=" << solution.allowed_internal_edges
            << std::endl;

  // The position hash is no longer needed after child resolution.
  {
    PreparedBook empty;
    prepared.positions.swap(empty.positions);
  }

  const OutputStats output_stats =
      WriteOutput(options, book, graph, solution);
  std::cout << "Wrote " << output_stats.positions << " Gote positions: "
            << output_stats.forced << " force an exit ("
            << output_stats.immediate << " immediately), "
            << output_stats.fallback << " use normal-best fallback"
            << ", max finite distance=" << output_stats.max_distance << '\n'
            << "Completed in "
            << FormatDuration(std::chrono::steady_clock::now() - started)
            << std::endl;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return Run(ParseOptions(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "gote_book_generator: " << error.what() << '\n';
    return 1;
  }
}
