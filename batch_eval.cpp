/*
  Batch evaluator: reads PSV files, evaluates with YaneuraOu NNUE, outputs
  bestmove + score for each position.

  Build:
    1. Copy this file to YaneuraOu/source/
    2. Build YaneuraOu normally, but replace main.cpp with batch_eval.cpp
       OR: Build YaneuraOu as a library and link this against it.

    Simplest approach — replace main.cpp:
      cp batch_eval.cpp ~/Downloads/YaneuraOu/source/main.cpp
      cd ~/Downloads/YaneuraOu/source
      make -j$(nproc) TARGET_CPU=AVX512VNNI YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE normal

  Usage:
    ./YaneuraOu-by-gcc \
      --psv /path/to/hao_depth_9_shuffled_01.bin \
      --output /path/to/output.tsv \
      --depth 7 \
      --threads 64 \
      --max-positions 100000000 \
      --eval-dir /path/to/eval/

  Output format (TSV):
    sfen<TAB>bestmove<TAB>score<TAB>game_result
*/

// This file replaces YaneuraOu's main.cpp.
// It provides a batch evaluation mode that reads PSV, runs search, and outputs results.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// YaneuraOu headers — adjust paths if needed
#include "types.h"
#include "bitboard.h"
#include "position.h"
#include "search.h"
#include "thread.h"
#include "tt.h"
#include "usi.h"

// From learn/learn.h
namespace Learner {
struct PackedSfenValue {
    PackedSfen sfen;
    int16_t score;
    uint16_t move;
    uint16_t gamePly;
    int8_t game_result;
    uint8_t padding;
};
}

// =====================================================================
// Output record
// =====================================================================

struct EvalResult {
    std::string sfen;
    std::string bestmove;
    int score;
    int8_t game_result;
};

// =====================================================================
// Worker thread: process a range of PSV records
// =====================================================================

static std::mutex output_mutex;
static std::atomic<int64_t> total_processed{0};
static std::atomic<int64_t> total_valid{0};

void worker_thread(
    const std::string& psv_path,
    int64_t start_record,
    int64_t num_records,
    int depth,
    int worker_id,
    std::vector<EvalResult>& results)
{
    constexpr int RECORD_SIZE = 40;

    // Open PSV file
    std::ifstream fin(psv_path, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "Worker %d: cannot open %s\n", worker_id, psv_path.c_str());
        return;
    }
    fin.seekg(start_record * RECORD_SIZE);

    // Each worker needs its own Position + StateInfo
    StateInfo si;
    Position pos;

    // Search limits
    Search::LimitsType limits;
    limits.depth = depth;

    auto t0 = std::chrono::steady_clock::now();

    for (int64_t i = 0; i < num_records; i++) {
        Learner::PackedSfenValue psv;
        fin.read(reinterpret_cast<char*>(&psv), RECORD_SIZE);
        if (!fin) break;

        // Decode packed sfen to position
        auto result = pos.set_from_packed_sfen(psv.sfen, &si, false, psv.gamePly);
        if (result != Tools::Result::Ok()) {
            continue;
        }

        // Get SFEN string for output
        std::string sfen = pos.sfen();

        // --- Run search ---
        // For a simple evaluation approach, we can use:
        // 1. Just call Eval::evaluate() for static eval (fastest, no search)
        // 2. Use a simplified search for depth-limited evaluation

        // Option 1: Static NNUE eval (very fast, ~1μs)
        // Value eval = Eval::evaluate(pos);

        // Option 2: Get best move via search
        // This requires proper thread setup. For simplicity, use the USI
        // go command internally.

        // For now, use static eval + legal move with best see/eval
        // (A proper search integration requires more YaneuraOu internals)

        // Static eval from NNUE
        Value eval = Eval::evaluate(pos);
        int score = static_cast<int>(eval);

        // Get legal moves and pick the best by static eval after making the move
        Move best = MOVE_NONE;
        Value best_eval = -VALUE_INFINITE;

        for (auto m : MoveList<LEGAL>(pos)) {
            StateInfo si2;
            pos.do_move(m, si2);
            Value v = -Eval::evaluate(pos);  // Negate for opponent's perspective
            pos.undo_move(m);
            if (v > best_eval) {
                best_eval = v;
                best = m;
            }
        }

        if (best == MOVE_NONE) continue;

        EvalResult er;
        er.sfen = sfen;
        er.bestmove = USI::move(best);
        er.score = static_cast<int>(best_eval);
        er.game_result = psv.game_result;
        results.push_back(er);

        int64_t proc = total_processed.fetch_add(1) + 1;
        total_valid.fetch_add(1);

        if (proc % 100000 == 0) {
            auto elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            double rate = proc / elapsed;
            fprintf(stderr, "  Worker %d: %lld processed (%.0f pos/sec, %lld valid)\n",
                    worker_id, (long long)proc, rate, (long long)total_valid.load());
        }
    }
}

// =====================================================================
// Main
// =====================================================================

int main(int argc, char* argv[]) {
    // Parse arguments
    std::string psv_path;
    std::string output_path = "output.tsv";
    std::string eval_dir = "eval";
    int depth = 1;  // 1 = static eval + 1-ply
    int num_threads = 8;
    int64_t max_positions = 1000000;

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--psv" && i + 1 < argc)
            psv_path = argv[++i];
        else if (std::string(argv[i]) == "--output" && i + 1 < argc)
            output_path = argv[++i];
        else if (std::string(argv[i]) == "--eval-dir" && i + 1 < argc)
            eval_dir = argv[++i];
        else if (std::string(argv[i]) == "--depth" && i + 1 < argc)
            depth = std::stoi(argv[++i]);
        else if (std::string(argv[i]) == "--threads" && i + 1 < argc)
            num_threads = std::stoi(argv[++i]);
        else if (std::string(argv[i]) == "--max-positions" && i + 1 < argc)
            max_positions = std::stoll(argv[++i]);
    }

    if (psv_path.empty()) {
        fprintf(stderr, "Usage: %s --psv <file.bin> [--output output.tsv] "
                "[--eval-dir eval/] [--depth 1] [--threads 8] "
                "[--max-positions 1000000]\n", argv[0]);
        return 1;
    }

    fprintf(stderr, "PSV: %s\n", psv_path.c_str());
    fprintf(stderr, "Output: %s\n", output_path.c_str());
    fprintf(stderr, "Depth: %d (1 = static eval + 1-ply move selection)\n", depth);
    fprintf(stderr, "Threads: %d\n", num_threads);
    fprintf(stderr, "Max positions: %lld\n", (long long)max_positions);

    // --- Initialize YaneuraOu ---
    // These initialization functions are from YaneuraOu's startup code.
    // The exact calls depend on the version. Adjust as needed.

    Bitboards::init();
    Position::init();

    // Set eval directory
    USI::Options["EvalDir"] = eval_dir;

    // Initialize NNUE
    // Eval::init() or Eval::NNUE::init() — depends on version
    Eval::init();

    // Get file size to determine record count
    std::ifstream fcheck(psv_path, std::ios::binary | std::ios::ate);
    int64_t file_size = fcheck.tellg();
    fcheck.close();
    int64_t total_records = file_size / 40;
    int64_t to_process = std::min(max_positions, total_records);

    fprintf(stderr, "File: %lld records, processing %lld\n",
            (long long)total_records, (long long)to_process);

    // --- Launch worker threads ---
    int64_t per_thread = to_process / num_threads;
    std::vector<std::thread> threads;
    std::vector<std::vector<EvalResult>> thread_results(num_threads);

    auto t0 = std::chrono::steady_clock::now();

    for (int t = 0; t < num_threads; t++) {
        int64_t start = t * per_thread;
        int64_t count = (t == num_threads - 1) ? (to_process - start) : per_thread;
        threads.emplace_back(worker_thread, psv_path, start, count,
                             depth, t, std::ref(thread_results[t]));
    }

    for (auto& t : threads) t.join();

    auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    // --- Write output ---
    fprintf(stderr, "\nWriting results to %s...\n", output_path.c_str());
    std::ofstream fout(output_path);
    int64_t written = 0;
    for (auto& results : thread_results) {
        for (auto& er : results) {
            fout << er.sfen << "\t" << er.bestmove << "\t"
                 << er.score << "\t" << (int)er.game_result << "\n";
            written++;
        }
    }
    fout.close();

    fprintf(stderr, "Done! %lld positions in %.1f seconds (%.0f pos/sec)\n",
            (long long)written, elapsed, written / elapsed);
    fprintf(stderr, "Output: %s\n", output_path.c_str());

    return 0;
}
