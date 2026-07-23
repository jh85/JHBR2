#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 SOURCE.ybb [OUTPUT.ybb] [generator options]" >&2
  echo "Example: $0 /data/books/user_book1.ybb user_book1_gote_exit.ybb --force" >&2
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
source_book=$1
shift

if [[ ! -f $source_book || ! -r $source_book ]]; then
  echo "Source YBB is not a readable file: $source_book" >&2
  exit 2
fi

if [[ $# -gt 0 && $1 != --* ]]; then
  output_book=$1
  shift
else
  output_book="$repo_dir/user_book1_gote_exit.ybb"
fi

eval_margin=${GOTE_EXIT_EVAL_MARGIN:-30}
generator_threads=${GOTE_EXIT_THREADS:-$(nproc)}
build_dir=${GOTE_BOOK_BUILD_DIR:-"$repo_dir/build-book"}

cmake -S "$repo_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" --target gote_book_generator \
  --parallel "${GOTE_BOOK_BUILD_JOBS:-16}"

exec "$build_dir/gote_book_generator" \
  --input "$source_book" \
  --output "$output_book" \
  --eval-margin "$eval_margin" \
  --threads "$generator_threads" \
  "$@"
