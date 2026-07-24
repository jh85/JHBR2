#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  tools/run_strength_test.sh openings SOURCE.ybb OUTPUT.txt [generator options]
  tools/run_strength_test.sh match [strength_test.py options]

Examples:
  tools/run_strength_test.sh openings /data/books/user_book1.ybb \
    strength-openings.txt --count 512

  tools/run_strength_test.sh match \
    --engine-a ./build-trt/jhbr2 \
    --engine-b ./build-trt/jhbr2 \
    --openings strength-openings.txt --pairs 200 --nodes 100000 \
    --option-a OnnxModel=/data/model.engine \
    --option-b OnnxModel=/data/model.engine

The wrapper reuses a Python with cshogi when one is available. Otherwise it
creates build-strength/venv and installs cshogi there. Set STRENGTH_PYTHON to
force a particular interpreter.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

subcommand=$1
if [[ $subcommand == "-h" || $subcommand == "--help" || $subcommand == "help" ]]; then
  usage
  exit 0
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
venv_dir=${STRENGTH_VENV:-"$repo_dir/build-strength/venv"}

works_with_cshogi() {
  "$1" -c 'import cshogi' >/dev/null 2>&1
}

if [[ -n ${STRENGTH_PYTHON:-} ]]; then
  python_bin=$STRENGTH_PYTHON
  if ! works_with_cshogi "$python_bin"; then
    echo "STRENGTH_PYTHON cannot import cshogi: $python_bin" >&2
    exit 2
  fi
elif [[ -x $venv_dir/bin/python ]] && works_with_cshogi "$venv_dir/bin/python"; then
  python_bin=$venv_dir/bin/python
elif works_with_cshogi python3; then
  python_bin=$(command -v python3)
else
  mkdir -p -- "$(dirname -- "$venv_dir")"
  if command -v uv >/dev/null 2>&1; then
    uv venv "$venv_dir"
    uv pip install --python "$venv_dir/bin/python" "cshogi==1.0.4"
  else
    python3 -m venv "$venv_dir"
    "$venv_dir/bin/python" -m pip install --upgrade pip
    "$venv_dir/bin/python" -m pip install "cshogi==1.0.4"
  fi
  python_bin=$venv_dir/bin/python
fi

shift
case "$subcommand" in
  openings)
    exec "$python_bin" "$script_dir/generate_strength_openings.py" "$@"
    ;;
  match)
    exec "$python_bin" "$script_dir/strength_test.py" "$@"
    ;;
  *)
    echo "Unknown subcommand: $subcommand" >&2
    usage
    exit 2
    ;;
esac
