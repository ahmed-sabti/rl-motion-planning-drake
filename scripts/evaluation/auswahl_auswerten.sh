#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s CHECKPOINT_COUNT SERVER EXPERIMENT_ROOT OUTPUT_DIR\n' "$0"
  printf 'Experiment root contains run64, run256, and run1024 (2-link: run64-2r, ... with PLANAR2D_ROBOT=2r).\n'
  printf 'Set PLANAR_EVAL_PYTHON if needed.\n'
}
if [[ "${1:-}" == '--help' ]]; then
  usage
  exit 0
fi
if [[ $# -ne 4 || ! "$1" =~ ^[1-9][0-9]*$ ]]; then
  usage >&2
  exit 2
fi
kmax=$1
server=$2
experiment=$3
if [[ ! "$experiment" =~ ^(~?/)[A-Za-z0-9._/-]+$ || "$server" == -* ]]; then
  printf 'Use a server name and a simple absolute or ~/ experiment path.\n' >&2
  exit 2
fi
code=$(cd "$(dirname "$0")/../.." && pwd)
python_bin=${PLANAR_EVAL_PYTHON:-python3}
if [[ -z "${PLANAR_EVAL_PYTHON:-}" && -x "$code/.venv/bin/python" ]]; then
  python_bin="$code/.venv/bin/python"
fi
mkdir -p "$4"
output=$(cd "$4" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
ssh -n -o ConnectTimeout=10 -- "$server" true || exit 2
export PLANAR2D_ROBOT=${PLANAR2D_ROBOT:-3r}
if [[ "$PLANAR2D_ROBOT" == 2r ]]; then prefix=zg suffix=-2r; else prefix=kto suffix=; fi
for width in 64 256 1024; do
  run="run$width$suffix"
  for ((k = 1; k <= kmax; k++)); do
    checkpoint="$prefix$width-s963-at$((k * 1000000))"
    [[ ! -d "$output/$checkpoint" ]] || continue
    source_checkpoint=$checkpoint
    if ! ssh -n -- "$server" "test -f $experiment/$run/$checkpoint.zip"; then
      final="$prefix$width-s963"
      steps=$(ssh -n -- "$server" "test -f $experiment/$run/$final.zip && python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[\"training\"][\"timesteps\"])' $experiment/$run/$final.json" 2>/dev/null || true)
      [[ "$steps" == "$((k * 1000000))" ]] || continue
      source_checkpoint=$final
    fi
    ssh -n -- "$server" "tar cf - -C $experiment/$run $source_checkpoint.zip $source_checkpoint.json" | tar xf - -C "$work"
    if [[ "$source_checkpoint" != "$checkpoint" ]]; then
      mv "$work/$source_checkpoint.zip" "$work/$checkpoint.zip"
      mv "$work/$source_checkpoint.json" "$work/$checkpoint.json"
    fi
    pids=()
    for shard in 0 1 2 3 4; do
      OMP_NUM_THREADS=1 "$python_bin" "$code/scripts/evaluation/run_descriptive.py" select \
        --candidate "$work/$checkpoint" --output "$work/eval/$checkpoint/shard$shard" --shard "$shard/5" \
        >"$work/$checkpoint-$shard.log" 2>&1 &
      pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do
      wait "$pid" || failed=1
    done
    if [[ "$failed" -ne 0 ]]; then
      printf 'Evaluation failed: %s\n' "$checkpoint" >&2
      exit 1
    fi
    for shard in 0 1 2 3 4; do
      [[ -f "$work/eval/$checkpoint/shard$shard/done" ]] || exit 1
    done
    cp -R "$work/eval/$checkpoint" "$output/"
    printf 'Evaluated: %s\n' "$checkpoint"
  done
done
