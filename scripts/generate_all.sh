#!/usr/bin/env bash
set -euo pipefail

MODEL="deepseek-v4-flash"
LANGUAGE="cpp"
PROBLEMS_DIR="data/problems"

for problem_dir in "$PROBLEMS_DIR"/*; do
  [ -d "$problem_dir" ] || continue
  [ -f "$problem_dir/meta.json" ] || continue

  problem_id="$(basename "$problem_dir")"

  python generate.py \
    --problem-id "$problem_id" \
    --problems-dir "$PROBLEMS_DIR" \
    --models "$MODEL" \
    --language "$LANGUAGE"
done
