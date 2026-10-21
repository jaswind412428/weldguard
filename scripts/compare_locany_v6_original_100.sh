#!/usr/bin/env bash
# Compare a short native-format prompt with saved original-model validation runs.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv-vlm/bin/activate

base='runs/vlm/prompt_four_100_original/predictions.jsonl'
t1='runs/vlm/prompt_four_100_t1_size/predictions.jsonl'
v4='runs/vlm/prompt_four_100_balanced_v4d_size/predictions.jsonl'
v5='runs/vlm/prompt_v5_compact_size_original_100/predictions.jsonl'
for input in "$base" "$t1" "$v4" "$v5"; do
  [[ -f "$input" ]] || { echo "Missing baseline: $input" >&2; exit 1; }
done

output='runs/vlm/prompt_v6_native_size_original_100'
if [[ -e "$output/predictions.jsonl" ]]; then
  echo "Existing predictions would be overwritten: $output/predictions.jsonl" >&2
  exit 1
fi
mkdir -p "$output"

python scripts/preview_locany_val.py \
  --model nvidia/LocateAnything-3B \
  --prompt-file prompts/locany_weld_native_v6.txt \
  --include-image-size \
  --split datasets/weld-dataset/kfold_high/fold0/val \
  --limit 100 \
  --generation-mode slow \
  --temperature 0.7 --top-p 0.9 --repetition-penalty 1.1 \
  --max-new-tokens 512 --seed 42 \
  --output "$output" 2>&1 | tee "$output/run.log"

python scripts/evaluate_locany_predictions.py \
  "$base" "$t1" "$v4" "$v5" "$output/predictions.jsonl" \
  --iou 0.5 --include-class-agnostic \
  --output runs/vlm/prompt_v6_original_100_scores.csv \
  2>&1 | tee runs/vlm/prompt_v6_original_100_scores.txt

python - "$output/predictions.jsonl" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding='utf-8').splitlines() if line]
classes = Counter(box['class'] for row in rows for box in row['predictions'])
print('Accepted boxes by class:', dict(classes))
print('Images with no accepted boxes:', sum(not row['predictions'] for row in rows))
print('Rejected numeric boxes:', sum(row['rejected_parsed_boxes'] for row in rows))
PY
