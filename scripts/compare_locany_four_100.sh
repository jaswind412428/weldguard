#!/usr/bin/env bash
# Four-arm comparison on the same deterministic 100 fold0 validation images.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv-vlm/bin/activate

settings=(
  --split datasets/weld-dataset/kfold_high/fold0/val
  --limit 100
  --generation-mode slow
  --temperature 0.7 --top-p 0.9 --repetition-penalty 1.1
  --max-new-tokens 512 --seed 42
)
prefix='runs/vlm/prompt_four_100'
base="${prefix}_original"
v3="${prefix}_v3"
t1="${prefix}_t1_size"
detailed="${prefix}_base_detailed"

for output in "$base" "$v3" "$t1" "$detailed"; do
  mkdir -p "$output"
done

python scripts/preview_locany_val.py \
  "${settings[@]}" --model nvidia/LocateAnything-3B --output "$base" \
  2>&1 | tee "$base/run.log"

python scripts/preview_locany_val.py \
  "${settings[@]}" --model runs/vlm/locany_fold0_5epoch_equiv --output "$v3" \
  2>&1 | tee "$v3/run.log"

python scripts/preview_locany_val.py \
  "${settings[@]}" --model nvidia/LocateAnything-3B \
  --include-image-size --output "$t1" \
  2>&1 | tee "$t1/run.log"

python scripts/preview_locany_val.py \
  "${settings[@]}" --model nvidia/LocateAnything-3B \
  --prompt-file prompts/locany_weld_training_v4.txt --output "$detailed" \
  2>&1 | tee "$detailed/run.log"

python scripts/evaluate_locany_predictions.py \
  "$base/predictions.jsonl" \
  "$v3/predictions.jsonl" \
  "$t1/predictions.jsonl" \
  "$detailed/predictions.jsonl" \
  --iou 0.5 --output "${prefix}_scores.csv" \
  2>&1 | tee "${prefix}_scores.txt"

python - "$detailed/predictions.jsonl" <<'PY' | tee "${prefix}_detailed_diagnostics.txt"
import json
import sys
from collections import Counter
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding='utf-8').splitlines() if line]
print('Detailed prompt diagnostics')
print('images:', len(rows))
print('images with accepted boxes:', sum(bool(row['predictions']) for row in rows))
print('rejected numeric boxes:', sum(row['rejected_parsed_boxes'] for row in rows))
print('answer prefixes for first 5 images:')
for row in rows[:5]:
    print(Path(row['image']).name, repr(str(row['answer'])[:450]))
PY
