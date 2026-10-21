#!/usr/bin/env bash
# Pilot a concise weld-defect prompt on the same eight validation images.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv-vlm/bin/activate

output='runs/vlm/prompt_balanced_pilot_v4d'
mkdir -p "$output"

python scripts/preview_locany_val.py \
  --model nvidia/LocateAnything-3B \
  --prompt-file prompts/locany_weld_balanced_v4d.txt \
  --split datasets/weld-dataset/kfold_high/fold0/val \
  --images 25_12.jpg 25_125.jpg 25_142.jpg 25_159.jpg \
           25_160.jpg 25_161.jpg 25_188.jpg 25_208.jpg \
  --generation-mode slow \
  --temperature 0.7 --top-p 0.9 --repetition-penalty 1.1 \
  --max-new-tokens 512 --seed 42 \
  --output "$output" \
  2>&1 | tee "$output/run.log"

python scripts/evaluate_locany_predictions.py \
  runs/vlm/prompt_ab_pilot_original/predictions.jsonl \
  runs/vlm/prompt_four_pilot_v3_original_prompt/predictions.jsonl \
  runs/vlm/prompt_d_pilot_all_classes_size/predictions.jsonl \
  runs/vlm/prompt_example_pilot_v4b/predictions.jsonl \
  "$output/predictions.jsonl" \
  --iou 0.5 --output runs/vlm/prompt_balanced_pilot_scores.csv \
  2>&1 | tee runs/vlm/prompt_balanced_pilot_scores.txt

python - "$output/predictions.jsonl" <<'PY'
import json
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding='utf-8').splitlines() if line]
print('Accepted boxes:', sum(len(row['predictions']) for row in rows))
print('Rejected numeric boxes:', sum(row['rejected_parsed_boxes'] for row in rows))
for row in rows[:2]:
    print(Path(row['image']).name, repr(str(row['answer'])[:900]))
PY
