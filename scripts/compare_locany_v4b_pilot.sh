#!/usr/bin/env bash
# Check revised detailed prompt on eight fixed validation images before larger runs.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv-vlm/bin/activate

output='runs/vlm/prompt_four_pilot_base_detailed_v4b'
mkdir -p "$output"

python scripts/preview_locany_val.py \
  --model nvidia/LocateAnything-3B \
  --prompt-file prompts/locany_weld_training_v4b.txt \
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
  "$output/predictions.jsonl" \
  --iou 0.5 --output runs/vlm/prompt_four_pilot_v4b_scores.csv \
  2>&1 | tee runs/vlm/prompt_four_pilot_v4b_scores.txt

python - "$output/predictions.jsonl" <<'PY'
import json
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding='utf-8').splitlines() if line]
for row in rows[:3]:
    print(Path(row['image']).name, 'predicted=', len(row['predictions']),
          'rejected=', row['rejected_parsed_boxes'], 'answer=', repr(str(row['answer'])[:700]))
PY
