#!/usr/bin/env bash
# Compare detailed prompts without and with one valid format example.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv-vlm/bin/activate

settings=(
  --model nvidia/LocateAnything-3B
  --split datasets/weld-dataset/kfold_high/fold0/val
  --images 25_12.jpg 25_125.jpg 25_142.jpg 25_159.jpg
           25_160.jpg 25_161.jpg 25_188.jpg 25_208.jpg
  --generation-mode slow
  --temperature 0.7 --top-p 0.9 --repetition-penalty 1.1
  --max-new-tokens 512 --seed 42
)

for variant in v4b v4c_example; do
  output="runs/vlm/prompt_example_pilot_${variant}"
  mkdir -p "$output"
  python scripts/preview_locany_val.py \
    "${settings[@]}" \
    --prompt-file "prompts/locany_weld_training_${variant}.txt" \
    --output "$output" \
    2>&1 | tee "$output/run.log"
done

python scripts/evaluate_locany_predictions.py \
  runs/vlm/prompt_ab_pilot_original/predictions.jsonl \
  runs/vlm/prompt_four_pilot_v3_original_prompt/predictions.jsonl \
  runs/vlm/prompt_d_pilot_all_classes_size/predictions.jsonl \
  runs/vlm/prompt_example_pilot_v4b/predictions.jsonl \
  runs/vlm/prompt_example_pilot_v4c_example/predictions.jsonl \
  --iou 0.5 --output runs/vlm/prompt_example_pilot_scores.csv \
  2>&1 | tee runs/vlm/prompt_example_pilot_scores.txt

python - <<'PY'
import json
from pathlib import Path

example = [0.1, 0.2, 0.15, 0.25]
for variant in ('v4b', 'v4c_example'):
    path = Path(f'runs/vlm/prompt_example_pilot_{variant}/predictions.jsonl')
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
    copied = sum(pred['class'] == 'pore' and pred['box_01'] == example
                 for row in rows for pred in row['predictions'])
    print(f'{variant}: accepted example box copied {copied} times')
    for row in rows[:2]:
        print(Path(row['image']).name, 'predicted=', len(row['predictions']),
              'rejected=', row['rejected_parsed_boxes'],
              'answer=', repr(str(row['answer'])[:600]))
PY
