#!/usr/bin/env bash
# Compare one-query, four-class prompts with and without image dimensions.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv-vlm/bin/activate

output='runs/vlm/prompt_d_pilot_all_classes_size'
mkdir -p "$output"

python scripts/preview_locany_val.py \
  --model 'nvidia/LocateAnything-3B' \
  --split datasets/weld-dataset/kfold_high/fold0/val \
  --images 25_12.jpg 25_125.jpg 25_142.jpg 25_159.jpg \
           25_160.jpg 25_161.jpg 25_188.jpg 25_208.jpg \
  --include-image-size \
  --generation-mode slow \
  --temperature 0.7 --top-p 0.9 --repetition-penalty 1.1 \
  --max-new-tokens 512 --seed 42 \
  --output "$output" \
  2>&1 | tee "$output/run.log"

python scripts/evaluate_locany_predictions.py \
  runs/vlm/prompt_ab_pilot_original/predictions.jsonl \
  "$output/predictions.jsonl" \
  --iou 0.5 --output runs/vlm/prompt_all_classes_size_scores.csv \
  2>&1 | tee runs/vlm/prompt_all_classes_size_scores.txt
