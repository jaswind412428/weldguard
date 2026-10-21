#!/usr/bin/env bash
# Eight-image prompt pilot: original model, four separate class queries per image.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv-vlm/bin/activate

for variant in one_class one_class_size; do
  output="runs/vlm/prompt_c_pilot_${variant}"
  mkdir -p "$output"
  extra=()
  if [[ "$variant" == one_class_size ]]; then
    extra=(--include-image-size)
  fi
  python scripts/preview_locany_val.py \
    --model 'nvidia/LocateAnything-3B' \
    --split datasets/weld-dataset/kfold_high/fold0/val \
    --images 25_12.jpg 25_125.jpg 25_142.jpg 25_159.jpg \
             25_160.jpg 25_161.jpg 25_188.jpg 25_208.jpg \
    --one-class-at-a-time "${extra[@]}" \
    --generation-mode slow \
    --temperature 0.7 --top-p 0.9 --repetition-penalty 1.1 \
    --max-new-tokens 512 --seed 42 \
    --output "$output" \
    2>&1 | tee "$output/run.log"
done

python scripts/evaluate_locany_predictions.py \
  runs/vlm/prompt_ab_pilot_original/predictions.jsonl \
  runs/vlm/prompt_ab_pilot_detailed/predictions.jsonl \
  runs/vlm/prompt_c_pilot_one_class/predictions.jsonl \
  runs/vlm/prompt_c_pilot_one_class_size/predictions.jsonl \
  --iou 0.5 --output runs/vlm/prompt_abcd_pilot_scores.csv \
  2>&1 | tee runs/vlm/prompt_abcd_pilot_scores.txt
