#!/usr/bin/env bash
# Compare V5 and V5-image using P04 only at inference (no example image).
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
source "$repo/.venv-vlm/bin/activate"
cd "$repo"
base='runs/vlm/v5_image_p04_val100_no_example'
mkdir -p "$base"
settings=(
  --split datasets/weld-dataset/kfold_high/fold0/val --limit 100
  --prompt-file prompts/locany_p04_balanced.txt --include-image-size
  --generation-mode slow --temperature 0.7 --top-p 0.9
  --repetition-penalty 1.1 --max-new-tokens 512 --seed 42
)

for model in runs/vlm/locany_v5_p04_epoch1 runs/vlm/locany_v5_image_p04_epoch1; do
  [[ -f "$model/config.json" ]] || { echo "Missing completed model: $model" >&2; exit 1; }
done

for label in V5_P04 V5_image_P04; do
  [[ ! -e "$base/$label/predictions.jsonl" ]] || {
    echo "Output already exists: $base/$label" >&2; exit 1;
  }
done

python scripts/preview_locany_val.py "${settings[@]}" \
  --model runs/vlm/locany_v5_p04_epoch1 \
  --output "$base/V5_P04" 2>&1 | tee "$base/V5_P04.log"

python scripts/preview_locany_val.py "${settings[@]}" \
  --model runs/vlm/locany_v5_image_p04_epoch1 \
  --output "$base/V5_image_P04" 2>&1 | tee "$base/V5_image_P04.log"

python scripts/evaluate_locany_predictions.py \
  "$base/V5_P04/predictions.jsonl" \
  "$base/V5_image_P04/predictions.jsonl" \
  --iou 0.5 --include-class-agnostic --output "$base/scores.csv" \
  2>&1 | tee "$base/scores.txt"
