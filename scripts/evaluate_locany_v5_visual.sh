#!/usr/bin/env bash
# Exploratory fold0 validation; four visual queries per image, never test.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
source "$repo/.venv-vlm/bin/activate"
cd "$repo"
output='runs/vlm/v5_visual_val100'
baseline='runs/vlm/v5_image_p04_val100_no_example/V5_P04/predictions.jsonl'
model='runs/vlm/locany_v5_visual_epoch1'
manifest='datasets/derived/locany_v5_visual/manifest.json'
[[ -f "$model/config.json" ]] || { echo "Missing trained V5-visual model: $model" >&2; exit 1; }
[[ -f "$manifest" ]] || { echo "Missing train-only reference manifest: $manifest" >&2; exit 1; }
[[ ! -e "$output/V5_visual/predictions.jsonl" ]] || {
  echo "Predictions already exist: $output/V5_visual/predictions.jsonl" >&2; exit 1;
}
mkdir -p "$output"

python scripts/preview_locany_v5_visual_val.py \
  --model "$model" --manifest "$manifest" \
  --split datasets/weld-dataset/kfold_high/fold0/val \
  --output "$output/V5_visual" --limit 100 \
  --generation-mode slow --temperature 0.7 --top-p 0.9 \
  --repetition-penalty 1.1 --max-new-tokens 512 --seed 42 \
  2>&1 | tee "$output/V5_visual.log"

if [[ -f "$baseline" ]]; then
  python scripts/evaluate_locany_predictions.py \
    "$baseline" "$output/V5_visual/predictions.jsonl" \
    --iou 0.5 --include-class-agnostic --output "$output/scores.csv" \
    2>&1 | tee "$output/scores.txt"
else
  python scripts/evaluate_locany_predictions.py \
    "$output/V5_visual/predictions.jsonl" \
    --iou 0.5 --include-class-agnostic --output "$output/scores.csv" \
    2>&1 | tee "$output/scores.txt"
  echo "V5 baseline missing at $baseline; scores.txt contains V5-visual only."
fi
