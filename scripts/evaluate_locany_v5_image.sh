#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
source "$repo/.venv-vlm/bin/activate"
cd "$repo"
manifest="datasets/derived/locany_v5_image/manifest.json"
[[ -f "$manifest" ]] || { echo "Run scripts/prepare_locany_v5_image.py first" >&2; exit 1; }
base='runs/vlm/v5_image_p04_val100'
mkdir -p "$base"
settings=(
  --split datasets/weld-dataset/kfold_high/fold0/val --limit 100
  --prompt-file prompts/locany_p04_balanced.txt --include-image-size
  --generation-mode slow --temperature 0.7 --top-p 0.9
  --repetition-penalty 1.1 --max-new-tokens 512 --seed 42
  --max-input-tokens 2048
)

for label in V5_P04_oneshot V5_image_P04_oneshot; do
  [[ ! -e "$base/$label/predictions.jsonl" ]] || {
    echo "Output already exists: $base/$label" >&2; exit 1;
  }
done

python scripts/preview_locany_val.py "${settings[@]}" \
  --few-shot-manifest "$manifest" \
  --model runs/vlm/locany_v5_p04_epoch1 \
  --output "$base/V5_P04_oneshot" 2>&1 | tee "$base/V5_P04_oneshot.log"

python scripts/preview_locany_val.py "${settings[@]}" \
  --few-shot-manifest "$manifest" \
  --model runs/vlm/locany_v5_image_p04_epoch1 \
  --output "$base/V5_image_P04_oneshot" 2>&1 | tee "$base/V5_image_P04_oneshot.log"

python scripts/evaluate_locany_predictions.py \
  "$base/V5_P04_oneshot/predictions.jsonl" \
  "$base/V5_image_P04_oneshot/predictions.jsonl" \
  --iou 0.5 --include-class-agnostic --output "$base/scores.csv" \
  2>&1 | tee "$base/scores.txt"
