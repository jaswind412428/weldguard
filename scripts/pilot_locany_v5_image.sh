#!/usr/bin/env bash
# Inference-only A/B on V5 weights before image-example training.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
source "$repo/.venv-vlm/bin/activate"
cd "$repo"
limit="${1:-100}"
[[ "$limit" =~ ^[1-9][0-9]*$ ]] || { echo "Usage: $0 [positive image count]" >&2; exit 2; }
python scripts/prepare_locany_v5_p04.py
python scripts/prepare_locany_v5_image.py

base="runs/vlm/v5_image_p04_pilot${limit}_random653"
mkdir -p "$base"
settings=(
  --model runs/vlm/locany_v5_p04_epoch1
  --split datasets/weld-dataset/kfold_high/fold0/val --limit "$limit"
  --prompt-file prompts/locany_p04_balanced.txt --include-image-size
  --generation-mode slow --temperature 0.7 --top-p 0.9
  --repetition-penalty 1.1 --max-new-tokens 512 --seed 42
  --max-input-tokens 2048
)
for label in V5_P04 V5_P04_oneshot; do
  [[ ! -e "$base/$label/predictions.jsonl" ]] || {
    echo "Output already exists: $base/$label" >&2; exit 1;
  }
done

python scripts/preview_locany_val.py "${settings[@]}" \
  --output "$base/V5_P04" 2>&1 | tee "$base/V5_P04.log"
python scripts/preview_locany_val.py "${settings[@]}" \
  --few-shot-manifest datasets/derived/locany_v5_image/manifest.json \
  --output "$base/V5_P04_oneshot" 2>&1 | tee "$base/V5_P04_oneshot.log"
python scripts/evaluate_locany_predictions.py \
  "$base/V5_P04/predictions.jsonl" \
  "$base/V5_P04_oneshot/predictions.jsonl" \
  --iou 0.5 --include-class-agnostic --output "$base/scores.csv" \
  2>&1 | tee "$base/scores.txt"
