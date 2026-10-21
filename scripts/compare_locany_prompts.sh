#!/usr/bin/env bash
# Compare two prompts with the same original LocateAnything model and inference settings.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv-vlm/bin/activate

stage="${1:-pilot}"
case "$stage" in
  pilot)
    selection=(--images 25_12.jpg 25_125.jpg 25_142.jpg 25_159.jpg
      25_160.jpg 25_161.jpg 25_188.jpg 25_208.jpg)
    ;;
  full)
    selection=(--limit 164)
    ;;
  *)
    echo "Usage: bash scripts/compare_locany_prompts.sh [pilot|full]" >&2
    exit 2
    ;;
esac

model='nvidia/LocateAnything-3B'
split='datasets/weld-dataset/kfold_high/fold0/val'
prefix="runs/vlm/prompt_ab_${stage}"
base="${prefix}_original"
detailed="${prefix}_detailed"
mkdir -p "$base" "$detailed"

settings=(
  --model "$model"
  --split "$split"
  --generation-mode slow
  --temperature 0.7
  --top-p 0.9
  --repetition-penalty 1.1
  --max-new-tokens 512
  --seed 42
  "${selection[@]}"
)

python scripts/preview_locany_val.py \
  "${settings[@]}" --output "$base" \
  2>&1 | tee "$base/run.log"

python scripts/preview_locany_val.py \
  "${settings[@]}" \
  --prompt-file prompts/locany_weld_detailed_v1.txt \
  --output "$detailed" \
  2>&1 | tee "$detailed/run.log"

python scripts/evaluate_locany_predictions.py \
  "$base/predictions.jsonl" \
  "$detailed/predictions.jsonl" \
  --iou 0.5 --output "${prefix}_scores.csv" \
  2>&1 | tee "${prefix}_scores.txt"

wc -l "$base/predictions.jsonl" "$detailed/predictions.jsonl"
echo "Comparison saved to ${prefix}_scores.csv"
