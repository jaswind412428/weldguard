#!/usr/bin/env bash
# Evaluate the two 50-step pilot checkpoints on the same 100 validation images.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv-vlm/bin/activate

v3a_model='runs/vlm/locany_50step_v3a'
v4_model='runs/vlm/locany_50step_v4'
for model in "$v3a_model" "$v4_model"; do
  if [[ ! -f "$model/config.json" ]]; then
    echo "Missing completed model directory: $model" >&2
    exit 1
  fi
done

settings=(
  --split datasets/weld-dataset/kfold_high/fold0/val
  --limit 100
  --generation-mode slow
  --temperature 0.7 --top-p 0.9 --repetition-penalty 1.1
  --max-new-tokens 512 --seed 42
)
v3a_out='runs/vlm/locany_50step_v3a_val100'
v4_out='runs/vlm/locany_50step_v4_val100'
mkdir -p "$v3a_out" "$v4_out"

python scripts/preview_locany_val.py \
  "${settings[@]}" --model "$v3a_model" --output "$v3a_out" \
  2>&1 | tee "$v3a_out/run.log"

python scripts/preview_locany_val.py \
  "${settings[@]}" --model "$v4_model" \
  --prompt-file prompts/locany_weld_balanced_v4d.txt \
  --include-image-size --output "$v4_out" \
  2>&1 | tee "$v4_out/run.log"

python scripts/evaluate_locany_predictions.py \
  "$v3a_out/predictions.jsonl" \
  "$v4_out/predictions.jsonl" \
  --iou 0.5 --include-class-agnostic \
  --output runs/vlm/locany_50step_pair_val100_scores.csv \
  2>&1 | tee runs/vlm/locany_50step_pair_val100_scores.txt
