#!/usr/bin/env bash
# Compare V3 inference prompts on the same 100 fold0 validation images.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv-vlm/bin/activate

v3_model='runs/vlm/locany_fold0_5epoch_equiv'
v3_base='runs/vlm/prompt_four_100_v3/predictions.jsonl'
base_t1='runs/vlm/prompt_four_100_t1_size/predictions.jsonl'
base_balanced='runs/vlm/prompt_four_100_balanced_v4d_size/predictions.jsonl'
for existing in "$v3_base" "$base_t1" "$base_balanced"; do
  if [[ ! -f "$existing" ]]; then
    echo "Missing earlier 100-image result: $existing" >&2
    exit 1
  fi
done

settings=(
  --model "$v3_model"
  --split datasets/weld-dataset/kfold_high/fold0/val
  --limit 100
  --generation-mode slow
  --temperature 0.7 --top-p 0.9 --repetition-penalty 1.1
  --max-new-tokens 512 --seed 42
)

v3_t1='runs/vlm/prompt_v3_100_t1_size'
v3_balanced='runs/vlm/prompt_v3_100_balanced_v4d_size'
mkdir -p "$v3_t1" "$v3_balanced"

python scripts/preview_locany_val.py \
  "${settings[@]}" --include-image-size --output "$v3_t1" \
  2>&1 | tee "$v3_t1/run.log"

python scripts/preview_locany_val.py \
  "${settings[@]}" --prompt-file prompts/locany_weld_balanced_v4d.txt \
  --include-image-size --output "$v3_balanced" \
  2>&1 | tee "$v3_balanced/run.log"

python scripts/evaluate_locany_predictions.py \
  "$v3_base" \
  "$v3_t1/predictions.jsonl" \
  "$v3_balanced/predictions.jsonl" \
  "$base_t1" \
  "$base_balanced" \
  --iou 0.5 --include-class-agnostic \
  --output runs/vlm/prompt_v3_100_comparison.csv \
  2>&1 | tee runs/vlm/prompt_v3_100_comparison.txt

echo 'Order: V3 original, V3 + T1 size, V3 + balanced V4d size, original model + T1 size, original model + balanced V4d size.'
