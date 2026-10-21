#!/usr/bin/env bash
# Four-way eight-image validation pilot. Existing original and T1 results are reused.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv-vlm/bin/activate

original='runs/vlm/prompt_ab_pilot_original/predictions.jsonl'
t1='runs/vlm/prompt_d_pilot_all_classes_size/predictions.jsonl'
for existing in "$original" "$t1"; do
  if [[ ! -f "$existing" ]]; then
    echo "Missing prior pilot result: $existing" >&2
    exit 1
  fi
done

selection=(--images 25_12.jpg 25_125.jpg 25_142.jpg 25_159.jpg
  25_160.jpg 25_161.jpg 25_188.jpg 25_208.jpg)
settings=(
  --split datasets/weld-dataset/kfold_high/fold0/val
  --generation-mode slow
  --temperature 0.7 --top-p 0.9 --repetition-penalty 1.1
  --max-new-tokens 512 --seed 42
  "${selection[@]}"
)

v3_dir='runs/vlm/prompt_four_pilot_v3_original_prompt'
prompt_dir='runs/vlm/prompt_four_pilot_base_detailed_training_prompt'
mkdir -p "$v3_dir" "$prompt_dir"

python scripts/preview_locany_val.py \
  "${settings[@]}" \
  --model runs/vlm/locany_fold0_5epoch_equiv \
  --output "$v3_dir" \
  2>&1 | tee "$v3_dir/run.log"

python scripts/preview_locany_val.py \
  "${settings[@]}" \
  --model nvidia/LocateAnything-3B \
  --prompt-file prompts/locany_weld_training_v4.txt \
  --output "$prompt_dir" \
  2>&1 | tee "$prompt_dir/run.log"

python scripts/evaluate_locany_predictions.py \
  "$original" \
  "$v3_dir/predictions.jsonl" \
  "$t1" \
  "$prompt_dir/predictions.jsonl" \
  --iou 0.5 --output runs/vlm/prompt_four_pilot_scores.csv \
  2>&1 | tee runs/vlm/prompt_four_pilot_scores.txt

echo 'Model order: original, V3, T1 (original + image size), original + detailed training prompt.'
echo 'The detailed prompt arm uses the original weights; it is not a newly trained V4 model.'
