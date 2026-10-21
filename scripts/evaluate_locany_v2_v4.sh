#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv-vlm/bin/activate
base='runs/vlm/v2_v4_p05_val100'
settings=(--split datasets/weld-dataset/kfold_high/fold0/val --limit 100
  --generation-mode slow --temperature 0.7 --top-p 0.9
  --repetition-penalty 1.1 --max-new-tokens 512 --seed 42 --localization-only)
for model in runs/vlm/locany_fold0_train653_v2 runs/vlm/locany_v4_p05_epoch1; do
  [[ -f "$model/config.json" ]] || { echo "Missing completed model: $model"; exit 1; }
done
for label in V2_P01 V2_P05 V4_P05; do
  [[ ! -e "$base/$label/predictions.jsonl" ]] || { echo "Output already exists: $base/$label"; exit 1; }
done
for label in V2_P01 V2_P05 V4_P05; do
  model='runs/vlm/locany_fold0_train653_v2'
  [[ "$label" != V4_P05 ]] || model='runs/vlm/locany_v4_p05_epoch1'
  prompt_args=()
  [[ "$label" == V2_P01 ]] || prompt_args=(--prompt-file prompts/locany_p05_gt_features.txt --include-image-size)
  mkdir -p "$base/$label"
  python scripts/preview_locany_val.py "${settings[@]}" "${prompt_args[@]}" \
    --model "$model" --output "$base/$label" 2>&1 | tee "$base/$label/run.log"
done
python scripts/evaluate_locany_predictions.py \
  "$base/V2_P01/predictions.jsonl" "$base/V2_P05/predictions.jsonl" "$base/V4_P05/predictions.jsonl" \
  --localization-only --iou 0.5 --output "$base/scores.csv" 2>&1 | tee "$base/scores.txt"
echo 'Order: V2+P01 reference; V2+P05 vs V4+P05 uses the same inference prompt.'
