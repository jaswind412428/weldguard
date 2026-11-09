#!/usr/bin/env bash
# Four conditions distinguish the effect of one-shot training from one-shot inference.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
source "$repo/.venv-vlm/bin/activate"
cd "$repo"
base='runs/vlm/v6_xy_image_p04_val100'
baseline='runs/vlm/v5_v6_xy_p04_val100/V6_XY_P04/predictions.jsonl'
manifest='datasets/derived/locany_v6_xy_image/manifest.json'
v6='runs/vlm/locany_v6_xy_p04_epoch1'
v6_image='runs/vlm/locany_v6_xy_image_p04_epoch1'
settings=(
  --split datasets/weld-dataset/kfold_high/fold0/val --limit 100
  --prompt-file prompts/locany_p04_balanced.txt --include-image-size
  --generation-mode slow --temperature 0.7 --top-p 0.9
  --repetition-penalty 1.1 --max-new-tokens 512 --seed 42
)

for model in "$v6" "$v6_image"; do
  [[ -f "$model/config.json" ]] || { echo "Missing completed model: $model" >&2; exit 1; }
done
[[ -f "$manifest" ]] || { echo "Missing one-shot reference manifest: $manifest" >&2; exit 1; }
for label in V6_P04 V6_one_shot V6_image_P04 V6_image_one_shot; do
  [[ ! -e "$base/$label/predictions.jsonl" ]] || {
    echo "Predictions already exist: $base/$label" >&2; exit 1;
  }
done

if [[ ! -f "$baseline" ]]; then
  mkdir -p "$base/V6_P04"
  python scripts/preview_locany_val.py "${settings[@]}" \
    --model "$v6" --output "$base/V6_P04" 2>&1 | tee "$base/V6_P04.log"
  baseline="$base/V6_P04/predictions.jsonl"
fi

mkdir -p "$base/V6_one_shot" "$base/V6_image_P04" "$base/V6_image_one_shot"
python scripts/preview_locany_val.py "${settings[@]}" \
  --few-shot-manifest "$manifest" --model "$v6" \
  --output "$base/V6_one_shot" 2>&1 | tee "$base/V6_one_shot.log"
python scripts/preview_locany_val.py "${settings[@]}" \
  --model "$v6_image" --output "$base/V6_image_P04" \
  2>&1 | tee "$base/V6_image_P04.log"
python scripts/preview_locany_val.py "${settings[@]}" \
  --few-shot-manifest "$manifest" --model "$v6_image" \
  --output "$base/V6_image_one_shot" 2>&1 | tee "$base/V6_image_one_shot.log"

python scripts/evaluate_locany_predictions.py \
  "$baseline" \
  "$base/V6_one_shot/predictions.jsonl" \
  "$base/V6_image_P04/predictions.jsonl" \
  "$base/V6_image_one_shot/predictions.jsonl" \
  --iou 0.5 --include-class-agnostic --output "$base/scores.csv" \
  2>&1 | tee "$base/scores.txt"
