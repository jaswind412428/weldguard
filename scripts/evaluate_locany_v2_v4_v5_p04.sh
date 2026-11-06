#!/usr/bin/env bash
# Compare the three checkpoints using the same P04 inference prompt and validation images.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
source "$repo/.venv-vlm/bin/activate"
cd "$repo"

base='runs/vlm/v2_v4_v5_p04_val100'
settings=(
  --split datasets/weld-dataset/kfold_high/fold0/val --limit 100
  --prompt-file prompts/locany_p04_balanced.txt --include-image-size
  --generation-mode slow --temperature 0.7 --top-p 0.9
  --repetition-penalty 1.1 --max-new-tokens 512 --seed 42
)
labels=(V2_P04 V4_P04 V5_P04)
models=(
  runs/vlm/locany_fold0_train653_v2
  runs/vlm/locany_v4_p05_epoch1
  runs/vlm/locany_v5_p04_epoch1
)

for i in "${!labels[@]}"; do
  [[ -f "${models[$i]}/config.json" ]] || {
    echo "Missing completed model: ${models[$i]}" >&2
    exit 1
  }
  [[ ! -e "$base/${labels[$i]}/predictions.jsonl" ]] || {
    echo "Output already exists: $base/${labels[$i]}" >&2
    exit 1
  }
done

for i in "${!labels[@]}"; do
  mkdir -p "$base/${labels[$i]}"
  python scripts/preview_locany_val.py "${settings[@]}" \
    --model "${models[$i]}" --output "$base/${labels[$i]}" \
    2>&1 | tee "$base/${labels[$i]}/run.log"
done

python scripts/evaluate_locany_predictions.py \
  "$base/V2_P04/predictions.jsonl" \
  "$base/V4_P04/predictions.jsonl" \
  "$base/V5_P04/predictions.jsonl" \
  --iou 0.5 --include-class-agnostic --output "$base/scores.csv" \
  2>&1 | tee "$base/scores.txt"
