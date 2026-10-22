#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv-vlm/bin/activate

baseline='runs/vlm/prompt_four_100_v3/predictions.jsonl'
model='runs/vlm/locany_fold0_5epoch_equiv'
output='runs/vlm/prompt_v3_size_only_100'

[[ -f "$baseline" ]] || { echo "Missing V3 original baseline: $baseline" >&2; exit 1; }
[[ -f "$model/config.json" ]] || { echo "Missing V3 model: $model" >&2; exit 1; }
[[ ! -e "$output/predictions.jsonl" ]] || { echo "Output already exists: $output" >&2; exit 1; }
mkdir -p "$output"

python scripts/preview_locany_val.py \
  --model "$model" --split datasets/weld-dataset/kfold_high/fold0/val \
  --limit 100 --generation-mode slow --temperature 0.7 --top-p 0.9 \
  --repetition-penalty 1.1 --max-new-tokens 512 --seed 42 \
  --prompt-file prompts/locany_v3_original_prompt.txt --image-size-only \
  --output "$output" 2>&1 | tee "$output/run.log"

python scripts/evaluate_locany_predictions.py \
  "$baseline" "$output/predictions.jsonl" --iou 0.5 \
  --include-class-agnostic --output "$output/scores.csv" \
  2>&1 | tee "$output/scores.txt"

echo 'Compare V3 original with V3 plus image dimensions only. Native box parser still reads 0–1000 tokens.'
