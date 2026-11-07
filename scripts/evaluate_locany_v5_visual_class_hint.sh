#!/usr/bin/env bash
# A/B on the same V5-visual weights and train-only crops; change text only.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
source "$repo/.venv-vlm/bin/activate"
cd "$repo"
baseline='runs/vlm/v5_visual_val100/V5_visual/predictions.jsonl'
output='runs/vlm/v5_visual_class_hint_ab100'
named="$output/named_class/predictions.jsonl"
[[ -f "$baseline" ]] || { echo "Missing crop-only baseline: $baseline" >&2; exit 1; }
[[ -f runs/vlm/locany_v5_visual_epoch1/config.json ]] || {
  echo 'Missing V5-visual trained model' >&2; exit 1;
}
[[ ! -e "$named" ]] || { echo "Output already exists: $named" >&2; exit 1; }
mkdir -p "$output"

python scripts/preview_locany_v5_visual_val.py \
  --model runs/vlm/locany_v5_visual_epoch1 \
  --manifest datasets/derived/locany_v5_visual/manifest.json \
  --split datasets/weld-dataset/kfold_high/fold0/val \
  --output "$output/named_class" --limit 100 --name-reference-class \
  --generation-mode slow --temperature 0.7 --top-p 0.9 \
  --repetition-penalty 1.1 --max-new-tokens 512 --seed 42 \
  2>&1 | tee "$output/named_class.log"

python - "$baseline" "$named" <<'PY'
import json
import sys
from pathlib import Path

old = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding='utf-8').splitlines() if line]
new = [json.loads(line) for line in Path(sys.argv[2]).read_text(encoding='utf-8').splitlines() if line]
if len(old) != 100 or len(new) != 100:
    raise SystemExit(f'Expected 100 images in each group, found {len(old)} and {len(new)}')
for left, right in zip(old, new, strict=True):
    if (left['image'] != right['image'] or left['references'] != right['references']
            or left['generation'] != right['generation']):
        raise SystemExit(f'Inputs differ for {left["image"]}; not a text-only A/B')
print('Confirmed: identical images, reference crops, and generation settings.')
PY

python scripts/evaluate_locany_predictions.py \
  "$baseline" "$named" \
  --iou 0.5 --include-class-agnostic --output "$output/scores.csv" \
  2>&1 | tee "$output/scores.txt"
