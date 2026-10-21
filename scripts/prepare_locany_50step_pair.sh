#!/usr/bin/env bash
# Prepare matched 653-image training recipes for the V3A/V4 50-step pilot.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv-vlm/bin/activate

v3a='datasets/derived/locany_50step_v3a_original'
v4='datasets/derived/locany_50step_v4_balanced_size'

python scripts/prepare_locany_pilot.py --all --output "$v3a"
python scripts/prepare_locany_pilot.py --all \
  --prompt-file prompts/locany_weld_balanced_v4d.txt \
  --include-image-size --output "$v4"

python - "$v3a/train_full.jsonl" "$v4/train_full.jsonl" <<'PY'
import json
import sys
from pathlib import Path

left, right = [
    [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]
    for path in sys.argv[1:]
]
if len(left) != len(right):
    raise SystemExit(f'Different record counts: {len(left)} vs {len(right)}')
for index, (original, modified) in enumerate(zip(left, right), 1):
    if original['image'] != modified['image']:
        raise SystemExit(f'Image order differs at record {index}')
    if original['conversations'][1] != modified['conversations'][1]:
        raise SystemExit(f'Ground-truth answer differs at record {index}')
    if original['conversations'][0] == modified['conversations'][0]:
        raise SystemExit(f'Prompt unexpectedly identical at record {index}')
print(f'Matched {len(left)} images and ground-truth answers; only human prompts differ.')
print('V3A recipe:', Path(sys.argv[1]).parent / 'full_recipe.json')
print('V4 recipe:', Path(sys.argv[2]).parent / 'full_recipe.json')
PY
