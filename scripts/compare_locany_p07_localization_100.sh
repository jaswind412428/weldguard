#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv-vlm/bin/activate

output='runs/vlm/prompt_p07_localization_original_100'
[[ ! -e "$output/predictions.jsonl" ]] || { echo "Output already exists: $output"; exit 1; }
mkdir -p "$output"

# Reparse baseline raw text with the same class-independent geometry rules.
python - <<'PY'
import json
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from preview_locany_val import parse_predictions
sources = {
    'P03': Path('runs/vlm/prompt_four_100_t1_size/predictions.jsonl'),
    'P05': Path('runs/vlm/prompt_p05_gt_features_size_original_100/predictions.jsonl'),
}
if not sources['P05'].is_file():
    sources['P05'] = Path('runs/vlm/prompt_v6_native_size_original_100/predictions.jsonl')
for label, source in sources.items():
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line]
    for row in rows:
        row['predictions'], row['rejected_parsed_boxes'] = parse_predictions(row['answer'], True)
        row['task'] = 'localization'
        row['model'] += f' [{label}, reparsed for localization]'
    target = Path('runs/vlm/prompt_p07_localization_original_100') / f'{label}_localization_baseline.jsonl'
    target.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
PY

python scripts/preview_locany_val.py \
  --model nvidia/LocateAnything-3B \
  --prompt-file prompts/locany_p07_local_bounds.txt \
  --include-image-size --localization-only \
  --split datasets/weld-dataset/kfold_high/fold0/val --limit 100 \
  --generation-mode slow --temperature 0.7 --top-p 0.9 \
  --repetition-penalty 1.1 --max-new-tokens 512 --seed 42 \
  --output "$output" 2>&1 | tee "$output/run.log"

python scripts/evaluate_locany_predictions.py \
  "$output/P03_localization_baseline.jsonl" \
  "$output/P05_localization_baseline.jsonl" \
  "$output/predictions.jsonl" \
  --localization-only --iou 0.5 \
  --output "$output/scores.csv" 2>&1 | tee "$output/scores.txt"

python - "$output/predictions.jsonl" <<'PY'
import json
import sys
from pathlib import Path
rows = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding='utf-8').splitlines() if line]
boxes = [box for row in rows for box in row['predictions']]
large = sum((b['box_01'][2]-b['box_01'][0])*(b['box_01'][3]-b['box_01'][1]) >= .8 for b in boxes)
print('P07 accepted boxes:', len(boxes))
print('Empty images:', sum(not row['predictions'] for row in rows))
print('Rejected numeric boxes:', sum(row['rejected_parsed_boxes'] for row in rows))
print('Boxes covering at least 80% of the image:', large)
PY
