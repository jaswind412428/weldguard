#!/usr/bin/env bash
# Evaluate V8/V8-G/V8-N with P10 and three input modes on held-out test.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
source "$repo/.venv-vlm/bin/activate"
cd "$repo"

for variant in v8 v8-g v8-n; do
  run="$repo/runs/vlm/locany_${variant//-/_}_p10_xy"
  [[ -f "$run/done.txt" && -f "$run/loss_summary.json" ]] || {
    echo "Incomplete training run: $run" >&2
    exit 1
  }
  python - "$run" "$variant" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
variant = sys.argv[2]
repo = Path.cwd()
state = json.loads((run / "trainer_state.json").read_text(encoding="utf-8"))
loss = json.loads((run / "loss_summary.json").read_text(encoding="utf-8"))
manifest = json.loads((run / "data_manifest.json").read_text(encoding="utf-8"))
prompt_hash = hashlib.sha256(
    (repo / "prompts/locany_p10_task_format_features.json").read_bytes()).hexdigest()
expected = 2612 if variant == "v8-n" else 653
if state.get("global_step") != expected or loss.get("best_metric_eval_loss") is None:
    raise SystemExit(f"Incomplete validation or training steps: {run}: "
                     f"{state.get('global_step')} / {expected}")
if manifest.get("prompt_sha256") != prompt_hash:
    raise SystemExit(f"P10 changed since {variant} was trained; refuse mixed-prompt comparison")
print(f"{run.name}: steps={expected}, best val loss={loss['best_metric_eval_loss']:.4f}, "
      f"checkpoint={loss['best_model_checkpoint']}")
PY
done

if [[ ! -f datasets/derived/locany_v8/gt_references/train_0000.jpg ]]; then
  python scripts/prepare_locany_v8_family.py --variant v8-g
fi
if [[ ! -f datasets/derived/locany_v5_visual/manifest.json ]]; then
  python scripts/prepare_locany_v5_visual.py
fi

output="$repo/runs/vlm/v8_p10_nine_test"
report="$repo/results/v8_p10_nine_test_report.zip"
mkdir -p "$output"
for variant in v8 v8-g v8-n; do
  for mode in none gt visual; do
    python scripts/preview_locany_v7_nine_test.py \
      --variant "$variant" --mode "$mode" --prompt-version P10 \
      --output "$output/${variant}_${mode}" \
      --generation-mode slow --temperature 0.7 --top-p 0.9 \
      --repetition-penalty 1.1 --max-new-tokens 512 \
      --max-input-tokens 2048 --seed 42
  done
done

inputs=()
for variant in v8 v8-g v8-n; do
  for mode in none gt visual; do
    inputs+=("$output/${variant}_${mode}/predictions.jsonl")
  done
done
python scripts/evaluate_locany_predictions.py "${inputs[@]}" \
  --iou 0.5 --output "$output/scores.csv" 2>&1 | tee "$output/scores.txt"
python scripts/summarize_locany_v7_nine_counts.py --family v8 --run "$output" \
  --prompt-version P10 --output "$output/counts.csv" \
  2>&1 | tee "$output/counts.txt"
python scripts/pack_locany_v7_nine_test.py --family v8 --run "$output" \
  --prompt-version P10 --output "$report"
