#!/usr/bin/env bash
# One final, class-aware evaluation on the held-out 205-image test set.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
source "$repo/.venv-vlm/bin/activate"
cd "$repo"

for variant in v7 v7-g v7-n; do
  run="$repo/runs/vlm/locany_${variant//-/_}_p04_xy"
  [[ -f "$run/done.txt" && -f "$run/loss_summary.json" ]] || {
    echo "Incomplete training run: $run" >&2
    exit 1
  }
  python - "$run" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
state = json.loads((run / "trainer_state.json").read_text(encoding="utf-8"))
loss = json.loads((run / "loss_summary.json").read_text(encoding="utf-8"))
expected = 2612 if "_v7_n_" in run.name else 653
if state.get("global_step") != expected or loss.get("best_metric_eval_loss") is None:
    raise SystemExit(f"Incomplete validation or training steps: {run}: {state.get('global_step')} / {expected}")
print(f"{run.name}: steps={expected}, best val loss={loss['best_metric_eval_loss']:.4f}, "
      f"checkpoint={loss['best_model_checkpoint']}")
PY
done

if [[ ! -f datasets/derived/locany_v7/gt_references/train_0000.jpg ]]; then
  python scripts/prepare_locany_v7_family.py --variant v7-g
fi
if [[ ! -f datasets/derived/locany_v5_visual/manifest.json ]]; then
  python scripts/prepare_locany_v5_visual.py
fi

output="$repo/runs/vlm/v7_nine_test"
mkdir -p "$output"
for variant in v7 v7-g v7-n; do
  for mode in none gt visual; do
    python scripts/preview_locany_v7_nine_test.py \
      --variant "$variant" --mode "$mode" \
      --output "$output/${variant}_${mode}" \
      --generation-mode slow --temperature 0.7 --top-p 0.9 \
      --repetition-penalty 1.1 --max-new-tokens 512 \
      --max-input-tokens 2048 --seed 42
  done
done

inputs=()
for variant in v7 v7-g v7-n; do
  for mode in none gt visual; do
    inputs+=("$output/${variant}_${mode}/predictions.jsonl")
  done
done
python scripts/evaluate_locany_predictions.py "${inputs[@]}" \
  --iou 0.5 --output "$output/scores.csv" 2>&1 | tee "$output/scores.txt"
python scripts/pack_locany_v7_nine_test.py --run "$output" \
  --output "$repo/results/v7_nine_test_report.zip"
