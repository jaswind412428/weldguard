#!/usr/bin/env bash
# Class-aware evaluation of three V7 weights and three inference input modes.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
prompt_version="${1:-P04}"
if [[ $# -gt 1 || ( "$prompt_version" != P04 && "$prompt_version" != P09 && "$prompt_version" != P10 ) ]]; then
  echo "Usage: bash scripts/evaluate_locany_v7_nine_test.sh [P04|P09|P10]" >&2
  exit 2
fi
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

if [[ "$prompt_version" == P04 ]]; then
  output="$repo/runs/vlm/v7_nine_test"
  report="$repo/results/v7_nine_test_report.zip"
elif [[ "$prompt_version" == P09 ]]; then
  output="$repo/runs/vlm/v7_p09_nine_test"
  report="$repo/results/v7_p09_nine_test_report.zip"
else
  output="$repo/runs/vlm/v7_p10_nine_test"
  report="$repo/results/v7_p10_nine_test_report.zip"
fi
mkdir -p "$output"
for variant in v7 v7-g v7-n; do
  for mode in none gt visual; do
    python scripts/preview_locany_v7_nine_test.py \
      --variant "$variant" --mode "$mode" --prompt-version "$prompt_version" \
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
python scripts/summarize_locany_v7_nine_counts.py --run "$output" \
  --prompt-version "$prompt_version" --output "$output/counts.csv" \
  2>&1 | tee "$output/counts.txt"
if [[ "$prompt_version" != P04 && -f "$repo/runs/vlm/v7_nine_test/scores.csv" ]]; then
  prompt_lower="${prompt_version,,}"
  python scripts/compare_locany_v7_prompt_scores.py \
    --p04 "$repo/runs/vlm/v7_nine_test/scores.csv" \
    --other "$output/scores.csv" --other-version "$prompt_version" \
    --output "$output/p04_vs_${prompt_lower}.csv"
fi
python scripts/pack_locany_v7_nine_test.py --run "$output" \
  --prompt-version "$prompt_version" --output "$report"
