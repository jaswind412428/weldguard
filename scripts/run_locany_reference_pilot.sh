#!/usr/bin/env bash
# Default: six controlled conditions on the SAME fixed 20 validation images.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
source "$repo/.venv-vlm/bin/activate"
cd "$repo"
limit="${LIMIT:-20}"
case "$limit" in 20|164) ;; *) echo 'LIMIT must be 20 or 164' >&2; exit 2;; esac
run="$repo/runs/vlm/reference_alignment_val${limit}"
mkdir -p "$run"

# Diagnose the old nine TEST outputs without re-running or selecting new test settings.
old_inputs=()
for variant in v8 v8-g v8-n; do
  for mode in none gt visual; do
    path="$repo/runs/vlm/v8_p10_nine_test/${variant}_${mode}/predictions.jsonl"
    if [[ -f "$path" ]]; then old_inputs+=("$path"); fi
  done
done
if [[ ${#old_inputs[@]} -eq 9 ]]; then
  python scripts/analyze_locany_reference_errors.py "${old_inputs[@]}" \
    --output "$run/old_test_error_analysis" | tee "$run/old_test_error_analysis.log"
fi
python scripts/prepare_locany_reference_alignment.py | tee "$run/preparation.log"

conditions=(
  'v8 none P10 original'
  'v8-g gt P10 original'
  'v8-g gt P10 context'
  'v8-n visual P10 original'
  'v8-n visual P11 original'
  'v8-n visual P11 context'
)
if [[ "${ALL_WEIGHTS:-0}" == 1 ]]; then
  conditions=()
  for variant in v8 v8-g v8-n; do
    conditions+=("$variant none P10 original" "$variant gt P10 original"
      "$variant gt P10 context" "$variant visual P10 original"
      "$variant visual P11 original" "$variant visual P11 context")
  done
fi
inputs=()
for condition in "${conditions[@]}"; do
  read -r variant mode prompt bank <<< "$condition"
  output="$run/${variant}_${mode}_${prompt}_${bank}"
  python scripts/preview_locany_reference_alignment.py --variant "$variant" \
    --mode "$mode" --prompt "$prompt" --reference-bank "$bank" \
    --limit "$limit" --output "$output" 2>&1 | tee "$run/${variant}_${mode}_${prompt}_${bank}.log"
  inputs+=("$output/predictions.jsonl")
done
python scripts/analyze_locany_reference_errors.py "${inputs[@]}" \
  --output "$run/analysis" | tee "$run/analysis.log"
python scripts/pack_locany_reference_pilot.py --run "$run"
