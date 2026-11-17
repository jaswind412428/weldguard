#!/usr/bin/env bash
# Optional: compare retained checkpoints by actual class-aware validation F1.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
source "$repo/.venv-vlm/bin/activate"
cd "$repo"
limit="${LIMIT:-20}"
case "$limit" in 20|164) ;; *) echo 'LIMIT must be 20 or 164' >&2; exit 2;; esac
profile="${1:-p10-original}"
case "$profile" in
  p10-original) model="$repo/runs/vlm/locany_reference_aligned_p10_original_epoch1"; variant=aligned-p10; prompt=P10; bank=original;;
  p11-context) model="$repo/runs/vlm/locany_reference_aligned_p11_epoch1"; variant=aligned; prompt=P11; bank=context;;
  *) echo 'Choose p10-original or p11-context' >&2; exit 2;;
esac
[[ -f "$model/done.txt" ]] || { echo 'Finish aligned training first' >&2; exit 1; }
run="$repo/runs/vlm/reference_checkpoints_${profile}_val${limit}"
inputs=()
shopt -s nullglob
checkpoints=("$model"/checkpoint-*)
for checkpoint in "${checkpoints[@]}"; do
  [[ -f "$checkpoint/config.json" ]] || continue
  name="$(basename "$checkpoint")"
  python scripts/preview_locany_reference_alignment.py --variant "$variant" --mode visual \
    --prompt "$prompt" --reference-bank "$bank" --limit "$limit" --checkpoint "$name" \
    --output "$run/$name"
  inputs+=("$run/$name/predictions.jsonl")
done
[[ ${#inputs[@]} -gt 0 ]] || { echo 'No retained checkpoints found' >&2; exit 1; }
python scripts/analyze_locany_reference_errors.py "${inputs[@]}" --output "$run/analysis"
python scripts/pack_locany_reference_pilot.py --run "$run"
