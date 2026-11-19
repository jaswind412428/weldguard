#!/usr/bin/env bash
# Compare matched V7-G continuations with the original and doubled pore sampling weight.
set -euo pipefail

phase="${1:-train}"
case "$phase" in
  train|test) ;;
  *) echo 'Usage: bash scripts/train_locany_v7_g_pore_balance.sh [train|test]' >&2; exit 2;;
esac
repo="$(cd "$(dirname "$0")/.." && pwd)"
source "$repo/.venv-vlm/bin/activate"
cd "$repo"
base="$repo/runs/vlm/locany_v7_g_p04_xy"
source_data="$repo/datasets/derived/locany_v7/v7-g"
data="$repo/datasets/derived/locany_v7_g_pore_balance"
run="$repo/runs/vlm/v7_g_pore_balance"
steps=653

for name in config.json model.safetensors.index.json done.txt; do
  [[ -f "$base/$name" ]] || { echo "Missing V7-G starting checkpoint file: $base/$name" >&2; exit 1; }
done

if [[ "$phase" == train ]]; then
  if [[ ! -f "$source_data/train_recipe.json" || ! -f "$source_data/val_recipe.json" ]]; then
    python scripts/prepare_locany_v7_family.py --variant v7-g
  fi
  python scripts/prepare_locany_v7_g_pore_balance.py
  cd "$repo/eagle-vlm/Embodied"
  export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
  python "$repo/scripts/make_locany_v7_eval_trainer.py" --continue-llm-lora v7-g --show-hash
  export WELDGUARD_V7_VAL_RECIPE="$source_data/val_recipe.json"
  python "$repo/scripts/preflight_locany_v7.py" --variant v7-g --model "$base" \
    --recipe "$WELDGUARD_V7_VAL_RECIPE" --max-tokens 2048
  for arm in control pore2x; do
    output="$run/$arm"
    if [[ -e "$output" ]] && ! cmp -s "$data/$arm/manifest.json" "$output/data_manifest.json"; then
      echo "Existing V7-G continuation has different data: $output" >&2
      exit 1
    fi
    if [[ ! -f "$output/done.txt" ]]; then
      python "$repo/scripts/preflight_locany_v7.py" --variant v7-g --model "$base" \
        --recipe "$data/$arm/train_recipe.json" --max-tokens 2048
      mkdir -p "$output"
      cp "$data/$arm/manifest.json" "$output/data_manifest.json"
      LAUNCHER=pytorch python -m torch.distributed.run --nproc_per_node=1 \
        eaglevl/train/locany_v7_g_pore_eval_stream.py \
        --model_name_or_path "$base" --meta_path "$data/$arm/train_recipe.json" \
        --output_dir "$output" --overwrite_output_dir False --max_steps "$steps" --num_train_epochs 1 \
        --block_size 4 --causal_attn False --attn_implementation sdpa \
        --freeze_llm True --freeze_backbone True --freeze_mlp False \
        --use_llm_lora 8 --use_backbone_lora 0 --vision_select_layer -1 \
        --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 1 \
        --learning_rate 2e-5 --lr_scheduler_type linear --warmup_steps 0 --warmup_ratio 0 \
        --weight_decay 0 --max_grad_norm 1 --optim adamw_torch_fused \
        --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-8 \
        --save_strategy steps --save_steps 100 --save_total_limit 3 \
        --eval_strategy steps --eval_steps 100 --do_eval True \
        --load_best_model_at_end True --metric_for_best_model eval_loss --greater_is_better False \
        --prediction_loss_only True --eval_accumulation_steps 1 --logging_steps 10 \
        --remove_unused_columns False --bf16 True --fp16 False --seed 42 \
        --dataloader_num_workers 0 --group_by_length False --gradient_checkpointing False \
        --grad_checkpoint False --max_seq_length 2048 --max_num_tokens_per_sample 2048 \
        --max_num_tokens 2048 --packing_buffer_size 4 --do_train True \
        --deepspeed deepspeed_configs/zero_stage2_config.json --report_to none \
        --run_name "V7_G_P04_$arm" 2>&1 | tee -a "$output/train.log"
    fi
    python "$repo/scripts/export_v3_training_metadata.py" --run "$output" --output "$output/training_metadata.json"
    python "$repo/scripts/summarize_locany_v7_loss.py" --run "$output"
  done
  cd "$repo"
fi

python - "$steps" "$run/control" "$run/pore2x" <<'PY'
import json
import sys
from pathlib import Path

steps, *outputs = sys.argv[1:]
for output in map(Path, outputs):
    if not (output / "done.txt").is_file() or not (output / "loss_summary.json").is_file():
        raise SystemExit(f"Finish training before scoring: {output}")
    loss = json.loads((output / "loss_summary.json").read_text(encoding="utf-8"))
    if loss["global_step"] != int(steps) or loss["best_metric_eval_loss"] is None:
        raise SystemExit(f"Training or checkpoint selection is incomplete: {output}")
PY

split=val
limit=164
scores="$run/val164"
if [[ "$phase" == test ]]; then
  split=test
  limit=205
  scores="$run/test205"
fi
inputs=()
for arm in original control pore2x; do
  variant="v7-g-$arm"
  [[ "$arm" != original ]] || variant=v7-g
  predictions="$scores/$arm/predictions.jsonl"
  python scripts/preview_locany_v7_nine_test.py --variant "$variant" --split "$split" \
    --mode gt --prompt-version P04 --generation-mode slow --temperature 0.7 --top-p 0.9 \
    --repetition-penalty 1.1 --max-new-tokens 512 --max-input-tokens 2048 --seed 42 \
    --output "$scores/$arm"
  [[ "$(wc -l < "$predictions")" -eq "$limit" ]] || { echo "Incomplete predictions: $predictions" >&2; exit 1; }
  inputs+=("$predictions")
done
python scripts/evaluate_locany_predictions.py "${inputs[@]}" \
  --iou 0.5 --output "$scores/scores.csv" | tee "$scores/scores.txt"
