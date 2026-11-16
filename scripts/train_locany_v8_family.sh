#!/usr/bin/env bash
# Train one V8 weight with P10 and held-out validation loss.
set -euo pipefail

variant="${1:-}"
case "$variant" in
  v8)   steps=653;  interval=100; run_name=V8_P10_XY ;;
  v8-g) steps=653;  interval=100; run_name=V8_G_P10_GT_boxed_reference ;;
  v8-n) steps=2612; interval=400; run_name=V8_N_P10_visual_prompt ;;
  *) echo 'Usage: bash scripts/train_locany_v8_family.sh v8|v8-g|v8-n' >&2; exit 2 ;;
esac

repo="$(cd "$(dirname "$0")/.." && pwd)"
source "$repo/.venv-vlm/bin/activate"
cd "$repo"

if [[ ! -f datasets/derived/locany_v6_xy_p04/train_full.jsonl ]]; then
  if [[ ! -f datasets/derived/locany_v5_p04/train_full.jsonl ]]; then
    python scripts/prepare_locany_v5_p04.py
  fi
  python scripts/prepare_locany_v6_xy.py
fi

data="$repo/datasets/derived/locany_v8/$variant"
python scripts/prepare_locany_v8_family.py --variant "$variant"
[[ -f "$data/train_recipe.json" && -f "$data/val_recipe.json" ]] || {
  echo "Missing V8 recipes: $data" >&2; exit 1;
}

output="$repo/runs/vlm/locany_${variant//-/_}_p10_xy"
if [[ -f "$output/done.txt" ]]; then
  echo "Completed run already exists: $output" >&2
  exit 1
fi

cd "$repo/eagle-vlm/Embodied"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# This existing trainer adds a finite validation set; its filename is historical.
python "$repo/scripts/make_locany_v7_eval_trainer.py" --show-hash
python "$repo/scripts/preflight_locany_v7.py" --variant "$variant" \
  --recipe "$data/train_recipe.json" --max-tokens 2048
python "$repo/scripts/preflight_locany_v7.py" --variant "$variant" \
  --recipe "$data/val_recipe.json" --max-tokens 2048

mkdir -p "$output"
if [[ -f "$output/data_manifest.json" ]]; then
  cmp "$data/manifest.json" "$output/data_manifest.json" || {
    echo "Training data or P10 changed during this V8 run: $output" >&2
    exit 1
  }
else
  cp "$data/manifest.json" "$output/data_manifest.json"
fi

export WELDGUARD_V7_VAL_RECIPE="$data/val_recipe.json"
LAUNCHER=pytorch python -m torch.distributed.run --nproc_per_node=1 \
  eaglevl/train/locany_v7_eval_stream.py \
  --model_name_or_path nvidia/LocateAnything-3B \
  --meta_path "$data/train_recipe.json" \
  --output_dir "$output" --overwrite_output_dir False \
  --max_steps "$steps" --num_train_epochs 1 \
  --block_size 4 --causal_attn False --attn_implementation sdpa \
  --freeze_llm True --freeze_backbone True --freeze_mlp False \
  --use_llm_lora 8 --use_backbone_lora 0 --vision_select_layer -1 \
  --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 --lr_scheduler_type linear \
  --warmup_steps 0 --warmup_ratio 0 --weight_decay 0 --max_grad_norm 1 \
  --optim adamw_torch_fused --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-8 \
  --save_strategy steps --save_steps "$interval" --save_total_limit 3 \
  --eval_strategy steps --eval_steps "$interval" --do_eval True \
  --load_best_model_at_end True --metric_for_best_model eval_loss \
  --greater_is_better False --prediction_loss_only True --eval_accumulation_steps 1 \
  --logging_steps 10 --remove_unused_columns False \
  --bf16 True --fp16 False --seed 42 --dataloader_num_workers 0 \
  --group_by_length False --gradient_checkpointing False --grad_checkpoint False \
  --max_seq_length 2048 --max_num_tokens_per_sample 2048 --max_num_tokens 2048 \
  --packing_buffer_size 4 --do_train True \
  --deepspeed deepspeed_configs/zero_stage2_config.json \
  --report_to none --run_name "$run_name" 2>&1 | tee -a "$output/train.log"

cd "$repo"
python scripts/export_v3_training_metadata.py --run "$output" --output "$output/training_metadata.json"
python scripts/summarize_locany_v7_loss.py --run "$output"
