#!/usr/bin/env bash
# Train two matched 50-step pilots from the same original LocateAnything model.
# Run prepare_locany_50step_pair.sh first. Never resume into these output dirs.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
source "$repo/.venv-vlm/bin/activate"
cd "$repo/eagle-vlm/Embodied"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

train_one() {
  local label="$1"
  local recipe="$2"
  local output="$3"
  if [[ ! -f "$recipe" ]]; then
    echo "Missing training recipe: $recipe" >&2
    exit 1
  fi
  if [[ -e "$output" ]]; then
    echo "Output already exists; refusing to resume or overwrite: $output" >&2
    exit 1
  fi
  mkdir -p "$output"

  echo "Training $label: original model, 50 steps, recipe $recipe" | tee "$output/run.log"
  LAUNCHER=pytorch python -m torch.distributed.run \
    --nproc_per_node=1 \
    eaglevl/train/locany_finetune_magi_stream.py \
    --model_name_or_path nvidia/LocateAnything-3B \
    --meta_path "$recipe" \
    --output_dir "$output" \
    --overwrite_output_dir False \
    --max_steps 50 \
    --num_train_epochs 5 \
    --block_size 4 \
    --causal_attn False \
    --attn_implementation sdpa \
    --freeze_llm True \
    --freeze_backbone True \
    --freeze_mlp False \
    --use_llm_lora 8 \
    --use_backbone_lora 0 \
    --vision_select_layer -1 \
    --dataloader_num_workers 0 \
    --bf16 True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear \
    --warmup_steps 0 \
    --weight_decay 0.01 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --max_seq_length 2048 \
    --max_num_tokens_per_sample 2048 \
    --max_num_tokens 2048 \
    --packing_buffer_size 4 \
    --grad_checkpoint False \
    --group_by_length False \
    --do_train True \
    --deepspeed deepspeed_configs/zero_stage2_config.json \
    --report_to none \
    --seed 42 \
    --run_name "$label" \
    2>&1 | tee -a "$output/run.log"
}

train_one V3A \
  "$repo/datasets/derived/locany_50step_v3a_original/full_recipe.json" \
  "$repo/runs/vlm/locany_50step_v3a"

train_one V4 \
  "$repo/datasets/derived/locany_50step_v4_balanced_size/full_recipe.json" \
  "$repo/runs/vlm/locany_50step_v4"
