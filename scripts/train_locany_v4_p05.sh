#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
source "$repo/.venv-vlm/bin/activate"
cd "$repo"
output="$repo/runs/vlm/locany_v4_p05_epoch1"
[[ ! -e "$output" ]] || { echo "Output already exists: $output"; exit 1; }
python scripts/prepare_locany_v4_p05.py
mkdir -p "$output"
cp results/v2_training_metadata.json "$output/v2_reference_parameters.json"
cp prompts/locany_p05_gt_features.txt "$output/training_prompt.txt"
cd "$repo/eagle-vlm/Embodied"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
LAUNCHER=pytorch python -m torch.distributed.run --nproc_per_node=1 \
  eaglevl/train/locany_finetune_magi_stream.py \
  --model_name_or_path nvidia/LocateAnything-3B \
  --meta_path "$repo/datasets/derived/locany_v4_p05/full_recipe.json" \
  --output_dir "$output" --overwrite_output_dir False \
  --max_steps 653 --num_train_epochs 1 \
  --block_size 4 --causal_attn False --attn_implementation sdpa \
  --freeze_llm True --freeze_backbone True --freeze_mlp False \
  --use_llm_lora 8 --use_backbone_lora 0 --vision_select_layer -1 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 --lr_scheduler_type linear \
  --warmup_steps 0 --warmup_ratio 0 --weight_decay 0 --max_grad_norm 1 \
  --optim adamw_torch_fused --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-8 \
  --save_strategy steps --save_steps 100 --save_total_limit 3 --logging_steps 10 \
  --bf16 True --fp16 False --seed 42 --dataloader_num_workers 0 \
  --group_by_length False --gradient_checkpointing False --grad_checkpoint False \
  --max_seq_length 2048 --max_num_tokens_per_sample 2048 --max_num_tokens 2048 \
  --packing_buffer_size 4 --do_train True \
  --deepspeed deepspeed_configs/zero_stage2_config.json \
  --report_to none --run_name V4_P05_epoch1 2>&1 | tee "$output/train.log"
cd "$repo"
python scripts/export_v3_training_metadata.py --run "$output" --output "$output/training_metadata.json"
