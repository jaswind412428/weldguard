#!/usr/bin/env bash
# Separate, optional training run after inspecting the validation pilot.
# Hyperparameters match V8-N. P11 + external references are the experimental changes.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
source "$repo/.venv-vlm/bin/activate"
cd "$repo"
python scripts/prepare_locany_reference_alignment.py
data="$repo/datasets/derived/locany_reference_alignment"
output="$repo/runs/vlm/locany_reference_aligned_p11_epoch1"
if [[ -e "$output" ]]; then
  echo "Output already exists; keep this experiment immutable: $output" >&2
  exit 1
fi
cd "$repo/eagle-vlm/Embodied"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python "$repo/scripts/make_locany_v7_eval_trainer.py" --show-hash
# Reuse the existing TWO-image check; this is not V8-G training data.
for split in train val; do
  python "$repo/scripts/preflight_locany_v7.py" --variant v8-g \
    --recipe "$data/${split}_recipe.json" --max-tokens 2048
done
mkdir -p "$output"
cp "$data/manifest.json" "$output/data_manifest.json"
cp "$repo/prompts/locany_p11_class_reference.txt" "$output/training_prompt.txt"
export WELDGUARD_V7_VAL_RECIPE="$data/val_recipe.json"
LAUNCHER=pytorch python -m torch.distributed.run --nproc_per_node=1 \
  eaglevl/train/locany_v7_eval_stream.py \
  --model_name_or_path nvidia/LocateAnything-3B \
  --meta_path "$data/train_recipe.json" --output_dir "$output" --overwrite_output_dir False \
  --max_steps 2612 --num_train_epochs 1 \
  --block_size 4 --causal_attn False --attn_implementation sdpa \
  --freeze_llm True --freeze_backbone True --freeze_mlp False \
  --use_llm_lora 8 --use_backbone_lora 0 --vision_select_layer -1 \
  --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 --lr_scheduler_type linear --warmup_steps 0 --warmup_ratio 0 \
  --weight_decay 0 --max_grad_norm 1 --optim adamw_torch_fused \
  --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-8 \
  --save_strategy steps --save_steps 400 --save_total_limit 3 \
  --eval_strategy steps --eval_steps 400 --do_eval True \
  --load_best_model_at_end True --metric_for_best_model eval_loss --greater_is_better False \
  --prediction_loss_only True --eval_accumulation_steps 1 --logging_steps 10 \
  --remove_unused_columns False --bf16 True --fp16 False --seed 42 \
  --dataloader_num_workers 0 --group_by_length False --gradient_checkpointing False \
  --grad_checkpoint False --max_seq_length 2048 --max_num_tokens_per_sample 2048 \
  --max_num_tokens 2048 --packing_buffer_size 4 --do_train True \
  --deepspeed deepspeed_configs/zero_stage2_config.json --report_to none \
  --run_name Reference_aligned_P11_epoch1 2>&1 | tee "$output/train.log"
cd "$repo"
python scripts/export_v3_training_metadata.py --run "$output" --output "$output/training_metadata.json"
python scripts/summarize_locany_v7_loss.py --run "$output"
# Use exactly the same held-out validation selection; no test evaluation here.
python scripts/preview_locany_reference_alignment.py --variant aligned --mode visual \
  --prompt P11 --reference-bank context --limit 20 \
  --output "$repo/runs/vlm/reference_alignment_val20/aligned_visual_P11_context"
pilot="$repo/runs/vlm/reference_alignment_val20"
inputs=("$pilot"/*/predictions.jsonl)
python scripts/analyze_locany_reference_errors.py "${inputs[@]}" --output "$pilot/analysis"
python scripts/pack_locany_reference_pilot.py --run "$pilot"
