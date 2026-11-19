#!/usr/bin/env bash
# Continue V8-N2 on whole images, then compare both input modes on the test split.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
source "$repo/.venv-vlm/bin/activate"
cd "$repo"

base="$repo/runs/vlm/locany_reference_aligned_p10_original_epoch1"
data="$repo/datasets/derived/locany_v8_n2_whole"
output="$repo/runs/vlm/locany_v8_n2_whole_p10"
test_run="$repo/runs/vlm/v8_n2_whole_p10_test"
prompt="$repo/prompts/locany_p10_task_format_features.json"

if [[ ! -f "$output/done.txt" ]]; then
  for name in done.txt config.json model.safetensors.index.json; do
    [[ -f "$base/$name" ]] || { echo "Missing V8-N2 starting checkpoint file: $base/$name" >&2; exit 1; }
  done
  python scripts/prepare_locany_v8_n2_whole.py
  if [[ -e "$output" ]] && ! cmp -s "$data/manifest.json" "$output/data_manifest.json"; then
    echo "Existing V8-N2 run has different training data: $output" >&2
    exit 1
  fi
  cd "$repo/eagle-vlm/Embodied"
  export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
  python "$repo/scripts/make_locany_v7_eval_trainer.py" --continue-llm-lora --show-hash
  for split in train val; do
    python "$repo/scripts/preflight_locany_v7.py" --variant v8-n2-whole --model "$base" \
      --recipe "$data/${split}_recipe.json" --max-tokens 2048
  done
  mkdir -p "$output"
  cp "$data/manifest.json" "$output/data_manifest.json"
  cp "$prompt" "$output/training_prompt_source.txt"
  export WELDGUARD_V7_VAL_RECIPE="$data/val_recipe.json"
  LAUNCHER=pytorch python -m torch.distributed.run --nproc_per_node=1 \
    eaglevl/train/locany_v8_n2_whole_eval_stream.py \
    --model_name_or_path "$base" --meta_path "$data/train_recipe.json" \
    --output_dir "$output" --overwrite_output_dir False --max_steps 1306 --num_train_epochs 1 \
    --block_size 4 --causal_attn False --attn_implementation sdpa \
    --freeze_llm True --freeze_backbone True --freeze_mlp False \
    --use_llm_lora 8 --use_backbone_lora 0 --vision_select_layer -1 \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 --lr_scheduler_type linear --warmup_steps 0 --warmup_ratio 0 \
    --weight_decay 0 --max_grad_norm 1 --optim adamw_torch_fused \
    --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-8 \
    --save_strategy steps --save_steps 200 --save_total_limit 3 \
    --eval_strategy steps --eval_steps 200 --do_eval True \
    --load_best_model_at_end True --metric_for_best_model eval_loss --greater_is_better False \
    --prediction_loss_only True --eval_accumulation_steps 1 --logging_steps 10 \
    --remove_unused_columns False --bf16 True --fp16 False --seed 42 \
    --dataloader_num_workers 0 --group_by_length False --gradient_checkpointing False \
    --grad_checkpoint False --max_seq_length 2048 --max_num_tokens_per_sample 2048 \
    --max_num_tokens 2048 --packing_buffer_size 4 --do_train True \
    --deepspeed deepspeed_configs/zero_stage2_config.json --report_to none \
    --run_name V8_N2_whole_image_P10 2>&1 | tee -a "$output/train.log"
  cd "$repo"
fi

python scripts/export_v3_training_metadata.py --run "$output" --output "$output/training_metadata.json"
python scripts/summarize_locany_v7_loss.py --run "$output"
python - "$output" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
state = json.loads((run / "trainer_state.json").read_text(encoding="utf-8"))
loss = json.loads((run / "loss_summary.json").read_text(encoding="utf-8"))
if state.get("global_step") != 1306 or loss.get("best_metric_eval_loss") is None:
    raise SystemExit("Complete the 1,306 training steps and validation before testing")
PY

for mode in none gt; do
  python scripts/preview_locany_v7_nine_test.py --variant v8-n2-whole \
    --mode "$mode" --prompt-version P10 --output "$test_run/$mode" \
    --generation-mode slow --temperature 0.7 --top-p 0.9 \
    --repetition-penalty 1.1 --max-new-tokens 512 --max-input-tokens 2048 --seed 42
done
python scripts/evaluate_locany_predictions.py \
  "$test_run/none/predictions.jsonl" "$test_run/gt/predictions.jsonl" \
  --iou 0.5 --output "$test_run/scores.csv" | tee "$test_run/scores.txt"
