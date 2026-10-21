"""Export a small, allowlisted V3 training-parameter report; never export weights/data."""

from __future__ import annotations

import argparse
import json
import re
from enum import Enum
from pathlib import Path

import torch


TRAINING_FIELDS = (
    "max_steps", "num_train_epochs", "per_device_train_batch_size",
    "gradient_accumulation_steps", "learning_rate", "lr_scheduler_type",
    "warmup_steps", "warmup_ratio", "weight_decay", "max_grad_norm",
    "save_strategy", "save_steps", "save_total_limit", "logging_steps",
    "bf16", "fp16", "seed", "dataloader_num_workers", "group_by_length",
    "gradient_checkpointing", "report_to", "optim", "adam_beta1",
    "adam_beta2", "adam_epsilon", "do_train", "deepspeed",
)
CUSTOM_FIELDS = (
    "model_name_or_path", "meta_path", "block_size", "causal_attn",
    "attn_implementation", "freeze_llm", "freeze_mlp", "freeze_backbone",
    "use_llm_lora", "use_backbone_lora", "vision_select_layer",
    "packing_buffer_size", "max_seq_length", "max_num_tokens_per_sample",
    "max_num_tokens", "grad_checkpoint", "mlp_connector_layers",
)


def plain(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path,
                        default=Path("runs/vlm/locany_fold0_5epoch_equiv"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/v3_training_metadata.json"))
    args = parser.parse_args()
    run = args.run
    training_path = run / "training_args.bin"
    if not training_path.is_file():
        parser.error(f"Missing {training_path}")

    # This is a checkpoint file created by this project's own previous run.
    training = torch.load(training_path, map_location="cpu", weights_only=False)
    standard = {
        key: plain(getattr(training, key))
        for key in TRAINING_FIELDS if hasattr(training, key)
    }
    # DeepSpeed may be a dictionary rather than a path; only its stage matters.
    if isinstance(standard.get("deepspeed"), dict):
        standard["deepspeed"] = {
            "zero_optimization": standard["deepspeed"].get("zero_optimization")
        }

    custom = {}
    log_path = run / "train.log"
    if log_path.is_file():
        line_pattern = re.compile(r"^\s*([a-z][a-z0-9_]*)=(.*?),?\s*$")
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = line_pattern.match(line)
            if match and match.group(1) in CUSTOM_FIELDS:
                custom[match.group(1)] = match.group(2).removesuffix(",")

    state = {}
    state_path = run / "trainer_state.json"
    if state_path.is_file():
        raw_state = json.loads(state_path.read_text(encoding="utf-8"))
        state = {key: raw_state.get(key) for key in ("global_step", "epoch", "max_steps")}

    report = {
        "run_name": run.name,
        "training_args": standard,
        "custom_args_found_in_log": custom,
        "trainer_state": state,
        "missing_custom_args": sorted(set(CUSTOM_FIELDS) - set(custom)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved parameter-only report to {args.output}")
    print(f"Training fields: {len(standard)}; custom fields from log: {len(custom)}")
    print("No model weights, dataset records, or raw log lines were exported.")


if __name__ == "__main__":
    main()
