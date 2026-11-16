"""Export V7 train/validation loss by step without model weights or images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    state_path = args.run / "trainer_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    losses = {}
    for entry in state.get("log_history", []):
        step = entry.get("step")
        if step is None:
            continue
        row = losses.setdefault(int(step), {"step": int(step), "train_loss": "", "eval_loss": ""})
        if "loss" in entry:
            row["train_loss"] = entry["loss"]
        if "eval_loss" in entry:
            row["eval_loss"] = entry["eval_loss"]
    if not any(row["eval_loss"] != "" for row in losses.values()):
        raise ValueError(f"No eval_loss found in {state_path}; validation was not recorded")
    output = args.run / "loss_curve.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("step", "train_loss", "eval_loss"))
        writer.writeheader()
        writer.writerows(losses[step] for step in sorted(losses))
    report = {"run": args.run.name, "global_step": state.get("global_step"),
              "best_metric_eval_loss": state.get("best_metric"),
              "best_model_checkpoint": state.get("best_model_checkpoint"),
              "validation_loss_steps": [step for step, row in sorted(losses.items()) if row["eval_loss"] != ""],
              "test_used": False}
    (args.run / "loss_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}")
    print(f"Best validation loss: {report['best_metric_eval_loss']} at {report['best_model_checkpoint']}")


if __name__ == "__main__":
    main()
