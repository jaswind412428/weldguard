"""Count accepted boxes and empty outputs for a nine-way V7/V8 inference run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODES = ("none", "gt", "visual")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--family", choices=("v7", "v8"), default="v7")
    parser.add_argument("--prompt-version", choices=("P04", "P09", "P10"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = []
    for variant in (args.family, f"{args.family}-g", f"{args.family}-n"):
        for mode in MODES:
            source = args.run / f"{variant}_{mode}" / "predictions.jsonl"
            if not source.is_file():
                parser.error(f"Missing predictions: {source}")
            rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
            label = f"{variant.upper()} / {mode} / {args.prompt_version}"
            if len(rows) != 205 or any(row.get("model") != label for row in rows):
                parser.error(f"Expected 205 {label} records: {source}")
            accepted = sum(len(row["predictions"]) for row in rows)
            rejected = sum(row["rejected_parsed_boxes"] for row in rows)
            zero = sum(not row["predictions"] for row in rows)
            summary.append({"model": label, "images": len(rows), "queries": len(rows) * (4 if mode == "visual" else 1),
                            "accepted_boxes": accepted, "rejected_numeric_boxes": rejected,
                            "images_zero_accepted": zero})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(f"{row['model']}: accepted={row['accepted_boxes']} "
              f"rejected_numeric={row['rejected_numeric_boxes']} "
              f"images_zero_accepted={row['images_zero_accepted']} "
              f"queries={row['queries']}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
