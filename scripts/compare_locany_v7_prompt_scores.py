"""Make paired P04 versus another prompt's V7 scores for the same nine runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read(path: Path, version: str) -> dict[tuple[str, str, str], dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 45:
        raise ValueError(f"Expected 45 class rows in {path}; found {len(rows)}")
    result = {}
    for row in rows:
        parts = row["model"].split(" / ")
        if len(parts) != 3 or parts[2] != version or row["images"] != "205" or float(row["iou"]) != 0.5:
            raise ValueError(f"Incompatible score row in {path}: {row}")
        key = (parts[0], parts[1], row["class"])
        if key in result:
            raise ValueError(f"Duplicate row in {path}: {key}")
        result[key] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p04", type=Path, required=True)
    parser.add_argument("--other", type=Path, required=True)
    parser.add_argument("--other-version", choices=("P09", "P10"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old = read(args.p04, "P04")
    new = read(args.other, args.other_version)
    if old.keys() != new.keys():
        parser.error("Both prompts must have the same nine modes and five class rows")
    rows = []
    for (variant, mode, category), previous in old.items():
        current = new[(variant, mode, category)]
        rows.append({"variant": variant, "mode": mode, "class": category,
                     "p04_precision": previous["precision"], "other_precision": current["precision"],
                     "p04_recall": previous["recall"], "other_recall": current["recall"],
                     "p04_f1": previous["f1"], "other_f1": current["f1"],
                     "delta_f1": round(float(current["f1"]) - float(previous["f1"]), 6)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved paired P04/{args.other_version} comparison to {args.output}")
    for row in rows:
        if row["class"] == "all":
            print(f"{row['variant']} / {row['mode']}: P04 F1={float(row['p04_f1']):.3f}, "
                  f"{args.other_version} F1={float(row['other_f1']):.3f}, delta={row['delta_f1']:+.3f}")


if __name__ == "__main__":
    main()
