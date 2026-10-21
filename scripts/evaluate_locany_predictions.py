"""Evaluate LocateAnything preview JSONL with class-aware IoU matching.

Each prediction can match at most one ground-truth box of the same class.
Reports micro precision, recall and F1 at one IoU threshold; this is not mAP.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CLASSES = ("pore", "deposit", "discontinuity", "stain")


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def count_matches(truth: list[dict], predictions: list[dict], threshold: float,
                  ignore_class: bool = False) -> tuple[int, int, int]:
    # Maximum cardinality bipartite matching, so overlapping predictions cannot
    # claim the same ground-truth box twice.
    edges = [
        [j for j, gt in enumerate(truth)
         if (ignore_class or pred["class"] == gt["class"])
         and iou(pred["box_01"], gt["box_01"]) >= threshold]
        for pred in predictions
    ]
    claimed_by: dict[int, int] = {}

    def assign(pred_index: int, seen: set[int]) -> bool:
        for gt_index in edges[pred_index]:
            if gt_index in seen:
                continue
            seen.add(gt_index)
            if gt_index not in claimed_by or assign(claimed_by[gt_index], seen):
                claimed_by[gt_index] = pred_index
                return True
        return False

    for index in range(len(predictions)):
        assign(index, set())
    tp = len(claimed_by)
    return tp, len(predictions) - tp, len(truth) - tp


def scores(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="One predictions.jsonl per model")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--include-class-agnostic", action="store_true",
                        help="Add one overall localization row that ignores defect class")
    parser.add_argument("--output", type=Path, help="Optional CSV summary path")
    parser.add_argument("--localization-only", action="store_true",
                        help="Report only the class-agnostic localization row")
    args = parser.parse_args()
    if not 0 < args.iou <= 1:
        parser.error("--iou must be in (0, 1]")

    rows: list[dict] = []
    expected_images: set[str] | None = None
    expected_truth: dict[str, list[dict]] | None = None
    for source in args.inputs:
        if not source.is_file():
            parser.error(f"Missing input: {source}")
        records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        images = [Path(record["image"]).name for record in records]
        if len(images) != len(set(images)):
            parser.error(f"Duplicate image in {source}")
        if not records:
            parser.error(f"Empty input: {source}")
        image_set = set(images)
        truth_by_image = {Path(record["image"]).name: record["ground_truth"] for record in records}
        if expected_images is None:
            expected_images = image_set
            expected_truth = truth_by_image
        elif image_set != expected_images:
            parser.error(f"Image sets differ: {source} has {len(image_set)} images, expected {len(expected_images)}")
        elif truth_by_image != expected_truth:
            parser.error(f"Ground-truth boxes differ: {source}")

        model = str(records[0].get("model", source.parent.name))
        generation = records[0].get("generation")
        for record in records:
            if record.get("generation") != generation:
                parser.error(f"Mixed generation settings in {source}")
        print(f"\n{model} ({len(records)} images, IoU >= {args.iou:g})")
        categories = (("all_no_class",) if args.localization_only else
                      (*CLASSES, "all", "all_no_class") if args.include_class_agnostic else (*CLASSES, "all"))
        for category in categories:
            total = [0, 0, 0]
            for record in records:
                overall = category in {"all", "all_no_class"}
                gt = [box for box in record["ground_truth"] if overall or box["class"] == category]
                pred = [box for box in record["predictions"] if overall or box["class"] == category]
                counts = count_matches(gt, pred, args.iou, ignore_class=category == "all_no_class")
                total = [a + b for a, b in zip(total, counts)]
            row = {"model": model, "source": str(source), "images": len(records), "class": category,
                   "iou": args.iou, **scores(*total)}
            rows.append(row)
            print(f"{category:15} TP={row['tp']:4} FP={row['fp']:4} FN={row['fn']:4} "
                  f"P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
