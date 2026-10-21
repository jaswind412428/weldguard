"""Measure defect localization without requiring the predicted defect class to match.

Each predicted box can match at most one GT box at the chosen IoU threshold.
The matching is maximum-cardinality; valid edges prefer higher IoU, then a
matching class. This is a detection diagnostic, not mAP or class accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluate_locany_predictions import count_matches, iou, scores


def locate_matches(truth: list[dict], predictions: list[dict], threshold: float) -> dict[int, int]:
    edges = []
    for pred in predictions:
        candidates = [
            (index, iou(pred["box_01"], gt["box_01"]), pred["class"] == gt["class"])
            for index, gt in enumerate(truth)
        ]
        edges.append([
            index for index, overlap, same_class in sorted(
                (item for item in candidates if item[1] >= threshold),
                key=lambda item: (-item[1], -int(item[2]), item[0]),
            )
        ])

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

    for pred_index in range(len(predictions)):
        assign(pred_index, set())
    return claimed_by


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 < args.iou <= 1:
        parser.error("--iou must be in (0, 1]")

    expected_images = None
    expected_truth = None
    rows = []
    for source in args.inputs:
        if not source.is_file():
            parser.error(f"Missing input: {source}")
        records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records:
            parser.error(f"Empty input: {source}")
        by_image = {Path(record["image"]).name: record for record in records}
        if len(by_image) != len(records):
            parser.error(f"Duplicate image in {source}")
        truths = {name: record["ground_truth"] for name, record in by_image.items()}
        if expected_images is None:
            expected_images, expected_truth = set(by_image), truths
        elif set(by_image) != expected_images or truths != expected_truth:
            parser.error(f"Image set or GT differs: {source}")

        total_gt = total_pred = localized = matched_wrong_class = class_aware = 0
        for record in records:
            truth, predictions = record["ground_truth"], record["predictions"]
            pairs = locate_matches(truth, predictions, args.iou)
            total_gt += len(truth)
            total_pred += len(predictions)
            localized += len(pairs)
            matched_wrong_class += sum(
                truth[gt_index]["class"] != predictions[pred_index]["class"]
                for gt_index, pred_index in pairs.items()
            )
            class_aware += count_matches(truth, predictions, args.iou)[0]

        row = {
            "model": records[0].get("model", source.parent.name),
            "source": str(source), "images": len(records), "iou": args.iou,
            "ground_truth_boxes": total_gt, "predicted_boxes": total_pred,
            **scores(localized, total_pred - localized, total_gt - localized),
            "matched_wrong_class": matched_wrong_class,
            "class_aware_tp": class_aware,
        }
        rows.append(row)
        print(f"\n{source.parent.name} ({len(records)} images, class-agnostic IoU >= {args.iou:g})")
        print(f"GT={total_gt} predicted={total_pred} localized TP={row['tp']} "
              f"FP={row['fp']} FN={row['fn']}")
        print(f"Localization P={row['precision']:.3f} R={row['recall']:.3f} "
              f"F1={row['f1']:.3f}")
        print(f"Among localized matches, wrong class={matched_wrong_class}; "
              f"strict class-aware TP={class_aware}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
