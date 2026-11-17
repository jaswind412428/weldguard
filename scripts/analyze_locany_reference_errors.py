"""Analyze existing class-aware predictions; no GPU or new inference needed."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_locany_predictions import CLASSES, scores, count_matches
from locany_reference_diagnostics import diagnose, parse_answer, prediction_views


def write_csv(path, rows):
    if rows:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def with_queries(record):
    if "queries" in record:
        return record
    answers = record.get("answer", {})
    if isinstance(answers, dict) and set(answers) == set(CLASSES):
        record = dict(record)
        record["queries"] = [{"requested_class": c, "answer": answers[c],
                               "prompt": record.get("prompt", {}).get(c, ""),
                               "predictions": parse_answer(answers[c])[0]} for c in CLASSES]
        reconstructed = [p for q in record["queries"] for p in q["predictions"]]
        if reconstructed != record["predictions"]:
            raise ValueError("Raw query reparse differs from stored predictions; do not silently change parser")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--dedup-iou", type=float, default=0.9)
    args = parser.parse_args()
    if not 0.1 < args.iou <= 1 or not 0 < args.dedup_iou <= 1:
        parser.error("Require 0.1 < match IoU <= 1 and 0 < dedup IoU <= 1")
    args.output.mkdir(parents=True, exist_ok=True)
    score_rows, error_rows, query_rows, confusion_rows, sensitivity = [], [], [], [], []
    expected_truth = None
    details = args.output / "per_image_errors.jsonl"
    with details.open("w", encoding="utf-8") as detail_file:
        for source in args.inputs:
            records = [with_queries(json.loads(line)) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not records:
                raise ValueError(f"Empty input: {source}")
            truth_map = {Path(r["image"]).name: r["ground_truth"] for r in records}
            if len(truth_map) != len(records):
                raise ValueError(f"Duplicate images: {source}")
            if expected_truth is None:
                expected_truth = truth_map
            elif truth_map != expected_truth:
                raise ValueError(f"Different target images or GT: {source}")
            model = records[0]["model"]
            totals = defaultdict(lambda: [0, 0, 0])
            errors, confusions = Counter(), Counter()
            for record in records:
                if record["model"] != model:
                    raise ValueError(f"Mixed model labels: {source}")
                truth = record["ground_truth"]
                raw_diag = diagnose(truth, record["predictions"], args.iou)
                if tuple(raw_diag["scores"][k] for k in ("tp", "fp", "fn")) != count_matches(truth, record["predictions"], args.iou):
                    raise AssertionError("Matching disagrees with existing evaluator")
                for view, predictions in prediction_views(record, args.dedup_iou).items():
                    for category in (*CLASSES, "all"):
                        gt = truth if category == "all" else [g for g in truth if g["class"] == category]
                        pred = predictions if category == "all" else [p for p in predictions if p["class"] == category]
                        counts = count_matches(gt, pred, args.iou)
                        for i, value in enumerate(counts):
                            totals[(view, category)][i] += value
                for side, label in (("fp", "predicted_class"), ("fn", "gt_class")):
                    for error in raw_diag[side]:
                        errors[(side, error[label], error["reason"])] += 1
                for c in raw_diag["classification_confusion_candidates"]:
                    confusions[(c["gt_class"], c["predicted_class"])] += c["count"]
                detail_file.write(json.dumps({"model": model, "image": record["image"],
                    "ground_truth": truth, "predictions": record["predictions"],
                    "diagnostics": raw_diag}, ensure_ascii=False) + "\n")
                queries = record.get("queries", [])
                if queries:
                    sensitivity.append({"model": model, "image": Path(record["image"]).name,
                        "queries": len(queries),
                        "same_prompt_across_queries": all(q.get("prompt") for q in queries) and len({q["prompt"] for q in queries}) == 1,
                        "has_input_trace": all(q.get("trace") for q in queries),
                        "unique_raw_answers": len({q.get("answer", "") for q in queries}),
                        "unique_reference_hashes": len({q.get("trace", {}).get("reference_pixel_sha256")
                                                       for q in queries if q.get("trace", {}).get("reference_pixel_sha256")}),
                        "unique_processed_pixel_hashes": len({q.get("trace", {}).get("processed_pixel_sha256")
                            for q in queries if q.get("trace", {}).get("processed_pixel_sha256")})})
                for q in queries:
                    requested = q.get("requested_class")
                    if requested is None:
                        continue
                    pred = q["predictions"]
                    query_rows.append({"model": model, "image": Path(record["image"]).name,
                        "requested_class": requested, "predictions": len(pred),
                        "off_query_class_predictions": sum(p["class"] != requested for p in pred),
                        "none_answer": "<box>none</box>" in q.get("answer", "").lower(),
                        "ended_with_eos": q.get("trace", {}).get("ended_with_eos"),
                        "generated_tokens_reencoded": q.get("trace", {}).get("generated_tokens_reencoded")})
            for (view, category), counts in totals.items():
                score_rows.append({"model": model, "view": view, "class": category,
                                   "images": len(records), **scores(*counts)})
            error_rows.extend({"model": model, "side": side, "class": c, "reason": reason, "count": n}
                              for (side, c, reason), n in sorted(errors.items()))
            confusion_rows.extend({"model": model, "gt_class": g, "predicted_class": p,
                                   "classification_fp_candidates": n}
                                  for (g, p), n in sorted(confusions.items()))
            all_score = scores(*totals[("raw", "all")])
            print(f"{model}: TP={all_score['tp']} FP={all_score['fp']} FN={all_score['fn']} "
                  f"P={all_score['precision']:.3f} R={all_score['recall']:.3f} F1={all_score['f1']:.3f}")
    for name, rows in (("scores.csv", score_rows), ("error_types.csv", error_rows),
                       ("query_compliance.csv", query_rows), ("confusion_candidates.csv", confusion_rows),
                       ("reference_sensitivity.csv", sensitivity)):
        write_csv(args.output / name, rows)
    (args.output / "method.json").write_text(json.dumps({"match_iou": args.iou,
        "background_iou": 0.1, "dedup_iou": args.dedup_iou,
        "official_TIDE": False, "confidence_available": False,
        "fp_priority": ["duplicate", "localization", "classification", "classification_and_localization", "background"],
        "dedup_rule": "same-class high-IoU, keep first in original generation order; no invented confidence",
        "query_filter": "independent ablation; may remove true positives; labels never rewritten",
        "confusion_note": "Unmatched prediction candidates, not a one-to-one GT confusion matrix",
        "fn_note": "Nearest candidate reasons; not proof of causation or additive counterfactual gains",
        "reference_sensitivity_note": "Different hashes verify changed input, not semantic use; identical answers suggest insensitivity, not proof of a processor bug",
        "sources": [str(p) for p in args.inputs]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
