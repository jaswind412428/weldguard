"""Class-aware matching and TIDE-inspired fixed-threshold diagnostics.

No confidence scores are available, so this is not official TIDE or delta-AP.
Raw P/R/F1 use the existing maximum-cardinality, same-class IoU matching.
"""
from __future__ import annotations

from collections import Counter
import re

from evaluate_locany_predictions import CLASSES, iou, scores


def parse_answer(answer):
    result, rejected, category = [], 0, None
    pattern = r"<ref>(.*?)</ref>|<box><(\d{1,4})><(\d{1,4})><(\d{1,4})><(\d{1,4})></box>"
    for match in re.finditer(pattern, answer, re.DOTALL):
        if match.group(1) is not None:
            category = match.group(1).strip().lower()
            continue
        box = list(map(int, match.groups()[1:]))
        if category not in CLASSES or max(box) > 1000 or box[0] >= box[2] or box[1] >= box[3]:
            rejected += 1
        else:
            result.append({"class": category, "box_01": [v / 1000 for v in box]})
    return result, rejected


def matching(truth, predictions, threshold=0.5):
    edges = [[j for j, gt in enumerate(truth)
              if gt["class"] == pred["class"] and iou(pred["box_01"], gt["box_01"]) >= threshold]
             for pred in predictions]
    claimed = {}

    def assign(p, seen):
        for g in edges[p]:
            if g in seen:
                continue
            seen.add(g)
            if g not in claimed or assign(claimed[g], seen):
                claimed[g] = p
                return True
        return False

    for p in range(len(predictions)):
        assign(p, set())
    return {p: g for g, p in claimed.items()}


def deduplicate(predictions, threshold=0.9):
    """Keep first same-class high-IoU box, never rank with GT or invented confidence."""
    if not 0 < threshold <= 1:
        raise ValueError("dedup IoU must be in (0, 1]")
    kept, removed = [], []
    for index, pred in enumerate(predictions):
        if any(pred["class"] == old["class"] and
               iou(pred["box_01"], old["box_01"]) >= threshold for old in kept):
            removed.append(index)
        else:
            kept.append(pred)
    return kept, removed


def prediction_views(record, threshold=0.9):
    raw = record["predictions"]
    views = {"raw": raw, "dedup": deduplicate(raw, threshold)[0]}
    queries = record.get("queries")
    if queries is not None and all("predictions" in q and "requested_class" in q for q in queries):
        filtered = [pred for q in queries for pred in q["predictions"]
                    if q["requested_class"] is None or pred["class"] == q["requested_class"]]
        views["query_filter"] = filtered
        views["query_filter_dedup"] = deduplicate(filtered, threshold)[0]
    return views


def diagnose(truth, predictions, threshold=0.5, background=0.1):
    if not 0 < background < threshold <= 1:
        raise ValueError("Require 0 < background IoU < match IoU <= 1")
    matches = matching(truth, predictions, threshold)
    used_gt = set(matches.values())
    fp, fn = [], []
    confusion = Counter()
    for p, pred in enumerate(predictions):
        if p in matches:
            continue
        overlaps = [(iou(pred["box_01"], gt["box_01"]), g) for g, gt in enumerate(truth)]
        same = [(value, g) for value, g in overlaps if truth[g]["class"] == pred["class"]]
        other = [(value, g) for value, g in overlaps if truth[g]["class"] != pred["class"]]
        same_iou, same_g = max(same, default=(0, None))
        other_iou, other_g = max(other, default=(0, None))
        if same_iou >= threshold:
            reason, g, value = "duplicate", same_g, same_iou
        elif same_iou >= background:
            reason, g, value = "localization", same_g, same_iou
        elif other_iou >= threshold:
            reason, g, value = "classification", other_g, other_iou
            confusion[(truth[g]["class"], pred["class"])] += 1
        elif other_iou >= background:
            reason, g, value = "classification_and_localization", other_g, other_iou
        else:
            value, g = max(overlaps, default=(0, None))
            reason = "background"
        fp.append({"prediction_index": p, "predicted_class": pred["class"],
                   "reason": reason, "nearest_gt_index": g, "iou": value})
    for g, gt in enumerate(truth):
        if g in used_gt:
            continue
        # FN attribution uses only unmatched predictions and may share candidates;
        # it is not an additive counterfactual performance improvement.
        candidates = [(iou(gt["box_01"], pred["box_01"]), p)
                      for p, pred in enumerate(predictions) if p not in matches]
        same = [(value, p) for value, p in candidates if predictions[p]["class"] == gt["class"]]
        other = [(value, p) for value, p in candidates if predictions[p]["class"] != gt["class"]]
        same_iou, same_p = max(same, default=(0, None))
        other_iou, other_p = max(other, default=(0, None))
        if other_iou >= threshold:
            reason, p, value = "wrong_class_at_gt", other_p, other_iou
        elif same_iou >= background:
            reason, p, value = "same_class_poor_overlap", same_p, same_iou
        elif other_iou >= background:
            reason, p, value = "wrong_class_poor_overlap", other_p, other_iou
        else:
            # A broad already-matched box can cover multiple GTs. Surface this
            # instead of calling it an entirely unseen object.
            value, p = max([(iou(gt["box_01"], pred["box_01"]), p)
                            for p, pred in enumerate(predictions)], default=(0, None))
            reason = "only_already_matched_overlap" if value >= background else "no_overlap_above_background"
        fn.append({"gt_index": g, "gt_class": gt["class"], "reason": reason,
                   "candidate_prediction_index": p, "iou": value})
    return {"scores": scores(len(matches), len(fp), len(fn)), "matches": matches,
            "fp": fp, "fn": fn,
            "fp_reasons": dict(Counter(item["reason"] for item in fp)),
            "fn_reasons": dict(Counter(item["reason"] for item in fn)),
            "classification_confusion_candidates": [
                {"gt_class": g, "predicted_class": p, "count": n}
                for (g, p), n in sorted(confusion.items())]}


def class_scores(truth, predictions, threshold=0.5):
    return {c: diagnose([g for g in truth if g["class"] == c],
                        [p for p in predictions if p["class"] == c], threshold)["scores"]
            for c in CLASSES}
