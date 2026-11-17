"""Select high-disagreement V8/P10 test images and package visual evidence.

Reads the nine completed prediction JSONL files on the 4090. The ZIP contains
only selected comparison sheets and per-image metrics, not model weights or the
full test dataset/predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageDraw, ImageOps

from evaluate_locany_predictions import count_matches


REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "runs/vlm/v8_p10_nine_test"
OUTPUT = REPO / "results/v8_p10_difference_cases.zip"
VARIANTS = ("v8", "v8-g", "v8-n")
MODES = ("none", "gt", "visual")
KEYS = tuple(f"{variant}_{mode}" for variant in VARIANTS for mode in MODES)
COLORS = {"pore": "#ff4040", "deposit": "#00b8d9", "discontinuity": "#f5a623", "stain": "#21b573"}
CELL_W, CELL_H = 640, 360
HEADER_H = 32


def load_records(run: Path) -> dict[str, dict[str, dict]]:
    result = {}
    expected_names = None
    expected_truth = None
    for variant in VARIANTS:
        for mode in MODES:
            key = f"{variant}_{mode}"
            path = run / key / "predictions.jsonl"
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}; finish all nine inference modes first")
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
            by_name = {Path(row["image"]).name: row for row in rows}
            if len(rows) != 205 or len(by_name) != 205:
                raise ValueError(f"Expected 205 unique images in {path}; found {len(rows)}")
            if any(row.get("model") != f"{variant.upper()} / {mode} / P10" for row in rows):
                raise ValueError(f"Unexpected model or prompt label in {path}")
            names = set(by_name)
            truth = {name: row["ground_truth"] for name, row in by_name.items()}
            if expected_names is None:
                expected_names, expected_truth = names, truth
            elif names != expected_names or truth != expected_truth:
                raise ValueError(f"Different image set or GT boxes in {path}")
            result[key] = by_name
    return result


def score(truth: list[dict], predictions: list[dict]) -> dict:
    tp, fp, fn = count_matches(truth, predictions, 0.5)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "predictions": len(predictions)}


def image_metrics(records: dict[str, dict[str, dict]]) -> dict[str, dict]:
    names = sorted(next(iter(records.values())))
    result = {}
    for name in names:
        truth = records[KEYS[0]][name]["ground_truth"]
        modes = {}
        for key in KEYS:
            row = records[key][name]
            modes[key] = {
                **score(truth, row["predictions"]),
                "rejected_numeric": row.get("rejected_parsed_boxes", 0),
                "predicted_classes": dict(Counter(item["class"] for item in row["predictions"])),
                "per_class": {
                    category: score([item for item in truth if item["class"] == category],
                                    [item for item in row["predictions"] if item["class"] == category])
                    for category in COLORS
                },
            }
        result[name] = {"image": name, "gt_boxes": len(truth),
                        "gt_classes": dict(Counter(item["class"] for item in truth)),
                        "modes": modes}
    return result


def choose_cases(metrics: dict[str, dict], per_reason: int) -> list[dict]:
    # Each contrast names one visible behavior to inspect. Require at least
    # four GT boxes so a single match cannot dominate the per-image F1.
    contrasts = (
        ("V8: GT image reduces F1", "v8_none", "v8_gt", "f1"),
        ("V8: visual input adds false boxes", "v8_visual", "v8_none", "fp"),
        ("V8-G: GT image restores F1", "v8-g_gt", "v8-g_none", "f1"),
        ("V8-G: visual input adds false boxes", "v8-g_visual", "v8-g_gt", "fp"),
        ("V8-N: visual input restores F1", "v8-n_visual", "v8-n_none", "f1"),
        ("V8 versus V8-N without a reference", "v8_none", "v8-n_none", "f1"),
    )
    selected: dict[str, dict] = {}
    for reason, high, low, field in contrasts:
        ranked = sorted(
            ((row["modes"][high][field] - row["modes"][low][field], name)
             for name, row in metrics.items() if row["gt_boxes"] >= 4),
            key=lambda item: (-item[0], -metrics[item[1]]["gt_boxes"], item[1]),
        )
        taken = 0
        for delta, name in ranked:
            if delta <= 0 or name in selected:
                continue
            selected[name] = {"image": name, "reason": reason,
                              "contrast": [high, low], "delta": round(delta, 4)}
            taken += 1
            if taken == per_reason:
                break
    if len(selected) < 6:
        raise ValueError(f"Only {len(selected)} distinct contrast cases; inspect metrics before packing")
    return list(selected.values())


def fit_image(source: Image.Image, width: int, height: int) -> tuple[Image.Image, int, int]:
    resized = ImageOps.contain(source.convert("RGB"), (width, height), Image.Resampling.LANCZOS)
    return resized, (width - resized.width) // 2, (height - resized.height) // 2


def panel(source: Image.Image, boxes: list[dict], title: str) -> Image.Image:
    result = Image.new("RGB", (CELL_W, CELL_H), "#161b22")
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, CELL_W, HEADER_H - 1), fill="#293241")
    draw.text((8, 9), title, fill="white")
    content_h = CELL_H - HEADER_H
    resized, left, top = fit_image(source, CELL_W, content_h)
    result.paste(resized, (left, HEADER_H + top))
    draw = ImageDraw.Draw(result)
    for item in boxes:
        x1, y1, x2, y2 = item["box_01"]
        coords = (left + round(x1 * resized.width), HEADER_H + top + round(y1 * resized.height),
                  left + round(x2 * resized.width), HEADER_H + top + round(y2 * resized.height))
        color = COLORS.get(item["class"], "white")
        draw.rectangle(coords, outline=color, width=2)
        draw.text((coords[0], max(HEADER_H, coords[1] - 12)), item["class"], fill=color,
                  stroke_width=1, stroke_fill="black")
    resized.close()
    return result


def image_panel(path: Path | None, title: str) -> Image.Image:
    if path is None or not path.is_file():
        return panel(Image.new("RGB", (640, 300), "#161b22"), [], f"{title}: unavailable")
    with Image.open(path) as opened:
        return panel(opened, [], title)


def gt_reference_map() -> dict[str, Path]:
    source = REPO / "datasets/derived/locany_v8/v8/train.jsonl"
    folder = REPO / "datasets/derived/locany_v8/gt_references"
    if not source.is_file():
        return {}
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    return {row["image"]: folder / f"train_{index:04d}.jpg" for index, row in enumerate(rows)}


def crop_gallery(references: dict) -> Image.Image:
    result = Image.new("RGB", (CELL_W, CELL_H), "#161b22")
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, CELL_W, HEADER_H - 1), fill="#293241")
    draw.text((8, 9), "visual: four training crops", fill="white")
    for index, category in enumerate(COLORS):
        ref = references.get(category, {})
        path = Path(ref["crop"]) if isinstance(ref, dict) and ref.get("crop") else None
        x = (index % 2) * (CELL_W // 2)
        y = HEADER_H + (index // 2) * ((CELL_H - HEADER_H) // 2)
        draw.text((x + 7, y + 3), category, fill=COLORS[category])
        if path is None or not path.is_file():
            continue
        with Image.open(path) as opened:
            resized, left, top = fit_image(opened, CELL_W // 2 - 12, (CELL_H - HEADER_H) // 2 - 26)
            result.paste(resized, (x + 6 + left, y + 22 + top))
            resized.close()
    return result


def render_sheet(name: str, records: dict[str, dict[str, dict]],
                 metrics: dict[str, dict], gt_refs: dict[str, Path]) -> Image.Image:
    source = Path(records[KEYS[0]][name]["image"])
    if not source.is_file():
        raise FileNotFoundError(source)
    with Image.open(source) as opened:
        target = opened.convert("RGB")
    truth = records[KEYS[0]][name]["ground_truth"]
    sheet = Image.new("RGB", (CELL_W * 3, CELL_H * 4), "white")
    gt = panel(target, truth, f"{name}  GT={len(truth)}")
    sheet.paste(gt, (0, 0))
    gt.close()
    ref_source = records["v8_gt"][name].get("references", {}).get("gt")
    reference = image_panel(gt_refs.get(ref_source), f"GT reference: {ref_source or 'none'}")
    sheet.paste(reference, (CELL_W, 0))
    reference.close()
    gallery = crop_gallery(records["v8_visual"][name].get("references", {}))
    sheet.paste(gallery, (CELL_W * 2, 0))
    gallery.close()
    for row_index, variant in enumerate(VARIANTS, start=1):
        for col_index, mode in enumerate(MODES):
            key = f"{variant}_{mode}"
            result = metrics["modes"][key]
            title = (f"{variant.upper()} / {mode}: TP={result['tp']} FP={result['fp']} "
                     f"FN={result['fn']} F1={result['f1']:.2f}")
            rendered = panel(target, records[key][name]["predictions"], title)
            sheet.paste(rendered, (CELL_W * col_index, CELL_H * row_index))
            rendered.close()
    target.close()
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--per-reason", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.per_reason <= 5:
        parser.error("--per-reason must be between 1 and 5")
    records = load_records(args.run)
    metrics = image_metrics(records)
    selected = choose_cases(metrics, args.per_reason)
    gt_refs = gt_reference_map()
    selected_data = []
    for item in selected:
        name = item["image"]
        selected_data.append({
            **item, **metrics[name],
            "references": {key: records[key][name].get("references", {}) for key in KEYS},
            "answer_excerpt": {
                key: {query: str(answer)[:400] for query, answer in
                      records[key][name].get("answer", {}).items()}
                for key in KEYS
            },
        })
    all_rows = []
    for name, row in metrics.items():
        flat = {"image": name, "gt_boxes": row["gt_boxes"]}
        for key in KEYS:
            for field in ("tp", "fp", "fn", "f1", "predictions", "rejected_numeric"):
                flat[f"{key}_{field}"] = row["modes"][key][field]
        all_rows.append(flat)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("selection.json", json.dumps(selected_data, ensure_ascii=False, indent=2) + "\n")
        archive.writestr("README.txt", "V8/P10, 205 test images, class-aware IoU >= 0.5. "
                         "Each sheet shows target GT, one train GT reference, four train visual crops, "
                         "and all nine model/input outputs. TP/FP/FN are per image.\n")
        import io
        csv_data = io.StringIO()
        writer = csv.DictWriter(csv_data, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
        archive.writestr("all_image_metrics.csv", csv_data.getvalue())
        for item in selected:
            name = item["image"]
            sheet = render_sheet(name, records, metrics[name], gt_refs)
            with io.BytesIO() as buffer:
                sheet.save(buffer, format="JPEG", quality=88)
                archive.writestr(f"sheets/{Path(name).stem}.jpg", buffer.getvalue())
            sheet.close()
            print(f"{name}: {item['reason']}, delta={item['delta']}")
    if args.output.stat().st_size > 90_000_000:
        raise ValueError(f"Report ZIP exceeds 90 MB: {args.output}")
    print(f"Saved {len(selected)} high-disagreement sheets and 205-image metrics to {args.output}")


if __name__ == "__main__":
    main()
