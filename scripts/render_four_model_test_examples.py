"""Draw GT and four model predictions for the same held-out test images.

Use after four_models_test_fixed025_scores.csv has been produced. This creates
evidence images for a report; it does not alter predictions or scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {"pore": "#e53935", "deposit": "#168aad", "discontinuity": "#e69f00", "stain": "#20a35b"}
SOURCES = {
    "YOLO11s": "runs/detect/runs/detect/yolo11s_test_fixed025/predictions.jsonl",
    "YOLO26l": "runs/detect/runs/detect/yolo26l_test_fixed025/predictions.jsonl",
    "Author_YOLOv7": "runs/author_test/high_fold0_aug_1280_fixed025/predictions.jsonl",
    "VLM_LocateAnything": "runs/vlm/locany_fold0_v2_test_slow/predictions.jsonl",
}


def read_records(path: Path) -> dict[str, dict]:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            name = Path(item["image"]).name
            if name in records:
                raise ValueError(f"Duplicate image: {name} in {path}")
            records[name] = item
    return records


def render(image: Image.Image, boxes: list[dict], title: str) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    width, height = output.size
    stroke = max(3, round(min(width, height) / 120))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(17, round(min(width, height) / 28)))
    except OSError:
        font = ImageFont.load_default()
    for item in boxes:
        x1, y1, x2, y2 = item["box_01"]
        rect = (round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height))
        color = COLORS[item["class"]]
        draw.rectangle(rect, outline=color, width=stroke)
        label = item["class"]
        if "confidence" in item:
            label += f" {item['confidence']:.2f}"
        label_y = max(32, rect[1] - 24)
        bounds = draw.textbbox((rect[0], label_y), label, font=font)
        draw.rectangle(bounds, fill="#111111")
        draw.text((rect[0], label_y), label, fill=color, font=font)
    title_height = max(32, round(min(width, height) / 17))
    draw.rectangle((0, 0, width, title_height), fill="#111111")
    draw.text((8, 5), f"{title}: {len(boxes)} boxes", fill="white", font=font)
    return output


def choose_examples(records: dict[str, dict], limit: int) -> list[str]:
    # First cover rare classes, then fill deterministically by filename.
    chosen = []
    for category in ("pore", "deposit", "stain", "discontinuity"):
        for name in sorted(records):
            if name not in chosen and any(box["class"] == category for box in records[name]["ground_truth"]):
                chosen.append(name)
                break
        if len(chosen) >= limit:
            return chosen
    chosen.extend(name for name in sorted(records) if name not in chosen)
    return chosen[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.home() / "weldguard")
    parser.add_argument("--output", type=Path, default=Path.home() / "weldguard/results/report_test_examples")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--images", nargs="*", help="Optional exact test image filenames")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    records = {label: read_records(args.project / relative) for label, relative in SOURCES.items()}
    baseline = records["YOLO11s"]
    names = set(baseline)
    for label, model_records in records.items():
        if set(model_records) != names:
            raise ValueError(f"Image set differs for {label}")
    selected = args.images or choose_examples(baseline, args.limit)
    for name in selected:
        if name not in baseline or Path(name).name != name:
            raise ValueError(f"Unknown image: {name}")
        source = args.project / "datasets/weld-dataset/kfold_high/test/images" / name
        with Image.open(source) as image_file:
            image = image_file.convert("RGB")
        all_boxes = {"GT": baseline[name]["ground_truth"]}
        all_boxes.update({label: model_records[name]["predictions"] for label, model_records in records.items()})
        for label, boxes in all_boxes.items():
            target_dir = args.output / label
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / name
            render(image, boxes, label).save(target, quality=94)
            print(target)
    print(f"Rendered {len(selected)} test images for GT and all four models under {args.output}")


if __name__ == "__main__":
    main()
