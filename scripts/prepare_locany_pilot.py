"""Convert a few LoHi-WELD YOLO labels to LocateAnything training JSONL.

Run on Linux from ~/weldguard after installing Pillow in the VLM environment.
The source dataset is read only. Outputs default under datasets/, which Git ignores.
"""

from __future__ import annotations

import argparse
import json
import random
import warnings
from pathlib import Path

from PIL import Image, ImageDraw


NAMES = {0: "pore", 1: "deposit", 2: "discontinuity", 3: "stain"}
COLORS = {0: "#ff4040", 1: "#00b8d9", 2: "#f5a623", 3: "#21b573"}
PROMPT = "Detect all weld defects in <image-1>."
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def read_boxes(label_path: Path) -> list[tuple[int, tuple[int, int, int, int]]]:
    boxes = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 YOLO fields")
        try:
            class_id = int(fields[0])
            cx, cy, width, height = (float(value) for value in fields[1:])
        except ValueError as error:
            raise ValueError(f"{label_path}:{line_number}: invalid number") from error
        if class_id not in NAMES:
            raise ValueError(f"{label_path}:{line_number}: unknown class {class_id}")
        x1, y1 = cx - width / 2, cy - height / 2
        x2, y2 = cx + width / 2, cy + height / 2
        if not (x1 < x2 and y1 < y2):
            raise ValueError(f"{label_path}:{line_number}: invalid normalized box")
        clipped = (
            max(0.0, min(1.0, x1)),
            max(0.0, min(1.0, y1)),
            max(0.0, min(1.0, x2)),
            max(0.0, min(1.0, y2)),
        )
        if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
            warnings.warn(f"{label_path}:{line_number}: box outside image; skipped")
            continue
        if clipped != (x1, y1, x2, y2):
            warnings.warn(f"{label_path}:{line_number}: box clipped to image boundary")
        box = tuple(round(value * 1000) for value in clipped)
        if box[0] >= box[2] or box[1] >= box[3]:
            warnings.warn(f"{label_path}:{line_number}: box too small after rounding; skipped")
            continue
        boxes.append((class_id, box))
    return boxes


def make_preview(image_path: Path, boxes: list, output_path: Path) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    stroke = max(2, round(min(width, height) / 150))
    for class_id, (x1, y1, x2, y2) in boxes:
        pixel_box = (
            round(x1 * width / 1000),
            round(y1 * height / 1000),
            round(x2 * width / 1000),
            round(y2 * height / 1000),
        )
        color = COLORS[class_id]
        draw.rectangle(pixel_box, outline=color, width=stroke)
        draw.text((pixel_box[0], max(0, pixel_box[1] - 12)), NAMES[class_id], fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.home() / "weldguard/datasets/weld-dataset/kfold_high",
        help="LoHi-WELD kfold_high directory",
    )
    parser.add_argument("--split", default="fold0/train", help="Only use a training split")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "weldguard/datasets/derived/locany_pilot",
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.split != "fold0/train":
        parser.error("pilot mode only accepts --split fold0/train; val/test must stay held out")

    root = args.root.expanduser().resolve()
    image_dir = root / args.split / "images"
    label_dir = root / args.split / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        parser.error(f"missing images or labels: {image_dir}, {label_dir}")

    candidates = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
        boxes = read_boxes(label_path)
        if boxes:
            candidates.append((image_path, boxes))
    if len(candidates) < args.limit:
        raise ValueError(f"Only {len(candidates)} annotated train images; asked for {args.limit}")

    rng = random.Random(args.seed)
    shuffled = candidates[:]
    rng.shuffle(shuffled)
    selected = []
    covered = set()
    for image_path, boxes in shuffled:
        new_classes = {class_id for class_id, _ in boxes} - covered
        if new_classes and len(selected) < args.limit:
            selected.append((image_path, boxes))
            covered.update(class_id for class_id, _ in boxes)
        if covered == set(NAMES):
            break
    for item in shuffled:
        if len(selected) >= args.limit:
            break
        if item not in selected:
            selected.append(item)
    output = args.output.expanduser().resolve()
    preview_dir = output / "previews"
    output.mkdir(parents=True, exist_ok=True)
    jsonl_path = output / "train_pilot.jsonl"
    manifest_path = output / "manifest.txt"
    count_by_class = {name: 0 for name in NAMES.values()}

    with jsonl_path.open("w", encoding="utf-8") as jsonl, manifest_path.open(
        "w", encoding="utf-8"
    ) as manifest:
        for image_path, boxes in selected:
            answer = "".join(
                f"<ref>{NAMES[class_id]}</ref><box>({','.join(map(str, box))})</box>"
                for class_id, box in boxes
            )
            record = {
                "conversations": [
                    {"from": "human", "value": PROMPT},
                    {"from": "gpt", "value": answer},
                ],
                "image": image_path.relative_to(root).as_posix(),
            }
            jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            classes = ", ".join(sorted({NAMES[class_id] for class_id, _ in boxes}))
            manifest.write(f"{image_path.name}: {len(boxes)} boxes ({classes})\n")
            make_preview(image_path, boxes, preview_dir / image_path.name)
            for class_id, _ in boxes:
                count_by_class[NAMES[class_id]] += 1

    recipe = {
        "weldguard_pilot": {
            "annotation": str(jsonl_path),
            "root": str(root),
            "repeat_time": 1.0,
            "data_augment": False,
        }
    }
    (output / "pilot_recipe.json").write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Created {len(selected)} training records in {jsonl_path}")
    print(f"Boxes by class: {count_by_class}")
    print(f"Review all previews in {preview_dir} before training")


if __name__ == "__main__":
    main()
