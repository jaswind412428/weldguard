"""Prepare three P04/X-Y V7 training recipes and held-out validation recipes.

This script reads fold0 train/val only. The 205-image test split is untouched.
Generated images and JSONL live below datasets/derived and stay out of Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from locany_image_example import choose_reference
from prepare_locany_pilot import NAMES, read_boxes


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "datasets/weld-dataset/kfold_high"
OUT = REPO / "datasets/derived/locany_v7"
CLASSES = tuple(NAMES.values())
COLORS = {"pore": "#ff4040", "deposit": "#00b8d9", "discontinuity": "#f5a623", "stain": "#21b573"}
ENTRY = re.compile(r"<ref>(pore|deposit|discontinuity|stain)</ref><box><(\d{1,4})><(\d{1,4})><(\d{1,4})><(\d{1,4})></box>")
SIZE = " The image is {width} pixels wide and {height} pixels high. Return boxes using the model's normalized 0 to 1000 coordinate scale."


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_answer(answer: str) -> list[tuple[str, tuple[int, int, int, int]]]:
    matches = list(ENTRY.finditer(answer))
    if not matches or "".join(match.group(0) for match in matches) != answer:
        raise ValueError(f"Unexpected GT format: {answer[:100]!r}")
    boxes = []
    for match in matches:
        box = tuple(map(int, match.groups()[1:]))
        if not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000):
            raise ValueError(f"Bad GT box: {box}")
        boxes.append((match.group(1), box))
    return boxes


def answer_for(boxes: list[tuple[str, tuple[int, int, int, int]]]) -> str:
    return "".join(f"<ref>{name}</ref><box><{x1}><{y1}><{x2}><{y2}></box>"
                   for name, (x1, y1, x2, y2) in boxes)


def sorted_boxes(boxes: list[tuple[str, tuple[int, int, int, int]]]):
    return sorted(boxes, key=lambda item: (item[1][0], item[1][1]))


def read_train() -> list[dict]:
    source = REPO / "datasets/derived/locany_v6_xy_p04/train_full.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"Missing {source}; run scripts/prepare_locany_v6_xy.py first")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 653:
        raise ValueError(f"Expected 653 train images; got {len(rows)}")
    for row in rows:
        image = row["image"]
        if not isinstance(image, str) or not image.startswith("fold0/train/images/"):
            raise ValueError(f"Train split violation: {image!r}")
        if not (ROOT / image).is_file():
            raise FileNotFoundError(ROOT / image)
        boxes = parse_answer(row["conversations"][1]["value"])
        if boxes != sorted_boxes(boxes):
            raise ValueError(f"GT not in X-Y order: {image}")
    return rows


def read_val(p04: str) -> list[dict]:
    image_dir = ROOT / "fold0/val/images"
    label_dir = ROOT / "fold0/val/labels"
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(images) != 164:
        raise ValueError(f"Expected 164 validation images; got {len(images)}")
    rows = []
    for image in images:
        label = label_dir / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(label)
        boxes = sorted_boxes([(NAMES[class_id], box) for class_id, box in read_boxes(label)])
        if not boxes:
            raise ValueError(f"No usable GT in {label}")
        with Image.open(image) as opened:
            width, height = opened.size
        rows.append({
            "conversations": [
                {"from": "human", "value": p04 + SIZE.format(width=width, height=height)},
                {"from": "gpt", "value": answer_for(boxes)},
            ],
            "image": f"fold0/val/images/{image.name}",
        })
    return rows


def render_reference(row: dict, index: int) -> dict:
    source = row["image"]
    boxes = parse_answer(row["conversations"][1]["value"])
    with Image.open(ROOT / source) as opened:
        image = opened.convert("RGB")
    original_size = image.size
    image.thumbnail((640, 640), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    stroke = max(2, round(min(width, height) / 160))
    for name, (x1, y1, x2, y2) in boxes:
        pixel = (round(x1 * width / 1000), round(y1 * height / 1000),
                 round(x2 * width / 1000), round(y2 * height / 1000))
        color = COLORS[name]
        draw.rectangle(pixel, outline=color, width=stroke)
        tx, ty = pixel[0], max(0, pixel[1] - 13)
        label_width = 8 * len(name) + 4
        draw.rectangle((tx, ty, min(width - 1, tx + label_width), min(height - 1, ty + 12)), fill="black")
        draw.text((tx + 2, ty), name, fill=color)
    path = OUT / "gt_references" / f"train_{index:04d}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=95)
    image.close()
    return {"source_image": source, "image": str(path.resolve()), "classes": sorted({name for name, _ in boxes}),
            "original_size": list(original_size), "rendered_size": [width, height], "sha256": sha(path)}


def with_reference(row: dict, refs: list[dict], p04: str) -> tuple[dict, str]:
    target = row["image"]
    reference = choose_reference(refs, target)
    if target == reference["source_image"]:
        raise ValueError(f"Self-reference: {target}")
    with Image.open(ROOT / target) as opened:
        width, height = opened.size
    prompt = (p04 + SIZE.format(width=width, height=height)
              + "\n\nThe second image <image-2> is a different training photograph with its correct defects "
                "already outlined and named. Use it only as a visual example of the four classes. "
                "Locate and classify defects in the first, unmarked image only. Do not copy the "
                "example boxes or their coordinates into the answer.")
    return {"conversations": [{"from": "human", "value": prompt}, row["conversations"][1]],
            "image": [target, reference["image"]]}, reference["source_image"]


def classwise(rows: list[dict], p04: str) -> tuple[list[dict], dict]:
    paragraphs = p04.split("\n\n")
    if len(paragraphs) != 3 or "pore</c>deposit</c>discontinuity</c>stain" not in paragraphs[0]:
        raise ValueError("P04 changed; review classwise adaptation")
    output, counts = [], {"positive": Counter(), "negative": Counter()}
    for row in rows:
        target = row["image"]
        with Image.open(ROOT / target) as opened:
            width, height = opened.size
        boxes = parse_answer(row["conversations"][1]["value"])
        for category in CLASSES:
            selected = [(name, box) for name, box in boxes if name == category]
            # NVIDIA's visual-prompt converter matches this exact category-set
            # sentence and replaces its category with the cropped image query.
            prompt = (f"<image-1> Detect all the objects in the image that belong to the category set: {category}.\n\n"
                      + paragraphs[1] + "\n\n" + paragraphs[2]
                      + " Search for the requested visual pattern only. Return boxes for that pattern; "
                        "if absent, return the class ref tag followed by <box>none</box>."
                      + SIZE.format(width=width, height=height))
            answer = answer_for(selected) if selected else f"<ref>{category}</ref><box>none</box>"
            output.append({"conversations": [{"from": "human", "value": prompt},
                                              {"from": "gpt", "value": answer}], "image": target})
            counts["positive" if selected else "negative"][category] += 1
    return output, {key: dict(value) for key, value in counts.items()}


def write_recipe(name: str, train_rows: list[dict], val_rows: list[dict], visual: bool = False) -> dict:
    directory = OUT / name
    directory.mkdir(parents=True, exist_ok=True)
    info = {}
    for split, rows in (("train", train_rows), ("val", val_rows)):
        path = directory / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        recipe = {f"v7_{name}_{split}": {"annotation": str(path.resolve()), "root": str(ROOT.resolve()),
                   "repeat_time": 1.0, "data_augment": False, "visual_prompt": visual}}
        recipe_path = directory / f"{split}_recipe.json"
        recipe_path.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
        info[split] = {"rows": len(rows), "annotation": str(path.resolve()), "sha256": sha(path)}
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("v7", "v7-g", "v7-n"), required=True)
    args = parser.parse_args()
    p04 = (REPO / "prompts/locany_p04_balanced.txt").read_text(encoding="utf-8").strip()
    train = read_train()
    val = read_val(p04)
    refs = []
    pairings = {}
    counts = {}
    if args.variant == "v7-g":
        refs = [render_reference(row, index) for index, row in enumerate(train)]
        train, train_pairings = zip(*(with_reference(row, refs, p04) for row in train))
        val, val_pairings = zip(*(with_reference(row, refs, p04) for row in val))
        pairings = {"train": list(train_pairings), "val": list(val_pairings)}
    elif args.variant == "v7-n":
        train, train_counts = classwise(train, p04)
        val, val_counts = classwise(val, p04)
        counts = {"train": train_counts, "val": val_counts}
    info = write_recipe(args.variant, list(train), list(val), args.variant == "v7-n")
    manifest = {"variant": args.variant, "base": "nvidia/LocateAnything-3B", "prompt": "P04 + image size",
                "answer_order": "x1, then y1", "train_split": "fold0/train", "validation_split": "fold0/val",
                "test_used": False, "data": info, "reference_count": len(refs),
                "reference_class_coverage": dict(Counter(cls for ref in refs for cls in ref["classes"])),
                "reference_pairings": pairings, "classwise_counts": counts,
                "official_visual_prompt_conversion": args.variant == "v7-n"}
    (OUT / args.variant / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{args.variant}: train={info['train']['rows']} val={info['val']['rows']} refs={len(refs)}")
    print(OUT / args.variant)


if __name__ == "__main__":
    main()
