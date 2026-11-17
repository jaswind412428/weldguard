"""Prepare external train-only references for both positive and negative queries.

This is an experimental extension of NVIDIA's data format, not an unchanged
official training recipe. No test images/labels are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

import prepare_locany_v7_family as v7

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "datasets/derived/locany_reference_alignment"
PROMPT = REPO / "prompts/locany_p11_class_reference.txt"
COLORS = {"pore": "#ff4040", "deposit": "#00b8d9", "discontinuity": "#f5a623", "stain": "#21b573"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_reference(bank, category, target, seed=42):
    eligible = [r for r in bank if r["class"] == category and r["source_image"] != target]
    if not eligible:
        raise ValueError(f"No non-self train reference: {target} / {category}")
    key = int.from_bytes(hashlib.sha256(f"{seed}:{target}:{category}".encode()).digest()[:8], "big")
    return random.Random(key).choice(eligible)


def make_class_rows(rows, bank, root, template, seed=42):
    output, counts = [], {"positive": Counter(), "negative": Counter()}
    for row in rows:
        target = row["image"]
        with Image.open(root / target) as im:
            width, height = im.size
        boxes = v7.parse_answer(row["conversations"][1]["value"])
        for category in v7.CLASSES:
            reference = choose_reference(bank, category, target, seed)
            selected = [(c, box) for c, box in boxes if c == category]
            answer = v7.answer_for(selected) if selected else f"<ref>{category}</ref><box>none</box>"
            output.append({"image": [target, reference["image"]], "conversations": [
                {"from": "human", "value": template.format(category=category, width=width, height=height)},
                {"from": "gpt", "value": answer}]})
            counts["positive" if selected else "negative"][category] += 1
    return output, {k: dict(v) for k, v in counts.items()}


def make_bank(train, root, out, padding):
    bank, gt_refs = [], []
    (out / "contexts").mkdir(parents=True, exist_ok=True)
    (out / "gt_references").mkdir(exist_ok=True)
    train_dir = (root / "fold0/train/images").resolve()
    for index, row in enumerate(train):
        source = row["image"]
        source_path = (root / source).resolve()
        if not source_path.is_relative_to(train_dir):
            raise ValueError(f"Reference outside train: {source}")
        boxes = v7.parse_answer(row["conversations"][1]["value"])
        with Image.open(source_path) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        # Preserve every GT instance as a candidate instead of only the first.
        for j, (category, box) in enumerate(boxes):
            x1, y1, x2, y2 = [v / 1000 for v in box]
            bw, bh = (x2 - x1) * width, (y2 - y1) * height
            dx, dy = max(2, bw * padding), max(2, bh * padding)
            pixel = (max(0, int(x1 * width - dx)), max(0, int(y1 * height - dy)),
                     min(width, int(x2 * width + dx + 1)), min(height, int(y2 * height + dy + 1)))
            crop = image.crop(pixel)
            path = out / "contexts" / f"train_{index:04d}_{j:03d}_{category}.png"
            crop.save(path)
            bank.append({"class": category, "source_image": source, "image": str(path.resolve()),
                         "gt_box_1000": box, "crop_pixels": pixel, "crop_size": list(crop.size),
                         "gt_pixel_size": [bw, bh], "source_sha256": sha(source_path),
                         "crop_sha256": sha(path)})
            crop.close()
        # A larger GT reference with class names in an added legend, not on defects.
        ref_image = image.copy()
        ref_image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (ref_image.width, ref_image.height + 40), "#161b22")
        canvas.paste(ref_image, (0, 0))
        draw = ImageDraw.Draw(canvas)
        for category, box in boxes:
            draw.rectangle(tuple(round(v * (ref_image.width if k % 2 == 0 else ref_image.height) / 1000)
                                 for k, v in enumerate(box)), outline=COLORS[category], width=2)
        for k, (category, color) in enumerate(COLORS.items()):
            draw.text((8 + k * canvas.width // 4, ref_image.height + 10), category, fill=color)
        ref_path = out / "gt_references" / f"train_{index:04d}.jpg"
        canvas.save(ref_path, quality=95)
        gt_refs.append({"source_image": source, "image": str(ref_path.resolve()), "sha256": sha(ref_path)})
        image.close()
        ref_image.close()
        canvas.close()
    return bank, gt_refs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--padding", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.padding <= 0.5:
        parser.error("padding must be between 0 and 0.5")
    train, val = v7.read_train(), v7.read_val("")
    template = PROMPT.read_text(encoding="utf-8").strip()
    config = {"padding": args.padding, "seed": args.seed, "prompt_sha256": sha(PROMPT),
              "train_source_sha256": sha(REPO / "datasets/derived/locany_v6_xy_p04/train_full.jsonl")}
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8"))["configuration"] != config:
        raise ValueError("Existing preparation has different settings; use a separate experiment folder")
    bank, gt_refs = make_bank(train, v7.ROOT, OUT, args.padding)
    info = {}
    for name, source_rows in (("train", train), ("val", val)):
        rows, counts = make_class_rows(source_rows, bank, v7.ROOT, template, args.seed)
        annotation = OUT / f"{name}.jsonl"
        annotation.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        recipe = {f"reference_aligned_{name}": {"annotation": str(annotation.resolve()),
                   "root": str(v7.ROOT.resolve()), "repeat_time": 1.0,
                   "data_augment": False, "visual_prompt": False}}
        (OUT / f"{name}_recipe.json").write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
        info[name] = {"images": len(source_rows), "queries": len(rows), "class_counts": counts,
                      "annotation_sha256": sha(annotation)}
    validation = sorted(Path(row["image"]).name for row in val)
    random.Random(args.seed).shuffle(validation)
    manifest = {"configuration": config, "data": info, "prompt": "P11",
        "test_used": False, "reference_source": "fold0/train only",
        "positive_and_negative_both_two_images": True, "official_auto_conversion": False,
        "reference_bank": bank, "gt_references": gt_refs, "validation_order": validation,
        "note": "Experimental external-reference extension. Raw tight-crop P10 baseline remains available."}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"data": info, "reference_crops": len(bank), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
