"""Create classwise visual-prompt training data, including absent-class answers.

The official auto-converter crops a positive box from the target image and
leaves negative queries as text. Here every query instead receives an
unannotated crop from a different fold0 training image of the requested class.
This is a documented extension to NVIDIA's two-image training format.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from locany_visual_prompt import (
    CLASSES, answer_for_class, boxes_by_class, choose_reference,
    p04_visual_template, prompt_for_target,
)


def crop_box(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = box
    left = max(0, min(width - 1, round(x1 * width / 1000)))
    top = max(0, min(height - 1, round(y1 * height / 1000)))
    right = max(left + 1, min(width, round(x2 * width / 1000)))
    bottom = max(top + 1, min(height, round(y2 * height / 1000)))
    return image.crop((left, top, right, bottom)).convert("RGB")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    source = repo / "datasets/derived/locany_v5_p04/full_recipe.json"
    recipe = json.loads(source.read_text(encoding="utf-8"))
    if len(recipe) != 1:
        raise ValueError("Expected exactly one V5 P04 training dataset")
    entry = next(iter(recipe.values()))
    root = Path(entry["root"]).resolve()
    annotation = Path(entry["annotation"])
    rows = [json.loads(line) for line in annotation.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 653 or entry["repeat_time"] != 1.0 or entry["data_augment"]:
        raise ValueError("V5 P04 recipe differs from the recorded 653-image experiment")

    output = repo / "datasets/derived/locany_v5_visual"
    reference_dir = output / "references"
    reference_dir.mkdir(parents=True, exist_ok=True)
    template = p04_visual_template(repo / "prompts/locany_p04_balanced.txt")
    references = []
    samples = []

    for index, row in enumerate(rows):
        if len(row["conversations"]) != 2 or row["conversations"][1]["from"] != "gpt":
            raise ValueError(f"Unexpected conversation in training row {index}")
        image_name = row["image"]
        if not isinstance(image_name, str):
            raise ValueError(f"Expected one training image in row {index}")
        path = (root / image_name).resolve()
        if not path.is_relative_to(root / "fold0/train/images"):
            raise ValueError(f"Not a fold0 training image: {path}")
        boxes = boxes_by_class(row["conversations"][1]["value"])
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        for category in CLASSES:
            if not boxes[category]:
                continue
            crop = crop_box(image, boxes[category][0])
            crop_path = reference_dir / f"ref_{index:04d}_{category}.png"
            crop.save(crop_path)
            references.append({
                "class": category,
                "source_image": image_name,
                "image": str(crop_path.resolve()),
                "box_1000": boxes[category][0],
            })
        samples.append((image_name, image.size, boxes))
        image.close()

    bank_counts = Counter(item["class"] for item in references)
    if any(bank_counts[category] < 2 for category in CLASSES):
        raise ValueError(f"Insufficient non-self train references: {bank_counts}")

    output_rows = []
    positive = Counter()
    negative = Counter()
    for image_name, (width, height), boxes in samples:
        for category in CLASSES:
            reference = choose_reference(references, category, image_name)
            target = (root / image_name).resolve()
            answer = answer_for_class(category, boxes[category])
            output_rows.append({
                "conversations": [
                    {"from": "human", "value": prompt_for_target(template, width, height)},
                    {"from": "gpt", "value": answer},
                ],
                "image": [str(target), reference["image"]],
            })
            (positive if boxes[category] else negative)[category] += 1

    target_annotation = output / "train_full.jsonl"
    target_annotation.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    entry["annotation"] = str(target_annotation.resolve())
    entry["visual_prompt"] = False  # Already a two-image visual query; do not auto-crop.
    (output / "full_recipe.json").write_text(
        json.dumps(recipe, indent=2) + "\n", encoding="utf-8"
    )
    (output / "training_prompt_template.txt").write_text(template + "\n", encoding="utf-8")
    manifest = {
        "experiment": "V5-visual",
        "base_model": "nvidia/LocateAnything-3B",
        "prompt_id": "P04-derived visual query",
        "source_prompt": "prompts/locany_p04_balanced.txt",
        "training_images": len(samples),
        "training_queries": len(output_rows),
        "queries_per_target": len(CLASSES),
        "class_order": list(CLASSES),
        "positive_queries_by_class": dict(positive),
        "negative_queries_by_class": dict(negative),
        "reference_bank_by_class": dict(bank_counts),
        "references": references,
        "reference_selection": "first GT crop per present class per fold0 train image; seeded draw from a different train image for each target/class",
        "official_method_deviation": "NVIDIA visual_prompt=true auto-crops a positive from the same image and leaves negative samples text-only; this prepared two-image dataset uses external train-only crops for both positive and negative targets",
        "validation_or_test_used_in_training": False,
        "source_annotation": str(annotation),
        "source_annotation_sha256": hashlib.sha256(annotation.read_bytes()).hexdigest(),
        "output_annotation_sha256": hashlib.sha256(target_annotation.read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"V5-visual: {len(samples)} target images; {len(output_rows)} queries")
    print(f"Positive queries: {dict(positive)}")
    print(f"Absent-class queries: {dict(negative)}")
    print(f"Train-only reference crops: {dict(bank_counts)}")
    print(target_annotation)


if __name__ == "__main__":
    main()
