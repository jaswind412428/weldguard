"""Shared P04 image-example prompt and deterministic training-only references."""

from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image


CLASSES = ("pore", "deposit", "discontinuity", "stain")
REF_PATTERN = re.compile(r"<ref>([^<]+)</ref>")


def select_reference_rows(rows: list[dict], count: int = 2) -> list[dict]:
    """Pick short, class-diverse answers without looking at val or test data."""
    ranked = []
    for row in rows:
        answer = row["conversations"][-1]["value"]
        present = set(REF_PATTERN.findall(answer)) & set(CLASSES)
        boxes = answer.count("<box>")
        if not present or boxes == 0:
            continue
        ranked.append((-len(present), boxes, row["image"], row))
    ranked.sort(key=lambda item: item[:3])
    if len(ranked) < count:
        raise ValueError(f"Need {count} labeled training examples")
    return [item[3] for item in ranked[:count]]


def save_reference_images(rows: list[dict], root: Path, output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    references = []
    for index, row in enumerate(rows):
        source = (root / row["image"]).resolve()
        if not source.is_relative_to(root / "fold0/train/images"):
            raise ValueError(f"Reference must be a fold0 training image: {source}")
        output = output_dir / f"reference_{index}.jpg"
        with Image.open(source) as opened:
            image = opened.convert("RGB")
            original_size = image.size
            image.thumbnail((640, 640), Image.Resampling.LANCZOS)
            image.save(output, quality=95)
        references.append({
            "source_image": row["image"],
            "image": str(output.resolve()),
            "answer": row["conversations"][-1]["value"],
            "source_size": list(original_size),
            "display_size": list(image.size),
        })
    return references


def choose_reference(references: list[dict], target_image: str) -> dict:
    for reference in references:
        if reference["source_image"] != target_image:
            return reference
    raise ValueError(f"No non-self reference for {target_image}")


def build_prompt(p04_prompt: str, width: int, height: int, example_answer: str) -> str:
    image_size = (
        f" The image is {width} pixels wide and {height} pixels high."
        " Return boxes using the model's normalized 0 to 1000 coordinate scale."
    )
    return (
        p04_prompt.strip() + image_size
        + "\n\nLabeled training example, separate from the target image: <image-2>"
        + "\nThe correct answer for that example image is: " + example_answer
        + "\nNow output detections for the target image only."
        + " The example coordinates belong only to the example image."
    )


def load_references(manifest_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    references = manifest["references"]
    for reference in references:
        if not Path(reference["image"]).is_file():
            raise FileNotFoundError(reference["image"])
    return references
