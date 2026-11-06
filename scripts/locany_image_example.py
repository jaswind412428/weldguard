"""Shared P04 image-example prompt and deterministic training-only references."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

from PIL import Image


REF_PATTERN = re.compile(r"<ref>([^<]+)</ref>")


def select_reference_rows(rows: list[dict]) -> list[dict]:
    """Use the entire labeled train split as the reproducible reference pool."""
    selected = []
    seen = set()
    for row in rows:
        answer = row["conversations"][-1]["value"]
        path = row["image"]
        if path in seen or not REF_PATTERN.search(answer) or "<box>" not in answer:
            raise ValueError(f"Invalid or duplicate training reference: {path}")
        selected.append(row)
        seen.add(path)
    return selected


def save_reference_images(rows: list[dict], root: Path, output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    references = []
    for index, row in enumerate(rows):
        source = (root / row["image"]).resolve()
        if not source.is_relative_to(root / "fold0/train/images"):
            raise ValueError(f"Reference must be a fold0 training image: {source}")
        output = output_dir / f"reference_{index:03d}.jpg"
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


def choose_reference(references: list[dict], target_image: str, seed: int = 42) -> dict:
    """Pseudorandomly sample one train image, excluding the target itself."""
    eligible = [reference for reference in references if reference["source_image"] != target_image]
    if not eligible:
        raise ValueError(f"No non-self training reference for {target_image}")
    digest = hashlib.sha256(f"{seed}:{target_image}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    return eligible[rng.randrange(len(eligible))]


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
