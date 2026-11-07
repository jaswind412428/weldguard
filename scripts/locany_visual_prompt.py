"""P04-derived, classwise visual queries for LocateAnything.

The reference is an unannotated crop from a *different training image*.
This keeps the two-image NVIDIA visual-prompt format while allowing
negative target images, which the official same-image auto-crop cannot make.
"""

from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path


CLASSES = ("pore", "deposit", "discontinuity", "stain")
TOKEN_PATTERN = re.compile(
    r"<ref>([^<]+)</ref>|<box><(\d+)><(\d+)><(\d+)><(\d+)></box>",
    re.DOTALL,
)


def boxes_by_class(answer: str) -> dict[str, list[tuple[int, int, int, int]]]:
    """Read all normalized GT boxes without changing their coordinates."""
    result: dict[str, list[tuple[int, int, int, int]]] = {name: [] for name in CLASSES}
    current = None
    for match in TOKEN_PATTERN.finditer(answer):
        if match.group(1) is not None:
            current = match.group(1).strip()
            if current not in result:
                raise ValueError(f"Unknown GT class: {current!r}")
            continue
        if current is None:
            raise ValueError("GT box appears before a <ref> class")
        box = tuple(int(value) for value in match.groups()[1:])
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            raise ValueError(f"Invalid normalized GT box: {box}")
        result[current].append(box)
    if sum(map(len, result.values())) == 0:
        raise ValueError("GT answer contains no valid boxes")
    return result


def answer_for_class(category: str, boxes: list[tuple[int, int, int, int]]) -> str:
    if category not in CLASSES:
        raise ValueError(category)
    suffix = ("".join(f"<box><{x1}><{y1}><{x2}><{y2}></box>"
                      for x1, y1, x2, y2 in boxes)
              if boxes else "<box>None</box>")
    return f"<ref>{category}</ref>{suffix}"


def p04_visual_template(p04_path: Path) -> str:
    """Retain P04's weld definitions and box rules, adapting its query to one crop."""
    paragraphs = [part.strip() for part in p04_path.read_text(encoding="utf-8").split("\n\n")]
    if len(paragraphs) != 3 or "pore</c>deposit</c>discontinuity</c>stain" not in paragraphs[0]:
        raise ValueError("Unexpected P04 prompt; review its visual-query adaptation")
    definitions = paragraphs[1].replace(
        "Find every distinct visible defect, including multiple small instances of the same class.",
        "Find every distinct visible instance of the category shown by the reference crop.",
    )
    if "Find every distinct visible instance of the category shown" not in definitions:
        raise ValueError("P04 definitions changed; review prompt adaptation")
    return (
        "<image-1> Detect all the objects in the image that belong to the category set: <image-2>."
        "\n\n" + definitions + "\n\n"
        "The second image is one unmarked example crop from a separate training photograph. "
        "Find only defects matching its visual category in the first image; include every "
        "distinct instance of that category. Use only pore, deposit, discontinuity, or stain "
        "as the class name. Return tight boxes in the model's native format, using normalized "
        "0 to 1000 coordinates. If no matching defect is visible, return the category with "
        "a None box. Do not mark the whole image or add prose."
    )


def prompt_for_target(template: str, width: int, height: int) -> str:
    if template.count("<image-1>") != 1 or template.count("<image-2>") != 1:
        raise ValueError("Visual prompt requires one target and one reference placeholder")
    return (
        template
        + f" The first image is {width} pixels wide and {height} pixels high."
        + " Return its boxes using the normalized 0 to 1000 coordinate scale."
    )


def choose_reference(references: list[dict], category: str, target_image: str,
                     seed: int = 42) -> dict:
    eligible = [item for item in references
                if item["class"] == category and item["source_image"] != target_image]
    if not eligible:
        raise ValueError(f"No non-self reference for {category} / {target_image}")
    digest = hashlib.sha256(f"{seed}:{target_image}:{category}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    return eligible[rng.randrange(len(eligible))]
