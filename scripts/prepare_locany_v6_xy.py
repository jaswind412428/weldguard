"""Build a V5-identical LocateAnything dataset with GT entries sorted by x1, then y1."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "datasets/derived/locany_v5_p04"
OUTPUT = REPO / "datasets/derived/locany_v6_xy_p04"
ENTRY = re.compile(
    r"<ref>(pore|deposit|discontinuity|stain)</ref>"
    r"<box><(\d{1,4})><(\d{1,4})><(\d{1,4})><(\d{1,4})></box>"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sort_answer(answer: str, image: str) -> tuple[str, int]:
    matches = list(ENTRY.finditer(answer))
    if not matches or "".join(match.group(0) for match in matches) != answer:
        raise ValueError(f"Unexpected answer format: {image}")

    for match in matches:
        x1, y1, x2, y2 = map(int, match.groups()[1:])
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            raise ValueError(f"Invalid GT box: {image}: {match.group(0)}")

    # Stable sort: keep the original order when two boxes share the same top-left corner.
    ordered = sorted(matches, key=lambda match: (int(match.group(2)), int(match.group(3))))
    if Counter(match.group(0) for match in matches) != Counter(match.group(0) for match in ordered):
        raise ValueError(f"GT entries changed: {image}")
    return "".join(match.group(0) for match in ordered), len(matches)


def main() -> None:
    source_annotation = SOURCE / "train_full.jsonl"
    source_recipe = SOURCE / "full_recipe.json"
    if not source_annotation.is_file() or not source_recipe.is_file():
        raise FileNotFoundError(
            f"Missing V5 training data in {SOURCE}; run scripts/prepare_locany_v5_p04.py first"
        )

    recipe = json.loads(source_recipe.read_text(encoding="utf-8"))
    if len(recipe) != 1:
        raise ValueError("Expected one V5 training recipe")
    settings = next(iter(recipe.values()))
    if (Path(settings["annotation"]).resolve() != source_annotation.resolve()
            or settings["repeat_time"] != 1.0 or settings["data_augment"]):
        raise ValueError("V5 recipe differs from the recorded experiment")

    rows = [json.loads(line) for line in source_annotation.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 653:
        raise ValueError(f"Expected 653 V5 training records; found {len(rows)}")

    changed = 0
    total_boxes = 0
    for row in rows:
        image = row["image"]
        if not image.startswith("fold0/train/images/"):
            raise ValueError(f"Non-training image in V5 data: {image}")
        turns = row["conversations"]
        if (len(turns) != 2 or turns[0]["from"] != "human" or turns[1]["from"] != "gpt"):
            raise ValueError(f"Unexpected conversation structure: {image}")
        original = turns[1]["value"]
        ordered, count = sort_answer(original, image)
        turns[1]["value"] = ordered
        total_boxes += count
        changed += ordered != original

    if changed == 0:
        raise ValueError("All V5 answers are already X-Y sorted; there is no ordering A/B to run")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_annotation = OUTPUT / "train_full.jsonl"
    output_annotation.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    settings["annotation"] = str(output_annotation.resolve())
    (OUTPUT / "full_recipe.json").write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "model": "nvidia/LocateAnything-3B",
        "experiment": "V6_XY_vs_V5_P04",
        "only_change": "Sort complete GT entries by top-left x1, then y1",
        "train_images": len(rows),
        "gt_boxes": total_boxes,
        "answers_reordered": changed,
        "source_annotation": str(source_annotation.resolve()),
        "source_sha256": digest(source_annotation),
        "sorted_sha256": digest(output_annotation),
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"V6 X-Y: {len(rows)} images, {total_boxes} unchanged GT boxes, {changed} answers reordered")
    print(output_annotation)


if __name__ == "__main__":
    main()
