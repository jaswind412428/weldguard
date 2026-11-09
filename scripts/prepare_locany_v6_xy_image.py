"""Add one labeled train-only image example to each V6 X-Y sorted P04 sample."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from locany_image_example import (
    build_prompt, choose_reference, save_reference_images, select_reference_rows,
)


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "datasets/derived/locany_v6_xy_p04"
OUTPUT = REPO / "datasets/derived/locany_v6_xy_image"


def main() -> None:
    recipe_path = SOURCE / "full_recipe.json"
    if not recipe_path.is_file():
        raise FileNotFoundError(f"Missing {recipe_path}; run scripts/prepare_locany_v6_xy.py first")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if len(recipe) != 1:
        raise ValueError("Expected one V6 training dataset")
    entry = next(iter(recipe.values()))
    root = Path(entry["root"]).resolve()
    source_annotation = Path(entry["annotation"]).resolve()
    if (source_annotation != (SOURCE / "train_full.jsonl").resolve()
            or entry["repeat_time"] != 1.0 or entry["data_augment"]):
        raise ValueError("V6 recipe differs from the recorded X-Y experiment")
    rows = [json.loads(line) for line in source_annotation.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 653:
        raise ValueError(f"Expected 653 V6 training records; found {len(rows)}")

    references = save_reference_images(select_reference_rows(rows), root, OUTPUT)
    p04_prompt = (REPO / "prompts/locany_p04_balanced.txt").read_text(encoding="utf-8").strip()
    new_rows = []
    reference_use_counts = {reference["source_image"]: 0 for reference in references}
    for source_row in rows:
        target = source_row["image"]
        target_path = (root / target).resolve()
        if not target_path.is_relative_to(root / "fold0/train/images"):
            raise ValueError(f"Target must be a fold0 training image: {target_path}")
        with Image.open(target_path) as opened:
            width, height = opened.size
        reference = choose_reference(references, target)
        if reference["source_image"] == target:
            raise ValueError(f"Self-reference is forbidden: {target}")
        reference_use_counts[reference["source_image"]] += 1
        prompt = build_prompt(p04_prompt, width, height, reference["answer"])
        if prompt.count("<image-1>") != 1 or prompt.count("<image-2>") != 1:
            raise ValueError(f"Expected one target and one example image: {target}")
        answer = source_row["conversations"][-1]["value"]
        new_rows.append({
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": answer},
            ],
            "image": [target, reference["image"]],
        })

    annotation = OUTPUT / "train_full.jsonl"
    annotation.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in new_rows),
        encoding="utf-8",
    )
    entry["annotation"] = str(annotation.resolve())
    (OUTPUT / "full_recipe.json").write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "experiment": "V6-image",
        "base_model": "nvidia/LocateAnything-3B",
        "training_prompt": "P04 + image size + one labeled train-image example",
        "target_answer_order": "top-left x1, then y1",
        "training_images": len(new_rows),
        "source_annotation": str(source_annotation),
        "source_annotation_sha256": hashlib.sha256(source_annotation.read_bytes()).hexdigest(),
        "output_annotation_sha256": hashlib.sha256(annotation.read_bytes()).hexdigest(),
        "references": references,
        "reference_selection": "uniform pseudorandom train-only draw excluding target; seed 42 plus target image path",
        "reference_use_counts": reference_use_counts,
        "validation_or_test_examples_used": False,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"V6-image: {len(new_rows)} sorted-GT targets; {len(references)} train-only references")
    print(f"Unique references used: {sum(count > 0 for count in reference_use_counts.values())}")
    print(annotation)


if __name__ == "__main__":
    main()
