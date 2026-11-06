"""Add one labeled train-image example to each V5 P04 target, preserving GT."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from locany_image_example import (
    build_prompt, choose_reference, save_reference_images, select_reference_rows,
)


repo = Path(__file__).resolve().parents[1]
source_recipe_path = repo / "datasets/derived/locany_v5_p04/full_recipe.json"
recipe = json.loads(source_recipe_path.read_text(encoding="utf-8"))
if len(recipe) != 1:
    raise ValueError("Expected exactly one V5 training dataset")
entry = next(iter(recipe.values()))
root = Path(entry["root"]).resolve()
source_annotation = Path(entry["annotation"])
rows = [json.loads(line) for line in source_annotation.read_text(encoding="utf-8").splitlines() if line]
if len(rows) != 653 or entry["repeat_time"] != 1.0 or entry["data_augment"]:
    raise ValueError("V5 recipe differs from the recorded 653-image experiment")

output = repo / "datasets/derived/locany_v5_image"
references = save_reference_images(select_reference_rows(rows), root, output)
p04_prompt = (repo / "prompts/locany_p04_balanced.txt").read_text(encoding="utf-8").strip()
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
    reference_use_counts[reference["source_image"]] += 1
    prompt = build_prompt(p04_prompt, width, height, reference["answer"])
    if prompt.count("<image-1>") != 1 or prompt.count("<image-2>") != 1:
        raise ValueError("Exactly one placeholder per image is required")
    row = {
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": source_row["conversations"][-1]["value"]},
        ],
        "image": [target, reference["image"]],
    }
    if row["conversations"][-1]["value"] != source_row["conversations"][-1]["value"]:
        raise AssertionError("GT answer changed")
    new_rows.append(row)

annotation = output / "train_full.jsonl"
annotation.write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in new_rows), encoding="utf-8"
)
entry["annotation"] = str(annotation.resolve())
(output / "full_recipe.json").write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
manifest = {
    "experiment": "V5-image",
    "base_model": "nvidia/LocateAnything-3B",
    "prompt_id": "P04 + one labeled training-image example",
    "training_images": len(rows),
    "source_annotation": str(source_annotation),
    "source_annotation_sha256": hashlib.sha256(source_annotation.read_bytes()).hexdigest(),
    "output_annotation_sha256": hashlib.sha256(annotation.read_bytes()).hexdigest(),
    "references": references,
    "reference_selection": "uniform pseudorandom draw from all 653 train images excluding target; seed 42 plus target image path",
    "reference_use_counts": reference_use_counts,
    "validation_or_test_examples_used": False,
}
(output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"V5-image: {len(new_rows)} targets; {len(references)} train-only references in the pool.")
print(f"Unique references used: {sum(count > 0 for count in reference_use_counts.values())}")
print(annotation)
