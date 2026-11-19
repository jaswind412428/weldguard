"""Duplicate pore-positive target rows while preserving the original V7-G training pairs."""

from __future__ import annotations

import json
import random
from pathlib import Path

from prepare_locany_v7_family import OUT as V7_DATA
from prepare_locany_v7_family import REPO, ROOT, parse_answer, sha


SOURCE = V7_DATA / "v7-g"
OUT = REPO / "datasets/derived/locany_v7_g_pore_balance"
SEED = 42


def read_original_split(split, expected_count):
    recipe = json.loads((SOURCE / f"{split}_recipe.json").read_text(encoding="utf-8"))
    if len(recipe) != 1:
        raise ValueError(f"Expected one original V7-G {split} dataset")
    entry = next(iter(recipe.values()))
    if (Path(entry["root"]).resolve() != ROOT.resolve() or entry["repeat_time"] != 1.0
            or entry["data_augment"] or entry["visual_prompt"]):
        raise ValueError(f"V7-G {split} recipe differs from the original settings")
    annotation = Path(entry["annotation"])
    rows = [json.loads(line) for line in annotation.read_text(encoding="utf-8").splitlines() if line]
    target_dir = (ROOT / f"fold0/{split}/images").resolve()
    reference_dir = (SOURCE.parent / "gt_references").resolve()
    targets = set()
    for row in rows:
        target, reference = [(ROOT / image).resolve() for image in row["image"]]
        if target.parent != target_dir or not target.is_file() or target in targets:
            raise ValueError(f"Invalid or repeated V7-G {split} target: {target}")
        if reference.parent != reference_dir or not reference.is_file():
            raise ValueError(f"Invalid V7-G GT reference: {reference}")
        targets.add(target)
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} V7-G {split} targets, found {len(rows)}")
    return rows, annotation, recipe


def main():
    train, train_source, recipe = read_original_split("train", 653)
    _, val_source, _ = read_original_split("val", 164)
    pore_rows = [
        row for row in train
        if any(name == "pore" for name, _ in parse_answer(row["conversations"][1]["value"]))
    ]
    if not 0 < len(pore_rows) < len(train):
        raise ValueError("Expected target images both with and without pore")
    entry = next(iter(recipe.values()))
    for arm, weight in (("control", 1), ("pore2x", 2)):
        directory = OUT / arm
        directory.mkdir(parents=True, exist_ok=True)
        samples = train + pore_rows * (weight - 1)
        random.Random(SEED).shuffle(samples)
        annotation = directory / "train.jsonl"
        annotation.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples),
                              encoding="utf-8")
        recipe_out = {f"v7_g_{arm}": {**entry, "annotation": str(annotation.resolve())}}
        (directory / "train_recipe.json").write_text(json.dumps(recipe_out, indent=2) + "\n", encoding="utf-8")
        manifest = {
            "experiment": f"V7-G P04/GT {arm}", "starting_checkpoint": "runs/vlm/locany_v7_g_p04_xy",
            "train_split": "fold0/train", "validation_split": "fold0/val", "test_used": False,
            "seed": SEED, "pore_positive_weight": weight, "unique_images": len(train),
            "train_rows": len(samples), "pore_positive_rows": len(pore_rows) * weight,
            "pore_absent_rows": len(train) - len(pore_rows),
            "source_sha256": sha(train_source), "train_sha256": sha(annotation),
            "validation_sha256": sha(val_source),
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"{arm}: {manifest['pore_positive_rows']}/{len(samples)} rows have target pores; "
              f"{len(train)} unique training images")


if __name__ == "__main__":
    main()
