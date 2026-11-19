"""Build paired whole-image P10 training and validation data for V8-N2."""

from __future__ import annotations

import json

from PIL import Image

import prepare_locany_v7_family as v7
from locany_image_example import choose_reference
from prepare_locany_v8_family import PROMPT_FILE, load_prompts


OUT = v7.REPO / "datasets/derived/locany_v8_n2_whole"
SEED = 42


def paired_rows(rows, references, prompts, fixed_reference=None):
    for row in rows:
        source = row["image"]
        reference = fixed_reference or choose_reference(references, source, SEED)
        if reference["source_image"] == source:
            raise ValueError(f"Reference is the target: {source}")
        with Image.open(v7.ROOT / source) as opened:
            width, height = opened.size
        for mode, images in (("none", source), ("gt", [source, reference["image"]])):
            yield {
                "image": images,
                "conversations": [
                    {"from": "human", "value": prompts[mode].format(width=width, height=height)},
                    row["conversations"][1],
                ],
            }


def main():
    train = v7.read_train()
    val = v7.read_val("")
    prompts = load_prompts()
    complete = [
        row for row in train
        if {name for name, _ in v7.parse_answer(row["conversations"][1]["value"])} == set(v7.CLASSES)
    ]
    if len({row["image"] for row in complete}) < 2:
        raise ValueError("Expected at least two training images containing all four classes")
    train_dir = (v7.ROOT / "fold0/train/images").resolve()
    for row in complete:
        if (v7.ROOT / row["image"]).resolve().parent != train_dir:
            raise ValueError(f"Reference is outside the training split: {row['image']}")

    v7.OUT = OUT
    references = [v7.render_reference(row, i) for i, row in enumerate(complete)]
    fixed_reference = choose_reference(references, "whole-image held-out reference", SEED)
    splits = {}
    for split, rows in (("train", train), ("val", val)):
        records = list(paired_rows(rows, references, prompts, fixed_reference if split == "val" else None))
        annotation = OUT / f"{split}.jsonl"
        annotation.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
        )
        recipe = {f"v8_n2_whole_{split}": {
            "annotation": str(annotation.resolve()), "root": str(v7.ROOT.resolve()),
            "repeat_time": 1.0, "data_augment": False, "visual_prompt": False,
        }}
        (OUT / f"{split}_recipe.json").write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
        splits[split] = {
            "images": len(rows), "without_reference": len(rows), "with_reference": len(rows),
            "annotation_sha256": v7.sha(annotation),
        }
    manifest = {
        "model_version": "V8-N2", "experiment": "whole-image continuation", "seed": SEED,
        "initial_weights": "runs/vlm/locany_reference_aligned_p10_original_epoch1",
        "prompt": "P10", "prompt_sha256": v7.sha(PROMPT_FILE),
        "train_split": "fold0/train", "validation_split": "fold0/val", "test_used": False,
        "reference_source": "fold0/train only", "four_class_reference_count": len(references),
        "fixed_reference": fixed_reference, "splits": splits,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
