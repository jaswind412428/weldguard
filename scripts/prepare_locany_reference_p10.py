"""P10 + original train crops, with two-image positive AND negative queries.

Uses the same prompt/reference selection as the validation baseline. This is
an experimental external-reference extension, not NVIDIA's auto-crop recipe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import prepare_locany_v7_family as v7
from prepare_locany_reference_alignment import REPO, make_class_rows, sha

OUT = REPO / "datasets/derived/locany_reference_p10_original"
PROMPT = REPO / "prompts/locany_p10_task_format_features.json"
BANK = REPO / "datasets/derived/locany_v5_visual/manifest.json"


def load_original_bank(root, manifest):
    if manifest.get("validation_or_test_used_in_training") is not False:
        raise ValueError("Reference manifest must certify train-only sources")
    bank = manifest["references"]
    train_dir = (root / "fold0/train/images").resolve()
    fingerprints = []
    for ref in bank:
        source = (root / ref["source_image"]).resolve()
        crop = Path(ref["image"])
        if ref["class"] not in v7.CLASSES or not source.is_relative_to(train_dir):
            raise ValueError(f"Invalid train-only reference: {ref}")
        if not source.is_file() or not crop.is_file():
            raise FileNotFoundError(f"Missing source/crop: {source} / {crop}")
        fingerprints.append([ref["class"], ref["source_image"], sha(crop)])
    if any(sum(r["class"] == c for r in bank) < 2 for c in v7.CLASSES):
        raise ValueError("Need at least two non-self reference sources per class")
    digest = hashlib.sha256(json.dumps(fingerprints, separators=(",", ":")).encode()).hexdigest()
    return bank, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train, val = v7.read_train(), v7.read_val("")
    template = json.loads(PROMPT.read_text(encoding="utf-8"))["visual"]
    bank, crop_hash = load_original_bank(v7.ROOT, json.loads(BANK.read_text(encoding="utf-8")))
    config = {"seed": args.seed, "prompt_sha256": sha(PROMPT),
              "train_source_sha256": sha(REPO / "datasets/derived/locany_v6_xy_p04/train_full.jsonl"),
              "reference_manifest_sha256": sha(BANK), "reference_crop_contents_sha256": crop_hash}
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8"))["configuration"] != config:
        raise ValueError("Existing P10 preparation differs; use a separate experiment folder")
    data = {}
    for split, source_rows in (("train", train), ("val", val)):
        rows, counts = make_class_rows(source_rows, bank, v7.ROOT, template, args.seed)
        annotation = OUT / f"{split}.jsonl"
        annotation.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        recipe = {f"p10_original_aligned_{split}": {"annotation": str(annotation.resolve()),
                  "root": str(v7.ROOT.resolve()), "repeat_time": 1.0,
                  "data_augment": False, "visual_prompt": False}}
        (OUT / f"{split}_recipe.json").write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
        data[split] = {"images": len(source_rows), "queries": len(rows),
                       "class_counts": counts, "annotation_sha256": sha(annotation)}
    manifest = {"configuration": config, "data": data, "prompt": "P10 visual",
        "reference_bank": "original tight train crops", "test_used": False,
        "positive_and_negative_both_two_images": True, "official_auto_conversion": False,
        "answer_order": "x1, then y1", "base_model": "nvidia/LocateAnything-3B",
        "note": "Inference remains P10/original. Training target and reference pairing, including negatives, are aligned; this does not prove a single isolated cause."}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"data": data, "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
