"""Prepare P10 training and validation data for V8, V8-G, and V8-N.

The images, labels, answer order, and splits match V7. Only the human prompts
change to P10. The 205-image test split is never read by this script.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

import prepare_locany_v7_family as v7
from locany_image_example import choose_reference


REPO = Path(__file__).resolve().parents[1]
ROOT = v7.ROOT
OUT = REPO / "datasets/derived/locany_v8"
PROMPT_FILE = REPO / "prompts/locany_p10_task_format_features.json"
VISUAL_PREFIX = (
    "<image-1> Locate every visible welding defect in the first image that is similar "
    "to the defect shown in the second image. <image-2> Use this image as a visual reference. "
)
SIZE_SENTENCE = "The first image is {width} pixels wide and {height} pixels high."


def load_prompts() -> dict[str, str]:
    prompts = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    if set(prompts) != {"none", "gt", "visual"}:
        raise ValueError("P10 must contain none, gt, and visual prompts")
    for mode, prompt in prompts.items():
        if (prompt.count("<image-1>") != 1 or
                prompt.count("<image-2>") != (mode != "none") or
                prompt.count("{width}") != 1 or prompt.count("{height}") != 1 or
                not prompt.endswith(SIZE_SENTENCE)):
            raise ValueError(f"P10 {mode} format changed; review V8 preparation")
    if not prompts["visual"].startswith(VISUAL_PREFIX):
        raise ValueError("P10 visual opening changed; review NVIDIA crop conversion")
    return prompts


def image_size(image: str) -> tuple[int, int]:
    with Image.open(ROOT / image) as opened:
        return opened.size


def prepare_rows(rows: list[dict], prompt: str) -> list[dict]:
    result = []
    for row in rows:
        width, height = image_size(row["image"])
        result.append({
            "conversations": [
                {"from": "human", "value": prompt.format(width=width, height=height)},
                row["conversations"][1],
            ],
            "image": row["image"],
        })
    return result


def add_gt(rows: list[dict], refs: list[dict], prompt: str) -> tuple[list[dict], list[str]]:
    result, pairings = [], []
    for row in rows:
        target = row["image"]
        reference = choose_reference(refs, target)
        if reference["source_image"] == target:
            raise ValueError(f"Self-reference: {target}")
        width, height = image_size(target)
        result.append({
            "conversations": [
                {"from": "human", "value": prompt.format(width=width, height=height)},
                row["conversations"][1],
            ],
            "image": [target, reference["image"]],
        })
        pairings.append(reference["source_image"])
    return result, pairings


def classwise(rows: list[dict], visual_prompt: str) -> tuple[list[dict], dict]:
    # NVIDIA's visual_prompt converter needs the exact category-set opener.
    # It replaces positive categories with an image crop and leaves absent
    # categories as one-image text queries.
    body = visual_prompt[len(VISUAL_PREFIX):]
    if not body.endswith(SIZE_SENTENCE):
        raise ValueError("P10 visual dimensions must remain last")
    body = body[:-len(SIZE_SENTENCE)].rstrip()
    result, counts = [], {"positive": Counter(), "negative": Counter()}
    for row in rows:
        target = row["image"]
        width, height = image_size(target)
        boxes = v7.parse_answer(row["conversations"][1]["value"])
        for category in v7.CLASSES:
            selected = [(name, box) for name, box in boxes if name == category]
            prompt = (
                f"<image-1> Detect all the objects in the image that belong to the category set: {category}. "
                + body
                + " Search only for the requested visual category. If absent, return "
                  "its English name in <ref>...</ref> followed by <box>none</box>. "
                + SIZE_SENTENCE.format(width=width, height=height)
            )
            answer = (v7.answer_for(selected) if selected else
                      f"<ref>{category}</ref><box>none</box>")
            result.append({
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": answer},
                ],
                "image": target,
            })
            counts["positive" if selected else "negative"][category] += 1
    return result, {key: dict(value) for key, value in counts.items()}


def write_recipe(variant: str, train: list[dict], val: list[dict]) -> dict:
    directory = OUT / variant
    directory.mkdir(parents=True, exist_ok=True)
    info = {}
    for split, rows in (("train", train), ("val", val)):
        path = directory / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                        encoding="utf-8")
        recipe = {f"v8_{variant}_{split}": {
            "annotation": str(path.resolve()), "root": str(ROOT.resolve()),
            "repeat_time": 1.0, "data_augment": False,
            "visual_prompt": variant == "v8-n",
        }}
        (directory / f"{split}_recipe.json").write_text(
            json.dumps(recipe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        info[split] = {"rows": len(rows), "annotation": str(path.resolve()),
                       "sha256": v7.sha(path)}
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("v8", "v8-g", "v8-n"), required=True)
    args = parser.parse_args()
    prompts = load_prompts()
    train = v7.read_train()
    val = v7.read_val("")
    references, pairings, counts = [], {}, {}
    if args.variant == "v8":
        train = prepare_rows(train, prompts["none"])
        val = prepare_rows(val, prompts["none"])
    elif args.variant == "v8-g":
        v7.OUT = OUT
        references = [v7.render_reference(row, index) for index, row in enumerate(train)]
        train, train_pairs = add_gt(train, references, prompts["gt"])
        val, val_pairs = add_gt(val, references, prompts["gt"])
        pairings = {"train": train_pairs, "val": val_pairs}
    else:
        train, train_counts = classwise(train, prompts["visual"])
        val, val_counts = classwise(val, prompts["visual"])
        counts = {"train": train_counts, "val": val_counts}
    info = write_recipe(args.variant, train, val)
    manifest = {
        "variant": args.variant, "base": "nvidia/LocateAnything-3B",
        "prompt": "P10", "prompt_sha256": v7.sha(PROMPT_FILE),
        "answer_order": "x1, then y1", "train_split": "fold0/train",
        "validation_split": "fold0/val", "test_used": False, "data": info,
        "reference_count": len(references),
        "reference_class_coverage": dict(Counter(cls for ref in references for cls in ref["classes"])),
        "reference_pairings": pairings, "classwise_counts": counts,
        "official_visual_prompt_conversion": args.variant == "v8-n",
    }
    (OUT / args.variant / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{args.variant}: train={info['train']['rows']} val={info['val']['rows']} "
          f"references={len(references)}")
    print(OUT / args.variant)


if __name__ == "__main__":
    main()
