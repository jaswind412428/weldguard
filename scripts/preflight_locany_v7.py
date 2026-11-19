"""Fail before V7/V8 training if samples are truncated or visual conversion is wrong."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from eaglevl.train.augmentation import apply_resize_augmentation
from eaglevl.train.locany_finetune_magi_stream import LazySupervisedDatasetMTP
from eaglevl.train.tools import process_multimodal_sample
from eaglevl.utils.locany.processing_locateanything import LocateAnythingProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--variant", choices=("v7", "v7-g", "v7-n", "v8", "v8-g", "v8-n", "v8-n2-whole"), required=True)
    parser.add_argument("--model", default="nvidia/LocateAnything-3B")
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    if len(recipe) != 1:
        raise ValueError("Expected exactly one recipe entry")
    meta = next(iter(recipe.values()))
    rows = [json.loads(line) for line in Path(meta["annotation"]).read_text(encoding="utf-8").splitlines() if line]
    processor = LocateAnythingProcessor.from_pretrained(args.model, trust_remote_code=True)
    dataset = LazySupervisedDatasetMTP("v7_preflight", meta, processor, block_size=4)
    counts = Counter()
    longest = (0, "")
    oversize = []
    for index, row in enumerate(rows):
        messages = process_multimodal_sample(row, str(meta["root"]), dataset.max_frames,
                                             dataset.target_fps, dataset.video_total_pixels,
                                             visual_prompt=bool(meta.get("visual_prompt", False)))
        rendered = processor.py_apply_chat_template(messages, tokenize=False)
        images, videos = processor.process_vision_info(messages)
        image_count = 0 if images is None else len(images)
        counts[image_count] += 1
        if args.variant == "v8-n2-whole":
            expected = len(row["image"]) if isinstance(row["image"], list) else 1
            if expected not in (1, 2):
                raise ValueError(f"Row {index}: expected one target and at most one reference")
        else:
            expected = 2 if args.variant.endswith("-g") else 1
        if not args.variant.endswith("-n") and image_count != expected:
            raise ValueError(f"Row {index}: expected {expected} images, got {image_count}")
        if args.variant.endswith("-n"):
            positive = "<box><" in row["conversations"][1]["value"]
            expected = 2 if positive else 1
            if image_count != expected:
                raise ValueError(
                    f"Row {index}: visual conversion expected {expected} images "
                    f"but got {image_count}; prompt={row['conversations'][0]['value'][:180]!r}"
                )
        if images is not None:
            images = [apply_resize_augmentation(img, data_augment=False, min_long_edge=640,
                                                max_long_edge=2560, augment_prob=0.5)
                      for img in images]
        tokenized = processor(text=rendered, images=images, videos=videos,
                              return_tensors="pt", padding=False, truncation=False)
        ids = tokenized["input_ids"][0]
        if images is not None:
            dataset._validate_image_token_alignment(ids, tokenized["pixel_values"], tokenized["image_grid_hws"])
        sample = dataset.get_targets_flag_with_mtp(ids)
        length = int(sample["input_ids"].numel())
        if length > longest[0]:
            longest = (length, str(row["image"]))
        if length > args.max_tokens:
            oversize.append((str(row["image"]), length))
        if (index + 1) % 100 == 0:
            print(f"Checked {index + 1}/{len(rows)}", flush=True)
    print(f"Checked {len(rows)} samples, image counts={dict(counts)}, longest={longest}")
    if args.variant == "v8-n2-whole" and (counts[1] == 0 or counts[1] != counts[2]):
        raise ValueError(f"Expected equal counts of one-image and two-image rows: {counts}")
    if args.variant.endswith("-n"):
        # The official converter should make positive class queries two-image
        # samples and keep absent classes as one-image text queries.
        if counts[1] == 0 or counts[2] == 0:
            raise ValueError(f"Visual prompt conversion did not produce both positive and negative modes: {counts}")
    if oversize:
        for item in oversize[:20]:
            print(f"Over token limit: {item}")
        raise ValueError(f"{len(oversize)} samples exceed {args.max_tokens} tokens")


if __name__ == "__main__":
    main()
