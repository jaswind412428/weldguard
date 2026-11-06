"""Check actual training sample lengths before starting the one-shot run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eaglevl.train.locany_finetune_magi_stream import LazySupervisedDatasetMTP
from eaglevl.train.tools import process_multimodal_sample
from eaglevl.utils.locany.processing_locateanything import LocateAnythingProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--model", default="nvidia/LocateAnything-3B")
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    if len(recipe) != 1:
        raise ValueError("Expected one dataset in recipe")
    meta = next(iter(recipe.values()))
    annotation = Path(meta["annotation"])
    rows = [json.loads(line) for line in annotation.read_text(encoding="utf-8").splitlines() if line]
    processor = LocateAnythingProcessor.from_pretrained(args.model, trust_remote_code=True)
    dataset = LazySupervisedDatasetMTP("v5_image_preflight", meta, processor, block_size=4)
    oversize = []
    longest = (0, "")
    for index, row in enumerate(rows):
        messages = process_multimodal_sample(
            row, str(meta["root"]), dataset.max_frames, dataset.target_fps,
            dataset.video_total_pixels, visual_prompt=False,
        )
        sample = dataset.multi_modal_get_item(messages)
        length = int(sample["input_ids"].numel())
        if length > longest[0]:
            longest = (length, row["image"][0])
        if length > args.max_tokens:
            oversize.append((row["image"][0], length))
        if (index + 1) % 100 == 0:
            print(f"Checked {index + 1}/{len(rows)}", flush=True)
    print(f"Checked {len(rows)} records; longest={longest[0]} tokens ({longest[1]})")
    print(f"Over {args.max_tokens} tokens: {len(oversize)}")
    for path, length in oversize[:20]:
        print(f"  {path}: {length}")
    if oversize:
        raise SystemExit("Preflight failed: training would skip overlength samples")


if __name__ == "__main__":
    main()
