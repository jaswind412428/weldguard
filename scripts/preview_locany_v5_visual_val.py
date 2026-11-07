"""Evaluate a trained V5-visual model with four train-only crop queries per image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from locany_visual_prompt import CLASSES, choose_reference, prompt_for_target
from preview_locany_val import (
    choose_images, draw_panel, ground_truth, parse_predictions, predict,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--generation-mode", choices=("hybrid", "slow", "fast"), default="slow")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name-reference-class", action="store_true",
                        help="Add the reference crop's known class name to each query")
    args = parser.parse_args()
    if args.limit < 1 or args.max_new_tokens < 1 or args.max_input_tokens < 1:
        parser.error("--limit and token limits must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("experiment") != "V5-visual" or manifest.get("validation_or_test_used_in_training"):
        parser.error("Expected a train-only V5-visual manifest")
    references = manifest["references"]
    recipe = json.loads((args.manifest.parent / "full_recipe.json").read_text(encoding="utf-8"))
    root = Path(next(iter(recipe.values()))["root"]).resolve()
    train_dir = root / "fold0/train/images"
    if args.split.resolve() != (root / "fold0/val").resolve():
        parser.error("Use fold0 validation; leave the independent test set untouched")
    for ref in references:
        if not (root / ref["source_image"]).resolve().is_relative_to(train_dir):
            parser.error(f"Reference source is outside train: {ref['source_image']}")
        if not Path(ref["image"]).is_file():
            parser.error(f"Missing reference crop: {ref['image']}")
    template = (args.manifest.parent / "training_prompt_template.txt").read_text(encoding="utf-8").strip()
    selected = choose_images(args.split / "images", args.split / "labels", args.limit)
    args.output.mkdir(parents=True, exist_ok=True)
    compare_dir = args.output / "compare"
    compare_dir.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(str(args.model), trust_remote_code=True)
    model = AutoModel.from_pretrained(
        str(args.model), trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for image_path in selected:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            truth = ground_truth(args.split / "labels" / f"{image_path.stem}.txt")
            answers = {}
            used_references = {}
            predictions = []
            rejected = 0
            base_prompt = prompt_for_target(template, image.width, image.height)
            prompts = {}
            for category in CLASSES:
                prompt = base_prompt
                if args.name_reference_class:
                    prompt += (
                        f" The second image shows an example of the {category} defect class."
                        f" Detect {category} instances in the first image."
                    )
                reference = choose_reference(references, category, image_path.name, args.seed)
                with Image.open(reference["image"]) as opened:
                    crop = opened.convert("RGB")
                answer = predict(model, processor, tokenizer, image, args, prompt, crop)
                crop.close()
                parsed, invalid = parse_predictions(answer)
                accepted = [item for item in parsed if item["class"] == category]
                rejected += invalid + len(parsed) - len(accepted)
                predictions.extend(accepted)
                answers[category] = answer
                prompts[category] = prompt
                used_references[category] = {
                    "source_image": reference["source_image"],
                    "reference_crop": reference["image"],
                }
            left = draw_panel(image, truth, f"Ground truth: {len(truth)}")
            right = draw_panel(image, predictions, f"Prediction: {len(predictions)}")
            comparison = Image.new("RGB", (image.width * 2, image.height))
            comparison.paste(left, (0, 0))
            comparison.paste(right, (image.width, 0))
            comparison.save(compare_dir / image_path.name, quality=95)
            record = {
                "image": str(image_path),
                "model": ("V5-visual (P04, four train-crop queries, named class)"
                          if args.name_reference_class else
                          "V5-visual (P04, four train-crop queries)"),
                "prompt": prompts if args.name_reference_class else base_prompt,
                "references": used_references,
                "answer": answers,
                "ground_truth": truth,
                "predictions": predictions,
                "task": "class_detection",
                "rejected_parsed_boxes": rejected,
                "generation": {
                    "mode": args.generation_mode,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "repetition_penalty": args.repetition_penalty,
                    "max_new_tokens": args.max_new_tokens,
                    "seed": args.seed,
                    "queries_per_image": len(CLASSES),
                },
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{image_path.name}: GT={len(truth)} predicted={len(predictions)} rejected={rejected}", flush=True)
    print(f"Saved {len(selected)} comparisons to {compare_dir}")


if __name__ == "__main__":
    main()
