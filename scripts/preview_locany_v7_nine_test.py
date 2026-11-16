"""Run one V7 weight/input combination on the held-out 205-image test split.

The two reference modes use only fold0/train images. Predictions are written
incrementally so rerunning the same command resumes after an interruption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from locany_image_example import choose_reference as choose_gt_reference
from locany_visual_prompt import CLASSES, choose_reference as choose_crop_reference
from preview_locany_val import draw_panel, ground_truth, parse_predictions, predict


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "datasets/weld-dataset/kfold_high"
SPLIT = ROOT / "test"
MODELS = {
    "v7": REPO / "runs/vlm/locany_v7_p04_xy",
    "v7-g": REPO / "runs/vlm/locany_v7_g_p04_xy",
    "v7-n": REPO / "runs/vlm/locany_v7_n_p04_xy",
}
SIZE = " The image is {width} pixels wide and {height} pixels high. Return boxes using the model's normalized 0 to 1000 coordinate scale."
GT_SUFFIX = (
    "\n\nThe second image <image-2> is a different training photograph with its correct defects "
    "already outlined and named. Use it only as a visual example of the four classes. "
    "Locate and classify defects in the first, unmarked image only. Do not copy the "
    "example boxes or their coordinates into the answer."
)
VISUAL_SUFFIX = (
    " Search for the requested visual pattern only. Return boxes for that pattern; "
    "if absent, return the class ref tag followed by <box>none</box>."
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gt_references() -> list[dict]:
    annotation = REPO / "datasets/derived/locany_v7/v7/train.jsonl"
    folder = REPO / "datasets/derived/locany_v7/gt_references"
    rows = [json.loads(line) for line in annotation.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 653:
        raise ValueError(f"Expected 653 train references; got {len(rows)}")
    refs = []
    for index, row in enumerate(rows):
        source = row["image"]
        path = folder / f"train_{index:04d}.jpg"
        if not source.startswith("fold0/train/images/") or not (ROOT / source).is_file() or not path.is_file():
            raise ValueError(f"Invalid train-only GT reference {source}: {path}")
        refs.append({"source_image": source, "image": str(path.resolve())})
    return refs


def load_crop_references() -> list[dict]:
    source = REPO / "datasets/derived/locany_v5_visual/manifest.json"
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if manifest.get("validation_or_test_used_in_training") is not False:
        raise ValueError("Visual reference bank must contain training images only")
    refs = manifest["references"]
    present = {category: 0 for category in CLASSES}
    for ref in refs:
        image = (ROOT / ref["source_image"]).resolve()
        if (ref["class"] not in present or
                not image.is_relative_to(ROOT / "fold0/train/images") or
                not image.is_file() or not Path(ref["image"]).is_file()):
            raise ValueError(f"Invalid train-only crop reference: {ref}")
        present[ref["class"]] += 1
    if any(count == 0 for count in present.values()):
        raise ValueError(f"Missing crop reference class: {present}")
    return refs


def visual_prompt(p04: str, width: int, height: int) -> str:
    paragraphs = p04.split("\n\n")
    if len(paragraphs) != 3 or "pore</c>deposit</c>discontinuity</c>stain" not in paragraphs[0]:
        raise ValueError("P04 changed; review the visual query adaptation")
    return ("<image-1> Detect all the objects in the image that belong to the category set: <image-2>.\n\n"
            + paragraphs[1] + "\n\n" + paragraphs[2] + VISUAL_SUFFIX
            + SIZE.format(width=width, height=height))


def selected_test_images() -> list[Path]:
    images = sorted(path for path in (SPLIT / "images").iterdir()
                    if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(images) != 205:
        raise ValueError(f"Expected exactly 205 test images, found {len(images)}")
    for image in images:
        if not (SPLIT / "labels" / f"{image.stem}.txt").is_file():
            raise FileNotFoundError(f"Missing test label: {image.name}")
    return images


def previous_records(path: Path, images: list[Path], model_label: str, settings: dict) -> int:
    if not path.is_file():
        return 0
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) > len(images):
        raise ValueError(f"Too many existing records in {path}")
    for index, row in enumerate(rows):
        if (Path(row["image"]).name != images[index].name or row["model"] != model_label
                or row["generation"] != settings):
            raise ValueError(f"Existing predictions do not match this run at row {index}: {path}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=MODELS, required=True)
    parser.add_argument("--mode", choices=("none", "gt", "visual"), required=True)
    parser.add_argument("--prompt-version", choices=("P04", "P09"), default="P04")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-count", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--generation-mode", choices=("slow", "hybrid", "fast"), default="slow")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    model_path = MODELS[args.variant]
    for name in ("config.json", "done.txt", "loss_summary.json"):
        if not (model_path / name).is_file():
            parser.error(f"Training run is incomplete: missing {model_path / name}")
    images = selected_test_images()
    prompt_path = (REPO / "prompts/locany_p04_balanced.txt" if args.prompt_version == "P04"
                   else REPO / "prompts/locany_p09_inference.json")
    if args.prompt_version == "P04":
        p04 = prompt_path.read_text(encoding="utf-8").strip()
        p09 = None
    else:
        p04 = None
        p09 = json.loads(prompt_path.read_text(encoding="utf-8"))
        if set(p09) != {"none", "gt", "visual"}:
            parser.error("P09 must contain exactly the three inference modes")
        for mode, prompt in p09.items():
            if (not isinstance(prompt, str) or prompt.count("<image-1>") != 1
                    or prompt.count("<image-2>") != (mode != "none")):
                parser.error(f"Invalid P09 image placeholders for {mode}")
    gt_refs = load_gt_references() if args.mode == "gt" else None
    crop_refs = load_crop_references() if args.mode == "visual" else None
    settings = {"mode": args.generation_mode, "temperature": args.temperature,
                "top_p": args.top_p, "repetition_penalty": args.repetition_penalty,
                "max_new_tokens": args.max_new_tokens, "max_input_tokens": args.max_input_tokens,
                "seed": args.seed, "queries_per_image": 4 if args.mode == "visual" else 1}
    label = f"{args.variant.upper()} / {args.mode} / {args.prompt_version}"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    record_path = output / "predictions.jsonl"
    start = previous_records(record_path, images, label, settings)
    metadata_path = output / "run_metadata.json"
    metadata = {"model": label, "model_path": str(model_path), "test_split": str(SPLIT),
                "test_images": len(images), "test_used_for_training": False,
                "prompt_file": str(prompt_path), "prompt_sha256": digest(prompt_path),
                "reference_source": "fold0/train only" if args.mode != "none" else None,
                "generation": settings}
    if metadata_path.is_file() and json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
        raise ValueError(f"Run metadata changed; refusing to append to {record_path}")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render_names = {path.name for path in random.Random(args.seed).sample(images, min(args.render_count, len(images)))}
    if start == len(images):
        print(f"Already complete: {label}, {start} images")
        return
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True,
                                      torch_dtype=torch.bfloat16,
                                      attn_implementation="sdpa").to("cuda").eval()
    compare = output / "compare"
    compare.mkdir(exist_ok=True)
    with record_path.open("a", encoding="utf-8") as stream, torch.inference_mode():
        for image_path in images[start:]:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            truth = ground_truth(SPLIT / "labels" / f"{image_path.stem}.txt")
            references = {}
            prompts = {}
            answers = {}
            predictions = []
            rejected = 0
            if args.mode in ("none", "gt"):
                prompt = (p04 + SIZE.format(width=image.width, height=image.height)
                          if args.prompt_version == "P04" else p09[args.mode])
                example = None
                if args.mode == "gt":
                    ref = choose_gt_reference(gt_refs, f"test/images/{image_path.name}", args.seed)
                    with Image.open(ref["image"]) as opened:
                        example = opened.convert("RGB")
                    if args.prompt_version == "P04":
                        prompt += GT_SUFFIX
                    references["gt"] = ref["source_image"]
                torch.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)
                answer = predict(model, processor, tokenizer, image, args, prompt, example)
                if example is not None:
                    example.close()
                predictions, rejected = parse_predictions(answer)
                prompts["all"] = prompt
                answers["all"] = answer
            else:
                prompt = (visual_prompt(p04, image.width, image.height)
                          if args.prompt_version == "P04" else p09["visual"])
                for category in CLASSES:
                    ref = choose_crop_reference(crop_refs, category, image_path.name, args.seed)
                    with Image.open(ref["image"]) as opened:
                        example = opened.convert("RGB")
                    torch.manual_seed(args.seed)
                    torch.cuda.manual_seed_all(args.seed)
                    answer = predict(model, processor, tokenizer, image, args, prompt, example)
                    example.close()
                    parsed, invalid = parse_predictions(answer)
                    predictions.extend(parsed)
                    rejected += invalid
                    prompts[category] = prompt
                    answers[category] = answer
                    references[category] = {"source_image": ref["source_image"],
                                            "crop": ref["image"]}
            if image_path.name in render_names:
                left = draw_panel(image, truth, f"GT: {len(truth)}")
                right = draw_panel(image, predictions, f"Prediction: {len(predictions)}")
                canvas = Image.new("RGB", (image.width * 2, image.height))
                canvas.paste(left, (0, 0))
                canvas.paste(right, (image.width, 0))
                canvas.save(compare / image_path.name, quality=90)
                left.close()
                right.close()
                canvas.close()
            record = {"image": str(image_path), "model": label, "prompt": prompts,
                      "references": references, "answer": answers,
                      "ground_truth": truth, "predictions": predictions,
                      "task": "class_detection", "rejected_parsed_boxes": rejected,
                      "generation": settings}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            image.close()
            print(f"{label}: {image_path.name} GT={len(truth)} predicted={len(predictions)} rejected={rejected}", flush=True)
    print(f"Saved {len(images)} test predictions to {record_path}")


if __name__ == "__main__":
    main()
