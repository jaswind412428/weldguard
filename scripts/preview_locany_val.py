"""Save LocateAnything predictions beside YOLO ground truth on fold0 validation images.

Run once per model from eagle-vlm/Embodied with .venv-vlm active. This is a
qualitative preview, not a validation score. Raw model text is retained.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor, AutoTokenizer


NAMES = {0: "pore", 1: "deposit", 2: "discontinuity", 3: "stain"}
COLORS = {"pore": "#ff4040", "deposit": "#00b8d9", "discontinuity": "#f5a623", "stain": "#21b573"}
PROMPT = (
    "<image-1> Locate all the instances that matches the following description: "
    "pore</c>deposit</c>discontinuity</c>stain."
)
OUTPUT_PATTERN = re.compile(
    r"<ref>(.*?)</ref>|<box><(\d{1,4})><(\d{1,4})><(\d{1,4})><(\d{1,4})></box>",
    flags=re.DOTALL,
)


def ground_truth(label_path: Path) -> list[dict]:
    result = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO label in {label_path}: {line}")
        class_id = int(fields[0])
        cx, cy, width, height = map(float, fields[1:])
        if class_id not in NAMES or width <= 0 or height <= 0:
            raise ValueError(f"Invalid YOLO label in {label_path}: {line}")
        box = [
            max(0.0, cx - width / 2),
            max(0.0, cy - height / 2),
            min(1.0, cx + width / 2),
            min(1.0, cy + height / 2),
        ]
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError(f"Invalid YOLO box in {label_path}: {line}")
        result.append({"class": NAMES[class_id], "box_01": box})
    return result


def parse_predictions(answer: str) -> tuple[list[dict], int]:
    predictions = []
    rejected = 0
    current_class = None
    for match in OUTPUT_PATTERN.finditer(answer):
        if match.group(1) is not None:
            current_class = match.group(1).strip().lower()
            continue
        coordinates = list(map(int, match.groups()[1:]))
        if (
            current_class not in COLORS
            or any(value > 1000 for value in coordinates)
            or coordinates[0] >= coordinates[2]
            or coordinates[1] >= coordinates[3]
        ):
            rejected += 1
            continue
        predictions.append(
            {"class": current_class, "box_01": [value / 1000 for value in coordinates]}
        )
    return predictions, rejected


def choose_images(image_dir: Path, label_dir: Path, limit: int) -> list[Path]:
    images = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise ValueError(f"No validation images in {image_dir}")
    labels = {path: ground_truth(label_dir / f"{path.stem}.txt") for path in images}
    chosen = []
    uncovered = set(NAMES.values())
    for path in images:
        present = {item["class"] for item in labels[path]}
        if present & uncovered:
            chosen.append(path)
            uncovered -= present
        if not uncovered or len(chosen) >= limit:
            break
    chosen_set = set(chosen)
    chosen.extend(path for path in images if path not in chosen_set and len(chosen) < limit)
    return chosen[:limit]


def draw_panel(image: Image.Image, boxes: list[dict], title: str) -> Image.Image:
    panel = image.copy()
    draw = ImageDraw.Draw(panel)
    width, height = panel.size
    stroke = max(2, round(min(width, height) / 150))
    for item in boxes:
        x1, y1, x2, y2 = item["box_01"]
        box = (round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height))
        color = COLORS[item["class"]]
        draw.rectangle(box, outline=color, width=stroke)
        draw.text((box[0], max(0, box[1] - 12)), item["class"], fill=color)
    draw.rectangle((0, 0, min(width, 290), 25), fill="black")
    draw.text((6, 5), title, fill="white")
    return panel


def predict(model, processor, tokenizer, image: Image.Image, args, prompt_text: str) -> str:
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt_text}, {"type": "image", "image": image}
    ]}]
    rendered = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = processor.process_vision_info(messages)
    inputs = processor(text=[rendered], images=images, videos=videos, return_tensors="pt").to("cuda")
    response = model.generate(
        pixel_values=inputs["pixel_values"].to(torch.bfloat16),
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_grid_hws=inputs.get("image_grid_hws"),
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        generation_mode=args.generation_mode,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        verbose=False,
    )
    return response[0] if isinstance(response, tuple) else response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face ID or trained model directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path,
                        help="UTF-8 text file used instead of the original prompt")
    parser.add_argument("--one-class-at-a-time", action="store_true",
                        help="Query each of the four classes separately, then combine predictions")
    parser.add_argument("--include-image-size", action="store_true",
                        help="Include image pixel dimensions in each prompt")
    parser.add_argument("--split", type=Path, default=Path.home() / "weldguard/datasets/weld-dataset/kfold_high/fold0/val")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--generation-mode", choices=("hybrid", "slow", "fast"), default="hybrid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--images", nargs="*", help="Exact validation filenames to inspect")
    args = parser.parse_args()
    if args.one_class_at_a_time and args.prompt_file:
        parser.error("--one-class-at-a-time cannot be combined with --prompt-file")
    if args.prompt_file:
        try:
            args.prompt_text = args.prompt_file.read_text(encoding="utf-8").strip()
        except OSError as error:
            parser.error(f"Cannot read prompt file: {error}")
        if not args.prompt_text:
            parser.error("Prompt file is empty")
        if "<image-1>" not in args.prompt_text:
            parser.error("Prompt file must contain <image-1>")
    else:
        args.prompt_text = PROMPT
    if args.limit < 1 or args.max_new_tokens < 1:
        parser.error("--limit and --max-new-tokens must be positive")
    if args.temperature < 0 or not 0 < args.top_p <= 1 or args.repetition_penalty <= 0:
        parser.error("Invalid generation parameters")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if args.images:
        selected = [args.split / "images" / name for name in args.images]
        for image_path in selected:
            if image_path.name != str(image_path.relative_to(args.split / "images")) or not image_path.is_file():
                parser.error(f"Missing validation image: {image_path}")
    else:
        selected = choose_images(args.split / "images", args.split / "labels", args.limit)
    output = args.output.expanduser()
    image_output = output / "compare"
    image_output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as records:
        for image_path in selected:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            truth = ground_truth(args.split / "labels" / f"{image_path.stem}.txt")
            if args.one_class_at_a_time:
                answers = {}
                prompts = {}
                predictions = []
                rejected = 0
                for category in NAMES.values():
                    prompt_text = (
                        "<image-1> Locate all the instances that matches the following "
                        f"description: {category}."
                    )
                    if args.include_image_size:
                        prompt_text += (
                            f" The image is {image.width} pixels wide and {image.height} pixels high."
                            " Return boxes using the model's normalized 0 to 1000 coordinate scale."
                        )
                    answer = predict(model, processor, tokenizer, image, args, prompt_text)
                    parsed, invalid = parse_predictions(answer)
                    # Keep only boxes explicitly named as the requested class.
                    # Other classes generated in this call are counted as rejected.
                    accepted = [item for item in parsed if item["class"] == category]
                    rejected += invalid + len(parsed) - len(accepted)
                    predictions.extend(accepted)
                    answers[category] = answer
                    prompts[category] = prompt_text
            else:
                prompt_text = args.prompt_text
                if args.include_image_size:
                    prompt_text += (
                        f" The image is {image.width} pixels wide and {image.height} pixels high."
                        " Return boxes using the model's normalized 0 to 1000 coordinate scale."
                    )
                answer = predict(model, processor, tokenizer, image, args, prompt_text)
                predictions, rejected = parse_predictions(answer)
            left = draw_panel(image, truth, f"Ground truth: {len(truth)}")
            right = draw_panel(image, predictions, f"Prediction: {len(predictions)}")
            comparison = Image.new("RGB", (image.width * 2, image.height))
            comparison.paste(left, (0, 0))
            comparison.paste(right, (image.width, 0))
            comparison.save(image_output / image_path.name, quality=95)
            record = {
                "image": str(image_path),
                "model": args.model + (
                    " (one class per query, with image size)" if args.one_class_at_a_time and args.include_image_size
                    else " (one class per query)" if args.one_class_at_a_time
                    else " (all classes, with image size)" if args.include_image_size
                    else ""
                ),
                "prompt": prompts if args.one_class_at_a_time else prompt_text,
                "answer": answers if args.one_class_at_a_time else answer,
                "ground_truth": truth, "predictions": predictions,
                "rejected_parsed_boxes": rejected,
                "generation": {
                    "mode": args.generation_mode, "temperature": args.temperature,
                    "top_p": args.top_p, "repetition_penalty": args.repetition_penalty,
                    "max_new_tokens": args.max_new_tokens, "seed": args.seed,
                    "queries_per_image": 4 if args.one_class_at_a_time else 1,
                },
            }
            records.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.flush()
            print(f"{image_path.name}: GT={len(truth)} predicted={len(predictions)} rejected={rejected}", flush=True)
    print(f"Saved {len(selected)} comparisons to {image_output}")


if __name__ == "__main__":
    main()
