"""Validation-only reference ablations, with complete answers and input traces."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageOps
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer

from evaluate_locany_predictions import count_matches
from locany_image_example import choose_reference as choose_gt
from locany_reference_diagnostics import parse_answer, trace_grid_to_list
from locany_visual_prompt import CLASSES, choose_reference as choose_crop
from prepare_locany_reference_alignment import OUT, choose_reference, sha
from preview_locany_v7_nine_test import MODELS, ROOT, REPO, load_crop_references, load_gt_references
from preview_locany_val import draw_panel, ground_truth


def pixel_hash(image):
    return hashlib.sha256(str(image.size).encode() + image.convert("RGB").tobytes()).hexdigest()


def predict_traced(model, processor, tokenizer, target, example, prompt, args):
    # Retain the legacy message construction to isolate prompt/reference changes.
    content = [{"type": "text", "text": prompt}, {"type": "image", "image": target}]
    if example is not None:
        content.append({"type": "image", "image": example})
    messages = [{"role": "user", "content": content}]
    rendered = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = processor.process_vision_info(messages)
    if images is None or len(images) != (2 if example is not None else 1):
        raise ValueError("Processor did not retain the expected target/reference image count")
    inputs = processor(text=[rendered], images=images, videos=videos, return_tensors="pt")
    input_tokens = int(inputs["input_ids"].shape[-1])
    if input_tokens > args.max_input_tokens:
        raise ValueError(f"Input too long: {input_tokens} > {args.max_input_tokens}; no truncation allowed")
    pixels = inputs["pixel_values"].detach().cpu().contiguous()
    pixel_digest = hashlib.sha256(pixels.view(torch.uint8).numpy().tobytes()).hexdigest()
    grid = inputs.get("image_grid_hws")
    trace = {"target_pixel_sha256": pixel_hash(target),
        "reference_pixel_sha256": pixel_hash(example) if example is not None else None,
        "source_image_sizes": [list(target.size)] + ([list(example.size)] if example is not None else []),
        "processed_images": len(images), "input_tokens": input_tokens,
        "pixel_values_shape": list(pixels.shape), "pixel_values_dtype": str(pixels.dtype),
        "processed_pixel_sha256": pixel_digest,
        "image_grid_hws": trace_grid_to_list(grid),
        "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest()}
    inputs = inputs.to("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    response = model.generate(pixel_values=inputs["pixel_values"].to(torch.bfloat16),
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
        image_grid_hws=inputs.get("image_grid_hws"), tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens, use_cache=True, generation_mode="slow",
        temperature=0.7, top_p=0.9, repetition_penalty=1.1, verbose=False)
    answer = response[0] if isinstance(response, tuple) else response
    if not isinstance(answer, str):
        raise TypeError(f"Unexpected generate result: {type(answer)}")
    encoded = len(tokenizer.encode(answer, add_special_tokens=False))
    trace.update({"generated_tokens_reencoded": encoded,
                  "ended_with_eos": answer.rstrip().endswith(("<|im_end|>", "<|endoftext|>")),
                  "at_or_above_token_limit_reencoded": encoded >= args.max_new_tokens})
    return answer, trace


def comparison(target, truth, predictions, references, path):
    canvas = Image.new("RGB", (1920, 320), "#161b22")
    for index, (boxes, title) in enumerate(((truth, f"Target GT: {len(truth)}"),
                                             (predictions, f"Prediction: {len(predictions)}"))):
        panel = draw_panel(target, boxes, title)
        thumb = ImageOps.contain(panel, (640, 280))
        canvas.paste(thumb, (index * 640, 30 + (280 - thumb.height) // 2))
        panel.close()
        thumb.close()
    draw = ImageDraw.Draw(canvas)
    draw.text((1285, 5), "TRAIN references only", fill="white")
    for index, (category, ref) in enumerate(references.items()):
        with Image.open(ref["image"]) as im:
            thumb = ImageOps.contain(im.convert("RGB"), (300, 115))
        left, top = 1285 + index % 2 * 315, 30 + index // 2 * 140
        draw.text((left, top), category, fill="white")
        canvas.paste(thumb, (left, top + 18))
        thumb.close()
    canvas.save(path, quality=92)
    canvas.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=(*MODELS, "aligned"), required=True)
    parser.add_argument("--mode", choices=("none", "gt", "visual"), required=True)
    parser.add_argument("--prompt", choices=("P10", "P11"), default="P10")
    parser.add_argument("--reference-bank", choices=("original", "context"), default="original")
    parser.add_argument("--limit", type=int, choices=(20, 164), default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", help="Saved checkpoint-N inside the aligned training run")
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.prompt == "P11" and args.mode != "visual":
        parser.error("P11 is a single-class visual query; use P10 for none/GT baselines")
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["configuration"]["seed"] != args.seed:
        parser.error("Seed differs from reference preparation")
    model_path = (REPO / "runs/vlm/locany_reference_aligned_p11_epoch1" if args.variant == "aligned"
                  else MODELS[args.variant])
    if not all((model_path / name).is_file() for name in ("config.json", "done.txt", "loss_summary.json")):
        parser.error(f"Incomplete model: {model_path}")
    model_root = model_path
    if args.checkpoint:
        if args.variant != "aligned" or not re.fullmatch(r"checkpoint-\d+", args.checkpoint):
            parser.error("--checkpoint requires aligned and a checkpoint-N directory name")
        model_path = model_path / args.checkpoint
        if not (model_path / "config.json").is_file():
            parser.error(f"Missing checkpoint config: {model_path}")
    paths = {"P10": REPO / "prompts/locany_p10_task_format_features.json",
             "P11": REPO / "prompts/locany_p11_class_reference.txt"}
    prompt_text = paths[args.prompt].read_text(encoding="utf-8").strip()
    p10 = json.loads(prompt_text) if args.prompt == "P10" else None
    images = [ROOT / "fold0/val/images" / name for name in manifest["validation_order"][:args.limit]]
    if len(images) != args.limit or len(set(images)) != args.limit:
        parser.error("Invalid fixed validation selection")
    for image in images:
        if not image.is_file():
            raise FileNotFoundError(image)
    crop_refs = (load_crop_references() if args.reference_bank == "original" else manifest["reference_bank"]) if args.mode == "visual" else []
    gt_refs = (load_gt_references("v8") if args.reference_bank == "original" else manifest["gt_references"]) if args.mode == "gt" else []
    label = f"{args.variant.upper()} / {args.mode} / {args.prompt} / {args.reference_bank} / val{args.limit}"
    if args.checkpoint:
        label += f" / {args.checkpoint}"
    metadata = {"model": label, "model_path": str(model_path), "prompt_sha256": sha(paths[args.prompt]),
        "model_config_sha256": sha(model_path / "config.json"), "seed": args.seed,
        "manifest_sha256": sha(manifest_path), "images": [p.name for p in images],
        "split": "fold0/val", "test_used": False, "generation_mode": "slow",
        "temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.1,
        "max_new_tokens": args.max_new_tokens, "max_input_tokens": args.max_input_tokens}
    args.output.mkdir(parents=True, exist_ok=True)
    meta_path, record_path = args.output / "run_metadata.json", args.output / "predictions.jsonl"
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Run settings changed; choose a new output directory")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    existing = [json.loads(line) for line in record_path.read_text(encoding="utf-8").splitlines() if line.strip()] if record_path.exists() else []
    if len(existing) > len(images) or any(r["model"] != label or Path(r["image"]).name != images[i].name for i, r in enumerate(existing)):
        raise ValueError("Existing records do not match this validation run")
    if len(existing) == len(images):
        print(f"Already complete: {label}")
        return
    if not torch.cuda.is_available():
        parser.error("Run model inference in the 4090 VLM environment")
    processor = AutoProcessor.from_pretrained(str(model_root), trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_root), trust_remote_code=True)
    if args.checkpoint:
        # Intermediate checkpoints do not contain all copied inference .py files.
        # Load the architecture from the completed run and weights from checkpoint.
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        config = AutoConfig.from_pretrained(str(model_root), trust_remote_code=True)
        model_class = get_class_from_dynamic_module(config.auto_map["AutoModel"], str(model_root))
        model = model_class.from_pretrained(str(model_path), config=config,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    else:
        model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    compare = args.output / "compare"
    compare.mkdir(exist_ok=True)
    with record_path.open("a", encoding="utf-8") as stream, torch.inference_mode():
        for index, image_path in enumerate(images[len(existing):], start=len(existing)):
            with Image.open(image_path) as opened:
                target = opened.convert("RGB")
            truth = ground_truth(ROOT / "fold0/val/labels" / f"{image_path.stem}.txt")
            queries, references, predictions, answers = [], {}, [], {}
            for category in (CLASSES if args.mode == "visual" else (None,)):
                example, ref = None, None
                if args.mode == "visual":
                    choose = choose_crop if args.reference_bank == "original" else choose_reference
                    ref = choose(crop_refs, category, f"fold0/val/images/{image_path.name}", args.seed)
                elif args.mode == "gt":
                    ref = choose_gt(gt_refs, f"fold0/val/images/{image_path.name}", args.seed)
                if ref:
                    if not (ROOT / ref["source_image"]).resolve().is_relative_to((ROOT / "fold0/train/images").resolve()):
                        raise ValueError("Reference is not from training split")
                    with Image.open(ref["image"]) as opened:
                        example = opened.convert("RGB")
                    references[category or "gt"] = ref
                prompt = (p10[args.mode].format(width=target.width, height=target.height) if p10 else
                          prompt_text.format(category=category, width=target.width, height=target.height))
                answer, trace = predict_traced(model, processor, tokenizer, target, example, prompt, args)
                parsed, rejected = parse_answer(answer)
                queries.append({"requested_class": category, "prompt": prompt, "answer": answer,
                    "predictions": parsed, "rejected_numeric": rejected, "reference": ref, "trace": trace})
                predictions.extend(parsed)
                answers[category or "all"] = answer
                if example is not None:
                    example.close()
            record = {"image": str(image_path), "image_size": list(target.size), "model": label,
                "ground_truth": truth, "predictions": predictions, "queries": queries,
                "answer": answers, "references": references, "generation": metadata,
                "rejected_parsed_boxes": sum(q["rejected_numeric"] for q in queries)}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            if index < 6:
                comparison(target, truth, predictions, references, compare / image_path.name)
            tp, fp, fn = count_matches(truth, predictions, 0.5)
            print(f"{label}: {image_path.name} TP={tp} FP={fp} FN={fn}", flush=True)
            target.close()


if __name__ == "__main__":
    main()
