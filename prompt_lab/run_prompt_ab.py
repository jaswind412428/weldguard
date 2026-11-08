"""Run several weld prompts on the exact same unmarked image through OpenAI Responses."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw


LAB = Path(__file__).resolve().parent
REPO = LAB.parent
CLASSES = ("pore", "deposit", "discontinuity", "stain")
COLORS = {
    "pore": "#ed4848",
    "deposit": "#00a7c2",
    "discontinuity": "#f0a23b",
    "stain": "#36b46b",
}
SIZE_SUFFIX = (
    "\nThe input image is {width} pixels wide and {height} pixels high."
    " Return boxes using the model's normalized 0 to 1000 coordinate scale."
    " For this API comparison, each detection must have a <ref> tag containing"
    " one of the four class names followed by a <box> tag containing four separate"
    " integer coordinate tokens in x1, y1, x2, y2 order. Return detections only;"
    " do not output prose or a box for an absent class."
)
TAG_PATTERN = re.compile(
    r"<ref>\s*([^<]+?)\s*</ref>|<box>\s*<(\d{1,4})>\s*<(\d{1,4})>"
    r"\s*<(\d{1,4})>\s*<(\d{1,4})>\s*</box>",
    flags=re.IGNORECASE | re.DOTALL,
)


def parse_answer(answer: str) -> tuple[list[dict], int]:
    """Keep valid LocateAnything-style boxes, including multiple boxes per ref."""
    predictions: list[dict] = []
    rejected = 0
    current_class = None
    numeric_box_count = 0
    for match in TAG_PATTERN.finditer(answer):
        if match.group(1) is not None:
            current_class = match.group(1).strip().lower()
            continue
        numeric_box_count += 1
        box = [int(value) for value in match.groups()[1:]]
        if (
            current_class not in CLASSES
            or not (0 <= box[0] < box[2] <= 1000)
            or not (0 <= box[1] < box[3] <= 1000)
        ):
            rejected += 1
            continue
        predictions.append({"class": current_class, "box_1000": box})
    # Also count malformed or nonnumeric <box> entries, which the regex cannot parse.
    rejected += max(0, len(re.findall(r"<box\s*>", answer, re.I)) - numeric_box_count)
    return predictions, rejected


def read_yolo_gt(path: Path) -> list[dict]:
    """YOLO labels are used only for local scoring; they are never sent to the API."""
    truth = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected class cx cy width height")
        cls = int(fields[0])
        if cls not in range(len(CLASSES)):
            raise ValueError(f"{path}:{line_number}: unknown class {cls}")
        cx, cy, width, height = (float(value) for value in fields[1:])
        if width <= 0 or height <= 0:
            raise ValueError(f"{path}:{line_number}: nonpositive box size")
        box = [
            max(0.0, (cx - width / 2) * 1000),
            max(0.0, (cy - height / 2) * 1000),
            min(1000.0, (cx + width / 2) * 1000),
            min(1000.0, (cy + height / 2) * 1000),
        ]
        if not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000):
            raise ValueError(f"{path}:{line_number}: invalid YOLO box")
        truth.append({"class": CLASSES[cls], "box_1000": box})
    return truth


def iou(a: list[float], b: list[float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return overlap / (area_a + area_b - overlap) if overlap else 0.0


def score(predictions: list[dict], truth: list[dict], class_aware: bool) -> dict:
    """Maximum one-to-one matching at IoU >= 0.5."""
    edges = []
    for pred in predictions:
        options = [
            (iou(pred["box_1000"], gt["box_1000"]), index)
            for index, gt in enumerate(truth)
            if not class_aware or pred["class"] == gt["class"]
        ]
        edges.append([index for overlap, index in sorted(options, reverse=True) if overlap >= 0.5])
    matched_gt: dict[int, int] = {}

    def augment(pred_index: int, seen: set[int]) -> bool:
        for gt_index in edges[pred_index]:
            if gt_index in seen:
                continue
            seen.add(gt_index)
            if gt_index not in matched_gt or augment(matched_gt[gt_index], seen):
                matched_gt[gt_index] = pred_index
                return True
        return False

    for pred_index in range(len(predictions)):
        augment(pred_index, set())
    tp = len(matched_gt)
    fp, fn = len(predictions) - tp, len(truth) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate(predictions: list[dict], truth: list[dict]) -> dict:
    by_class = {}
    for name in CLASSES:
        class_predictions = [item for item in predictions if item["class"] == name]
        class_truth = [item for item in truth if item["class"] == name]
        by_class[name] = score(class_predictions, class_truth, class_aware=True)
    overall_tp = sum(item["TP"] for item in by_class.values())
    overall_fp = sum(item["FP"] for item in by_class.values())
    overall_fn = sum(item["FN"] for item in by_class.values())
    precision = overall_tp / (overall_tp + overall_fp) if overall_tp + overall_fp else 0.0
    recall = overall_tp / (overall_tp + overall_fn) if overall_tp + overall_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "by_class": by_class,
        "all": {"TP": overall_tp, "FP": overall_fp, "FN": overall_fn,
                "precision": precision, "recall": recall, "f1": f1},
        "all_no_class": score(predictions, truth, class_aware=False),
        "iou_threshold": 0.5,
    }


def draw_boxes(image: Image.Image, boxes: list[dict], title: str) -> Image.Image:
    panel = image.convert("RGB").copy()
    draw = ImageDraw.Draw(panel)
    width, height = panel.size
    stroke = max(2, round(min(width, height) / 150))
    for item in boxes:
        x1, y1, x2, y2 = item["box_1000"]
        rectangle = [x1 * width / 1000, y1 * height / 1000,
                     x2 * width / 1000, y2 * height / 1000]
        color = COLORS[item["class"]]
        draw.rectangle(rectangle, outline=color, width=stroke)
        draw.text((rectangle[0], max(26, rectangle[1] - 14)), item["class"], fill=color)
    draw.rectangle((0, 0, min(width, 320), 25), fill="black")
    draw.text((6, 5), title, fill="white")
    return panel


def save_panel(image_path: Path, output: Path, panels: list[tuple[str, list[dict]]]) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    canvas = Image.new("RGB", (image.width * len(panels), image.height), "white")
    for index, (title, boxes) in enumerate(panels):
        canvas.paste(draw_boxes(image, boxes, title), (index * image.width, 0))
    canvas.save(output)


def prompt_specs(values: list[str] | None) -> list[tuple[str, Path]]:
    if not values:
        return [
            ("P04", REPO / "prompts/locany_p04_balanced.txt"),
            ("P08", REPO / "prompts/locany_p08_detailed.txt"),
        ]
    result = []
    for value in values:
        if "=" not in value:
            raise ValueError("Use --prompt NAME=path/to/prompt.txt")
        name, path = value.split("=", 1)
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
            raise ValueError(f"Invalid prompt name: {name!r}")
        if name in [existing for existing, _ in result]:
            raise ValueError(f"Duplicate prompt name: {name}")
        result.append((name, Path(path).expanduser().resolve()))
    return result


def call_responses_api(
    *, api_key: str, model: str, prompt: str, image_url: str, detail: str,
    max_output_tokens: int
) -> dict:
    """Use the documented Responses endpoint without adding an SDK dependency."""
    body = {
        "model": model,
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": image_url, "detail": detail},
        ]}],
        "max_output_tokens": max_output_tokens,
        "store": False,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as incoming:
            return json.load(incoming)
    except urllib.error.HTTPError as error:
        detail_text = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API returned HTTP {error.code}: {detail_text}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="Unmarked JPG or PNG used for every prompt")
    parser.add_argument("--model", required=True, help="The same vision-capable OpenAI model for all prompts")
    parser.add_argument("--prompt", action="append", help="NAME=path; repeat to override default P04/P08")
    parser.add_argument("--gt-yolo", type=Path, help="Optional YOLO .txt GT, used only after inference")
    parser.add_argument("--detail", choices=("original", "high", "auto"), default="original")
    parser.add_argument("--max-output-tokens", type=int, default=2500)
    parser.add_argument("--output-root", type=Path, default=LAB / "runs")
    parser.add_argument("--dry-run", action="store_true", help="Write expanded prompts without calling the API")
    args = parser.parse_args()

    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        parser.error(f"Missing image: {image_path}")
    mime_type = mimetypes.guess_type(image_path.name)[0]
    if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
        parser.error("Use a JPG, PNG, or WEBP image")
    with Image.open(image_path) as image:
        width, height = image.size
    raw_image = image_path.read_bytes()
    specs = prompt_specs(args.prompt)
    prompts = {
        name: path.read_text(encoding="utf-8").strip()
        + SIZE_SUFFIX.format(width=width, height=height)
        for name, path in specs
    }
    key_file = LAB / ".api_key"
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key and key_file.is_file():
        api_key = key_file.read_text(encoding="utf-8").strip()
    if not args.dry_run and not api_key:
        parser.error("Set OPENAI_API_KEY or save your key in prompt_lab/.api_key")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = args.output_root.expanduser().resolve() / f"{image_path.stem}_{run_id}"
    run_dir.mkdir(parents=True)
    truth = read_yolo_gt(args.gt_yolo.expanduser().resolve()) if args.gt_yolo else None
    manifest = {
        "image": str(image_path),
        "image_sha256": hashlib.sha256(raw_image).hexdigest(),
        "image_size": {"width": width, "height": height},
        "model": args.model,
        "detail": args.detail,
        "coordinate_system": "LocateAnything native 0-1000 grid",
        "gt_yolo": str(args.gt_yolo.expanduser().resolve()) if args.gt_yolo else None,
        "dry_run": args.dry_run,
        "prompts": {},
    }
    for name, path in specs:
        (run_dir / f"{name}_prompt.txt").write_text(prompts[name] + "\n", encoding="utf-8")
        manifest["prompts"][name] = {"source": str(path), "status": "pending"}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.dry_run:
        print(f"Expanded prompts saved to {run_dir}; no API call made.")
        return

    image_url = f"data:{mime_type};base64,{base64.b64encode(raw_image).decode('ascii')}"
    panels = [("Original unmarked image", [])]
    if truth is not None:
        panels.append(("GT (local only)", truth))
    for name, _ in specs:
        response = call_responses_api(
            api_key=api_key,
            model=args.model,
            prompt=prompts[name],
            image_url=image_url,
            detail=args.detail,
            max_output_tokens=args.max_output_tokens,
        )
        answer = "\n".join(
            content.get("text", "")
            for item in response.get("output", [])
            if item.get("type") == "message"
            for content in item.get("content", [])
            if content.get("type") == "output_text"
        )
        predictions, rejected = parse_answer(answer)
        (run_dir / f"{name}_response.json").write_text(
            json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (run_dir / f"{name}_raw.txt").write_text(answer, encoding="utf-8")
        (run_dir / f"{name}_boxes.json").write_text(
            json.dumps({"predictions": predictions, "rejected_boxes": rejected}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if truth is not None:
            (run_dir / f"{name}_scores.json").write_text(
                json.dumps(evaluate(predictions, truth), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        panels.append((f"{name} predictions", predictions))
        manifest["prompts"][name] = {
            "source": manifest["prompts"][name]["source"],
            "status": response.get("status", "unknown"),
            "response_id": response.get("id"),
            "usage": response.get("usage"),
            "accepted_boxes": len(predictions),
            "rejected_boxes": rejected,
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"{name}: accepted={len(predictions)} rejected={rejected}")
    save_panel(image_path, run_dir / "comparison.png", panels)
    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()
