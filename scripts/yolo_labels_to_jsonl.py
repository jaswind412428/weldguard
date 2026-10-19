"""Convert saved YOLO prediction labels to the common WeldGuard JSONL format.

Accepts class cx cy width height [confidence] lines. Missing prediction files
mean no detections. The split must contain images/ and labels/ directories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


NAMES = {0: "pore", 1: "deposit", 2: "discontinuity", 3: "stain"}
SUFFIXES = {".jpg", ".jpeg", ".png"}


def boxes_from_file(path: Path, *, prediction: bool, min_confidence: float) -> list[dict]:
    if not path.is_file():
        if prediction:
            return []
        raise FileNotFoundError(path)
    boxes = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) not in ((5, 6) if prediction else (5,)):
            raise ValueError(f"{path}:{number}: expected YOLO box, got {len(fields)} fields")
        class_id = int(fields[0])
        if class_id not in NAMES:
            raise ValueError(f"{path}:{number}: unexpected class {class_id}")
        cx, cy, width, height = map(float, fields[1:5])
        confidence = float(fields[5]) if len(fields) == 6 else None
        if prediction and confidence is not None and confidence < min_confidence:
            continue
        x1, y1, x2, y2 = cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            # YOLO exports round normalized values, so tiny overflow is okay.
            if min(x1, y1) < -1e-4 or max(x2, y2) > 1 + 1e-4:
                raise ValueError(f"{path}:{number}: invalid normalized box")
        item = {"class": NAMES[class_id], "box_01": [max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)]}
        if item["box_01"][0] >= item["box_01"][2] or item["box_01"][1] >= item["box_01"][3]:
            raise ValueError(f"{path}:{number}: zero-area box")
        if confidence is not None:
            item["confidence"] = confidence
        boxes.append(item)
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--prediction-labels", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-confidence", type=float, default=0.25)
    args = parser.parse_args()
    if not 0 <= args.min_confidence <= 1:
        parser.error("--min-confidence must be between 0 and 1")
    images = sorted(path for path in (args.split / "images").iterdir() if path.suffix.lower() in SUFFIXES)
    if not images:
        parser.error(f"No images in {args.split / 'images'}")
    if len({path.stem for path in images}) != len(images):
        parser.error("Image stems are not unique")
    if not args.prediction_labels.is_dir():
        parser.error(f"Missing prediction directory: {args.prediction_labels}")
    unknown = {path.stem for path in args.prediction_labels.glob("*.txt")} - {path.stem for path in images}
    if unknown:
        parser.error(f"Prediction labels include {len(unknown)} unknown image stems")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for image in images:
            truth = boxes_from_file(args.split / "labels" / f"{image.stem}.txt", prediction=False, min_confidence=0)
            pred = boxes_from_file(args.prediction_labels / f"{image.stem}.txt", prediction=True,
                                   min_confidence=args.min_confidence)
            handle.write(json.dumps({"image": str(image), "model": args.model,
                                     "ground_truth": truth, "predictions": pred,
                                     "confidence_threshold": args.min_confidence}, ensure_ascii=False) + "\n")
    print(f"Saved {len(images)} images to {args.output}")


if __name__ == "__main__":
    main()
