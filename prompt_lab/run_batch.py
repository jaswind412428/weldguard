"""Compare P04/P08 on a fixed subset of the transferred fold0 validation images."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path


LAB = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = LAB / "transfer/prompt_lab_val100.zip"


def unpack_selected(archive_path: Path, destination: Path) -> list[str]:
    """Extract only declared images and labels; reject unexpected archive paths."""
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        names = manifest["images"]
        if manifest.get("split") != "fold0/val" or len(names) != 100 or len(set(names)) != 100:
            raise ValueError("Expected exactly 100 unique fold0 validation images")
        expected = {"manifest.json"}
        for name in names:
            if Path(name).name != name or Path(name).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                raise ValueError(f"Unsafe image name: {name}")
            expected.add(f"images/{name}")
            expected.add(f"labels/{Path(name).stem}.txt")
        if set(archive.namelist()) != expected:
            raise ValueError("Archive contents differ from manifest")
        for member in sorted(expected - {"manifest.json"}):
            target = destination / member
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
        (destination / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return names


def combine_counts(score_files: list[Path]) -> dict:
    data = [json.loads(path.read_text(encoding="utf-8")) for path in score_files]
    keys = ["pore", "deposit", "discontinuity", "stain", "all", "all_no_class"]
    combined = {}
    for key in keys:
        items = [entry["by_class"][key] if key in entry["by_class"] else entry[key] for entry in data]
        tp, fp, fn = (sum(item[field] for item in items) for field in ("TP", "FP", "FN"))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        combined[key] = {
            "TP": tp, "FP": fp, "FN": fn,
            "precision": precision, "recall": recall, "f1": f1,
        }
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--limit", type=int, choices=range(1, 101), default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", required=True)
    parser.add_argument("--detail", choices=("original", "high", "auto"), default="original")
    parser.add_argument("--batch-dir", type=Path, help="Reuse this directory to resume an interrupted batch")
    args = parser.parse_args()
    archive_path = args.archive.expanduser().resolve()
    if not archive_path.is_file():
        parser.error(f"Missing transferred archive: {archive_path}")
    if not os.environ.get("OPENAI_API_KEY") and not (LAB / ".api_key").is_file():
        parser.error("Set OPENAI_API_KEY or create prompt_lab/.api_key before running the batch")

    local_data = LAB / "transfer/val100"
    names = unpack_selected(archive_path, local_data)
    selected = names if args.limit == 100 else sorted(random.Random(args.seed).sample(names, args.limit))
    batch_dir = (
        args.batch_dir.expanduser().resolve() if args.batch_dir
        else LAB / "runs" / f"batch_val{args.limit}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_manifest_path = batch_dir / "batch_manifest.json"
    batch_manifest = {
        "archive": str(archive_path), "selected_images": selected,
        "model": args.model, "detail": args.detail, "seed": args.seed,
        "split": "fold0/val", "iou_threshold": 0.5,
    }
    if batch_manifest_path.is_file():
        previous = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
        if previous != batch_manifest:
            raise ValueError("Existing batch directory has different settings or image selection")
    else:
        batch_manifest_path.write_text(
            json.dumps(batch_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    for index, name in enumerate(selected, 1):
        stem = Path(name).stem
        completed = [
            path for path in batch_dir.glob(f"{stem}_*")
            if (path / "comparison.png").is_file()
            and (path / "P04_scores.json").is_file()
            and (path / "P08_scores.json").is_file()
        ]
        if completed:
            print(f"[{index}/{len(selected)}] {name}: already complete")
            continue
        print(f"[{index}/{len(selected)}] {name}: querying P04 and P08", flush=True)
        subprocess.run(
            [sys.executable, str(LAB / "run_prompt_ab.py"),
             "--image", str(local_data / "images" / name),
             "--gt-yolo", str(local_data / "labels" / f"{stem}.txt"),
             "--model", args.model, "--detail", args.detail,
             "--output-root", str(batch_dir)],
            check=True,
        )

    rows = []
    summary = {}
    for prompt_name in ("P04", "P08"):
        files = []
        for name in selected:
            candidates = sorted(
                path / f"{prompt_name}_scores.json"
                for path in batch_dir.glob(f"{Path(name).stem}_*")
                if (path / "comparison.png").is_file()
                and (path / f"{prompt_name}_scores.json").is_file()
            )
            if len(candidates) != 1:
                raise ValueError(f"Expected one complete {prompt_name} result for {name}")
            files.append(candidates[0])
        summary[prompt_name] = combine_counts(files)
        for class_name, counts in summary[prompt_name].items():
            rows.append({"prompt": prompt_name, "class": class_name, **counts})
    (batch_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (batch_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("prompt", "class", "TP", "FP", "FN",
                                                    "precision", "recall", "f1"))
        writer.writeheader()
        writer.writerows(rows)
    for name, scores in summary.items():
        print(f"{name}: classified F1={scores['all']['f1']:.3f}, "
              f"localization F1={scores['all_no_class']['f1']:.3f}")
    print(f"Batch summary saved to {batch_dir}")


if __name__ == "__main__":
    main()
