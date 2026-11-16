"""Package nine-way score evidence and a small set of rendered examples.

Raw test prediction records, model weights, and dataset files stay on the 4090.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


MODELS = ("v7", "v7-g", "v7-n")
MODES = ("none", "gt", "visual")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prompt-version", choices=("P04", "P09"), default="P04")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scores = args.run / "scores.csv"
    if not scores.is_file():
        parser.error(f"Missing nine-way scores: {scores}")
    with scores.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_models = {f"{variant.upper()} / {mode} / {args.prompt_version}"
                       for variant in MODELS for mode in MODES}
    if len(rows) != 45 or {row["model"] for row in rows} != expected_models:
        parser.error("scores.csv must contain nine model/input combinations and five class-aware rows each")
    files = [(scores, "scores.csv")]
    if (args.run / "scores.txt").is_file():
        files.append((args.run / "scores.txt", "scores.txt"))
    for name in ("counts.csv", "counts.txt", "p04_vs_p09.csv"):
        source = args.run / name
        if source.is_file():
            files.append((source, name))
    if args.prompt_version == "P09":
        prompt = Path(__file__).resolve().parents[1] / "prompts/locany_p09_inference.json"
        files.append((prompt, "prompts/locany_p09_inference.json"))
    for variant in MODELS:
        model_run = args.run.parent / f"locany_{variant.replace('-', '_')}_p04_xy"
        loss = model_run / "loss_summary.json"
        manifest = model_run / "data_manifest.json"
        for source in (loss, manifest):
            if not source.is_file():
                parser.error(f"Missing training evidence: {source}")
            files.append((source, f"training/{variant}/{source.name}"))
        for mode in MODES:
            folder = args.run / f"{variant}_{mode}"
            metadata = folder / "run_metadata.json"
            predictions = folder / "predictions.jsonl"
            if not metadata.is_file() or not predictions.is_file():
                parser.error(f"Missing result for {variant}/{mode}")
            count = sum(1 for line in predictions.open(encoding="utf-8") if line.strip())
            if count != 205:
                parser.error(f"Expected 205 predictions for {variant}/{mode}, got {count}")
            info = json.loads(metadata.read_text(encoding="utf-8"))
            if info.get("test_images") != 205 or info.get("test_used_for_training") is not False:
                parser.error(f"Invalid test metadata for {variant}/{mode}")
            files.append((metadata, f"inference/{variant}_{mode}/run_metadata.json"))
            for image in sorted((folder / "compare").glob("*.jpg")):
                files.append((image, f"examples/{variant}_{mode}/{image.name}"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for source, name in files:
            archive.write(source, name)
    print(f"Saved {args.output} ({len(files)} files; no raw 205-image records or model weights)")


if __name__ == "__main__":
    main()
