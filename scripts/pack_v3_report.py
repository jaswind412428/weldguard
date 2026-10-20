"""Collect completed V3 evidence for a report, without weights or source images."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from evaluate_locany_predictions import count_matches, scores
from render_four_model_test_examples import read_records, render


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.home() / "weldguard")
    args = parser.parse_args()
    root = args.project.resolve()
    train = root / "runs/vlm/locany_fold0_5epoch_equiv"
    sources = {
        "V2": root / "runs/vlm/locany_fold0_v2_test_slow/predictions.jsonl",
        "V3": root / "runs/vlm/locany_v3_test_slow/predictions.jsonl",
    }
    records = {name: read_records(path) for name, path in sources.items()}
    expected = {
        p.name for p in (root / "datasets/weld-dataset/kfold_high/test/images").iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    if len(expected) != 205:
        raise ValueError(f"Expected 205 test images, found {len(expected)}")
    for label, data in records.items():
        if set(data) != expected:
            raise ValueError(f"{label} is incomplete or has a different test image set. Wait for inference to finish.")
    for name in expected:
        if records["V2"][name]["ground_truth"] != records["V3"][name]["ground_truth"]:
            raise ValueError(f"Ground truth differs: {name}")
    generation = records["V2"][sorted(expected)[0]]["generation"]
    prompt = records["V2"][sorted(expected)[0]]["prompt"]
    for data in records.values():
        if any(row.get("generation") != generation or row.get("prompt") != prompt for row in data.values()):
            raise ValueError("V2/V3 generation settings or prompts differ")
    state = json.loads((train / "trainer_state.json").read_text())
    if state.get("global_step") != 3265:
        raise ValueError(f"V3 final global_step is {state.get('global_step')}, expected 3265")
    comparisons = root / "runs/vlm/locany_v3_test_slow/compare"
    if any(not (comparisons / name).is_file() for name in expected):
        raise ValueError("Some V3 comparison images are missing")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    bundle = root / "runs/report_exports" / f"v3_report_{stamp}"
    bundle.mkdir(parents=True, exist_ok=False)
    for label, path in sources.items():
        shutil.copy2(path, bundle / f"{label}_predictions.jsonl")
    training_out = bundle / "training"
    training_out.mkdir()
    for name in ("train_results.json", "all_results.json", "trainer_state.json",
                 "train.log", "config.json"):
        if (train / name).is_file():
            shutil.copy2(train / name, training_out / name)
    recipe = root / "datasets/derived/locany_full_v2/full_recipe.json"
    if recipe.is_file():
        shutil.copy2(recipe, training_out / "full_recipe.json")
    loss_rows = [
        {k: row.get(k, "") for k in ("step", "epoch", "loss", "grad_norm", "learning_rate")}
        for row in state.get("log_history", []) if "loss" in row
    ]
    if loss_rows:
        write_csv(training_out / "loss_history.csv", loss_rows)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True, check=True)
    (training_out / "environment.txt").write_text(freeze.stdout, encoding="utf-8")
    shutil.copytree(comparisons, bundle / "V3_all_205_comparisons")

    evaluator = Path(__file__).with_name("evaluate_locany_predictions.py")
    evaluation = subprocess.run([
        sys.executable, str(evaluator), *map(str, sources.values()),
        "--iou", "0.5", "--output", str(bundle / "v2_v3_scores.csv"),
    ], capture_output=True, text=True, check=True)
    (bundle / "v2_v3_scores.txt").write_text(evaluation.stdout, encoding="utf-8")
    print(evaluation.stdout)
    # Include all existing model records to permit a common five-model table.
    optional = {
        "YOLO11s": "runs/detect/runs/detect/yolo11s_test_fixed025/predictions.jsonl",
        "YOLO26l": "runs/detect/runs/detect/yolo26l_test_fixed025/predictions.jsonl",
        "Author_YOLOv7": "runs/author_test/high_fold0_aug_1280_fixed025/predictions.jsonl",
    }
    missing = []
    for label, relative in optional.items():
        source = root / relative
        if source.is_file():
            shutil.copy2(source, bundle / f"{label}_predictions.jsonl")
        else:
            missing.append(relative)
    if not missing:
        subprocess.run([
            sys.executable, str(evaluator),
            *[str(bundle / f"{label}_predictions.jsonl") for label in (*optional, "V2", "V3")],
            "--iou", "0.5", "--output", str(bundle / "five_models_scores.csv"),
        ], check=True)

    per_image = []
    for name in sorted(expected):
        row = {"image": name, "gt": len(records["V3"][name]["ground_truth"])}
        for label in ("V2", "V3"):
            record = records[label][name]
            row[f"{label}_predicted"] = len(record["predictions"])
            row[f"{label}_rejected"] = record.get("rejected_parsed_boxes", 0)
            metrics = scores(*count_matches(record["ground_truth"], record["predictions"], 0.5))
            row.update({f"{label}_{k}": v for k, v in metrics.items()})
        row["F1_change"] = row["V3_f1"] - row["V2_f1"]
        per_image.append(row)
    write_csv(bundle / "per_image_counts.csv", per_image)
    selected = {name: "Existing report example" for name in
                ("25_105.jpg", "25_106.jpg", "25_108.jpg", "25_128.jpg") if name in expected}
    for row in sorted(per_image, key=lambda r: (-r["F1_change"], r["image"]))[:2]:
        selected.setdefault(row["image"], "Two largest per-image F1 changes")
    for row in sorted(per_image, key=lambda r: (r["F1_change"], r["image"]))[:2]:
        selected.setdefault(row["image"], "Two smallest per-image F1 changes")
    for name in selected:
        image_path = root / "datasets/weld-dataset/kfold_high/test/images" / name
        with Image.open(image_path) as f:
            image = f.convert("RGB")
        boxes = {"GT": records["V3"][name]["ground_truth"]}
        boxes.update({label: records[label][name]["predictions"] for label in ("V2", "V3")})
        for label, items in boxes.items():
            target = bundle / "selected_examples" / label
            target.mkdir(parents=True, exist_ok=True)
            render(image, items, label).save(target / name, quality=94)
    (bundle / "selection.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    (bundle / "README.txt").write_text(
        "V3 report evidence; 205 fixed test images; class-aware IoU >= 0.5.\n"
        "Training: final global_step 3265. Epoch field is a streaming trainer value.\n"
        "Actual arguments: inspect training/train.log; unspecified flags are not verified.\n"
        "Current environment.txt describes the export environment, not guaranteed historical state.\n"
        "Examples include prior report images plus two largest/smallest per-image F1 changes.\n"
        "Selected examples do not represent average performance. Use full-set metrics.\n"
        "Includes derived GT/prediction overlays and prediction metadata; no raw dataset or model weights.\n"
        f"Generation settings: {json.dumps(generation)}\n"
        f"Missing optional model results: {missing}\n", encoding="utf-8")
    # Include evaluator/renderer code to document how the evidence was produced.
    for name in ("pack_v3_report.py", "evaluate_locany_predictions.py",
                 "render_four_model_test_examples.py", "preview_locany_val.py"):
        source = Path(__file__).with_name(name)
        if source.is_file():
            target = bundle / "scripts" / name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(source, target)
    manifest = {
        str(p.relative_to(bundle)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(bundle.rglob("*")) if p.is_file()
    }
    (bundle / "sha256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive = root / "results" / f"{bundle.name}.zip"
    archive.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(bundle.rglob("*")):
            if p.is_file():
                z.write(p, str(p.relative_to(bundle)))
    print(f"\nREPORT_ARCHIVE={archive.relative_to(root)}")
    print(f"Archive size: {archive.stat().st_size / 1024**2:.1f} MiB")
    if archive.stat().st_size >= 95 * 1024**2:
        raise ValueError("Archive is too large for normal GitHub upload; do not push it.")
    print("Ready to upload this archive only after checking git status.")


if __name__ == "__main__":
    main()
