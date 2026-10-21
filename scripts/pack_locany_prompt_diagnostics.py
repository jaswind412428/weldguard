"""Pack existing original-model prompt predictions and a few comparison JPGs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from evaluate_locany_predictions import count_matches


SOURCES = {
    "original": Path("runs/vlm/prompt_four_100_original"),
    "t1_size": Path("runs/vlm/prompt_four_100_t1_size"),
    "v4d_size": Path("runs/vlm/prompt_four_100_balanced_v4d_size"),
    "v5_compact_size": Path("runs/vlm/prompt_v5_compact_size_original_100"),
    "p05_gt_features_size": Path("runs/vlm/prompt_p05_gt_features_size_original_100"),
}
P05_OLD_PATH = Path("runs/vlm/prompt_v6_native_size_original_100")


def resolve_sources() -> dict[str, Path]:
    sources = SOURCES.copy()
    current = sources["p05_gt_features_size"] / "predictions.jsonl"
    old = P05_OLD_PATH / "predictions.jsonl"
    if not current.is_file() and old.is_file():
        sources["p05_gt_features_size"] = P05_OLD_PATH
        print(f"Using historical P05 output path: {old}")
    return sources


def read_records(directory: Path) -> dict[str, dict]:
    path = directory / "predictions.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; locate the existing predictions.jsonl before rerunning inference")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    by_name = {Path(row["image"]).name: row for row in records}
    if len(by_name) != len(records):
        raise ValueError(f"Duplicate image names in {path}")
    return by_name


def pick_examples(records: dict[str, dict], count: int) -> list[dict]:
    cases = []
    for name, row in records.items():
        truth, predictions = row["ground_truth"], row["predictions"]
        aware, _, _ = count_matches(truth, predictions, 0.5)
        located, _, _ = count_matches(truth, predictions, 0.5, ignore_class=True)
        cases.append({
            "image": name,
            "gt": len(truth),
            "predicted": len(predictions),
            "rejected": row["rejected_parsed_boxes"],
            "class_aware_tp": aware,
            "location_tp": located,
            "location_fp": len(predictions) - located,
            "location_fn": len(truth) - located,
            "located_wrong_class": located - aware,
        })
    groups = (
        sorted(cases, key=lambda c: (-c["location_fp"], c["image"])),
        sorted(cases, key=lambda c: (-c["located_wrong_class"], c["image"])),
        sorted(cases, key=lambda c: (-c["location_fn"], c["image"])),
        sorted(cases, key=lambda c: (-c["location_tp"], c["image"])),
    )
    selected = []
    seen = set()
    for index in range(count):
        for case in groups[index % len(groups)]:
            if case["image"] not in seen:
                selected.append(case)
                seen.add(case["image"])
                break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("results/locany_p05_original_prompt_diagnostics.zip"))
    parser.add_argument("--focus", choices=tuple(SOURCES), default="p05_gt_features_size")
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args()
    if args.examples < 1:
        parser.error("--examples must be positive")

    sources = resolve_sources()
    records = {label: read_records(path) for label, path in sources.items()}
    p05_prompt = Path("prompts/locany_p05_gt_features.txt").read_text(encoding="utf-8").strip()
    for row in records["p05_gt_features_size"].values():
        if not row.get("prompt", "").startswith(p05_prompt):
            raise ValueError(
                "Saved P05 predictions use a different prompt than prompts/locany_p05_gt_features.txt; "
                "do not label these results P05"
            )
    image_sets = [set(group) for group in records.values()]
    if any(image_set != image_sets[0] for image_set in image_sets[1:]):
        parser.error("The prediction files have different image sets")
    for name in image_sets[0]:
        truths = [group[name]["ground_truth"] for group in records.values()]
        if any(truth != truths[0] for truth in truths[1:]):
            parser.error(f"Ground truth differs for {name}")

    examples = pick_examples(records[args.focus], min(args.examples, len(image_sets[0])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, "w", ZIP_DEFLATED) as archive:
        for label, directory in sources.items():
            archive.write(directory / "predictions.jsonl", f"{label}/predictions.jsonl")
            for case in examples:
                source = directory / "compare" / case["image"]
                if not source.is_file():
                    raise FileNotFoundError(f"Missing comparison JPG: {source}")
                archive.write(source, f"{label}/compare/{case['image']}")
        archive.writestr("selected_examples.json", json.dumps(examples, indent=2) + "\n")
    print(f"Packed {len(image_sets[0])} prediction records per prompt and {len(examples)} JPGs per prompt")
    print(f"Saved {args.output} ({args.output.stat().st_size / 1024 / 1024:.2f} MiB)")
    print("Selected images:", ", ".join(case["image"] for case in examples))


if __name__ == "__main__":
    main()
