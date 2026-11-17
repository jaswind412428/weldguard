"""Package result JSONL/CSV, trace summaries and comparison JPGs, not datasets/weights."""
from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=REPO / "runs/vlm/reference_alignment_val20")
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to((REPO / "runs/vlm").resolve()) or not run.is_dir():
        parser.error("Run must be an existing project runs/vlm directory")
    output = REPO / "results" / f"{run.name}_report.zip"
    output.parent.mkdir(exist_ok=True)
    allowed = {".json", ".jsonl", ".csv", ".jpg"}
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(run.rglob("*")):
            if path.is_file() and path.suffix.lower() in allowed:
                archive.write(path, path.relative_to(run).as_posix())
        archive.writestr("README.txt", "Validation pilot plus error diagnosis of historical test predictions. "
            "All scores remain class-aware at IoU 0.5. Dedup/query filtering are separately labeled ablations. "
            "No weights, raw training dataset, API credentials, or pixel tensors are included.\n")
    if output.stat().st_size > 90_000_000:
        raise ValueError(f"Report exceeds 90 MB: {output}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
