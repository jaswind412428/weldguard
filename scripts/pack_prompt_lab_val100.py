"""On the 4090, pack the exact 100 fold0-val images used in earlier prompt runs.

Only raw validation images and their YOLO labels are included. No model weights,
train images, or independent test images enter this archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SOURCE_ZIP = REPO / "results/locany_p05_original_prompt_diagnostics.zip"
SOURCE_MEMBER = "original/predictions.jsonl"
DEFAULT_SPLIT = REPO / "datasets/weld-dataset/kfold_high/fold0/val"
DEFAULT_OUTPUT = REPO / "datasets/derived/prompt_lab_val100.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    split = args.split.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not SOURCE_ZIP.is_file():
        parser.error(f"Missing historical prompt archive: {SOURCE_ZIP}")

    with zipfile.ZipFile(SOURCE_ZIP) as archive:
        rows = [json.loads(line) for line in archive.read(SOURCE_MEMBER).decode("utf-8").splitlines() if line]
    names = [Path(row["image"]).name for row in rows]
    if len(rows) != 100 or len(set(names)) != 100:
        raise ValueError("Historical prompt archive must list 100 unique images")
    # The source must be the fold0 validation split, never train or test.
    expected = "datasets/weld-dataset/kfold_high/fold0/val/images/"
    if any(
        not row["image"].replace("\\", "/").startswith(expected)
        or Path(row["image"]).name != name
        for row, name in zip(rows, names)
    ):
        raise ValueError("Historical selections must come only from fold0/val/images")

    files = []
    for name in names:
        if Path(name).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError(f"Unsupported image extension: {name}")
        image_path = split / "images" / name
        label_path = split / "labels" / f"{Path(name).stem}.txt"
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing image or label: {name}")
        files.append((name, image_path, label_path))

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "selection": "historical P01/P03/P04/P05 prompt validation set",
        "selection_source": f"{SOURCE_ZIP.name}:{SOURCE_MEMBER}",
        "selection_source_sha256": hashlib.sha256(SOURCE_ZIP.read_bytes()).hexdigest(),
        "split": "fold0/val",
        "count": len(files),
        "images": names,
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, image_path, label_path in files:
            archive.write(image_path, f"images/{name}")
            archive.write(label_path, f"labels/{label_path.name}")
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Packed {len(files)} unmarked validation images and GT labels into {output}")
    print(f"Archive size: {output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
