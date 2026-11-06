"""Reuse V2 training images and answers, changing only the human prompt to P04."""

import hashlib
import json
from pathlib import Path

from PIL import Image


repo = Path(__file__).resolve().parents[1]
source_recipe = repo / "datasets/derived/locany_full_v2/full_recipe.json"
recipe = json.loads(source_recipe.read_text(encoding="utf-8"))
if len(recipe) != 1:
    raise ValueError("Expected exactly one V2 training dataset")

entry = next(iter(recipe.values()))
root = Path(entry["root"]).resolve()
source_annotation = Path(entry["annotation"])
rows = [json.loads(line) for line in source_annotation.read_text(encoding="utf-8").splitlines() if line]
if len(rows) != 653 or entry["repeat_time"] != 1.0 or entry["data_augment"]:
    raise ValueError("V2 recipe differs from the recorded 653-image experiment")

prompt = (repo / "prompts/locany_p04_balanced.txt").read_text(encoding="utf-8").strip()
suffix_template = (
    " The image is {width} pixels wide and {height} pixels high."
    " Return boxes using the model's normalized 0 to 1000 coordinate scale."
)
for row in rows:
    image_path = (root / row["image"]).resolve()
    if not image_path.is_relative_to(root / "fold0/train/images"):
        raise ValueError(f"Not a fold0 training image: {image_path}")
    conversation = row["conversations"]
    if (len(conversation) != 2 or conversation[0]["from"] != "human"
            or conversation[1]["from"] != "gpt"):
        raise ValueError(f"Unexpected V2 conversation structure: {image_path}")
    with Image.open(image_path) as image:
        width, height = image.size
    conversation[0]["value"] = prompt + suffix_template.format(width=width, height=height)

out = repo / "datasets/derived/locany_v5_p04"
out.mkdir(parents=True, exist_ok=True)
annotation = out / "train_full.jsonl"
annotation.write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    encoding="utf-8",
)
entry["annotation"] = str(annotation)
(out / "full_recipe.json").write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
(out / "training_prompt_template.txt").write_text(
    prompt + suffix_template + "\n", encoding="utf-8"
)
manifest = {
    "model": "nvidia/LocateAnything-3B",
    "prompt_id": "P04",
    "images": len(rows),
    "source_annotation": str(source_annotation),
    "source_annotation_sha256": hashlib.sha256(source_annotation.read_bytes()).hexdigest(),
    "output_annotation_sha256": hashlib.sha256(annotation.read_bytes()).hexdigest(),
    "prompt_file": "prompts/locany_p04_balanced.txt",
    "image_size_suffix": suffix_template,
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"V5: {len(rows)} V2 training images and answers preserved; P04 + image size applied.")
print(annotation)
