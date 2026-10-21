"""Reuse V2 image order and answers, replacing only the human prompt for V4."""
import json
from pathlib import Path
from PIL import Image

repo = Path(__file__).resolve().parents[1]
source = repo / 'datasets/derived/locany_full_v2/full_recipe.json'
recipe = json.loads(source.read_text(encoding='utf-8'))
if len(recipe) != 1:
    raise ValueError('Expected one V2 training dataset')
entry = next(iter(recipe.values()))
root = Path(entry['root']).resolve()
rows = [json.loads(line) for line in Path(entry['annotation']).read_text(encoding='utf-8').splitlines() if line]
if len(rows) != 653 or entry['repeat_time'] != 1.0 or entry['data_augment']:
    raise ValueError('V2 recipe differs from the recorded 653-image experiment')
prompt = (repo / 'prompts/locany_p05_gt_features.txt').read_text(encoding='utf-8').strip()
for row in rows:
    image_path = (root / row['image']).resolve()
    if not image_path.is_relative_to(root / 'fold0/train/images'):
        raise ValueError(f'Not a fold0 training image: {image_path}')
    if len(row['conversations']) != 2 or row['conversations'][0]['from'] != 'human' or row['conversations'][1]['from'] != 'gpt':
        raise ValueError('Unexpected V2 conversation structure')
    with Image.open(image_path) as image:
        width, height = image.size
    row['conversations'][0]['value'] = prompt + (
        f' The image is {width} pixels wide and {height} pixels high.'
        " Return boxes using the model's normalized 0 to 1000 coordinate scale."
    )
out = repo / 'datasets/derived/locany_v4_p05'
out.mkdir(parents=True, exist_ok=True)
annotation = out / 'train_full.jsonl'
annotation.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
entry['annotation'] = str(annotation)
(out / 'full_recipe.json').write_text(json.dumps(recipe, indent=2) + '\n', encoding='utf-8')
(out / 'training_prompt.txt').write_text(prompt + '\n', encoding='utf-8')
print(f'V4: {len(rows)} V2 images and answers preserved; P05 + image size applied.')
print(out / 'full_recipe.json')
