"""Random paired visual review, without running inference or selecting by score."""
import csv
import io
import json
import random
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from PIL import Image, ImageDraw, ImageFont
from evaluate_locany_predictions import count_matches

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/vlm/v2_v4_p05_val100'
LABELS = ('V2_P01', 'V4_P05')
COLORS = {'pore': '#ff5252', 'deposit': '#00bfff',
          'discontinuity': '#ffb300', 'stain': '#28ce65'}


def panel(image, boxes, title):
    canvas = Image.new('RGB', (image.width, image.height + 48), 'white')
    canvas.paste(image, (0, 48))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 18)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 12), title, fill='black', font=font)
    for i, box in enumerate(boxes, 1):
        a, b, c, d = box['box_01']
        rect = (a * image.width, b * image.height + 48,
                c * image.width, d * image.height + 48)
        color = COLORS.get(box['class'], '#ff00ff')
        draw.rectangle(rect, outline=color, width=3)
        text = f"{i} {box['class']}"
        position = (min(rect[0], max(0, image.width - 200)), max(48, rect[1] - 22))
        draw.rectangle(draw.textbbox(position, text, font=font), fill='black')
        draw.text(position, text, fill=color, font=font)
    return canvas


def main():
    groups = {}
    for label in LABELS:
        source = BASE / label / 'predictions_classified.jsonl'
        rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
        groups[label] = {Path(r['image']).name: r for r in rows}
        if len(groups[label]) != len(rows):
            raise ValueError(f'Duplicate images: {source}')
    names = sorted(groups[LABELS[0]])
    if len(names) != 100 or set(names) != set(groups[LABELS[1]]):
        raise ValueError('Expected matching 100-image validation sets')
    for name in names:
        if groups[LABELS[0]][name]['ground_truth'] != groups[LABELS[1]][name]['ground_truth']:
            raise ValueError(f'GT mismatch: {name}')
    selected = random.Random(20261021).sample(names, 20)
    target = ROOT / 'results/v2_v4_random20_review.zip'
    target.parent.mkdir(parents=True, exist_ok=True)
    statistics = []
    with ZipFile(target, 'w', ZIP_DEFLATED) as archive:
        archive.writestr('selection.json', json.dumps({'seed': 20261021, 'population': names,
            'selected': selected, 'models': LABELS}, indent=2))
        for label in LABELS:
            archive.writestr(f'{label}/records.jsonl', ''.join(
                json.dumps(groups[label][n], ensure_ascii=False) + '\n' for n in selected))
        for index, name in enumerate(selected, 1):
            source = ROOT / 'datasets/weld-dataset/kfold_high/fold0/val/images' / name
            with Image.open(source) as im:
                image = im.convert('RGB')
            gt = groups[LABELS[0]][name]['ground_truth']
            panels = [panel(image, [], f'{name} | Original'), panel(image, gt, f'GT | {len(gt)} boxes')]
            for label in LABELS:
                pred = groups[label][name]['predictions']
                tp, fp, fn = count_matches(gt, pred, .5)
                lt, lf, ln = count_matches(gt, pred, .5, ignore_class=True)
                statistics.append(dict(image=name, model=label, tp=tp, fp=fp, fn=fn,
                    localization_tp=lt, localization_fp=lf, localization_fn=ln))
                panels.append(panel(image, pred, f'{label} | {len(pred)} boxes | TP {tp} FP {fp} FN {fn}'))
            combined = Image.new('RGB', (image.width, sum(p.height for p in panels)), 'white')
            y = 0
            for p in panels:
                combined.paste(p, (0, y))
                y += p.height
            buffer = io.BytesIO()
            combined.save(buffer, format='JPEG', quality=95)
            archive.writestr(f'compare/{index:02d}_{Path(name).stem}.jpg', buffer.getvalue())
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(statistics[0]))
        writer.writeheader()
        writer.writerows(statistics)
        archive.writestr('per_image_scores.csv', buffer.getvalue())
    print('Selected:', ', '.join(selected))
    print(f'Saved {target} ({target.stat().st_size / 1024**2:.2f} MiB)')


if __name__ == '__main__':
    main()
