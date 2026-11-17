"""Meaningful CPU checks for matching, error attribution, and two-image negatives."""
from __future__ import annotations

import json
import random
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from evaluate_locany_predictions import count_matches
from locany_reference_diagnostics import diagnose, matching, prediction_views
from prepare_locany_reference_alignment import make_bank, make_class_rows, choose_reference


def box(category, coordinates):
    return {"class": category, "box_01": coordinates}


class ReferenceDiagnosticsTests(unittest.TestCase):
    def test_same_place_wrong_class_counts_both_fp_fn(self):
        d = diagnose([box("pore", [0, 0, .2, .2])], [box("deposit", [0, 0, .2, .2])])
        self.assertEqual((d["scores"]["tp"], d["scores"]["fp"], d["scores"]["fn"]), (0, 1, 1))
        self.assertEqual(d["fp_reasons"], {"classification": 1})
        self.assertEqual(d["fn_reasons"], {"wrong_class_at_gt": 1})

    def test_matching_agrees_with_existing_evaluator_on_ambiguous_overlaps(self):
        rng = random.Random(42)
        for _ in range(60):
            truth, predictions = [], []
            for dest in (truth, predictions):
                for _ in range(12):
                    x, y = rng.random() * .6, rng.random() * .6
                    dest.append(box(rng.choice(("pore", "deposit")), [x, y, x+.4, y+.4]))
            matches = matching(truth, predictions)
            self.assertEqual((len(matches), len(predictions)-len(matches), len(truth)-len(matches)),
                             count_matches(truth, predictions, .5))
        truth = [box("pore", [0, 0, .6, 1]), box("pore", [.4, 0, 1, 1])]
        predictions = [box("pore", [0, 0, 1, 1]), box("pore", [0, 0, .6, 1])]
        self.assertEqual(len(matching(truth, predictions)), 2)

    def test_error_buckets_partition_unmatched_predictions(self):
        truth = [box("pore", [0, 0, .2, .2]), box("stain", [.5, .5, 1, 1])]
        predictions = [truth[0], truth[0], box("stain", [.5, .5, .7, .7]),
                       box("deposit", [.5, .5, .7, .7]), box("deposit", [.3, .3, .4, .4])]
        d = diagnose(truth, predictions)
        self.assertEqual(d["fp_reasons"], {"duplicate": 1, "localization": 1,
                         "classification_and_localization": 1, "background": 1})
        self.assertEqual(sum(d["fp_reasons"].values()), d["scores"]["fp"])
        self.assertEqual(sum(d["fn_reasons"].values()), d["scores"]["fn"])

    def test_query_filter_can_remove_tp_and_never_relabels(self):
        prediction = box("deposit", [.1, .1, .2, .2])
        record = {"predictions": [prediction], "queries": [
            {"requested_class": "stain", "predictions": [prediction]}]}
        views = prediction_views(record)
        self.assertEqual(views["raw"], [prediction])
        self.assertEqual(views["query_filter"], [])
        self.assertEqual(prediction["class"], "deposit")

    def test_dedup_keeps_overlapping_different_classes(self):
        p = box("pore", [.1, .1, .2, .2])
        d = box("deposit", [.1, .1, .2, .2])
        views = prediction_views({"predictions": [p, p, d]})
        self.assertEqual(views["dedup"], [p, d])

    def test_analysis_cli_preserves_raw_tp_and_exposes_filter_loss(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prediction = box("deposit", [.1, .1, .2, .2])
            answers = {c: f"<ref>{c}</ref><box>none</box>" for c in
                       ("pore", "deposit", "discontinuity", "stain")}
            answers["stain"] = "<ref>deposit</ref><box><100><100><200><200></box>"
            source = root / "predictions.jsonl"
            source.write_text(json.dumps({"model": "fixture", "image": "sample.jpg",
                "ground_truth": [prediction], "predictions": [prediction], "answer": answers}) + "\n", encoding="utf-8")
            subprocess.run([sys.executable, str(Path(__file__).with_name("analyze_locany_reference_errors.py")),
                            str(source), "--output", str(root / "out")], check=True, capture_output=True)
            with (root / "out/scores.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            raw = next(r for r in rows if r["view"] == "raw" and r["class"] == "all")
            filtered = next(r for r in rows if r["view"] == "query_filter" and r["class"] == "all")
            self.assertEqual((raw["tp"], raw["fp"], raw["fn"]), ("1", "0", "0"))
            self.assertEqual((filtered["tp"], filtered["fp"], filtered["fn"]), ("0", "0", "1"))

    def test_train_only_external_refs_and_negative_queries(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "root"
            directory = root / "fold0/train/images"
            directory.mkdir(parents=True)
            rows = []
            for index in range(2):
                Image.new("RGB", (80, 30), (index * 30, 50, 70)).save(directory / f"{index}.png")
                answer = "".join(f"<ref>{c}</ref><box><{10+j*200}><100><{150+j*200}><800></box>"
                                 for j, c in enumerate(("pore", "deposit", "discontinuity", "stain")))
                rows.append({"image": f"fold0/train/images/{index}.png", "conversations": [
                    {"from": "human", "value": "unused"}, {"from": "gpt", "value": answer}]})
            out = Path(folder) / "out"
            bank, refs = make_bank(rows, root, out, .2)
            with Image.open(refs[0]["image"]) as rendered:
                self.assertEqual(rendered.height, 70)  # 30px image + external legend
            template = "<image-1> Find {category}. Example <image-2>. Size {width} {height}."
            negatives = [{"image": rows[0]["image"], "conversations": [rows[0]["conversations"][0],
                {"from": "gpt", "value": "<ref>pore</ref><box><10><100><150><800></box>"}]}]
            prepared, counts = make_class_rows(negatives, bank, root, template)
            self.assertEqual(len(prepared), 4)
            self.assertEqual(sum(counts["negative"].values()), 3)
            for row in prepared:
                self.assertEqual(len(row["image"]), 2)
                self.assertNotEqual(row["image"][0], row["image"][1])
                self.assertIn("<image-2>", row["conversations"][0]["value"])
            reference = choose_reference(bank, "pore", rows[0]["image"])
            self.assertEqual(reference["source_image"], rows[1]["image"])


if __name__ == "__main__":
    unittest.main()
