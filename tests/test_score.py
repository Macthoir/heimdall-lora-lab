"""Checks for scripts/score.py: Wilson interval, macro-F1, parse failures, missing ids, off-by-one."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import score  # noqa: E402
from score import COMPLEXITIES, TASK_TYPES, confusion, evaluate, macro_f1, update_summary, wilson  # noqa: E402


def gold_row(i, task="debugging", complexity="low"):
    return {"id": f"g{i}", "prompt": f"prompt {i}", "task_type": task, "complexity": complexity}


def pred_row(i, task="debugging", complexity="low", ok=True):
    if not ok:
        return {"id": f"g{i}", "task_type": None, "complexity": None, "raw": "garbage", "parse_ok": False}
    return {"id": f"g{i}", "task_type": task, "complexity": complexity,
            "raw": json.dumps({"task_type": task, "complexity": complexity}), "parse_ok": True}


def write_jsonl(path, rows, eol="\n"):
    Path(path).write_text("".join(json.dumps(r) + eol for r in rows), encoding="utf-8", newline="")


class Wilson(unittest.TestCase):
    def test_half(self):
        lo, hi = wilson(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=4)
        self.assertAlmostEqual(hi, 0.7634, places=4)

    def test_edges_stay_in_range(self):
        lo, hi = wilson(0, 10)
        self.assertAlmostEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.2775, places=4)
        lo, hi = wilson(10, 10)
        self.assertAlmostEqual(lo, 0.7225, places=4)
        self.assertAlmostEqual(hi, 1.0)
        self.assertLessEqual(hi, 1.0)

    def test_empty(self):
        self.assertIsNone(wilson(0, 0))


class MacroF1(unittest.TestCase):
    def test_hand_computed(self):
        # low: tp2 fp0 fn1 -> 0.8; medium: tp1 fp1 fn1 -> 0.5; high: tp2 fp1 fn0 -> 0.8
        pairs = [("low", "low")] * 2 + [("low", "medium")] + [("medium", "medium"), ("medium", "high")] + [("high", "high")] * 2
        cm = confusion(COMPLEXITIES, pairs)
        self.assertAlmostEqual(macro_f1(COMPLEXITIES, cm), 0.7)

    def test_perfect_and_all_wrong(self):
        cm = confusion(COMPLEXITIES, [(c, c) for c in COMPLEXITIES])
        self.assertEqual(macro_f1(COMPLEXITIES, cm), 1.0)
        cm = confusion(COMPLEXITIES, [("low", "high"), ("medium", "low"), ("high", "medium")])
        self.assertEqual(macro_f1(COMPLEXITIES, cm), 0.0)

    def test_labels_without_support_are_skipped(self):
        cm = confusion(COMPLEXITIES, [("low", "low"), ("low", "medium")])
        # Only "low" has support: tp1 fn1 -> F1 = 2/3. medium and high do not pull the mean down.
        self.assertAlmostEqual(macro_f1(COMPLEXITIES, cm), 2 / 3)

    def test_invalid_predictions_count_as_misses(self):
        cm = confusion(COMPLEXITIES, [("low", "low"), ("low", None)])
        self.assertAlmostEqual(macro_f1(COMPLEXITIES, cm), 2 / 3)


class Evaluate(unittest.TestCase):
    def test_perfect(self):
        gold = [gold_row(1), gold_row(2, "research_qa", "high")]
        preds = {"g1": pred_row(1), "g2": pred_row(2, "research_qa", "high")}
        m = evaluate(gold, preds)
        self.assertEqual(m["task_type"]["correct"], 2)
        self.assertEqual(m["complexity"]["correct"], 2)
        self.assertEqual(m["cell"]["correct"], 2)
        self.assertEqual(m["parse_failures"], 0)
        self.assertEqual(m["missing"], 0)

    def test_parse_failure_is_wrong_on_both_fields(self):
        gold = [gold_row(1), gold_row(2)]
        preds = {"g1": pred_row(1), "g2": pred_row(2, ok=False)}
        m = evaluate(gold, preds)
        self.assertEqual(m["parse_failures"], 1)
        self.assertEqual(m["missing"], 0)
        self.assertEqual((m["task_type"]["correct"], m["task_type"]["n"]), (1, 2))
        self.assertEqual((m["complexity"]["correct"], m["complexity"]["n"]), (1, 2))
        self.assertEqual(m["cell"]["correct"], 1)
        self.assertEqual(m["task_type"]["confusion"]["debugging"]["invalid"], 1)

    def test_parse_ok_true_with_bad_label_still_fails(self):
        bad = pred_row(1, "not_a_task", "low")
        m = evaluate([gold_row(1)], {"g1": bad})
        self.assertEqual(m["parse_failures"], 1)
        self.assertEqual(m["cell"]["correct"], 0)

    def test_missing_ids_are_wrong_and_counted_separately(self):
        gold = [gold_row(1), gold_row(2), gold_row(3)]
        m = evaluate(gold, {"g1": pred_row(1)})
        self.assertEqual(m["missing"], 2)
        self.assertEqual(m["parse_failures"], 0)
        self.assertEqual(m["task_type"]["n"], 3)
        self.assertEqual(m["task_type"]["correct"], 1)
        self.assertEqual(m["complexity"]["correct"], 1)

    def test_extra_pred_ids_are_ignored(self):
        m = evaluate([gold_row(1)], {"g1": pred_row(1), "zzz": pred_row(9)})
        self.assertEqual(m["n"], 1)

    def test_off_by_one_and_two(self):
        gold = [gold_row(i, complexity=c) for i, c in enumerate(["low", "low", "medium", "high", "medium", "low"])]
        preds = {
            "g0": pred_row(0, complexity="medium"),  # off by one
            "g1": pred_row(1, complexity="high"),    # off by two
            "g2": pred_row(2, complexity="medium"),  # right
            "g3": pred_row(3, complexity="medium"),  # off by one
            "g4": pred_row(4, ok=False),             # error with no distance
            "g5": pred_row(5, complexity="low"),     # right
        }
        c = evaluate(gold, preds)["complexity"]
        self.assertEqual(c["errors"], 4)
        self.assertEqual(c["errors_with_valid_pred"], 3)
        self.assertEqual(c["off_by_one"], 2)
        self.assertEqual(c["off_by_two"], 1)
        self.assertAlmostEqual(c["off_by_one_rate"], 2 / 3)

    def test_no_errors_gives_no_rate(self):
        c = evaluate([gold_row(1)], {"g1": pred_row(1)})["complexity"]
        self.assertIsNone(c["off_by_one_rate"])

    def test_task_na_skips_task_and_cell(self):
        row = pred_row(1)
        row["task_type"] = None
        m = evaluate([gold_row(1)], {"g1": row}, task_na=True)
        self.assertIsNone(m["task_type"])
        self.assertIsNone(m["cell"])
        self.assertEqual(m["complexity"]["correct"], 1)
        self.assertEqual(m["parse_failures"], 0)

    def test_per_cell_has_all_18_cells(self):
        gold = [gold_row(1, "writing_docs", "high")]
        m = evaluate(gold, {"g1": pred_row(1, "writing_docs", "high")})
        self.assertEqual(len(m["per_cell"]), len(TASK_TYPES) * len(COMPLEXITIES))
        cell = next(c for c in m["per_cell"] if c["task_type"] == "writing_docs" and c["complexity"] == "high")
        self.assertEqual((cell["n"], cell["cell_correct"]), (1, 1))


class Summary(unittest.TestCase):
    def test_row_is_replaced_not_duplicated(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "summary.md"
            self.assertFalse(update_summary(p, "a", "| a | 1 |"))
            self.assertFalse(update_summary(p, "b", "| b | 1 |"))
            self.assertTrue(update_summary(p, "a", "| a | 2 |"))
            lines = p.read_text(encoding="utf-8").splitlines()
            self.assertEqual([x for x in lines if x.startswith("| a |")], ["| a | 2 |"])
            self.assertEqual([x for x in lines if x.startswith("| b |")], ["| b | 1 |"])
            self.assertEqual(sum(x.startswith("| Model |") for x in lines), 1)


class EndToEnd(unittest.TestCase):
    def run_main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = score.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_scores_file_summary_and_borderline_subset(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            gold = [gold_row(1), gold_row(2, complexity="medium"), gold_row(3, "research_qa", "high")]
            write_jsonl(d / "gold.jsonl", gold, eol="\r\n")
            write_jsonl(d / "border.jsonl", [{"id": "g2"}])
            write_jsonl(d / "p.jsonl", [pred_row(1), pred_row(2, ok=False)])  # g3 is missing
            code, stdout, _ = self.run_main([
                "--gold", str(d / "gold.jsonl"), "--preds", str(d / "p.jsonl"), "--name", "toy",
                "--borderline", str(d / "border.jsonl"), "--results-dir", str(d / "res"),
            ])
            self.assertEqual(code, 0)
            self.assertIn("borderline subset", stdout)
            scores = json.loads((d / "res" / "toy_val_scores.json").read_text(encoding="utf-8"))
            self.assertEqual(scores["all"]["missing"], 1)
            self.assertEqual(scores["all"]["parse_failures"], 1)
            self.assertEqual(scores["all"]["cell"]["correct"], 1)
            self.assertEqual(scores["borderline_subset"]["n"], 1)
            self.assertEqual(scores["borderline_subset"]["parse_failures"], 1)
            summary = (d / "res" / "summary.md").read_text(encoding="utf-8")
            self.assertIn("| toy |", summary)
            self.assertIn("1 (+1 missing)", summary)

    def test_from_length_marks_task_na_and_fits_on_the_train_file(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            sizes = [(2, "low"), (3, "low"), (10, "medium"), (11, "medium"), (30, "high"), (31, "high")]
            train = [{"id": f"t{i}", "prompt": "w " * n, "complexity": c, "task_type": "debugging"}
                     for i, (n, c) in enumerate(sizes)]
            gold = [dict(gold_row(1), prompt="w " * 2),
                    dict(gold_row(2, complexity="medium"), prompt="w " * 10),
                    dict(gold_row(3, complexity="high"), prompt="w " * 40)]
            write_jsonl(d / "train.jsonl", train)
            write_jsonl(d / "gold.jsonl", gold)
            code, _, _ = self.run_main([
                "--gold", str(d / "gold.jsonl"), "--from-length", "--train", str(d / "train.jsonl"),
                "--borderline", str(d / "none.jsonl"), "--results-dir", str(d / "res"),
            ])
            self.assertEqual(code, 0)
            scores = json.loads((d / "res" / "length_only_val_scores.json").read_text(encoding="utf-8"))
            self.assertTrue(scores["all"]["task_na"])
            self.assertEqual(scores["all"]["complexity"]["correct"], 3)
            self.assertIsNone(scores["borderline_subset"])
            self.assertIn("| length_only | n/a |", (d / "res" / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
