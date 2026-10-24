"""Checks for the length-only baseline: thresholds, determinism, output shape, I/O."""

import builtins
import contextlib
import io
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))

import length_only  # noqa: E402
from length_only import fit_thresholds, main, predict_complexity, word_count  # noqa: E402


def row(n_words, complexity, **extra):
    return {"id": f"r{n_words}-{complexity}", "prompt": " ".join(["w"] * n_words), "complexity": complexity, **extra}


# Short prompts are low, mid-length medium, long high, with one noisy row.
TRAIN = (
    [row(n, "low") for n in (3, 4, 5, 6)]
    + [row(n, "medium") for n in (10, 11, 12, 13)]
    + [row(n, "high") for n in (30, 31, 32, 33)]
    + [row(20, "low")]
)


class TempDirCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def write_jsonl(self, name, rows, newline="\n"):
        path = self.dir / name
        with open(path, "w", encoding="utf-8", newline="") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + newline)
        return path

    def run_main(self, fit, predict, thresholds_name="thresholds.json"):
        out, err = io.StringIO(), io.StringIO()
        argv = ["--fit", str(fit), "--predict", str(predict), "--thresholds-out", str(self.dir / thresholds_name)]
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            main(argv)
        return out.getvalue(), err.getvalue()


class FitThresholds(unittest.TestCase):
    def test_thresholds_are_ordered(self):
        t1, t2, _ = fit_thresholds(TRAIN)
        self.assertLess(t1, t2)

    def test_recovers_obvious_split(self):
        t1, t2, acc = fit_thresholds(TRAIN)
        self.assertEqual((t1, t2), (6, 13))
        self.assertAlmostEqual(acc, 12 / 13)

    def test_matches_brute_force_with_lower_threshold_tie_break(self):
        rows = [row(n, c) for n, c in [(2, "low"), (4, "medium"), (4, "low"), (7, "high"), (9, "medium"), (12, "high")]]
        counts = sorted({word_count(r["prompt"]) for r in rows})
        best = None
        for t1, t2 in itertools.combinations(counts, 2):
            correct = sum(predict_complexity(word_count(r["prompt"]), t1, t2) == r["complexity"] for r in rows)
            if best is None or correct > best[0]:
                best = (correct, t1, t2)
        t1, t2, acc = fit_thresholds(rows)
        self.assertEqual((t1, t2), (best[1], best[2]))
        self.assertAlmostEqual(acc, best[0] / len(rows))

    def test_ties_go_to_lower_thresholds(self):
        # Every pair that puts both rows right scores the same; the lowest pair wins.
        rows = [row(1, "low"), row(5, "medium"), row(9, "high")]
        t1, t2, acc = fit_thresholds(rows)
        self.assertEqual((t1, t2, acc), (1, 5, 1.0))

    def test_needs_two_distinct_counts(self):
        with self.assertRaises(ValueError):
            fit_thresholds([row(4, "low"), row(4, "high")])

    def test_rejects_unknown_label(self):
        with self.assertRaises(ValueError):
            fit_thresholds([row(1, "low"), row(5, "extreme")])

    def test_predict_boundaries(self):
        self.assertEqual(predict_complexity(6, 6, 13), "low")
        self.assertEqual(predict_complexity(7, 6, 13), "medium")
        self.assertEqual(predict_complexity(13, 6, 13), "medium")
        self.assertEqual(predict_complexity(14, 6, 13), "high")


class Cli(TempDirCase):
    def test_deterministic(self):
        train = self.write_jsonl("train.jsonl", TRAIN)
        val = self.write_jsonl("val.jsonl", [row(n, "low") for n in (2, 9, 40)])
        first = self.run_main(train, val)
        second = self.run_main(train, val)
        self.assertEqual(first, second)
        self.assertEqual(
            json.loads((self.dir / "thresholds.json").read_text(encoding="utf-8"))["t1"],
            6,
        )

    def test_output_fields_have_no_task_type(self):
        train = self.write_jsonl("train.jsonl", TRAIN)
        val = self.write_jsonl("val.jsonl", [{"id": "a", "prompt": "one two three"}, {"prompt": "x " * 50}])
        out, _ = self.run_main(train, val)
        lines = [json.loads(line) for line in out.splitlines()]
        self.assertEqual([set(r) for r in lines], [{"id", "complexity"}] * 2)
        self.assertEqual([r["id"] for r in lines], ["a", None])
        self.assertEqual([r["complexity"] for r in lines], ["low", "high"])

    def test_saves_and_prints_thresholds(self):
        train = self.write_jsonl("train.jsonl", TRAIN)
        val = self.write_jsonl("val.jsonl", [row(3, "low")])
        out_path = self.dir / "nested" / "thresholds.json"
        _, err = self.run_main(train, val, thresholds_name="nested/thresholds.json")
        saved = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual((saved["t1"], saved["t2"]), (6, 13))
        self.assertIn("t1=6", err)
        self.assertIn("t2=13", err)

    def test_utf8_prompts(self):
        train = self.write_jsonl("train.jsonl", TRAIN)
        # Non-ASCII words count as one word each; the emoji and CJK must not break reading or counting.
        val = self.write_jsonl("val.jsonl", [{"id": "u", "prompt": "naïve café 日本語 🚀 résumé " * 2}])
        out, _ = self.run_main(train, val)
        self.assertEqual(json.loads(out)["complexity"], "medium")  # 10 words

    def test_crlf_line_endings(self):
        train = self.write_jsonl("train.jsonl", TRAIN, newline="\r\n")
        val = self.write_jsonl("val.jsonl", [row(3, "low"), row(40, "high")], newline="\r\n")
        out, _ = self.run_main(train, val)
        self.assertEqual([json.loads(line)["complexity"] for line in out.splitlines()], ["low", "high"])

    def test_crlf_inside_prompt_counts_words(self):
        self.assertEqual(word_count("one two\r\nthree\r\n\r\nfour"), 4)

    def test_report_on_labeled_file(self):
        train = self.write_jsonl("train.jsonl", TRAIN)
        val = self.write_jsonl("val.jsonl", [row(3, "low"), row(11, "medium"), row(40, "high"), row(3, "high")])
        _, err = self.run_main(train, val)
        self.assertIn("accuracy: 0.7500", err)
        self.assertIn("macro-F1", err)
        self.assertIn("confusion matrix", err)

    def test_no_report_without_labels(self):
        train = self.write_jsonl("train.jsonl", TRAIN)
        val = self.write_jsonl("val.jsonl", [{"id": "a", "prompt": "hello there"}])
        _, err = self.run_main(train, val)
        self.assertNotIn("confusion matrix", err)

    def test_refuses_to_fit_on_test_file(self):
        test = self.write_jsonl("test.jsonl", TRAIN)
        val = self.write_jsonl("val.jsonl", [row(3, "low")])
        with self.assertRaises(SystemExit):
            self.run_main(test, val)
        self.assertFalse((self.dir / "thresholds.json").exists())


class MacroF1(unittest.TestCase):
    def test_perfect_and_empty_class(self):
        perfect = length_only.confusion_matrix(["low", "medium", "high"], ["low", "medium", "high"])
        self.assertEqual(length_only.macro_f1(perfect), 1.0)
        # "high" never appears or is predicted, so it scores 0 and drags the mean to 2/3.
        two_class = length_only.confusion_matrix(["low", "medium"], ["low", "medium"])
        self.assertAlmostEqual(length_only.macro_f1(two_class), 2 / 3)


class TestFileIsolation(TempDirCase):
    def opened_names(self, fit, predict):
        opened = []
        real_open = builtins.open

        def spy(file, *args, **kwargs):
            opened.append(Path(str(file)).name)
            return real_open(file, *args, **kwargs)

        with mock.patch.object(builtins, "open", spy):
            self.run_main(fit, predict)
        return opened

    def test_predicting_never_touches_test_jsonl(self):
        train = self.write_jsonl("train.jsonl", TRAIN)
        val = self.write_jsonl("val.jsonl", [row(3, "low")])
        self.write_jsonl("test.jsonl", [row(3, "low")])  # sits next to the others
        opened = self.opened_names(train, val)
        self.assertIn("val.jsonl", opened)
        self.assertNotIn("test.jsonl", opened)

    def test_predict_reads_test_jsonl_when_pointed_at_it(self):
        train = self.write_jsonl("train.jsonl", TRAIN)
        test = self.write_jsonl("test.jsonl", [row(3, "low")])
        opened = self.opened_names(train, test)
        self.assertIn("test.jsonl", opened)

    def test_source_never_hardcodes_a_test_path(self):
        source = Path(length_only.__file__).read_text(encoding="utf-8")
        self.assertNotIn("data/test", source)


if __name__ == "__main__":
    unittest.main()
