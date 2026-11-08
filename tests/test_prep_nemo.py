"""Checks for scripts/prep_nemo.py: borderline rows dropped, chat and input-output shapes, label format."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import prep_nemo  # noqa: E402
from common import load_template, parse_output  # noqa: E402


def row(i, task="debugging", complexity="low", prompt=None):
    return {"id": f"x{i}", "prompt": prompt or f"prompt {i}", "task_type": task, "complexity": complexity}


class PrepNemo(unittest.TestCase):
    def run_prep(self, d, extra=()):
        train = [row(1), row(2, "research_qa", "high"), row(3, prompt="café \"quoted\"\nsecond line")]
        val = [row(4), row(5, "writing_docs", "medium")]
        for name, rows in (("train", train), ("val", val), ("border", [{"id": "x2"}, {"id": "x5"}])):
            (d / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\r\n" for r in rows), encoding="utf-8", newline="")
        argv = ["--train", str(d / "train.jsonl"), "--val", str(d / "val.jsonl"),
                "--borderline", str(d / "border.jsonl"), *extra]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(prep_nemo.main(argv), 0)
        return out.getvalue()

    def read(self, path):
        raw = Path(path).read_bytes()
        self.assertNotIn(b"\r", raw)
        return [json.loads(line) for line in raw.decode("utf-8").splitlines()]

    def test_chat_format_drops_borderline_and_uses_template(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            out = self.run_prep(d, ["--out-dir", str(d / "o")])
            train, val = self.read(d / "o" / "train.jsonl"), self.read(d / "o" / "val.jsonl")
            self.assertEqual(len(train), 2)
            self.assertEqual(len(val), 1)
            self.assertIn("2 written", out)
            ex = train[0]["messages"]
            self.assertEqual([m["role"] for m in ex], ["system", "user", "assistant"])
            self.assertEqual(ex[0]["content"], load_template())
            self.assertEqual(ex[1]["content"], "prompt 1")
            self.assertEqual(ex[2]["content"], '{"task_type":"debugging","complexity":"low"}')
            self.assertEqual(parse_output(ex[2]["content"])[1], True)
            # The unicode, quotes and newline survive the round trip.
            self.assertEqual(train[1]["messages"][1]["content"], "café \"quoted\"\nsecond line")

    def test_input_output_format(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self.run_prep(d, ["--format", "input-output", "--out-dir", str(d / "o")])
            train = self.read(d / "o" / "train.jsonl")
            self.assertEqual(set(train[0]), {"input", "output"})
            self.assertEqual(train[0]["input"], load_template() + "\n\nprompt 1")
            self.assertEqual(train[0]["output"], '{"task_type":"debugging","complexity":"low"}')

    def test_bad_label_stops(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "train.jsonl").write_text(json.dumps(row(1, "nope", "low")) + "\n", encoding="utf-8")
            (d / "val.jsonl").write_text("", encoding="utf-8")
            (d / "border.jsonl").write_text("", encoding="utf-8")
            with self.assertRaises(SystemExit):
                prep_nemo.main(["--train", str(d / "train.jsonl"), "--val", str(d / "val.jsonl"),
                                "--borderline", str(d / "border.jsonl"), "--out-dir", str(d / "o")])


if __name__ == "__main__":
    unittest.main()
