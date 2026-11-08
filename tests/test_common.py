"""Checks for scripts/common.py: JSONL reading, template loading, output parsing."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from common import COMPLEXITIES, TASK_TYPES, load_template, parse_output, read_jsonl  # noqa: E402

GOOD = {"task_type": "debugging", "complexity": "low"}


class ParseOutput(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(parse_output('{"task_type":"debugging","complexity":"low"}'), (GOOD, True))

    def test_whitespace_and_spaced_json(self):
        self.assertEqual(parse_output('  \n{"task_type": "debugging", "complexity": "low"}\n '), (GOOD, True))

    def test_json_fence(self):
        text = '```json\n{"task_type": "debugging", "complexity": "low"}\n```'
        self.assertEqual(parse_output(text), (GOOD, True))

    def test_json_fence_crlf(self):
        text = '```json\r\n{"task_type": "debugging", "complexity": "low"}\r\n```'
        self.assertEqual(parse_output(text), (GOOD, True))

    def test_extra_keys_are_dropped(self):
        text = '{"task_type":"debugging","complexity":"low","why":"x"}'
        self.assertEqual(parse_output(text), (GOOD, True))

    def test_failures(self):
        bad = [
            "",
            None,
            "debugging, low",
            'Sure! {"task_type":"debugging","complexity":"low"}',
            '{"task_type":"debugging"}',
            '{"complexity":"low"}',
            '{"task_type":"bug_fixing","complexity":"low"}',
            '{"task_type":"debugging","complexity":"Low"}',
            '{"task_type":"debugging","complexity":"easy"}',
            '{"task_type":["debugging"],"complexity":"low"}',
            '["debugging","low"]',
            '{"task_type":"debugging","complexity":"low"',  # truncated
            '```\n{"task_type":"debugging","complexity":"low"}\n```',  # bare fence is not the allowed one
            '```json\n```json\n{"task_type":"debugging","complexity":"low"}\n```\n```',  # nested fences
        ]
        for text in bad:
            with self.subTest(text=text):
                self.assertEqual(parse_output(text), (None, False))

    def test_every_label_accepted(self):
        for t in TASK_TYPES:
            for c in COMPLEXITIES:
                out, ok = parse_output(json.dumps({"task_type": t, "complexity": c}))
                self.assertTrue(ok)
                self.assertEqual(out, {"task_type": t, "complexity": c})


class ReadJsonl(unittest.TestCase):
    def test_crlf_bom_and_blank_lines(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.jsonl"
            p.write_bytes(b'\xef\xbb\xbf{"id": "a"}\r\n\r\n{"id": "b"}\r\n')
            self.assertEqual(read_jsonl(p), [{"id": "a"}, {"id": "b"}])


class Template(unittest.TestCase):
    def test_template_names_every_label_and_output_rule(self):
        t = load_template()
        for label in TASK_TYPES + COMPLEXITIES:
            self.assertIn(label, t)
        self.assertIn("ONLY a JSON object", t)
        self.assertFalse(t.endswith("\n"))
        self.assertNotIn("\r", t)

    def test_template_is_short(self):
        # Rough ceiling: ~4 chars per token puts 350 tokens near 1400 chars.
        self.assertLess(len(load_template()), 1500)


if __name__ == "__main__":
    unittest.main()
