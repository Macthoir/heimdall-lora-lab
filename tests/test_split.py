"""Tests for scripts/split.py, using a synthetic 540-row dataset (no real data, no test set)."""

import contextlib
import io
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import split  # noqa: E402
from split import COMPLEXITIES, TASK_TYPES  # noqa: E402

PER_CELL = 30


def make_rows(per_cell=PER_CELL, seed=0):
    """One row per slot, each with 20 random distinct-looking words, so no two are near-duplicates."""
    rng = random.Random(seed)
    rows = []
    for task in TASK_TYPES:
        for cx in COMPLEXITIES:
            for _ in range(per_cell):
                words = [f"w{rng.randrange(10**6)}" for _ in range(20)]
                rows.append({
                    "id": f"tv-{len(rows) + 1:04d}",
                    "prompt": " ".join(words),
                    "task_type": task,
                    "complexity": cx,
                    "label_source": "target",
                    "meta": {},
                })
    return rows


def make_group(rows, indices):
    """Make rows[indices[1:]] near-duplicates of rows[indices[0]]: each swaps one word (Jaccard ~0.9).
    Two variants differ from each other by two words (~0.82), so groups of 3+ also test chaining."""
    base = rows[indices[0]]["prompt"].split()
    for k, i in enumerate(indices[1:]):
        words = list(base)
        words[k] = f"swap{i}"
        rows[i]["prompt"] = " ".join(words)


def entries_of(rows):
    return [split.Entry(n, json.dumps(r), r) for n, r in enumerate(rows, start=1)]


def checks(problems):
    return [p[:3] for p in problems]


class ValidationTests(unittest.TestCase):
    def test_clean_dataset_passes(self):
        self.assertEqual(split.validate(entries_of(make_rows())), [])

    def test_wrong_row_count(self):
        problems = split.validate(entries_of(make_rows()[:-1]))
        self.assertTrue(any("[1] expected 540 rows, found 539" in p for p in problems))
        self.assertTrue(any(p.startswith("[4]") and "research_qa/high has 29" in p for p in problems))

    def test_duplicate_ids(self):
        rows = make_rows()
        rows[1]["id"] = rows[0]["id"]
        self.assertIn("[1] id 'tv-0001' appears 2 times", split.validate(entries_of(rows)))

    def test_bad_labels(self):
        rows = make_rows()
        rows[0]["task_type"] = "chitchat"
        rows[1]["complexity"] = "extreme"
        rows[2]["task_type"] = ["debugging"]
        got = checks(split.validate(entries_of(rows)))
        self.assertEqual(got.count("[2]"), 2)
        self.assertEqual(got.count("[3]"), 1)

    def test_cell_imbalance(self):
        rows = make_rows()
        rows[0]["complexity"] = "high"  # one cell has 29, its neighbour 31
        problems = split.validate(entries_of(rows))
        self.assertEqual(checks(problems), ["[4]", "[4]"])
        self.assertTrue(any("code_generation/low has 29" in p for p in problems))
        self.assertTrue(any("code_generation/high has 31" in p for p in problems))

    def test_exact_duplicate_after_case_and_whitespace(self):
        rows = make_rows()
        rows[5]["prompt"] = "  " + rows[0]["prompt"].upper().replace(" ", "  ") + "\n"
        problems = split.validate(entries_of(rows))
        self.assertEqual(checks(problems), ["[5]"])
        self.assertIn("tv-0001", problems[0])
        self.assertIn("tv-0006", problems[0])

    def test_bad_prompts(self):
        rows = make_rows()
        for row, bad in zip(rows, ["", "   \n", 42, None]):
            row["prompt"] = bad
        self.assertEqual(checks(split.validate(entries_of(rows))), ["[6]"] * 4)

    def test_every_problem_is_reported(self):
        rows = make_rows()
        rows[0]["task_type"] = "chitchat"
        rows[1]["prompt"] = ""
        rows[3]["prompt"] = rows[2]["prompt"]
        found = set(checks(split.validate(entries_of(rows))))
        self.assertTrue({"[2]", "[4]", "[5]", "[6]"} <= found)

    def test_near_duplicates_are_reported_not_rejected(self):
        rows = make_rows()
        make_group(rows, [0, 1, 2])
        # Same words as row 0 bar one, but in another cell: not a near-duplicate pair.
        rows[40]["prompt"] = rows[0]["prompt"].replace(rows[0]["prompt"].split()[5], "other")
        entries = entries_of(rows)
        self.assertEqual(split.validate(entries), [])
        self.assertEqual([(i, j) for i, j, _ in split.find_near_duplicates(entries)], [(0, 1), (0, 2)])

    def test_jaccard_threshold_is_inclusive(self):
        words = [f"a{i}" for i in range(20)]
        for kept, expected in [(17, [(0, 1)]), (16, [])]:  # 17/20 = 0.85 exactly, 16/20 = 0.80
            with self.subTest(kept=kept):
                rows = make_rows()
                rows[0]["prompt"] = " ".join(words)
                rows[1]["prompt"] = " ".join(words[:kept])
                pairs = split.find_near_duplicates(entries_of(rows))
                self.assertEqual([(i, j) for i, j, _ in pairs], expected)

    def test_case_and_punctuation_do_not_hide_near_duplicates(self):
        rows = make_rows()
        rows[0]["prompt"] = "Fix the Kubernetes pod, it keeps crashing on startup!"
        rows[1]["prompt"] = "fix the kubernetes pod; it keeps crashing on startup?"
        pairs = split.find_near_duplicates(entries_of(rows))
        self.assertEqual(pairs, [(0, 1, 1.0)])


class SplitTests(unittest.TestCase):
    def test_cells_are_balanced(self):
        entries = entries_of(make_rows())
        train, val = split.split_entries(entries, [], seed=42)
        self.assertEqual((len(train), len(val)), (432, 108))
        for task in TASK_TYPES:
            for cx in COMPLEXITIES:
                with self.subTest(cell=(task, cx)):
                    self.assertEqual(sum(split.cell_of(e.row) == (task, cx) for e in val), 6)
                    self.assertEqual(sum(split.cell_of(e.row) == (task, cx) for e in train), 24)

    def test_train_and_val_partition_the_input(self):
        entries = entries_of(make_rows())
        train, val = split.split_entries(entries, [], seed=42)
        ids = [e.row["id"] for e in train + val]
        self.assertEqual(sorted(ids), [e.row["id"] for e in entries])
        self.assertEqual(ids, sorted(ids[:432]) + sorted(ids[432:]))  # each side sorted by id

    def grouped_entries(self):
        rows = make_rows()
        for cell in range(18):
            base = cell * PER_CELL
            make_group(rows, [base, base + 1, base + 2])
            make_group(rows, [base + 3, base + 4])
        entries = entries_of(rows)
        return entries, split.find_near_duplicates(entries)

    def test_near_duplicate_groups_never_straddle_the_split(self):
        entries, pairs = self.grouped_entries()
        self.assertEqual(len(pairs), 18 * 3)
        for seed in range(20):
            with self.subTest(seed=seed):
                train, val = split.split_entries(entries, pairs, seed=seed)
                val_ids = {e.row["id"] for e in val}
                for i, j, _ in pairs:
                    self.assertEqual(entries[i].row["id"] in val_ids, entries[j].row["id"] in val_ids)
                self.assertEqual({len(val)}, {108})

    def test_chained_group_stays_together(self):
        # Rows 1 and 2 are only ~0.82 similar to each other; they are joined through row 0.
        entries, pairs = self.grouped_entries()
        self.assertNotIn((1, 2), {(i, j) for i, j, _ in pairs})
        for seed in range(20):
            _, val = split.split_entries(entries, pairs, seed=seed)
            in_val = [e.row["id"] in {v.row["id"] for v in val} for e in entries[:3]]
            self.assertIn(sum(in_val), (0, 3))

    def test_group_larger_than_val_quota_goes_to_train(self):
        rows = make_rows()
        make_group(rows, list(range(8)))
        entries = entries_of(rows)
        pairs = split.find_near_duplicates(entries)
        for seed in range(5):
            train, val = split.split_entries(entries, pairs, seed=seed)
            self.assertTrue({f"tv-{n:04d}" for n in range(1, 9)} <= {e.row["id"] for e in train})
            self.assertEqual(len(val), 108)

    def test_unreachable_val_quota_reports_actual_counts(self):
        rows = make_rows()
        for g in range(6):  # cell 0 is six groups of five: val can take one group, not 6 rows
            make_group(rows, list(range(g * 5, g * 5 + 5)))
        entries = entries_of(rows)
        pairs = split.find_near_duplicates(entries)
        train, val = split.split_entries(entries, pairs, seed=1)
        summary = split.summarize(entries, train, val, pairs, 1, "x")
        self.assertEqual(summary["per_cell"]["code_generation"]["low"], {"total": 30, "train": 25, "val": 5})
        self.assertEqual(summary["val_shortfalls"],
                         [{"task_type": "code_generation", "complexity": "low", "val": 5, "target": 6}])
        self.assertEqual((summary["near_duplicates"]["pairs"], summary["near_duplicates"]["groups"]), (24, 6))

    def test_same_seed_gives_same_split(self):
        entries = entries_of(make_rows())
        first = split.split_entries(entries, [], seed=7)
        second = split.split_entries(entries, [], seed=7)
        self.assertEqual(first, second)

    def test_different_seed_gives_different_split(self):
        entries = entries_of(make_rows())
        _, val_a = split.split_entries(entries, [], seed=1)
        _, val_b = split.split_entries(entries, [], seed=2)
        self.assertNotEqual({e.row["id"] for e in val_a}, {e.row["id"] for e in val_b})

    def test_input_order_does_not_matter(self):
        rows = make_rows()
        shuffled = list(rows)
        random.Random(3).shuffle(shuffled)
        _, val_a = split.split_entries(entries_of(rows), [], seed=42)
        _, val_b = split.split_entries(entries_of(shuffled), [], seed=42)
        self.assertEqual([e.row["id"] for e in val_a], [e.row["id"] for e in val_b])


class CliTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.input = self.dir / "trainval.jsonl"
        self.out = self.dir / "out"

    def write_input(self, rows, eol="\n", ensure_ascii=True):
        text = "".join(json.dumps(r, ensure_ascii=ensure_ascii) + eol for r in rows)
        self.input.write_bytes(text.encode("utf-8"))

    def run_cli(self, *extra):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = split.main(["--input", str(self.input), "--out-dir", str(self.out), *extra])
        return code, stdout.getvalue(), stderr.getvalue()

    def outputs(self):
        return sorted(p.name for p in self.out.iterdir()) if self.out.exists() else []

    def test_failed_validation_exits_1_and_writes_nothing(self):
        rows = make_rows()
        self.write_input(rows[:-1])
        code, out, _ = self.run_cli()
        self.assertEqual(code, 1)
        self.assertIn("[1] expected 540 rows, found 539", out)
        self.assertEqual(self.outputs(), [])

    def test_invalid_json_line_is_a_reported_problem(self):
        self.write_input(make_rows())
        with self.input.open("ab") as f:
            f.write(b"{not json}\n[1, 2]\n")
        code, out, _ = self.run_cli()
        self.assertEqual(code, 1)
        self.assertIn("invalid JSON", out)
        self.assertIn("not a JSON object", out)
        self.assertEqual(self.outputs(), [])

    def test_writes_train_val_and_report(self):
        rows = make_rows()
        self.write_input(rows)
        code, out, _ = self.run_cli()
        self.assertEqual(code, 0)
        self.assertEqual(self.outputs(), ["split_report.json", "train.jsonl", "val.jsonl"])
        train = (self.out / "train.jsonl").read_bytes()
        val = (self.out / "val.jsonl").read_bytes()
        self.assertNotIn(b"\r", train + val)
        train_rows = [json.loads(line) for line in train.decode().splitlines()]
        val_rows = [json.loads(line) for line in val.decode().splitlines()]
        self.assertEqual((len(train_rows), len(val_rows)), (432, 108))
        self.assertEqual(train_rows, sorted(train_rows, key=lambda r: r["id"]))
        self.assertEqual(sorted(train_rows + val_rows, key=lambda r: r["id"]), rows)
        report = json.loads((self.out / "split_report.json").read_text())
        self.assertEqual((report["seed"], report["train_rows"], report["val_rows"]), (42, 432, 108))
        self.assertEqual(report["per_cell"]["debugging"]["medium"], {"total": 30, "train": 24, "val": 6})
        self.assertEqual(report["val_shortfalls"], [])
        self.assertIn("Seed: 42", out)

    def test_crlf_blank_lines_and_unicode_survive(self):
        rows = make_rows()
        rows[0]["prompt"] = "caf\u00e9 \u2028 line separator inside the prompt " + rows[0]["prompt"]
        self.input.write_bytes(
            ("\r\n".join(json.dumps(r, ensure_ascii=False) + "\r\n" for r in rows)).encode("utf-8")
        )
        code, _, _ = self.run_cli("--seed", "5")
        self.assertEqual(code, 0)
        lines = (self.out / "train.jsonl").read_bytes().decode("utf-8").split("\n")[:-1]
        lines += (self.out / "val.jsonl").read_bytes().decode("utf-8").split("\n")[:-1]
        self.assertEqual(sorted(lines), sorted(json.dumps(r, ensure_ascii=False) for r in rows))

    def test_dry_run_writes_nothing(self):
        self.write_input(make_rows())
        code, out, _ = self.run_cli("--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("Dry run", out)
        self.assertEqual(self.outputs(), [])

    def test_refuses_to_overwrite_without_force(self):
        self.write_input(make_rows())
        self.assertEqual(self.run_cli()[0], 0)
        before = (self.out / "val.jsonl").read_bytes()
        code, _, err = self.run_cli("--seed", "99")
        self.assertEqual(code, 2)
        self.assertIn("--force", err)
        self.assertEqual((self.out / "val.jsonl").read_bytes(), before)
        self.assertEqual(self.run_cli("--seed", "99", "--force")[0], 0)
        self.assertNotEqual((self.out / "val.jsonl").read_bytes(), before)

    def test_same_seed_gives_identical_files(self):
        self.write_input(make_rows())
        self.assertEqual(self.run_cli("--seed", "3")[0], 0)
        first = {n: (self.out / n).read_bytes() for n in self.outputs()}
        self.assertEqual(self.run_cli("--seed", "3", "--force")[0], 0)
        self.assertEqual({n: (self.out / n).read_bytes() for n in self.outputs()}, first)

    def test_missing_input_exits_2(self):
        code, _, err = self.run_cli()
        self.assertEqual(code, 2)
        self.assertIn("input not found", err)


if __name__ == "__main__":
    unittest.main()
