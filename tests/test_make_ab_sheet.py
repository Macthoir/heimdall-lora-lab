"""Checks for scripts/make_ab_sheet.py: row selection, blinding, CSV round trip and unblinding."""

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_ab_sheet as ab  # noqa: E402


def val_row(i, task="debugging", complexity="low"):
    return {"id": f"v{i}", "prompt": f"prompt {i}\n\"quoted\", café", "task_type": task, "complexity": complexity}


def pred(i, task, complexity, ok=True):
    if not ok:
        return {"id": f"v{i}", "task_type": None, "complexity": None, "raw": "x", "parse_ok": False}
    return {"id": f"v{i}", "task_type": task, "complexity": complexity, "raw": "{}", "parse_ok": True}


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\r\n" for r in rows), encoding="utf-8", newline="")


class AbSheet(unittest.TestCase):
    N = 40

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = self.dir = Path(self.tmp.name)
        val = [val_row(i) for i in range(self.N)]
        # Systems agree on every 10th row; borderline ids are v0..v9 (v0 and v10.. agree where i % 10 == 0).
        a = [pred(i, "debugging", "low") for i in range(self.N)]
        b = [pred(i, "debugging", "low") if i % 10 == 0 else pred(i, "research_qa", "high") for i in range(self.N)]
        b[1] = pred(1, None, None, ok=False)  # a parse failure counts as a different cell
        write_jsonl(d / "val.jsonl", val)
        write_jsonl(d / "a_val_preds.jsonl", a)
        write_jsonl(d / "b_val_preds.jsonl", b)
        write_jsonl(d / "border.jsonl", [{"id": f"v{i}"} for i in range(10)])
        gpt6 = [dict(r, task_type="research_qa", complexity="high") for r in val]
        gpt6[5] = dict(val[5])  # GPT-6 agrees with the target on one row
        write_jsonl(d / "gpt6.jsonl", gpt6)

    def argv(self, *extra):
        d = self.dir
        return ["--a", str(d / "a_val_preds.jsonl"), "--b", str(d / "b_val_preds.jsonl"), "--val", str(d / "val.jsonl"),
                "--borderline", str(d / "border.jsonl"), "--out", str(d / "eval" / "sheet.csv"),
                "--key", str(d / "eval" / "key.json"), "--gpt6", str(d / "gpt6.jsonl"),
                "--results-dir", str(d / "res"), *extra]

    def run_main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = ab.main(argv)
        return code, out.getvalue(), err.getvalue()

    def read_sheet(self):
        with open(self.dir / "eval" / "sheet.csv", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    def key(self):
        return json.loads((self.dir / "eval" / "key.json").read_text(encoding="utf-8"))

    def test_borderline_first_then_random_rest_and_only_differing_cells(self):
        code, out, _ = self.run_main(self.argv("--n", "12", "--seed", "42"))
        self.assertEqual(code, 0)
        rows = self.key()["rows"]
        self.assertEqual(len(rows), 12)
        ids = [r["id"] for r in rows]
        self.assertTrue(all(i in {f"v{k}" for k in range(1, 10)} for i in ids[:9]))
        self.assertTrue(all(i not in {f"v{k}" for k in range(10)} for i in ids[9:]))
        self.assertEqual([r["borderline"] for r in rows], [True] * 9 + [False] * 3)
        self.assertNotIn("v0", ids)
        self.assertTrue(all(int(i[1:]) % 10 != 0 for i in ids))
        self.assertEqual(len(set(ids)), 12)
        self.assertIn("9 from the borderline subset", out)

    def test_seeded_and_reproducible(self):
        self.run_main(self.argv("--n", "12", "--seed", "42"))
        first = (self.key(), self.read_sheet())
        self.run_main(self.argv("--n", "12", "--seed", "42"))
        self.assertEqual((self.key(), self.read_sheet()), first)
        self.run_main(self.argv("--n", "12", "--seed", "7"))
        self.assertNotEqual(self.key()["rows"], first[0]["rows"])

    def test_takes_all_eligible_when_fewer_than_n(self):
        code, out, _ = self.run_main(self.argv("--n", "50"))
        self.assertEqual(code, 0)
        self.assertEqual(len(self.key()["rows"]), 36)  # 40 rows minus the 4 where the cells match
        self.assertIn("only 36 eligible", out)

    def test_sides_are_randomised_and_sheet_hides_system_names(self):
        self.run_main(self.argv("--n", "30"))
        rows = self.key()["rows"]
        self.assertEqual({r["X"] for r in rows}, {"a", "b"})
        for r in rows:
            self.assertEqual({r["X"], r["Y"]}, {"a", "b"})
        sheet = self.read_sheet()
        self.assertEqual(list(sheet[0]), ab.COLUMNS)
        raw = (self.dir / "eval" / "sheet.csv").read_text(encoding="utf-8")
        self.assertNotIn("a_val", raw)
        # The shown cells match the real predictions for the side they sit on.
        preds = {"a": ("debugging", "low"), "b": ("research_qa", "high")}
        for s, k in zip(sheet, rows):
            if k["id"] != "v1":
                self.assertEqual((s["X_task"], s["X_complexity"]), preds[k["X"]])
                self.assertEqual((s["Y_task"], s["Y_complexity"]), preds[k["Y"]])

    def test_csv_round_trip_keeps_multiline_prompts_and_marks_parse_failures(self):
        self.run_main(self.argv("--n", "36"))
        sheet = self.read_sheet()
        k = {r["row"]: r for r in self.key()["rows"]}
        by_id = {k[int(s["row"])]["id"]: s for s in sheet}
        self.assertEqual(by_id["v3"]["prompt"], "prompt 3\n\"quoted\", café")
        v1, side = by_id["v1"], k[[i for i, r in k.items() if r["id"] == "v1"][0]]
        failed_side = "Y" if side["X"] == "a" else "X"
        self.assertEqual(v1[f"{failed_side}_task"], ab.FAILED)

    def fill(self, choices, labels=None):
        sheet = self.read_sheet()
        for s in sheet:
            s["choice"] = choices.get(int(s["row"]), "")
            s["my_task"], s["my_complexity"] = (labels or {}).get(int(s["row"]), ("", ""))
        with open(self.dir / "eval" / "sheet.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ab.COLUMNS)
            w.writeheader()
            w.writerows(sheet)

    def test_unblind_counts_and_agreement(self):
        self.run_main(self.argv("--n", "8"))
        key = {r["row"]: r for r in self.key()["rows"]}
        by_side = lambda row, sysname: "X" if key[row]["X"] == sysname else "Y"  # noqa: E731
        choices = {1: by_side(1, "a"), 2: by_side(2, "a"), 3: by_side(3, "b"), 4: "tie", 5: "both wrong", 6: "Y" if key[6]["Y"] == "b" else "X"}
        # rows 7 and 8 stay unrated
        labels = {1: ("debugging", "low"), 2: ("research_qa", "high"), 3: ("debugging", "high")}
        self.fill(choices, labels)
        code, out, err = self.run_main(["--unblind", str(self.dir / "eval" / "sheet.csv"), "--key", str(self.dir / "eval" / "key.json"),
                                        "--val", str(self.dir / "val.jsonl"), "--gpt6", str(self.dir / "gpt6.jsonl"),
                                        "--results-dir", str(self.dir / "res")])
        self.assertEqual(code, 0, err)
        res = json.loads((self.dir / "res" / "ab_results.json").read_text(encoding="utf-8"))
        self.assertEqual((res["unrated"], res["ties"], res["both_wrong"]), (2, 1, 1))
        self.assertEqual(res["systems"]["a"]["wins"], 2)
        self.assertEqual(res["systems"]["b"]["wins"], 2)
        self.assertEqual(res["decided"], 4)
        self.assertAlmostEqual(res["systems"]["a"]["win_rate_decided"], 0.5)
        lo, hi = res["systems"]["a"]["wilson95"]
        self.assertAlmostEqual(lo, 0.15, places=2)
        self.assertAlmostEqual(hi, 0.85, places=2)
        overall = res["hand_label_agreement"]["overall"]
        self.assertEqual(overall["task_type"]["labeled"], 3)
        # Targets are debugging/low on every row; GPT-6 says research_qa/high except val row v5.
        ids = {r: key[r]["id"] for r in (1, 2, 3)}
        exp_target_task = sum(1 for r, (t, _) in ((1, labels[1]), (2, labels[2]), (3, labels[3])) if t == "debugging")
        self.assertEqual(overall["task_type"]["vs_target"]["agree"], exp_target_task)
        exp_gpt6_task = sum(1 for r in (1, 2, 3) if labels[r][0] == ("debugging" if ids[r] == "v5" else "research_qa"))
        self.assertEqual(overall["task_type"]["vs_gpt6"]["agree"], exp_gpt6_task)
        self.assertEqual(overall["cell"]["labeled"], 3)
        border = res["hand_label_agreement"]["borderline_subset"]
        n_border = sum(1 for r in (1, 2, 3) if key[r]["borderline"])
        self.assertEqual(border["cell"]["labeled"], n_border)
        self.assertIn("win rate among decided rows", out)

    def test_unblind_rejects_bad_choices_and_labels(self):
        self.run_main(self.argv("--n", "4"))
        self.fill({1: "maybe", 2: "X"}, {2: ("nonsense", "")})
        code, _, err = self.run_main(["--unblind", str(self.dir / "eval" / "sheet.csv"), "--key", str(self.dir / "eval" / "key.json"),
                                      "--val", str(self.dir / "val.jsonl"), "--gpt6", str(self.dir / "gpt6.jsonl"),
                                      "--results-dir", str(self.dir / "res")])
        self.assertEqual(code, 1)
        self.assertIn("choice 'maybe'", err)
        self.assertIn("my_task 'nonsense'", err)
        self.assertFalse((self.dir / "res" / "ab_results.json").exists())

    def test_same_system_twice_is_an_error(self):
        argv = self.argv()
        argv[argv.index("--b") + 1] = argv[argv.index("--a") + 1]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            ab.main(argv)

    def test_system_name_from_path(self):
        self.assertEqual(ab.system_name("results/lora_val_preds.jsonl"), "lora")
        self.assertEqual(ab.system_name("results/qwen_base_val_preds.jsonl"), "qwen_base")


if __name__ == "__main__":
    unittest.main()
