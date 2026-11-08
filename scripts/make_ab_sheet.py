"""Blind human A/B sheet: LoRA vs a baseline on val rows where their predicted cells differ.

    python scripts/make_ab_sheet.py --a results/lora_val_preds.jsonl --b results/<best>_val_preds.jsonl --n 50 --seed 42
    python scripts/make_ab_sheet.py --unblind eval/ab_sheet.csv

Make mode picks val rows where the two systems' predicted (task_type, complexity)
cells differ: rows in data/borderline.jsonl first, then the rest, each group in
seeded random order, up to --n. Each row randomly puts system A or B on side X.
It writes eval/ab_sheet.csv (what you fill in) and eval/ab_key.json (which system
is X and Y; gitignored, do not open it until you have finished rating).

Fill in `choice` as X, Y, tie or both_wrong. my_task and my_complexity are
optional hand labels.

--unblind reads the filled sheet and the key and reports wins, ties and
both_wrong per system, a Wilson interval on each system's win rate (wins out of
decided rows, so ties and both_wrong are left out), and how often your hand
labels agree with the val target labels and with the GPT-6 relabels, overall
and on the borderline subset. Writes results/ab_results.json. Stdlib only.
"""

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import COMPLEXITIES, REPO, TASK_TYPES, read_jsonl  # noqa: E402
from score import wilson  # noqa: E402

COLUMNS = ["row", "prompt", "X_task", "X_complexity", "Y_task", "Y_complexity",
           "choice", "my_task", "my_complexity", "note"]
CHOICES = {"x": "X", "y": "Y", "tie": "tie", "both_wrong": "both_wrong"}
FAILED = "(parse failure)"


def system_name(path: str) -> str:
    stem = Path(path).name
    return re.sub(r"(_val)?_preds\.jsonl$", "", stem) if stem.endswith("_preds.jsonl") else Path(path).stem


def read_preds(path: str) -> dict:
    return {r["id"]: r for r in read_jsonl(path)}


def cell_of(pred):
    """(task_type, complexity), or None when the system gave no valid prediction."""
    if pred is None or pred.get("parse_ok") is not True:
        return None
    return pred.get("task_type"), pred.get("complexity")


def show(pred) -> tuple:
    c = cell_of(pred)
    return (FAILED, FAILED) if c is None else (c[0] if c[0] is not None else "n/a", c[1])


def pick_rows(val: list, preds_a: dict, preds_b: dict, border_ids: set, n: int, seed: int):
    """Returns (picked val rows, number eligible). Rows missing from either preds file are skipped."""
    eligible = [r for r in val
                if r["id"] in preds_a and r["id"] in preds_b
                and cell_of(preds_a[r["id"]]) != cell_of(preds_b[r["id"]])]
    rng = random.Random(seed)
    border = [r for r in eligible if r["id"] in border_ids]
    rest = [r for r in eligible if r["id"] not in border_ids]
    rng.shuffle(border)
    rng.shuffle(rest)
    return (border + rest)[:n], len(eligible), rng


def make_sheet(args) -> int:
    if not (args.a and args.b):
        sys.exit("make mode needs --a and --b")
    name_a, name_b = args.name_a or system_name(args.a), args.name_b or system_name(args.b)
    if name_a == name_b:
        sys.exit(f"both systems are called {name_a!r}; pass --name-a / --name-b")
    val = read_jsonl(args.val)
    preds_a, preds_b = read_preds(args.a), read_preds(args.b)
    border_ids = {r["id"] for r in read_jsonl(args.borderline)}

    picked, n_eligible, rng = pick_rows(val, preds_a, preds_b, border_ids, args.n, args.seed)
    n_border = sum(1 for r in picked if r["id"] in border_ids)
    print(f"{n_eligible} val rows have different predicted cells for {name_a} and {name_b}")
    if n_eligible < args.n:
        print(f"only {n_eligible} eligible, fewer than the {args.n} asked for: taking all of them")
    print(f"sheet has {len(picked)} rows, {n_border} from the borderline subset, then {len(picked) - n_border} others")

    sheet, key_rows = [], []
    for i, r in enumerate(picked, 1):
        x_is_a = rng.random() < 0.5
        x, y = (name_a, name_b) if x_is_a else (name_b, name_a)
        px, py = (preds_a, preds_b) if x_is_a else (preds_b, preds_a)
        xt, xc = show(px[r["id"]])
        yt, yc = show(py[r["id"]])
        sheet.append({"row": i, "prompt": r["prompt"], "X_task": xt, "X_complexity": xc,
                      "Y_task": yt, "Y_complexity": yc, "choice": "", "my_task": "", "my_complexity": "", "note": ""})
        key_rows.append({"row": i, "id": r["id"], "X": x, "Y": y, "borderline": r["id"] in border_ids})

    out, key_path = Path(args.out), Path(args.key)
    out.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:  # BOM so Excel reads the UTF-8
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(sheet)
    key = {"a": name_a, "b": name_b, "a_preds": args.a, "b_preds": args.b, "seed": args.seed,
           "n_requested": args.n, "n_eligible": n_eligible, "n": len(picked), "rows": key_rows}
    key_path.write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out} and {key_path} (keep the key closed until you have rated every row)")
    return 0


def agreement_block(k: int, n: int) -> dict:
    ci = wilson(k, n)
    return {"agree": k, "n": n, "rate": k / n if n else None, "wilson95": list(ci) if ci else None}


def label_agreement(rows: list) -> dict:
    """rows: dicts with my_task, my_complexity (may be ""), target and gpt6 label dicts."""
    out = {}
    for field, mine, key in (("task_type", "my_task", "task_type"), ("complexity", "my_complexity", "complexity")):
        sel = [r for r in rows if r[mine]]
        out[field] = {"labeled": len(sel),
                      "vs_target": agreement_block(sum(r[mine] == r["target"][key] for r in sel), len(sel)),
                      "vs_gpt6": agreement_block(sum(r[mine] == r["gpt6"][key] for r in sel), len(sel))}
    both = [r for r in rows if r["my_task"] and r["my_complexity"]]
    out["cell"] = {"labeled": len(both),
                   "vs_target": agreement_block(sum(r["my_task"] == r["target"]["task_type"]
                                                    and r["my_complexity"] == r["target"]["complexity"] for r in both), len(both)),
                   "vs_gpt6": agreement_block(sum(r["my_task"] == r["gpt6"]["task_type"]
                                                  and r["my_complexity"] == r["gpt6"]["complexity"] for r in both), len(both))}
    return out


def unblind(args) -> int:
    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    key_by_row = {k["row"]: k for k in key["rows"]}
    val = {r["id"]: r for r in read_jsonl(args.val)}
    gpt6 = {r["id"]: r for r in read_jsonl(args.gpt6)}
    with open(args.unblind, encoding="utf-8-sig", newline="") as f:
        sheet = list(csv.DictReader(f))

    names = [key["a"], key["b"]]
    wins = dict.fromkeys(names, 0)
    ties = both_wrong = unrated = 0
    errors, labelled = [], []
    seen = set()
    for r in sheet:
        try:
            row_no = int(str(r["row"]).strip())
        except (ValueError, KeyError):
            errors.append(f"bad row number {r.get('row')!r}")
            continue
        k = key_by_row.get(row_no)
        if k is None or row_no in seen:
            errors.append(f"row {row_no}: not in the key, or listed twice")
            continue
        seen.add(row_no)
        raw = (r.get("choice") or "").strip().lower().replace(" ", "_").replace("-", "_")
        my_task, my_comp = (r.get("my_task") or "").strip(), (r.get("my_complexity") or "").strip()
        if my_task and my_task not in TASK_TYPES:
            errors.append(f"row {row_no}: my_task {my_task!r} is not a task_type")
        if my_comp and my_comp not in COMPLEXITIES:
            errors.append(f"row {row_no}: my_complexity {my_comp!r} is not a complexity")
        if raw == "":
            unrated += 1
        elif raw not in CHOICES:
            errors.append(f"row {row_no}: choice {r.get('choice')!r} is not X, Y, tie or both_wrong")
        else:
            choice = CHOICES[raw]
            if choice in ("X", "Y"):
                wins[k[choice]] += 1
            elif choice == "tie":
                ties += 1
            else:
                both_wrong += 1
        if my_task or my_comp:  # any label problem is in `errors`, which stops the run below
            labelled.append({"my_task": my_task, "my_complexity": my_comp, "borderline": k["borderline"],
                             "target": val[k["id"]], "gpt6": gpt6[k["id"]]})
    missing_rows = sorted(set(key_by_row) - seen)
    if missing_rows:
        errors.append(f"key rows missing from the sheet: {missing_rows}")
    if errors:
        print("Fix the sheet first:", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
        return 1

    decided = sum(wins.values())
    systems = {}
    for name in names:
        ci = wilson(wins[name], decided)
        systems[name] = {"wins": wins[name], "win_rate_decided": wins[name] / decided if decided else None,
                         "wilson95": list(ci) if ci else None}
    results = {
        "a": key["a"], "b": key["b"], "seed": key["seed"], "rows": len(sheet), "unrated": unrated,
        "ties": ties, "both_wrong": both_wrong, "decided": decided, "systems": systems,
        "hand_label_agreement": {
            "overall": label_agreement(labelled),
            "borderline_subset": label_agreement([r for r in labelled if r["borderline"]]),
        },
        "note": "Win rate = wins / (wins of A + wins of B); ties and both_wrong are left out. "
                "Agreement is on rows where you filled in a hand label.",
    }
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.results_dir) / "ab_results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"{len(sheet)} rows, {unrated} unrated, {ties} tie, {both_wrong} both_wrong, {decided} decided")
    for name in names:
        s = systems[name]
        rate = "n/a" if s["win_rate_decided"] is None else \
            f"{s['win_rate_decided']:.3f} [{s['wilson95'][0]:.3f}, {s['wilson95'][1]:.3f}]"
        print(f"  {name}: {s['wins']} wins, win rate among decided rows {rate}")
    for title, block in results["hand_label_agreement"].items():
        print(f"hand labels vs target / GPT-6 relabel ({title.replace('_', ' ')}):")
        for field in ("task_type", "complexity", "cell"):
            b = block[field]
            t, g = b["vs_target"], b["vs_gpt6"]
            f = lambda x: "n/a" if x["rate"] is None else f"{x['agree']}/{x['n']} ({x['rate']:.3f})"  # noqa: E731
            print(f"  {field:<10} labeled {b['labeled']:>3}   target {f(t)}   GPT-6 {f(g)}")
    print(f"wrote {out_path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", help="preds for system A (the LoRA model)")
    ap.add_argument("--b", help="preds for system B (the best baseline)")
    ap.add_argument("--name-a", help="default: from the preds file name")
    ap.add_argument("--name-b")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--unblind", metavar="SHEET", help="score a filled-in sheet instead of making one")
    ap.add_argument("--out", default=str(REPO / "eval" / "ab_sheet.csv"))
    ap.add_argument("--key", default=str(REPO / "eval" / "ab_key.json"))
    ap.add_argument("--val", default=str(REPO / "data" / "val.jsonl"))
    ap.add_argument("--borderline", default=str(REPO / "data" / "borderline.jsonl"))
    ap.add_argument("--gpt6", default=str(REPO / "data" / "relabeled_gpt6.jsonl"))
    ap.add_argument("--results-dir", default=str(REPO / "results"))
    args = ap.parse_args(argv)
    return unblind(args) if args.unblind else make_sheet(args)


if __name__ == "__main__":
    sys.exit(main())
