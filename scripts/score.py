"""Score one system's predictions against the gold labels. One scorer for every row.

    python scripts/score.py --gold data/val.jsonl --preds results/<name>_val_preds.jsonl --name <name>
    python scripts/score.py --gold data/val.jsonl --from-rules
    python scripts/score.py --gold data/val.jsonl --from-length

Prediction rows: {"id", "task_type", "complexity", "raw", "parse_ok"}. task_type
and complexity are null on a parse failure. A parse failure, or an id missing
from the preds, counts as wrong on both fields.

Macro-F1 averages over the labels that appear in the gold rows being scored, so
a label that is absent from a small subset does not drag the mean down.
Wilson intervals are given on accuracies only; they do not apply to F1.

Writes results/<name>_val_scores.json and puts one row per model in
results/summary.md (re-scoring a name replaces its row). Stdlib only.
"""

import argparse
import json
import math
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import COMPLEXITIES, REPO, TASK_TYPES, TEMPLATE_PATH, read_jsonl  # noqa: E402

Z95 = 1.959963984540054
INVALID = "invalid"  # confusion-matrix column for parse failures and missing ids

SUMMARY_HEADER = (
    "| Model | task_type acc | complexity acc | cell exact-match | task_type macro-F1 "
    "| complexity off-by-one | parse failures |\n"
    "|---|---|---|---|---|---|---|\n"
)
SUMMARY_INTRO = (
    "Scored on `data/val.jsonl` (108 rows, generator target labels as ground truth), "
    "used as the held-out test set. Accuracies show a 95% Wilson interval. "
    "Written by `scripts/score.py`.\n\n"
)


def wilson(k: int, n: int, z: float = Z95):
    """95% Wilson score interval for k successes in n trials, or None if n == 0."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def confusion(labels, pairs) -> dict:
    """pairs = (true, pred-or-None). Rows are true labels; INVALID holds pred None."""
    cm = {t: dict.fromkeys(labels + [INVALID], 0) for t in labels}
    for t, p in pairs:
        cm[t][p if p is not None else INVALID] += 1
    return cm


def macro_f1(labels, cm: dict):
    """Mean F1 over labels with gold support. Invalid predictions are misses (fn), never hits for a label."""
    f1s = []
    for lab in labels:
        support = sum(cm[lab].values())
        if support == 0:
            continue
        tp = cm[lab][lab]
        fp = sum(cm[t][lab] for t in labels) - tp
        fn = support - tp
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(f1s) / len(f1s) if f1s else None


def acc_block(k: int, n: int) -> dict:
    ci = wilson(k, n)
    return {"correct": k, "n": n, "acc": k / n if n else None, "wilson95": list(ci) if ci else None}


def evaluate(gold: list, preds: dict, task_na: bool = False) -> dict:
    """Score gold rows against preds (id -> pred row). task_na: system makes no task_type prediction."""
    rows = []
    n_missing = n_parse_fail = 0
    for g in gold:
        p = preds.get(g["id"])
        task_p = comp_p = None
        if p is None:
            n_missing += 1
        else:
            ok = p.get("parse_ok") is True and p.get("complexity") in COMPLEXITIES
            ok = ok and (task_na or p.get("task_type") in TASK_TYPES)
            if ok:
                comp_p = p["complexity"]
                task_p = None if task_na else p["task_type"]
            else:
                n_parse_fail += 1
        rows.append((g, task_p, comp_p))

    n = len(rows)
    comp_pairs = [(g["complexity"], c) for g, _, c in rows]
    comp_cm = confusion(COMPLEXITIES, comp_pairs)
    comp_correct = sum(1 for t, p in comp_pairs if t == p)

    # Distance only exists when the system gave a valid complexity.
    errors = [(t, p) for t, p in comp_pairs if t != p]
    dists = [abs(COMPLEXITIES.index(t) - COMPLEXITIES.index(p)) for t, p in errors if p is not None]
    off1, off2 = dists.count(1), dists.count(2)
    out = {
        "n": n,
        "missing": n_missing,
        "parse_failures": n_parse_fail,
        "task_na": task_na,
        "complexity": {
            **acc_block(comp_correct, n),
            "macro_f1": macro_f1(COMPLEXITIES, comp_cm),
            "confusion": comp_cm,
            "errors": len(errors),
            "errors_with_valid_pred": len(dists),
            "off_by_one": off1,
            "off_by_two": off2,
            # Among errors where the system gave a valid level.
            "off_by_one_rate": off1 / len(dists) if dists else None,
        },
        "task_type": None,
        "cell": None,
        "per_cell": None,
    }
    if task_na:
        return out

    task_pairs = [(g["task_type"], t) for g, t, _ in rows]
    task_cm = confusion(TASK_TYPES, task_pairs)
    task_correct = sum(1 for t, p in task_pairs if t == p)
    cell_correct = sum(1 for g, t, c in rows if t == g["task_type"] and c == g["complexity"])
    out["task_type"] = {**acc_block(task_correct, n), "macro_f1": macro_f1(TASK_TYPES, task_cm), "confusion": task_cm}
    out["cell"] = acc_block(cell_correct, n)

    per_cell = []
    for tt in TASK_TYPES:
        for cc in COMPLEXITIES:
            sel = [(t, c) for g, t, c in rows if g["task_type"] == tt and g["complexity"] == cc]
            per_cell.append({
                "task_type": tt, "complexity": cc, "n": len(sel),
                "cell_correct": sum(1 for t, c in sel if t == tt and c == cc),
                "task_correct": sum(1 for t, _ in sel if t == tt),
                "complexity_correct": sum(1 for _, c in sel if c == cc),
            })
    out["per_cell"] = per_cell
    return out


# ---- reporting ----------------------------------------------------------

def fmt_acc(block, digits=4) -> str:
    if block is None or block["acc"] is None:
        return "n/a"
    lo, hi = block["wilson95"]
    return f"{block['acc']:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}] ({block['correct']}/{block['n']})"


def fmt_f1(x) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def print_headline(m: dict, out) -> None:
    if m["task_na"]:
        print("task_type acc:       n/a (no task_type prediction)", file=out)
        print("task_type macro-F1:  n/a", file=out)
    else:
        print(f"task_type acc:       {fmt_acc(m['task_type'])}", file=out)
        print(f"task_type macro-F1:  {fmt_f1(m['task_type']['macro_f1'])}", file=out)
    c = m["complexity"]
    print(f"complexity acc:      {fmt_acc(c)}", file=out)
    print(f"complexity macro-F1: {fmt_f1(c['macro_f1'])}", file=out)
    print(f"cell exact-match:    {'n/a' if m['task_na'] else fmt_acc(m['cell'])}", file=out)
    rate = "n/a" if c["off_by_one_rate"] is None else f"{c['off_by_one_rate']:.4f}"
    print(f"complexity errors:   {c['errors']} ({c['errors_with_valid_pred']} with a valid level): "
          f"off-by-one {c['off_by_one']} (rate {rate}), off-by-two {c['off_by_two']}", file=out)
    print(f"parse failures:      {m['parse_failures']}   missing ids: {m['missing']}", file=out)


def print_confusion(title, labels, cm, out) -> None:
    cols = labels + ([INVALID] if any(cm[t][INVALID] for t in labels) else [])
    w = max(len(c) for c in labels + [INVALID]) + 2  # row-label width
    cw = 11  # column width; longer headers are cut
    print(f"\n{title} (rows = true, cols = predicted)", file=out)
    print(" " * w + "".join(f"{c[:cw - 1]:>{cw}}" for c in cols), file=out)
    for t in labels:
        print(f"{t:<{w}}" + "".join(f"{cm[t][c]:>{cw}}" for c in cols), file=out)


def print_report(name, m, border, n_border, out) -> None:
    print(f"== {name}: {m['n']} rows ==", file=out)
    print_headline(m, out)
    if not m["task_na"]:
        print_confusion("task_type confusion", TASK_TYPES, m["task_type"]["confusion"], out)
    print_confusion("complexity confusion", COMPLEXITIES, m["complexity"]["confusion"], out)
    if m["per_cell"]:
        print("\nper-cell exact match (task_type, complexity: cell / task / complexity correct of n)", file=out)
        for c in m["per_cell"]:
            print(f"  {c['task_type']:<24}{c['complexity']:<8}{c['cell_correct']}/{c['n']}"
                  f"  task {c['task_correct']}/{c['n']}  complexity {c['complexity_correct']}/{c['n']}", file=out)
    print(f"\n== {name}: borderline subset ({n_border} val rows whose id is in the borderline file) ==", file=out)
    if border is None:
        print("no val rows in the borderline subset", file=out)
    else:
        print_headline(border, out)


def summary_row(name: str, m: dict) -> str:
    def cell(block, digits=3):
        if block is None or block["acc"] is None:
            return "n/a"
        lo, hi = block["wilson95"]
        return f"{block['acc']:.{digits}f} ({lo:.{digits}f}-{hi:.{digits}f})"

    c = m["complexity"]
    if m["task_na"]:
        task_acc = cell_acc = task_f1 = "n/a"
    else:
        task_acc = cell(m["task_type"])
        cell_acc = f"{m['cell']['correct']}/{m['cell']['n']} " + cell(m["cell"])
        task_f1 = f"{m['task_type']['macro_f1']:.3f}"
    rate = "n/a" if c["off_by_one_rate"] is None else f"{c['off_by_one_rate']:.3f}"
    fails = str(m["parse_failures"]) + (f" (+{m['missing']} missing)" if m["missing"] else "")
    return f"| {name} | {task_acc} | {cell(c)} | {cell_acc} | {task_f1} | {rate} | {fails} |"


def update_summary(path: Path, name: str, row: str) -> bool:
    """Put `row` in the summary table, replacing an existing row for `name`. Returns True if replaced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    if not any(line.startswith("| Model |") for line in lines):
        lines = (SUMMARY_INTRO + SUMMARY_HEADER).splitlines()
    replaced = False
    for i, line in enumerate(lines):
        if line.startswith(f"| {name} |"):
            lines[i] = row
            replaced = True
    if not replaced:
        lines.append(row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return replaced


# ---- baseline prediction modes ------------------------------------------

def preds_from_rules(gold: list) -> list:
    sys.path.insert(0, str(REPO / "baselines"))
    import rules
    out = []
    for g in gold:
        label = rules.classify(g["prompt"])
        out.append({"id": g["id"], **label, "raw": json.dumps(label), "parse_ok": True})
    return out


def preds_from_length(gold: list, train_path: str):
    sys.path.insert(0, str(REPO / "baselines"))
    import length_only
    t1, t2, train_acc = length_only.fit_thresholds(read_jsonl(train_path))
    out = []
    for g in gold:
        c = length_only.predict_complexity(length_only.word_count(g["prompt"]), t1, t2)
        out.append({"id": g["id"], "task_type": None, "complexity": c,
                    "raw": json.dumps({"complexity": c}), "parse_ok": True})
    return out, {"t1": t1, "t2": t2, "train_accuracy": train_acc, "fit_file": train_path}


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def git_sha() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True, help="gold JSONL (data/val.jsonl)")
    ap.add_argument("--preds", help="predictions JSONL; with --from-* it is where they get written")
    ap.add_argument("--name", help="model name; default rules / length_only with --from-*")
    ap.add_argument("--from-rules", action="store_true", help="generate preds with baselines/rules.py")
    ap.add_argument("--from-length", action="store_true",
                    help="generate preds with baselines/length_only.py (no task_type: scored n/a)")
    ap.add_argument("--no-task", action="store_true", help="the system makes no task_type prediction")
    ap.add_argument("--train", default=str(REPO / "data" / "train.jsonl"), help="fit file for --from-length")
    ap.add_argument("--borderline", default=str(REPO / "data" / "borderline.jsonl"))
    ap.add_argument("--results-dir", default=str(REPO / "results"))
    args = ap.parse_args(argv)

    if args.from_rules and args.from_length:
        ap.error("pick one of --from-rules and --from-length")
    results = Path(args.results_dir)
    name = args.name or ("rules" if args.from_rules else "length_only" if args.from_length else None)
    if not name:
        ap.error("--name is required")
    if not (args.from_rules or args.from_length) and not args.preds:
        ap.error("--preds is required unless --from-rules or --from-length is given")

    gold = read_jsonl(args.gold)
    if len({g["id"] for g in gold}) != len(gold):
        sys.exit("duplicate ids in the gold file")
    preds_path = Path(args.preds) if args.preds else results / f"{name}_val_preds.jsonl"

    extra = {}
    task_na = args.no_task
    if args.from_rules:
        write_jsonl(preds_path, preds_from_rules(gold))
        print(f"wrote {preds_path}", file=sys.stderr)
    elif args.from_length:
        rows, extra = preds_from_length(gold, args.train)
        write_jsonl(preds_path, rows)
        task_na = True
        print(f"wrote {preds_path} (thresholds t1={extra['t1']} t2={extra['t2']}, fitted on {args.train})", file=sys.stderr)

    preds = {}
    dupes = 0
    for p in read_jsonl(preds_path):
        dupes += p["id"] in preds
        preds[p["id"]] = p  # last row wins
    if dupes:
        print(f"warning: {dupes} duplicate ids in {preds_path}; the last row for each is used", file=sys.stderr)
    stray = set(preds) - {g["id"] for g in gold}
    if stray:
        print(f"warning: {len(stray)} pred ids are not in the gold file and are ignored", file=sys.stderr)

    scores_path = results / f"{name}_val_scores.json"
    if scores_path.exists():
        print(f"note: {name} was scored before; overwriting {scores_path}. "
              "The held-out set is meant to be scored once per model.", file=sys.stderr)

    border_ids = {r["id"] for r in read_jsonl(args.borderline)} if Path(args.borderline).exists() else set()
    border_gold = [g for g in gold if g["id"] in border_ids]
    full = evaluate(gold, preds, task_na)
    border = evaluate(border_gold, preds, task_na) if border_gold else None
    print_report(name, full, border, len(border_gold), sys.stdout)

    template_hash = sha256(TEMPLATE_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:12] if TEMPLATE_PATH.exists() else None
    scores = {
        "name": name, "gold": args.gold, "preds": str(preds_path),
        "git_sha": git_sha(), "template_sha256_12": template_hash,
        "all": full, "borderline_subset": border, "borderline_n": len(border_gold), **({"length_fit": extra} if extra else {}),
    }
    results.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8", newline="\n")
    replaced = update_summary(results / "summary.md", name, summary_row(name, full))
    print(f"\nwrote {scores_path}; {'replaced the' if replaced else 'added a'} row in {results / 'summary.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
