"""Length-only sanity baseline for complexity.

Predicts complexity from prompt word count alone, using two thresholds fitted
on the train split. It makes no task_type prediction. If the LoRA model can't
beat this, the model is mostly learning length.

Stdlib only.

Usage:
    python baselines/length_only.py --fit data/train.jsonl --predict data/val.jsonl > results/length_only_val_preds.jsonl

Rule: words <= t1 -> low, t1 < words <= t2 -> medium, words > t2 -> high.
Thresholds are fitted on --fit only, never on the file being predicted.
Input lines need a "prompt" field; an "id" field is passed through if present.
Predictions go to stdout. Thresholds and, when the --predict file has
complexity labels, accuracy / macro-F1 / confusion matrix go to stderr.
"""

import argparse
import json
import sys
from pathlib import Path

LABELS = ["low", "medium", "high"]
DEFAULT_THRESHOLDS_PATH = Path(__file__).resolve().parents[1] / "results" / "length_only_thresholds.json"
# The test split is scored once, so it must never be used to fit.
HELD_OUT_NAME = "test.jsonl"


def word_count(prompt: str) -> int:
    return len(prompt.split())


def predict_complexity(words: int, t1: int, t2: int) -> str:
    if words <= t1:
        return "low"
    if words <= t2:
        return "medium"
    return "high"


def read_jsonl(path: str) -> list:
    # utf-8-sig tolerates a BOM; text mode folds \r\n line endings to \n.
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def fit_thresholds(rows: list) -> tuple:
    """Return (t1, t2, train_accuracy), grid-searching over observed word counts.

    Ties go to the lower thresholds: t1 ascending, then t2 ascending, and only a
    strictly better accuracy replaces the current best.
    """
    if not rows:
        raise ValueError("no rows to fit on")
    for row in rows:
        if row.get("complexity") not in LABELS:
            raise ValueError(f"bad complexity label {row.get('complexity')!r} in fit data")

    counts = sorted({word_count(r["prompt"]) for r in rows})
    if len(counts) < 2:
        raise ValueError("need at least two distinct word counts to fit t1 < t2")

    # per_count[i][label] = rows with counts[i] words and that label.
    index = {c: i for i, c in enumerate(counts)}
    per_count = [dict.fromkeys(LABELS, 0) for _ in counts]
    for row in rows:
        per_count[index[word_count(row["prompt"])]][row["complexity"]] += 1

    # prefix[k][label] = rows whose count is among counts[0..k-1]; prefix[0] is all zeros.
    prefix = [dict.fromkeys(LABELS, 0)]
    for cell in per_count:
        prefix.append({lab: prefix[-1][lab] + cell[lab] for lab in LABELS})

    best = None  # (correct, t1, t2)
    for i in range(len(counts) - 1):
        for j in range(i + 1, len(counts)):
            low_ok = prefix[i + 1]["low"]
            med_ok = prefix[j + 1]["medium"] - prefix[i + 1]["medium"]
            high_ok = prefix[-1]["high"] - prefix[j + 1]["high"]
            correct = low_ok + med_ok + high_ok
            if best is None or correct > best[0]:
                best = (correct, counts[i], counts[j])
    correct, t1, t2 = best
    return t1, t2, correct / len(rows)


def confusion_matrix(true: list, pred: list) -> dict:
    matrix = {t: dict.fromkeys(LABELS, 0) for t in LABELS}
    for t, p in zip(true, pred):
        matrix[t][p] += 1
    return matrix


def macro_f1(matrix: dict) -> float:
    """Unweighted mean F1 over all three labels; a label never hit scores 0."""
    f1s = []
    for lab in LABELS:
        tp = matrix[lab][lab]
        fp = sum(matrix[t][lab] for t in LABELS) - tp
        fn = sum(matrix[lab].values()) - tp
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return sum(f1s) / len(f1s)


def report(true: list, pred: list, file=None) -> None:
    file = file or sys.stderr
    matrix = confusion_matrix(true, pred)
    accuracy = sum(matrix[lab][lab] for lab in LABELS) / len(true)
    print(f"complexity accuracy: {accuracy:.4f}", file=file)
    print(f"complexity macro-F1: {macro_f1(matrix):.4f}", file=file)
    print("confusion matrix (rows = true, cols = predicted):", file=file)
    print("          " + "".join(f"{lab:>8}" for lab in LABELS), file=file)
    for t in LABELS:
        print(f"{t:>8}  " + "".join(f"{matrix[t][p]:>8}" for p in LABELS), file=file)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Length-only complexity baseline.")
    parser.add_argument("--fit", required=True, help="JSONL with prompt + complexity to fit thresholds on (train only)")
    parser.add_argument("--predict", required=True, help="JSONL with prompt to predict on")
    parser.add_argument("--thresholds-out", default=str(DEFAULT_THRESHOLDS_PATH), help="where to save the fitted thresholds")
    args = parser.parse_args(argv)

    if Path(args.fit).name == HELD_OUT_NAME:
        sys.exit(f"refusing to fit on {HELD_OUT_NAME}: the test split is scored once, never tuned on")

    fit_rows = read_jsonl(args.fit)
    try:
        t1, t2, train_acc = fit_thresholds(fit_rows)
    except ValueError as e:
        sys.exit(f"fit failed: {e}")

    out_path = Path(args.thresholds_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fitted = {"t1": t1, "t2": t2, "train_accuracy": train_acc, "fit_file": args.fit, "n_fit": len(fit_rows)}
    out_path.write_text(json.dumps(fitted, indent=2) + "\n", encoding="utf-8")
    print(f"fitted thresholds: t1={t1} t2={t2} (train accuracy {train_acc:.4f}, n={len(fit_rows)})", file=sys.stderr)

    rows = read_jsonl(args.predict)
    preds = []
    for row in rows:
        pred = predict_complexity(word_count(row["prompt"]), t1, t2)
        preds.append(pred)
        print(json.dumps({"id": row.get("id"), "complexity": pred}))

    if rows and all(r.get("complexity") in LABELS for r in rows):
        report([r["complexity"] for r in rows], preds)


if __name__ == "__main__":
    main()
