"""Turn data/train.jsonl and data/val.jsonl into NeMo fine-tuning files.

    python scripts/prep_nemo.py                          # chat format -> data/nemo/
    python scripts/prep_nemo.py --format input-output    # -> data/nemo_io/

Every id in data/borderline.jsonl is dropped from both files (the train/val
labels and the GPT-6 relabel disagree on those prompts). The label is the
compact JSON the template asks for. The val file is for training-loss
monitoring only; the final checkpoint is used, never picked by val accuracy.

Stdlib only, so it can be rerun on the GPU box.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import COMPLEXITIES, REPO, TASK_TYPES, load_template, read_jsonl  # noqa: E402


def label_json(row: dict) -> str:
    return json.dumps({"task_type": row["task_type"], "complexity": row["complexity"]}, separators=(",", ":"))


def to_example(row: dict, template: str, fmt: str) -> dict:
    if fmt == "chat":
        return {"messages": [
            {"role": "system", "content": template},
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": label_json(row)},
        ]}
    return {"input": template + "\n\n" + row["prompt"], "output": label_json(row)}


def check_labels(rows: list, name: str) -> None:
    for r in rows:
        if r.get("task_type") not in TASK_TYPES or r.get("complexity") not in COMPLEXITIES:
            sys.exit(f"{name}: bad label on {r.get('id')}: {r.get('task_type')!r}, {r.get('complexity')!r}")


def cell_counts(rows: list) -> Counter:
    return Counter((r["task_type"], r["complexity"]) for r in rows)


def write_examples(path: Path, rows: list, template: str, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(to_example(r, template, fmt), ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", choices=["chat", "input-output"], default="chat")
    ap.add_argument("--train", default=str(REPO / "data" / "train.jsonl"))
    ap.add_argument("--val", default=str(REPO / "data" / "val.jsonl"))
    ap.add_argument("--borderline", default=str(REPO / "data" / "borderline.jsonl"))
    ap.add_argument("--out-dir", help="default data/nemo (chat) or data/nemo_io (input-output)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir) if args.out_dir else REPO / "data" / ("nemo" if args.format == "chat" else "nemo_io")
    template = load_template()
    borderline = {r["id"] for r in read_jsonl(args.borderline)}

    summary = {}
    for split, path in (("train", args.train), ("val", args.val)):
        rows = read_jsonl(path)
        check_labels(rows, split)
        kept = [r for r in rows if r["id"] not in borderline]
        write_examples(out_dir / f"{split}.jsonl", kept, template, args.format)
        summary[split] = (len(rows), kept)

    print(f"format: {args.format}   out: {out_dir}")
    for split, (total, kept) in summary.items():
        print(f"{split}: {total} rows read, {total - len(kept)} borderline dropped, {len(kept)} written")
    print("\nper-cell counts after dropping borderline rows (train / val):")
    train_cells, val_cells = cell_counts(summary["train"][1]), cell_counts(summary["val"][1])
    for t in TASK_TYPES:
        for c in COMPLEXITIES:
            print(f"  {t:<24}{c:<8}{train_cells[(t, c)]:>4} /{val_cells[(t, c)]:>3}")
    smallest = min(((train_cells[(t, c)], t, c) for t in TASK_TYPES for c in COMPLEXITIES))
    print(f"\nsmallest train cell: {smallest[1]}/{smallest[2]} with {smallest[0]} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
