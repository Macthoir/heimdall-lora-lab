"""Validate the 540-row train/val dataset, then split it into train and val.

    python scripts/split.py [--input PATH] [--seed N] [--force] [--dry-run]

Reads data/trainval.jsonl (or data/trainval.json1 if that is the file present).
The separate 180-prompt test set is not read, written or mixed in here.

Validation runs first and every problem is printed:
  1. exactly 540 rows with unique ids
  2. task_type is one of the six task types
  3. complexity is low, medium or high
  4. exactly 30 rows in each of the 18 (task_type, complexity) cells
  5. no exact duplicate prompts (after lowercasing and collapsing whitespace)
  6. prompt is a non-empty string
  7. near-duplicates (word-set Jaccard >= 0.85, same cell) are reported only
If checks 1-6 fail, the script exits 1 and writes nothing.

Split: 24 train and 6 val per cell. Near-duplicate groups (union-find over the
check 7 pairs) are never split across train and val. Per cell, groups are
shuffled with a seeded RNG and moved to val while they fit within 6 rows; a group
that would overshoot is skipped. If a cell can't reach 6, the actual counts are
reported instead of failing. Rows are written unchanged, sorted by id, to
data/train.jsonl and data/val.jsonl, plus data/split_report.json.

Existing outputs are not overwritten without --force. --dry-run validates and
prints the planned split without writing anything. --out-dir changes where the
outputs go (used by the tests).

Exit codes: 0 ok, 1 validation failed or input unreadable, 2 usage or refused overwrite.
"""

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import NamedTuple

# Same label space and exact-duplicate rule as the generator.
from generate import COMPLEXITIES, TASK_TYPES, normalize

REPO = Path(__file__).resolve().parent.parent
DEFAULT_INPUTS = [REPO / "data" / "trainval.jsonl", REPO / "data" / "trainval.json1"]
EXPECTED_PER_CELL = 30
VAL_PER_CELL = 6
NEAR_DUP_JACCARD = 0.85


class Entry(NamedTuple):
    line: int  # 1-based line number in the input file
    raw: str  # the line as read, without its line ending
    row: dict


def load_entries(path: Path):
    """Return (entries, problems). Raises ValueError if the file isn't UTF-8."""
    try:
        text = path.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ValueError(f"{path} is not valid UTF-8: {e}") from e
    entries, problems = [], []
    # Split on \n only: str.splitlines would also break on U+2028 inside a prompt.
    for n, line in enumerate(text.split("\n"), start=1):
        line = line.rstrip("\r")
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"line {n}: invalid JSON ({e.msg})")
            continue
        if not isinstance(row, dict):
            problems.append(f"line {n}: not a JSON object")
            continue
        entries.append(Entry(n, line, row))
    return entries, problems


def cell_of(row: dict):
    """(task_type, complexity) if both labels are valid, else None."""
    task, cx = row.get("task_type"), row.get("complexity")
    if isinstance(task, str) and isinstance(cx, str) and task in TASK_TYPES and cx in COMPLEXITIES:
        return task, cx
    return None


def name(e: Entry) -> str:
    id_ = e.row.get("id")
    return id_ if isinstance(id_, str) and id_ else f"line {e.line}"


def validate(entries, per_cell: int = EXPECTED_PER_CELL) -> list:
    """Checks 1-6. Returns every problem found, each prefixed with its check number."""
    problems = []

    expected = per_cell * len(TASK_TYPES) * len(COMPLEXITIES)
    if len(entries) != expected:
        problems.append(f"[1] expected {expected} rows, found {len(entries)}")
    id_counts = Counter()
    for e in entries:
        id_ = e.row.get("id")
        if isinstance(id_, str) and id_:
            id_counts[id_] += 1
        else:
            problems.append(f"[1] line {e.line}: id is missing or not a non-empty string")
    for id_, n in sorted(id_counts.items()):
        if n > 1:
            problems.append(f"[1] id {id_!r} appears {n} times")

    for e in entries:
        task = e.row.get("task_type")
        if not (isinstance(task, str) and task in TASK_TYPES):
            problems.append(f"[2] {name(e)}: task_type {task!r} is not one of {', '.join(TASK_TYPES)}")
    for e in entries:
        cx = e.row.get("complexity")
        if not (isinstance(cx, str) and cx in COMPLEXITIES):
            problems.append(f"[3] {name(e)}: complexity {cx!r} is not one of {', '.join(COMPLEXITIES)}")

    cell_counts = Counter(c for c in (cell_of(e.row) for e in entries) if c)
    for task in TASK_TYPES:
        for cx in COMPLEXITIES:
            if cell_counts[(task, cx)] != per_cell:
                problems.append(f"[4] cell {task}/{cx} has {cell_counts[(task, cx)]} rows, expected {per_cell}")

    seen = defaultdict(list)
    for e in entries:
        prompt = e.row.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            seen[normalize(prompt)].append(name(e))
    for ids in seen.values():
        if len(ids) > 1:
            problems.append(f"[5] duplicate prompt in {', '.join(ids)}")

    for e in entries:
        prompt = e.row.get("prompt")
        if not (isinstance(prompt, str) and prompt.strip()):
            problems.append(f"[6] {name(e)}: prompt is not a non-empty string")

    return problems


def word_set(prompt: str) -> set:
    return set(re.findall(r"\w+", prompt.lower()))


def find_near_duplicates(entries, threshold: float = NEAR_DUP_JACCARD) -> list:
    """Check 7: (i, j, jaccard) for entry indices i < j in the same cell."""
    by_cell = defaultdict(list)
    for i, e in enumerate(entries):
        cell, prompt = cell_of(e.row), e.row.get("prompt")
        if cell and isinstance(prompt, str):
            words = word_set(prompt)
            if words:
                by_cell[cell].append((i, words))
    pairs = []
    for members in by_cell.values():
        for (i, a), (j, b) in combinations(members, 2):
            jaccard = len(a & b) / len(a | b)
            if jaccard >= threshold:
                pairs.append((i, j, jaccard))
    return sorted(pairs)


def group_roots(n: int, pairs) -> list:
    """Union-find over entry indices; returns each index's group root."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in pairs:
        parent[find(i)] = find(j)
    return [find(i) for i in range(n)]


def split_entries(entries, pairs, seed: int, val_per_cell: int = VAL_PER_CELL):
    """Stratified split keeping near-duplicate groups together. Needs validated entries."""
    roots = group_roots(len(entries), pairs)
    cells = defaultdict(lambda: defaultdict(list))
    for i, e in enumerate(entries):
        cells[cell_of(e.row)][roots[i]].append(i)

    train, val = [], []
    for task in TASK_TYPES:
        for cx in COMPLEXITIES:
            # Order groups by their lowest id before shuffling, so the result
            # doesn't depend on the row order in the input file.
            groups = sorted(cells[(task, cx)].values(), key=lambda g: min(entries[i].row["id"] for i in g))
            random.Random(f"{seed}:{task}:{cx}").shuffle(groups)
            n_val = 0
            for g in groups:
                to_val = n_val + len(g) <= val_per_cell
                (val if to_val else train).extend(entries[i] for i in g)
                if to_val:
                    n_val += len(g)
    train.sort(key=lambda e: e.row["id"])
    val.sort(key=lambda e: e.row["id"])
    return train, val


def summarize(entries, train, val, pairs, seed: int, input_path, val_per_cell: int = VAL_PER_CELL) -> dict:
    total, in_train, in_val = (Counter(cell_of(e.row) for e in es) for es in (entries, train, val))
    per_cell = {
        task: {
            cx: {"total": total[(task, cx)], "train": in_train[(task, cx)], "val": in_val[(task, cx)]}
            for cx in COMPLEXITIES
        }
        for task in TASK_TYPES
    }
    roots = group_roots(len(entries), pairs)
    members = defaultdict(list)
    for i, root in enumerate(roots):
        members[root].append(entries[i].row["id"])
    groups = sorted(sorted(ids) for ids in members.values() if len(ids) > 1)
    shortfalls = [
        {"task_type": task, "complexity": cx, "val": cells["val"], "target": val_per_cell}
        for task, row in per_cell.items()
        for cx, cells in row.items()
        if cells["val"] != val_per_cell
    ]
    return {
        "input": str(input_path),
        "seed": seed,
        "total_rows": len(entries),
        "train_rows": len(train),
        "val_rows": len(val),
        "per_cell": per_cell,
        "near_duplicates": {
            "jaccard_threshold": NEAR_DUP_JACCARD,
            "pairs": len(pairs),
            "groups": len(groups),
            "pair_list": [
                {"a": entries[i].row["id"], "b": entries[j].row["id"], "jaccard": round(jac, 3)}
                for i, j, jac in pairs
            ],
            "group_list": groups,
        },
        "val_shortfalls": shortfalls,
    }


def format_summary(s: dict) -> str:
    nd = s["near_duplicates"]
    lines = [
        f"Input: {s['input']}",
        f"Seed: {s['seed']}",
        f"Rows: {s['total_rows']} (train {s['train_rows']}, val {s['val_rows']})",
        f"Near-duplicates (word-set Jaccard >= {nd['jaccard_threshold']}, same cell): "
        f"{nd['pairs']} pairs in {nd['groups']} groups",
    ]
    lines += [f"  {p['a']} ~ {p['b']} ({p['jaccard']})" for p in nd["pair_list"]]
    lines += ["", f"{'cell':<36}{'total':>6}{'train':>7}{'val':>5}"]
    for task, row in s["per_cell"].items():
        for cx, c in row.items():
            lines.append(f"{task + ' / ' + cx:<36}{c['total']:>6}{c['train']:>7}{c['val']:>5}")
    for sf in s["val_shortfalls"]:
        lines.append(f"Val short in {sf['task_type']}/{sf['complexity']}: {sf['val']} of {sf['target']}")
    return "\n".join(lines)


def write_lines(path: Path, entries) -> None:
    path.write_bytes(("\n".join(e.raw for e in entries) + "\n").encode("utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, help="default: data/trainval.jsonl, else data/trainval.json1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    ap.add_argument("--dry-run", action="store_true", help="validate and print the plan, write nothing")
    ap.add_argument("--out-dir", type=Path, default=REPO / "data", help="where train/val/report go (default: data)")
    args = ap.parse_args(argv)

    path = args.input or next((p for p in DEFAULT_INPUTS if p.is_file()), DEFAULT_INPUTS[0])
    if not path.is_file():
        print(f"input not found: {path} (use --input)", file=sys.stderr)
        return 2
    try:
        entries, problems = load_entries(path)
    except ValueError as e:
        print(e)
        return 1
    problems += validate(entries)
    pairs = find_near_duplicates(entries)

    if problems:
        print(f"Validation FAILED: {len(problems)} problems in {path}")
        print("\n".join(f"  {p}" for p in problems))
        print(f"Near-duplicate pairs (report only): {len(pairs)}")
        print("Nothing written.")
        return 1
    print(f"Validation passed: {len(entries)} rows, {EXPECTED_PER_CELL} per cell, unique ids, no duplicate prompts.")

    outputs = [args.out_dir / n for n in ("train.jsonl", "val.jsonl", "split_report.json")]
    existing = [p for p in outputs if p.exists()]
    if existing and not args.force and not args.dry_run:
        print(f"Refusing to overwrite {', '.join(str(p) for p in existing)}; pass --force.", file=sys.stderr)
        return 2

    train, val = split_entries(entries, pairs, args.seed)
    summary = summarize(entries, train, val, pairs, args.seed, path)
    print(format_summary(summary))

    if args.dry_run:
        print("\nDry run: nothing written.")
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_lines(outputs[0], train)
    write_lines(outputs[1], val)
    outputs[2].write_bytes((json.dumps(summary, indent=2) + "\n").encode("utf-8"))
    print("\nWrote " + ", ".join(str(p) for p in outputs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
