"""Shared helpers for the scoring, prompting and data-prep scripts. Stdlib only."""

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = REPO / "shared" / "template_v1.txt"

TASK_TYPES = [
    "code_generation",
    "debugging",
    "code_explanation_review",
    "writing_docs",
    "data_analysis",
    "research_qa",
]
COMPLEXITIES = ["low", "medium", "high"]

_JSON_FENCE = re.compile(r"^```json[ \t]*\r?\n(.*?)\r?\n?[ \t]*```$", re.DOTALL)


def read_jsonl(path) -> list:
    """Read a JSONL file. Tolerates a BOM, \r\n line endings and blank lines."""
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()  # also drops a stray trailing \r
            if line:
                rows.append(json.loads(line))
    return rows


def load_template() -> str:
    # Text mode folds \r\n; strip() drops the file's trailing newline.
    return TEMPLATE_PATH.read_text(encoding="utf-8").strip()


def parse_output(text):
    """Parse a model reply into ({"task_type", "complexity"}, True), or (None, False).

    Strip whitespace and one surrounding ```json fence if present, then
    json.loads, then require both keys with allowed values. Anything else is a
    parse failure. Extra keys are ignored.
    """
    if not isinstance(text, str):
        return None, False
    text = text.strip()
    m = _JSON_FENCE.match(text)
    if m:
        text = m.group(1).strip()
    try:
        obj = json.loads(text)
    except ValueError:
        return None, False
    if not isinstance(obj, dict):
        return None, False
    task, complexity = obj.get("task_type"), obj.get("complexity")
    if not (isinstance(task, str) and task in TASK_TYPES):
        return None, False
    if not (isinstance(complexity, str) and complexity in COMPLEXITIES):
        return None, False
    return {"task_type": task, "complexity": complexity}, True
