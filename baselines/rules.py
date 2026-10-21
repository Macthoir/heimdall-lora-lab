"""Rules-only baseline for {task_type, complexity}.

Implements docs/labeling-guidelines.md with keyword and regex signals only.
Stdlib only, so it runs anywhere the gateway does.

Usage:
    python baselines/rules.py data/test.jsonl > results/rules_preds.jsonl

Input lines need a "prompt" field; an "id" field is passed through if present.
"""

import json
import re
import sys

# Tie-break order from the guidelines: earlier wins when scores are equal.
PRIORITY = [
    "debugging",
    "code_generation",
    "data_analysis",
    "code_explanation_review",
    "writing_docs",
    "research_qa",
]

# Each pattern that matches adds its weight to the class score.
TASK_RULES = {
    "debugging": [
        (r"\b(error|exception|traceback|stack ?trace|segfault|panic)\b", 2),
        (r"\b(fails?|failing|failed|broken|crash(es|ing)?|hangs?)\b", 2),
        (r"(doesn'?t|does not|isn'?t|won'?t|not) (work|working|run|start|compile)", 2),
        (r"\b(bug|fix|debug|troubleshoot|root cause)\b", 2),
        (r"\b(crashloopbackoff|oomkilled|timed? ?out|exit code|status code 5\d\d)\b", 2),
        (r"\bwhy (is|does|am|are)\b.*\b(wrong|fail|error|slow|null|none|undefined)\b", 1),
    ],
    "code_generation": [
        (r"\b(write|create|generate|implement|build|scaffold)\b.{0,40}\b(function|script|class|module|query|endpoint|api|cli|test|dockerfile|terraform|helm|manifest|yaml|config|pipeline|workflow|regex|code)\b", 3),
        (r"\bhow (do|can) i\b.{0,60}\b(in|with|using) (python|go|bash|sql|terraform|typescript|javascript|rust|java)\b", 1),
        (r"\b(give me|show me) (the )?(code|a snippet|an example)\b", 2),
        (r"\bconvert\b.{0,40}\b(to|into) (python|go|bash|terraform|typescript)\b", 2),
    ],
    "code_explanation_review": [
        (r"\b(review|critique|audit)\b.{0,30}\b(code|pr|diff|function|script|module|change|implementation)\b", 3),
        (r"\bwhat does (this|the following|that)\b", 2),
        (r"\b(explain|walk me through)\b.{0,30}\b(code|function|script|snippet|line|regex|query|diff)\b", 3),
        (r"\b(code smells?|readab(le|ility)|refactor suggestions?)\b", 1),
        (r"\bcompare (these|the two|both) (implementations|approaches|functions|versions)\b", 2),
    ],
    "writing_docs": [
        (r"\b(readme|docs|documentation|runbook|playbook|release notes|changelog|postmortem|rfc|design doc)\b", 2),
        (r"\b(email|slack message|announcement|blog post|status update)\b", 2),
        (r"\b(docstrings?|code comments?)\b", 3),
        (r"\b(rewrite|reword|proofread|polish|draft|summari[sz]e)\b", 1),
    ],
    "data_analysis": [
        (r"\b(csv|spreadsheet|dataset|dataframe|table below|these numbers|these metrics)\b", 2),
        (r"\b(trend|anomal(y|ies)|outliers?|distribution|correlat\w+|aggregate|breakdown)\b", 2),
        (r"\b(p50|p95|p99|latency by|cost by|usage by|per (region|service|team|day|week))\b", 2),
        (r"\b(analy[sz]e|interpret)\b.{0,30}\b(data|metrics|logs|results|numbers|usage)\b", 2),
    ],
    "research_qa": [
        (r"\b(what is|what are|what's) (a|an|the)?\b", 1),
        (r"\b(difference between|vs\.?|versus|pros and cons|tradeoffs? between)\b", 2),
        (r"\b(how does|how do)\b.{0,40}\b(work|differ)\b", 2),
        (r"\b(best practices?|when should i use|is it better to)\b", 1),
    ],
}

# Complexity: signal-based, with "pick the lower level" as the default.
HIGH_RULES = [
    r"\b(architect\w*|system design|design (a|an|the|our))\b",
    r"\b(migrat\w+|multi-?region|failover|disaster recovery|zero-?downtime)\b",
    r"\b(trade-?offs?|best approach|which approach|evaluate options)\b",
    r"\b(across|between) (services|teams|clusters|regions|accounts)\b",
    r"\b(end-to-end|at scale|scal(e|ing) to|high availability)\b",
]
MEDIUM_RULES = [
    r"\b(and then|after that|also|additionally|as well as)\b",
    r"\b(must|should|without|make sure|ensure)\b",
    r"\b(edge cases?|handle (errors|retries|failures))\b",
]
CODE_FENCE = re.compile(r"```")

_compiled_tasks = {
    label: [(re.compile(p, re.IGNORECASE), w) for p, w in rules]
    for label, rules in TASK_RULES.items()
}
_compiled_high = [re.compile(p, re.IGNORECASE) for p in HIGH_RULES]
_compiled_medium = [re.compile(p, re.IGNORECASE) for p in MEDIUM_RULES]


def has_pasted_context(prompt: str) -> bool:
    """Code fence or more than 8 lines suggests the user pasted code, logs or data."""
    return bool(CODE_FENCE.search(prompt)) or prompt.count("\n") > 8


def task_scores(prompt: str) -> dict:
    scores = {label: 0 for label in PRIORITY}
    for label, rules in _compiled_tasks.items():
        for pattern, weight in rules:
            if pattern.search(prompt):
                scores[label] += weight
    # Pasted context plus a "why" question is the classic debugging shape.
    if has_pasted_context(prompt) and re.search(r"\bwhy\b", prompt, re.IGNORECASE):
        scores["debugging"] += 1
    return scores


def classify_task(prompt: str) -> str:
    scores = task_scores(prompt)
    best = max(scores.values())
    if best == 0:
        return "research_qa"
    # PRIORITY order breaks ties.
    return next(label for label in PRIORITY if scores[label] == best)


def classify_complexity(prompt: str) -> str:
    if any(p.search(prompt) for p in _compiled_high):
        return "high"
    medium_hits = sum(1 for p in _compiled_medium if p.search(prompt))
    if has_pasted_context(prompt) or medium_hits >= 2:
        return "medium"
    return "low"


def classify(prompt: str) -> dict:
    return {"task_type": classify_task(prompt), "complexity": classify_complexity(prompt)}


def main(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out = {"id": row.get("id"), **classify(row["prompt"])}
            print(json.dumps(out))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python baselines/rules.py <input.jsonl>")
    main(sys.argv[1])
