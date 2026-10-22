"""Generate synthetic gateway prompts, one (task_type, complexity) cell at a time.

Implements the data plan in docs/run-log.md (2026-10-20):
- Generator is Claude Sonnet 5.5. Labels come from the target cell
  (label_source = "target"). Train/val get relabeled by a different-family
  model in a later step; test gets hand-labeled.
- Diversity axes (persona, domain, length, noise, artifact) are sampled per
  prompt from a seeded RNG, so a run is reproducible given its seed.
- About a quarter of each cell is written as near-misses: prompts that look
  like a neighboring class or level but still belong to the target cell.

Two separate runs, with different seeds and output files:
    python scripts/generate.py --run trainval --per-cell 30 --seed 1 --out data/raw/trainval.jsonl
    python scripts/generate.py --run test     --per-cell 10 --seed 2 --out data/raw/test_unlabeled.jsonl

Add --dry-run to print the first request without calling the API.
The script is resumable: rerun the same command and it fills in only the
cells that are still short.

Needs: pip install anthropic, and ANTHROPIC_API_KEY in the environment.
"""

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

MODEL = "claude-sonnet-5-5"
REPO = Path(__file__).resolve().parent.parent
GUIDELINES = REPO / "docs" / "labeling-guidelines.md"

TASK_TYPES = [
    "code_generation",
    "debugging",
    "code_explanation_review",
    "writing_docs",
    "data_analysis",
    "research_qa",
]
COMPLEXITIES = ["low", "medium", "high"]

PERSONAS = [
    "backend engineer", "SRE on call", "platform engineer", "data engineer",
    "ML engineer", "frontend developer", "junior developer", "security engineer",
    "engineering manager", "analyst who codes a little",
]
DOMAINS = [
    "Kubernetes", "Terraform / IaC", "CI/CD pipelines", "Python services",
    "SQL and databases", "networking and DNS", "observability and alerting",
    "LLM gateway and model serving", "frontend web app", "batch data pipelines",
    "auth and secrets management", "Linux and bash",
]
LENGTHS = {
    "short": "one sentence, no pasted material beyond what the artifact needs",
    "medium": "two to five sentences",
    "long": "a longer message; any pasted artifact is 10-40 lines",
}
NOISE = {
    "clean": "clean, well-written",
    "sloppy": "lowercase, a couple of typos, minimal punctuation",
    "terse": "terse shorthand, like a Slack message",
    "rambly": "includes some irrelevant backstory before the actual ask",
    "non_native": "written by a fluent but non-native English speaker",
}
# Which pasted artifacts make sense for each task type. research_qa means no
# code or data is given, so it never gets one.
ARTIFACTS = {
    "code_generation": ["none", "none", "config file", "short code snippet to extend"],
    "debugging": ["stack trace", "error message", "log excerpt", "code snippet", "config file"],
    "code_explanation_review": ["code snippet", "diff", "config file", "SQL query"],
    "writing_docs": ["none", "bullet notes", "code snippet", "existing draft text"],
    "data_analysis": ["CSV table", "metrics table", "log excerpt", "JSON records"],
    "research_qa": ["none"],
}
# Neighbors a near-miss should resemble (from "Common confusions" in the guidelines).
TASK_NEIGHBORS = {
    "code_generation": ["research_qa", "debugging"],
    "debugging": ["code_explanation_review", "data_analysis"],
    "code_explanation_review": ["debugging", "writing_docs"],
    "writing_docs": ["code_explanation_review", "research_qa"],
    "data_analysis": ["debugging", "research_qa"],
    "research_qa": ["code_generation", "code_explanation_review"],
}
COMPLEXITY_NEIGHBORS = {"low": ["medium"], "medium": ["low", "high"], "high": ["medium"]}
NEAR_MISS_RATE = 0.25
BATCH = 5  # prompts per API request


def sample_spec(rng: random.Random, task: str, complexity: str) -> dict:
    spec = {
        "persona": rng.choice(PERSONAS),
        "domain": rng.choice(DOMAINS),
        "length": rng.choice(list(LENGTHS)),
        "noise": rng.choice(list(NOISE)),
        "artifact": rng.choice(ARTIFACTS[task]),
        "near_miss": None,
    }
    if rng.random() < NEAR_MISS_RATE:
        if rng.random() < 0.5:
            spec["near_miss"] = {"looks_like_task": rng.choice(TASK_NEIGHBORS[task])}
        else:
            spec["near_miss"] = {"looks_like_complexity": rng.choice(COMPLEXITY_NEIGHBORS[complexity])}
    return spec


def describe_spec(i: int, spec: dict) -> str:
    lines = [
        f"{i}. persona: {spec['persona']}; domain: {spec['domain']}",
        f"   length: {LENGTHS[spec['length']]}",
        f"   style: {NOISE[spec['noise']]}",
        f"   pasted artifact: {spec['artifact']}",
    ]
    nm = spec["near_miss"]
    if nm and "looks_like_task" in nm:
        lines.append(f"   near-miss: on the surface it should resemble a `{nm['looks_like_task']}` "
                     "request, but by the guidelines its correct label is still the target cell")
    elif nm:
        lines.append(f"   near-miss: on the surface it should look `{nm['looks_like_complexity']}` "
                     "complexity, but by the guidelines its correct complexity is still the target")
    return "\n".join(lines)


SYSTEM_TEMPLATE = """You write synthetic prompts that engineers might send through an internal LLM gateway.
The prompts are training and evaluation data for a classifier that labels each prompt
with a task_type and a complexity. All content is fictional: invent service names,
hostnames, data and code. Never include real company names, real credentials or real people.

These are the labeling guidelines the classifier and human raters follow:

<guidelines>
{guidelines}
</guidelines>"""

USER_TEMPLATE = """Write {n} prompts whose correct label under the guidelines is:
task_type = {task}
complexity = {complexity}

Write one prompt per spec below, in order. Each prompt is the user's full message,
including any pasted artifact inline. Do not mention labels, task types or complexity
in the prompts themselves.

{specs}

Return only a JSON array of {n} strings, with no commentary and no code fences."""


def build_request(guidelines: str, task: str, complexity: str, specs: list) -> dict:
    return {
        "system": SYSTEM_TEMPLATE.format(guidelines=guidelines),
        "user": USER_TEMPLATE.format(
            n=len(specs), task=task, complexity=complexity,
            specs="\n".join(describe_spec(i + 1, s) for i, s in enumerate(specs)),
        ),
    }


def parse_prompts(text: str, n: int) -> list:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    items = json.loads(text)
    if not isinstance(items, list) or len(items) != n or not all(isinstance(x, str) and x.strip() for x in items):
        raise ValueError(f"expected a JSON array of {n} non-empty strings")
    return [x.strip() for x in items]


def call_model(client, request: dict, retries: int = 4) -> str:
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=8000,
                temperature=1.0,
                system=[{"type": "text", "text": request["system"],
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": request["user"]}],
            )
            return "".join(b.text for b in resp.content if b.type == "text")
        except Exception as e:  # network or rate-limit errors: back off and retry
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt * 5
            print(f"  API error ({e.__class__.__name__}), retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def load_existing(path: Path):
    rows = []
    if path.exists():
        with path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
    return rows


def main(argv=None, client=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", choices=["trainval", "test"], required=True)
    ap.add_argument("--per-cell", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true", help="print the first request and exit")
    args = ap.parse_args(argv)

    guidelines = GUIDELINES.read_text()
    existing = load_existing(args.out)
    have = Counter((r["task_type"], r["complexity"]) for r in existing)
    seen = {normalize(r["prompt"]) for r in existing}
    next_num = len(existing) + 1
    prefix = "tv" if args.run == "trainval" else "test"

    if not args.dry_run and client is None:
        import anthropic  # imported late so --dry-run works without the SDK
        client = anthropic.Anthropic()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as out:
        for task in TASK_TYPES:
            for complexity in COMPLEXITIES:
                # Seed per cell, so resuming or reordering never changes a cell's specs.
                rng = random.Random(f"{args.seed}:{task}:{complexity}")
                needed = args.per_cell - have[(task, complexity)]
                # Burn the specs already used, so a resumed run doesn't repeat them.
                for _ in range(have[(task, complexity)]):
                    sample_spec(rng, task, complexity)
                while needed > 0:
                    specs = [sample_spec(rng, task, complexity) for _ in range(min(BATCH, needed))]
                    request = build_request(guidelines, task, complexity, specs)
                    if args.dry_run:
                        print("=== SYSTEM ===\n" + request["system"] + "\n\n=== USER ===\n" + request["user"])
                        return
                    try:
                        prompts = parse_prompts(call_model(client, request), len(specs))
                    except ValueError as e:
                        print(f"  {task}/{complexity}: bad output ({e}), retrying batch", file=sys.stderr)
                        continue
                    for prompt, spec in zip(prompts, specs):
                        if normalize(prompt) in seen:
                            continue  # exact duplicate; the cell loop asks for a replacement
                        seen.add(normalize(prompt))
                        row = {
                            "id": f"{prefix}-{next_num:04d}",
                            "prompt": prompt,
                            "task_type": task,
                            "complexity": complexity,
                            "label_source": "target",
                            "meta": {"run": args.run, "seed": args.seed, "model": MODEL, "spec": spec},
                        }
                        out.write(json.dumps(row) + "\n")
                        next_num += 1
                        needed -= 1
                    out.flush()
                    print(f"{task}/{complexity}: {args.per_cell - needed}/{args.per_cell}", file=sys.stderr)


if __name__ == "__main__":
    main()
