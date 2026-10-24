# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

**heimdall-lora-lab**: a one-week exercise to fine-tune **Qwen2.5-1.5B-Instruct** with LoRA (NVIDIA NeMo) to classify LLM gateway prompts as `{task_type, complexity}`.

The classifier serves the Bifrost LLM/MCP gateway in two ways:
- **Routing:** complexity informs which model tier a prompt goes to.
- **Attribution:** it is the Tier 2 batch classifier in the productivity attribution framework, feeding batch processing and sentiment categorization.

This is a homelab portfolio project. Write docs and commit messages plainly. Never describe it as production-grade or overstate results.

**Label space:** 6 task types × 3 complexity levels = 18 combinations.
- `task_type`: `code_generation`, `debugging`, `code_explanation_review`, `writing_docs`, `data_analysis`, `research_qa`
- `complexity`: `low`, `medium`, `high`

## Where things live

- `README.md` is the short index. Read it first.
- Open files under `docs/` only when the current task needs them:
  - `docs/scope.md`: scope, must-ship vs stretch, timeline
  - `docs/labeling-guidelines.md`: full labeling rules, tie-break order, canonical examples
  - `docs/run-log.md`: log every working session here (decisions, costs, surprises)

## How to work with me

- Work one step at a time, following the current day in `docs/scope.md`.
- Ask at most one or two questions at a time, and propose a sensible default for each so I can answer "yes."
- Don't ask deep technical questions up front. Build outlines and fill them in progressively.
- "Done" means **measured and written down**. There are no pass/fail thresholds. If a baseline beats the LoRA model, that is a result to report, not something to fix by tuning against the test set.
- If time slips, cut stretch items first (second run, second rater, hybrid validator, GGUF/Ollama serving). Never cut the baselines or the human eval.

## Key constraints

- Synthetic data only. No real company prompts or traces.
- Training runs on a rented 4090-class GPU (~$25 budget). Log GPU hours and cost in `docs/run-log.md`. Data generation, eval and serving run locally on AMD.
- Stdlib only for `baselines/rules.py` so it can run anywhere the gateway does.
- Tune rules, prompt template and hyperparameters on val only. Score test once. Details and versioning tags are in `docs/run-log.md`.

## Commands

Run the test suite:
```
python -m unittest tests/test_rules.py
```

Run a single test:
```
python -m unittest tests.test_rules.GuidelineExamples.test_task_type_examples
```

Score the rules baseline against a JSONL file:
```
python baselines/rules.py data/test.jsonl > results/rules_preds.jsonl
```

## Architecture

**Comparisons being tracked** (in order):
1. Rules-only baseline (`baselines/rules.py`)
2. Prompted base model (Qwen2.5-1.5B-Instruct, no fine-tune)
3. Prompted frontier model
4. LoRA fine-tuned model
5. Blind human A/B review on ~50 borderline prompts

All comparisons are scored on the same held-out test set.

**`baselines/rules.py`** is the rules-only classifier. It uses weighted keyword/regex signals per class, accumulates scores, and breaks ties via the fixed `PRIORITY` list from the guidelines. Complexity uses a separate signal set with a "pick lower by default" rule. Input is JSONL with a `"prompt"` field; output is JSONL with `{id, task_type, complexity}`.

**`tests/test_rules.py`** is a unittest suite that asserts the rules baseline correctly classifies every canonical example from `docs/labeling-guidelines.md`. Any change to `TASK_RULES`, `HIGH_RULES`, or `MEDIUM_RULES` should keep these tests passing.

Data, training configs, eval scripts and model artifacts will live under `data/`, `training/`, `results/` and `models/` respectively (not yet created).