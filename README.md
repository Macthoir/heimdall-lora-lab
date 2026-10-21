# heimdall-lora-lab

A week-long LoRA fine-tuning exercise: a prompt **task type + complexity classifier** for the LLM gateway (Bifrost) in my homelab.

> In Norse myth, Heimdall guards the Bifröst and decides who crosses. This model reads a prompt and helps the gateway decide where it should go.

**Status:** in progress (week of 2026-10-19). Results will be filled in as they exist.

---

## What this is

- A small classifier, fine-tuned from Qwen2.5-1.5B-Instruct with **NVIDIA NeMo + LoRA**, that labels each prompt passing through the gateway:
  ```json
  {"task_type": "<one of 6>", "complexity": "low" | "medium" | "high"}
  ```
  - `task_type`: `code_generation`, `debugging`, `code_explanation_review`, `writing_docs`, `data_analysis`, `research_qa`
  - 6 task types × 3 complexity levels = 18 combinations
- **Why:** in Bifrost, routing (cheap model vs. strong model) and downstream batch processing / sentiment categorization both depend on knowing what a prompt is for and how hard it is.
- **How it's judged:** against simple baselines on a hand-labeled test set, plus a 50-example human review.

## What this is not

- Not a production system. It's a homelab experiment built in about 15 hours.
- Not trained on real data. **All prompts are synthetic**, with no company prompts or traces.
- Not a claim that LoRA wins. If a baseline beats it, that's the finding.

## Constraints

| | |
|---|---|
| Data | Synthetic only |
| Compute | Rented NVIDIA GPU (4090-class) for training; local machine for data, eval, serving |
| Budget | ~$25 GPU rental |
| Time | ~15 hours across one week |

## Approach

- **Train/val:** prompts generated for a target label (task type × complexity), with varied persona, domain, length, noise and artifact. A model from a different family relabels them. Prompts where the two labels disagree are kept out of train.
- **Test:** a separate generation run, deduped against train, hand-labeled blind. The hand label is ground truth.
- **Comparison:** all models see the same system prompt and output format, and all are scored on the same test set. Test is scored once.

Decisions and their reasons are in the run log.

## This week: must ship

"Done" means **measured and written down**. There are no pass/fail thresholds.

- [ ] **Dataset:** synthetic prompts, labeled against written guidelines, split into train/val/test
- [ ] **Baselines:** rules-only, prompted base model, prompted frontier models, scored on the test set
- [ ] **One LoRA run:** trained with NeMo, config and cost logged
- [ ] **Comparison:** LoRA vs. baselines on the same test set, with error analysis
- [ ] **Human eval:** 50 borderline examples reviewed by hand
- [ ] **Write-up:** what worked, what didn't, what I'd do next (~2 pages)

## Stretch (cut first if time runs short)

- [ ] Second training run (v2 adapter), shown as a v1 → v2 swap behind the gateway
- [ ] Second human rater
- [ ] Hybrid validator (rules + model)
- [ ] Serve via GGUF / Ollama inside Bifrost

## Timeline

| Day | Step |
|---|---|
| Mon 10/19 | Lock scope, create repo and run log |
| Tue 10/20 | Label taxonomy, labeling guidelines, rules-only baseline, report skeleton |
| Wed 10/21 | Generate, review, split data; score baselines |
| Thu 10/22 | Set up NeMo, first LoRA run |
| Fri 10/23 | Evaluate, error analysis, optional second run |
| Sat 10/24 | Human evaluation, optional hybrid validator |
| Sun 10/25 | Write-up, cleanup, stretch goals |

## Results

_TBD. Filled in once the comparison exists._

## Future work

- Split task type and complexity into separate adapters that can be retrained and redeployed independently
- Wire the classifier into Bifrost routing for real (homelab) traffic

## Docs index

| Doc | Read when you need… |
|---|---|
| [docs/scope.md](docs/scope.md) | Scope, must-ship vs stretch, timeline |
| [docs/labeling-guidelines.md](docs/labeling-guidelines.md) | Label definitions, tie-break order, canonical examples |
| [docs/run-log.md](docs/run-log.md) | Day-by-day decisions, costs, surprises |
| [docs/report.md](docs/report.md) | The write-up (skeleton, filled in as results exist) |
| [GLOSSARY.md](GLOSSARY.md) | What terms like cell, relabel and borderline mean here |

More docs (baselines, training, results, human eval) will be added under `docs/` as each step is done.

## Commands

```
python -m unittest tests/test_rules.py
python baselines/rules.py data/test.jsonl > results/rules_preds.jsonl
```
