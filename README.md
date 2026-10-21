# heimdall-lora-lab

A week-long LoRA fine-tuning exercise: a prompt **intent + complexity classifier** for the LLM gateway (Bifrost) in my homelab.

> In Norse myth, Heimdall guards the Bifröst and decides who crosses. This model reads a prompt and helps the gateway decide where it should go.

**Status:** in progress (week of 2026-10-19). Results will be filled in as they exist.

---

## What this is

- A small classifier, fine-tuned with **NVIDIA NeMo + LoRA**, that labels each prompt passing through the gateway:
  ```json
  {"intent": "<one of a small label set>", "complexity": "simple" | "complex"}
  ```
- **Why:** in Bifrost, routing (cheap model vs. strong model) and downstream batch processing / sentiment categorization both depend on knowing what a prompt is for and how hard it is.
- **How it's judged:** against simple baselines, plus a 50-example human review.

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

## This week: must ship

"Done" means **measured and written down**. There are no pass/fail thresholds.

- [ ] **Dataset:** synthetic prompts, labeled against written guidelines, split into train/val/test → [docs/data.md](docs/data.md)
- [ ] **Baselines:** at least a rules-only baseline, scored on the test set → [docs/baselines.md](docs/baselines.md)
- [ ] **One LoRA run:** trained with NeMo, config and cost logged → [docs/training.md](docs/training.md)
- [ ] **Comparison:** LoRA vs. baselines on the same test set, with error analysis → [docs/results.md](docs/results.md)
- [ ] **Human eval:** 50 examples reviewed by hand → [docs/human-eval.md](docs/human-eval.md)
- [ ] **Write-up:** what worked, what didn't, what I'd do next → [docs/report.md](docs/report.md)

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

Running notes, decisions and costs go in the run log → [docs/run-log.md](docs/run-log.md)

## Results

_TBD. Filled in from [docs/results.md](docs/results.md) once the comparison exists._

## Future work

- Split intent and complexity into separate adapters that can be retrained and redeployed independently
- Wire the classifier into Bifrost routing for real (homelab) traffic

## Docs index

| Doc | Read when you need… |
|---|---|
| [docs/run-log.md](docs/run-log.md) | Day-by-day decisions, costs, surprises |
| [docs/data.md](docs/data.md) | Label taxonomy, guidelines, how data was generated and split |
| [docs/baselines.md](docs/baselines.md) | How the baselines work |
| [docs/training.md](docs/training.md) | NeMo setup, LoRA config, GPU rental details |
| [docs/results.md](docs/results.md) | Scores and error analysis |
| [docs/human-eval.md](docs/human-eval.md) | Human review protocol and findings |
| [docs/report.md](docs/report.md) | The write-up |