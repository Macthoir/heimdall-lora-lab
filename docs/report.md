# heimdall-lora-lab: write-up

Homelab experiment on synthetic data, not a production evaluation.

Fill each section from the named source, not from memory. Delete these notes when the report is done.

## Summary

TBD: write last, 3-4 sentences. What was built, the headline comparison, the main caveat.

## Setup

- **Labels:** 6 task types × 3 complexity levels. TBD: one paragraph, from `docs/labeling-guidelines.md`.
- **Data:** TBD: generation by target cell, axes (persona, domain, length, noise, artifact), relabel model, disagreement rate. Source: `docs/run-log.md` and `data/MANIFEST.json`.
- **Split:** TBD: train/val sizes, dedupe. There is no separate test set: `data/val.jsonl` (108 rows, 6 per cell) is the held-out test set and is scored once per model. See Limitations. Source: `data/MANIFEST.json`.
- **Shared template:** TBD: system prompt, JSON-only output, malformed output counts as wrong. Source: `shared/template_v1.txt`.
- **Versions:** TBD: git SHA, `data-v1`, `template-v1`, `rules-v1`.

## Results

TBD: source is the scoring script output in `results/`.

| Model                          | task_type acc | complexity acc | cell exact-match | task_type macro-F1 | complexity off-by-one | parse failures |
| ------------------------------ | ------------- | -------------- | ---------------- | ------------------ | --------------------- | -------------- |
| Rules (`rules-v1`)             | TBD           | TBD            | TBD              | TBD                | TBD                   | TBD            |
| Prompted Qwen2.5-1.5B-Instruct | TBD           | TBD            | TBD              | TBD                | TBD                   | TBD            |
| Prompted Opus 5.5              | TBD           | TBD            | TBD              | TBD                | TBD                   | TBD            |
| LoRA                           | TBD           | TBD            | TBD              | TBD                | TBD                   | TBD            |

Each accuracy is reported with a 95% Wilson interval; macro-F1 has no interval. 108 test prompts, so differences of a few points are within noise. No pass/fail thresholds.

TBD: 18-cell confusion matrix for the LoRA model and for the best baseline.

## Error analysis

TBD: which cells and which axis fail, and whether errors cluster on borderline prompts. Source: `results/`.

## Human eval

TBD: 50-prompt blind A/B result. Agreement between hand label, target cell and relabel, and disagreement rate by cell. Source: `docs/run-log.md`.

## Cost and effort

TBD: GPU hours and dollars against the $25 budget, total hours against the 15-hour budget. Source: `docs/run-log.md`.

## Limitations

- Synthetic data only. No real gateway traffic.
- Train excludes prompts where the target cell and relabel disagreed, so it is easier than real traffic.
- Generator and some baselines are Claude family models, so same-family bias is possible. The relabeler is a different family.
- The "test set" is `data/val.jsonl`: 108 prompts, 6 per cell, with the generator's target labels as ground truth, not hand labels. This replaces the planned separate 180-prompt hand-labeled test set, which was cut for time. It was never used for tuning (`rules-v1` was frozen untuned). The GPT-6 relabel disagrees with the target on 21 of the 108 rows, so some ground truth is arguable; the human A/B review is the check on that. 6 per cell makes per-cell scores very noisy.
- The same val file also supplied the training-loss curve for the LoRA run (minor leak). The final checkpoint is used; none was picked by val accuracy.
- GPT-6 is not a baseline: it relabeled the data, so it would be a biased judge. The frontier baseline is Opus 5.5 only, run with `effort=low` and no temperature (Opus 5.5 rejects sampling parameters).
- TBD: anything that changed after the test set was scored, and which version saw it.

## What I'd do next

TBD: write last.
