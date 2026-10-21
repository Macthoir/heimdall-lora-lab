# heimdall-lora-lab: write-up

Homelab experiment on synthetic data, not a production evaluation.

Fill each section from the named source, not from memory. Delete these notes when the report is done.

## Summary

TBD: write last, 3-4 sentences. What was built, the headline comparison, the main caveat.

## Setup

- **Labels:** 6 task types × 3 complexity levels. TBD: one paragraph, from `docs/labeling-guidelines.md`.
- **Data:** TBD: generation by target cell, axes (persona, domain, length, noise, artifact), relabel model, disagreement rate. Source: `docs/run-log.md` and `data/MANIFEST.json`.
- **Split:** TBD: train/val sizes, test size (10 per cell, hand-labeled blind), dedupe. Source: `data/MANIFEST.json`.
- **Shared template:** TBD: system prompt, JSON-only output, malformed output counts as wrong. Source: `shared/template_v1.txt`.
- **Versions:** TBD: git SHA, `data-v1`, `template-v1`, `rules-v1`.

## Results

TBD: source is the scoring script output in `results/`.

| Model | task_type acc | complexity acc | cell exact-match | task_type macro-F1 | complexity off-by-one | parse failures |
|---|---|---|---|---|---|---|
| Rules (`rules-v1`) | TBD | TBD | TBD | TBD | TBD | TBD |
| Prompted Qwen2.5-1.5B-Instruct | TBD | TBD | TBD | TBD | TBD | TBD |
| Prompted Opus 5.5 | TBD | TBD | TBD | TBD | TBD | TBD |
| Prompted GPT-6 | TBD | TBD | TBD | TBD | TBD | TBD |
| LoRA | TBD | TBD | TBD | TBD | TBD | TBD |

Each accuracy and macro-F1 value is reported with a 95% Wilson interval. 180 test prompts, so differences of a few points are within noise. No pass/fail thresholds.

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
- Test set is small (10 per cell), so per-cell scores are noisy.
- TBD: anything that changed after the test set was scored, and which version saw it.

## What I'd do next

TBD: write last.
