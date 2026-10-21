# Scope (locked 2026-10-20)

## Goal

Fine-tune Qwen2.5-1.5B-Instruct with LoRA (NVIDIA NeMo) to classify LLM gateway
prompts as `{task_type, complexity}`. It is the Tier 2 classifier in a zero-touch
AI productivity attribution framework built around the Bifrost LLM/MCP gateway.

## Label space: 6 x 3 = 18 combinations

- `task_type`: `code_generation`, `debugging`, `code_explanation_review`,
  `writing_docs`, `data_analysis`, `research_qa`
- `complexity`: `low`, `medium`, `high`

Exact definitions and tie-break rules go in `docs/labeling-guidelines.md` (Tue 10/20).

## Comparisons

1. Rules-only baseline
2. Prompted base model (Qwen2.5-1.5B-Instruct, no fine-tune)
3. Prompted frontier model
4. LoRA fine-tuned model
5. Blind human A/B review on ~50 borderline prompts

## Constraints

- Synthetic data only. No real company prompts or traces.
- Training on a rented 4090-class NVIDIA GPU. Data, eval and serving run locally (AMD).
- GPU budget: ~$25.
- Time: ~15 h total (2 h weeknights, 3-3.5 h weekend days). Ends ~2026-10-26.

## Must ship

Dataset, one LoRA run, baseline comparison, 50-example human eval, ~2 page write-up.

## Stretch (cut first if slipping)

Second training run, second rater, hybrid validator, GGUF/Ollama serving in Bifrost.

## Timeline

| Day       | Plan                                                           |
| --------- | -------------------------------------------------------------- |
| Mon 10/19 | Lock scope, repo, run log                                      |
| Tue 10/20 | Taxonomy, labeling guidelines, rules baseline, report skeleton |
| Wed 10/21 | Generate, review, split data; score baselines                  |
| Thu 10/22 | NeMo setup, first LoRA run                                     |
| Fri 10/23 | Evaluate, error analysis, optional second run                  |
| Sat 10/24 | Human eval, optional hybrid validator                          |
| Sun 10/25 | Write-up, cleanup, stretch                                     |
