# Labeling guidelines

Every prompt gets exactly one `task_type` and one `complexity`. The same rules
apply to synthetic data generation, the rules baseline and human raters.

## Main rule: label by the main thing the user wants back

Ask "what is the user expecting to receive?" and label that.

- "Here's my stack trace, explain why it fails and write a fix" -> `debugging` (the fix is the deliverable).
- "Explain what this function does" -> `code_explanation_review`.

**Tie-break.** If two parts carry equal weight, the higher class in this list wins:

1. `debugging`
2. `code_generation`
3. `data_analysis`
4. `code_explanation_review`
5. `writing_docs`
6. `research_qa`

## task_type

| Label                     | Definition                                                                            | Example                                                                       |
| ------------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `code_generation`         | Write new code, scripts, configs or queries from a description.                       | "Write a Python function that retries an HTTP call with exponential backoff." |
| `debugging`               | Find or fix the cause of an error, failing test or unexpected behavior.               | "My Terraform apply fails with 'cycle detected', here's the plan output."     |
| `code_explanation_review` | Explain, review, critique or compare existing code without the main goal being a fix. | "Review this PR diff for security issues."                                    |
| `writing_docs`            | Write or edit prose: docs, READMEs, emails, summaries, release notes.                 | "Turn these bullet points into a runbook for on-call."                        |
| `data_analysis`           | Interpret, transform or summarize data, metrics, logs or tables.                      | "Here's a CSV of p95 latency by region, what changed last week?"              |
| `research_qa`             | Answer a conceptual or factual question without code or data given.                   | "What's the difference between a Kubernetes Deployment and a StatefulSet?"    |

**Common confusions**

- Code pasted plus "why doesn't this work?" -> `debugging`, not `code_explanation_review`.
- "How do I do X in Python?" with no code given -> `code_generation` if they want code back, `research_qa` if they want a concept explained.
- Logs pasted to find a failure -> `debugging`. Logs pasted to find trends or counts -> `data_analysis`.
- Docstrings or code comments only -> `writing_docs`.

## complexity

Judge the **effort needed to do the task well**, not how long the prompt is. A short
prompt can be `high` ("design a multi-region failover for our Postgres"), and a long
paste can be `low` ("what does line 3 do?").

| Label    | Anchor                                                                                                     | Signals                                                                                    |
| -------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `low`    | One self-contained ask that a competent engineer answers in minutes with no context gathering.             | A single function or fact, an obvious fix, a short rewrite.                                |
| `medium` | Several steps, or needs reasoning over provided context (a file, a diff, a table), but the scope is clear. | Multi-step logic, moderate pasted context, a couple of constraints.                        |
| `high`   | Spans several components, involves design tradeoffs, ambiguous requirements or large context.              | Architecture or system design, cross-service bugs, migrations, "what's the best approach". |

**Between two levels:** pick the lower one unless a `high`/`medium` signal is clearly present.
This keeps labels conservative and makes disagreements easy to spot in error analysis.

## Output format

```json
{ "task_type": "debugging", "complexity": "medium" }
```
