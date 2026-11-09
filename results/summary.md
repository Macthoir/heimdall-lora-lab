Scored on `data/val.jsonl` (108 rows, generator target labels as ground truth), used as the held-out test set. Accuracies show a 95% Wilson interval. Written by `scripts/score.py`.

| Model | task_type acc | complexity acc | cell exact-match | task_type macro-F1 | complexity off-by-one | parse failures |
|---|---|---|---|---|---|---|
| qwen_prompted | 0.639 (0.545-0.723) | 0.435 (0.346-0.529) | 24/108 0.222 (0.154-0.309) | 0.643 | 1.000 | 2 |
| lora | 0.991 (0.949-0.998) | 0.806 (0.721-0.869) | 86/108 0.796 (0.711-0.861) | 0.991 | 0.952 | 0 |
