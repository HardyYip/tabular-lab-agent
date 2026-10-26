# Tabular Lab experiment report

Strategy: **codegen**; provider: **deepseek**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.034288; calls: 3; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.4314942729736819 | 5.98 |
| 1 | logistic | ok | 0.7804373713745776 | 5.64 |
| 2 | generated | ok | 0.8096158021043067 | 12.36 |
| 3 | generated | ok | 0.8067641464574418 | 8.34 |
| 4 | generated | ok | 0.8127111661510814 | 10.06 |

## Decisions

- Trial 0: Majority-class reference.
- Trial 1: Linear baseline with training-only preprocessing.
- Trial 2: CatBoost with native categorical handling typically outperforms a scaled linear baseline on this adult-style dataset for macro-F1, while still fitting preprocessing within the pipeline.
- Trial 3: CatBoost with native categorical handling and ordered boosting plus moderate depth/iterations and internal early stopping should exceed the current macro-F1. Add frequency-based feature engineering inside the pipeline and pass categorical columns as strings with missing filled to a sentinel so CatBoost handles them natively. Use thread_count=4 and allow_writing_files=False.
- Trial 4: A tuned CatBoost with native categorical handling, plus engineered features (capital net, hours buckets) and robust preprocessing inside the pipeline, should improve macro-F1 over prior trials. I'll add a feature-engineering transformer and pass categoricals as strings.

## Final test (not used for selection)

```json
{
  "accuracy": 0.8735949880228487,
  "macro_f1": 0.8148827405906285,
  "roc_auc": 0.9277821478761845
}
```

## Limitations

Generated pipelines in CPU Docker containers; one validation split per run. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
