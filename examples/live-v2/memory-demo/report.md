# Tabular Lab experiment report

Strategy: **codegen**; provider: **deepseek**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.038528; calls: 4; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.424 | 4.67 |
| 1 | logistic | ok | 0.8074866310160428 | 4.39 |
| 2 | generated | ok | 0.8301441677588467 | 2.91 |
| 3 | generated | failed | — | 2.95 |
| 4 | generated | ok | 0.8888888888888888 | 2.98 |

## Decisions

- Trial 0: Majority-class reference.
- Trial 1: Linear baseline with training-only preprocessing.
- Trial 2: Logistic regression with median imputation, one-hot encoding for the categorical column, and standardization, all fitted inside a pipeline. This mirrors the strong linear baseline while robustly handling missing values and unseen categories.
- Trial 3: Improve on the best logistic baseline by using a CatBoost classifier with native categorical/missing handling alongside a regularized logistic model averaged in an ensemble; CatBoost captures nonlinearities while the linear model stabilizes small-data performance. Keep all preprocessing inside the pipeline and use 4 threads with allow_writing_files=False.
- Trial 4: The CatBoost repair failed because its ColumnTransformer output remained a numeric/object mixed numpy array with object dtype columns CatBoost could not parse as floats. I will instead keep the best logistic baseline (which scored macro_f1 0.830) and add a small, safe ensemble: a tuned logistic model plus a HistGradientBoosting model on fully numeric one-hot-encoded data, all fitted inside the pipeline. This adds nonlinear capacity without CatBoost's dtype fragility, handles missing values/unseen categories via imputers with handle_unknown, and stays within 4 threads.

## Final test (not used for selection)

```json
{
  "accuracy": 0.9166666666666666,
  "macro_f1": 0.8794642857142857,
  "roc_auc": 0.9900695134061569
}
```

## Limitations

Generated pipelines in CPU Docker containers; one validation split per run. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
