# Tabular Lab experiment report

Strategy: **random**; provider: **deepseek**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.000000; calls: 0; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.4314942729736819 | 3.94 |
| 1 | logistic | ok | 0.7804373713745776 | 4.05 |
| 2 | catboost | ok | 0.79663534375374 | 23.69 |
| 3 | logistic | ok | 0.7500301010723955 | 5.38 |
| 4 | logistic | ok | 0.7784772714341638 | 6.58 |
| 5 | hist_gradient_boosting | ok | 0.8030318974231423 | 8.89 |
| 6 | logistic | ok | 0.7784772714341638 | 6.11 |
| 7 | extra_trees | ok | 0.7645412852280511 | 21.06 |

## Decisions

- Trial 0: Majority-class reference.
- Trial 1: Linear baseline with training-only preprocessing.
- Trial 2: Test native categorical boosting.
- Trial 3: Test a preprocessing alternative to the linear baseline.
- Trial 4: Test linear regularization and class weighting.
- Trial 5: Test boosted nonlinear interactions.
- Trial 6: Test a preprocessing alternative to the linear baseline.
- Trial 7: Test nonlinear tree interactions.

## Final test (not used for selection)

```json
{
  "accuracy": 0.8741477796204165,
  "macro_f1": 0.8128440497603318,
  "roc_auc": 0.9255061002600941
}
```

## Limitations

Finite search space; observed training devices: cpu, gpu; one validation split per run. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
