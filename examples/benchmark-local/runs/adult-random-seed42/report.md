# Tabular Lab experiment report

Strategy: **random**; provider: **mock**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.000000; calls: 0; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.4314942729736819 | 3.55 |
| 1 | logistic | ok | 0.7804373713745776 | 3.25 |
| 2 | catboost | ok | 0.79663534375374 | 28.38 |
| 3 | logistic | ok | 0.7500301010723955 | 4.75 |
| 4 | logistic | ok | 0.7784772714341638 | 3.19 |
| 5 | hist_gradient_boosting | ok | 0.8030318974231423 | 4.53 |
| 6 | logistic | ok | 0.7784772714341638 | 3.34 |
| 7 | extra_trees | ok | 0.7645412852280511 | 15.81 |

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

Finite search space; one validation split per run; CPU execution. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
