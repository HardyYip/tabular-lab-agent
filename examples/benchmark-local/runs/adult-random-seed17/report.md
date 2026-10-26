# Tabular Lab experiment report

Strategy: **random**; provider: **mock**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.000000; calls: 0; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.4314942729736819 | 3.08 |
| 1 | logistic | ok | 0.7804373713745776 | 3.03 |
| 2 | logistic | ok | 0.7804373713745776 | 3.19 |
| 3 | hist_gradient_boosting | ok | 0.7946087435942714 | 4.94 |
| 4 | logistic | ok | 0.7784772714341638 | 3.31 |
| 5 | logistic | ok | 0.7500301010723955 | 4.44 |
| 6 | catboost | ok | 0.8068510438175169 | 32.86 |
| 7 | logistic | ok | 0.778108066886554 | 4.44 |

## Decisions

- Trial 0: Majority-class reference.
- Trial 1: Linear baseline with training-only preprocessing.
- Trial 2: Test a preprocessing alternative to the linear baseline.
- Trial 3: Test boosted nonlinear interactions.
- Trial 4: Test linear regularization and class weighting.
- Trial 5: Test a preprocessing alternative to the linear baseline.
- Trial 6: Test native categorical boosting.
- Trial 7: Test linear regularization and class weighting.

## Final test (not used for selection)

```json
{
  "accuracy": 0.8745163073521283,
  "macro_f1": 0.8147046506069705,
  "roc_auc": 0.9261659642099394
}
```

## Limitations

Finite search space; one validation split per run; CPU execution. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
