# Tabular Lab experiment report

Strategy: **random**; provider: **deepseek**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.000000; calls: 0; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.3837837837837838 | 4.88 |
| 1 | logistic | ok | 0.9814935064935065 | 5.02 |
| 2 | logistic | ok | 0.9814935064935065 | 4.98 |
| 3 | hist_gradient_boosting | ok | 0.9444805194805195 | 7.72 |
| 4 | logistic | ok | 0.9723547005092555 | 5.05 |
| 5 | logistic | ok | 0.9629870129870131 | 5.28 |
| 6 | catboost | ok | 0.9444805194805195 | 19.38 |
| 7 | logistic | ok | 0.9542792973449907 | 3.97 |

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
  "accuracy": 0.9824561403508771,
  "macro_f1": 0.9811507936507937,
  "roc_auc": 0.9957010582010581
}
```

## Limitations

Finite search space; observed training devices: cpu, gpu; one validation split per run. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
