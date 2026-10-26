# Tabular Lab experiment report

Strategy: **random**; provider: **deepseek**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.000000; calls: 0; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.18666666666666668 | 5.25 |
| 1 | logistic | ok | 1.0 | 4.30 |
| 2 | catboost | ok | 0.9229629629629629 | 15.75 |
| 3 | logistic | ok | 0.9474747474747475 | 4.23 |
| 4 | logistic | ok | 1.0 | 4.20 |
| 5 | hist_gradient_boosting | ok | 0.974320987654321 | 4.59 |
| 6 | logistic | ok | 1.0 | 5.52 |
| 7 | extra_trees | ok | 1.0 | 5.95 |

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
  "accuracy": 0.9722222222222222,
  "macro_f1": 0.9709618874773139
}
```

## Limitations

Finite search space; observed training devices: cpu, gpu; one validation split per run. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
