# Tabular Lab experiment report

Strategy: **random**; provider: **deepseek**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.000000; calls: 0; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.18666666666666668 | 4.17 |
| 1 | logistic | ok | 0.945824706694272 | 4.31 |
| 2 | logistic | ok | 0.945824706694272 | 4.42 |
| 3 | hist_gradient_boosting | ok | 0.9717813051146384 | 6.03 |
| 4 | logistic | ok | 0.945824706694272 | 5.67 |
| 5 | logistic | ok | 0.945824706694272 | 5.56 |
| 6 | catboost | ok | 0.9717813051146384 | 21.11 |
| 7 | logistic | ok | 0.945824706694272 | 5.09 |

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
  "accuracy": 1.0,
  "macro_f1": 1.0
}
```

## Limitations

Finite search space; observed training devices: cpu, gpu; one validation split per run. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
