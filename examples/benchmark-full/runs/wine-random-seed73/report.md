# Tabular Lab experiment report

Strategy: **random**; provider: **deepseek**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.000000; calls: 0; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.18666666666666668 | 5.42 |
| 1 | logistic | ok | 1.0 | 5.48 |
| 2 | logistic | ok | 1.0 | 5.95 |
| 3 | logistic | ok | 0.9487179487179488 | 5.44 |
| 4 | catboost | ok | 1.0 | 23.27 |
| 5 | logistic | ok | 0.9487179487179488 | 5.16 |
| 6 | logistic | ok | 0.974320987654321 | 4.23 |
| 7 | logistic | ok | 1.0 | 4.30 |

## Decisions

- Trial 0: Majority-class reference.
- Trial 1: Linear baseline with training-only preprocessing.
- Trial 2: Test a preprocessing alternative to the linear baseline.
- Trial 3: Test a preprocessing alternative to the linear baseline.
- Trial 4: Test native categorical boosting.
- Trial 5: Test a preprocessing alternative to the linear baseline.
- Trial 6: Test a preprocessing alternative to the linear baseline.
- Trial 7: Test a preprocessing alternative to the linear baseline.

## Final test (not used for selection)

```json
{
  "accuracy": 1.0,
  "macro_f1": 1.0
}
```

## Limitations

Finite search space; observed training devices: cpu, gpu; one validation split per run. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
