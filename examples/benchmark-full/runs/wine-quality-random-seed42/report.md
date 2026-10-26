# Tabular Lab experiment report

Strategy: **random**; provider: **deepseek**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.000000; calls: 0; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.20244328097731237 | 4.05 |
| 1 | logistic | ok | 0.5472515556042574 | 3.91 |
| 2 | catboost | ok | 0.5703661647466662 | 16.11 |
| 3 | logistic | ok | 0.5327276390568917 | 4.20 |
| 4 | logistic | ok | 0.5356593705701027 | 3.89 |
| 5 | hist_gradient_boosting | ok | 0.568796825123212 | 4.25 |
| 6 | logistic | ok | 0.5356593705701027 | 3.86 |
| 7 | extra_trees | ok | 0.5780714593566784 | 4.76 |

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
  "accuracy": 0.6325187969924813,
  "macro_f1": 0.6042710508827245
}
```

## Limitations

Finite search space; observed training devices: cpu, gpu; one validation split per run. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
