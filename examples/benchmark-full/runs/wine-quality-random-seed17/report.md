# Tabular Lab experiment report

Strategy: **random**; provider: **deepseek**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.000000; calls: 0; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.20244328097731237 | 4.44 |
| 1 | logistic | ok | 0.5655501875094702 | 4.25 |
| 2 | logistic | ok | 0.5655501875094702 | 4.17 |
| 3 | hist_gradient_boosting | ok | 0.592328705618356 | 4.97 |
| 4 | logistic | ok | 0.5364723481560998 | 4.05 |
| 5 | logistic | ok | 0.5273614576699633 | 4.64 |
| 6 | catboost | ok | 0.6051377100808825 | 17.73 |
| 7 | logistic | ok | 0.5365172057664631 | 4.08 |

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
  "accuracy": 0.6080827067669173,
  "macro_f1": 0.5801184288414151
}
```

## Limitations

Finite search space; observed training devices: cpu, gpu; one validation split per run. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
