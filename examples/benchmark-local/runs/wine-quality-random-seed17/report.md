# Tabular Lab experiment report

Strategy: **random**; provider: **mock**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.000000; calls: 0; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.20244328097731237 | 2.88 |
| 1 | logistic | ok | 0.5655501875094702 | 2.84 |
| 2 | logistic | ok | 0.5655501875094702 | 2.67 |
| 3 | hist_gradient_boosting | ok | 0.592328705618356 | 3.58 |
| 4 | logistic | ok | 0.5364723481560998 | 3.11 |
| 5 | logistic | ok | 0.5273614576699633 | 3.58 |
| 6 | catboost | ok | 0.6051377100808825 | 19.75 |
| 7 | logistic | ok | 0.5365172057664631 | 3.44 |

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

Finite search space; one validation split per run; CPU execution. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
