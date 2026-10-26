# Tabular Lab experiment report

Strategy: **random**; provider: **deepseek**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.000000; calls: 0; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.3837837837837838 | 5.14 |
| 1 | logistic | ok | 0.99062114356232 | 5.19 |
| 2 | catboost | ok | 0.9626596790042581 | 16.48 |
| 3 | logistic | ok | 0.9439895185063871 | 4.33 |
| 4 | logistic | ok | 0.9721203228173148 | 4.22 |
| 5 | hist_gradient_boosting | ok | 0.9535338713621913 | 4.30 |
| 6 | logistic | ok | 0.9721203228173148 | 4.14 |
| 7 | extra_trees | ok | 0.9626596790042581 | 4.84 |

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
  "accuracy": 0.9824561403508771,
  "macro_f1": 0.9811507936507937,
  "roc_auc": 0.9953703703703703
}
```

## Limitations

Finite search space; observed training devices: cpu, gpu; one validation split per run. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
