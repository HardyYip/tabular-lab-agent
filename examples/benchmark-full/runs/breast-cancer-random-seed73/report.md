# Tabular Lab experiment report

Strategy: **random**; provider: **deepseek**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.000000; calls: 0; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.3837837837837838 | 4.22 |
| 1 | logistic | ok | 0.981329839502129 | 4.20 |
| 2 | logistic | ok | 0.9721203228173148 | 3.97 |
| 3 | logistic | ok | 0.9526381387619444 | 4.01 |
| 4 | catboost | ok | 0.9434523809523809 | 15.58 |
| 5 | logistic | ok | 0.9718634306869601 | 5.25 |
| 6 | logistic | ok | 0.9721203228173148 | 5.61 |
| 7 | logistic | ok | 0.981329839502129 | 5.03 |

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
  "accuracy": 0.9912280701754386,
  "macro_f1": 0.9905276277523889,
  "roc_auc": 0.9986772486772486
}
```

## Limitations

Finite search space; observed training devices: cpu, gpu; one validation split per run. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
