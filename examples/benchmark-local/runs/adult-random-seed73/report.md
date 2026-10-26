# Tabular Lab experiment report

Strategy: **random**; provider: **mock**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.000000; calls: 0; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.4314942729736819 | 3.69 |
| 1 | logistic | ok | 0.7804373713745776 | 3.52 |
| 2 | logistic | ok | 0.7784772714341638 | 2.97 |
| 3 | logistic | ok | 0.7584431503122168 | 4.25 |
| 4 | catboost | ok | 0.7942638645645561 | 23.62 |
| 5 | logistic | ok | 0.7500301010723955 | 4.62 |
| 6 | logistic | ok | 0.7776548030590623 | 3.45 |
| 7 | logistic | ok | 0.7804373713745776 | 3.39 |

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
  "accuracy": 0.8331183588231681,
  "macro_f1": 0.7959844490494585,
  "roc_auc": 0.9263991999165291
}
```

## Limitations

Finite search space; one validation split per run; CPU execution. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
