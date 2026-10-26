# Tabular Lab experiment report

Strategy: **random**; provider: **deepseek**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.000000; calls: 0; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.20244328097731237 | 3.88 |
| 1 | logistic | ok | 0.5834368061967594 | 3.91 |
| 2 | logistic | ok | 0.5746020442519463 | 3.86 |
| 3 | logistic | ok | 0.5562883252518436 | 4.17 |
| 4 | catboost | ok | 0.5890623814750873 | 15.75 |
| 5 | logistic | ok | 0.552081277651865 | 4.34 |
| 6 | logistic | ok | 0.5419395189070296 | 4.02 |
| 7 | logistic | ok | 0.5834368061967594 | 3.98 |

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
  "accuracy": 0.5845864661654135,
  "macro_f1": 0.5805468001266654
}
```

## Limitations

Finite search space; observed training devices: cpu, gpu; one validation split per run. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
