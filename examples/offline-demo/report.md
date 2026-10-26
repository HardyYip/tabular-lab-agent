# Tabular Lab experiment report

Strategy: **agent**; provider: **mock**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.000000; calls: 0; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.424 | 2.69 |
| 1 | logistic | ok | 0.8074866310160428 | 2.61 |
| 2 | extra_trees | ok | 0.8561233228661147 | 3.16 |
| 3 | hist_gradient_boosting | ok | 0.8888888888888888 | 2.36 |
| 4 | catboost | ok | 0.8726951250315736 | 12.05 |

## Decisions

- Trial 0: Majority-class reference.
- Trial 1: Linear baseline with training-only preprocessing.
- Trial 2: Test nonlinear tree interactions.
- Trial 3: Test boosted nonlinear interactions.
- Trial 4: Test native categorical boosting.

## Final test (not used for selection)

```json
{
  "accuracy": 0.9027777777777778,
  "macro_f1": 0.8678899082568807,
  "roc_auc": 0.9334657398212513
}
```

## Limitations

Finite search space; one validation split per run; CPU execution. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
