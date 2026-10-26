# Tabular Lab experiment report

Strategy: **baseline**; provider: **deepseek**. Mock/replay results are not live LLM results.

Status: completed; stop reason: baseline_complete

Estimated model cost: CNY 0.000000; calls: 0; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.3837837837837838 | 5.08 |
| 1 | logistic | ok | 0.99062114356232 | 5.61 |

## Decisions

- Trial 0: Majority-class reference.
- Trial 1: Linear baseline with training-only preprocessing.

## Final test (not used for selection)

```json
{
  "accuracy": 0.9824561403508771,
  "macro_f1": 0.9811507936507937,
  "roc_auc": 0.9953703703703703
}
```

## Limitations

Finite search space; observed training devices: cpu; one validation split per run. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
