# Tabular Lab experiment report

Strategy: **baseline**; provider: **deepseek**. Mock/replay results are not live LLM results.

Status: completed; stop reason: baseline_complete

Estimated model cost: CNY 0.000000; calls: 0; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.18666666666666668 | 5.02 |
| 1 | logistic | ok | 0.945824706694272 | 4.03 |

## Decisions

- Trial 0: Majority-class reference.
- Trial 1: Linear baseline with training-only preprocessing.

## Final test (not used for selection)

```json
{
  "accuracy": 1.0,
  "macro_f1": 1.0
}
```

## Limitations

Finite search space; observed training devices: cpu; one validation split per run. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
