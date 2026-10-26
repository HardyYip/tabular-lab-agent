# Tabular Lab experiment report

Strategy: **baseline**; provider: **mock**. Mock/replay results are not live LLM results.

Status: completed; stop reason: baseline_complete

Estimated model cost: CNY 0.000000; calls: 0; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.20244328097731237 | 3.58 |
| 1 | logistic | ok | 0.5472515556042574 | 3.69 |

## Decisions

- Trial 0: Majority-class reference.
- Trial 1: Linear baseline with training-only preprocessing.

## Final test (not used for selection)

```json
{
  "accuracy": 0.606203007518797,
  "macro_f1": 0.5693136314608144
}
```

## Limitations

Finite search space; one validation split per run; CPU execution. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
