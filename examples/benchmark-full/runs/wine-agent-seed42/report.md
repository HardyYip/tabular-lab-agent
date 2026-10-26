# Tabular Lab experiment report

Strategy: **agent**; provider: **deepseek**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.070232; calls: 6; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.18666666666666668 | 5.48 |
| 1 | logistic | ok | 1.0 | 5.44 |
| 2 | logistic | ok | 1.0 | 5.16 |
| 3 | hist_gradient_boosting | ok | 0.974320987654321 | 5.80 |
| 4 | logistic | ok | 0.9474747474747475 | 5.34 |
| 5 | extra_trees | ok | 1.0 | 4.45 |
| 6 | logistic | ok | 1.0 | 4.05 |
| 7 | catboost | ok | 0.9740129935032483 | 14.06 |

## Decisions

- Trial 0: Majority-class reference.
- Trial 1: Linear baseline with training-only preprocessing.
- Trial 2: The scaled logistic baseline already reached perfect macro F1 with C=1.0. To probe robustness, test lighter regularization (C=0.1): if the separation is genuine rather than overfit, stronger regularization should sustain high macro F1 while verifying the linear model is not sensitive to the penalty strength.
- Trial 3: The scaled logistic baseline achieved perfect macro F1 at both C=1.0 and C=0.1, showing the classes are highly separable with a linear decision boundary and insensitive to regularization strength. Since lighter regularization didn't degrade performance, the remaining question is whether this is genuinely linear or whether non-linear structure exists. A shallow boosted tree ensemble (hist_gradient_boosting with a low learning rate) is the natural next probe: on 106 rows with 13 clean numeric features, if the perfect separation is truly linear we expect comparable perfect or near-perfect macro F1; any drop would be regularization-driven and suggest the linear boundary is the right inductive bias.
- Trial 4: Scaled logistic regression reached perfect macro F1 at both C=1.0 and C=0.1, showing a highly separable, regularization-insensitive linear signal, while hist_gradient_boosting matched but did not exceed it (0.974). Testing whether the perfect linear separation depends on feature scaling is the next diagnostic: if scaling is essential, an unscaled logistic with C=1.0 should degrade, confirming the linear boundary operates in the scaled space; if it remains perfect, the signal is scale-invariant and the linear baseline is robust.
- Trial 5: Scaled logistic regression reached perfect macro F1 at both C=1.0 and C=0.1, and remained strong (0.947) even unscaled, showing a highly separable, regularization-insensitive linear signal. Dropping scaling cost a small amount, so scaling is modestly helpful but not essential. The one untested direction that could improve on the perfect linear baseline is whether non-linear feature interactions exist: extra_trees probes this with a different inductive bias than the already-tested hist_gradient_boosting. If the separation is genuinely linear, extra_trees should match or fall below the logistic result; if it exceeds perfect on held-out folds or reveals complementary structure, interactions matter. This is a new model not yet attempted.
- Trial 6: Across trials, macro F1 is at or near perfect for both the linear model (logistic C=1.0 and C=0.1, scaled) and extra_trees, while a moderate hist_gradient_boosting (0.974) and unscaled logistic (0.947) fall slightly short. This indicates a highly separable signal where the linear boundary is the right inductive bias and scaling matters modestly. Since most expressive models already saturate near ceiling, the remaining question worth testing is whether a more strongly regularized linear boundary (C=10.0) can sustain perfect macro F1 — distinguishing a genuinely linear, low-variance separation from one that depends on penalty strength. This new untested C value for the scaled logistic directly probes regularization sensitivity at the loose end.
- Trial 7: Scaled logistic regression saturates at perfect macro F1 across C in {0.1, 1, 10}, extra_trees also reaches 1.0, and unscaled logistic drops to 0.947, indicating a highly separable, near-linear signal where feature scaling is modestly helpful. A different model family not yet attempted is CatBoost; a shallow depth-4 boosted model with mild L2 provides a strong regularizer on only 106 rows and probes whether gradient-boosted interactions can sustain the ceiling without overfitting, complementing the earlier hist_gradient_boosting probe with a distinct implementation.

## Final test (not used for selection)

```json
{
  "accuracy": 0.9722222222222222,
  "macro_f1": 0.9709618874773139
}
```

## Limitations

Finite search space; observed training devices: cpu, gpu; one validation split per run. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
