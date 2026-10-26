# Tabular Lab experiment report

Strategy: **agent**; provider: **deepseek**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.071066; calls: 6; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.18666666666666668 | 4.19 |
| 1 | logistic | ok | 1.0 | 4.03 |
| 2 | logistic | ok | 1.0 | 4.23 |
| 3 | logistic | ok | 1.0 | 5.34 |
| 4 | logistic | ok | 0.9487179487179488 | 5.78 |
| 5 | logistic | ok | 1.0 | 5.17 |
| 6 | logistic | ok | 0.974320987654321 | 5.34 |
| 7 | extra_trees | ok | 1.0 | 6.03 |

## Decisions

- Trial 0: Majority-class reference.
- Trial 1: Linear baseline with training-only preprocessing.
- Trial 2: Baseline logistic regression with scaled, median-imputed features achieved perfect macro F1 (1.0), so the classes are linearly separable with this preprocessing. Rather than adding complexity or changing preprocessing that could only hurt, the most informative next step is to probe regularization strength at the low end (C=0.1) to test whether the learned boundary is stable and whether stronger shrinkage preserves performance, which would indicate robustness to noise/overfitting on this small 106-row dataset.
- Trial 3: The scaled, median-imputed logistic regression already reaches macro F1 1.0 at both C=1.0 and C=0.1, so the classes are linearly separable and the ceiling has been hit on the training metric. Before concluding separability generalizes, probe the upper end of regularization (C=10.0): if performance stays at 1.0, the linear boundary is robust across the shrinkage range rather than an artifact of a specific C, supporting that the simple linear model is sufficient and nonlinear alternatives are unnecessary on this small 106-row dataset.
- Trial 4: Logistic regression with scaled, median-imputed features already achieves macro F1 1.0 at C=0.1, C=1.0, and C=10.0, so the linear boundary is robust across all regularization strengths and the metric ceiling has been hit. Since the data has no missing values, the only untested dimension that could still affect a linear model is imputation choice (most_frequent vs median); this is effectively redundant here, but the most informative remaining untested probe is whether dropping feature scaling preserves the perfect score. If unscaled performance degrades, it confirms scaling is essential to the linear separation; if it stays at 1.0, the ceiling is fully model-robust and no further experiments are warranted.
- Trial 5: Logistic regression with scaled, median-imputed features hits macro F1 1.0 at C=0.1, 1.0, and 10.0, so the classes are linearly separable and the metric ceiling is reached; dropping scaling degraded it to 0.949, confirming scaling is essential for the linear boundary. With no missing values, imputation choice is moot, so the remaining meaningful test is whether the perfect score survives class weighting (balanced) at the default C=1.0 given mild class imbalance (35/43/28). If balanced weighting keeps 1.0, the ceiling is robust to imbalance-aware weighting; if it drops, unweighted logistic is the confirmed best configuration, and the stopping criterion (repeated 1.0) is satisfied.
- Trial 6: Logistic regression with scaled, median-imputed features reaches macro F1 1.0 across C in {0.1, 1.0, 10.0} and also under balanced class weighting at C=1.0, while unscaled features degrade to 0.949 — confirming a robust linear boundary and that scaling is essential. The metric ceiling has been hit repeatedly, so the only remaining informative probe is whether feature selection preserves the perfect separation. Reducing to select_percentile=75 tests whether the boundary survives discarding the weakest quarter of features; if macro F1 stays at 1.0 with balanced weighting at C=1.0, the linear signal is concentrated in a feature subset and the ceiling is confirmed robust, supporting stopping; if it drops, the full feature set is needed and the stopping decision is reinforced.
- Trial 7: Logistic regression with scaled, median-imputed features reaches macro F1 1.0 across C in {0.1, 1.0, 10.0} and under balanced weighting, while unscaled features degrade to 0.949 and select_percentile=75 degrades to 0.97443 — establishing a robust, scaled, full-feature linear boundary. Since the linear ceiling is confirmed, the remaining informative probe is whether a nonlinear model can match or exceed it: extra_trees with balanced weighting and capped depth (8) tests whether class-imbalance-aware nonlinear interactions add value beyond the linear solution on this small 106-row dataset. If it falls short of 1.0, the linear model is confirmed sufficient and experiments can stop.

## Final test (not used for selection)

```json
{
  "accuracy": 1.0,
  "macro_f1": 1.0
}
```

## Limitations

Finite search space; observed training devices: cpu; one validation split per run. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
