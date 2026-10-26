# Tabular Lab experiment report

Strategy: **agent**; provider: **deepseek**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.070070; calls: 6; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.18666666666666668 | 5.17 |
| 1 | logistic | ok | 0.945824706694272 | 5.27 |
| 2 | logistic | ok | 0.945824706694272 | 4.23 |
| 3 | extra_trees | ok | 1.0 | 4.62 |
| 4 | extra_trees | ok | 1.0 | 5.34 |
| 5 | catboost | ok | 0.9717813051146384 | 19.74 |
| 6 | hist_gradient_boosting | ok | 0.945824706694272 | 5.83 |
| 7 | hist_gradient_boosting | ok | 0.9717813051146384 | 5.84 |

## Decisions

- Trial 0: Majority-class reference.
- Trial 1: Linear baseline with training-only preprocessing.
- Trial 2: The scaled logistic baseline reached macro_f1 0.946 with only 106 training rows and 13 numeric features, showing a strong near-linear class boundary. Given the small sample size, higher regularization (C=0.1) should reduce variance and may match or slightly improve macro_f1 versus the C=1.0 baseline; class weighting is unnecessary since class counts (35/43/28) are already fairly balanced.
- Trial 3: Logistic regression with scaling reached macro_f1 0.946 and higher regularization (C=0.1) gave an identical score, so the linear boundary is stable and well-regularized. To probe whether additional model capacity can capture non-linear interactions among the 13 numeric features, try ExtraTrees (200 trees, depth 8, sqrt features) with the same preprocessing; with only 106 rows, the shallow depth cap and random splits should limit overfitting while allowing interaction effects beyond the linear baseline.
- Trial 4: Logistic baselines plateaued at macro_f1 0.946 (both C=0.1 and C=1.0), while ExtraTrees (200 trees, max_depth=8, sqrt features) achieved macro_f1 1.0, indicating nonlinear feature interactions exist. Removing the depth cap lets ExtraTrees fully exploit interactions that a depth-8 cap may still smooth; with 200 randomized trees and sqrt feature subsetting, variance is controlled by averaging, and min_samples_leaf=1 retains fine splits on the 106-row training set. This isolates whether depth is the limiting factor for the tree model before exploring boosting alternatives.
- Trial 5: Logistic models plateaued at macro_f1 0.946 and ExtraTrees reached perfect 1.0 with both depth caps, showing strong nonlinear structure. CatBoost with shallow depth (4) and strong regularization adds a different, lower-variance nonlinear hypothesis via ordered boosting suited to the 106-row regime, testing whether gradient boosting generalizes as well without the overfitting risk of deeper trees.
- Trial 6: ExtraTrees saturated macro_f1 at 1.0 with both depth caps, and CatBoost (depth 4, lr 0.1) reached 0.972, so a boosted variant with more capacity is a natural next probe. HistGradientBoosting at a faster learning rate (0.1) with 31 leaf nodes exploits leaner residual structure than CatBoost's ordered boosting, testing whether a different nonlinear learner also generalizes tightly on the 106-row, 13-numeric-feature set. This tests boosted nonlinear interactions beyond the single CatBoost configuration tried so far.
- Trial 7: ExtraTrees with 200 trees, max_depth=8 and sqrt features reached macro_f1 1.0, and removing the depth cap (max_depth=null) also gave 1.0, so unregularized tree depth is not the constraint. CatBoost (depth 4) scored 0.972 and HistGradientBoosting (lr 0.1, 31 leaves) 0.946, while HistGradientBoosting at lr 0.03 has not been tried; a slower learning rate with the same leaf capacity adds more, smaller residual corrections that typically reduce variance on tiny datasets (106 rows). This tests whether a gentler boosted-learner schedule closes the gap to the perfect ExtraTrees result without tree-depth overfitting; class weighting is unnecessary since class counts (35/43/28) are nearly balanced.

## Final test (not used for selection)

```json
{
  "accuracy": 1.0,
  "macro_f1": 1.0
}
```

## Limitations

Finite search space; observed training devices: cpu, gpu; one validation split per run. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
