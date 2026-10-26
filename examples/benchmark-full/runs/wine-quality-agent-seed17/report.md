# Tabular Lab experiment report

Strategy: **agent**; provider: **deepseek**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.070401; calls: 6; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.20244328097731237 | 4.09 |
| 1 | logistic | ok | 0.5655501875094702 | 3.95 |
| 2 | extra_trees | ok | 0.5144062511702111 | 4.27 |
| 3 | hist_gradient_boosting | ok | 0.6013955351748351 | 4.31 |
| 4 | hist_gradient_boosting | ok | 0.5890717831500295 | 4.34 |
| 5 | hist_gradient_boosting | ok | 0.592328705618356 | 4.58 |
| 6 | catboost | ok | 0.6051377100808825 | 15.12 |
| 7 | catboost | ok | 0.6075797447461375 | 13.89 |

## Decisions

- Trial 0: Majority-class reference.
- Trial 1: Linear baseline with training-only preprocessing.
- Trial 2: The linear baseline (logistic, C=1.0) reached macro-F1 0.566, far above the majority dummy (0.202), but macro-F1 is sensitive to the imbalanced 3-class distribution (605/1193/1394). A nonlinear tree ensemble should capture interactions and threshold effects among the 12 mixed-scale features that a linear model cannot, testing whether non-linearity improves macro-F1 over the linear baseline.
- Trial 3: The linear logistic baseline (C=1.0, macro-F1 0.566) clearly beat random Extra Trees (depth 8, macro-F1 0.514), suggesting the signal here is largely smooth/linear and unregularized trees just overfit noise. A gradient-boosted learner with shallow trees and a low learning rate should combine additive linear-like splits with mild nonlinear corrections, potentially exceeding the linear baseline without the variance of random forests.
- Trial 4: Histogram gradient boosting (lr=0.03, 150 iters, 31 leaves, no class weighting) achieved the best macro-F1 so far at 0.601, beating logistic (0.566) and Extra Trees (0.514). This suggests boosted nonlinear corrections help. A faster learning rate (0.1) with the same boosting capacity should reach comparable or better fit within the same iteration budget without overfitting, since regularization comes from the ensemble itself. Class imbalance (605/1193/1394) motivates keeping a follow-up option of balanced weighting, but first test whether the higher learning rate alone improves macro-F1 over the 0.03 run.
- Trial 5: HistGradientBoosting at lr=0.03 (150 iters, 31 leaves, unweighted) is the best run so far at macro-F1 0.601, beating the faster lr=0.1 variant (0.589) and logistic (0.566), indicating that gentle boosting with many small steps generalizes better here. However, the unweighted objective optimizes accuracy-like loss, while macro-F1 on an imbalanced 3-class split (605/1193/1394) rewards balanced per-class recall. Applying class_weight='balanced' to the winning config should lift the minority-class recall and thus macro-F1 without changing the model family, isolating the effect of class weighting as the next controlled step.
- Trial 6: No previous trial used CatBoost, which handles categorical features natively and is a distinct regularized boosted-tree family from HistGradientBoosting. The best result so far came from gentle boosting (HistGB lr=0.03, unweighted, macro-F1 0.601), while balanced weighting did not help (0.592). A shallow unweighted CatBoost (depth 4, lr=0.1, l2_leaf_reg=3) offers stronger regularization than the default depth and may better exploit the one categorical feature and 12-feature mixed-scale signal, testing whether a regularized native-categorical booster beats the current best without class weighting.
- Trial 7: CatBoost with depth 4, lr=0.1, l2=3 and unweighted objective is the best configuration so far (macro-F1 0.605), narrowly surpassing HistGradientBoosting lr=0.03 (0.601); balanced class weighting did not help HistGB (0.592 vs 0.601), so the unweighted direction is favored. The untested remaining lever within the winning family is model capacity: the candidate with depth 6 keeps the same learning rate, iterations and regularization while adding representational depth, testing whether more capacity captures additional feature interactions among the 12 mixed-scale features and the categorical column without eroding the mild gains seen at depth 4.

## Final test (not used for selection)

```json
{
  "accuracy": 0.606203007518797,
  "macro_f1": 0.5790136272920938
}
```

## Limitations

Finite search space; observed training devices: cpu, gpu; one validation split per run. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
