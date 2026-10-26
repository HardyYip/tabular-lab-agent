# Tabular Lab experiment report

Strategy: **agent**; provider: **deepseek**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.071438; calls: 6; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.4314942729736819 | 7.05 |
| 1 | logistic | ok | 0.7804373713745776 | 4.97 |
| 2 | hist_gradient_boosting | ok | 0.8107211187845934 | 5.24 |
| 3 | catboost | ok | 0.8086522468847519 | 23.50 |
| 4 | hist_gradient_boosting | ok | 0.8083612338869389 | 5.36 |
| 5 | catboost | ok | 0.8087318792429428 | 19.72 |
| 6 | hist_gradient_boosting | ok | 0.797181215140308 | 5.08 |
| 7 | extra_trees | ok | 0.7014307491269165 | 4.98 |

## Decisions

- Trial 0: Majority-class reference.
- Trial 1: Linear baseline with training-only preprocessing.
- Trial 2: The linear baseline achieved macro_f1 0.780 with roc_auc 0.904, indicating strong signal but likely underfitting nonlinear/interaction structure across the 6 categorical features and skewed numeric columns. Gradient boosted trees handle mixed categorical/numeric data and interactions well, so a hist_gradient_boosting model with moderate learning rate should improve macro_f1 over the linear baseline.
- Trial 3: The linear baseline reached macro_f1 0.780 and the gradient-boosted trees improved to 0.811 with roc_auc 0.923, confirming nonlinear interactions matter across the mixed categorical/numeric features. CatBoost's native categorical handling and ordered boosting are well suited to the 7 categorical columns (some with up to 41 categories) and skewed numeric columns, so it should capture these interactions more effectively than histogram gradient boosting at a moderate depth, yielding higher macro_f1.
- Trial 4: History shows nonlinear trees beat the linear baseline (logistic macro_f1 0.780 vs hist_gradient_boosting 0.811, roc_auc 0.904 vs 0.923), and CatBoost at depth 6 matched but did not exceed hist_gradient_boosting at lr=0.1 (0.8087 vs 0.8107), suggesting the latter's lr=0.1/150-iter setting was near-optimal on the boosting side but not yet tuned. Since the current best remains the default-ish hist_gradient_boosting, the promising untried direction is a smoother, more conservative boosting schedule: lowering learning_rate to 0.03 with the same 31 leaves and 150 iterations regularizes each step and often improves macro_f1 for imbalanced binary data (class ratio ~3:1) by reducing overfitting to the majority class, while preserving the interaction modeling that clearly outperforms linear models.
- Trial 5: History: logistic baseline macro_f1 0.780; hist_gradient_boosting lr=0.1/max_iter=150 reached 0.8107, the current best; lowering lr to 0.03 left it flat (0.8084) and CatBoost depth 6 matched but did not exceed it (0.8087), so boosting is near a plateau under scratch-built splits. The data has 7 categorical columns (one with 41 levels) and a class imbalance of roughly 3:1; native categorical handling was not yet shown to beat one-hot tree building, but CatBoost with a shallower depth (4) and strong ordered boosting may generalize better on these high-cardinality categoricals while the lower depth regularizes against the majority class and lifts macro_f1. This probe isolates depth effects for native categorical boosting, which the existing depth-6 run did not test.
- Trial 6: Tried so far: logistic (macro_f1 0.780), hist_gradient_boosting at lr 0.1 and 0.03 (0.8107 best, 0.8084), and catboost at depth 4 and 6 (0.8087, 0.8087). Boosting has plateaued around 0.81 on this ~3:1 imbalanced binary task, and all tree runs used null class weighting, optimizing raw accuracy rather than macro_f1. Since the metric rewards minority-class recall, a balanced class_weight on the current best model is a principled, untried lever: hist_gradient_boosting with lr=0.1/150 iters and class_weight='balanced' should reweight leaf gradients to favor the minority class and raise macro_f1 without changing the interaction structure that already beat the linear baseline.
- Trial 7: Boosting has plateaued near macro_f1 0.81 across hist_gradient_boosting (lr 0.1 and 0.03) and catboost (depth 4 and 6), while the linear logistic baseline reached 0.780 with roc_auc 0.904. Nonlinear trees clearly win, but no untried model family has been probed beyond boosting. A bagged, heavily randomized tree ensemble (extra_trees) with balanced class weighting is the principled next step on this ~3:1 imbalanced mixed categorical/numeric data: extra-random splits reduce variance and the 'balanced' weighting directly targets macro_f1 by favoring minority-class recall, testing whether a decorrelated ensemble can exceed the boosting plateau that explicit class weighting hurt for hist_gradient_boosting.

## Final test (not used for selection)

```json
{
  "accuracy": 0.8729193538480438,
  "macro_f1": 0.8139874930466997,
  "roc_auc": 0.9276108881106351
}
```

## Limitations

Finite search space; observed training devices: cpu, gpu; one validation split per run. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
