# Tabular Lab experiment report

Strategy: **agent**; provider: **deepseek**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.081633; calls: 7; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.4314942729736819 | 5.73 |
| 1 | logistic | ok | 0.7804373713745776 | 5.84 |
| 2 | hist_gradient_boosting | ok | 0.8091270552864586 | 7.42 |
| 3 | catboost | ok | 0.8087318792429428 | 31.78 |
| 4 | catboost | ok | 0.8068510438175169 | 20.36 |
| 5 | hist_gradient_boosting | ok | 0.8091180949981045 | 5.48 |
| 6 | hist_gradient_boosting | ok | 0.7967159119802779 | 4.86 |
| 7 | extra_trees | ok | 0.7643286048859588 | 13.84 |

## Decisions

- Trial 0: Majority-class reference.
- Trial 1: Linear baseline with training-only preprocessing.
- Trial 2: The logistic baseline reached macro_f1 0.780 and ROC AUC 0.904, showing a strong linear signal. The remaining question is whether nonlinear interactions and native categorical handling improve on this. A gradient-boosted model captures nonlinearities and interactions across the 14 features (several categorical with meaningful cardinality, plus skewed numerics) while remaining robust on 26k rows; its boosting capacity should lift macro_f1 beyond the linear baseline, and this serves as the first nonlinear probe before tuning.
- Trial 3: Histogram gradient boosting improved macro_f1 over the logistic baseline (0.809 vs 0.780) with strong ROC AUC 0.923, confirming nonlinear interactions add value. With 14 features (7 categorical with meaningful cardinality) and 26k rows, native categorical handling should further improve over one-hot/ordinal encoding. CatBoost with ordered boosting and categorical-aware splits is a natural next probe; depth 6 with 200 iterations is a middle-ground capacity setting to test this before tuning learning rate or depth.
- Trial 4: History shows a strong linear signal (logistic macro_f1 0.780, ROC AUC 0.904) and that nonlinear boosting adds value (HGB 0.809/0.923; CatBoost depth 6 0.809/0.923, essentially tied). A shallower CatBoost (depth 4) was the untested capacity neighbor of the tied depth-6 run; with 7 categorical features and only 14 total features, lower tree depth should reduce overfitting on native ordered-boosting categorical splits and may edge macro_f1. Testing the smaller-depth CatBoost isolates capacity as the driver before changing learning rate or class weighting.
- Trial 5: CatBoost depth 6 (macro_f1 0.809, ROC AUC 0.923) and HGB (0.809/0.923) are essentially tied, and reducing depth to 4 lowered macro_f1 (0.807), indicating depth 6 is near-optimal capacity. The training set is imbalanced (19742 vs 6266, ~24% minority), so unweighted boosting may under-optimize the minority class, which macro_f1 weights equally. A learning-rate reduction (0.1 -> 0.03) with same max_iter is the untested neighbor to the trial-2 HGB run; slower boosting often improves probabilistic calibration and minority-class recall, potentially lifting macro_f1 above the 0.809 plateau before exploring class weighting.
- Trial 6: Boosted models plateaued at macro_f1 ~0.809 (HGB lr=0.1 and 0.03; CatBoost depth 6), and lower CatBoost depth (4) or lower learning rate did not break the plateau. All boosting trials used unweighted loss, yet the training set is imbalanced (19742 vs 6266, ~24% minority) and macro_f1 weights classes equally, so unweighted boosting likely under-optimizes the minority class. Testing class_weight='balanced' on the best-performing HGB configuration (lr=0.1, 150 iters, 31 leaves) isolates class weighting as the untested lever while keeping the strongest baseline capacity, aiming to lift minority-class recall and push macro_f1 above 0.809.
- Trial 7: Boosting plateaued at macro_f1 ~0.809 across HGB (lr 0.1 and 0.03), CatBoost depth 6, and depth 4; class weighting on HGB hurt (0.797). The untested model family in the candidate space is extremely randomized trees, which offer a different bias-variance tradeoff (random split thresholds, deeper variance control) on this 14-feature, 26k-row, 7-categorical dataset. Extra trees with unlimited depth and sqrt features is the capacity-unconstrained probe; comparing against the depth-8 variant will reveal whether the plateau is a model-family limit or a capacity issue.

## Final test (not used for selection)

```json
{
  "accuracy": 0.8736564093114674,
  "macro_f1": 0.8147661307990167,
  "roc_auc": 0.9276885148586481
}
```

## Limitations

Finite search space; observed training devices: cpu, gpu; one validation split per run. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
