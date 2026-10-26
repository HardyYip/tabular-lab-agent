# Tabular Lab experiment report

Strategy: **agent**; provider: **deepseek**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.088404; calls: 6; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.3837837837837838 | 6.77 |
| 1 | logistic | ok | 0.99062114356232 | 5.91 |
| 2 | extra_trees | ok | 0.9626596790042581 | 5.56 |
| 3 | logistic | ok | 0.9626596790042581 | 5.44 |
| 4 | logistic | ok | 0.9811507936507937 | 5.14 |
| 5 | logistic | ok | 0.9721203228173148 | 5.25 |
| 6 | logistic | ok | 0.9715828832571666 | 5.25 |
| 7 | hist_gradient_boosting | ok | 0.9535338713621913 | 5.97 |

## Decisions

- Trial 0: Majority-class reference.
- Trial 1: Linear baseline with training-only preprocessing.
- Trial 2: The linear baseline already achieves very high macro-F1 (~0.99) with no missing values, so remaining gains are likely marginal. Testing a nonlinear tree ensemble (extra_trees, depth 8) checks whether feature interactions can match or slightly exceed the strong linear fit; chosen as the first step away from the well-regularized linear decision boundary before trying higher-capacity boosting.
- Trial 3: The scaled logistic baseline (trial 1) hit near-perfect macro-F1 (~0.991) with the default C=1.0, while extra_trees underperformed (~0.963), confirming the decision boundary is essentially linear and feature interactions add nothing. Rather than chase nonlinearity, refine along the dominant linear axis: the C=1.0 baseline is already well-fit, so testing a less-regularized C=10.0 checks whether relaxing the L2 penalty slightly improves the fit without the risk of overfitting that C=0.1 (stronger regularization) would likely introduce on 341 rows. This isolates the regularization dimension while holding the proven preprocessing (median impute, scaling, no feature selection) fixed.
- Trial 4: Trials 1-3 establish that the decision boundary is essentially linear (logistic ~0.991 macro-F1; extra_trees ~0.963) and that relaxing or tightening L2 (C=10 vs C=1) yields no gain. The one untested axis on the proven linear model is feature-space reduction: with 30 numeric features and only 341 rows, plus a clear class imbalance (127/214), a univariate 75th-percentile selection may drop noisy dimensions and modestly improve macro-F1 by reducing overfitting, while preserving the same linear boundary and preprocessing that already score highest.
- Trial 5: Trials show the boundary is essentially linear: logistic at C=1 (0.9906 macro-F1) beats extra_trees (0.9627), C=10 matches C=1 (0.9627 here), and 75th-percentile feature selection slightly hurts (0.9812). None of the tried logistic variants address the moderate class imbalance (127/214) that macro-F1 is sensitive to, and tuning C did not help. The remaining untested axis on the proven linear model is class weighting: 'balanced' reweights the minority class to equalize per-class recall, which should lift macro-F1 if the default boundary under-serves the minority class, while keeping the identical preprocessing that already scores best. Choosing a separate untested direction rather than repeating prior attempts.
- Trial 6: Trials 0-5 establish that the decision boundary is essentially linear: logistic at C=1.0 with scaled features reaches the best macro-F1 (~0.991), while extra_trees (~0.963), C=10 relaxation, 75th-percentile feature selection (~0.981), and balanced class weighting (~0.972) all fail to improve on it. Since stronger feature selection (75th percentile) slightly hurt and the baseline is already near-perfect, the one remaining untested axis is stronger regularization on the proven linear model: testing C=0.1 checks whether a tighter L2 penalty reduces variance on the 341-row, 30-feature training set without degrading fit, while holding the identified-best preprocessing (median impute, scaling, no selection) fixed.
- Trial 7: History shows the scaled logistic baseline (C=1.0, median impute, no selection) is best at ~0.991 macro-F1, and every deviation so far fails to improve: extra_trees (~0.963) confirms an essentially linear boundary, C=10 and C=0.1 both lose (~0.963, ~0.972), 75th-percentile selection hurts (~0.981), and balanced class weighting hurts (~0.972). The remaining untested axis is boosting capacity: hist_gradient_boosting with a slower learning rate (0.03) explores whether a well-regularized boosted model can capture any residual structure beyond the linear fit, chosen over the 0.1 variant to limit overfitting on only 341 rows.

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
