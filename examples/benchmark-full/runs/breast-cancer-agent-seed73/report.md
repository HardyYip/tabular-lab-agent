# Tabular Lab experiment report

Strategy: **agent**; provider: **deepseek**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.089500; calls: 6; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.3837837837837838 | 5.08 |
| 1 | logistic | ok | 0.981329839502129 | 5.86 |
| 2 | hist_gradient_boosting | ok | 0.933693394266722 | 6.33 |
| 3 | logistic | ok | 0.9526381387619444 | 7.23 |
| 4 | logistic | ok | 0.9721203228173148 | 5.91 |
| 5 | logistic | ok | 0.9718634306869601 | 5.27 |
| 6 | logistic | ok | 0.9721203228173148 | 5.38 |
| 7 | logistic | ok | 0.981329839502129 | 5.30 |

## Decisions

- Trial 0: Majority-class reference.
- Trial 1: Linear baseline with training-only preprocessing.
- Trial 2: The linear baseline already achieves near-ceiling macro F1 (0.981) and ROC AUC (0.996), suggesting the classes are close to linearly separable in scaled feature space. With only 341 training rows and mild class imbalance (127/214, not severe), the most valuable probe is a nonlinear model to check whether it can match or exceed the linear baseline; gradient boosting with a conservative learning rate (0.03) and moderate depth is a strong, low-variance nonlinear candidate that can reveal any residual nonlinear signal the linear model misses.
- Trial 3: The logistic baseline with C=1.0 already hit macro F1 0.981 and ROC AUC 0.996, while hist_gradient_boosting underperformed (macro F1 0.934), reinforcing that the classes are near-linearly separable and that added nonlinear capacity mostly adds variance on only 341 rows. Since a lower regularization strength C=0.1 was not yet tried, a tighter linear model may generalize slightly better by reducing coefficient variance without sacrificing the already-strong separability; this is a low-risk refinement of the best-performing model family.
- Trial 4: Linear logistic with C=1.0 on scaled features reached the best macro F1 (0.981) and ROC AUC (0.996), while hist_gradient_boosting (0.934) and tighter regularization C=0.1 (0.953) both degraded performance, confirming near-linear separability and that reduced regularization capacity hurts. The remaining untried linear direction is looser regularization: C=10.0 stays in the same successful linear family but lets the model fit the separable structure more fully without the variance penalty that hurt the nonlinear models. This directly probes whether the default C=1.0 underfits the near-linear boundary.
- Trial 5: Logistic regression with C=1.0 on scaled features remains the best performer (macro F1 0.981, ROC AUC 0.996); C=0.1 (0.953) and C=10.0 (0.972) both degraded, and nonlinear hist_gradient_boosting fell to 0.934, confirming near-linear separability where capacity changes hurt. With no missing values, imputation choice is irrelevant, so untried leverage points are feature-space reductions and class weighting. A select_percentile=75 with C=1.0 was not yet tried and keeps the strongest linear config while dropping the least-informative quartile of features, which may reduce noise from the many collinear cellular-dimension columns without changing model family.
- Trial 6: Logistic regression with C=1.0 on scaled features remains the best performer (macro F1 0.981, ROC AUC 0.996); C=0.1 (0.953) and C=10.0 (0.972) both degraded, hist_gradient_boosting fell to 0.934, and select_percentile=75 also dropped to 0.972, confirming near-linear separability where capacity/feature changes hurt. The mild imbalance (127/214) makes macro F1 sensitive to per-class recall, so the untried lever most likely to help is class_weight='balanced' at the strongest regularization C=1.0. It preserves the winning linear config and feature space while explicitly correcting any majority-class bias, which directly targets macro F1 without adding variance.
- Trial 7: Logistic regression on scaled features is clearly the best family: C=1.0 (0.981 macro F1), C=10.0 (0.972), C=0.1 (0.953), class_weight=balanced (0.972), select_percentile=75 (0.972) all cluster near the top, while nonlinear hist_gradient_boosting fell to 0.934, confirming near-linear separability and that capacity/feature changes tend to hurt. Notably every perturbation away from the plain scaled C=1.0 config has been slightly worse, so the informative untested route is the complementary regularization direction with the balancing that helped little-but-positively at C=1.0: a moderately regularized linear model combined with class weighting at C=10 and balanced was never tried. However C=10.0 and balanced at C=1.0 each individually hurt, so combining them is unlikely to exceed C=1.0; instead the remaining untested knob that keeps the winning config is min_frequency-based feature filtering. With 30 features and 341 rows, raising min_frequency to 5 removes ultra-rare categorical-like columns (already all numeric, but guards against low-support splits/normalization artifacts) while preserving the strong linear model, a low-risk variance-reduction probe aimed at the one remaining lever not yet explored in the winning family.

## Final test (not used for selection)

```json
{
  "accuracy": 0.9912280701754386,
  "macro_f1": 0.9905276277523889,
  "roc_auc": 0.9986772486772486
}
```

## Limitations

Finite search space; observed training devices: cpu; one validation split per run. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
