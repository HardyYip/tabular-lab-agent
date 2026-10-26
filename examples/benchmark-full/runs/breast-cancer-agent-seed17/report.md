# Tabular Lab experiment report

Strategy: **agent**; provider: **deepseek**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.089774; calls: 6; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.3837837837837838 | 4.03 |
| 1 | logistic | ok | 0.9814935064935065 | 4.17 |
| 2 | logistic | ok | 0.9542792973449907 | 4.14 |
| 3 | logistic | ok | 0.9718634306869601 | 4.47 |
| 4 | extra_trees | ok | 0.9718634306869601 | 5.69 |
| 5 | logistic | ok | 0.9721203228173148 | 5.19 |
| 6 | extra_trees | ok | 0.9718634306869601 | 5.55 |
| 7 | logistic | ok | 0.981329839502129 | 5.22 |

## Decisions

- Trial 0: Majority-class reference.
- Trial 1: Linear baseline with training-only preprocessing.
- Trial 2: The linear baseline (trial 1) already achieved macro_f1 0.981 with ROC_AUC 0.999 on 30 numeric features and 341 rows, suggesting the classes are largely linearly separable and the problem is nearly saturated. With no missing values, class rebalancing and preprocessing variants are unlikely to add signal; a moderate increase in regularization strength could match or slightly improve on C=1.0 without risk. Candidate C=10.0 keeps the same training-only preprocessing pipeline and tests whether a slightly less constrained linear boundary captures residual structure without overfitting.
- Trial 3: Trial 1 (logistic C=1.0, scaled, median-imputed) is the clear leader at macro_f1 0.981 / ROC_AUC 0.999, and trial 2 (C=10.0) degraded to 0.954, indicating weaker regularization overfits the small 341-row dataset. The trend favors stronger, not weaker, regularization. The untested candidate C=0.1 explores the opposite direction (more regularization) to see if a tighter linear boundary generalizes even better on these 30 numeric features; since logistic is near-saturated, this is the most informative next probe before branching to nonlinear models.
- Trial 4: Logistic regression on scaled features is near-saturated: C=1.0 reached macro_f1 0.981 (best so far), C=0.1 reached 0.972, and C=10.0 dropped to 0.954, indicating the linear boundary generalizes best around moderate regularization and weaker regularization overfits the small 341-row set. Since the problem is essentially linearly separable, untested nonlinear candidates (extra trees, gradient boosting, catboost) may add capacity but also overfitting risk. Before committing to nonlinear models, an untested nonlinear baseline with strong averaging and limited depth—extra_trees with max_depth=8, sqrt feature sampling, 200 trees—tests whether tree interactions capture any residual structure beyond the linear model while keeping variance controlled. If it fails to beat 0.981, it confirms the linear model is the right ceiling and steering toward more complex boosting is unwarranted.
- Trial 5: Trials show a near-saturated linear problem: logistic C=1.0 scored macro_f1 0.981 (best), C=0.1 gave 0.972, and C=10.0 dropped to 0.954, while extra_trees with a moderate depth-8 bound matched only 0.972. This suggests residual errors are not from needing more nonlinear capacity but possibly from the mild class imbalance (127 vs 214), which macro_f1 penalizes. No prior trial used class rebalancing. Testing the best-regularized linear model (C=0.1) with class_weight='balanced' checks whether correcting the imbalance improves macro_f1 over the 0.981 ceiling without changing the well-performing preprocessing pipeline.
- Trial 6: Logistic regression with scaled median-imputed features is near-saturated: C=1.0 gave the best macro_f1 0.981, stronger regularization C=0.1 dropped to 0.972, C=10.0 dropped to 0.954, and balanced class weighting added nothing. The depth-8 extra_trees matched only 0.972, so added nonlinear capacity has not helped. Since the linear boundary is strong and the only untested tree variant without a depth cap may capture residual interactions, testing extra_trees with max_depth=null and sqrt feature sampling checks whether unbounded averaged trees extract any structure beyond the linear ceiling while still controlling variance via averaging; failure to beat 0.981 would confirm the linear model is the right ceiling.
- Trial 7: History shows the linear model is near-saturated: logistic C=1.0 with scaling leads at macro_f1 0.981, C=0.1 gives 0.972, C=10.0 drops to 0.954, and balanced weighting plus depth-bounded and unbounded extra_trees all plateau around 0.971-0.972. This indicates added nonlinear capacity is not the bottleneck and the best region is moderate regularization on scaled features. All prior trials used full feature sets; the linear model's ceiling at C=1.0 may partly reflect redundancy among the 30 features. Testing the same best-performing linear model with feature selection retaining 75% of features (select_percentile=75) checks whether pruning the weakest features reduces variance and improves macro_f1 over 0.981 without altering the strong scaled, median-imputed pipeline.

## Final test (not used for selection)

```json
{
  "accuracy": 0.9824561403508771,
  "macro_f1": 0.9811507936507937,
  "roc_auc": 0.9957010582010581
}
```

## Limitations

Finite search space; observed training devices: cpu; one validation split per run. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
