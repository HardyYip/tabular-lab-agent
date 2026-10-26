# Tabular Lab experiment report

Strategy: **agent**; provider: **deepseek**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.069342; calls: 6; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.4314942729736819 | 5.77 |
| 1 | logistic | ok | 0.7804373713745776 | 4.73 |
| 2 | extra_trees | ok | 0.726984495788525 | 5.44 |
| 3 | extra_trees | ok | 0.7632234192871632 | 14.06 |
| 4 | logistic | ok | 0.778919104066512 | 4.03 |
| 5 | logistic | ok | 0.7809401339372005 | 4.11 |
| 6 | logistic | ok | 0.7784772714341638 | 4.08 |
| 7 | hist_gradient_boosting | ok | 0.8076953908007936 | 4.92 |

## Decisions

- Trial 0: Majority-class reference.
- Trial 1: Linear baseline with training-only preprocessing.
- Trial 2: The linear baseline (C=1.0) already achieves macro_f1 0.78 with roc_auc 0.90, showing strong linearly separable signal from the numeric and low-cardinality features. To probe whether nonlinear interactions among the 14 features (several numeric with wide ranges plus categorical ones) yield additional gains, start with an ExtraTrees ensemble, which handles mixed-type scaled features and captures interactions while being robust. Default max_depth=8 with 200 trees limits overfitting on the 75/25 imbalance without class weighting initially, giving a clean comparison to the linear baseline.
- Trial 3: The linear logistic baseline (C=1.0) reached macro_f1 0.780 with roc_auc 0.904, while a capped ExtraTrees (max_depth=8) underperformed at macro_f1 0.727, indicating the depth cap limited nonlinear gains rather than nonlinearity being unhelpful. Before abandoning the tree family, test whether removing the depth constraint on ExtraTrees lets it capture deep interactions; if uncapped ExtraTrees still trails logistic, that confirms the signal is largely linear and we should focus on tuning the logistic model instead.
- Trial 4: Logistic C=1.0 leads at macro_f1 0.780 (roc_auc 0.904); ExtraTrees improved from macro_f1 0.727 to 0.763 when unconstrained but still trails logistic, suggesting the signal is largely linear yet trees capture some structure without linearization. The best linear model used default regularization; testing C=0.1 probes whether stronger regularization improves generalization on the 26k-row, 14-feature problem with a 76/24 class imbalance, which could yield a cleaner decision boundary than C=1.0.
- Trial 5: Logistic regression remains the strongest family: C=1.0 gives macro_f1 0.780 (roc_auc 0.904) and C=0.1 gives 0.779, while uncapped ExtraTrees topped out at 0.763, confirming the signal is largely linear. The remaining untested direction within the best family is boosting capacity with less regularization. Repeated tree models suggest extra flexibility does not help, so the lowest-risk high-upside move is to try C=10.0 to see whether relaxing regularization lets the linear model capture more structure without overfitting, since C=0.1 vs C=1.0 showed essentially a plateau and C=10.0 was never probed.
- Trial 6: Logistic regression dominates: C=10.0 reaches macro_f1 0.781 (roc_auc 0.905), C=1.0 and C=0.1 give ~0.78/0.779, and uncapped ExtraTrees trailed at 0.763, confirming a largely linear signal. With regularization strength now effectively plateaued across three orders of magnitude without class weighting, the highest-value untested lever for macro_f1 on this 76/24 imbalance is class weighting. Testing C=1.0 with class_weight='balanced' probes whether rebalancing the decision threshold for the minority class lifts macro_f1 beyond 0.781, independent of the regularization plateau already observed.
- Trial 7: The logistic family dominates (C=10.0 macro_f1 0.781, C=1.0 0.780, C=0.1 0.779), with a flat plateau across three orders of magnitude, while balanced class weighting at C=1.0 did not improve (0.778). ExtraTrees maxed at 0.763, so nonlinearity has not beaten the linear signal. The key remaining question is whether a modern gradient-boosted model can capture any interaction structure the linear models and trees miss; logistic has plateaued, so the best untested nonlinear candidate worth probing is HistGradientBoosting with a moderate learning rate. Comparing its boosted interactions against the 0.781 linear ceiling tells us whether to keep optimizing the linear family or invest in boosting.

## Final test (not used for selection)

```json
{
  "accuracy": 0.8732264602911369,
  "macro_f1": 0.8140035119943991,
  "roc_auc": 0.9273568996640043
}
```

## Limitations

Finite search space; observed training devices: cpu; one validation split per run. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
