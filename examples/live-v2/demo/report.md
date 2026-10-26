# Tabular Lab experiment report

Strategy: **codegen**; provider: **deepseek**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.026672; calls: 3; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.424 | 1.95 |
| 1 | logistic | ok | 0.8074866310160428 | 1.98 |
| 2 | generated | failed | — | 6.52 |
| 3 | generated | failed | — | 2.33 |
| 4 | generated | ok | 0.8570009930486594 | 2.62 |

## Decisions

- Trial 0: Majority-class reference.
- Trial 1: Linear baseline with training-only preprocessing.
- Trial 2: Logistic regression with median imputation, standard scaling, and one-hot encoding already achieves 0.807 macro F1; adding a small ensemble (logistic + catboost) with class_weight balancing and ordinal/one-hot handling of the categorical column should improve robustness on this small, slightly imbalanced dataset.
- Trial 3: The previous CatBoost failure likely stems from class_weight='balanced' combined with auto_class_weights='Balanced', which can cause parameter conflicts or invalid labels for CatBoost's balanced weighting. Removing class_weight from CatBoost (and using explicit class_weights instead) or dropping the ensemble in favor of a robust logistic regression plus a safer CatBoost configuration should fix the issue while maintaining strong macro F1.
- Trial 4: The CatBoost failure under VotingClassifier is likely because class labels must be non-negative contiguous integers; combined with the imbalanced small dataset, a robust alternative is a soft-voting ensemble of a balanced logistic regression and a balanced RandomForest, both wrapped so that the whole preprocessing (median/most-frequent imputation, scaling, one-hot with handle_unknown) is fitted inside the pipeline. RandomForest is stable, handles missing categories via the fitted OneHotEncoder, and gives strong macro-F1 on small tabular data without CatBoost-specific label constraints.

## Final test (not used for selection)

```json
{
  "accuracy": 0.9444444444444444,
  "macro_f1": 0.9229946524064171,
  "roc_auc": 0.9890764647467727
}
```

## Limitations

Generated pipelines in CPU Docker containers; one validation split per run. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
