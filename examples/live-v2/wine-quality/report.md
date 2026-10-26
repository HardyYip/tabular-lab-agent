# Tabular Lab experiment report

Strategy: **codegen**; provider: **deepseek**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.061325; calls: 4; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.20244328097731237 | 5.44 |
| 1 | logistic | ok | 0.5472515556042574 | 5.28 |
| 2 | generated | ok | 0.5827990251315431 | 5.58 |
| 3 | generated | ok | 0.5867353867353867 | 6.56 |
| 4 | generated | failed | — | 4.08 |

## Decisions

- Trial 0: Majority-class reference.
- Trial 1: Linear baseline with training-only preprocessing.
- Trial 2: CatBoost multiclass handles numeric/categorical mix with missing values natively; minimal preprocessing and balanced class weights for macro-F1.
- Trial 3: CatBoost with native categorical handling, class weights, and ordinal-encoded categoricals inside a pipeline may be improved by adding feature engineering and tuned hyperparameters; ensemble CatBoost with a logistic regression on scaled features to boost macro-F1.
- Trial 4: Improve over trial 3 by engineering wine-chemistry ratio/interaction features fitted inside the pipeline, using CatBoost with native categorical handling, balanced class weights, and an ensemble with logistic regression on scaled imputed features to raise macro-F1.

## Final test (not used for selection)

```json
{
  "accuracy": 0.6221804511278195,
  "macro_f1": 0.6131776159932715
}
```

## Limitations

Generated pipelines in CPU Docker containers; one validation split per run. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
