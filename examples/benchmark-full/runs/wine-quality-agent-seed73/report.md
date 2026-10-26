# Tabular Lab experiment report

Strategy: **agent**; provider: **deepseek**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.083700; calls: 7; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.20244328097731237 | 4.05 |
| 1 | logistic | ok | 0.5834368061967594 | 4.02 |
| 2 | logistic | ok | 0.5789788228605053 | 3.97 |
| 3 | hist_gradient_boosting | ok | 0.6141971724371946 | 4.39 |
| 4 | hist_gradient_boosting | ok | 0.6226473922902495 | 4.31 |
| 5 | catboost | ok | 0.6133096954687018 | 15.77 |
| 6 | catboost | ok | 0.6157901420010382 | 14.08 |
| 7 | hist_gradient_boosting | ok | 0.5991163988800373 | 4.45 |

## Decisions

- Trial 0: Majority-class reference.
- Trial 1: Linear baseline with training-only preprocessing.
- Trial 2: Logistic regression already beats the dummy baseline substantially (macro_f1 0.583 vs 0.202), indicating informative linear structure. With 3 classes and mild imbalance (605/1193/1394), first testing stronger regularization (C=0.1) with unchanged preprocessing isolates whether the linear baseline is overfitting; this is the cheapest informative next step before adding class weighting or moving to nonlinear models.
- Trial 3: History shows logistic regression at C=1.0 (macro_f1 0.583) and C=0.1 (0.579) nearly matching, so regularization strength is not the bottleneck; the linear model is stable but may underfit nonlinear feature interactions. Since stronger regularization did not help, the next informative step is to test a nonlinear tree ensemble to capture interactions among the 11 numeric + 1 categorical features. Using regularized boosting (hist_gradient_boosting, lr=0.1, 150 iters, 31 leaves) gives a strong, commonly effective step up from the linear baseline while keeping runtime modest on 3192 rows.
- Trial 4: Logistic baselines plateaued (~0.583) and regularization strength was not the bottleneck; hist_gradient_boosting at lr=0.1 improved macro_f1 to 0.614, showing nonlinear interactions help. The natural next step is to test a lower learning rate (0.03) for the same 150-iteration boosting budget, which typically yields better-calibrated, less-overfit trees. With mild class imbalance (605/1193/1394) and macro_f1 as metric, this isolates whether the current HGB is underfit on the optimization side before trying class weighting or a different ensemble.
- Trial 5: Boosting dominates the search so far: hist_gradient_boosting improved from 0.614 (lr=0.1) to 0.6226 (lr=0.03), and lower-lr boosting over 150 iterations was still gaining, suggesting the model benefits from slower, more regularized fitting rather than class weighting or linear tweaks. CatBoost with native categorical handling (there is one binary categorical feature) and stronger default regularization is a complementary boosted model that often outperforms HGB on small tabular data with mixed feature types; testing depth=4 with 200 iterations provides a regularized comparison on a fresh model family rather than repeating HGB variants already explored.
- Trial 6: History shows boosting wins: HGB lr=0.03 reached macro_f1 0.6226, better than lr=0.1 (0.614) and logistic plateau (~0.583). The slow-lr run was still improving, suggesting boosted models benefit from slower fitting. CatBoost at depth 4 matched HGB but underperformed it (0.613); a deeper CatBoost (depth=6) with native categorical handling may capture richer interactions. Testing this fresh-depth CatBoost with the proven median/scaled preprocessing isolates whether CatBoost's tree depth is the limiting factor versus its default regularization.
- Trial 7: Boosting is the clear winner so far: hist_gradient_boosting with lr=0.03 reached macro_f1 0.6226, beating lr=0.1 (0.614), logistic (~0.583), and both CatBoost depths (0.613, 0.616). The slow-lr run was still improving, indicating the data benefits from boosted modeling of nonlinear interactions. With macro_f1 as metric and mild class imbalance (605/1193/1394), addressing minority-class recall is a promising unexplored axis: no trial has yet applied class weighting. Testing the proven best HGB config (lr=0.03) with class_weight='balanced' isolates whether per-class reweighting lifts macro_f1 over the unweighted champion, rather than repeating captured HGB lr/depth or CatBoost variants.

## Final test (not used for selection)

```json
{
  "accuracy": 0.5986842105263158,
  "macro_f1": 0.5820821120187909
}
```

## Limitations

Finite search space; observed training devices: cpu, gpu; one validation split per run. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
