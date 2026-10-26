# Tabular Lab experiment report

Strategy: **agent**; provider: **deepseek**. Mock/replay results are not live LLM results.

Status: completed; stop reason: trial_limit

Estimated model cost: CNY 0.071362; calls: 6; unknown usage: False

| Trial | Model | Status | Validation macro-F1 | Seconds |
| --- | --- | --- | ---: | ---: |
| 0 | dummy | ok | 0.20244328097731237 | 3.94 |
| 1 | logistic | ok | 0.5472515556042574 | 4.02 |
| 2 | extra_trees | ok | 0.5038265168127064 | 4.28 |
| 3 | logistic | ok | 0.5356593705701027 | 4.05 |
| 4 | logistic | ok | 0.5508451013660943 | 4.02 |
| 5 | hist_gradient_boosting | ok | 0.551661293112939 | 4.52 |
| 6 | catboost | ok | 0.570192916422068 | 14.00 |
| 7 | catboost | ok | 0.5753738270955516 | 16.27 |

## Decisions

- Trial 0: Majority-class reference.
- Trial 1: Linear baseline with training-only preprocessing.
- Trial 2: The scaled logistic baseline reached macro_f1 0.547 versus 0.202 for the dummy, so linear signal is strong, but macro_f1 trails accuracy (0.574) with imbalanced classes (605/1193/1394), and features span very different scales/ranges. A nonlinear tree model should capture interactions and threshold effects that the linear model misses; ExtraTrees with bounded depth (8) and sqrt feature sampling offers a good bias-variance tradeoff on this 3192-row, 12-feature set and is a distinct model family from the linear baseline.
- Trial 3: Logistic regression remains the strongest model so far (macro_f1 0.547), while the tuned ExtraTrees underperformed (0.504), suggesting the signal is largely linear and tree interactions add noise. However, macro_f1 still trails accuracy (0.574), indicating the minority classes are underpredicted under imbalance (605/1193/1394). Since the linear structure already serves the data well, the next lever is class imbalance: keeping the proven scaled-median logistic setup with C=1.0 but adding balanced class weights should lift minority-class recall and thus macro_f1 without changing the model family or discarding the linear signal.
- Trial 4: Unregularized logistic (C=1.0) is still the best result (macro_f1 0.547); balanced weighting at C=1.0 dropped to 0.536 and ExtraTrees underperformed (0.504), confirming signal is largely linear and features vary widely in scale (e.g. std of col7 ~0.003 vs col6 ~56.7), so scaling matters. Moving to weaker regularization C=10.0 keeps the proven scaled-median setup and nearest-neighbor preprocessing but lets the linear boundary fit sharper class distinctions, which may lift macro_f1 on the imbalanced 3-class problem without an over-regularization penalty at C=1.0.
- Trial 5: Logistic regression with scaled-median preprocessing dominates: C=1.0 gave macro_f1 0.547 and C=10.0 improved further to 0.551, while class-weight balancing (0.536) and ExtraTrees (0.504) both hurt, confirming a largely linear signal with weak over-regularization cost. Since increasing C from 1 to 10 helped, continuing in the same direction is the most evidence-consistent next step, but the candidate space has no logistic row beyond C=10.0 with null class_weight. The nearest unexplored multi-class nonlinear option is boosted trees: histogram gradient boosting with a moderate 0.1 learning rate can capture any residual interactions while its shrinkage controls overfitting, exploiting the categorical column without abandoning the scaled-median pipeline.
- Trial 6: Evidence shows a largely linear signal: logistic C=10.0 (0.551) and C=1.0 (0.547) lead, while ExtraTrees (0.504) and balanced weighting (0.536) hurt. HistGradientBoosting at lr 0.1 already matched the best logistic (0.552), suggesting mild nonlinear gains are accessible. Since no logistic candidate beyond C=10.0 with null class_weight remains untried, the least-redundant novel exploration in the boosted family is CatBoost, which handles the categorical column natively and generally differs from histogram GB in regularization/leaf growth. depth 4 with 200 iterations, lr 0.1, l2 3.0, no class weights keeps settings closest to the best-performing unweighted configurations and avoids the imbalance handling that previously degraded results.
- Trial 7: CatBoost depth 4 with unweighted loss is the clear leader (macro_f1 0.570), beating linear (0.551) and histogram GB (0.552), while ExtraTrees (0.504) and balanced class weights (0.536) hurt. Boosting clearly extracts useful nonlinearity that linear models miss. Depth 4 worked; the untried depth-6 CatBoost with identical otherwise-default settings is the natural next step to test whether greater tree depth captures additional interactions without destabilizing. All tried balance-weight variants degraded results, so keep null class weights and the proven scaled-median pipeline.

## Final test (not used for selection)

```json
{
  "accuracy": 0.6231203007518797,
  "macro_f1": 0.5939693522011641
}
```

## Limitations

Finite search space; observed training devices: cpu, gpu; one validation split per run. Scores on these datasets do not establish broad generalization or LLM superiority. Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons.
