# Live V2 code-generation evidence

These are real DeepSeek calls and actual Docker executions, with private data, binary models and
complete prompts excluded. Recorded costs use the configured pricing/FX snapshot, not provider invoices.

| Task | Generated attempts | Failed attempts | API calls | Test macro-F1 | Seconds | Estimated CNY |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Synthetic demo | 3 | 2 | 3 | 0.922995 | 32.2 | 0.026672 |
| Wine Quality | 3 | 1 | 4 | 0.613178 | 69.3 | 0.061325 |
| Adult | 3 | 0 | 3 | 0.814883 | 83.8 | 0.034288 |
| Additional memory-validation demo | 3 | 1 | 4 | 0.879464 | 44.0 | 0.038528 |

Each run also executed Dummy and logistic-regression baselines. The evaluator selected a generated
candidate using validation macro-F1, refitted it on train + validation, and evaluated the test once.
Every exported model was reloaded by a new process inside Docker: all 72 / 1,064 / 16,281 predictions
matched the original test predictions. See each task's `export-check.json`.

`candidate.py.txt` preserves the selected generated source as an audit artifact; it is not imported on
the host. `events.sample.jsonl` retains usage and experiment metadata without full prompts/responses.
The full local logs remain private and support replay. The code, hashes and actual pre-release package
versions are recorded in the corresponding manifests; these runs preceded the release commit.

The original demo's two CatBoost failures were caused by default training-file output in a read-only
working directory. Later controlled experiments reproduced both failures; disabling CatBoost file
output or changing to writable scratch fixed them. The successful third demo attempt switched to a
different estimator, so it is evidence of a working feedback/replanning loop, not a demonstrated fix
of that specific CatBoost cause. The executor now starts candidates in bounded `/tmp` scratch and
the root filesystem stays read-only. A real-container regression test covers default CatBoost output.

Wine Quality's last generated attempt failed with `TypeError`; the trial limit was reached and the
already successful best candidate was exported. Its four API calls include a rejected plan, so API
calls need not equal executed generated attempts. Failures are retained, not filtered out.

These are single-seed functional validations, not matched-budget comparisons between V1 and V2.
The additional `memory-demo` run verifies that three real requests received the best successful
generated source and its validation metrics. Its lower test score is retained, not replaced by the
earlier better demo result. This is a separate functional regression, not an additional independent
dataset or evidence of monotonic improvement. Its 72 exported predictions also matched in a new process.

V2 can generate models outside V1's finite candidate registry. See the separate V1 benchmark for
random-search versus LLM comparisons. Dataset preparation follows [the dataset protocol](../../docs/DATASETS.md).
