# Architecture and trust boundaries

The task contract is YAML validated by Pydantic. CSV paths resolve relative to that YAML; CLI overrides
are revalidated together. The profile describes **training data only**. Integer class encoding and
feature ordering are fixed by the evaluator, not the planner.

## V1

`runner` owns the state machine: baselines → proposal → validation → bounded worker → feedback →
selection → train+validation refit → final test. Ties retain the earlier candidate. Invalid plans do not
consume executed trial slots but three consecutive invalid plans end search. All actual LLM responses,
including malformed JSON, are accounted for before semantic validation.

`ExperimentSpec` permits only known algorithms, checked parameter bounds and preprocessing fields.
The shared finite space supports a meaningful random-vs-LLM comparison. Canonical fingerprints include
default parameters and exclude hypotheses. No task-name routing is used. Dataset preparation is separate
from algorithm selection.

Trusted subprocesses receive per-trial input artifacts and a minimal environment without API secrets.
They have wall-clock timeouts and thread limits; they are **not a security sandbox**. Normal sklearn
pipelines fit imputation/scaling/selection inside the training call. CatBoost uses native categorical
columns with training-only numeric imputation. Sparse encoding is preserved where possible; a shape
guard rejects oversized dense conversions for histogram boosting.

The full labeled test set stays out of workers during search. The evaluator receives no candidate
test metrics until selection is finished. External split overlap checks use feature hashes only.
Internal repeated feature vectors are grouped into approximately 60/20/20 folds.

## V2

Only the Docker worker imports generated source. A candidate defines `build_pipeline(seed,
feature_schema)`; the fixed worker calls fit/predict. Validation labels stay on the host and are used
by the same metric implementation as V1. Immutable image IDs are recorded with each candidate and
prediction bundle. Generated pickle artifacts never enter host `joblib.load`.

The executor sets the working directory to bounded writable `/tmp` scratch. The root and input mounts
remain read-only. This supports libraries such as CatBoost that otherwise try to create training logs
in the read-only image working directory. Syntax failures preserve source and return line/offset
without forwarding source excerpts or arbitrary error messages.

Each candidate can be repaired at most twice; repairs count toward the overall executed-trial limit.
Repair context contains exception types and stack locations, avoiding arbitrary exception strings
that could expose raw feature values. Container execution is isolated from the API client and caller
environment. The restriction is appropriate for a local learning tool; adversarial multiuser hosting
would require stronger isolation and output/disk quotas.

## Cost and reproduction

On Windows, `credentials.get_api_key()` checks an environment override and then the current user's
Credential Manager store. Keys are resolved in the host HTTP client, never placed in task YAML,
experiment logs or the generated-code container. Linux/CI use environment injection. The test suite
blocks both saved-key reads and construction of the real Windows credential store.

Reservations use conservative input-byte estimates plus a maximum completion allowance. Actual usage
is saved per response at configured peak/cache-miss prices. Unknown usage stops all further calls.
HTTP/transport errors do not automatically retry and are never assumed free. This prioritizes accounting
integrity over availability. Validly billed but malformed JSON can be retried within the invalid-plan cap.

Reproducibility has two levels: replay the recorded decisions, or rerun live planning. Only the former
fixes the decision sequence. Record code dirty state, versions, data hashes, seed, device and pricing;
GPU floating-point reductions can still change results slightly.
