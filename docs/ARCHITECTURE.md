# Architecture and trust boundaries

The task contract is YAML validated by Pydantic. CSV paths resolve relative to that YAML; CLI overrides
are revalidated together. The profile describes **training data only**. Integer class encoding and
feature ordering are fixed by the evaluator, not the planner.

## Optional local media inputs

Explicit modality columns are deterministically encoded before splitting and profiling. Character hashes
learn no vocabulary/IDF; bounded image descriptors use only each image's own pixels. Raw media/text columns
are removed, and learned imputation/scaling remains inside the fit pipeline. Raw input schema and encoding
version are exported for prediction. Legacy tabular bundles retain their existing path.

Image bytes/pixels, path roots, text length and encoded feature cells are bounded. Exact decoded image
content and optional subject IDs define outer disjoint groups; subject grouping is not claimed for the
helper's internal holdout. CSV/image fingerprints enter manifests and resume protocols. Generated workers
receive encoded features rather than original media files, and the planner receives training summaries.

## V1

`runner` owns the state machine: baselines → proposal → validation → bounded worker → feedback →
selection → train+validation refit → final test. Ties retain the earlier candidate. Invalid plans do not
consume executed trial slots but three consecutive invalid plans end search. All actual LLM responses,
including malformed JSON, are accounted for before semantic validation.

The planner receives untried `candidate_options` with canonical fingerprint IDs. An ID resolves to the
executor's registered specification; the model supplies a hypothesis rather than arbitrary parameters.
Legacy experiment objects remain supported and undergo the same registry/duplicate checks. Rejected
plans receive bounded error feedback on the next attempt. Exhaustion stops without another API request.

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

V2 reuses task `device`: `cpu` does not request a GPU, `auto` resolves to GPU or CPU using an isolated
container `nvidia-smi` probe, and `gpu` requires that probe to succeed before any LLM call. Resolved GPU
access adds `--gpus=all`; the planner receives this capability but may still choose CPU estimators.
Training errors remain repair feedback rather than automatic source rewriting. The original requested
policy remains in `config.json`; manifests and summaries separately record resolved availability.
Candidate and exported image records preserve the resolved device for refitting and new-process
prediction. Prediction checks the saved image ID rather than requiring a mutable tag. Legacy image
records default to CPU. GPU-enabled bundles currently require GPU access even for prediction.

The fixed worker records fitted-model devices and bounded internal-training metadata. The host reads
only schema-validated JSON, never the generated model. Known CatBoost models report fitted `task_type`;
unsupported objects are unknown rather than inferred from GPU access. Runtime records feed subsequent
proposals and are copied from the model actually exported, including the candidate fallback on refit failure.
CatBoost early stopping uses the supplied helper with an internal, stratified/group-disjoint monitoring
fold and preprocessing fitted on the internal training fold. It then refits on all current fit data using
the selected iterations and learning rate. External validation labels never enter this process. Detected
CatBoost detectors without the helper's training record are rejected; fixed-round training is allowed.

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

Live planning requires the full configured HTTP timeout plus one full trial window to remain before the
search deadline. The finalization reserve is outside that deadline. The HTTP client rechecks after setup;
the runner rechecks the trial window after receiving a plan. Normal time admission stops do not become
unknown usage or planning failures. Default production HTTP runs in a separate spawned process with a
total wait deadline, a bounded response and no redirects. Timeouts terminate it and retain unknown usage;
startup/cleanup overhead still exists, and local termination does not imply free or cancelled server work.
Injected test/custom clients retain phase timeouts. Events record which path enforces the total deadline.

Benchmark output directories are locked and protected against overwrite. Resume compares source,
configuration/order, input assets, environment and immutable image IDs, then verifies saved runs. A completed
orphan can be recovered with unknown outer wall time; unfinished or unknown-usage runs are never retried
automatically. Atomic JSON is the authoritative checkpoint and CSV summaries are replaceable derivatives.

Reproducibility has two levels: replay the recorded decisions, or rerun live planning. Only the former
fixes the decision sequence. Record code dirty state, versions, data hashes, seed, device and pricing;
GPU floating-point reductions can still change results slightly.
