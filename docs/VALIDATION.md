# Validation status (2026-10-27)

Validated locally on Windows, Python 3.11.16, RTX 5070 Laptop GPU (~8 GB VRAM), ~32 GB RAM.

- Offline end-to-end demo: five candidates, successful final model export and prediction.
- Local full test suite: **49 passed, 0 skipped**. This includes 45 offline tests and four actual Docker
  integration tests. A separate network-disabled Linux container passed all 45 offline tests.
  No paid API requests are made by these tests.
- Static checks: `ruff check .` passes.
- GPU: CatBoost successfully trained with `task_type=GPU`; real-data benchmark workers also recorded GPU use.
- Real data: official Wine Quality and Adult downloaded, SHA-256/provenance saved, preparation completed.
- Reference benchmarks: 12/12 runs completed, no candidate failures, three seeds per dataset/strategy.
  See [published records](../examples/benchmark-local/README.md). No live LLM calls were made.
- Full V1 comparison: **36/36 runs completed**, four datasets (Breast Cancer, Wine, Wine Quality, Adult),
  three seeds and three strategies. The 12 live Agent runs made 74 API calls; estimated cost CNY 0.92692296.
  All 216 executed candidates succeeded. The audit independently recalculated test metrics, matched
  splits between strategies, verified validation-based selection, registry membership and the usage ledger.
  See [all records](../examples/benchmark-full/README.md) and [audit](../examples/benchmark-full/audit.json).
  Agent tied baseline/random on the small datasets, trailed random on Wine Quality and had a higher
  Adult mean; three seeds do not establish statistical significance. The earlier live Wine Quality
  smoke run is retained separately and is not substituted into this matrix.
- Docker V2: **actual training, host scoring, export and prediction pass locally on Windows**, in addition
  to the earlier Linux CI
  ([verified CI run](https://github.com/HardyYip/tabular-lab-agent/actions/runs/40275274895)). The local image
  is `tabular-lab-worker:0.1.0`, running Linux/amd64 as UID/GID 65534, with the configured 4 CPU/8 GB limits.
  The repair path in the local integration test uses recorded responses. The suite additionally covers
  default CatBoost file output and timeout cleanup. The Python base image is pinned by digest.
- Live DeepSeek V2: **one real demo completed end to end** (`configs/codegen-smoke.yaml`,
  `runs/20261026T133749Z-0f212fa3`). Two fixed baselines and three generated attempts ran; the generated
  attempts made three real API calls. Trials 2 and 3 failed with `CatBoostError`; the next requests carried
  `repair_attempt: 1` and `2` feedback, and trial 4 (the third generated attempt) succeeded with the best
  validation macro-F1 0.8570010. The selected candidate was retrained in Docker and exported, and loading
  it in a separate process to predict the 72 test rows reproduced the exported predictions
  (`export-check.json`). Final test macro-F1 was 0.9229947 at an estimated CNY 0.02667168 over 32.2 s. This is synthetic 360-row demo data: it shows one successful
  generate/repair/score/export loop, not general repair ability or an advantage over random search. The
  `CatBoostError` root cause was subsequently reproduced: default file output could not create
  `catboost_info` in the read-only working directory. Both original candidates failed unchanged and
  succeeded with either disabled file output or writable scratch. The executor now sets `/tmp` as its
  working directory, preserving the read-only root and bounded tmpfs. The successful third demo
  candidate changed estimator, so it did not repair that specific cause.
- Live V2 on real data: Wine Quality test macro-F1 **0.6131776**, Adult **0.8148827**; five candidates
  per task, four/three real API calls, estimated CNY 0.06132456 / 0.03428784. Both selected generated
  pipelines were refitted/exported, and new-process Docker predictions matched all 1,064 / 16,281 rows.
  Wine Quality retained one failed candidate. These are single-seed functional checks, not fair V1/V2
  performance comparisons. [Public summaries and generated source](../examples/live-v2/README.md).
- Successful-code memory: a separate real demo made four calls; three requests included the best
  successful candidate's actual source and validation metrics. Its final test macro-F1 **0.8794643**
  was lower than the original demo and is retained. New-process predictions matched. A replay regression
  verifies failed-source repair followed by successful-source feedback. Syntax failures preserve source
  and return safe numeric line/offset instead of raw source excerpts or exception strings.
- Credential persistence: `auth login` / `auth status` / `auth logout` save the DeepSeek key in Windows
  Credential Manager for the current user through native `CredReadW` / `CredWriteW` / `CredDeleteW`; an
  explicit `DEEPSEEK_API_KEY` still takes priority and no plaintext `.env` is loaded. Tests cover read
  priority, hidden input, log secrecy, stopping before any HTTP call when the store fails, and container
  environment isolation. A round trip was also exercised locally with a randomly named throwaway
  credential: written in one process, read in a new process, then deleted. The real user key was then
  saved through the same hidden-input command and read back in a further new process without being
  printed. The no-network `scripts/verify_credentials.py` check confirms header delivery and log secrecy;
  subsequent real V1/V2 experiments also succeeded using the saved credential. The mock check alone
  is not evidence of provider authentication.
- Windows CLI discovery: fresh per-user/all-users Docker Desktop installations are detected even when
  an already-running terminal has not inherited the installer's PATH update. A regression test covers this case.
- Local build required an isolated anonymous Docker configuration to fetch the public official Python
  base image after the signed-in token request was reset. The existing Docker login and system proxy were not changed.

These limitations are part of the delivery. No mock result is presented as a live-LLM result, and no
training-set GPU smoke-check score is presented as generalization performance.
