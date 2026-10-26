# Validation status (2026-10-26)

Validated locally on Windows, Python 3.11.16, RTX 5070 Laptop GPU (~8 GB VRAM), ~32 GB RAM.

- Offline end-to-end demo: five candidates, successful final model export and prediction.
- Local full test suite: **33 passed, 0 skipped** after Docker setup. This includes 31 offline tests
  and two actual Docker integration tests; no paid API requests are made by these tests.
- Static checks: `ruff check .` passes.
- GPU: CatBoost successfully trained with `task_type=GPU`; real-data benchmark workers also recorded GPU use.
- Real data: official Wine Quality and Adult downloaded, SHA-256/provenance saved, preparation completed.
- Reference benchmarks: 12/12 runs completed, no candidate failures, three seeds per dataset/strategy.
  See [published records](../examples/benchmark-local/README.md). No live LLM calls were made.
- Live DeepSeek V1: **one Wine Quality run completed**, with eight candidate experiments and six successful
  API responses. Estimated model cost was CNY 0.07165368; final-test macro-F1 was 0.5991108. This single
  run verifies the live loop, not superiority over random search. Multi-dataset, three-seed live comparisons remain pending.
- Docker V2: **actual training, host scoring, export, prediction and replay-driven repair now pass locally
  on Windows**, in addition to the earlier Linux CI
  ([verified CI run](https://github.com/HardyYip/tabular-lab-agent/actions/runs/40275274895)). The local image
  is `tabular-lab-worker:0.1.0`, running Linux/amd64 as UID/GID 65534, with the configured 4 CPU/8 GB limits.
  The repair test uses recorded responses; real DeepSeek-generated V2 code remains to be validated.
- Windows CLI discovery: fresh per-user/all-users Docker Desktop installations are detected even when
  an already-running terminal has not inherited the installer's PATH update. A regression test covers this case.
- Local build required an isolated anonymous Docker configuration to fetch the public official Python
  base image after the signed-in token request was reset. The existing Docker login and system proxy were not changed.

These limitations are part of the delivery. No mock result is presented as a live-LLM result, and no
training-set GPU smoke-check score is presented as generalization performance.
