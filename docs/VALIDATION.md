# Validation status (2026-10-26)

Validated locally on Windows, Python 3.11.16, RTX 5070 Laptop GPU (~8 GB VRAM), ~32 GB RAM.

- Offline end-to-end demo: five candidates, successful final model export and prediction.
- Offline tests: 30 tests cover training, prediction, accounting, search and publication sanitization;
  the real-Docker integration tests skip on this machine.
- Static checks: `ruff check .` passes.
- GPU: CatBoost successfully trained with `task_type=GPU`; real-data benchmark workers also recorded GPU use.
- Real data: official Wine Quality and Adult downloaded, SHA-256/provenance saved, preparation completed.
- Reference benchmarks: 12/12 runs completed, no candidate failures, three seeds per dataset/strategy.
  See [published records](../examples/benchmark-local/README.md). No live LLM calls were made.
- Live DeepSeek: **not executed**; no `DEEPSEEK_API_KEY` was configured during development. HTTP behavior,
  usage accounting, failure handling and feedback control were tested with mock HTTP responses.
- Docker V2: **Linux CI passed actual container training, host scoring, export and container prediction**
  ([verified run](https://github.com/HardyYip/tabular-lab-agent/actions/runs/40260328568)). Local Docker
  integration remains unavailable because Docker is not installed. A second integration scenario exercises
  the full controller with a recorded broken candidate followed by its repair; this uses replay, not live LLM calls.

These limitations are part of the delivery. No mock result is presented as a live-LLM result, and no
training-set GPU smoke-check score is presented as generalization performance.
