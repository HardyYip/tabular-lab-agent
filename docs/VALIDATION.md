# Validation status (2026-10-26)

Validated locally on Windows, Python 3.11.16, RTX 5070 Laptop GPU (~8 GB VRAM), ~32 GB RAM.

- Offline end-to-end demo: five candidates, successful final model export and prediction.
- Offline tests: 30 tests cover training, prediction, accounting, search and publication sanitization;
  the separate real-Docker integration test skips on this machine.
- Static checks: `ruff check .` passes.
- GPU: CatBoost successfully trained with `task_type=GPU`; real-data benchmark workers also recorded GPU use.
- Real data: official Wine Quality and Adult downloaded, SHA-256/provenance saved, preparation completed.
- Reference benchmarks: 12/12 runs completed, no candidate failures, three seeds per dataset/strategy.
  See [published records](../examples/benchmark-local/README.md). No live LLM calls were made.
- Live DeepSeek: **not executed**; no `DEEPSEEK_API_KEY` was configured during development. HTTP behavior,
  usage accounting, failure handling and feedback control were tested with mock HTTP responses.
- Docker V2: source, command construction and boundary tests implemented; **local container integration
  not executed** because Docker is not installed. The integration test skips explicitly. A separate
  Linux CI job builds the worker image and attempts this test; do not call it passing until CI confirms.

These limitations are part of the delivery. No mock result is presented as a live-LLM result, and no
training-set GPU smoke-check score is presented as generalization performance.
