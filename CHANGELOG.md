# Changelog

## 0.2.0 — 2026-10-27

- Complete live V1 comparisons: four datasets, three seeds, baseline/random/Agent, 36 completed runs.
  Keep ties and regressions: Agent slightly trails random on Wine Quality and has a higher Adult mean.
- Validate V2 on synthetic data, Wine Quality and Adult, including fresh-process Docker prediction.
- Persist DeepSeek keys in Windows Credential Manager; keep environment overrides and isolated tests.
- Fix default CatBoost file output with writable bounded scratch and a read-only container root.
- Feed the best successful generated source, validation metrics and execution limits into later proposals.
- Preserve syntax-failed source and safe line/offset feedback; verify real container timeout cleanup.
- Stop a benchmark suite after unknown API usage rather than initiating additional paid runs.
- Publish sanitized experiment records, archived generated source, comparisons, costs and independent audits.
- Pin the Docker Python base by digest; keep the recorded old image available for existing models.
- Verify 49 Windows tests, 45 Linux offline tests, credential header delivery and wheel packaging.

## 0.1.0

Initial structured-experiment Agent, trusted evaluators, reproducible local references and optional
Docker-generated pipeline execution. Early mock/replay validations were separated from live evidence.
