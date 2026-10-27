# Changelog

## Unreleased

- Add bounded, stateless local image descriptors and character-hash text features for mixed classification,
  raw-schema-aware exported prediction, input-asset fingerprints and outer image/subject grouping.
- Protect benchmark outputs with a process lock, atomic checkpoints, protocol-checked resume and read-only
  planning; never retry an unfinished or unknown-usage run automatically.
- Bound production HTTP with a separate-process total deadline, response-size cap and conservative usage
  handling; test with a real loopback slow-response server without any external API call.
- Reserve complete request and candidate-training windows before live planning; recheck time after setup
  and before training, and retain detailed timeout, elapsed-time and error-status evidence.
- Offer untried canonical candidate IDs and bounded rejection feedback to V1; retain legacy specs,
  fixed search-space validation and the three-invalid-plan limit.
- Audit bounded custom member collections, cycles and unknown structures conservatively; keep nested
  CatBoost early-stopping evidence scoped to the helper's fitted model.
- Add fitted-device reports, internal-monitoring/full-refit CatBoost helpers and V2 GPU availability policy.
- Keep the incomplete V1/V2 comparison and unknown usage unchanged; no new paid benchmark is included.

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
