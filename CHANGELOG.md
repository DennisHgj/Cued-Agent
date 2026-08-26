# Repository change log

This file records the maintainer-facing changes made while preparing Cued-Agent
for reproducible use after the ACM Multimedia 2025 paper release. Earlier
research snapshots remain available in the Git history.

## 2026-08-26 — Reproduction training contract

- Prevented lip/decoder reproduction training from silently falling back to
  random initialization when the base VSR checkpoint path is omitted.
- Added explicit `allow_random_initialization=true` opt-in for from-scratch
  controls and accepted full-checkpoint resume as the other valid start mode.
- Required the historical base checkpoint to match at least 99% of model
  tensors before any weights are loaded; the validated base transfers 762/767.
- Documented the original dynamic frame budget and moved hand-prompt beam/PER
  evaluation to post-training checkpoint selection instead of every epoch.
- Added regression tests for initialization, checkpoint, frame-budget, and
  compatibility-threshold failures.

## 2026-08-18 — Documentation and project links

- Added this repository change log.
- Added the maintainer's [personal homepage](https://dennishgj.github.io/) to
  the README.

## 2026-08-18 — External API contract hardening

Merged in [PR #3](https://github.com/DennisHgj/Cued-Agent/pull/3), commit
[`63dc4c5`](https://github.com/DennisHgj/Cued-Agent/commit/63dc4c529068c49376a5334c02de2d111675c4ab).

- Migrated the default hand-recognition model from the deprecated GPT-4o
  snapshot to `gpt-4.1`, while retaining an explicit paper-era override.
- Changed hand support images and keyframes from low-detail to high-detail
  vision inputs.
- Added hand-label range, frame-order, result-count, and structured-output
  validation.
- Added compatibility for both current OpenAI structured parsing and the beta
  parser exposed by OpenAI SDK 1.x in the original `auto_avsr` environment.
- Migrated the P2W default to `deepseek-v4-pro`, enabled JSON Output, and added
  explicit handling for empty, truncated, malformed, and incomplete responses.
- Added real OpenAI Python SDK request/response contract tests backed by
  in-memory HTTP transports, without making paid API requests.
- Validated all 18 tests and Python 3.8 compilation in the research server's
  `auto_avsr` environment.

## 2026-08-18 — Repository reorganization and release validation

Merged in [PR #2](https://github.com/DennisHgj/Cued-Agent/pull/2), commit
[`78099b3`](https://github.com/DennisHgj/Cued-Agent/commit/78099b35da8431be975e58f0e625bd0bde4dc7ae).

- Added the maintained `cued_agent` inference package and single-video and
  batch command-line entry points.
- Rebuilt the end-to-end path from video preprocessing through lip encoding,
  hand cue recognition or precomputed hand labels, prompt fusion, joint
  CTC/attention decoding, and optional P2W correction.
- Clarified that only the lip encoder, CTC projection, and attention decoder are
  trained; hand prompt construction and P2W correction remain inference-only.
- Added checkpoint validation, API-free ablation modes, reproducible hand-result
  input, and clearer failure messages.
- Added repository-owned unit and CLI tests plus inference, checkpoint, and
  training documentation.
- Validated pure-lip inference with the 2.87 GB research checkpoint, offline
  hand-prompt fusion across eight slow-motion groups, and one real training and
  validation batch on an RTX A6000.
