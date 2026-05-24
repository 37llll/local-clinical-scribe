# Changelog

## 0.3.1 - 2026-05-24

- Fixed public CI lightweight dependency set by adding `python-multipart`.

## 0.3.0 - 2026-05-24

- Prepared repository for public visibility.
- Added MIT license, security policy, third-party notices, and public release
  checklist.
- Added synthetic demo data policy.
- Added tracked-file public safety scanner.
- Added lightweight CI workflow for safety scan and smoke tests.
- Removed stale local private-history refs and ignored local runtime artifacts
  before publication.

## 0.2.0 - 2026-05-24

- Added local encounter persistence under `data/encounters`.
- Draft generation now stores a retrievable encounter record.
- Added clinician review finalization payload.
- Added Markdown and JSON export support.
- Added encounter list/get/export API endpoints.
- Added smoke test for persistence, finalization, and export.

## 0.1.0 - 2026-05-24

- Created a clean product repository baseline.
- Added lazy ASR service initialization.
- Added structured clinical note draft API.
- Added transcript evidence spans to generated note sections.
- Added privacy-focused repository ignores for audio, speaker embeddings, logs,
  runtime data, and model artifacts.
