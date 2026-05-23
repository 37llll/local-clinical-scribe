# Changelog

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
