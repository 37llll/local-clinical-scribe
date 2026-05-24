# Product Plan

## Product Thesis

Local Clinical Scribe should become a local-first clinical documentation engine
for small teams, research workflows, and private deployments. Its strongest
positioning is not "another ASR demo"; it is a reviewable clinical note drafting
workflow with transcript evidence and local data control.

## v0.1 Scope

- Keep the inherited FunASR ASR, VAD, punctuation, and speaker services.
- Prevent model loading at app import time.
- Add a clinical note draft API that works from existing transcript segments.
- Mark every generated section as review-required.
- Attach transcript evidence to every populated section.
- Explicitly mark missing content instead of inventing facts.

## v0.2 Scope

- Store encounters locally with audit-friendly metadata.
- Add clinician finalization state.
- Export Markdown and JSON.
- Keep DOCX and browser-based review UI for v0.3.

## v0.3 Scope

- Prepare the repository for public visibility.
- Add release safety scanning.
- Add security and third-party notices.
- Define synthetic demo data policy.

## v0.4 Scope

- Build a clinician review UI.
- Add editable sections in the browser.
- Add DOCX export.
- Add evidence viewer with segment playback hooks.

## v0.5 Scope

- Add quality metrics:
  - ASR word/character error rate hooks.
  - diarization accuracy hooks.
  - clinician edit distance.
  - unsupported claim checks.

## v0.6 Scope

- Add optional LLM-backed structuring.
- Enforce JSON schema validation.
- Require evidence ids for every generated statement.
- Add "not mentioned" and "uncertain" handling.

## v0.7 Scope

- Add FHIR-oriented export.
- Add webhooks and API keys.
- Add deployment recipes for private clinical environments.
