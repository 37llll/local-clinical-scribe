# Local Clinical Scribe

Local Clinical Scribe is a local-first clinical conversation tool that turns
recorded or streamed Chinese medical conversations into:

- diarized transcript segments
- timestamped ASR output
- reviewable structured note drafts
- evidence links back to source transcript spans

Version `0.2.0` turns the first draft API into a small product loop:
transcription, deterministic clinical note drafting, local encounter storage,
clinician finalization, and Markdown/JSON export. The note draft is never a
diagnosis or final medical record. It is clinician-reviewed documentation
support.

## Product Boundary

This project is designed for documentation support:

1. Capture or upload audio.
2. Run local ASR, VAD, punctuation, and optional speaker diarization.
3. Generate a structured draft from transcript evidence.
4. Let a clinician review, edit, and approve the final note.

It should not be used to automate diagnosis, prescribe medication, or make
clinical decisions without licensed clinician review.

## Current Capabilities

- FastAPI backend with REST and WebSocket endpoints.
- Offline ASR endpoint for uploaded audio.
- Streaming ASR/VAD WebSocket endpoints.
- Speaker enrollment, listing, deletion, and verification.
- Clinical note draft API:
  - `POST /api/clinical_note/from_transcript`
  - `POST /api/clinical_note/from_audio`
- Encounter review and export API:
  - `GET /api/clinical_note/encounters`
  - `GET /api/clinical_note/encounters/{encounter_id}`
  - `POST /api/clinical_note/finalize`
  - `GET /api/clinical_note/encounters/{encounter_id}/export?format=markdown`
- Lazy model loading, so the API can start before models are downloaded.
- Local-only runtime data directories for audio and speaker embeddings.

## Repository Hygiene

This repository intentionally does not commit:

- patient or sample audio
- speaker embeddings
- logs
- downloaded model artifacts
- historical internal Git history

Runtime data lives under `data/` by default and is ignored by Git.

## Quick Start

Create and activate a Python environment, then install dependencies:

```powershell
pip install -r requirements_cpu.txt
```

Download models when you want to run ASR:

```powershell
cd pretrained_models
python download_models.py
```

Start the API:

```powershell
cd code/service
python start_backend.py
```

Open:

- API docs: `http://localhost:63100/docs`
- Health check: `http://localhost:63100/health`
- Runtime capabilities: `http://localhost:63100/capabilities`

## Configuration

Useful environment variables:

```powershell
$env:PROJECT_ROOT="C:\path\to\local-clinical-scribe"
$env:LOCAL_CLINICAL_SCRIBE_DEVICE="cpu"
$env:PRELOAD_MODELS="false"
$env:BACKEND_PORT="63100"
$env:FRONTEND_PORT="63101"
$env:ENCOUNTER_DIR="C:\path\to\local-clinical-scribe\data\encounters"
```

Use `LOCAL_CLINICAL_SCRIBE_DEVICE=cuda:0` when you have a compatible GPU and
local models.

The core clinical note APIs can start without audio dependencies installed.
Audio endpoints appear only when optional runtime dependencies such as FunASR
and librosa are available; check `/capabilities` for the current state.

## Review And Export

Draft endpoints now save a local encounter record under `data/encounters`.
Review clients can submit edited sections to:

```text
POST /api/clinical_note/finalize
```

Then export the reviewed note:

```text
GET /api/clinical_note/encounters/{encounter_id}/export?format=markdown
GET /api/clinical_note/encounters/{encounter_id}/export?format=json
```

## Example: Draft From Transcript

```json
{
  "encounter_id": "demo-001",
  "template": "soap",
  "segments": [
    {
      "id": "seg_0001",
      "start": 0.0,
      "end": 3.2,
      "speaker": "patient",
      "text": "我这两天咳嗽，晚上有点发热。"
    },
    {
      "id": "seg_0002",
      "start": 3.5,
      "end": 8.1,
      "speaker": "doctor",
      "text": "建议先做血常规和胸片检查，注意休息，多喝水。"
    }
  ]
}
```

Send it to:

```powershell
Invoke-RestMethod `
  -Uri http://localhost:63100/api/clinical_note/from_transcript `
  -Method Post `
  -ContentType "application/json" `
  -Body (Get-Content examples/clinical_note_request.json -Raw)
```

## Version Plan

- `0.1.x`: local transcription, speaker handling, deterministic note draft.
- `0.2.x`: local encounter persistence, clinician review finalization,
  Markdown/JSON export.
- `0.3.x`: clinician review UI and richer export templates.
- `0.4.x`: evidence viewer, audit log, quality metrics.
- `0.5.x`: optional LLM structuring with schema validation and citations.
- `0.6.x`: FHIR-compatible export and integration hooks.

See [docs/PRODUCT_PLAN.md](docs/PRODUCT_PLAN.md) for the working roadmap.
