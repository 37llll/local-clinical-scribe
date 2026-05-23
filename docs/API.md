# API Notes

## Capabilities

`GET /capabilities`

Shows whether optional audio endpoints are available in the current runtime.
The clinical note, encounter store, and export APIs are core capabilities and do
not require ASR model dependencies.

## Clinical Note Draft From Transcript

`POST /api/clinical_note/from_transcript`

Use this endpoint when ASR has already produced transcript segments.

## Clinical Note Draft From Audio

`POST /api/clinical_note/from_audio`

Use this endpoint when you want the backend to run offline ASR first, then draft
a structured note.

## Encounter Review

`GET /api/clinical_note/encounters`

List saved local encounters.

`GET /api/clinical_note/encounters/{encounter_id}`

Fetch a saved encounter with draft, transcript metadata, finalized note, and
exports.

`POST /api/clinical_note/finalize`

Submit clinician-reviewed sections as the final note. The request body uses:

```json
{
  "encounter_id": "demo-001",
  "reviewer": "Dr. Demo",
  "review_notes": "Reviewed and corrected.",
  "sections": [
    {
      "key": "chief_complaint",
      "title": "Chief Complaint",
      "status": "reviewed",
      "content": "咳嗽伴夜间发热两天。",
      "evidence": []
    }
  ]
}
```

## Encounter Export

`GET /api/clinical_note/encounters/{encounter_id}/export?format=markdown`

`GET /api/clinical_note/encounters/{encounter_id}/export?format=json`

## Existing ASR Endpoints

- `POST /api/offline_asr/offline_asr`
- `WS /api/stream_asr/vad`
- `WS /api/stream_asr/streaming_asr`
- `WS /api/stream_asr/pipeline`

## Speaker Endpoints

- `POST /api/speaker/enroll`
- `GET /api/speaker/speakers`
- `DELETE /api/speaker/speakers/{speaker_name}`
- `POST /api/speaker/verify`
