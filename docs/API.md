# API Notes

## Clinical Note Draft From Transcript

`POST /api/clinical_note/from_transcript`

Use this endpoint when ASR has already produced transcript segments.

## Clinical Note Draft From Audio

`POST /api/clinical_note/from_audio`

Use this endpoint when you want the backend to run offline ASR first, then draft
a structured note.

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

