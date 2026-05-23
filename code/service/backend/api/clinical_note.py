"""Clinical note draft API."""

from typing import List, Optional

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from backend.clinical.draft_generator import ClinicalNoteDraftGenerator
from backend.clinical.schemas import ClinicalNoteRequest, TranscriptSegment
from backend.utils.file_utils import parse_speaker_list
from backend.utils.logger_manager import LoggerManager

logger = LoggerManager.get_backend_logger()

router = APIRouter(prefix="/api/clinical_note", tags=["结构化病历草稿"])
draft_generator = ClinicalNoteDraftGenerator()


@router.post("/from_transcript")
async def draft_from_transcript(payload: ClinicalNoteRequest):
    """Generate a reviewable note draft from transcript text or segments."""
    try:
        draft = draft_generator.generate(payload)
        return JSONResponse(content={"success": True, "clinical_note": draft.model_dump()})
    except Exception as exc:
        logger.error(f"[ClinicalNote] draft_from_transcript failed: {exc}")
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": str(exc)},
        )


@router.post("/from_audio")
async def draft_from_audio(
    audio_file: UploadFile = File(..., description="Audio file"),
    encounter_id: Optional[str] = Form(None),
    template: str = Form("soap"),
    enable_speaker_diarization: bool = Form(True),
    speaker_mode: str = Form("cluster"),
    registered_speakers: str = Form(None),
    similarity_threshold: float = Form(0.5),
):
    """Transcribe audio and return a structured clinical note draft."""
    try:
        from backend.api.offline_asr import get_offline_asr_service

        registered_speakers_list = parse_speaker_list(registered_speakers)
        transcript = await get_offline_asr_service().process_uploaded_file(
            upload_file=audio_file,
            enable_speaker_diarization=enable_speaker_diarization,
            speaker_mode=speaker_mode,
            registered_speakers=registered_speakers_list,
            similarity_threshold=similarity_threshold,
        )

        segments = _segments_from_transcript(transcript)
        request = ClinicalNoteRequest(
            encounter_id=encounter_id,
            template=template,
            text=transcript.get("text"),
            segments=segments,
            metadata={"source": "offline_asr"},
        )
        draft = draft_generator.generate(request)
        return JSONResponse(
            content={
                "success": True,
                "transcript": transcript,
                "clinical_note": draft.model_dump(),
            }
        )
    except Exception as exc:
        logger.error(f"[ClinicalNote] draft_from_audio failed: {exc}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(exc)},
        )


def _segments_from_transcript(transcript: dict) -> List[TranscriptSegment]:
    raw_segments = transcript.get("segments") or []
    if raw_segments:
        return [
            TranscriptSegment(
                id=f"seg_{index + 1:04d}",
                start=segment.get("start"),
                end=segment.get("end"),
                speaker=segment.get("speaker"),
                text=segment.get("text", ""),
            )
            for index, segment in enumerate(raw_segments)
            if segment.get("text")
        ]

    text = transcript.get("text") or ""
    if not text:
        return []

    return [
        TranscriptSegment(
            id="full_transcript",
            text=text,
            start=None,
            end=transcript.get("duration"),
            speaker=None,
        )
    ]
