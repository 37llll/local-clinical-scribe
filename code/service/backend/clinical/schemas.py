"""Pydantic schemas for structured clinical note drafting and review."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TranscriptSegment(BaseModel):
    """One diarized transcript segment."""

    model_config = ConfigDict(extra="allow")

    id: Optional[str] = Field(default=None, description="Stable segment id")
    start: Optional[float] = Field(default=None, description="Start time in seconds")
    end: Optional[float] = Field(default=None, description="End time in seconds")
    speaker: Optional[str] = Field(default=None, description="Speaker label")
    text: str = Field(..., description="Transcript text")


class ClinicalNoteRequest(BaseModel):
    """Draft a note from a transcript."""

    model_config = ConfigDict(extra="forbid")

    encounter_id: Optional[str] = None
    template: str = Field(default="soap", description="Current v0.1 template")
    language: str = Field(default="zh-CN")
    text: Optional[str] = Field(
        default=None,
        description="Fallback full transcript when segments are unavailable",
    )
    segments: List[TranscriptSegment] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_transcript(self):
        if not self.segments and not (self.text and self.text.strip()):
            raise ValueError("Either segments or text must be provided.")
        return self


class EvidenceSpan(BaseModel):
    """A quote from the source transcript that supports a note field."""

    segment_id: str
    start: Optional[float] = None
    end: Optional[float] = None
    speaker: Optional[str] = None
    quote: str


class ClinicalNoteSection(BaseModel):
    """One section in the generated note draft."""

    key: str
    title: str
    status: str = Field(
        default="draft",
        description="draft, missing, or needs_review",
    )
    content: str
    evidence: List[EvidenceSpan] = Field(default_factory=list)


class ClinicalNoteDraft(BaseModel):
    """Structured clinical note draft."""

    encounter_id: Optional[str] = None
    template: str = "soap"
    status: str = "draft_requires_clinician_review"
    sections: List[ClinicalNoteSection]
    warnings: List[str] = Field(default_factory=list)
    transcript_stats: Dict[str, Any] = Field(default_factory=dict)
    disclaimer: str = (
        "This is a documentation draft generated from the transcript. "
        "A licensed clinician must review, edit, and approve it before use."
    )


class ClinicalNoteFinalizeRequest(BaseModel):
    """Clinician-reviewed final note payload."""

    model_config = ConfigDict(extra="forbid")

    encounter_id: str
    sections: List[ClinicalNoteSection]
    reviewer: Optional[str] = None
    review_notes: Optional[str] = None

    @model_validator(mode="after")
    def require_sections(self):
        if not self.sections:
            raise ValueError("At least one reviewed section is required.")
        return self


class EncounterRecord(BaseModel):
    """Persisted encounter record."""

    model_config = ConfigDict(extra="allow")

    encounter_id: str
    status: str
    created_at: str
    updated_at: str
    draft: ClinicalNoteDraft
    transcript: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    finalized_note: Optional[Dict[str, Any]] = None
    exports: Dict[str, Any] = Field(default_factory=dict)


class EncounterSummary(BaseModel):
    """Small record used by list endpoints."""

    encounter_id: str
    status: str
    created_at: str
    updated_at: str
    template: str
    section_count: int

