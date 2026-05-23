"""Rule-based v0.1 clinical note draft generator.

The first product version intentionally stays deterministic and evidence-led.
It does not diagnose or recommend treatment. It organizes transcript evidence
into a reviewable note draft and marks missing fields explicitly.
"""

import re
from typing import Callable, Iterable, List, Optional

from .schemas import (
    ClinicalNoteDraft,
    ClinicalNoteRequest,
    ClinicalNoteSection,
    EvidenceSpan,
    TranscriptSegment,
)


SYMPTOM_KEYWORDS = [
    "疼",
    "痛",
    "咳",
    "发热",
    "发烧",
    "头晕",
    "胸闷",
    "气短",
    "恶心",
    "呕吐",
    "腹泻",
    "乏力",
    "麻",
    "痒",
    "出血",
    "睡眠",
    "食欲",
]

PAST_HISTORY_KEYWORDS = [
    "既往",
    "以前",
    "病史",
    "高血压",
    "糖尿病",
    "冠心病",
    "手术",
    "住院",
    "过敏",
    "家族",
]

PLAN_KEYWORDS = [
    "建议",
    "需要",
    "检查",
    "复查",
    "用药",
    "药",
    "治疗",
    "观察",
    "注意",
    "先",
]

ASSESSMENT_KEYWORDS = [
    "考虑",
    "诊断",
    "可能",
    "倾向",
    "问题",
    "判断",
]


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _sentence_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[。！？!?；;])\s*", _compact_text(text))
    return [part.strip() for part in parts if part.strip()]


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _speaker_is_patient(segment: TranscriptSegment) -> bool:
    speaker = (segment.speaker or "").lower()
    return any(token in speaker for token in ["patient", "患者", "病人"])


def _speaker_is_clinician(segment: TranscriptSegment) -> bool:
    speaker = (segment.speaker or "").lower()
    return any(token in speaker for token in ["doctor", "clinician", "医生", "医师"])


class ClinicalNoteDraftGenerator:
    """Generate a reviewable clinical note draft from transcript segments."""

    def generate(self, request: ClinicalNoteRequest) -> ClinicalNoteDraft:
        segments = self._normalize_segments(request)
        warnings: List[str] = []

        sections = [
            self._build_section(
                key="subjective",
                title="Subjective",
                segments=segments,
                matcher=lambda seg: _speaker_is_patient(seg)
                or _contains_any(seg.text, SYMPTOM_KEYWORDS),
                fallback_title="No clear patient-reported subjective history found.",
            ),
            self._build_section(
                key="chief_complaint",
                title="Chief Complaint",
                segments=segments,
                matcher=lambda seg: _contains_any(seg.text, SYMPTOM_KEYWORDS),
                fallback_title="Chief complaint was not clearly stated in the transcript.",
                sentence_filter=lambda sentence: _contains_any(
                    sentence, SYMPTOM_KEYWORDS
                ),
            ),
            self._build_section(
                key="history",
                title="History",
                segments=segments,
                matcher=lambda seg: _contains_any(seg.text, PAST_HISTORY_KEYWORDS),
                fallback_title="Past history/allergy/family history not clearly found.",
            ),
            self._build_section(
                key="assessment",
                title="Assessment",
                segments=segments,
                matcher=lambda seg: _speaker_is_clinician(seg)
                and _contains_any(seg.text, ASSESSMENT_KEYWORDS),
                fallback_title="Assessment was not clearly documented in the transcript.",
            ),
            self._build_section(
                key="plan",
                title="Plan",
                segments=segments,
                matcher=lambda seg: _speaker_is_clinician(seg)
                and _contains_any(seg.text, PLAN_KEYWORDS),
                fallback_title="Plan was not clearly documented in the transcript.",
            ),
        ]

        for section in sections:
            if section.status == "missing":
                warnings.append(section.content)

        return ClinicalNoteDraft(
            encounter_id=request.encounter_id,
            template=request.template,
            sections=sections,
            warnings=warnings,
            transcript_stats={
                "segments": len(segments),
                "characters": sum(len(seg.text or "") for seg in segments),
                "speakers": sorted(
                    {seg.speaker for seg in segments if seg.speaker is not None}
                ),
            },
        )

    def _normalize_segments(
        self, request: ClinicalNoteRequest
    ) -> List[TranscriptSegment]:
        if request.segments:
            return [
                segment.model_copy(
                    update={"id": segment.id or "seg_%04d" % (index + 1)}
                )
                for index, segment in enumerate(request.segments)
                if _compact_text(segment.text)
            ]

        return [
            TranscriptSegment(
                id="full_transcript",
                start=None,
                end=None,
                speaker=None,
                text=request.text or "",
            )
        ]

    def _build_section(
        self,
        key: str,
        title: str,
        segments: List[TranscriptSegment],
        matcher: Callable[[TranscriptSegment], bool],
        fallback_title: str,
        sentence_filter: Optional[Callable[[str], bool]] = None,
    ) -> ClinicalNoteSection:
        matched = [segment for segment in segments if matcher(segment)]
        if not matched:
            return ClinicalNoteSection(
                key=key,
                title=title,
                status="missing",
                content=fallback_title,
                evidence=[],
            )

        content_parts: List[str] = []
        evidence: List[EvidenceSpan] = []

        for segment in matched[:8]:
            sentences = _sentence_split(segment.text)
            if sentence_filter:
                sentences = [
                    sentence for sentence in sentences if sentence_filter(sentence)
                ]
            text = _compact_text(" ".join(sentences) if sentences else segment.text)
            if not text:
                continue
            content_parts.append(text)
            evidence.append(
                EvidenceSpan(
                    segment_id=segment.id or "",
                    start=segment.start,
                    end=segment.end,
                    speaker=segment.speaker,
                    quote=text[:300],
                )
            )

        if not content_parts:
            return ClinicalNoteSection(
                key=key,
                title=title,
                status="missing",
                content=fallback_title,
                evidence=[],
            )

        return ClinicalNoteSection(
            key=key,
            title=title,
            status="needs_review",
            content=" ".join(content_parts),
            evidence=evidence,
        )

