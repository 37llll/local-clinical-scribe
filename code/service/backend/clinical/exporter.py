"""Export helpers for clinical note drafts and reviewed notes."""

import json
from typing import Any, Dict, Iterable, List, Optional

from .schemas import ClinicalNoteDraft, ClinicalNoteSection


class ClinicalNoteExporter:
    """Render clinical notes to portable formats."""

    def draft_to_markdown(
        self,
        draft: ClinicalNoteDraft,
        *,
        title: str = "Clinical Note",
        reviewer: Optional[str] = None,
        review_notes: Optional[str] = None,
    ) -> str:
        lines: List[str] = [
            f"# {title}",
            "",
            f"- Encounter ID: {draft.encounter_id or 'N/A'}",
            f"- Template: {draft.template}",
            f"- Status: {draft.status}",
        ]
        if reviewer:
            lines.append(f"- Reviewer: {reviewer}")
        if review_notes:
            lines.extend(["", "## Review Notes", "", review_notes.strip()])

        lines.extend(["", "## Sections", ""])
        for section in draft.sections:
            lines.extend(self._section_to_markdown(section))

        if draft.warnings:
            lines.extend(["", "## Warnings", ""])
            for warning in draft.warnings:
                lines.append(f"- {warning}")

        lines.extend(["", "## Disclaimer", "", draft.disclaimer, ""])
        return "\n".join(lines)

    def draft_to_json(self, draft: ClinicalNoteDraft) -> Dict[str, Any]:
        return draft.model_dump()

    def record_to_json(self, record: Dict[str, Any]) -> str:
        return json.dumps(record, ensure_ascii=False, indent=2)

    def _section_to_markdown(self, section: ClinicalNoteSection) -> List[str]:
        lines = [
            f"### {section.title}",
            "",
            f"Status: `{section.status}`",
            "",
            section.content or "",
        ]

        if section.evidence:
            lines.extend(["", "Evidence:"])
            for item in section.evidence:
                timestamp = self._format_time_range(item.start, item.end)
                speaker = item.speaker or "unknown"
                lines.append(f"- `{item.segment_id}` {timestamp} {speaker}: {item.quote}")

        lines.append("")
        return lines

    def sections_to_markdown(
        self,
        sections: Iterable[ClinicalNoteSection],
        *,
        encounter_id: str,
        reviewer: Optional[str] = None,
        review_notes: Optional[str] = None,
    ) -> str:
        draft = ClinicalNoteDraft(
            encounter_id=encounter_id,
            status="finalized",
            sections=list(sections),
            warnings=[],
        )
        return self.draft_to_markdown(
            draft,
            title="Final Clinical Note",
            reviewer=reviewer,
            review_notes=review_notes,
        )

    def _format_time_range(
        self, start: Optional[float], end: Optional[float]
    ) -> str:
        if start is None and end is None:
            return ""
        if start is None:
            return f"[? - {end:.2f}s]"
        if end is None:
            return f"[{start:.2f}s - ?]"
        return f"[{start:.2f}s - {end:.2f}s]"
