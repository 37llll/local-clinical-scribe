"""Local JSON encounter store."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import ENCOUNTER_DIR

from .exporter import ClinicalNoteExporter
from .schemas import (
    ClinicalNoteDraft,
    ClinicalNoteFinalizeRequest,
    EncounterRecord,
    EncounterSummary,
)


class EncounterStore:
    """Persist encounter drafts and reviewed notes on the local filesystem."""

    def __init__(self, root_dir: Optional[str] = None):
        self.root_dir = Path(root_dir or ENCOUNTER_DIR)
        self.exporter = ClinicalNoteExporter()

    def save_draft(
        self,
        draft: ClinicalNoteDraft,
        *,
        transcript: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EncounterRecord:
        now = self._now()
        encounter_id = draft.encounter_id or self._new_encounter_id()
        draft = draft.model_copy(update={"encounter_id": encounter_id})

        existing = self._load_raw(encounter_id)
        created_at = existing.get("created_at") if existing else now

        record = EncounterRecord(
            encounter_id=encounter_id,
            status=draft.status,
            created_at=created_at,
            updated_at=now,
            draft=draft,
            transcript=transcript or {},
            metadata=metadata or {},
            finalized_note=existing.get("finalized_note") if existing else None,
            exports=existing.get("exports", {}) if existing else {},
        )
        self._save_record(record)
        return record

    def finalize(self, payload: ClinicalNoteFinalizeRequest) -> EncounterRecord:
        existing = self._load_raw(payload.encounter_id)
        if not existing:
            raise FileNotFoundError(f"Encounter not found: {payload.encounter_id}")

        draft = ClinicalNoteDraft(**existing["draft"])
        finalized_at = self._now()
        markdown = self.exporter.sections_to_markdown(
            payload.sections,
            encounter_id=payload.encounter_id,
            reviewer=payload.reviewer,
            review_notes=payload.review_notes,
        )

        finalized_note = {
            "status": "finalized",
            "finalized_at": finalized_at,
            "reviewer": payload.reviewer,
            "review_notes": payload.review_notes,
            "sections": [section.model_dump() for section in payload.sections],
        }
        existing["status"] = "finalized"
        existing["updated_at"] = finalized_at
        existing["draft"] = draft.model_copy(update={"status": "finalized"}).model_dump()
        existing["finalized_note"] = finalized_note
        existing["exports"] = {
            **existing.get("exports", {}),
            "markdown": markdown,
            "json": finalized_note,
        }

        record = EncounterRecord(**existing)
        self._save_record(record)
        return record

    def get(self, encounter_id: str) -> EncounterRecord:
        raw = self._load_raw(encounter_id)
        if not raw:
            raise FileNotFoundError(f"Encounter not found: {encounter_id}")
        return EncounterRecord(**raw)

    def list(self) -> List[EncounterSummary]:
        records = []
        for path in sorted(self.root_dir.glob("*.json"), reverse=True):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                draft = raw.get("draft", {})
                records.append(
                    EncounterSummary(
                        encounter_id=raw["encounter_id"],
                        status=raw.get("status", "unknown"),
                        created_at=raw.get("created_at", ""),
                        updated_at=raw.get("updated_at", ""),
                        template=draft.get("template", "soap"),
                        section_count=len(draft.get("sections", [])),
                    )
                )
            except Exception:
                continue
        return records

    def export(self, encounter_id: str, fmt: str) -> Any:
        record = self.get(encounter_id)
        fmt = fmt.lower()
        if fmt in ("md", "markdown"):
            markdown = record.exports.get("markdown")
            if markdown:
                return markdown
            return self.exporter.draft_to_markdown(record.draft)
        if fmt == "json":
            return record.model_dump()
        raise ValueError("Unsupported export format. Use markdown or json.")

    def _save_record(self, record: EncounterRecord):
        self.root_dir.mkdir(parents=True, exist_ok=True)
        path = self._record_path(record.encounter_id)
        path.write_text(
            json.dumps(record.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_raw(self, encounter_id: str) -> Dict[str, Any]:
        path = self._record_path(encounter_id)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _record_path(self, encounter_id: str) -> Path:
        safe_id = "".join(
            char for char in encounter_id if char.isalnum() or char in ("-", "_")
        )
        if not safe_id:
            raise ValueError("Invalid encounter id.")
        return self.root_dir / f"{safe_id}.json"

    def _new_encounter_id(self) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"enc_{stamp}_{uuid.uuid4().hex[:8]}"

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
