"""Smoke test for local encounter persistence, review, and export."""

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVICE_DIR = ROOT / "code" / "service"
sys.path.insert(0, str(SERVICE_DIR))

from backend.clinical.draft_generator import ClinicalNoteDraftGenerator
from backend.clinical.encounter_store import EncounterStore
from backend.clinical.schemas import (
    ClinicalNoteFinalizeRequest,
    ClinicalNoteRequest,
)


def main():
    payload = json.loads(
        (ROOT / "examples" / "clinical_note_request.json").read_text(
            encoding="utf-8"
        )
    )
    request = ClinicalNoteRequest(**payload)
    draft = ClinicalNoteDraftGenerator().generate(request)

    with tempfile.TemporaryDirectory() as tmp_dir:
        store = EncounterStore(tmp_dir)
        record = store.save_draft(
            draft,
            transcript={
                "segments": [segment.model_dump() for segment in request.segments]
            },
            metadata={"test": True},
        )

        assert record.encounter_id == "demo-001"
        assert store.list()[0].encounter_id == "demo-001"

        reviewed_sections = [
            section.model_copy(update={"status": "reviewed"})
            for section in record.draft.sections
            if section.status != "missing"
        ]
        finalized = store.finalize(
            ClinicalNoteFinalizeRequest(
                encounter_id=record.encounter_id,
                sections=reviewed_sections,
                reviewer="Dr. Demo",
                review_notes="Reviewed in smoke test.",
            )
        )

        assert finalized.status == "finalized"
        assert finalized.finalized_note is not None
        assert finalized.exports["markdown"].startswith("# Final Clinical Note")
        assert "Dr. Demo" in store.export(record.encounter_id, "markdown")
        assert store.export(record.encounter_id, "json")["status"] == "finalized"

        print(json.dumps(finalized.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
