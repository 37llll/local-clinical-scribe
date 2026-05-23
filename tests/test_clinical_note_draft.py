"""Smoke test for the v0.1 clinical note draft generator."""

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVICE_DIR = ROOT / "code" / "service"
sys.path.insert(0, str(SERVICE_DIR))

from backend.clinical.draft_generator import ClinicalNoteDraftGenerator
from backend.clinical.schemas import ClinicalNoteRequest


def main():
    payload_path = ROOT / "examples" / "clinical_note_request.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    request = ClinicalNoteRequest(**payload)
    draft = ClinicalNoteDraftGenerator().generate(request)

    assert draft.sections
    assert any(section.evidence for section in draft.sections)
    assert draft.status == "draft_requires_clinician_review"

    print(json.dumps(draft.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
