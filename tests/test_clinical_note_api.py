"""API smoke test for transcript draft, finalize, and export."""

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVICE_DIR = ROOT / "code" / "service"
sys.path.insert(0, str(SERVICE_DIR))

from fastapi.testclient import TestClient

from backend.api import clinical_note
from backend.clinical.encounter_store import EncounterStore
from backend.main import app


def main():
    payload = json.loads(
        (ROOT / "examples" / "clinical_note_request.json").read_text(
            encoding="utf-8"
        )
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        clinical_note.encounter_store = EncounterStore(tmp_dir)
        client = TestClient(app)

        draft_response = client.post("/api/clinical_note/from_transcript", json=payload)
        assert draft_response.status_code == 200
        draft_body = draft_response.json()
        assert draft_body["success"] is True
        encounter_id = draft_body["encounter_id"]

        sections = [
            {
                **section,
                "status": "reviewed",
            }
            for section in draft_body["clinical_note"]["sections"]
            if section["status"] != "missing"
        ]
        finalize_response = client.post(
            "/api/clinical_note/finalize",
            json={
                "encounter_id": encounter_id,
                "reviewer": "Dr. API",
                "sections": sections,
            },
        )
        assert finalize_response.status_code == 200
        assert finalize_response.json()["encounter"]["status"] == "finalized"

        export_response = client.get(
            f"/api/clinical_note/encounters/{encounter_id}/export?format=markdown"
        )
        assert export_response.status_code == 200
        assert "# Final Clinical Note" in export_response.text
        assert "Dr. API" in export_response.text

        print(export_response.text)


if __name__ == "__main__":
    main()
