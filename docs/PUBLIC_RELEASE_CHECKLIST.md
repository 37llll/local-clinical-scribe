# Public Release Checklist

Run this checklist before changing repository visibility or publishing a
release.

## Git Hygiene

- `git ls-remote --heads origin` shows only intended public branches.
- Local stale branches from private/internal history have been deleted.
- `git log --all` does not contain private project history.
- `git status --short --ignored` does not show tracked sensitive files.

## Data Hygiene

- No real patient audio is tracked.
- No speaker embeddings or biometric vectors are tracked.
- No local `data/encounters` records are tracked.
- No logs or temporary files are tracked.
- Demo records are synthetic and contain no real patient identifiers.

## Secret And Environment Hygiene

- No `.env` file is tracked.
- No GitHub, cloud, API, database, or SSH tokens are tracked.
- No private network addresses or internal filesystem paths are tracked.

## Product Boundary

- README states that generated notes are clinician-reviewed drafts.
- The product does not claim to diagnose, prescribe, or automate clinical
  decisions.
- Public docs explain how runtime patient data is stored and ignored.

## Required Commands

```powershell
python scripts/public_safety_scan.py
$cache = Join-Path $env:TEMP 'local_clinical_scribe_pycache_verify'
$env:PYTHONPYCACHEPREFIX = $cache
python -m compileall -q code/service tests
python tests/test_clinical_note_draft.py
python tests/test_encounter_store.py
python tests/test_clinical_note_api.py
```

