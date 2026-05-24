# Demo Data

The repository only includes synthetic demo transcript JSON under `examples/`.
It is intentionally minimal and does not represent a real patient, real visit,
or real clinician.

## Rules For Demo Contributions

- Use fictional names only, or avoid names entirely.
- Do not include dates of birth, national IDs, addresses, phone numbers,
  hospital numbers, admission numbers, insurance numbers, or medical record
  numbers.
- Do not include real audio.
- Do not include real speaker embeddings.
- Prefer short synthetic transcript snippets that exercise the product flow.

## Runtime Data

Any real or experimental runtime data should stay under ignored local paths:

- `data/audio`
- `data/speaker_embedding`
- `data/encounters`
- `logs`

