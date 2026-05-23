# Security And Privacy Notes

This repository is designed to keep sensitive data out of Git.

## Never Commit

- patient audio
- real clinical transcripts
- speaker embeddings
- logs containing PHI
- downloaded model artifacts
- access tokens or local environment files

## Runtime Data

Runtime data defaults to:

- `data/audio`
- `data/speaker_embedding`

Both are ignored by Git.

## Product Boundary

Generated notes are drafts. A licensed clinician must review and approve any
clinical documentation before it is used in care, billing, or official records.

## Deployment Notes

For a real deployment, add:

- user authentication
- role-based access control
- encryption at rest
- audit logging
- retention policy
- consent workflow
- backup and deletion workflow

