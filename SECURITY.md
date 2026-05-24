# Security Policy

Local Clinical Scribe is designed as a local-first documentation tool. Public
issues and pull requests must not include protected health information (PHI),
patient audio, speaker embeddings, access tokens, or private deployment
details.

## Supported Versions

Security fixes are handled on the latest public release branch.

## Reporting

If you find a security or privacy issue, do not open a public issue containing
sensitive details. Report it privately to the repository owner.

## Data Handling

The repository must remain free of:

- real patient audio or transcripts
- speaker embeddings or biometric identifiers
- local logs that may contain transcript text
- downloaded model artifacts
- `.env` files or access tokens
- private network addresses or internal deployment paths

Run the public safety scan before release:

```powershell
python scripts/public_safety_scan.py
```

