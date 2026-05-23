"""API package.

Routers are imported explicitly by `backend.main`. Keep this package initializer
light so tests and non-ASR utilities can import individual API modules without
pulling model dependencies such as FunASR.
"""

__all__ = []

