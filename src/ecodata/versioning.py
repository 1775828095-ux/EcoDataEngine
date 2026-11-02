"""Read the release version without machine-specific paths."""
from pathlib import Path


def version() -> str:
    path = Path(__file__).resolve().parents[2] / "VERSION"
    return path.read_text(encoding="utf-8").strip() if path.exists() else "unknown"
