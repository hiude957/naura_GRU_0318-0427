"""Run-directory and logging helpers."""

from datetime import datetime
from pathlib import Path


def make_run_id(name: str) -> str:
    """Create a stable timestamped run id."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = name.strip().replace(" ", "-")
    return f"{stamp}_{safe_name}"


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out

