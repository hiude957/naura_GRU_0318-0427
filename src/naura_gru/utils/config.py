"""YAML config loading helpers."""

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML config file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

