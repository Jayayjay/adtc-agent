"""Miscellaneous small helpers that don't warrant their own module yet."""

from __future__ import annotations

import json
from pathlib import Path


def load_json(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str | Path, data: dict, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def truncate(text: str, max_chars: int = 200) -> str:
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "..."
