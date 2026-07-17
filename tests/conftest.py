"""Shared pytest fixtures. Path setup so `src` imports work regardless of
where pytest is invoked from."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def sandbox_tmp_path(tmp_path):
    """A temp directory for FilesystemTool tests -- avoids touching real files."""
    d = tmp_path / "sandbox"
    d.mkdir()
    return d


@pytest.fixture
def sample_tasks():
    import json
    fixtures_dir = Path(__file__).parent / "fixtures"
    with open(fixtures_dir / "test_prompts.json") as f:
        return json.load(f)
