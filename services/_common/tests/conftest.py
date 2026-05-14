from __future__ import annotations

import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
DICTIONARY_PATH = REPO_ROOT / "data" / "dictionaries" / "manchester_terms.csv"
PROMPTS_PATH = REPO_ROOT / "data" / "prompts"


@pytest.fixture(autouse=True)
def _set_repo_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANCHESTER_DICTIONARY_PATH", str(DICTIONARY_PATH))
    monkeypatch.setenv("PROMPTS_DIR", str(PROMPTS_PATH))


@pytest.fixture
def dictionary_path() -> Path:
    return DICTIONARY_PATH


@pytest.fixture
def prompts_path() -> Path:
    return PROMPTS_PATH
