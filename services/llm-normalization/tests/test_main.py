from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
DICTIONARY_PATH = REPO_ROOT / "data" / "dictionaries" / "manchester_terms.csv"


@pytest.fixture(autouse=True)
def _set_dictionary(monkeypatch):
    monkeypatch.setenv("MANCHESTER_DICTIONARY_PATH", str(DICTIONARY_PATH))
    from triage_common import dictionary
    dictionary.reset_cache()


@pytest.fixture
def patched(monkeypatch):
    import main

    db_mock = MagicMock()
    monkeypatch.setattr(main, "db", db_mock)

    client_mock = MagicMock()
    monkeypatch.setattr(main, "get_client", lambda: client_mock)

    return main, db_mock, client_mock


def test_health(patched):
    main, _, _ = patched
    assert TestClient(main.app).get("/health").json() == {"status": "ok"}


def test_run_uses_exact_dictionary_first(patched):
    main, db_mock, client_mock = patched
    response = TestClient(main.app).post(
        "/run",
        json={
            "guid": "g1",
            "entidades_extraidas": ["me ahogo", "presion fuerte"],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    terms = [e["termino_clinico"] for e in body["entidades_normalizadas"]]
    assert "disnea" in terms
    assert "dolor_toracico_opresivo" in terms
    assert body["no_mapeadas"] == []
    client_mock.render_and_generate_json.assert_not_called()


def test_run_falls_back_to_llm_for_unmapped(patched):
    main, db_mock, client_mock = patched
    client_mock.render_and_generate_json.return_value = {
        "mapeos": [
            {"sintoma_original": "estertores", "termino_clinico": "sibilancias"},
            {"sintoma_original": "raro", "termino_clinico": "no_mapeado"},
        ]
    }
    response = TestClient(main.app).post(
        "/run",
        json={"guid": "g2", "entidades_extraidas": ["estertores", "raro"]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    entry = next(e for e in body["entidades_normalizadas"] if e["termino_clinico"] == "sibilancias")
    assert entry["prioridad_sugerida"] == "C2"
    assert entry["grupo_clinico"] == "RES"
    assert body["no_mapeadas"] == ["raro"]
    client_mock.render_and_generate_json.assert_called_once()


def test_run_rejects_invented_terms(patched):
    main, db_mock, client_mock = patched
    client_mock.render_and_generate_json.return_value = {
        "mapeos": [
            {"sintoma_original": "raro", "termino_clinico": "TERMINO_INVENTADO"},
        ]
    }
    response = TestClient(main.app).post(
        "/run", json={"guid": "g3", "entidades_extraidas": ["raro"]}
    )
    body = response.json()
    assert body["entidades_normalizadas"] == []
    assert body["no_mapeadas"] == ["raro"]


def test_run_persists_clinical_terms_only(patched):
    main, db_mock, client_mock = patched
    TestClient(main.app).post(
        "/run", json={"guid": "g4", "entidades_extraidas": ["me ahogo"]}
    )
    args = db_mock.upsert_texto_procesado.call_args
    assert args.args[0] == "g4"
    assert args.args[1]["entidades_normalizadas_es"] == ["disnea"]
