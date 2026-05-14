from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


VALID_ENTITY = {
    "termino_clinico": "dolor_toracico_opresivo",
    "prioridad_sugerida": "C2",
    "grupo_clinico": "CAR",
    "sintoma_original": "presion fuerte",
}


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


def test_run_returns_triage_c2(patched):
    main, db_mock, client_mock = patched
    client_mock.render_and_generate_json.return_value = {
        "triage": "C2",
        "justificacion": "Dolor toracico opresivo: Manchester C2.",
    }
    response = TestClient(main.app).post(
        "/run",
        json={
            "guid": "g1",
            "resumen_es": "Paciente con dolor toracico opresivo.",
            "entidades_normalizadas": [VALID_ENTITY],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["triage"] == "C2"
    assert "Manchester C2" in body["justificacion"]

    args = db_mock.upsert_texto_procesado.call_args
    assert args.args[0] == "g1"
    assert args.args[1]["triage_real"] == "C2"


def test_run_rejects_invalid_triage(patched):
    main, db_mock, client_mock = patched
    client_mock.render_and_generate_json.return_value = {
        "triage": "X9",
        "justificacion": "?",
    }
    response = TestClient(main.app).post(
        "/run",
        json={
            "guid": "g2",
            "resumen_es": "x",
            "entidades_normalizadas": [VALID_ENTITY],
        },
    )
    assert response.status_code == 502
    log_args = db_mock.log_task.call_args
    assert log_args.args[0].status.value == "ERROR"


def test_run_uppercases_lowercase_triage(patched):
    main, _, client_mock = patched
    client_mock.render_and_generate_json.return_value = {
        "triage": "c4",
        "justificacion": "leve",
    }
    response = TestClient(main.app).post(
        "/run",
        json={
            "guid": "g3",
            "resumen_es": "tobillo hinchado",
            "entidades_normalizadas": [VALID_ENTITY],
        },
    )
    assert response.status_code == 200
    assert response.json()["triage"] == "C4"
