from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


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
    response = TestClient(main.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_run_returns_entidades(patched):
    main, db_mock, client_mock = patched
    client_mock.render_and_generate_json.return_value = {
        "entidades": ["me ahogo", "presion fuerte"]
    }
    response = TestClient(main.app).post(
        "/run", json={"guid": "g1", "texto": "Doctor, me ahogo y noto presion fuerte."}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["guid"] == "g1"
    assert body["entidades"] == ["me ahogo", "presion fuerte"]
    db_mock.upsert_texto_procesado.assert_called_once()
    args = db_mock.upsert_texto_procesado.call_args
    assert args.args[0] == "g1"
    assert args.args[1]["entidades_extraidas_es"] == ["me ahogo", "presion fuerte"]


def test_run_strips_and_filters_empties(patched):
    main, _, client_mock = patched
    client_mock.render_and_generate_json.return_value = {
        "entidades": ["  miedo  ", "", "  ", "fatiga"]
    }
    body = TestClient(main.app).post(
        "/run", json={"guid": "g2", "texto": "..."}
    ).json()
    assert body["entidades"] == ["miedo", "fatiga"]


def test_run_handles_alternative_key(patched):
    main, _, client_mock = patched
    client_mock.render_and_generate_json.return_value = {"entities": ["disnea"]}
    body = TestClient(main.app).post(
        "/run", json={"guid": "g3", "texto": "x"}
    ).json()
    assert body["entidades"] == ["disnea"]


def test_run_returns_502_on_llm_error(patched):
    main, db_mock, client_mock = patched
    from triage_common import llm

    client_mock.render_and_generate_json.side_effect = llm.LLMError("boom")
    response = TestClient(main.app).post(
        "/run", json={"guid": "g4", "texto": "x"}
    )
    assert response.status_code == 502
    log_args = db_mock.log_task.call_args
    assert log_args.args[0].status.value == "ERROR"
