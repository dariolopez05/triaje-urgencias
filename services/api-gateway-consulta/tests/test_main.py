from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def patched(monkeypatch):
    import main

    db_mock = MagicMock()
    monkeypatch.setattr(main, "db", db_mock)
    return main, db_mock


def test_health(patched):
    main, _ = patched
    client = TestClient(main.app)
    assert client.get("/health").json() == {"status": "ok"}


def test_resultado_ok(patched):
    main, db_mock = patched
    db_mock.fetch_resultado_completo.return_value = {
        "guid_entrevista": "g1",
        "id_caso": "RES0051",
        "estado": "AUDITADO",
        "triage_real": "C2",
        "prediccion_ia": "C3",
        "validacion": "Under-triage",
    }
    client = TestClient(main.app)
    response = client.get("/resultado/g1")
    assert response.status_code == 200
    body = response.json()
    assert body["validacion"] == "Under-triage"
    db_mock.fetch_resultado_completo.assert_called_once_with("g1")


def test_resultado_not_found(patched):
    main, db_mock = patched
    db_mock.fetch_resultado_completo.return_value = None
    client = TestClient(main.app)
    response = client.get("/resultado/missing")
    assert response.status_code == 404
    assert "missing" in response.json()["detail"]
