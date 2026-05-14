from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from triage_common.contracts import AuditEthicsRequest, TriageLevel, Validacion


@pytest.fixture
def patched(monkeypatch):
    import main

    db_mock = MagicMock()
    monkeypatch.setattr(main, "db", db_mock)
    return main, db_mock


def test_health(patched):
    main, _ = patched
    assert TestClient(main.app).get("/health").json() == {"status": "ok"}


def test_acierto():
    import main

    req = AuditEthicsRequest(
        guid="g1",
        prediccion_ia=TriageLevel.C2,
        triage_real=TriageLevel.C2,
        score_ansiedad_ia=0.9,
    )
    result = main.evaluate(req)
    assert result.validacion == Validacion.ACIERTO
    assert result.sesgo_emocional_detectado is False


def test_under_triage_with_anxiety_bias():
    import main

    req = AuditEthicsRequest(
        guid="g1",
        prediccion_ia=TriageLevel.C3,
        triage_real=TriageLevel.C2,
        score_ansiedad_ia=0.95,
    )
    result = main.evaluate(req)
    assert result.validacion == Validacion.UNDER_TRIAGE
    assert result.sesgo_emocional_detectado is True
    assert "sesgo emocional" in result.motivo_fallo.lower()


def test_under_triage_without_bias():
    import main

    req = AuditEthicsRequest(
        guid="g1",
        prediccion_ia=TriageLevel.C3,
        triage_real=TriageLevel.C2,
        score_ansiedad_ia=0.2,
    )
    result = main.evaluate(req)
    assert result.validacion == Validacion.UNDER_TRIAGE
    assert result.sesgo_emocional_detectado is False
    assert "clinico" in result.motivo_fallo.lower()


def test_over_triage():
    import main

    req = AuditEthicsRequest(
        guid="g1",
        prediccion_ia=TriageLevel.C1,
        triage_real=TriageLevel.C3,
        score_ansiedad_ia=0.5,
    )
    result = main.evaluate(req)
    assert result.validacion == Validacion.OVER_TRIAGE
    assert result.sesgo_emocional_detectado is False


def test_run_persists(patched):
    main, db_mock = patched
    response = TestClient(main.app).post(
        "/run",
        json={
            "guid": "g1",
            "prediccion_ia": "C3",
            "triage_real": "C1",
            "score_ansiedad_ia": 0.9,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["sesgo_emocional_detectado"] is True
    args = db_mock.upsert_prediccion.call_args
    assert args.args[0] == "g1"
    assert args.kwargs["validacion"] == Validacion.UNDER_TRIAGE
