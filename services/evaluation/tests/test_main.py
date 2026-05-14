from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from triage_common.contracts import TriageLevel, Validacion


@pytest.fixture
def patched(monkeypatch):
    import main

    db_mock = MagicMock()
    monkeypatch.setattr(main, "db", db_mock)
    return main, db_mock


def test_health(patched):
    main, _ = patched
    assert TestClient(main.app).get("/health").json() == {"status": "ok"}


def test_classify_acierto():
    import main

    assert main.classify(TriageLevel.C2, TriageLevel.C2) == Validacion.ACIERTO


def test_classify_under_triage():
    import main

    assert main.classify(TriageLevel.C3, TriageLevel.C1) == Validacion.UNDER_TRIAGE


def test_classify_over_triage():
    import main

    assert main.classify(TriageLevel.C1, TriageLevel.C3) == Validacion.OVER_TRIAGE


def test_run_persists(patched):
    main, db_mock = patched
    response = TestClient(main.app).post(
        "/run", json={"guid": "g1", "prediccion_ia": "C3", "triage_real": "C1"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["validacion"] == "Under-triage"
    args = db_mock.upsert_prediccion.call_args
    assert args.args[0] == "g1"
    assert args.kwargs["validacion"] == Validacion.UNDER_TRIAGE
    db_mock.update_entrevista_estado.assert_called_once()
