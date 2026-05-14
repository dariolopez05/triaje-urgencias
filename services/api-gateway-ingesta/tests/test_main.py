from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def patched(monkeypatch):
    import main

    storage_client = MagicMock()
    storage_client.put_bytes.side_effect = (
        lambda bucket, name, data, ct: f"s3://{bucket}/{name}"
    )
    monkeypatch.setattr(main.storage, "StorageClient", lambda: storage_client)
    monkeypatch.setattr(main.storage, "BUCKET_AUDIO_ORIGINAL", "audio-original")
    monkeypatch.setattr(main.storage, "BUCKET_TEXTOS_ORIGINALES", "textos-originales")

    db_mock = MagicMock()
    monkeypatch.setattr(main, "db", db_mock)

    monkeypatch.setattr(main, "AIRFLOW_URL", "")

    return main, db_mock, storage_client


def test_health(patched):
    main, _, _ = patched
    client = TestClient(main.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingesta_texto_ok(patched):
    main, db_mock, storage_client = patched
    client = TestClient(main.app)

    response = client.post(
        "/ingesta",
        data={"texto": "Tengo dolor de pecho desde hace 20 minutos", "origen": "MVP"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["estado"] == "RECIBIDO"
    assert body["guid"]
    storage_client.put_bytes.assert_called_once()
    bucket_arg = storage_client.put_bytes.call_args.args[0]
    assert bucket_arg == "textos-originales"
    db_mock.insert_entrevista.assert_called_once()
    db_mock.log_task.assert_called_once()


def test_ingesta_audio_ok(patched):
    main, db_mock, storage_client = patched
    client = TestClient(main.app)

    response = client.post(
        "/ingesta",
        data={"origen": "Dataset", "id_caso": "RES0051", "grupo_clinico": "RES"},
        files={"audio": ("clip.wav", b"FAKE_WAV_BYTES", "audio/wav")},
    )

    assert response.status_code == 200, response.text
    storage_client.put_bytes.assert_called_once()
    assert storage_client.put_bytes.call_args.args[0] == "audio-original"
    kwargs = db_mock.insert_entrevista.call_args.kwargs
    assert kwargs["id_caso"] == "RES0051"
    assert kwargs["origen"] == "Dataset"
    assert kwargs["grupo_clinico"] == "RES"
    assert kwargs["url_audio_original"].startswith("s3://audio-original/")
    assert kwargs["url_texto_original"] is None


def test_ingesta_requires_one_input(patched):
    main, _, _ = patched
    client = TestClient(main.app)

    response = client.post("/ingesta", data={"origen": "MVP"})
    assert response.status_code == 400


def test_ingesta_rejects_both_inputs(patched):
    main, _, _ = patched
    client = TestClient(main.app)

    response = client.post(
        "/ingesta",
        data={"texto": "hola"},
        files={"audio": ("c.wav", b"x", "audio/wav")},
    )
    assert response.status_code == 400


def test_ingesta_rejects_empty_text(patched):
    main, _, _ = patched
    client = TestClient(main.app)

    response = client.post("/ingesta", data={"texto": "   "})
    assert response.status_code == 400


def test_ingesta_rejects_invalid_origen(patched):
    main, _, _ = patched
    client = TestClient(main.app)

    response = client.post("/ingesta", data={"texto": "x", "origen": "ZZZ"})
    assert response.status_code == 400
    assert "origen invalido" in response.json()["detail"]


def test_ingesta_rejects_invalid_grupo(patched):
    main, _, _ = patched
    client = TestClient(main.app)

    response = client.post("/ingesta", data={"texto": "x", "grupo_clinico": "XYZ"})
    assert response.status_code == 400


def test_trigger_dag_disabled_when_no_url(patched, monkeypatch):
    main, _, _ = patched
    monkeypatch.setattr(main, "AIRFLOW_URL", "")
    assert main._trigger_dag("dag_text_ingestion", "g1") is None


def test_trigger_dag_returns_run_id(patched, monkeypatch):
    main, _, _ = patched
    monkeypatch.setattr(main, "AIRFLOW_URL", "http://airflow:8080")
    fake_response = MagicMock()
    fake_response.json.return_value = {"dag_run_id": "triage-g1"}
    fake_response.raise_for_status = MagicMock()
    monkeypatch.setattr(main.httpx, "post", lambda *a, **k: fake_response)
    assert main._trigger_dag("dag_text_ingestion", "g1") == "triage-g1"
