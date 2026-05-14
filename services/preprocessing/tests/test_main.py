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


class TestPreprocessText:
    def test_strips_control_chars(self):
        import main

        out = main.preprocess_text("hola\x00 mundo\x07")
        assert out == "hola mundo"

    def test_strips_urls(self):
        import main

        out = main.preprocess_text("Mira http://malo.com y www.otro.es de prueba.")
        assert "http://malo.com" not in out
        assert "www.otro.es" not in out
        assert "de prueba." in out

    def test_collapses_whitespace_and_newlines(self):
        import main

        out = main.preprocess_text("  hola\n\tmundo   ya  ")
        assert out == "hola mundo ya"

    def test_preserves_accents_after_nfc(self):
        import main

        out = main.preprocess_text("Perdió el sentido y se desplomó")
        assert "Perdió" in out
        assert "desplomó" in out

    def test_empty_input(self):
        import main

        assert main.preprocess_text("") == ""
        assert main.preprocess_text(None) == ""


class TestRunEndpoint:
    def test_health(self, patched):
        main, _ = patched
        client = TestClient(main.app)
        assert client.get("/health").json() == {"status": "ok"}

    def test_run_writes_cleaned_text(self, patched):
        main, db_mock = patched
        client = TestClient(main.app)
        response = client.post(
            "/run",
            json={"guid": "g1", "texto": "  Me   ahogo\nmucho\x00"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["guid"] == "g1"
        assert body["texto_preprocesado"] == "Me ahogo mucho"

        db_mock.mark_timestamp.assert_called()
        upsert_args = db_mock.upsert_texto_procesado.call_args
        assert upsert_args.args[0] == "g1"
        assert upsert_args.args[1]["texto_preprocesado"] == "Me ahogo mucho"
        db_mock.log_task.assert_called_once()

    def test_run_rejects_missing_fields(self, patched):
        main, _ = patched
        client = TestClient(main.app)
        response = client.post("/run", json={"guid": "g1"})
        assert response.status_code == 422
