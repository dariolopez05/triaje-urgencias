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


class TestLexicon:
    def test_known_term(self):
        import main

        assert main.lexicon_score("Tengo mucho miedo de morirme") >= 0.7

    def test_panic_phrase(self):
        import main

        assert main.lexicon_score("Me muero, no aguanto este dolor") >= 0.9

    def test_no_match(self):
        import main

        assert main.lexicon_score("Hola que tal") == 0.0


class TestParseLlmScore:
    def test_decimal_with_dot(self):
        import main

        assert main.parse_llm_score("0.85") == 0.85

    def test_decimal_with_comma(self):
        import main

        assert main.parse_llm_score("0,7") == 0.7

    def test_out_of_one_to_ten_scale(self):
        import main

        assert main.parse_llm_score("8") == 0.8

    def test_clamps_above_one(self):
        import main

        assert main.parse_llm_score("99") == 1.0

    def test_negative_clamped(self):
        import main

        assert main.parse_llm_score("-0.5") == 0.0

    def test_garbage(self):
        import main

        assert main.parse_llm_score("no idea") == 0.0


class TestCombine:
    def test_weighted_average(self):
        import main

        assert main.combine(1.0, 0.0) == 0.4
        assert main.combine(0.0, 1.0) == 0.6
        assert main.combine(0.5, 0.5) == 0.5


class TestRun:
    def test_health(self, patched):
        main, _, _ = patched
        assert TestClient(main.app).get("/health").json() == {"status": "ok"}

    def test_run_persists_score(self, patched):
        main, db_mock, client_mock = patched
        client_mock.generate.return_value = "0.8"
        response = TestClient(main.app).post(
            "/run",
            json={"guid": "g1", "texto": "Me muero, tengo panico."},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["guid"] == "g1"
        assert 0.0 <= body["score_ansiedad"] <= 1.0
        args = db_mock.upsert_texto_procesado.call_args
        assert args.args[0] == "g1"
        assert "score_ansiedad" in args.args[1]
