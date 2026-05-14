from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from triage_common import db
from triage_common.contracts import (
    EntrevistaEstado,
    EntrevistaTimestamps,
    TaskLogEntry,
    TaskStatus,
    Validacion,
)


@pytest.fixture
def mocked_connection(monkeypatch):
    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.__exit__.return_value = False

    connection = MagicMock()
    connection.__enter__ = MagicMock(return_value=connection)
    connection.__exit__ = MagicMock(return_value=False)
    connection.cursor.return_value = cursor
    connection.commit = MagicMock()
    connection.rollback = MagicMock()
    connection.close = MagicMock()

    def fake_connect(**_kwargs):
        return connection

    monkeypatch.setattr(db.psycopg2, "connect", fake_connect)
    return connection, cursor


def _executed_sql(cursor: MagicMock) -> str:
    return cursor.execute.call_args.args[0]


def _executed_params(cursor: MagicMock):
    args = cursor.execute.call_args.args
    return args[1] if len(args) > 1 else None


class TestInsertEntrevista:
    def test_inserts_with_state_recibido(self, mocked_connection):
        _, cursor = mocked_connection
        db.insert_entrevista(
            guid="g1",
            id_caso="RES0051",
            origen="Dataset",
            url_audio_original="s3://audio-original/g1.wav",
        )
        sql = _executed_sql(cursor)
        params = _executed_params(cursor)
        assert "INSERT INTO Entrevista" in sql
        assert params[0] == "g1"
        assert params[1] == "RES0051"
        assert params[-1] == EntrevistaEstado.RECIBIDO.value


class TestUpdateEntrevistaEstado:
    def test_accepts_enum(self, mocked_connection):
        _, cursor = mocked_connection
        db.update_entrevista_estado("g1", EntrevistaEstado.TEXTO_ENRIQUECIDO)
        params = _executed_params(cursor)
        assert params == (EntrevistaEstado.TEXTO_ENRIQUECIDO.value, "g1")

    def test_accepts_string(self, mocked_connection):
        _, cursor = mocked_connection
        db.update_entrevista_estado("g1", "ERROR")
        params = _executed_params(cursor)
        assert params == ("ERROR", "g1")


class TestMarkTimestamp:
    def test_sets_inicio_column(self, mocked_connection):
        _, cursor = mocked_connection
        when = datetime(2026, 5, 14, 10, 0, 0)
        db.mark_timestamp("g1", EntrevistaTimestamps.TRANSCRIPCION, "inicio", when=when)
        sql = _executed_sql(cursor)
        params = _executed_params(cursor)
        assert "Inicio_Transcripcion" in sql
        assert params == (when, "g1")

    def test_sets_fin_column(self, mocked_connection):
        _, cursor = mocked_connection
        db.mark_timestamp("g1", EntrevistaTimestamps.ETIQUETADO, "fin")
        sql = _executed_sql(cursor)
        assert "Fin_Etiquetado" in sql

    def test_rejects_invalid_moment(self, mocked_connection):
        with pytest.raises(ValueError):
            db.mark_timestamp("g1", EntrevistaTimestamps.SOLICITUD, "middle")


class TestUpsertTextoProcesado:
    def test_wraps_jsonb_columns(self, mocked_connection):
        _, cursor = mocked_connection
        db.upsert_texto_procesado(
            "g1",
            {
                "resumen_es": "paciente con dolor",
                "entidades_extraidas_es": ["dolor"],
                "entidades_normalizadas_es": ["dolor_toracico_opresivo"],
                "triage_real": "C2",
            },
        )
        sql = _executed_sql(cursor)
        params = _executed_params(cursor)
        assert "INSERT INTO Texto_Procesado" in sql
        assert "ON CONFLICT (guid) DO UPDATE" in sql
        assert params[0] == "g1"
        json_params = [p for p in params if hasattr(p, "adapted")]
        assert len(json_params) == 2

    def test_noop_on_empty(self, mocked_connection):
        _, cursor = mocked_connection
        db.upsert_texto_procesado("g1", {})
        cursor.execute.assert_not_called()


class TestUpsertPrediccion:
    def test_passes_validacion_value(self, mocked_connection):
        _, cursor = mocked_connection
        db.upsert_prediccion(
            "g1",
            prediccion_ia="C2",
            score_ansiedad_ia=0.7,
            validacion=Validacion.UNDER_TRIAGE,
            motivo_fallo="sesgo emocional",
        )
        params = _executed_params(cursor)
        assert params[0] == "g1"
        assert params[1] == "C2"
        assert params[2] == 0.7
        assert params[3] == Validacion.UNDER_TRIAGE.value


class TestLogTask:
    def test_serializes_payload(self, mocked_connection):
        _, cursor = mocked_connection
        entry = TaskLogEntry(
            guid="g1",
            service_name="llm-extraction",
            timestamp_inicio=datetime(2026, 5, 14, 10, 0),
            timestamp_fin=datetime(2026, 5, 14, 10, 0, 5),
            status=TaskStatus.OK,
            payload_resultado={"entidades": ["disnea"]},
        )
        db.log_task(entry)
        sql = _executed_sql(cursor)
        params = _executed_params(cursor)
        assert "INSERT INTO Task_Log" in sql
        assert params[0] == "g1"
        assert params[1] == "llm-extraction"
        assert params[4] == "OK"


class TestFetchResultadoCompleto:
    def test_returns_none_when_missing(self, mocked_connection):
        _, cursor = mocked_connection
        cursor.fetchone.return_value = None
        cursor.description = []
        assert db.fetch_resultado_completo("missing") is None

    def test_returns_record_when_present(self, mocked_connection):
        _, cursor = mocked_connection
        cursor.fetchone.return_value = ("g1", "RES0051", "Dataset", "RES", "AUDITADO")
        cursor.description = [
            MagicMock(name="guid_entrevista"),
            MagicMock(name="id_caso"),
            MagicMock(name="origen"),
            MagicMock(name="grupo_clinico"),
            MagicMock(name="estado"),
        ]
        for descriptor, value in zip(
            cursor.description,
            ["guid_entrevista", "id_caso", "origen", "grupo_clinico", "estado"],
        ):
            descriptor.name = value
        result = db.fetch_resultado_completo("g1")
        assert result is not None
        assert result["guid_entrevista"] == "g1"
        assert result["estado"] == "AUDITADO"


class TestRollbackOnError:
    def test_rolls_back_on_exception(self, mocked_connection):
        connection, cursor = mocked_connection
        cursor.execute.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            db.update_entrevista_estado("g1", EntrevistaEstado.ERROR)
        connection.rollback.assert_called_once()
        connection.close.assert_called_once()
