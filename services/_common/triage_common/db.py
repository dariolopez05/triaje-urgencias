from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterator, Optional

import psycopg2
from psycopg2.extensions import connection as PgConnection
from psycopg2.extras import Json

from triage_common.contracts import (
    EntrevistaEstado,
    EntrevistaTimestamps,
    TaskLogEntry,
    TaskStatus,
    Validacion,
)


@dataclass(frozen=True)
class DbConfig:
    host: str
    port: int
    user: str
    password: str
    database: str

    @classmethod
    def from_env(cls) -> "DbConfig":
        return cls(
            host=os.getenv("POSTGRES_HOST", "postgres"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "triage"),
            password=os.getenv("POSTGRES_PASSWORD", "triage_pw"),
            database=os.getenv("POSTGRES_DB", "triage_db"),
        )


@contextmanager
def get_connection(config: DbConfig | None = None) -> Iterator[PgConnection]:
    cfg = config or DbConfig.from_env()
    conn = psycopg2.connect(
        host=cfg.host,
        port=cfg.port,
        user=cfg.user,
        password=cfg.password,
        dbname=cfg.database,
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def insert_entrevista(
    guid: str,
    id_caso: Optional[str] = None,
    origen: Optional[str] = None,
    grupo_clinico: Optional[str] = None,
    url_audio_original: Optional[str] = None,
    url_texto_original: Optional[str] = None,
    motor_workflow: Optional[str] = None,
    config: DbConfig | None = None,
) -> None:
    with get_connection(config) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO Entrevista (
                GUID_Entrevista, ID_CASO, Origen, Grupo_Clinico,
                URL_Audio_Original, URL_Texto_Original,
                Inicio_Solicitud, Motor_Workflow, Estado
            ) VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s, %s)
            """,
            (
                guid,
                id_caso,
                origen,
                grupo_clinico,
                url_audio_original,
                url_texto_original,
                motor_workflow,
                EntrevistaEstado.RECIBIDO.value,
            ),
        )


def update_entrevista_estado(
    guid: str,
    estado: EntrevistaEstado | str,
    config: DbConfig | None = None,
) -> None:
    value = estado.value if isinstance(estado, EntrevistaEstado) else estado
    with get_connection(config) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE Entrevista SET Estado = %s WHERE GUID_Entrevista = %s",
            (value, guid),
        )


def mark_timestamp(
    guid: str,
    stage: EntrevistaTimestamps,
    moment: str,
    when: Optional[datetime] = None,
    config: DbConfig | None = None,
) -> None:
    if moment not in ("inicio", "fin"):
        raise ValueError("moment must be 'inicio' or 'fin'")
    column = stage.inicio_column if moment == "inicio" else stage.fin_column
    value = when or datetime.utcnow()
    with get_connection(config) as conn, conn.cursor() as cur:
        cur.execute(
            f"UPDATE Entrevista SET {column} = %s WHERE GUID_Entrevista = %s",
            (value, guid),
        )


def upsert_texto_procesado(
    guid: str,
    fields: dict[str, Any],
    config: DbConfig | None = None,
) -> None:
    if not fields:
        return
    json_fields = {"entidades_extraidas_es", "entidades_normalizadas_es"}
    columns = list(fields.keys())
    values = [Json(fields[c]) if c in json_fields else fields[c] for c in columns]
    placeholders = ", ".join(["%s"] * (len(columns) + 1))
    column_list = ", ".join(["guid", *columns])
    update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns)
    sql = (
        f"INSERT INTO Texto_Procesado ({column_list}) VALUES ({placeholders}) "
        f"ON CONFLICT (guid) DO UPDATE SET {update_clause}"
    )
    with get_connection(config) as conn, conn.cursor() as cur:
        cur.execute(sql, [guid, *values])


def upsert_prediccion(
    guid: str,
    prediccion_ia: Optional[str] = None,
    score_ansiedad_ia: Optional[float] = None,
    validacion: Optional[Validacion | str] = None,
    motivo_fallo: Optional[str] = None,
    accion_correctiva: Optional[str] = None,
    config: DbConfig | None = None,
) -> None:
    validacion_value = (
        validacion.value if isinstance(validacion, Validacion) else validacion
    )
    with get_connection(config) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO Prediccion (guid, prediccion_ia, score_ansiedad_ia,
                                    validacion, motivo_fallo, accion_correctiva)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (guid) DO UPDATE SET
                prediccion_ia = COALESCE(EXCLUDED.prediccion_ia, Prediccion.prediccion_ia),
                score_ansiedad_ia = COALESCE(EXCLUDED.score_ansiedad_ia, Prediccion.score_ansiedad_ia),
                validacion = COALESCE(EXCLUDED.validacion, Prediccion.validacion),
                motivo_fallo = COALESCE(EXCLUDED.motivo_fallo, Prediccion.motivo_fallo),
                accion_correctiva = COALESCE(EXCLUDED.accion_correctiva, Prediccion.accion_correctiva),
                fecha = NOW()
            """,
            (guid, prediccion_ia, score_ansiedad_ia, validacion_value, motivo_fallo, accion_correctiva),
        )


def log_task(entry: TaskLogEntry, config: DbConfig | None = None) -> None:
    payload = Json(entry.payload_resultado) if entry.payload_resultado is not None else None
    status_value = entry.status.value if isinstance(entry.status, TaskStatus) else entry.status
    with get_connection(config) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO Task_Log (guid, service_name, timestamp_inicio, timestamp_fin,
                                  status, payload_resultado, error_msg)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                entry.guid,
                entry.service_name,
                entry.timestamp_inicio,
                entry.timestamp_fin,
                status_value,
                payload,
                entry.error_msg,
            ),
        )


def fetch_resultado_completo(
    guid: str, config: DbConfig | None = None
) -> Optional[dict[str, Any]]:
    with get_connection(config) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM v_resultado_completo WHERE GUID_Entrevista = %s",
            (guid,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        columns = [c.name for c in cur.description]
        record = dict(zip(columns, row))
    return _normalize_jsonb(record)


def _normalize_jsonb(record: dict[str, Any]) -> dict[str, Any]:
    for key, value in record.items():
        if isinstance(value, str) and key.startswith("entidades"):
            try:
                record[key] = json.loads(value)
            except (TypeError, ValueError):
                pass
    return record
