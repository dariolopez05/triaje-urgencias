from __future__ import annotations

import io
import uuid
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from triage_common import db, storage
from triage_common.contracts import (
    EntrevistaEstado,
    TaskLogEntry,
    TaskStatus,
)


app = FastAPI(title="TriageIA Dataset Builder", version="0.1.0")

_storage: Optional[storage.StorageClient] = None


def get_storage() -> storage.StorageClient:
    global _storage
    if _storage is None:
        _storage = storage.StorageClient()
    return _storage


class BuildRequest(BaseModel):
    min_rows: int = 1
    only_origin: Optional[str] = None


class BuildResponse(BaseModel):
    batch_id: str
    url: str
    rows: int
    triage_distribution: dict[str, int]


def _fetch_dataset(only_origin: Optional[str]) -> list[dict]:
    sql = """
        SELECT e.GUID_Entrevista AS guid,
               e.ID_CASO AS id_caso,
               e.Origen AS origen,
               e.Grupo_Clinico AS grupo_clinico,
               t.texto_original_en,
               t.resumen_es,
               t.texto_preprocesado,
               t.entidades_extraidas_es,
               t.entidades_normalizadas_es,
               t.triage_real,
               t.score_ansiedad
        FROM Entrevista e
        JOIN Texto_Procesado t ON t.guid = e.GUID_Entrevista
        WHERE t.triage_real IS NOT NULL
    """
    params: list = []
    if only_origin:
        sql += " AND e.Origen = %s"
        params.append(only_origin)
    sql += " ORDER BY e.Inicio_Solicitud"

    with db.get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [c.name for c in cur.description]
        rows = [dict(zip(columns, row)) for row in cur.fetchall()]
    return rows


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run", response_model=BuildResponse)
def run(req: BuildRequest) -> BuildResponse:
    started = datetime.utcnow()
    rows = _fetch_dataset(req.only_origin)

    if len(rows) < req.min_rows:
        _log(
            None,
            started,
            TaskStatus.ERROR,
            error=f"dataset insuficiente: {len(rows)} < {req.min_rows}",
        )
        raise HTTPException(
            status_code=400,
            detail=f"Dataset insuficiente ({len(rows)} filas, min {req.min_rows})",
        )

    df = pd.DataFrame(rows)
    if "entidades_extraidas_es" in df.columns:
        df["entidades_extraidas_es"] = df["entidades_extraidas_es"].apply(_as_list)
    if "entidades_normalizadas_es" in df.columns:
        df["entidades_normalizadas_es"] = df["entidades_normalizadas_es"].apply(_as_list)

    batch_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    object_name = f"{batch_id}.parquet"

    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    data = buf.getvalue()

    url = get_storage().put_bytes(
        storage.BUCKET_DATASETS, object_name, data, "application/octet-stream"
    )

    for guid in df["guid"].tolist():
        db.update_entrevista_estado(guid, EntrevistaEstado.DATASET_GENERADO)
        with db.get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE Entrevista SET URL_Dataset_Generado = %s WHERE GUID_Entrevista = %s",
                (url, guid),
            )

    distribution = (
        df["triage_real"].value_counts().to_dict() if "triage_real" in df.columns else {}
    )
    distribution = {str(k): int(v) for k, v in distribution.items()}

    _log(
        None,
        started,
        TaskStatus.OK,
        payload={"batch_id": batch_id, "url": url, "rows": len(df), "distribution": distribution},
    )

    return BuildResponse(
        batch_id=batch_id,
        url=url,
        rows=len(df),
        triage_distribution=distribution,
    )


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        import json
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (TypeError, ValueError):
            return []
    return []


def _log(
    guid: Optional[str],
    started: datetime,
    status: TaskStatus,
    payload: Optional[dict] = None,
    error: Optional[str] = None,
) -> None:
    db.log_task(
        TaskLogEntry(
            guid=guid,
            service_name="dataset-builder",
            timestamp_inicio=started,
            timestamp_fin=datetime.utcnow(),
            status=status,
            payload_resultado=payload,
            error_msg=error,
        )
    )
