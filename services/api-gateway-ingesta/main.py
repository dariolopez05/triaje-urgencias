from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Annotated, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from triage_common import db, storage
from triage_common.contracts import (
    EntrevistaEstado,
    IngestaResponse,
    Origen,
    TaskLogEntry,
    TaskStatus,
)


app = FastAPI(title="TriageIA API Gateway Ingesta", version="0.1.0")

AIRFLOW_URL = os.getenv("AIRFLOW_BASE_URL", "")
AIRFLOW_USER = os.getenv("AIRFLOW_ADMIN_USER", "admin")
AIRFLOW_PASS = os.getenv("AIRFLOW_ADMIN_PASSWORD", "admin")
DAG_AUDIO = os.getenv("DAG_AUDIO_INGESTION", "dag_audio_ingestion")
DAG_TEXT = os.getenv("DAG_TEXT_INGESTION", "dag_text_ingestion")


def _trigger_dag(dag_id: str, guid: str) -> Optional[str]:
    if not AIRFLOW_URL:
        return None
    try:
        response = httpx.post(
            f"{AIRFLOW_URL.rstrip('/')}/api/v1/dags/{dag_id}/dagRuns",
            json={"conf": {"guid": guid}, "dag_run_id": f"triage-{guid}"},
            auth=(AIRFLOW_USER, AIRFLOW_PASS),
            timeout=5.0,
        )
        response.raise_for_status()
        return response.json().get("dag_run_id")
    except httpx.HTTPError:
        return None


def _parse_origen(raw: str) -> Origen:
    try:
        return Origen(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"origen invalido: {raw}") from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingesta", response_model=IngestaResponse)
async def ingesta(
    texto: Annotated[Optional[str], Form()] = None,
    audio: Annotated[Optional[UploadFile], File()] = None,
    id_caso: Annotated[Optional[str], Form()] = None,
    origen: Annotated[str, Form()] = Origen.MVP.value,
) -> IngestaResponse:
    if (texto is None) == (audio is None):
        raise HTTPException(
            status_code=400,
            detail="Proporciona exactamente uno de 'texto' o 'audio'.",
        )

    origen_enum = _parse_origen(origen)

    guid = str(uuid.uuid4())
    started = datetime.utcnow()

    storage_client = storage.StorageClient()
    url_audio = None
    url_texto = None

    if audio is not None:
        data = await audio.read()
        if not data:
            raise HTTPException(status_code=400, detail="audio vacio")
        url_audio = storage_client.put_bytes(
            storage.BUCKET_AUDIO_ORIGINAL,
            f"{guid}.wav",
            data,
            audio.content_type or "audio/wav",
        )
    else:
        if not (texto or "").strip():
            raise HTTPException(status_code=400, detail="texto vacio")
        url_texto = storage_client.put_bytes(
            storage.BUCKET_TEXTOS_ORIGINALES,
            f"{guid}.txt",
            texto.encode("utf-8"),
            "text/plain; charset=utf-8",
        )

    motor = "airflow" if AIRFLOW_URL else "manual"

    db.insert_entrevista(
        guid=guid,
        id_caso=id_caso,
        origen=origen_enum.value,
        url_audio_original=url_audio,
        url_texto_original=url_texto,
        motor_workflow=motor,
    )

    dag_id = DAG_AUDIO if audio is not None else DAG_TEXT
    workflow_id = _trigger_dag(dag_id, guid)

    db.log_task(
        TaskLogEntry(
            guid=guid,
            service_name="api-gateway-ingesta",
            timestamp_inicio=started,
            timestamp_fin=datetime.utcnow(),
            status=TaskStatus.OK,
            payload_resultado={
                "url_audio_original": url_audio,
                "url_texto_original": url_texto,
                "workflow_id": workflow_id,
                "dag_id": dag_id,
            },
        )
    )

    return IngestaResponse(
        guid=guid,
        estado=EntrevistaEstado.RECIBIDO.value,
        workflow_id=workflow_id,
    )
