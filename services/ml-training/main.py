from __future__ import annotations

import io
import json
from datetime import datetime
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import pipeline as ml

from triage_common import db, storage
from triage_common.contracts import (
    EntrevistaEstado,
    EntrevistaTimestamps,
    TaskLogEntry,
    TaskStatus,
)


app = FastAPI(title="TriageIA ML Training", version="0.1.0")

_storage: Optional[storage.StorageClient] = None


def get_storage() -> storage.StorageClient:
    global _storage
    if _storage is None:
        _storage = storage.StorageClient()
    return _storage


class TrainRequest(BaseModel):
    dataset_url: str
    run_id: Optional[str] = None


class TrainResponse(BaseModel):
    run_id: str
    model_url: str
    metrics_url: str
    selected_model: str
    f1_macro: float
    recall_c1: float
    recall_c2: float


def _load_dataset(url: str) -> pd.DataFrame:
    bucket, key = storage.parse_uri(url)
    data = get_storage().get_bytes(bucket, key)
    return pd.read_parquet(io.BytesIO(data), engine="pyarrow")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run", response_model=TrainResponse)
def run(req: TrainRequest) -> TrainResponse:
    started = datetime.utcnow()
    run_id = req.run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%S")

    try:
        df = _load_dataset(req.dataset_url)
        artifacts = ml.train_best(df)
    except Exception as exc:
        _log_error(started, f"training failed: {exc}")
        raise HTTPException(status_code=400, detail=f"training failed: {exc}")

    model_buffer = io.BytesIO()
    joblib.dump(artifacts.pipeline, model_buffer)
    model_buffer.seek(0)
    model_bytes = model_buffer.getvalue()

    model_object = f"{run_id}.joblib"
    metrics_object = f"{run_id}-metrics.json"

    model_url = get_storage().put_bytes(
        storage.BUCKET_MODELOS, model_object, model_bytes, "application/octet-stream"
    )
    metrics_payload = {
        "run_id": run_id,
        "trained_at": datetime.utcnow().isoformat(),
        "dataset_url": req.dataset_url,
        "rows": len(df),
        "selected_model": artifacts.estimator_name,
        "metrics": artifacts.metrics,
    }
    metrics_url = get_storage().put_bytes(
        storage.BUCKET_MODELOS,
        metrics_object,
        json.dumps(metrics_payload, indent=2).encode("utf-8"),
        "application/json",
    )

    best = artifacts.metrics["best"]
    recall_c1 = float(best["recall_per_class"].get("C1", 0.0))
    recall_c2 = float(best["recall_per_class"].get("C2", 0.0))
    f1_macro = float(best["f1_macro"])

    if "guid" in df.columns:
        for guid in df["guid"].tolist():
            with db.get_connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE Entrevista SET URL_Modelo_Entrenado = %s, "
                    "Fin_Entrenamiento = NOW() WHERE GUID_Entrevista = %s",
                    (model_url, guid),
                )
            db.update_entrevista_estado(guid, EntrevistaEstado.MODELO_ENTRENADO)

    db.log_task(
        TaskLogEntry(
            guid=None,
            service_name="ml-training",
            timestamp_inicio=started,
            timestamp_fin=datetime.utcnow(),
            status=TaskStatus.OK,
            payload_resultado={
                "run_id": run_id,
                "model_url": model_url,
                "metrics_url": metrics_url,
                "selected_model": artifacts.estimator_name,
                "rows": len(df),
                "f1_macro": f1_macro,
                "recall_c1": recall_c1,
                "recall_c2": recall_c2,
            },
        )
    )

    return TrainResponse(
        run_id=run_id,
        model_url=model_url,
        metrics_url=metrics_url,
        selected_model=artifacts.estimator_name,
        f1_macro=f1_macro,
        recall_c1=recall_c1,
        recall_c2=recall_c2,
    )


def _log_error(started: datetime, msg: str) -> None:
    db.log_task(
        TaskLogEntry(
            guid=None,
            service_name="ml-training",
            timestamp_inicio=started,
            timestamp_fin=datetime.utcnow(),
            status=TaskStatus.ERROR,
            error_msg=msg,
        )
    )
