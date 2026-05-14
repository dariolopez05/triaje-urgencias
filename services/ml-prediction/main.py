from __future__ import annotations

import io
import os
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.sparse import hstack

from triage_common import db, storage
from triage_common.contracts import (
    EntrevistaEstado,
    PredictRequest,
    PredictResponse,
    TaskLogEntry,
    TaskStatus,
    TriageLevel,
)


app = FastAPI(title="TriageIA ML Prediction", version="0.1.0")

_storage: Optional[storage.StorageClient] = None
_model_cache: dict = {"pipeline": None, "url": None}


def get_storage() -> storage.StorageClient:
    global _storage
    if _storage is None:
        _storage = storage.StorageClient()
    return _storage


def _latest_model_url() -> Optional[str]:
    candidates: list[str] = []
    for obj in get_storage().list_objects(storage.BUCKET_MODELOS):
        if obj.endswith(".joblib"):
            candidates.append(obj)
    if not candidates:
        return None
    candidates.sort()
    return f"s3://{storage.BUCKET_MODELOS}/{candidates[-1]}"


def _load_pipeline(url: str) -> dict:
    bucket, key = storage.parse_uri(url)
    data = get_storage().get_bytes(bucket, key)
    return joblib.load(io.BytesIO(data))


def _ensure_loaded(force_url: Optional[str] = None) -> dict:
    target_url = force_url or _latest_model_url()
    if target_url is None:
        raise HTTPException(status_code=503, detail="No hay modelo entrenado todavia")
    if _model_cache["pipeline"] is None or _model_cache["url"] != target_url:
        _model_cache["pipeline"] = _load_pipeline(target_url)
        _model_cache["url"] = target_url
    return _model_cache


def _predict(pipeline: dict, texto: str, entidades: list[str]) -> tuple[str, dict[str, float]]:
    tfidf = pipeline["tfidf"]
    mlb = pipeline["mlb"]
    estimator = pipeline["estimator"]
    text_matrix = tfidf.transform([texto or ""])
    ent_matrix = mlb.transform([list(entidades or [])])
    features = hstack([text_matrix, ent_matrix]).tocsr()
    probs: dict[str, float] = {}
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(features)[0]
        for cls, value in zip(estimator.classes_, proba):
            probs[str(cls)] = float(value)
    pred = str(estimator.predict(features)[0])
    return pred, probs


class ReloadResponse(BaseModel):
    url: Optional[str]
    loaded: bool


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_url": _model_cache["url"] or ""}


@app.post("/reload", response_model=ReloadResponse)
def reload() -> ReloadResponse:
    url = _latest_model_url()
    if url is None:
        _model_cache["pipeline"] = None
        _model_cache["url"] = None
        return ReloadResponse(url=None, loaded=False)
    _model_cache["pipeline"] = _load_pipeline(url)
    _model_cache["url"] = url
    return ReloadResponse(url=url, loaded=True)


@app.post("/run", response_model=PredictResponse)
def run(req: PredictRequest) -> PredictResponse:
    started = datetime.utcnow()
    pipeline = _ensure_loaded()["pipeline"]

    try:
        pred, probs = _predict(pipeline, req.texto, req.entidades_normalizadas)
        triage = TriageLevel(pred)
    except ValueError as exc:
        _log_error(req.guid, started, f"prediction parse error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        _log_error(req.guid, started, f"prediction failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    score_anx = 0.0
    with db.get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT score_ansiedad FROM Texto_Procesado WHERE guid = %s", (req.guid,))
        row = cur.fetchone()
        if row and row[0] is not None:
            score_anx = float(row[0])

    db.upsert_prediccion(
        req.guid,
        prediccion_ia=triage.value,
        score_ansiedad_ia=score_anx,
    )
    db.update_entrevista_estado(req.guid, EntrevistaEstado.PREDICHO)

    db.log_task(
        TaskLogEntry(
            guid=req.guid,
            service_name="ml-prediction",
            timestamp_inicio=started,
            timestamp_fin=datetime.utcnow(),
            status=TaskStatus.OK,
            payload_resultado={"prediccion_ia": triage.value, "probabilidades": probs},
        )
    )

    return PredictResponse(
        guid=req.guid,
        prediccion_ia=triage,
        score_ansiedad_ia=score_anx,
        probabilidades=probs,
    )


def _log_error(guid: str, started: datetime, msg: str) -> None:
    db.log_task(
        TaskLogEntry(
            guid=guid,
            service_name="ml-prediction",
            timestamp_inicio=started,
            timestamp_fin=datetime.utcnow(),
            status=TaskStatus.ERROR,
            error_msg=msg,
        )
    )
