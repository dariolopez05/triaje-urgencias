from __future__ import annotations

import io
import os
import tempfile
from datetime import datetime
from typing import Optional

from faster_whisper import WhisperModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from triage_common import db, storage
from triage_common.contracts import (
    EntrevistaEstado,
    EntrevistaTimestamps,
    TaskLogEntry,
    TaskStatus,
    TranscribeRequest,
    TranscribeResponse,
)


app = FastAPI(title="TriageIA Transcripcion", version="0.1.0")

MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

_model: Optional[WhisperModel] = None
_storage: Optional[storage.StorageClient] = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
    return _model


def get_storage() -> storage.StorageClient:
    global _storage
    if _storage is None:
        _storage = storage.StorageClient()
    return _storage


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    compute_type: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model=MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )


@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe(req: TranscribeRequest) -> TranscribeResponse:
    started = datetime.utcnow()
    db.mark_timestamp(req.guid, EntrevistaTimestamps.TRANSCRIPCION, "inicio", when=started)

    try:
        bucket, key = storage.parse_uri(req.audio_url)
        data = get_storage().get_bytes(bucket, key)
    except Exception as exc:
        _log_error(req.guid, started, f"audio fetch failed: {exc}")
        raise HTTPException(status_code=400, detail=f"audio inaccesible: {exc}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        segments, info = get_model().transcribe(
            tmp_path,
            language=req.language,
            vad_filter=True,
        )
        pieces = [segment.text for segment in segments]
        texto = " ".join(piece.strip() for piece in pieces).strip()
        detected_language = info.language
        duration = float(info.duration or 0.0)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    fields = {"resumen_es": texto}
    if detected_language == "en":
        fields["texto_original_en"] = texto

    db.upsert_texto_procesado(req.guid, fields)

    finished = datetime.utcnow()
    db.mark_timestamp(req.guid, EntrevistaTimestamps.TRANSCRIPCION, "fin", when=finished)
    db.update_entrevista_estado(req.guid, EntrevistaEstado.TRANSCRITO)

    db.log_task(
        TaskLogEntry(
            guid=req.guid,
            service_name="transcripcion",
            timestamp_inicio=started,
            timestamp_fin=finished,
            status=TaskStatus.OK,
            payload_resultado={
                "language": detected_language,
                "duration_seconds": duration,
                "chars": len(texto),
            },
        )
    )

    return TranscribeResponse(
        guid=req.guid,
        texto=texto,
        language=detected_language,
        duration_seconds=duration,
    )


def _log_error(guid: str, started: datetime, msg: str) -> None:
    db.log_task(
        TaskLogEntry(
            guid=guid,
            service_name="transcripcion",
            timestamp_inicio=started,
            timestamp_fin=datetime.utcnow(),
            status=TaskStatus.ERROR,
            error_msg=msg,
        )
    )
