from __future__ import annotations

import re
import unicodedata
from datetime import datetime

from fastapi import FastAPI

from triage_common import db
from triage_common.contracts import (
    EntrevistaEstado,
    EntrevistaTimestamps,
    PreprocessRequest,
    PreprocessResponse,
    TaskLogEntry,
    TaskStatus,
)


app = FastAPI(title="TriageIA Preprocessing", version="0.1.0")

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")


def preprocess_text(text: str) -> str:
    if not text:
        return ""
    cleaned = unicodedata.normalize("NFC", text)
    cleaned = _CONTROL_RE.sub(" ", cleaned)
    cleaned = _URL_RE.sub(" ", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run", response_model=PreprocessResponse)
def run(req: PreprocessRequest) -> PreprocessResponse:
    started = datetime.utcnow()
    db.mark_timestamp(req.guid, EntrevistaTimestamps.PREPROCESAMIENTO, "inicio", when=started)

    cleaned = preprocess_text(req.texto)

    db.upsert_texto_procesado(req.guid, {"texto_preprocesado": cleaned})

    finished = datetime.utcnow()
    db.mark_timestamp(req.guid, EntrevistaTimestamps.PREPROCESAMIENTO, "fin", when=finished)
    db.update_entrevista_estado(req.guid, EntrevistaEstado.TEXTO_PREPROCESADO)

    db.log_task(
        TaskLogEntry(
            guid=req.guid,
            service_name="preprocessing",
            timestamp_inicio=started,
            timestamp_fin=finished,
            status=TaskStatus.OK,
            payload_resultado={"chars_in": len(req.texto or ""), "chars_out": len(cleaned)},
        )
    )

    return PreprocessResponse(guid=req.guid, texto_preprocesado=cleaned)
