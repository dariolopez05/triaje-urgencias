from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException

from triage_common import db, llm
from triage_common.contracts import (
    EntrevistaEstado,
    EntrevistaTimestamps,
    LabelRequest,
    LabelResponse,
    TaskLogEntry,
    TaskStatus,
    TriageLevel,
)


app = FastAPI(title="TriageIA LLM Labeling", version="0.1.0")

_client: Optional[llm.LLMClient] = None
VALID_TRIAGE = {level.value for level in TriageLevel}


def get_client() -> llm.LLMClient:
    global _client
    if _client is None:
        _client = llm.LLMClient()
    return _client


def _parse_label(payload: dict) -> tuple[TriageLevel, str]:
    triage_raw = str(payload.get("triage", "")).strip().upper()
    justificacion = str(payload.get("justificacion", "")).strip()
    if triage_raw not in VALID_TRIAGE:
        raise llm.LLMInvalidJSON(f"triage invalido: {triage_raw!r}")
    return TriageLevel(triage_raw), justificacion


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run", response_model=LabelResponse)
def run(req: LabelRequest) -> LabelResponse:
    started = datetime.utcnow()
    db.mark_timestamp(req.guid, EntrevistaTimestamps.ETIQUETADO, "inicio", when=started)

    entidades = [e.termino_clinico for e in req.entidades_normalizadas]
    try:
        payload = get_client().render_and_generate_json(
            "label_triage.j2",
            context={
                "resumen_es": req.resumen_es,
                "entidades_json": json.dumps(entidades, ensure_ascii=False),
            },
        )
        triage, justificacion = _parse_label(payload)
    except llm.LLMError as exc:
        _log_error(req.guid, started, str(exc))
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}")

    db.upsert_texto_procesado(
        req.guid,
        {"triage_real": triage.value, "justificacion_llm": justificacion},
    )

    finished = datetime.utcnow()
    db.mark_timestamp(req.guid, EntrevistaTimestamps.ETIQUETADO, "fin", when=finished)
    db.update_entrevista_estado(req.guid, EntrevistaEstado.ETIQUETADO)

    db.log_task(
        TaskLogEntry(
            guid=req.guid,
            service_name="llm-labeling",
            timestamp_inicio=started,
            timestamp_fin=finished,
            status=TaskStatus.OK,
            payload_resultado={"triage": triage.value, "justificacion": justificacion},
        )
    )

    return LabelResponse(guid=req.guid, triage=triage, justificacion=justificacion)


def _log_error(guid: str, started: datetime, msg: str) -> None:
    db.log_task(
        TaskLogEntry(
            guid=guid,
            service_name="llm-labeling",
            timestamp_inicio=started,
            timestamp_fin=datetime.utcnow(),
            status=TaskStatus.ERROR,
            error_msg=msg,
        )
    )
