from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException

from triage_common import db, llm
from triage_common.contracts import (
    EntrevistaEstado,
    EntrevistaTimestamps,
    ExtractRequest,
    ExtractResponse,
    TaskLogEntry,
    TaskStatus,
)


app = FastAPI(title="TriageIA LLM Extraction", version="0.1.0")

_client: Optional[llm.LLMClient] = None


def get_client() -> llm.LLMClient:
    global _client
    if _client is None:
        _client = llm.LLMClient()
    return _client


def _normalize_entities(payload: dict) -> list[str]:
    raw = payload.get("entidades") or payload.get("entities") or []
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run", response_model=ExtractResponse)
def run(req: ExtractRequest) -> ExtractResponse:
    started = datetime.utcnow()
    db.mark_timestamp(req.guid, EntrevistaTimestamps.EXTRACCION_ENTIDADES, "inicio", when=started)

    try:
        payload = get_client().render_and_generate_json(
            "extract_entities.j2",
            context={"texto": req.texto},
        )
    except llm.LLMError as exc:
        _log_error(req.guid, started, str(exc))
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}")

    entidades = _normalize_entities(payload)

    db.upsert_texto_procesado(req.guid, {"entidades_extraidas_es": entidades})

    finished = datetime.utcnow()
    db.mark_timestamp(req.guid, EntrevistaTimestamps.EXTRACCION_ENTIDADES, "fin", when=finished)
    db.update_entrevista_estado(req.guid, EntrevistaEstado.ENTIDADES_EXTRAIDAS)

    db.log_task(
        TaskLogEntry(
            guid=req.guid,
            service_name="llm-extraction",
            timestamp_inicio=started,
            timestamp_fin=finished,
            status=TaskStatus.OK,
            payload_resultado={"entidades": entidades, "raw": payload},
        )
    )

    return ExtractResponse(guid=req.guid, entidades=entidades)


def _log_error(guid: str, started: datetime, msg: str) -> None:
    db.log_task(
        TaskLogEntry(
            guid=guid,
            service_name="llm-extraction",
            timestamp_inicio=started,
            timestamp_fin=datetime.utcnow(),
            status=TaskStatus.ERROR,
            error_msg=msg,
        )
    )
