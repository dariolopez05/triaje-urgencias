from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException

from triage_common import db, dictionary, llm
from triage_common.contracts import (
    EntrevistaEstado,
    EntrevistaTimestamps,
    NormalizeRequest,
    NormalizeResponse,
    NormalizedEntity,
    TaskLogEntry,
    TaskStatus,
)


app = FastAPI(title="TriageIA LLM Normalization", version="0.1.0")

_client: Optional[llm.LLMClient] = None
UNMAPPED_TOKEN = "no_mapeado"


def get_client() -> llm.LLMClient:
    global _client
    if _client is None:
        _client = llm.LLMClient()
    return _client


def _call_llm_for_unmapped(unmapped: list[str]) -> dict[str, str]:
    terms = dictionary.list_clinical_terms()
    payload = get_client().render_and_generate_json(
        "normalize_entities.j2",
        context={
            "terminos_permitidos": terms,
            "sintomas_json": json.dumps(unmapped, ensure_ascii=False),
        },
    )
    mapeos = payload.get("mapeos") or payload.get("mappings") or []
    result: dict[str, str] = {}
    valid = set(terms) | {UNMAPPED_TOKEN}
    for item in mapeos:
        sintoma = str(item.get("sintoma_original", "")).strip()
        termino = str(item.get("termino_clinico", UNMAPPED_TOKEN)).strip()
        if not sintoma:
            continue
        if termino not in valid:
            termino = UNMAPPED_TOKEN
        result[sintoma] = termino
    return result


def _entity_from_term(term: str, sintoma_original: str) -> NormalizedEntity:
    entry = dictionary.get_entry_by_term(term)
    if entry is not None:
        return NormalizedEntity(
            termino_clinico=entry.termino_clinico,
            prioridad_sugerida=entry.prioridad_sugerida,
            grupo_clinico=entry.grupo_clinico,
            sintoma_original=sintoma_original,
        )
    from triage_common.contracts import GrupoClinico, TriageLevel
    return NormalizedEntity(
        termino_clinico=term,
        prioridad_sugerida=TriageLevel.C4,
        grupo_clinico=GrupoClinico.OTRO,
        sintoma_original=sintoma_original,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run", response_model=NormalizeResponse)
def run(req: NormalizeRequest) -> NormalizeResponse:
    started = datetime.utcnow()
    db.mark_timestamp(req.guid, EntrevistaTimestamps.NORMALIZACION, "inicio", when=started)

    mapeados, no_mapeados_directos = dictionary.normalize_many(req.entidades_extraidas)
    no_mapeadas_final: list[str] = []

    if no_mapeados_directos:
        try:
            llm_mappings = _call_llm_for_unmapped(no_mapeados_directos)
        except llm.LLMError as exc:
            _log_error(req.guid, started, str(exc))
            raise HTTPException(status_code=502, detail=f"LLM error: {exc}")

        for sintoma in no_mapeados_directos:
            termino = llm_mappings.get(sintoma, UNMAPPED_TOKEN)
            if termino == UNMAPPED_TOKEN:
                no_mapeadas_final.append(sintoma)
                continue
            mapeados.append(_entity_from_term(termino, sintoma))

    persisted_terms = [m.termino_clinico for m in mapeados]
    db.upsert_texto_procesado(req.guid, {"entidades_normalizadas_es": persisted_terms})

    finished = datetime.utcnow()
    db.mark_timestamp(req.guid, EntrevistaTimestamps.NORMALIZACION, "fin", when=finished)
    db.update_entrevista_estado(req.guid, EntrevistaEstado.ENTIDADES_NORMALIZADAS)

    db.log_task(
        TaskLogEntry(
            guid=req.guid,
            service_name="llm-normalization",
            timestamp_inicio=started,
            timestamp_fin=finished,
            status=TaskStatus.OK,
            payload_resultado={
                "entidades_normalizadas": persisted_terms,
                "no_mapeadas": no_mapeadas_final,
            },
        )
    )

    return NormalizeResponse(
        guid=req.guid,
        entidades_normalizadas=mapeados,
        no_mapeadas=no_mapeadas_final,
    )


def _log_error(guid: str, started: datetime, msg: str) -> None:
    db.log_task(
        TaskLogEntry(
            guid=guid,
            service_name="llm-normalization",
            timestamp_inicio=started,
            timestamp_fin=datetime.utcnow(),
            status=TaskStatus.ERROR,
            error_msg=msg,
        )
    )
