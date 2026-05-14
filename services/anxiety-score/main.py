from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException

from triage_common import db, llm
from triage_common.contracts import (
    EntrevistaEstado,
    EntrevistaTimestamps,
    ScoreRequest,
    ScoreResponse,
    TaskLogEntry,
    TaskStatus,
)


app = FastAPI(title="TriageIA Anxiety Score", version="0.1.0")

_client: Optional[llm.LLMClient] = None

ANXIETY_LEXICON = {
    "miedo": 0.7,
    "panico": 0.95,
    "pánico": 0.95,
    "panico extremo": 1.0,
    "no aguanto": 0.85,
    "no puedo respirar": 0.9,
    "me ahogo": 0.85,
    "me muero": 1.0,
    "se me va a parar el corazon": 0.95,
    "se me va a parar el corazón": 0.95,
    "agonia": 0.9,
    "agonía": 0.9,
    "desesperado": 0.8,
    "desesperada": 0.8,
    "angustia": 0.7,
    "ansiedad": 0.7,
    "horrible": 0.6,
    "terrible": 0.6,
    "muy mal": 0.5,
    "morirme": 0.95,
    "voy a morir": 1.0,
}

LEXICON_PROMPT = (
    "Eres un asistente clinico. Lee el texto del paciente y devuelve UN unico numero "
    "decimal entre 0 y 1 que represente su nivel de ansiedad o panico subjetivo. "
    "0 = totalmente calmado; 1 = panico extremo. Devuelve SOLO el numero, sin texto.\n\n"
    "Texto:\n\"{texto}\"\n\nNumero:"
)


def lexicon_score(texto: str) -> float:
    if not texto:
        return 0.0
    lower = texto.lower()
    matches = [weight for term, weight in ANXIETY_LEXICON.items() if term in lower]
    if not matches:
        return 0.0
    return min(1.0, max(matches))


def parse_llm_score(raw: str) -> float:
    if not raw:
        return 0.0
    match = re.search(r"-?\d+(?:[\.,]\d+)?", raw)
    if not match:
        return 0.0
    value = float(match.group(0).replace(",", "."))
    if value > 1.0 and value <= 10.0:
        value = value / 10.0
    return max(0.0, min(1.0, value))


def get_client() -> llm.LLMClient:
    global _client
    if _client is None:
        _client = llm.LLMClient()
    return _client


def llm_score(texto: str) -> float:
    raw = get_client().generate(LEXICON_PROMPT.format(texto=texto))
    return parse_llm_score(raw)


def combine(lex: float, llm_s: float) -> float:
    value = 0.4 * lex + 0.6 * llm_s
    return round(max(0.0, min(1.0, value)), 2)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run", response_model=ScoreResponse)
def run(req: ScoreRequest) -> ScoreResponse:
    started = datetime.utcnow()
    db.mark_timestamp(req.guid, EntrevistaTimestamps.SCORE, "inicio", when=started)

    lex = lexicon_score(req.texto)
    try:
        llm_s = llm_score(req.texto)
    except llm.LLMError as exc:
        _log_error(req.guid, started, str(exc))
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}")

    score = combine(lex, llm_s)

    db.upsert_texto_procesado(req.guid, {"score_ansiedad": score})

    finished = datetime.utcnow()
    db.mark_timestamp(req.guid, EntrevistaTimestamps.SCORE, "fin", when=finished)
    db.update_entrevista_estado(req.guid, EntrevistaEstado.TEXTO_ENRIQUECIDO)

    db.log_task(
        TaskLogEntry(
            guid=req.guid,
            service_name="anxiety-score",
            timestamp_inicio=started,
            timestamp_fin=finished,
            status=TaskStatus.OK,
            payload_resultado={
                "score_ansiedad": score,
                "lexicon_score": lex,
                "llm_score": llm_s,
            },
        )
    )

    return ScoreResponse(guid=req.guid, score_ansiedad=score)


def _log_error(guid: str, started: datetime, msg: str) -> None:
    db.log_task(
        TaskLogEntry(
            guid=guid,
            service_name="anxiety-score",
            timestamp_inicio=started,
            timestamp_fin=datetime.utcnow(),
            status=TaskStatus.ERROR,
            error_msg=msg,
        )
    )
