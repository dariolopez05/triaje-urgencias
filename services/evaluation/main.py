from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI

from triage_common import db
from triage_common.contracts import (
    EntrevistaEstado,
    EvaluationRequest,
    EvaluationResponse,
    TaskLogEntry,
    TaskStatus,
    Validacion,
    over_triage,
    under_triage,
)


app = FastAPI(title="TriageIA Evaluation", version="0.1.0")


def classify(prediccion, triage_real) -> Validacion:
    if prediccion == triage_real:
        return Validacion.ACIERTO
    if under_triage(prediccion, triage_real):
        return Validacion.UNDER_TRIAGE
    if over_triage(prediccion, triage_real):
        return Validacion.OVER_TRIAGE
    return Validacion.PENDIENTE


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run", response_model=EvaluationResponse)
def run(req: EvaluationRequest) -> EvaluationResponse:
    started = datetime.utcnow()
    validacion = classify(req.prediccion_ia, req.triage_real)

    db.upsert_prediccion(req.guid, validacion=validacion)
    db.update_entrevista_estado(req.guid, EntrevistaEstado.EVALUADO)

    db.log_task(
        TaskLogEntry(
            guid=req.guid,
            service_name="evaluation",
            timestamp_inicio=started,
            timestamp_fin=datetime.utcnow(),
            status=TaskStatus.OK,
            payload_resultado={
                "validacion": validacion.value,
                "prediccion_ia": req.prediccion_ia.value,
                "triage_real": req.triage_real.value,
            },
        )
    )

    return EvaluationResponse(guid=req.guid, validacion=validacion)
