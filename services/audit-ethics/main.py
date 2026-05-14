from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI

from triage_common import db
from triage_common.contracts import (
    AuditEthicsRequest,
    AuditEthicsResponse,
    EntrevistaEstado,
    TaskLogEntry,
    TaskStatus,
    Validacion,
    over_triage,
    under_triage,
)


app = FastAPI(title="TriageIA Audit Ethics", version="0.1.0")

ANXIETY_BIAS_THRESHOLD = 0.8


def evaluate(req: AuditEthicsRequest) -> AuditEthicsResponse:
    validacion = Validacion.PENDIENTE
    motivo: str | None = None
    accion: str | None = None
    sesgo = False

    if req.prediccion_ia == req.triage_real:
        validacion = Validacion.ACIERTO
    elif under_triage(req.prediccion_ia, req.triage_real):
        validacion = Validacion.UNDER_TRIAGE
        if req.score_ansiedad_ia >= ANXIETY_BIAS_THRESHOLD:
            sesgo = True
            motivo = (
                f"Under-triage por sesgo emocional: prediccion {req.prediccion_ia.value} "
                f"vs real {req.triage_real.value} con score ansiedad {req.score_ansiedad_ia:.2f}."
            )
            accion = "Reentrenar con class_weight reforzado en C1/C2 y revisar prompts del LLM."
        else:
            motivo = (
                f"Under-triage clinico: prediccion {req.prediccion_ia.value} vs real {req.triage_real.value}."
            )
            accion = "Revision humana del caso y refuerzo de features clinicas."
    elif over_triage(req.prediccion_ia, req.triage_real):
        validacion = Validacion.OVER_TRIAGE
        motivo = (
            f"Over-triage: prediccion {req.prediccion_ia.value} mas urgente que real {req.triage_real.value}."
        )
        accion = "Monitorizar tasa de falsos positivos. Sin riesgo clinico."

    return AuditEthicsResponse(
        guid=req.guid,
        validacion=validacion,
        motivo_fallo=motivo,
        accion_correctiva=accion,
        sesgo_emocional_detectado=sesgo,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run", response_model=AuditEthicsResponse)
def run(req: AuditEthicsRequest) -> AuditEthicsResponse:
    started = datetime.utcnow()
    result = evaluate(req)

    db.upsert_prediccion(
        req.guid,
        validacion=result.validacion,
        motivo_fallo=result.motivo_fallo,
        accion_correctiva=result.accion_correctiva,
    )
    db.update_entrevista_estado(req.guid, EntrevistaEstado.AUDITADO)

    db.log_task(
        TaskLogEntry(
            guid=req.guid,
            service_name="audit-ethics",
            timestamp_inicio=started,
            timestamp_fin=datetime.utcnow(),
            status=TaskStatus.OK,
            payload_resultado={
                "validacion": result.validacion.value,
                "sesgo_emocional_detectado": result.sesgo_emocional_detectado,
                "motivo_fallo": result.motivo_fallo,
            },
        )
    )

    return result
