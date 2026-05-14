from __future__ import annotations

from fastapi import FastAPI, HTTPException

from triage_common import db


app = FastAPI(title="TriageIA API Gateway Consulta", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/resultado/{guid}")
def resultado(guid: str) -> dict:
    record = db.fetch_resultado_completo(guid)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Entrevista no encontrada: {guid}")
    return record
