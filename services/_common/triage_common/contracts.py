from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TriageLevel(str, Enum):
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"
    C4 = "C4"
    C5 = "C5"

    @property
    def numeric(self) -> int:
        return int(self.value[1:])

    @property
    def color(self) -> str:
        return {
            "C1": "Rojo",
            "C2": "Naranja",
            "C3": "Amarillo",
            "C4": "Verde",
            "C5": "Azul",
        }[self.value]

    @property
    def max_minutes(self) -> int:
        return {"C1": 0, "C2": 10, "C3": 60, "C4": 120, "C5": 240}[self.value]


class GrupoClinico(str, Enum):
    RES = "RES"
    MSK = "MSK"
    CAR = "CAR"
    GAS = "GAS"
    OTRO = "OTRO"


class Origen(str, Enum):
    DATASET = "Dataset"
    SIMULACION = "Simulacion"
    MVP = "MVP"


class Validacion(str, Enum):
    ACIERTO = "Acierto"
    UNDER_TRIAGE = "Under-triage"
    OVER_TRIAGE = "Over-triage"
    PENDIENTE = "Pendiente"


class TaskStatus(str, Enum):
    OK = "OK"
    ERROR = "ERROR"
    RETRY = "RETRY"
    TIMEOUT = "TIMEOUT"


class NormalizedEntity(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    termino_clinico: str
    prioridad_sugerida: TriageLevel
    grupo_clinico: GrupoClinico
    sintoma_original: str


class IngestaRequest(BaseModel):
    texto: Optional[str] = None
    audio_url: Optional[str] = None
    id_caso: Optional[str] = None
    origen: Origen = Origen.MVP
    grupo_clinico: Optional[GrupoClinico] = None

    @model_validator(mode="after")
    def must_have_texto_or_audio(self) -> "IngestaRequest":
        if not self.texto and not self.audio_url:
            raise ValueError("Debe proporcionarse 'texto' o 'audio_url'")
        if self.texto and self.audio_url:
            raise ValueError("Solo uno de 'texto' o 'audio_url'")
        return self


class IngestaResponse(BaseModel):
    guid: str
    estado: str
    workflow_id: Optional[str] = None


class TranscribeRequest(BaseModel):
    guid: str
    audio_url: str
    language: Optional[str] = None


class TranscribeResponse(BaseModel):
    guid: str
    texto: str
    language: str
    duration_seconds: float


class PreprocessRequest(BaseModel):
    guid: str
    texto: str


class PreprocessResponse(BaseModel):
    guid: str
    texto_preprocesado: str


class ExtractRequest(BaseModel):
    guid: str
    texto: str
    language: str = "es"


class ExtractResponse(BaseModel):
    guid: str
    entidades: list[str] = Field(default_factory=list)


class NormalizeRequest(BaseModel):
    guid: str
    entidades_extraidas: list[str]


class NormalizeResponse(BaseModel):
    guid: str
    entidades_normalizadas: list[NormalizedEntity] = Field(default_factory=list)
    no_mapeadas: list[str] = Field(default_factory=list)


class LabelRequest(BaseModel):
    guid: str
    resumen_es: str
    entidades_normalizadas: list[NormalizedEntity]


class LabelResponse(BaseModel):
    guid: str
    triage: TriageLevel
    justificacion: str


class ScoreRequest(BaseModel):
    guid: str
    texto: str


class ScoreResponse(BaseModel):
    guid: str
    score_ansiedad: float = Field(ge=0.0, le=1.0)


class DatasetRow(BaseModel):
    id_caso: str
    origen: Origen
    texto_original_en: Optional[str] = None
    resumen_es: str
    entidades_extraidas_es: list[str]
    entidades_normalizadas_es: list[str]
    triage_real: TriageLevel
    grupo_clinico: Optional[GrupoClinico] = None
    score_ansiedad: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class PredictRequest(BaseModel):
    guid: str
    texto: str
    entidades_normalizadas: list[str] = Field(default_factory=list)


class PredictResponse(BaseModel):
    guid: str
    prediccion_ia: TriageLevel
    score_ansiedad_ia: float = Field(ge=0.0, le=1.0)
    probabilidades: dict[str, float] = Field(default_factory=dict)


class EvaluationRequest(BaseModel):
    guid: str
    prediccion_ia: TriageLevel
    triage_real: TriageLevel


class EvaluationResponse(BaseModel):
    guid: str
    validacion: Validacion


class AuditEthicsRequest(BaseModel):
    guid: str
    prediccion_ia: TriageLevel
    triage_real: TriageLevel
    score_ansiedad_ia: float = Field(ge=0.0, le=1.0)


class AuditEthicsResponse(BaseModel):
    guid: str
    validacion: Validacion
    motivo_fallo: Optional[str] = None
    accion_correctiva: Optional[str] = None
    sesgo_emocional_detectado: bool = False


class TaskLogEntry(BaseModel):
    guid: Optional[str] = None
    service_name: str
    timestamp_inicio: datetime
    timestamp_fin: Optional[datetime] = None
    status: TaskStatus = TaskStatus.OK
    payload_resultado: Optional[dict[str, Any]] = None
    error_msg: Optional[str] = None


class EntrevistaTimestamps(str, Enum):
    SOLICITUD = "Solicitud"
    TRANSCRIPCION = "Transcripcion"
    PREPROCESAMIENTO = "Preprocesamiento"
    EXTRACCION_ENTIDADES = "Extraccion_Entidades"
    NORMALIZACION = "Normalizacion"
    ETIQUETADO = "Etiquetado"
    SCORE = "Score"
    ENTRENAMIENTO = "Entrenamiento"

    @property
    def inicio_column(self) -> str:
        return f"Inicio_{self.value}"

    @property
    def fin_column(self) -> str:
        return f"Fin_{self.value}"


class EntrevistaEstado(str, Enum):
    RECIBIDO = "RECIBIDO"
    TRANSCRITO = "TRANSCRITO"
    TEXTO_PREPROCESADO = "TEXTO_PREPROCESADO"
    ENTIDADES_EXTRAIDAS = "ENTIDADES_EXTRAIDAS"
    ENTIDADES_NORMALIZADAS = "ENTIDADES_NORMALIZADAS"
    ETIQUETADO = "ETIQUETADO"
    SCORE_CALCULADO = "SCORE_CALCULADO"
    TEXTO_ENRIQUECIDO = "TEXTO_ENRIQUECIDO"
    DATASET_GENERADO = "DATASET_GENERADO"
    MODELO_ENTRENADO = "MODELO_ENTRENADO"
    PREDICHO = "PREDICHO"
    EVALUADO = "EVALUADO"
    AUDITADO = "AUDITADO"
    ERROR = "ERROR"


def under_triage(prediccion: TriageLevel, real: TriageLevel) -> bool:
    return prediccion.numeric > real.numeric


def over_triage(prediccion: TriageLevel, real: TriageLevel) -> bool:
    return prediccion.numeric < real.numeric
