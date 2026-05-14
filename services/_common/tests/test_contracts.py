from __future__ import annotations

import pytest
from pydantic import ValidationError

from triage_common.contracts import (
    AuditEthicsRequest,
    EntrevistaTimestamps,
    GrupoClinico,
    IngestaRequest,
    LabelResponse,
    NormalizedEntity,
    Origen,
    ScoreResponse,
    TriageLevel,
    Validacion,
    over_triage,
    under_triage,
)


class TestTriageLevel:
    def test_numeric_values(self):
        assert TriageLevel.C1.numeric == 1
        assert TriageLevel.C5.numeric == 5

    def test_colors_and_minutes(self):
        assert TriageLevel.C1.color == "Rojo"
        assert TriageLevel.C2.max_minutes == 10
        assert TriageLevel.C5.max_minutes == 240

    def test_under_and_over_triage(self):
        assert under_triage(TriageLevel.C3, TriageLevel.C1)
        assert over_triage(TriageLevel.C1, TriageLevel.C3)
        assert not under_triage(TriageLevel.C1, TriageLevel.C1)


class TestIngestaRequest:
    def test_accepts_text_only(self):
        req = IngestaRequest(texto="Tengo dolor de pecho")
        assert req.texto == "Tengo dolor de pecho"
        assert req.audio_url is None

    def test_accepts_audio_only(self):
        req = IngestaRequest(audio_url="s3://audio-original/x.wav", origen=Origen.DATASET)
        assert req.audio_url == "s3://audio-original/x.wav"
        assert req.origen == Origen.DATASET

    def test_rejects_neither(self):
        with pytest.raises(ValidationError):
            IngestaRequest()

    def test_rejects_both(self):
        with pytest.raises(ValidationError):
            IngestaRequest(texto="hola", audio_url="s3://a/b.wav")

    def test_grupo_clinico_optional(self):
        req = IngestaRequest(texto="x", grupo_clinico=GrupoClinico.RES)
        assert req.grupo_clinico == GrupoClinico.RES


class TestScoreResponse:
    def test_in_range(self):
        ScoreResponse(guid="g1", score_ansiedad=0.5)
        ScoreResponse(guid="g1", score_ansiedad=0.0)
        ScoreResponse(guid="g1", score_ansiedad=1.0)

    def test_out_of_range(self):
        with pytest.raises(ValidationError):
            ScoreResponse(guid="g1", score_ansiedad=1.5)
        with pytest.raises(ValidationError):
            ScoreResponse(guid="g1", score_ansiedad=-0.1)


class TestLabelResponse:
    def test_valid_triage_level(self):
        resp = LabelResponse(guid="g1", triage=TriageLevel.C2, justificacion="..")
        assert resp.triage == TriageLevel.C2

    def test_invalid_triage_string(self):
        with pytest.raises(ValidationError):
            LabelResponse(guid="g1", triage="X9", justificacion="..")


class TestNormalizedEntity:
    def test_round_trip(self):
        ent = NormalizedEntity(
            termino_clinico="disnea",
            prioridad_sugerida=TriageLevel.C1,
            grupo_clinico=GrupoClinico.RES,
            sintoma_original="me ahogo",
        )
        assert ent.termino_clinico == "disnea"
        assert ent.prioridad_sugerida == TriageLevel.C1


class TestEntrevistaTimestamps:
    def test_column_names(self):
        assert EntrevistaTimestamps.SOLICITUD.inicio_column == "Inicio_Solicitud"
        assert EntrevistaTimestamps.TRANSCRIPCION.fin_column == "Fin_Transcripcion"


class TestAuditEthicsRequest:
    def test_accepts_valid(self):
        req = AuditEthicsRequest(
            guid="g1",
            prediccion_ia=TriageLevel.C3,
            triage_real=TriageLevel.C1,
            score_ansiedad_ia=0.9,
        )
        assert under_triage(req.prediccion_ia, req.triage_real)


class TestValidacion:
    def test_values(self):
        assert Validacion.UNDER_TRIAGE.value == "Under-triage"
        assert Validacion.ACIERTO.value == "Acierto"
