from __future__ import annotations

import pandas as pd
import pytest

import pipeline as ml


def _sample_df() -> pd.DataFrame:
    rows = [
        {"resumen_es": "presion fuerte en el pecho", "entidades_normalizadas_es": ["dolor_toracico_opresivo"], "triage_real": "C1"},
        {"resumen_es": "se desmayo y no respira", "entidades_normalizadas_es": ["sincope", "disnea"], "triage_real": "C1"},
        {"resumen_es": "infarto agudo de miocardio", "entidades_normalizadas_es": ["dolor_toracico_opresivo", "sincope"], "triage_real": "C1"},
        {"resumen_es": "disnea aguda y dolor toracico", "entidades_normalizadas_es": ["disnea", "dolor_toracico_opresivo"], "triage_real": "C2"},
        {"resumen_es": "dolor opresivo y sudoracion", "entidades_normalizadas_es": ["dolor_toracico_opresivo"], "triage_real": "C2"},
        {"resumen_es": "fiebre alta y dolor de cabeza", "entidades_normalizadas_es": ["fiebre"], "triage_real": "C3"},
        {"resumen_es": "vomitos y diarrea", "entidades_normalizadas_es": ["vomitos", "diarrea"], "triage_real": "C3"},
        {"resumen_es": "lumbago tras cargar caja", "entidades_normalizadas_es": ["lumbalgia"], "triage_real": "C4"},
        {"resumen_es": "tobillo hinchado", "entidades_normalizadas_es": ["edema"], "triage_real": "C4"},
        {"resumen_es": "necesito certificado medico", "entidades_normalizadas_es": [], "triage_real": "C5"},
        {"resumen_es": "consulta administrativa", "entidades_normalizadas_es": [], "triage_real": "C5"},
        {"resumen_es": "vengo a recoger receta", "entidades_normalizadas_es": [], "triage_real": "C5"},
    ]
    return pd.DataFrame(rows)


def test_train_best_returns_artifacts():
    df = _sample_df()
    artifacts = ml.train_best(df)
    assert artifacts.estimator_name in {"logistic_regression", "random_forest", "gradient_boosting"}
    assert "best" in artifacts.metrics
    assert "f1_macro" in artifacts.metrics["best"]
    assert isinstance(artifacts.metrics["best"]["recall_per_class"], dict)


def test_predict_triage_returns_class_and_probs():
    df = _sample_df()
    artifacts = ml.train_best(df)
    pred, probs = ml.predict_triage(
        artifacts.pipeline,
        "presion fuerte en el pecho desmayo",
        ["dolor_toracico_opresivo", "sincope"],
    )
    assert pred in {"C1", "C2", "C3", "C4", "C5"}
    if probs:
        assert sum(probs.values()) == pytest.approx(1.0, abs=0.01)


def test_train_best_empty_raises():
    with pytest.raises(ValueError):
        ml.train_best(pd.DataFrame())
