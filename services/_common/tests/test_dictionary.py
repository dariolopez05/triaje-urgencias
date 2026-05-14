from __future__ import annotations

import pytest

from triage_common import dictionary
from triage_common.contracts import GrupoClinico, TriageLevel


@pytest.fixture(autouse=True)
def _reset_dictionary_cache():
    dictionary.reset_cache()
    yield
    dictionary.reset_cache()


class TestCanonical:
    def test_strips_accents_and_case(self):
        assert dictionary.canonical("PerdiO El SentIdo") == "perdio el sentido"
        assert dictionary.canonical("  sibIlancias  ") == "sibilancias"


class TestExactMappings:
    @pytest.mark.parametrize(
        "raw, expected_termino, expected_priority, expected_grupo",
        [
            ("me ahogo", "disnea", TriageLevel.C1, GrupoClinico.RES),
            ("no puedo respirar", "disnea", TriageLevel.C1, GrupoClinico.RES),
            ("falta de aire", "disnea", TriageLevel.C2, GrupoClinico.RES),
            ("pitos", "sibilancias", TriageLevel.C2, GrupoClinico.RES),
            ("presion fuerte", "dolor_toracico_opresivo", TriageLevel.C2, GrupoClinico.CAR),
            ("Perdio el sentido", "sincope", TriageLevel.C1, GrupoClinico.CAR),
            ("hinchado", "edema", TriageLevel.C4, GrupoClinico.MSK),
            ("vomitos", "vomitos", TriageLevel.C3, GrupoClinico.GAS),
        ],
    )
    def test_known_terms(self, raw, expected_termino, expected_priority, expected_grupo):
        result = dictionary.normalize(raw)
        assert result is not None
        assert result.termino_clinico == expected_termino
        assert result.prioridad_sugerida == expected_priority
        assert result.grupo_clinico == expected_grupo
        assert result.sintoma_original == raw


class TestFuzzyMatching:
    def test_substring_match(self):
        result = dictionary.normalize("creo que me ahogo bastante")
        assert result is not None
        assert result.termino_clinico == "disnea"


class TestUnknownTerm:
    def test_returns_none(self):
        assert dictionary.normalize("zzz sintoma inventado") is None


class TestNormalizeMany:
    def test_separates_mapped_and_unmapped(self):
        mapeados, no_mapeados = dictionary.normalize_many(
            ["me ahogo", "pitos", "esto no existe"]
        )
        assert {m.termino_clinico for m in mapeados} == {"disnea", "sibilancias"}
        assert no_mapeados == ["esto no existe"]

    def test_empty_input(self):
        mapeados, no_mapeados = dictionary.normalize_many([])
        assert mapeados == []
        assert no_mapeados == []


class TestListClinicalTerms:
    def test_contains_known_terms(self):
        terms = dictionary.list_clinical_terms()
        assert "disnea" in terms
        assert "sibilancias" in terms
        assert "dolor_toracico_opresivo" in terms
        assert "sincope" in terms


class TestGetEntryByTerm:
    def test_returns_entry_with_highest_priority(self):
        entry = dictionary.get_entry_by_term("disnea")
        assert entry is not None
        assert entry.termino_clinico == "disnea"
        assert entry.prioridad_sugerida == TriageLevel.C1

    def test_dolor_toracico_opresivo_is_car(self):
        entry = dictionary.get_entry_by_term("dolor_toracico_opresivo")
        assert entry is not None
        assert entry.grupo_clinico == GrupoClinico.CAR
        assert entry.prioridad_sugerida == TriageLevel.C2

    def test_case_insensitive(self):
        assert dictionary.get_entry_by_term("Disnea") is not None
        assert dictionary.get_entry_by_term("DISNEA") is not None

    def test_unknown_returns_none(self):
        assert dictionary.get_entry_by_term("inventado") is None
