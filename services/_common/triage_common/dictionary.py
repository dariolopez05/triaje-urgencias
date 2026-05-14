from __future__ import annotations

import csv
import os
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from triage_common.contracts import GrupoClinico, NormalizedEntity, TriageLevel


FALLBACK_DICTIONARY_PATH = "/app/data/dictionaries/manchester_terms.csv"


def _default_path() -> Path:
    return Path(os.getenv("MANCHESTER_DICTIONARY_PATH", FALLBACK_DICTIONARY_PATH))


@dataclass(frozen=True)
class DictionaryEntry:
    sintoma_coloquial: str
    termino_clinico: str
    prioridad_sugerida: TriageLevel
    grupo_clinico: GrupoClinico


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def canonical(text: str) -> str:
    return _strip_accents(text.strip().lower())


def _load_entries(path: Path) -> dict[str, DictionaryEntry]:
    if not path.exists():
        raise FileNotFoundError(f"Diccionario Manchester no encontrado en {path}")

    entries: dict[str, DictionaryEntry] = {}
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = canonical(row["sintoma_coloquial"])
            entries[key] = DictionaryEntry(
                sintoma_coloquial=row["sintoma_coloquial"],
                termino_clinico=row["termino_clinico"],
                prioridad_sugerida=TriageLevel(_first_priority(row["prioridad_sugerida"])),
                grupo_clinico=GrupoClinico(row["grupo_clinico"]),
            )
    return entries


def _first_priority(value: str) -> str:
    return value.split("/")[0].strip()


@lru_cache(maxsize=8)
def _cached_load(path_str: str) -> dict[str, DictionaryEntry]:
    return _load_entries(Path(path_str))


def load_dictionary(path: Path | None = None) -> dict[str, DictionaryEntry]:
    resolved = path or _default_path()
    return _cached_load(str(resolved))


def reset_cache() -> None:
    _cached_load.cache_clear()


def normalize(term: str, path: Path | None = None) -> NormalizedEntity | None:
    entries = load_dictionary(path)
    key = canonical(term)
    if key in entries:
        return _to_entity(entries[key], term)
    for dict_key, entry in entries.items():
        if dict_key in key or key in dict_key:
            return _to_entity(entry, term)
    return None


def normalize_many(
    terms: list[str], path: Path | None = None
) -> tuple[list[NormalizedEntity], list[str]]:
    mapeados: list[NormalizedEntity] = []
    no_mapeados: list[str] = []
    for raw in terms:
        result = normalize(raw, path)
        if result is None:
            no_mapeados.append(raw)
        else:
            mapeados.append(result)
    return mapeados, no_mapeados


def list_clinical_terms(path: Path | None = None) -> list[str]:
    entries = load_dictionary(path)
    seen: list[str] = []
    for entry in entries.values():
        if entry.termino_clinico not in seen:
            seen.append(entry.termino_clinico)
    return seen


def get_entry_by_term(termino_clinico: str, path: Path | None = None) -> DictionaryEntry | None:
    entries = load_dictionary(path)
    target = termino_clinico.strip().lower()
    best: DictionaryEntry | None = None
    for entry in entries.values():
        if entry.termino_clinico.lower() != target:
            continue
        if best is None or entry.prioridad_sugerida.numeric < best.prioridad_sugerida.numeric:
            best = entry
    return best


def _to_entity(entry: DictionaryEntry, sintoma_original: str) -> NormalizedEntity:
    return NormalizedEntity(
        termino_clinico=entry.termino_clinico,
        prioridad_sugerida=entry.prioridad_sugerida,
        grupo_clinico=entry.grupo_clinico,
        sintoma_original=sintoma_original,
    )
