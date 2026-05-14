# CLAUDE.md — TriageIA

Instrucciones permanentes para Claude Code al trabajar en este repositorio.

## Reglas de estilo (obligatorias)

- **No comentarios en el código.** Esto incluye `#` en Python/Bash/YAML/`.env`, `--` en SQL, `//` y `/* */` en JS/TS, y docstrings (`"""..."""`) en Python. El código debe explicarse por sí mismo: nombres descriptivos, funciones pequeñas, tipos cuando aplique. Si una decisión necesita explicación, va en este `CLAUDE.md` o en `docs/`, no en el fichero de código.
- **No emojis en el código.** Ni en identificadores, ni en strings, ni en logs, ni en mensajes que se rendericen al usuario final desde el código (Streamlit, FastAPI, errores, etc.). Markdown de documentación (`README.md`, `docs/*.md`, este `CLAUDE.md`) sí puede usarlos.

Estas dos reglas aplican a todo nuevo código y a cualquier refactor de código existente. Cuando edites un fichero que ya tenga comentarios o emojis previos, retíralos en la misma pasada.

## Convenciones del proyecto

- Idioma del código: identificadores en inglés cuando son técnicos genéricos (`guid`, `status`, `payload`), en español cuando son del dominio clínico (`entidades_normalizadas_es`, `triage_real`, `Entrevista`, `score_ansiedad`).
- Esquema de Postgres: ver `infra/postgres/01-init-triage.sql`. Tablas: `Entrevista`, `Texto_Procesado`, `Prediccion`, `Task_Log`. Vista: `v_resultado_completo`.
- Niveles Manchester: `C1, C2, C3, C4, C5` (mayúsculas, con prefijo C). No usar números sueltos.
- Grupos clínicos: `RES, MSK, CAR, GAS, OTRO`.
- Diccionario cerrado de síntomas: `data/dictionaries/manchester_terms.csv`. No se inventan términos clínicos.
- Trazabilidad: todo paso del pipeline arrastra `GUID_Entrevista` y registra `timestamp_inicio`/`timestamp_fin` + `status` en `Task_Log`.

## Stack y puertos

| Servicio | Puerto host | Notas |
|---|---|---|
| Postgres | 5432 | bases `triage_db` y `airflow_db` |
| minIO API / consola | 9000 / 9001 | buckets: `audio-original`, `textos-originales`, `datasets`, `modelos` |
| Ollama | 11434 | modelos `llama3`, `mistral` |
| Airflow webserver | 8080 | LocalExecutor |
| n8n | 5678 | basic auth |
| Transcripción (faster-whisper) | 9100 | `services/transcripcion/` |
| Streamlit MVP | 8501 | `services/streamlit-mvp/` |
| API Gateway ingesta | 8000 | pendiente, Iter 2 |
| API Gateway consulta | 8001 | pendiente, Iter 2 |

Levantar / parar: `docker compose up -d` / `docker compose down`. Variables en `.env` (plantilla `.env.example`).

## Plan vivo

Plan completo del proyecto en `/home/dorfin/.claude/plans/fizzy-fluttering-sunrise.md`. Estructura por iteraciones (0 a 7). Estado actual: Iteración 0 completada; siguiente es Iter 1 (librería común, contratos Pydantic, diccionario, prompts LLM).

## Fuentes del proyecto

- `guia.pdf` — infraestructura técnica (n8n+Airflow, microservicios, GUID, trazabilidad).
- `guia2.pdf` — dominio clínico (Protocolo Manchester, corpus Fareez et al., MVP Streamlit, auditoría ética). `guia2.pdf` manda en cualquier conflicto de dominio.
