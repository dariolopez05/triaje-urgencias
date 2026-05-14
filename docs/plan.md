# Plan ejecutable — Proyecto TriageIA (Sistema de Triaje Manchester con LLM + ML)

> Copia del plan vivo del proyecto. La versión maestra reside en
> `/home/dorfin/.claude/plans/fizzy-fluttering-sunrise.md`; este fichero del repo se mantiene sincronizado para servir de referencia visible en GitHub.

---

## Estado actual

**Fecha:** 2026-05-14
**Iteración en curso:** Iteración 2 (siguiente paso)
**Última iteración completada:** **Iteración 1 — Librería común y prompts**

| Iter | Contenido | Estado |
|---|---|---|
| 0 | Esqueleto e infraestructura (docker-compose, Postgres, minIO, Ollama, Airflow, n8n, Whisper, Streamlit, diccionario Manchester) | ✅ Completada |
| 1 | Librería común `triage_common` (contracts, db, storage, llm, dictionary), prompts Jinja2, 67 tests unitarios verdes | ✅ Completada |
| 2 | Microservicios FastAPI: A (gateways+prepro) + B (LLM+Whisper) + C (ML pipeline) ✅ con 132 tests + E2E pipeline completo; D (Streamlit UI) pendiente | ⏳ En curso |
| 3 | DAGs Airflow (audio, texto, training, evaluación, auditoría) | ⏳ Pendiente |
| 4 | Flujos n8n (webhook Fase 2, notificaciones de error) | ⏳ Pendiente |
| 5 | Descarga del corpus Fareez et al. + entrenamiento del modelo ML | ⏳ Pendiente |
| 6 | Tests de integración y E2E | ⏳ Pendiente |
| 7 | Documentación final, MVP Streamlit completo y presentación | ⏳ Pendiente |

**Hito alcanzado:** infraestructura levantando (Postgres, minIO, Whisper, Streamlit healthy) y librería compartida `triage_common` lista, instalable como paquete pip (`services/_common/pyproject.toml`), con 67 tests unitarios verdes ejecutados en un contenedor `python:3.11-slim` aislado. Cubre: contratos Pydantic (TriageLevel C1-C5, GrupoClinico, Origen, Validacion, modelos de request/response de todos los servicios), cliente Postgres con helpers (`insert_entrevista`, `update_entrevista_estado`, `mark_timestamp`, `upsert_texto_procesado`, `upsert_prediccion`, `log_task`, `fetch_resultado_completo`), cliente minIO (put/get bytes/file, presign, `ensure_buckets`, constantes de bucket), cliente Ollama (`generate`, `generate_json`, `render`, `render_and_generate*`, reintentos con tenacity), módulo de diccionario clínico cerrado (normalización con strip de acentos + fuzzy substring) y 3 plantillas Jinja2 en `data/prompts/` (extract_entities, normalize_entities, label_triage) con few-shot RES/MSK/CAR/GAS bilingüe EN/ES.

**Siguiente paso concreto:** crear los microservicios FastAPI bajo `services/<nombre>/` reutilizando `triage_common`. Empezar por `api-gateway-ingesta`, `transcripcion` (completar el endpoint `/transcribe` que actualmente solo tiene `/health`), y luego los servicios de LLM (extraction, normalization, labeling) y ML (training, prediction, evaluation, audit-ethics).

---

## Contexto

Proyecto definido por dos documentos en `/home/dorfin/proyecto_triage_6/`:
- **`guia.pdf`** — infraestructura técnica del sistema (n8n+Airflow, microservicios, Postgres, minIO, GUIDs, trazabilidad).
- **`guia2.pdf`** — dominio clínico y producto final (Protocolo Manchester, dataset Fareez et al., MVP Streamlit, auditoría ética). En caso de conflicto, manda `guia2.pdf`.

El directorio empezaba solo con esos PDFs; se construye el sistema desde cero.

**Qué se construye:** un sistema que recibe **audio o texto** de la voz del paciente, lo procesa con un pipeline orquestado por n8n + Airflow (Whisper → preprocesamiento → LLM extracción/normalización clínica → score de ansiedad → modelo ML), y devuelve un **nivel de triage Manchester C1-C5**. Incluye dashboard Streamlit para uso por personal sanitario y auditoría ética para detectar under-triage.

**Decisiones del usuario:**
- Motor de orquestación: **n8n + Airflow** (n8n recibe peticiones y notifica, Airflow ejecuta DAGs pesados).
- LLM: **Ollama local** (`llama3` / `mistral`).
- Dominio: **Triage Manchester C1-C5**.
- Dataset: **Fareez et al. (2022)** — 272 entrevistas OSCE desde Hugging Face.
- Whisper y Streamlit añadidos al docker-compose desde la Iteración 0.

---

## Dominio clínico (de guia2.pdf)

**Niveles Manchester:**

| Nivel | Color | Tiempo máx. | Descripción |
|---|---|---|---|
| C1 | Rojo | 0 min | Emergencia (paro, síncope, IAM) |
| C2 | Naranja | 10 min | Muy urgente (disnea aguda, dolor torácico opresivo) |
| C3 | Amarillo | 60 min | Urgente (fiebre alta, sibilancias) |
| C4 | Verde | 120 min | Menos urgente (edema, lumbago) |
| C5 | Azul | 240 min | No urgente |

**Grupos clínicos del corpus (con desbalance que SMOTE/class_weight debe compensar):**

| Grupo | Casos | Descripción |
|---|---|---|
| RES | 214 | Respiratorio (asma, neumonía, gripe) |
| MSK | 46 | Musculoesquelético (esguinces, lumbago, gota) |
| GAS | 6 | Gastrointestinal (gastroenteritis, apendicitis) |
| CAR | 5 | Cardíaco (angina, IAM) |

**Diccionario obligatorio de normalización** (`data/dictionaries/manchester_terms.csv`):

| Síntoma coloquial (input) | Término clínico (target) | Prioridad sugerida |
|---|---|---|
| "Me ahogo", "No puedo respirar", "Falta de aire" | Disnea | C1 / C2 |
| "Pitos", "Silbidos al inhalar" | Sibilancias | C2 / C3 |
| "Presión fuerte", "Como un elefante en el pecho" | Dolor Torácico Opresivo | C2 |
| "Ardiendo", "Tengo mucha calentura" | Fiebre / Hipertermia | C3 |
| "Perdió el sentido", "Se desplomó" | Síncope | C1 |
| "Hinchado", "Como un globo" | Edema / Inflamación | C4 |

**No se inventan términos** — todo lo que extrae el LLM debe mapearse a este diccionario cerrado.

---

## Arquitectura objetivo

```
┌──────────┐      ┌──────────┐      ┌──────────────────────┐
│ Cliente  │─HTTP─▶ n8n      │─HTTP▶│ Airflow DAGs         │
│ (Stream. │      │ webhooks │      │ - dag_text_ingestion │
│  o curl) │      └────┬─────┘      │ - dag_llm_enrichment │
└──────────┘           │            │ - dag_model_training │
                       │            │ - dag_predict_phase2 │
                       │            │ - dag_evaluation     │
                       │            │ - dag_audit_ethics   │
                       ▼            └──────────┬───────────┘
              ┌────────────────┐               │
              │ Microservicios │◀──── HTTP ────┘
              │  FastAPI:      │
              │  transcripcion │ (Whisper)
              │  preprocessing │
              │  llm-extract.  │
              │  llm-normaliz. │ (usa diccionario cerrado)
              │  llm-labeling  │ (Manchester C1-C5 + justif.)
              │  anxiety-score │ (para auditoría ética)
              │  dataset-build │
              │  ml-training   │ (class_weight/SMOTE)
              │  ml-prediction │
              │  evaluation    │ (Recall > 80% C1/C2)
              │  audit-ethics  │ (under-triage detection)
              └──────┬─────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   ┌────────┐  ┌──────────┐  ┌────────┐
   │Postgres│  │ minIO/S3 │  │ Ollama │
   │(estado)│  │ (audio,  │  │ (LLM)  │
   └────────┘  │ datasets,│  └────────┘
               │ modelos) │
               └──────────┘
                     │
              ┌──────▼──────┐
              │  Streamlit  │ (MVP — sube audio, ve triage)
              └─────────────┘
```

Todo levantado con un único `docker-compose.yml`.

---

## Estructura del repositorio

```
proyecto_triage_6/
├── docker-compose.yml
├── .env.example
├── README.md
├── CLAUDE.md
├── docs/
│   ├── plan.md                       # este fichero
│   ├── arquitectura.md
│   ├── diagrama-arquitectura.png
│   ├── pipeline-fase1.md
│   ├── pipeline-fase2.md
│   ├── gestion-errores.md
│   ├── modelo-ml.md
│   ├── auditoria-etica.md
│   └── presentacion.pptx
├── infra/
│   ├── postgres/01-init-triage.sql
│   ├── minio/bootstrap-buckets.sh
│   └── ollama/pull-models.sh
├── services/
│   ├── api-gateway-ingesta/
│   ├── api-gateway-consulta/
│   ├── transcripcion/
│   ├── preprocessing/
│   ├── llm-extraction/
│   ├── llm-normalization/
│   ├── llm-labeling/
│   ├── anxiety-score/
│   ├── dataset-builder/
│   ├── ml-training/
│   ├── ml-prediction/
│   ├── evaluation/
│   ├── audit-ethics/
│   ├── streamlit-mvp/
│   └── _common/
├── airflow/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── dags/
│       ├── dag_text_ingestion.py
│       ├── dag_audio_ingestion.py
│       ├── dag_llm_enrichment.py
│       ├── dag_model_training.py
│       ├── dag_prediction_phase_2.py
│       ├── dag_evaluation.py
│       └── dag_audit_ethics.py
├── n8n/workflows/
├── data/
│   ├── prompts/
│   ├── dictionaries/manchester_terms.csv
│   ├── samples/
│   └── fareez_dataset/
└── tests/{unit,integration,e2e}/
```

---

## Modelo de datos en Postgres

Tabla principal **`Entrevista`** (sección 4.3 de guia.pdf, extendida con campos de guia2.pdf):

```sql
CREATE TABLE Entrevista (
    GUID_Entrevista VARCHAR(255) PRIMARY KEY,
    ID_CASO VARCHAR(50),
    Origen VARCHAR(20),
    Grupo_Clinico VARCHAR(10),
    URL_Audio_Original VARCHAR(255),
    URL_Texto_Original VARCHAR(255),
    URL_Dataset_Generado VARCHAR(255),
    URL_Modelo_Entrenado VARCHAR(255),
    Inicio_Solicitud TIMESTAMP, Fin_Solicitud TIMESTAMP,
    Inicio_Transcripcion TIMESTAMP, Fin_Transcripcion TIMESTAMP,
    Inicio_Preprocesamiento TIMESTAMP, Fin_Preprocesamiento TIMESTAMP,
    Inicio_Extraccion_Entidades TIMESTAMP, Fin_Extraccion_Entidades TIMESTAMP,
    Inicio_Normalizacion TIMESTAMP, Fin_Normalizacion TIMESTAMP,
    Inicio_Etiquetado TIMESTAMP, Fin_Etiquetado TIMESTAMP,
    Inicio_Score TIMESTAMP, Fin_Score TIMESTAMP,
    Inicio_Entrenamiento TIMESTAMP, Fin_Entrenamiento TIMESTAMP,
    Motor_Workflow VARCHAR(50),
    Workflow_Id VARCHAR(255),
    Estado VARCHAR(50)
);
```

Tabla **`Texto_Procesado`** (alineada con el ejemplo de dataset integrado de guia2):

```sql
CREATE TABLE Texto_Procesado (
    guid VARCHAR(255) PRIMARY KEY REFERENCES Entrevista(GUID_Entrevista),
    texto_original_en TEXT,
    resumen_es TEXT,
    texto_preprocesado TEXT,
    entidades_extraidas_es JSONB,
    entidades_normalizadas_es JSONB,
    triage_real VARCHAR(5),
    score_ansiedad NUMERIC(3,2),
    justificacion_llm TEXT
);
```

Tabla **`Prediccion`** (Fase 2 + auditoría):

```sql
CREATE TABLE Prediccion (
    guid VARCHAR(255) PRIMARY KEY REFERENCES Entrevista(GUID_Entrevista),
    prediccion_ia VARCHAR(5),
    score_ansiedad_ia NUMERIC(3,2),
    validacion VARCHAR(20),
    motivo_fallo TEXT,
    accion_correctiva TEXT,
    fecha TIMESTAMP DEFAULT NOW()
);
```

Tabla **`Task_Log`** (trazabilidad de cada llamada HTTP del pipeline):

```sql
CREATE TABLE Task_Log (
    id SERIAL PRIMARY KEY,
    guid VARCHAR(255),
    service_name VARCHAR(100),
    timestamp_inicio TIMESTAMP,
    timestamp_fin TIMESTAMP,
    status VARCHAR(20),
    payload_resultado JSONB,
    error_msg TEXT
);
```

---

## Plan por iteraciones

### Iteración 0 — Esqueleto e infraestructura ✅

- [x] Estructura de carpetas.
- [x] `docker-compose.yml` con servicios: `postgres`, `minio`, `minio-bootstrap`, `ollama`, `ollama-init`, `airflow-init`, `airflow-webserver`, `airflow-scheduler`, `n8n`, `transcripcion`, `streamlit-mvp`. Red `triage_net`.
- [x] `.env.example` con POSTGRES_*, MINIO_*, AIRFLOW_*, N8N_*, OLLAMA_*, AIRFLOW_UID, WHISPER_*, STREAMLIT_PORT.
- [x] `infra/postgres/00-create-airflow-db.sh` + `01-init-triage.sql` con esquema completo (C1-C5, ID_CASO, transcripción, grupos clínicos) + Texto_Procesado + Prediccion + Task_Log + vista `v_resultado_completo`.
- [x] `infra/minio/bootstrap-buckets.sh` crea buckets `audio-original`, `textos-originales`, `datasets`, `modelos`.
- [x] `infra/ollama/pull-models.sh` con `ollama pull llama3` y `mistral`.
- [x] `services/transcripcion/` (FastAPI + faster-whisper) endpoint `/health` operativo; `/transcribe` se completa en Iter 2.
- [x] `services/streamlit-mvp/` placeholder con paleta Manchester.
- [x] `data/dictionaries/manchester_terms.csv` con el diccionario obligatorio.
- [x] `README.md` mínimo con quickstart.
- [x] `CLAUDE.md` con reglas de estilo (sin comentarios, sin emojis en código).
- [x] **Verificación parcial:** `docker compose config` válido; `postgres`, `minio`, `minio-bootstrap`, `transcripcion`, `streamlit-mvp` arrancan healthy y responden. Ollama y Airflow validados a nivel de YAML pero no arrancados aún (se hará al usarlos en Iter 2/3).

### Iteración 1 — Librería común, contratos y diccionario ✅

- [x] `services/_common/pyproject.toml` declara el paquete `triage_common` instalable con `pip install -e .[test]`.
- [x] `triage_common/contracts.py` — Pydantic: `TriageLevel` (C1-C5 con `numeric`, `color`, `max_minutes`), `GrupoClinico`, `Origen`, `Validacion`, `TaskStatus`, `NormalizedEntity`, `IngestaRequest/Response` (validación XOR texto/audio), `TranscribeRequest/Response`, `ExtractRequest/Response`, `NormalizeRequest/Response`, `LabelRequest/Response`, `ScoreRequest/Response`, `DatasetRow`, `PredictRequest/Response`, `EvaluationRequest/Response`, `AuditEthicsRequest/Response`, `TaskLogEntry`, `EntrevistaTimestamps`, `EntrevistaEstado` + helpers `under_triage`, `over_triage`.
- [x] `triage_common/db.py` — cliente psycopg2 con `DbConfig.from_env`, context manager `get_connection` (commit/rollback), helpers `insert_entrevista`, `update_entrevista_estado`, `mark_timestamp`, `upsert_texto_procesado` (envuelve campos JSONB), `upsert_prediccion`, `log_task`, `fetch_resultado_completo` (vía vista `v_resultado_completo`).
- [x] `triage_common/storage.py` — cliente minio-py con constantes de bucket, `put_bytes/put_file/put_stream`, `get_bytes/get_to_file`, `presign_url`, `list_objects`, `ensure_buckets` y `parse_uri` para `s3://bucket/key`.
- [x] `triage_common/llm.py` — cliente Ollama HTTP con `LLMConfig.from_env`, render Jinja2 estricto, `generate`, `generate_json` (modo JSON nativo de Ollama), `render_and_generate*`, `list_models` y reintentos exponenciales con tenacity (errores `LLMError`/`LLMInvalidJSON`).
- [x] `triage_common/dictionary.py` — carga `manchester_terms.csv` con caché LRU lazy (sensible a `MANCHESTER_DICTIONARY_PATH`), `canonical()` strip acentos+lowercase, `normalize()` match exacto + substring, `normalize_many()` separa mapeados/no_mapeados, `list_clinical_terms()`.
- [x] `data/prompts/extract_entities.j2` — system prompt clínico con few-shot bilingüe.
- [x] `data/prompts/normalize_entities.j2` — fuerza mapeo al diccionario cerrado (incluye `no_mapeado` como fallback explícito).
- [x] `data/prompts/label_triage.j2` — JSON `{triage, justificacion}` con criterios Manchester y principio precautorio.
- [x] **Verificación:** `docker run --rm -v $PWD:/repo python:3.11-slim` ejecuta `pytest` → **67 tests pasan** cubriendo contracts (TriageLevel, validators, enums), dictionary (8 casos exactos del PDF + fuzzy + no_mapeados), db (mock psycopg2, JSONB wrapping, rollback en error), storage (mock minio, presign, ensure_buckets, parse_uri) y llm (mock httpx, retries con backoff, render Jinja con StrictUndefined).

### Iteración 2 — Microservicios Python (3–4 días) ⏳ (en curso, bloque A completado)

Cada servicio: FastAPI + `Dockerfile`, expone `POST /run` (JSON con `guid` + payload), registra en `Task_Log`.

**Bloque A (gateways + preprocessing) ✅:**

- [x] `api-gateway-ingesta` — `POST /ingesta` multipart (audio.wav | texto) → GUID → minIO → fila `Entrevista` con `Estado=RECIBIDO` → dispara DAG vía REST API de Airflow (no-op si `AIRFLOW_BASE_URL` vacío). Tests: 10 verdes.
- [x] `api-gateway-consulta` — `GET /resultado/{guid}` agregado desde vista `v_resultado_completo`. Tests: 3 verdes.
- [x] `preprocessing` — `POST /run` con limpieza Unicode NFC, strip de caracteres de control `\x00-\x1f\x7f`, eliminación de URLs (`https?://`, `www.`), colapso de whitespace; escribe `texto_preprocesado` y marca timestamps `Inicio/Fin_Preprocesamiento` en `Entrevista`. Tests: 8 verdes.
- [x] Refactor `docker-compose.yml`: context raíz para servicios Python, los 3 nuevos arrancan en `:8000`, `:8001`, `:9101` y dependen de `postgres`/`minio` healthy.
- [x] **Smoke test end-to-end verde:** `curl -F texto=...` a `:8000/ingesta` → fila en Postgres + archivo en `minio://textos-originales/{guid}.txt` → `curl :9101/run` con el GUID → `texto_preprocesado` en Postgres + timestamps + 2 filas en `Task_Log` ('OK', 'OK') → `curl :8001/resultado/{guid}` devuelve la vista agregada.

**Bloque B (servicios LLM) ✅:**

- [x] `transcripcion` — endpoint `POST /transcribe` recibe `audio_url` (`s3://audio-original/...`), descarga via minIO, transcribe con `faster-whisper` (modelo `base`, CPU/int8, vad_filter), guarda `resumen_es` (+ `texto_original_en` si idioma EN) en `Texto_Procesado`, marca `Inicio/Fin_Transcripcion`.
- [x] `llm-extraction` — Ollama + `extract_entities.j2`, devuelve `entidades` (extracción bilingüe EN/ES). Tests: 5 verdes. E2E con qwen3:4b ~1s.
- [x] `llm-normalization` — mapeo exacto con `dictionary.normalize_many` → LLM (`normalize_entities.j2`) para los no mapeados con lista cerrada de términos → rechaza inventos del LLM (vuelven a `no_mapeadas`). Tests: 5 verdes.
- [x] `llm-labeling` — Ollama + `label_triage.j2` → `{triage, justificacion}`, valida `C1..C5`. Tests: 4 verdes. E2E aplica principio precautorio Manchester correctamente.
- [x] `anxiety-score` — lexicón emocional ponderado (`miedo`, `panico`, `no aguanto`, etc.) + LLM score 0-1; combinación 0.4·lex + 0.6·llm. Tests: 12 verdes.
- [x] **Cliente Ollama actualizado:** `think:false` por defecto en payload de `/api/generate` (los modelos qwen3 son "thinking" y devuelven en campo `thinking` por defecto).
- [x] **Smoke E2E verde**: `curl` real ingesta → preprocessing → extraction → normalization → labeling → anxiety-score → consulta. Texto `"el pecho como una losa, me cuesta respirar, miedo de morirme"` → **triage C1** con justificación clínica + 6 filas OK en `Task_Log`.

**Bloque C (datos y ML) ✅:**

- [x] `dataset-builder` (`:9120`) — query `Texto_Procesado` con `triage_real`, exporta parquet con columnas guia2, sube a `minio://datasets/{batch_id}.parquet`, marca `Estado=DATASET_GENERADO` + `URL_Dataset_Generado` en `Entrevista`.
- [x] `ml-training` (`:9121`) — TF-IDF n-gramas 1-2 + multi-hot entidades; comparativa LogReg/RandomForest/GradientBoosting con `class_weight='balanced'`; CV stratified (adaptativo según tamaño); selección por F1-macro; guarda `*.joblib` + `metrics.json` en `minio://modelos/`. Tests: 3 verdes.
- [x] `ml-prediction` (`:9122`) — caché en memoria del último modelo, endpoint `/reload`, devuelve `prediccion_ia` + `probabilidades` (probas por clase). Persiste en `Prediccion` y avanza estado a `PREDICHO`.
- [x] `evaluation` (`:9123`) — clasifica Acierto/Under-triage/Over-triage usando `under_triage`/`over_triage` helpers, escribe `Prediccion.validacion`, estado `EVALUADO`. Tests: 5 verdes.
- [x] `audit-ethics` (`:9124`) — detecta under-triage por sesgo emocional (`score_ansiedad_ia ≥ 0.8` + predicción peor que real), escribe `motivo_fallo` y `accion_correctiva`, estado `AUDITADO`. Tests: 6 verdes.
- [x] **Smoke E2E con seed 12 filas**: dataset → modelo LogReg (F1-macro 0.47, Recall(C1/C2) 0.33 — limitado por 12 muestras) → predicción `C1` correcta en caso clínico (presión torácica + síncope) → evaluation 'Acierto' → audit-ethics sin sesgo. Estado avanza hasta `AUDITADO`. La calidad subirá con dataset real Fareez (272 casos) en Iter 5.

**Bloque D (UI) ⏳ (siguiente):**

- [ ] `streamlit-mvp` — UI completa con upload, polling, visualización del triage.

- **Verificación bloque A:** `pytest` en docker (21 tests verdes) + smoke test con `curl` cubriendo ingesta → preprocessing → consulta.

### Iteración 3 — DAGs Airflow (Fase 1) (2–3 días)

`SimpleHttpOperator`; cada tarea registra timestamps; `retries=2`, `on_failure_callback` → webhook n8n.

- [ ] `dag_audio_ingestion.py` — manual trigger con `guid`: transcripcion → preprocessing → extraction → normalization → labeling → anxiety-score → `Estado=TEXTO_ENRIQUECIDO`.
- [ ] `dag_text_ingestion.py` — sin transcripción.
- [ ] `dag_llm_enrichment.py` — batch sobre `Estado=RECIBIDO`.
- [ ] `dag_model_training.py` — dataset-builder → ml-training → guarda URL del modelo.
- [ ] `dag_evaluation.py` — sobre `Estado=PREDICHO`.
- [ ] `dag_audit_ethics.py` — sobre `Estado=EVALUADO`.
- **Verificación:** disparar el DAG con audio de muestra y trazar en Postgres + Airflow UI.

### Iteración 4 — Flujos n8n (Fase 2 + notificaciones) (1–2 días)

- [ ] `webhook_fase2_prediccion.json` — `POST /webhook/predict` con audio/texto → ingesta → polling → ml-prediction → audit-ethics → respuesta `{guid, triage, justificacion, score, validacion}`.
- [ ] `webhook_consulta_resultado.json` — `GET /webhook/resultado/{guid}`.
- [ ] `error_notification.json` — recibe callbacks de Airflow.
- **Verificación:** `curl -X POST :5678/webhook/predict -F "audio=@samples/SIM_G1_01.wav"` devuelve triage Manchester.

### Iteración 5 — Datos Fareez et al. y Modelo ML (3 días)

**Dataset:**
- [ ] `scripts/download_fareez.py` (descarga desde Hugging Face).
- [ ] `scripts/seed_postgres.py` (ingesta los 272 casos lanzando `dag_audio_ingestion`).
- [ ] Ground truth: revisión humana del `triage_real` para 50 casos de control.

**Features:**
- `resumen_es` → embeddings `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` o TF-IDF n-gramas 1-2.
- `entidades_normalizadas_es` → multi-hot sobre los términos del diccionario.
- `score_ansiedad` → no se usa como feature (sólo en `audit-ethics`).

**Algoritmo:**
- Baseline `LogisticRegression(class_weight='balanced')`.
- Comparar `RandomForestClassifier` y `GradientBoostingClassifier`.
- SMOTE si `class_weight` no consigue Recall ≥ 0.80 en CAR/GAS.
- 5-fold stratified CV; criterio: F1-macro con restricciones `Recall(C1) ≥ 0.85` y `Recall(C2) ≥ 0.80`.

**Métricas en `metrics.json`:** accuracy global, P/R/F1 por clase, matriz de confusión PNG, Recall específico C1/C2, tasa de under-triage total y por sesgo emocional.

- **Verificación:** `metrics.json` cumple objetivos; under-triage por sesgo < 10%.

### Iteración 6 — Tests integración y E2E (2 días)

- [ ] `tests/integration/` (pytest-docker).
- [ ] `tests/e2e/test_fase1_audio.py` (sube .wav real → polling hasta `Estado=AUDITADO`).
- [ ] `tests/e2e/test_fase2_texto.py` (POST a webhook n8n → triage correcto en ≥80% de casos de control).
- [ ] `tests/e2e/test_reintentos.py` (parar Ollama, reintento, recuperación).
- [ ] `tests/e2e/test_diccionario.py` ("Me ahogo" → `["disnea"]`).
- **Verificación:** `pytest tests/` al 100%.

### Iteración 7 — Documentación, MVP final y presentación (2 días)

- [ ] `README.md` completo.
- [ ] `docs/arquitectura.md` + diagrama.
- [ ] `docs/pipeline-fase1.md`, `pipeline-fase2.md`.
- [ ] `docs/gestion-errores.md`.
- [ ] `docs/modelo-ml.md`.
- [ ] `docs/auditoria-etica.md`.
- [ ] `services/streamlit-mvp/app.py` final.
- [ ] `docs/presentacion.pptx` (8–10 slides).
- **Verificación:** demo end-to-end (audio → triage Manchester + auditoría).

---

## Verificación end-to-end (criterio de "hecho" global)

1. `docker compose up -d` levanta los ~12 servicios sin error.
2. `python scripts/download_fareez.py && python scripts/seed_postgres.py` ingesta los 272 casos.
3. `dag_audio_ingestion` en verde para los 272 casos.
4. `dag_model_training` produce modelo con **Recall(C1) ≥ 0.85, Recall(C2) ≥ 0.80, F1-macro ≥ 0.70**.
5. `dag_audit_ethics` detecta los casos de under-triage por sesgo (ej. el `SIM_G1_01` del PDF).
6. Streamlit `:8501` permite subir un `.wav` real, devuelve triage Manchester con color + justificación + score de ansiedad + alerta de auditoría si procede.
7. Webhook n8n responde a peticiones POST con audio en <30 s.
8. `pytest tests/` pasa en local.
9. Demo de errores: parar Ollama → DAG reintenta → notificación llega.
10. Presentación 8–10 slides exportada a PDF.

---

## Riesgos y mitigaciones

- **Latencia Ollama local (5–30 s)** — batch en `dag_llm_enrichment`, fallback `mistral`, caché de prompts.
- **Whisper en CPU lento** — `faster-whisper` modelo `base` (39M); GPU opcional vía `deploy.resources`.
- **Desbalance extremo CAR/GAS (5/6 casos)** — SMOTE + class_weight; aumento sintético con LLM si es necesario.
- **Bilingüe EN/ES** — embeddings multilingües y prompts con ejemplos en ambos idiomas.
- **Under-triage por sesgo emocional** — riesgo clínico real, mitigado por `audit-ethics` + revisión humana.
- **Persistencia de modelos** — volumen Docker named `minio_data`.
- **Doble complejidad n8n + Airflow** — n8n estrictamente para webhooks/notificaciones; toda la lógica batch en Airflow.
