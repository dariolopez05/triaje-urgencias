# TriageIA — Sistema de Triaje Manchester con LLM + Machine Learning

> Proyecto 3 · Curso de especialización IA-BD 25/26
> Equipos de 2 alumnos · 4 semanas (mayo 2026)

Sistema de procesamiento de texto/audio orquestado con **n8n + Apache Airflow** que toma la voz de un paciente, extrae síntomas mediante un LLM local (**Ollama**), los normaliza contra un diccionario clínico cerrado y predice un **nivel de triage Manchester (C1–C5)** con un modelo de Machine Learning, todo trazado mediante un identificador único (`GUID_Entrevista`) y almacenado en **Postgres + minIO/S3**.

Basado en el corpus **Fareez et al. (2022)** — 272 entrevistas OSCE publicadas en *Scientific Data (Nature)*.

---

## Arquitectura (vista rápida)

| Componente | Tecnología | Puerto |
|---|---|---|
| Orquestador batch | Apache Airflow 2.10 (LocalExecutor) | `8080` |
| Orquestador webhooks / notificaciones | n8n 1.74 | `5678` |
| Base de datos | Postgres 15 | `5432` |
| Almacén de objetos | minIO (S3-compatible) | API `9000`, consola `9001` |
| LLM local | Ollama (`llama3`, `mistral`) | `11434` |
| Transcripción audio → texto | faster-whisper (FastAPI) | `9100` |
| Dashboard MVP | Streamlit | `8501` |

Diagrama detallado y descripción de DAGs/flujos en [`docs/arquitectura.md`](docs/arquitectura.md) *(pendiente — Iteración 7)*.

---

## Quickstart

### 1. Requisitos

- Docker Engine ≥ 24 con Docker Compose v2
- 16 GB de RAM libres (Ollama + Whisper son los más pesados)
- ~10 GB de disco para los modelos LLM (`llama3` ~4.7 GB, `mistral` ~4.1 GB)

### 2. Configurar variables

```bash
cp .env.example .env
# Edita .env para ajustar contraseñas, AIRFLOW_UID (en Linux: `id -u`), modelos LLM...
```

> **AIRFLOW_UID:** en Linux pon el resultado de `id -u` (típicamente `1000`); en macOS/Windows deja `50000`.

### 3. Levantar el stack

```bash
docker compose up -d
```

La primera vez tarda varios minutos:
- Postgres ejecuta `infra/postgres/00-create-airflow-db.sh` + `01-init-triage.sql` (esquema completo).
- minIO arranca y `minio-bootstrap` crea los buckets (`audio-original`, `textos-originales`, `datasets`, `modelos`).
- Ollama arranca y `ollama-init` descarga los modelos (~9 GB en disco).
- Airflow migra su BD y crea el usuario admin.

Comprueba el estado con:

```bash
docker compose ps
```

### 4. Acceder a las interfaces

| Servicio | URL | Credenciales |
|---|---|---|
| Airflow | http://localhost:8080 | `admin / admin` (de `.env`) |
| n8n | http://localhost:5678 | `admin / admin` (de `.env`) |
| minIO consola | http://localhost:9001 | `minio_admin / minio_admin_pw` (de `.env`) |
| Streamlit MVP | http://localhost:8501 | — |
| Ollama API | http://localhost:11434/api/tags | — |
| Transcripción | http://localhost:9100/health | — |

### 5. Verificar que todo funciona

```bash
# Postgres: tablas creadas
docker compose exec postgres psql -U triage -d triage_db -c '\dt'

# minIO: buckets creados
docker compose exec minio-bootstrap mc ls triage/ 2>/dev/null || \
  docker compose run --rm minio-bootstrap

# Ollama: modelos descargados
curl -s http://localhost:11434/api/tags | python3 -m json.tool

# Airflow: DAG de humo
# (en la UI, activa y dispara `smoke_test`)

# Whisper: salud
curl -s http://localhost:9100/health
```

---

## Estructura del repositorio

```
proyecto_triage_6/
├── docker-compose.yml          # 12 servicios coordinados
├── .env.example                # plantilla de variables
├── infra/
│   ├── postgres/               # init.sql + script crear DB airflow
│   ├── minio/                  # bootstrap-buckets.sh
│   └── ollama/                 # pull-models.sh
├── services/                   # microservicios FastAPI
│   ├── transcripcion/          # Whisper (audio → texto) ✅ Iter 0
│   ├── streamlit-mvp/          # dashboard sanitario ✅ Iter 0 (placeholder)
│   ├── api-gateway-ingesta/    # ⏳ Iter 2
│   ├── llm-extraction/         # ⏳ Iter 2
│   ├── llm-normalization/      # ⏳ Iter 2
│   ├── llm-labeling/           # ⏳ Iter 2
│   ├── anxiety-score/          # ⏳ Iter 2
│   ├── ml-training/            # ⏳ Iter 2
│   ├── ml-prediction/          # ⏳ Iter 2
│   ├── evaluation/             # ⏳ Iter 2
│   ├── audit-ethics/           # ⏳ Iter 2 — under-triage detection
│   └── _common/                # contratos Pydantic, clientes
├── airflow/dags/               # DAGs del pipeline ⏳ Iter 3
├── n8n/workflows/              # webhooks Fase 2 ⏳ Iter 4
├── data/
│   ├── dictionaries/
│   │   └── manchester_terms.csv  # diccionario clínico cerrado ✅
│   ├── prompts/                # plantillas LLM ⏳ Iter 1
│   ├── samples/                # audios y textos de prueba
│   └── fareez_dataset/         # mirror del corpus ⏳ Iter 5
├── scripts/                    # download_fareez.py, seed_postgres.py ⏳ Iter 5
└── tests/{unit,integration,e2e}/
```

---

## Niveles Manchester

| Nivel | Color | Tiempo máx. | Descripción |
|---|---|---|---|
| C1 | 🔴 Rojo | 0 min | Emergencia |
| C2 | 🟠 Naranja | 10 min | Muy urgente |
| C3 | 🟡 Amarillo | 60 min | Urgente |
| C4 | 🟢 Verde | 120 min | Menos urgente |
| C5 | 🔵 Azul | 240 min | No urgente |

**Objetivo cuantitativo:** `Recall(C1) ≥ 0.85` y `Recall(C2) ≥ 0.80` sobre el conjunto de prueba.

---

## Plan de iteraciones

El plan completo (con estado actual de cada iteración) está en [`docs/plan.md`](docs/plan.md). Resumen:

| Iter | Contenido | Estado |
|---|---|---|
| 0 | Esqueleto e infraestructura (este README) | ✅ |
| 1 | Librería común, contratos Pydantic, diccionario, prompts LLM | ⏳ |
| 2 | Microservicios FastAPI (10) + Streamlit | ⏳ |
| 3 | DAGs Airflow (audio, texto, training, evaluación, auditoría) | ⏳ |
| 4 | Flujos n8n (webhook Fase 2, notificaciones) | ⏳ |
| 5 | Dataset Fareez et al. + entrenamiento modelo ML | ⏳ |
| 6 | Tests de integración y E2E | ⏳ |
| 7 | Documentación final + MVP Streamlit completo + presentación | ⏳ |

---

## Parar / limpiar

```bash
docker compose down                  # para los servicios, conserva datos
docker compose down -v               # ⚠️ borra también volúmenes (modelos, BD)
```
