# Plan: Flujo completo del proyecto de triage médico (Manchester)

## Contexto

**Qué es el proyecto.** `proyecto_triage` es un sistema de IA que toma una conversación médico-paciente (formato OSCE) y le asigna una categoría del **Sistema Manchester de Triage** (C1-Rojo / C2-Naranja / C3-Amarillo / C4-Verde / C5-Azul, con tiempos de espera de 0 / 10 / 60 / 120 / 240 min respectivamente). El input es una conversación; la salida es la urgencia clínica.

**Materiales que ya existen.**
- `dataset/audio/` — 272 MP3 (conversaciones simuladas OSCE, ~11 min media)
- `dataset/text/` — 272 transcritos crudos ASR (mayúsculas, sin puntuación)
- `dataset/cleantext/` — 272 transcritos limpios con etiquetas `D:`/`P:` y puntuación
- `paper.pdf` — descripción Scientific Data del dataset (categorías: RES 78.7%, MSK 16.9%, GAS 2.2%, CAR 1.8%, DER 0.4%, GEN)
- `guia/pagina-1..4.html` — 4 páginas de Gemini con las reglas Manchester (renderizadas vía JS — **no extraíbles desde el HTML, hay que copiarlas a mano**)
- `CLAUDE.md` — documentación del repo

**Distinción crítica.** Las categorías RES/MSK/CAR/GAS/DER/GEN del nombre de archivo son *especialidades clínicas*, **no** son las clases Manchester. Las etiquetas C1-C5 hay que derivarlas del contenido de la conversación (síntomas, severidad, banderas rojas).

**Supuestos técnicos asumidos (ajustar si difieren).**
- Input principal: `cleantext/` (más limpio, más rápido de iterar; audio queda como bonus E2E al final).
- Approach: **híbrido LLM + reglas** — extracción estructurada con LLM, decisión Manchester con árbol codificado, LLM como fallback en casos ambiguos.
- Las etiquetas C1-C5 **no existen aún** — generarlas (anotación manual de subset + auto-etiquetado del resto con verificación) es parte del proyecto.
- Lenguaje: Python 3.11+, ecosistema clásico (`pandas`, `pydantic`, `openai`/`anthropic`/`google-genai`, `whisper`, `scikit-learn`).

---

## Flujo del proyecto — 8 fases

### Fase 0 · Bootstrap (medio día)
1. Crear `pyproject.toml` o `requirements.txt` con dependencias mínimas.
2. Crear estructura de código:
   ```
   src/
     ingest/       # carga del dataset
     features/     # extracción NER / banderas rojas
     manchester/   # reglas codificadas + árbol de decisión
     triage/       # clasificador final
     audio/        # pipeline ASR (Whisper)
     eval/         # métricas y análisis
   data/
     interim/      # CSVs intermedios
     processed/    # output etiquetado
     labels/       # ground truth C1-C5 anotado a mano
   notebooks/      # exploración
   ```
3. **Acción del usuario requerida:** copiar el contenido de las 4 páginas Gemini (`guia/pagina-*.html`) a un archivo de texto plano `docs/manchester_rules.md`. Sin esto la Fase 2 no avanza.

### Fase 1 · Ingestión y EDA (1 día)
4. Construir `src/ingest/dataset.py` que lea los 272 casos y genere `data/interim/index.csv` con columnas:
   `case_id, specialty (RES/MSK/...), case_num, audio_path, text_path, cleantext_path, word_count, duration_min_estimate`.
5. Validación de integridad: confirmar que los 3 directorios tienen los mismos 272 IDs.
6. EDA en `notebooks/01_eda.ipynb`: distribución de longitudes, palabras frecuentes por especialidad, ejemplos representativos por categoría.

### Fase 2 · Codificación de reglas Manchester (1-2 días)
7. A partir de `docs/manchester_rules.md` (provisto por el usuario), construir `src/manchester/rules.py` con:
   - `Discriminator` (clase: nombre, descripción, prioridad C1-C5, especialidad aplicable).
   - `FlowChart` por presentación clínica (dolor torácico, disnea, dolor abdominal, traumatismo, etc.).
   - `decide(features) -> ManchesterCategory` que recorre el árbol y devuelve C1-C5 + razón.
8. Cobertura mínima: un flowchart por especialidad principal del dataset (RES, MSK, CAR, GAS, DER). Validar manualmente con 5 casos representativos.

### Fase 3 · Extracción de features clínicas (2 días)
9. `src/features/extractor.py` — para cada `cleantext` extrae a `data/interim/features.parquet`:
   - Motivo de consulta (chief complaint, texto libre).
   - Edad y sexo del paciente.
   - Localización del dolor, calidad (sharp/dull/burning), severidad 0-10, irradiación.
   - Tiempo de evolución (onset).
   - Síntomas asociados (disnea, síncope, fiebre, vómitos, hemoptisis, dolor torácico, etc.).
   - Banderas rojas Manchester (lista cerrada derivada de Fase 2).
   - Antecedentes relevantes (factor riesgo cardiovascular, embarazo, inmunosupresión).
10. Implementación: prompt estructurado a un LLM (Gemini Flash o Claude Haiku por coste/272 llamadas) con salida JSON validada por `pydantic`. Caché en disco para no re-procesar.
11. QA: revisar 20 casos a mano, calcular tasa de extracción correcta por campo.

### Fase 4 · Anotación ground truth (2 días, manual)
12. Seleccionar 80 casos estratificados por especialidad (proporcional al dataset: ~60 RES, ~14 MSK, 2 GAS, 2 CAR, 1 DER, 1 GEN extra) para anotación humana.
13. Crear `data/labels/ground_truth.csv` con columnas `case_id, manchester_label (C1-C5), rationale`. Anotación realizada por el usuario aplicando manualmente las reglas Manchester sobre el transcrito.
14. Doble revisión recomendada de al menos 20 casos para estimar acuerdo inter-anotador (Cohen's kappa).

### Fase 5 · Clasificador de triage (1-2 días)
15. `src/triage/classifier.py` con dos modos:
    - **Modo reglas**: pasa `features` extraídas de Fase 3 al árbol de Fase 2 (`manchester.decide`).
    - **Modo LLM-aumentado**: si el árbol devuelve confianza baja o ambigüedad, hace una segunda llamada a LLM con `cleantext` completo + reglas Manchester + features extraídas → respuesta forzada a C1-C5 + razón.
16. Pipeline: `cleantext → extract_features → manchester.decide → (fallback LLM) → ManchesterCategory + rationale`.
17. Ejecutar sobre los 272 casos → `data/processed/predictions.csv`.

### Fase 6 · Evaluación (1 día)
18. `src/eval/metrics.py` sobre los 80 casos del ground truth:
    - Accuracy global y por clase.
    - F1 macro y por C1-C5.
    - Matriz de confusión.
    - **Error de seguridad clínica**: % de casos donde el modelo predice una categoría *menos urgente* que la real (under-triage, peligroso) vs *más urgente* (over-triage, aceptable).
19. Reporte en `notebooks/02_evaluation.ipynb` con análisis de errores: revisar los 10 peores fallos a mano e identificar patrones (¿la regla no cubre el caso? ¿la extracción falló? ¿ambigüedad genuina?).

### Fase 7 · Pipeline audio E2E opcional (1 día)
20. `src/audio/asr.py` con `openai-whisper` (modelo `medium` o `large-v3`) sobre los MP3 → texto.
21. Re-ejecutar el clasificador sobre la salida ASR (no `cleantext`) para 30 casos y medir degradación de accuracy. Esto valida que el sistema funcionaría en producción a partir de audio real.

### Fase 8 · Demo y entrega (medio día)
22. `app.py` con Streamlit o Gradio: el usuario pega un transcrito (o sube un MP3), el sistema devuelve C1-C5 + razón + features detectadas + tiempo de espera.
23. README final con: cómo ejecutar end-to-end, métricas obtenidas, limitaciones (dataset sesgado a respiratorio, sin signos vitales reales, conversación simulada no espontánea).

---

## Archivos críticos a crear

| Archivo | Propósito |
|---|---|
| `pyproject.toml` | Dependencias y entrypoints |
| `docs/manchester_rules.md` | **A copiar manualmente desde `guia/`** |
| `src/ingest/dataset.py` | Indexa los 272 casos |
| `src/manchester/rules.py` | Árbol de decisión Manchester codificado |
| `src/features/extractor.py` | LLM-based clinical NER → JSON |
| `src/triage/classifier.py` | Orquestador del pipeline |
| `src/eval/metrics.py` | Métricas clínicas (incluye under-triage) |
| `data/labels/ground_truth.csv` | 80 casos anotados a mano |
| `app.py` | Demo Streamlit |

## Verificación end-to-end

1. **Smoke test**: `python -m src.ingest.dataset` produce `index.csv` con 272 filas, 0 huérfanos.
2. **Reglas**: `pytest tests/test_manchester.py` con 10 casos sintéticos cubre todas las salidas C1-C5.
3. **Features**: extraer 5 casos a mano y comparar contra LLM → coincidencia ≥80% por campo.
4. **Triage end-to-end**: `python -m src.triage.classifier --case CAR0001` imprime categoría + razón coherente con el contenido.
5. **Evaluación**: accuracy global ≥70%, under-triage <15%, sobre los 80 casos anotados.
6. **Demo**: `streamlit run app.py` arranca y procesa un transcrito de ejemplo en <30s.

## Riesgos y bloqueos conocidos

- **Bloqueante:** las reglas Manchester en `guia/*.html` no son extraíbles automáticamente (JS-rendered). El usuario debe transcribirlas a `docs/manchester_rules.md` antes de Fase 2.
- **Dataset desbalanceado:** 78.7% respiratorio. Las clases minoritarias (CAR, GAS, DER) tendrán pocos ejemplos para evaluar — interpretar métricas por clase con cuidado.
- **Sin signos vitales reales:** las conversaciones OSCE no incluyen TA/FC/SatO2/temperatura. Manchester depende de estos para muchos discriminadores → el clasificador trabajará solo con síntomas reportados.
- **Anotación ground truth subjetiva:** un solo anotador → considerar doble revisión en subset.
- **Coste LLM:** ~272 extracciones + ~80 fallbacks. Con Gemini Flash o Claude Haiku el coste total estimado es <5 USD.
