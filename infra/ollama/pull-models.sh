#!/bin/sh
set -eu

OLLAMA_HOST_URL="${OLLAMA_HOST:-http://ollama:11434}"
MODELS="${OLLAMA_MODELS:-llama3,mistral}"

echo "[ollama-init] esperando a Ollama en ${OLLAMA_HOST_URL}..."
until curl -fsS "${OLLAMA_HOST_URL}/api/tags" >/dev/null 2>&1; do
  sleep 3
done
echo "[ollama-init] Ollama disponible."

OLD_IFS=$IFS; IFS=','
for model in $MODELS; do
  echo "[ollama-init] descargando modelo '${model}'..."
  curl -fsS -X POST "${OLLAMA_HOST_URL}/api/pull" \
       -H 'Content-Type: application/json' \
       -d "{\"name\":\"${model}\"}" \
       --no-buffer | tail -n 1
  echo "[ollama-init] modelo '${model}' listo."
done
IFS=$OLD_IFS

echo "[ollama-init] hecho. Modelos disponibles:"
curl -fsS "${OLLAMA_HOST_URL}/api/tags"
