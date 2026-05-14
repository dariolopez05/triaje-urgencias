#!/bin/sh
set -eu

ALIAS=triage
ENDPOINT="${MINIO_ENDPOINT:-http://minio:9000}"
USER="${MINIO_ROOT_USER:?MINIO_ROOT_USER no definido}"
PASS="${MINIO_ROOT_PASSWORD:?MINIO_ROOT_PASSWORD no definido}"
BUCKETS="${MINIO_BUCKETS:-audio-original,textos-originales,datasets,modelos}"

echo "[bootstrap] esperando a minIO en ${ENDPOINT}..."
until mc alias set "${ALIAS}" "${ENDPOINT}" "${USER}" "${PASS}" >/dev/null 2>&1; do
  sleep 2
done
echo "[bootstrap] minIO disponible."

OLD_IFS=$IFS; IFS=','
for bucket in $BUCKETS; do
  if mc ls "${ALIAS}/${bucket}" >/dev/null 2>&1; then
    echo "[bootstrap] bucket '${bucket}' ya existe."
  else
    mc mb "${ALIAS}/${bucket}"
    echo "[bootstrap] bucket '${bucket}' creado."
  fi
done
IFS=$OLD_IFS

echo "[bootstrap] hecho."
