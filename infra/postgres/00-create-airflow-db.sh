#!/usr/bin/env bash
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE USER ${AIRFLOW_DB_USER:-airflow} WITH PASSWORD '${AIRFLOW_DB_PASSWORD:-airflow_pw}';
    CREATE DATABASE ${AIRFLOW_DB_NAME:-airflow_db} OWNER ${AIRFLOW_DB_USER:-airflow};
    GRANT ALL PRIVILEGES ON DATABASE ${AIRFLOW_DB_NAME:-airflow_db} TO ${AIRFLOW_DB_USER:-airflow};
EOSQL
