from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="smoke_test",
    description="Verifica que Airflow ejecuta DAGs. Reemplazar en Iteracion 3.",
    start_date=datetime(2026, 5, 1),
    schedule=None,
    catchup=False,
    tags=["triage", "smoke"],
) as dag:
    BashOperator(
        task_id="hello",
        bash_command='echo "TriageIA Airflow up. $(date -Iseconds)"',
    )
