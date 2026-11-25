"""Airflow DAG: pick up new sortie logs dropped by the flight-test team and ingest them.

Deploy by copying this file to $AIRFLOW_HOME/dags and `pip install flightlake` in the
Airflow image. Raw logs land in RAW_DIR (e.g. synced from the ground station / S3 / GCS).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task

RAW_DIR = Path(os.environ.get("FLIGHTLAKE_RAW", "/data/raw"))
LAKE_DIR = Path(os.environ.get("FLIGHTLAKE_ROOT", "/data/lake"))


@dag(schedule="*/15 * * * *", start_date=datetime(2026, 1, 1), catchup=False, max_active_runs=1,
     default_args={"retries": 2, "retry_delay": timedelta(minutes=2)}, tags=["flight-data"])
def flightlake_ingest():
    @task
    def discover() -> list[str]:
        return [str(p) for p in sorted(RAW_DIR.iterdir()) if p.is_dir() or p.suffix == ".ulg"]

    @task
    def ingest(path: str) -> dict:
        from flightlake.lake import Lake

        return Lake(LAKE_DIR).ingest(path)  # idempotent: unchanged logs are skipped

    @task
    def catalog(results: list[dict]) -> dict:
        from flightlake.lake import Lake

        Lake(LAKE_DIR).rebuild_catalog()
        low_qc = [r for r in results if r.get("qc_score", 100) < 70]
        return {"ingested": sum(r["status"] == "ingested" for r in results), "low_qc": low_qc}

    catalog(ingest.expand(path=discover()))


flightlake_ingest()
