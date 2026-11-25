"""REST query service over the lake (FastAPI).

Run:  FLIGHTLAKE_ROOT=data/lake uvicorn flightlake.api:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import math
import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from . import __version__
from .lake import Lake
from .schema import CHANNELS
from .sqlguard import UnsafeSQL, guard

app = FastAPI(title="flightlake", version=__version__,
              description="Query service for normalized aerospace flight telemetry")


@lru_cache(maxsize=1)
def lake() -> Lake:
    return Lake(os.environ.get("FLIGHTLAKE_ROOT", "data/lake"))


def _records(df):
    df = df.astype(object).where(df.notna(), None)
    return [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()}
            for r in df.to_dict(orient="records")]


@app.get("/health")
def health():
    return {"status": "ok", "flights": len(lake().flight_ids()), "version": __version__}


@app.get("/schema")
def schema():
    return [c.__dict__ for c in CHANNELS]


@app.get("/flights")
def flights(fault: str | None = None, min_qc: float = 0.0, site: str | None = None, limit: int = 500):
    cat = lake().catalog()
    if fault:
        cat = cat[cat.injected_faults.str.contains(fault)]
    if site:
        cat = cat[cat.site == site]
    cat = cat[cat.qc_score >= min_qc]
    return _records(cat.head(limit))


@app.get("/flights/{flight_id}")
def flight(flight_id: str):
    try:
        return lake().summary(flight_id)
    except FileNotFoundError:
        raise HTTPException(404, f"unknown flight {flight_id}")


@app.get("/flights/{flight_id}/telemetry")
def telemetry(flight_id: str, channels: str = Query("", description="comma-separated channel names"),
              t0: float | None = None, t1: float | None = None, every: int = Query(1, ge=1)):
    ch = [c for c in channels.split(",") if c] or None
    try:
        df = lake().telemetry(flight_id, ch, t0, t1)
    except KeyError:
        raise HTTPException(404, f"unknown flight {flight_id}")
    return _records(df.iloc[::every])


@app.get("/flights/{flight_id}/messages")
def messages(flight_id: str, level: str | None = None):
    df = lake().messages(flight_id)
    if level:
        df = df[df.level.str.upper() == level.upper()]
    return _records(df)


class SQLRequest(BaseModel):
    query: str
    max_rows: int = 5000


@app.post("/sql")
def sql(req: SQLRequest):
    try:
        q = guard(req.query, req.max_rows)
        return _records(lake().sql(q))
    except UnsafeSQL as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # duckdb errors -> 400 with message (useful for LLM self-correction)
        raise HTTPException(400, f"{type(e).__name__}: {e}")
