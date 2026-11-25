"""The flight data lake: partitioned Parquet + a JSON summary per flight + a DuckDB SQL layer.

    lake/
      telemetry/flight_id=<id>/part-0.parquet   # uniform-rate wide table (+ phase_est)
      messages/flight_id=<id>/part-0.parquet    # autopilot text messages
      labels/flight_id=<id>/part-0.parquet      # ground truth when available (sim / labeled)
      flights/<id>.json                         # meta + QC + metrics + phase timeline + lineage
      catalog.parquet                           # one row per flight (rebuilt from flights/*.json)

Ingestion is idempotent: a content hash of the raw source is stored and re-ingesting an
unchanged log is a no-op.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import duckdb
import pandas as pd

from . import adapters
from .metrics import flight_metrics
from .normalize import normalize
from .phases import detect_phases, phase_segments
from .schema import SCHEMA_VERSION


def _hash_source(path: Path) -> str:
    h = hashlib.sha1()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file() and "_truth" not in p.parts)
    for f in files:
        h.update(f.name.encode())
        h.update(str(f.stat().st_size).encode())
        with open(f, "rb") as fh:
            h.update(fh.read(1 << 20))
    return h.hexdigest()[:16]


class Lake:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        for sub in ("telemetry", "messages", "labels", "flights"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ ingest
    def ingest(self, path: str | Path, adapter: str | None = None, mapping=None, rate_hz: float = 10.0,
               overwrite: bool = False) -> dict:
        path = Path(path)
        t0 = time.time()
        src_hash = _hash_source(path)
        raw = adapters.load(path, adapter, mapping=mapping)
        fid = raw.flight_id
        summary_path = self.root / "flights" / f"{fid}.json"
        if summary_path.exists() and not overwrite:
            old = json.loads(summary_path.read_text())
            if old.get("lineage", {}).get("source_hash") == src_hash:
                return {"flight_id": fid, "status": "skipped_unchanged"}
        wide, qc = normalize(raw, rate_hz=rate_hz)
        wide["phase_est"] = detect_phases(wide, rate_hz)
        metrics = flight_metrics(wide, raw.messages, rate_hz)
        self._write(fid, "telemetry", wide)
        msgs = raw.messages.assign(flight_id=fid)[["flight_id", "t_s", "level", "text"]]
        self._write(fid, "messages", msgs)
        if raw.labels is not None:
            self._write(fid, "labels", raw.labels.assign(flight_id=fid))
        summary = {
            "flight_id": fid,
            "meta": raw.meta,
            "qc": qc,
            "metrics": metrics,
            "phases": phase_segments(wide["t_s"].to_numpy(), wide["phase_est"].to_numpy()),
            "lineage": {"source_path": str(path), "source_hash": src_hash, "adapter": adapter or
                        adapters.detect_adapter(path), "schema_version": SCHEMA_VERSION, "rate_hz": rate_hz,
                        "ingested_at": pd.Timestamp.now("UTC").isoformat(),
                        "ingest_seconds": round(time.time() - t0, 3)},
        }
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        return {"flight_id": fid, "status": "ingested", "qc_score": qc["qc_score"],
                "seconds": round(time.time() - t0, 2)}

    def ingest_many(self, paths, workers: int | None = None, **kw) -> list[dict]:
        paths = [str(p) for p in paths]
        workers = workers or max(1, (os.cpu_count() or 2) - 1)
        if workers == 1 or len(paths) < 2:
            res = [self.ingest(p, **kw) for p in paths]
        else:
            with ProcessPoolExecutor(workers) as ex:
                res = list(ex.map(_ingest_one, [(str(self.root), p, kw) for p in paths]))
        self.rebuild_catalog()
        return res

    def _write(self, fid: str, table: str, df: pd.DataFrame):
        d = self.root / table / f"flight_id={fid}"
        d.mkdir(parents=True, exist_ok=True)
        if table == "telemetry":  # float32 is ample for sensor data; keep time & position in float64
            keep64 = {"t_s", "lat_deg", "lon_deg"}
            df = df.astype({c: "float32" for c in df.select_dtypes("float64").columns if c not in keep64})
        df.to_parquet(d / "part-0.parquet", index=False, compression="zstd")

    # ------------------------------------------------------------------ catalog
    def rebuild_catalog(self) -> pd.DataFrame:
        rows = []
        for f in sorted((self.root / "flights").glob("*.json")):
            s = json.loads(f.read_text())
            meta = s["meta"]
            row = {
                "flight_id": s["flight_id"], "source": meta.get("source"), "airframe": meta.get("airframe"),
                "firmware": meta.get("firmware"), "site": meta.get("site"), "date": meta.get("date"),
                "qc_score": s["qc"]["qc_score"], "channels_present": s["qc"]["channels_present"],
                "injected_faults": ",".join(x["type"] for x in meta.get("faults", [])) or "none",
                "fault_onset_s": (meta.get("faults") or [{}])[0].get("t_start"),
                "has_labels": (self.root / "labels" / f"flight_id={s['flight_id']}").exists(),
                "ingested_at": s["lineage"]["ingested_at"],
            }
            row.update({k: v for k, v in s["metrics"].items()})
            rows.append(row)
        cat = pd.DataFrame(rows)
        if len(cat):
            cat.to_parquet(self.root / "catalog.parquet", index=False)
        return cat

    def catalog(self) -> pd.DataFrame:
        p = self.root / "catalog.parquet"
        return pd.read_parquet(p) if p.exists() else self.rebuild_catalog()

    def summary(self, flight_id: str) -> dict:
        return json.loads((self.root / "flights" / f"{flight_id}.json").read_text())

    def flight_ids(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "flights").glob("*.json"))

    # ------------------------------------------------------------------ reads
    def _read(self, table, fid, columns=None):
        p = self.root / table / f"flight_id={fid}" / "part-0.parquet"
        if not p.exists():
            return None
        return pd.read_parquet(p, columns=columns)

    def telemetry(self, flight_id: str, channels: list[str] | None = None, t0: float | None = None,
                  t1: float | None = None) -> pd.DataFrame:
        cols = None
        if channels is not None:
            cols = ["t_s", "phase_est", *[c for c in channels if c not in ("t_s", "phase_est")]]
        df = self._read("telemetry", flight_id, cols)
        if df is None:
            raise KeyError(flight_id)
        if t0 is not None:
            df = df[df.t_s >= t0]
        if t1 is not None:
            df = df[df.t_s <= t1]
        return df.reset_index(drop=True)

    def messages(self, flight_id: str) -> pd.DataFrame:
        df = self._read("messages", flight_id)
        return df if df is not None else pd.DataFrame(columns=["flight_id", "t_s", "level", "text"])

    def labels(self, flight_id: str) -> pd.DataFrame | None:
        return self._read("labels", flight_id)

    # ------------------------------------------------------------------ SQL
    def connect(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect()
        r = self.root.as_posix()
        for table in ("telemetry", "messages", "labels"):
            if any((self.root / table).glob("*/part-0.parquet")):
                con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{r}/{table}/*/part-0.parquet', "
                            f"union_by_name=true, hive_partitioning=false)")
        if (self.root / "catalog.parquet").exists():
            con.execute(f"CREATE VIEW flights AS SELECT * FROM read_parquet('{r}/catalog.parquet')")
        return con

    def sql(self, query: str) -> pd.DataFrame:
        with self.connect() as con:
            return con.execute(query).df()


def _ingest_one(args):
    root, path, kw = args
    try:
        return Lake(root).ingest(path, **kw)
    except Exception as e:  # keep the batch going; report the failure
        return {"flight_id": Path(path).name, "status": "failed", "error": f"{type(e).__name__}: {e}"}
