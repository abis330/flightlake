"""Generic adapter for a directory of per-topic CSV files (ROS bag exports, ArduPilot
``mavlogdump --format csv``, the CMU AirLab ALFA fault dataset, flight-test DAQ dumps ...).

The mapping lives in YAML so new sources are onboarded by config, not code::

    flight_id_from: dirname           # or 'parent'
    time: {column: "%time", scale: 1.0e-9}   # default time column + scale to seconds
    files:
      "*airspeed*.csv":
        airspeed_mps: {column: field.measured}
      "*battery*.csv":
        batt_v: {column: field.voltage}
        batt_a: {column: field.current, scale: -1.0}
"""
from __future__ import annotations

import fnmatch
from pathlib import Path

import pandas as pd
import yaml

from ..rawflight import RawFlight, empty_messages


def load_csvdir(path: str | Path, mapping: str | Path | dict | None = None, **_) -> RawFlight:
    p = Path(path)
    if mapping is None:
        raise ValueError("csvdir adapter needs a mapping YAML (see configs/alfa_mapping.example.yaml)")
    spec = mapping if isinstance(mapping, dict) else yaml.safe_load(Path(mapping).read_text())
    tdef = spec.get("time", {"column": "timestamp", "scale": 1.0})
    files = sorted(p.glob("*.csv"))
    topics, t_min = {}, None
    for pattern, cols in spec.get("files", {}).items():
        for f in files:
            if not fnmatch.fnmatch(f.name, pattern):
                continue
            d = pd.read_csv(f)
            tcol = cols.get("_time", tdef)
            t = d[tcol["column"]].astype(float) * float(tcol.get("scale", 1.0))
            out = {"t_s": t}
            for canon, c in cols.items():
                if canon.startswith("_") or c["column"] not in d:
                    continue
                out[canon] = d[c["column"]].astype(float) * float(c.get("scale", 1.0)) + float(c.get("offset", 0.0))
            if len(out) > 1:
                topics[f.stem] = pd.DataFrame(out)
                t_min = t.min() if t_min is None else min(t_min, t.min())
    for df in topics.values():
        df["t_s"] -= t_min
    fid = p.name if spec.get("flight_id_from", "dirname") == "dirname" else p.parent.name
    meta = {"flight_id": fid, "source": "csvdir", "raw_path": str(p), "airframe": spec.get("airframe", "unknown"),
            "firmware": spec.get("firmware", "unknown"), "faults": []}
    return RawFlight(fid, topics, empty_messages(), meta, None, {})
