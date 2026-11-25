import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flightlake.adapters import load
from flightlake.normalize import hampel, normalize
from flightlake.rawflight import quat_to_euler_deg
from flightlake.schema import CHANNEL_NAMES
from flightlake.sim.stol import FaultSpec, SimConfig, simulate
from flightlake.sqlguard import UnsafeSQL, guard

DATA = Path(__file__).parent / "data"


def test_quaternion_roundtrip():
    from flightlake.sim.writer import _euler_to_quat

    q = _euler_to_quat(np.array([10.0]), np.array([-5.0]), np.array([170.0]))
    r, p, y = quat_to_euler_deg(*q)
    assert np.allclose([r[0], p[0], y[0]], [10, -5, 170], atol=1e-6)


def test_hampel_removes_spike_keeps_step():
    x = pd.Series(np.r_[np.zeros(50), np.ones(50) * 10.0])
    x[20] = 500
    y, n = hampel(x, floor=2.0)
    assert n == 1 and np.isnan(y[20]) and y[60] == 10.0


def test_sim_is_deterministic_and_lands():
    a = simulate(SimConfig("A", seed=5))
    b = simulate(SimConfig("A", seed=5))
    assert a["truth"].equals(b["truth"])
    assert a["meta"]["landed"]
    assert set(a["truth"].phase) >= {"taxi", "takeoff_roll", "climb", "cruise", "approach", "landing_rollout"}


def test_fault_propagates_motor_failure():
    r = simulate(SimConfig("M", seed=2, faults=[FaultSpec("motor_failure", 250, None, {"motor": 3})]))
    tr = r["truth"]
    after = tr[(tr.t > 255) & (tr.phase == "cruise")]
    assert after.rpm3.max() < 50 and after.rpm1.mean() > 1000
    assert any("ESC 3" in m[2] for m in r["messages"])


def test_lake_ingest_schema_and_idempotency(fleet):
    lake = fleet["lake"]
    fid = lake.flight_ids()[0]
    tel = lake.telemetry(fid)
    assert list(tel.columns[:2]) == ["flight_id", "t_s"]
    assert set(CHANNEL_NAMES) <= set(tel.columns)
    assert np.allclose(np.diff(tel.t_s.values), 0.1, atol=1e-6)
    assert lake.ingest(fleet["paths"][0])["status"] == "skipped_unchanged"


def test_phase_estimate_matches_truth(fleet):
    lake = fleet["lake"]
    accs = []
    for fid in lake.flight_ids():
        m = lake.telemetry(fid).merge(lake.labels(fid), on="t_s")
        accs.append((m.phase == m.phase_est).mean())
    assert np.mean(accs) > 0.9


def test_metrics_reasonable(fleet):
    cat = fleet["lake"].catalog()
    assert cat.takeoff_ground_roll_m.between(50, 400).all()
    assert (cat.qc_score > 70).all()
    assert set(cat.injected_faults) != {"none"}


def test_legacy_firmware_units(tmp_path):
    from flightlake.sim.writer import LEGACY_FIRMWARE, write_flight

    sim = simulate(SimConfig("L", seed=1, firmware=LEGACY_FIRMWARE, cruise_duration_s=60))
    p = write_flight(sim, tmp_path, np.random.default_rng(0))
    assert "current_ca" in pd.read_csv(p / "topics" / "battery.csv").columns
    w, _ = normalize(load(p))
    assert w.batt_a.max() < 1500  # converted from centi-amps


def test_sql_and_guard(fleet):
    lake = fleet["lake"]
    df = lake.sql(guard("select flight_id, max(alt_rel_m) as h from telemetry group by 1"))
    assert len(df) == len(lake.flight_ids())
    for bad in ["drop table flights", "select 1; select 2", "select * from read_csv('/etc/passwd')",
                "COPY flights TO 'x.csv'"]:
        with pytest.raises(UnsafeSQL):
            guard(bad)


def test_px4_ulog_adapter(tmp_path):
    pytest.importorskip("pyulog")
    from flightlake.lake import Lake

    lake = Lake(tmp_path / "lake")
    res = lake.ingest(DATA / "sample_log_small.ulg")
    assert res["status"] == "ingested"
    s = lake.summary(res["flight_id"])
    assert s["qc"]["channels_present"] >= 8
    assert s["meta"]["source"] == "px4_ulog"


def test_api(fleet, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("FLIGHTLAKE_ROOT", str(fleet["root"] / "lake"))
    from flightlake import api

    api.lake.cache_clear()
    c = TestClient(api.app)
    assert c.get("/health").json()["flights"] == 4
    fid = c.get("/flights").json()[0]["flight_id"]
    tel = c.get(f"/flights/{fid}/telemetry", params={"channels": "airspeed_mps", "every": 50}).json()
    assert "airspeed_mps" in tel[0]
    assert c.post("/sql", json={"query": "delete from flights"}).status_code == 400
    assert json.dumps(c.get(f"/flights/{fid}").json())
