"""Turn clean simulator truth into *messy, realistic* raw flight logs.

Each flight becomes a directory that looks like what you would pull off a flight
computer SD card after a test sortie:

    <flight_id>/
        meta.json                 # airframe, firmware, site, test card info
        messages.log              # free-text autopilot/status messages
        topics/<topic>.csv        # multi-rate topics with firmware-native units
        _truth/labels.parquet     # ground-truth phase + fault (sim only, never ingested as telemetry)

Messiness deliberately injected (the things real pipelines must survive):
  * arbitrary boot-time timestamp origin, per-sample jitter, out-of-order chunks
  * duplicated rows, telemetry dropouts (gaps of 0.5-4 s), single-sample spikes
  * firmware-dependent schema drift (old firmware logs battery current in centi-amps)
  * quaternion attitude, NED-down altitude, radians, 1e7-scaled integer lat/lon
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .stol import SITES, FaultSpec, SimConfig, simulate

TOPIC_RATES = {
    "imu": 50, "attitude": 25, "actuators": 25, "local_position": 10, "airspeed": 10,
    "esc": 10, "estimator": 10, "gps": 5, "vibe": 5, "battery": 2, "generator": 2,
}
LEGACY_FIRMWARE = "lat-fc 2.1.0"  # logs battery current as integer centi-amps


def _euler_to_quat(roll, pitch, yaw):
    r, p, y = (np.radians(a) / 2 for a in (roll, pitch, yaw))
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def _sample(truth: pd.DataFrame, rate: float) -> pd.DataFrame:
    step = max(1, int(round((1 / rate) / (truth.t.iloc[1] - truth.t.iloc[0]))))
    return truth.iloc[::step].reset_index(drop=True)


def _messify(df: pd.DataFrame, rng, spike_cols=(), dropout_p=0.35) -> pd.DataFrame:
    n = len(df)
    # dropouts
    if rng.random() < dropout_p and n > 200:
        for _ in range(rng.integers(1, 4)):
            start = int(rng.integers(50, n - 50))
            width = int(len(df) * rng.uniform(0.5, 4.0) / max(df.t.iloc[-1], 1))
            df = df.drop(df.index[start:start + max(width, 2)], errors="ignore")
    # duplicates
    dup = df.sample(frac=0.003, random_state=int(rng.integers(1e9)))
    df = pd.concat([df, dup]).sort_index(kind="stable")
    # spikes
    for c in spike_cols:
        idx = df.sample(frac=0.0008, random_state=int(rng.integers(1e9))).index
        df.loc[idx, c] = df.loc[idx, c] * rng.choice([-1, 5, 10])
    return df


def write_flight(sim: dict, out_dir: Path, rng=None) -> Path:
    rng = rng or np.random.default_rng(0)
    truth, meta = sim["truth"], sim["meta"]
    fdir = Path(out_dir) / meta["flight_id"]
    (fdir / "topics").mkdir(parents=True, exist_ok=True)
    (fdir / "_truth").mkdir(exist_ok=True)
    boot_us = int(rng.integers(20e6, 120e6))
    legacy = meta["firmware"] == LEGACY_FIRMWARE

    def ts(t):
        jitter = rng.normal(0, 1500, size=len(t))
        return (boot_us + t.to_numpy() * 1e6 + jitter).astype(np.int64)

    n = lambda k: rng.standard_normal(k)  # noqa: E731
    topics: dict[str, pd.DataFrame] = {}

    d = _sample(truth, TOPIC_RATES["imu"])
    k, vib = len(d), d.vibe.to_numpy() / math.sqrt(3)
    topics["imu"] = pd.DataFrame({
        "t": d.t, "gyro_x_rad": np.radians(d.p) + 0.003 * n(k) + 0.002 * vib * n(k),
        "gyro_y_rad": np.radians(d.q) + 0.003 * n(k) + 0.002 * vib * n(k),
        "gyro_z_rad": np.radians(d.r) + 0.003 * n(k),
        "acc_x": d.v_dot + 9.81 * np.sin(np.radians(d.pitch)) + vib * n(k),
        "acc_y": 0.15 * n(k) + vib * n(k),
        "acc_z": -9.81 * np.cos(np.radians(d.pitch)) / np.cos(np.radians(d.roll)).clip(0.5) + vib * n(k),
    })
    d = _sample(truth, TOPIC_RATES["attitude"])
    q = _euler_to_quat(d.roll + 0.2 * n(len(d)), d.pitch + 0.2 * n(len(d)), d.yaw)
    topics["attitude"] = pd.DataFrame({"t": d.t, "q0": q[0], "q1": q[1], "q2": q[2], "q3": q[3]})
    d = _sample(truth, TOPIC_RATES["actuators"])
    topics["actuators"] = pd.DataFrame({
        "t": d.t, "throttle": d.throttle.clip(0, 1), "elevator_cmd_rad": np.radians(d.elev_cmd),
        "elevator_pos_rad": np.radians(d.elev_pos) + 0.002 * n(len(d)),
    })
    d = _sample(truth, TOPIC_RATES["local_position"])
    topics["local_position"] = pd.DataFrame({
        "t": d.t, "z_m": -(d.h + 0.4 * n(len(d))), "vz_m_s": -(d.h_dot + 0.15 * n(len(d))),
    })
    d = _sample(truth, TOPIC_RATES["airspeed"])
    topics["airspeed"] = pd.DataFrame({
        "t": d.t, "indicated_airspeed_m_s": (d.v_ias_meas + 0.3 * n(len(d))).clip(0)})
    d = _sample(truth, TOPIC_RATES["esc"])
    topics["esc"] = pd.DataFrame({"t": d.t, **{
        f"esc{i}_rpm": (d[f"rpm{i + 1}"] + 15 * n(len(d))).clip(0).round() for i in range(4)}})
    d = _sample(truth, TOPIC_RATES["estimator"])
    topics["estimator"] = pd.DataFrame({
        "t": d.t, "gps_hpos_innov_m": d.pos_innov, "gps_vel_innov_m_s": d.vel_innov, "eph_m": d.hacc})
    d = _sample(truth, TOPIC_RATES["gps"]).copy()
    k = len(d)
    lat = d.lat + 1.5 * n(k) / 111_320
    lon = d.lon + 1.5 * n(k) / (111_320 * math.cos(math.radians(d.lat.iloc[0])))
    nofix = d.gnss_fix.to_numpy() == 0
    lat, lon = lat.mask(nofix).ffill(), lon.mask(nofix).ffill()  # receivers hold last position
    topics["gps"] = pd.DataFrame({
        "t": d.t, "lat_e7": (lat * 1e7).round().astype("int64"), "lon_e7": (lon * 1e7).round().astype("int64"),
        "alt_mm": ((d.alt_msl + 2.5 * n(k)) * 1000).round().astype("int64"),
        "fix_type": d.gnss_fix.astype(int), "satellites_used": d.gnss_nsat.astype(int),
        "hdop": d.gnss_hdop.round(2), "vel_m_s": (d.groundspeed + 0.1 * n(k)).clip(0),
    })
    d = _sample(truth, TOPIC_RATES["vibe"])
    topics["vibe"] = pd.DataFrame({"t": d.t, "accel_vibration_metric": d.vibe * (1 + 0.05 * n(len(d)))})
    d = _sample(truth, TOPIC_RATES["battery"])
    cur = d.batt_a + 1.0 * n(len(d))
    topics["battery"] = pd.DataFrame({
        "t": d.t, "voltage_v": d.batt_v + 0.3 * n(len(d)),
        **({"current_ca": (cur * 100).round().astype("int64")} if legacy else {"current_a": cur}),
        "remaining": d.soc.round(3), "temperature_c": d.batt_temp.round(1),
    })
    d = _sample(truth, TOPIC_RATES["generator"])
    topics["generator"] = pd.DataFrame({"t": d.t, "power_w": (d.gen_kw * 1000 + 200 * n(len(d))).clip(0)})

    spikes = {"airspeed": ["indicated_airspeed_m_s"], "local_position": ["z_m"], "battery": ["voltage_v"]}
    for name, df in topics.items():
        df = _messify(df, rng, spikes.get(name, ()))
        df.insert(0, "timestamp_us", ts(df.t))
        df = df.drop(columns="t")
        if name == "imu" and len(df) > 1000:  # out-of-order chunk (buffered writer flush)
            i = int(rng.integers(100, len(df) - 200))
            df.iloc[i:i + 50] = df.iloc[i:i + 50].iloc[::-1].to_numpy()
        df.to_csv(fdir / "topics" / f"{name}.csv", index=False, float_format="%.6g")

    noise_msgs = ["Logger: write latency high", "Param PWM_MAIN_RATE changed", "Telemetry link RSSI low",
                  "Mission: uploaded 7 items", "CPU load 78%"]
    msgs = list(sim["messages"])
    for _ in range(int(rng.integers(1, 5))):
        msgs.append((float(rng.uniform(0, truth.t.max())), "INFO", str(rng.choice(noise_msgs))))
    msgs.sort(key=lambda m: m[0])
    with open(fdir / "messages.log", "w") as f:
        for t, lvl, text in msgs:
            f.write(f"[{int(boot_us + t * 1e6)}] {lvl}: {text}\n")

    meta = dict(meta, boot_time_us=boot_us, source="sim")
    (fdir / "meta.json").write_text(json.dumps(meta, indent=2))
    lab = _sample(truth, 10)[["t", "phase", "fault"]].rename(columns={"t": "t_s"})
    lab["t_s"] = lab["t_s"].round(1)
    lab.to_parquet(fdir / "_truth" / "labels.parquet", index=False)
    return fdir


def random_config(i: int, rng, fault_rate: float = 0.65, date="2026-01-15") -> SimConfig:
    from ..schema import FAULT_TYPES

    alt = float(rng.uniform(300, 600))
    cruise = float(rng.uniform(240, 540))
    cfg = SimConfig(
        flight_id=f"FLT-{i:05d}", seed=int(rng.integers(1e9)), site=str(rng.choice(list(SITES))),
        firmware=LEGACY_FIRMWARE if rng.random() < 0.12 else "lat-fc 2.3.1",
        date=str(np.datetime64(date) + np.timedelta64(int(i // 4), "D")),
        cruise_alt_m=alt, cruise_airspeed=float(rng.uniform(40, 48)), cruise_duration_s=cruise,
        wind_speed=float(rng.uniform(0, 10)), wind_dir_deg=float(rng.uniform(0, 360)),
        turbulence=float(rng.uniform(0.4, 1.6)), runway_heading_deg=float(rng.choice([90.0, 270.0, 180.0])),
    )
    if rng.random() < fault_rate:
        ftype = str(rng.choice(FAULT_TYPES))
        climb_end = 30 + alt / 2.8
        t0 = float(rng.uniform(climb_end * 0.6, climb_end + cruise - 40))
        params, t_end = {}, None
        if ftype == "motor_failure":
            params = {"motor": int(rng.integers(1, 5))}
        elif ftype == "gnss_jamming":
            t_end = t0 + float(rng.uniform(30, 120))
        elif ftype == "gnss_spoofing":
            params = {"drift_mps": float(rng.uniform(1.0, 3.0)), "drift_dir_deg": float(rng.uniform(0, 360))}
        elif ftype == "battery_degradation":
            params = {"r_factor": float(rng.uniform(2.5, 4.0)), "ocv_drop_v": float(rng.uniform(6.0, 14.0))}
        elif ftype == "imu_vibration":
            params = {"factor": float(rng.uniform(3.0, 7.0))}
        elif ftype == "pitot_blockage":
            params = {"decay_mps2": float(rng.uniform(0.4, 1.0))}
        if ftype in ("pitot_blockage", "generator_dropout", "elevator_stuck"):
            t0 = max(t0, climb_end * 0.7)
        cfg.faults = [FaultSpec(ftype, round(t0, 1), None if t_end is None else round(t_end, 1), params)]
    return cfg


def _sim_and_write(args):
    cfg, out_dir = args
    return write_flight(simulate(cfg), Path(out_dir), np.random.default_rng(cfg.seed))


def generate_fleet(n: int, out_dir: str | Path, seed: int = 7, fault_rate: float = 0.65,
                   start_index: int = 0, workers: int = 1) -> list[Path]:
    """Simulate ``n`` flights. Configs are drawn sequentially, so results are identical for
    any number of workers."""
    rng = np.random.default_rng(seed)
    cfgs = [random_config(i, rng, fault_rate) for i in range(start_index, start_index + n)]
    jobs = [(c, str(out_dir)) for c in cfgs]
    if workers <= 1:
        return [_sim_and_write(j) for j in jobs]
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_sim_and_write, jobs))
