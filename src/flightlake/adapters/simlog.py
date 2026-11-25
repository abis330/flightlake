"""Adapter for LAT flight-computer style logs (as produced by ``flightlake.sim.writer``)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..rawflight import RawFlight, empty_messages, quat_to_euler_deg

MSG_RE = re.compile(r"^\[(\d+)\]\s+(\w+):\s+(.*)$")


def _topic_map(name: str, d: pd.DataFrame) -> pd.DataFrame:
    """Firmware-native fields/units -> canonical channels."""
    o = pd.DataFrame({"t_s": d["t_s"]})
    if name == "imu":
        o["p_dps"], o["q_dps"], o["r_dps"] = (np.degrees(d[c]) for c in ("gyro_x_rad", "gyro_y_rad", "gyro_z_rad"))
        o["ax_mps2"], o["ay_mps2"], o["az_mps2"] = d["acc_x"], d["acc_y"], d["acc_z"]
    elif name == "attitude":
        o["roll_deg"], o["pitch_deg"], o["yaw_deg"] = quat_to_euler_deg(d.q0, d.q1, d.q2, d.q3)
    elif name == "actuators":
        o["throttle"] = d["throttle"]
        o["elevator_cmd_deg"] = np.degrees(d["elevator_cmd_rad"])
        o["elevator_pos_deg"] = np.degrees(d["elevator_pos_rad"])
    elif name == "local_position":
        o["alt_rel_m"], o["vz_mps"] = -d["z_m"], -d["vz_m_s"]  # NED down -> up
    elif name == "airspeed":
        o["airspeed_mps"] = d["indicated_airspeed_m_s"]
    elif name == "esc":
        for i in range(4):
            o[f"motor{i + 1}_rpm"] = d[f"esc{i}_rpm"]
    elif name == "estimator":
        o["ekf_pos_innov_m"], o["ekf_vel_innov_mps"] = d["gps_hpos_innov_m"], d["gps_vel_innov_m_s"]
        o["ekf_hacc_m"] = d["eph_m"]
    elif name == "gps":
        o["lat_deg"], o["lon_deg"] = d["lat_e7"] / 1e7, d["lon_e7"] / 1e7
        o["alt_msl_m"] = d["alt_mm"] / 1000.0
        o["gnss_fix"], o["gnss_nsat"], o["gnss_hdop"] = d["fix_type"], d["satellites_used"], d["hdop"]
        o["groundspeed_mps"] = d["vel_m_s"]
    elif name == "vibe":
        o["vibe_mps2"] = d["accel_vibration_metric"]
    elif name == "battery":
        o["batt_v"] = d["voltage_v"]
        # schema drift: legacy firmware logs integer centi-amps
        o["batt_a"] = d["current_a"] if "current_a" in d else d["current_ca"] / 100.0
        o["batt_soc"], o["batt_temp_c"] = d["remaining"], d["temperature_c"]
    elif name == "generator":
        o["gen_kw"] = d["power_w"] / 1000.0
    else:
        return pd.DataFrame()
    return o


def load_simlog(path: str | Path, **_) -> RawFlight:
    p = Path(path)
    meta = json.loads((p / "meta.json").read_text())
    raw = {f.stem: pd.read_csv(f) for f in sorted((p / "topics").glob("*.csv"))}
    t0 = meta.get("boot_time_us") or min(int(d.timestamp_us.min()) for d in raw.values())
    topics, stats = {}, {}
    for name, d in raw.items():
        ts = d["timestamp_us"].to_numpy()
        stats[name] = {"rows": int(len(d)), "non_monotonic": int((np.diff(ts) < 0).sum())}
        d = d.assign(t_s=(d["timestamp_us"] - t0) / 1e6)
        mapped = _topic_map(name, d)
        if not mapped.empty:
            topics[name] = mapped
    rows = []
    msg_file = p / "messages.log"
    if msg_file.exists():
        for line in msg_file.read_text().splitlines():
            m = MSG_RE.match(line.strip())
            if m:
                rows.append(((int(m.group(1)) - t0) / 1e6, m.group(2), m.group(3)))
    messages = pd.DataFrame(rows, columns=["t_s", "level", "text"]) if rows else empty_messages()
    labels = None
    lab = p / "_truth" / "labels.parquet"
    if lab.exists():
        labels = pd.read_parquet(lab)
    meta = dict(meta, source=meta.get("source", "sim"), raw_path=str(p))
    return RawFlight(meta["flight_id"], topics, messages, meta, labels, stats)
