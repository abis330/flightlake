"""PX4 ULog adapter (real autopilot logs, e.g. from https://review.px4.io).

PX4 renames topics and fields across releases (``ekf2_innovations`` ->
``estimator_innovations``, ``vehicle_gps_position.lat`` (int 1e7) -> ``sensor_gps.latitude_deg``
...). The mapping below lists *candidates* per canonical channel and takes the first that
exists, so the same code ingests logs from several firmware generations.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..rawflight import RawFlight, empty_messages, quat_to_euler_deg

ELEVATOR_MAX_DEG = 25.0  # normalized [-1, 1] actuator control -> degrees (per airframe!)

# topic candidates -> {canonical: [(field, scale), ...candidates]}
SPEC: list[tuple[list[str], dict[str, list[tuple[str, float]]]]] = [
    (["vehicle_angular_velocity"], {
        "p_dps": [("xyz[0]", np.degrees(1))], "q_dps": [("xyz[1]", np.degrees(1))],
        "r_dps": [("xyz[2]", np.degrees(1))]}),
    (["sensor_combined"], {
        "ax_mps2": [("accelerometer_m_s2[0]", 1.0)], "ay_mps2": [("accelerometer_m_s2[1]", 1.0)],
        "az_mps2": [("accelerometer_m_s2[2]", 1.0)],
        "p_dps": [("gyro_rad[0]", np.degrees(1))], "q_dps": [("gyro_rad[1]", np.degrees(1))],
        "r_dps": [("gyro_rad[2]", np.degrees(1))]}),
    (["vehicle_local_position"], {"alt_rel_m": [("z", -1.0)], "vz_mps": [("vz", -1.0)]}),
    (["vehicle_gps_position", "sensor_gps"], {
        "lat_deg": [("latitude_deg", 1.0), ("lat", 1e-7)], "lon_deg": [("longitude_deg", 1.0), ("lon", 1e-7)],
        "alt_msl_m": [("altitude_msl_m", 1.0), ("alt", 1e-3)], "gnss_fix": [("fix_type", 1.0)],
        "gnss_nsat": [("satellites_used", 1.0)], "gnss_hdop": [("hdop", 1.0)],
        "groundspeed_mps": [("vel_m_s", 1.0)]}),
    (["airspeed_validated", "airspeed"], {"airspeed_mps": [("indicated_airspeed_m_s", 1.0)]}),
    (["control_state"], {"airspeed_mps": [("airspeed", 1.0)]}),
    (["battery_status"], {
        "batt_v": [("voltage_filtered_v", 1.0), ("voltage_v", 1.0)], "batt_a": [("current_a", 1.0)],
        "batt_soc": [("remaining", 1.0)], "batt_temp_c": [("temperature", 1.0)]}),
    (["actuator_controls_0"], {
        "throttle": [("control[3]", 1.0)], "elevator_cmd_deg": [("control[1]", ELEVATOR_MAX_DEG)]}),
    (["vehicle_thrust_setpoint"], {"throttle": [("xyz[0]", 1.0)]}),
    (["esc_status"], {f"motor{i + 1}_rpm": [(f"esc[{i}].esc_rpm", 1.0)] for i in range(4)}),
    (["vehicle_global_position"], {"ekf_hacc_m": [("eph", 1.0)]}),
    (["vehicle_imu_status"], {"vibe_mps2": [("accel_vibration_metric", 1.0)]}),
]


def _first(data: dict, cands):
    for f, scale in cands:
        if f in data:
            return np.asarray(data[f], dtype=float) * scale
    return None


def load_ulog(path: str | Path, **_) -> RawFlight:
    from pyulog import ULog  # optional dependency for real PX4 logs

    p = Path(path)
    ulog = ULog(str(p))
    t0 = ulog.start_timestamp
    by_name: dict[str, dict] = {}
    for d in ulog.data_list:
        if d.multi_id == 0:  # primary instance only
            by_name[d.name] = d.data
    topics, stats = {}, {}
    taken: set[str] = set()
    for cands, fields in SPEC:
        name = next((c for c in cands if c in by_name), None)
        if name is None:
            continue
        data = by_name[name]
        out = {"t_s": (np.asarray(data["timestamp"], dtype=float) - t0) / 1e6}
        for canon, fc in fields.items():
            if canon in taken:
                continue
            v = _first(data, fc)
            if v is not None:
                out[canon] = v
        if len(out) > 1:
            taken.update(k for k in out if k != "t_s")
            topics[name] = pd.DataFrame(out)
            stats[name] = {"rows": len(out["t_s"]), "non_monotonic": int((np.diff(out["t_s"]) < 0).sum())}

    if "vehicle_attitude" in by_name:
        a = by_name["vehicle_attitude"]
        r, pch, y = quat_to_euler_deg(a["q[0]"], a["q[1]"], a["q[2]"], a["q[3]"])
        topics["vehicle_attitude"] = pd.DataFrame(
            {"t_s": (np.asarray(a["timestamp"], float) - t0) / 1e6, "roll_deg": r, "pitch_deg": pch, "yaw_deg": y})
    for name in ("ekf2_innovations", "estimator_innovations"):
        if name in by_name:
            e = by_name[name]
            t = (np.asarray(e["timestamp"], float) - t0) / 1e6
            if "vel_pos_innov[3]" in e:  # old: [vn, ve, vd, pn, pe, pd]
                pos = np.hypot(e["vel_pos_innov[3]"], e["vel_pos_innov[4]"])
                vel = np.hypot(e["vel_pos_innov[0]"], e["vel_pos_innov[1]"])
            elif "gps_hpos[0]" in e:
                pos = np.hypot(e["gps_hpos[0]"], e["gps_hpos[1]"])
                vel = np.hypot(e["gps_hvel[0]"], e["gps_hvel[1]"])
            else:
                continue
            topics[name] = pd.DataFrame({"t_s": t, "ekf_pos_innov_m": pos, "ekf_vel_innov_mps": vel})
            break

    msgs = [((m.timestamp - t0) / 1e6, m.log_level_str(), m.message) for m in ulog.logged_messages]
    messages = pd.DataFrame(msgs, columns=["t_s", "level", "text"]) if msgs else empty_messages()
    info = ulog.msg_info_dict
    meta = {
        "flight_id": p.stem, "source": "px4_ulog", "raw_path": str(p),
        "airframe": str(info.get("sys_name", "PX4")), "firmware": str(info.get("ver_sw", "unknown")),
        "hardware": str(info.get("ver_hw", "unknown")),
        "duration_s": float((ulog.last_timestamp - t0) / 1e6),
        "faults": [],
    }
    return RawFlight(p.stem, topics, messages, meta, None, stats)
