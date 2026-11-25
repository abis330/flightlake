"""Flight-level performance & health metrics (the numbers flight-test engineers ask for)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _f(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else round(x, 3)


def flight_metrics(w: pd.DataFrame, messages: pd.DataFrame | None = None, rate_hz: float = 10.0) -> dict:
    dt = 1.0 / rate_hz
    ph = w["phase_est"].to_numpy()
    air = np.isin(ph, ["climb", "cruise", "descent", "approach"])
    m: dict = {"duration_s": _f(w["t_s"].max()), "airborne_s": _f(air.sum() * dt)}

    def col(c, mask=None):
        s = w[c] if mask is None else w.loc[mask, c]
        return s.dropna()

    roll = ph == "takeoff_roll"
    land = ph == "landing_rollout"
    m["takeoff_ground_roll_m"] = _f(col("groundspeed_mps", roll).sum() * dt) if roll.any() else None
    m["landing_ground_roll_m"] = _f(col("groundspeed_mps", land).sum() * dt) if land.any() else None
    if roll.any():
        m["liftoff_airspeed_mps"] = _f(col("airspeed_mps", roll).tail(5).max())
    t_lift = w.loc[air, "t_s"].min() if air.any() else None
    cruise = ph == "cruise"
    m["time_to_cruise_s"] = _f(w.loc[cruise, "t_s"].min() - t_lift) if cruise.any() and t_lift is not None else None
    m["max_alt_rel_m"] = _f(col("alt_rel_m").max())
    climb = ph == "climb"
    m["mean_climb_rate_mps"] = _f(col("vz_mps", climb).mean()) if climb.any() else None
    m["cruise_airspeed_mean_mps"] = _f(col("airspeed_mps", cruise).mean()) if cruise.any() else None
    m["cruise_airspeed_std_mps"] = _f(col("airspeed_mps", cruise).std()) if cruise.any() else None
    m["cruise_alt_std_m"] = _f(col("alt_rel_m", cruise).std()) if cruise.any() else None
    if cruise.any():
        resid = (w.loc[cruise, "airspeed_mps"] - w.loc[cruise, "groundspeed_mps"]).dropna()
        m["cruise_airspeed_minus_gs_std"] = _f(resid.std())
    p_kw = (w["batt_v"] * w["batt_a"]).fillna(0) / 1000.0
    m["energy_batt_kwh"] = _f(p_kw.clip(lower=0).sum() * dt / 3600)
    m["energy_gen_kwh"] = _f(w["gen_kw"].fillna(0).sum() * dt / 3600)
    m["min_batt_v"] = _f(col("batt_v").min())
    m["max_batt_a"] = _f(col("batt_a").max())
    m["max_batt_temp_c"] = _f(col("batt_temp_c").max())
    soc = col("batt_soc")
    m["soc_used"] = _f(soc.iloc[0] - soc.iloc[-1]) if len(soc) > 1 else None
    m["max_vibe_mps2"] = _f(col("vibe_mps2").max())
    m["p95_vibe_mps2"] = _f(col("vibe_mps2").quantile(0.95)) if col("vibe_mps2").size else None
    fix = w["gnss_fix"]
    m["gnss_outage_s"] = _f(((fix < 3) & air).sum() * dt) if fix.notna().any() else None
    m["max_ekf_pos_innov_m"] = _f(col("ekf_pos_innov_m").max())
    m["max_ekf_hacc_m"] = _f(col("ekf_hacc_m").max())
    e = (w["elevator_cmd_deg"] - w["elevator_pos_deg"])[air].dropna()
    m["elevator_tracking_rmse_deg"] = _f(np.sqrt((e**2).mean())) if len(e) else None
    rpm = w.loc[air, [f"motor{i}_rpm" for i in range(1, 5)]].dropna()
    if len(rpm):
        m["min_motor_rpm_airborne"] = _f(rpm.min().min())
        mean = rpm.mean(axis=1).replace(0, np.nan)
        m["max_motor_rpm_imbalance"] = _f(((rpm.max(axis=1) - rpm.min(axis=1)) / mean).quantile(0.99))
    if w["lat_deg"].notna().sum() > 2:
        lat, lon = np.radians(w["lat_deg"].dropna()), np.radians(w["lon_deg"].dropna())
        d = 6371.0 * np.hypot(np.diff(lat), np.diff(lon) * np.cos(lat[1:]))
        m["distance_km"] = _f(d.sum())
    if messages is not None and len(messages):
        lv = messages["level"].str.upper()
        m["n_warnings"] = int((lv == "WARNING").sum())
        m["n_errors"] = int(lv.isin(["ERROR", "CRITICAL"]).sum())
    return m
