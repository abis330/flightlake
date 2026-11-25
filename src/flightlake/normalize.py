"""Clean multi-rate topics and fuse them onto one uniform time grid, with a QC report.

Design rules
  * never silently invent data: interpolation is only allowed across gaps shorter than
    ``max_gap_s``; longer gaps stay NaN and are reported.
  * flag, don't hide: every removed sample (duplicate, out-of-range, spike) is counted.
  * spike removal uses a Hampel filter, which removes isolated glitches but preserves
    genuine step changes (a motor failing is a step, not a spike).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rawflight import RawFlight
from .schema import CHANNEL_BY_NAME, CHANNEL_NAMES, DEFAULT_RATE_HZ, SCHEMA_VERSION

# channel -> absolute deviation floor: a sample is only a "spike" if it is far from the local
# median in *physical* terms too (prevents flagging sensor noise on near-constant signals)
HAMPEL_CHANNELS = {"alt_rel_m": 8.0, "airspeed_mps": 4.0, "batt_v": 8.0, "alt_msl_m": 15.0, "groundspeed_mps": 4.0}
ANGLE_CHANNELS = {"yaw_deg"}


def hampel(x: pd.Series, window: int = 7, k: float = 6.0, floor: float = 0.0) -> tuple[pd.Series, int]:
    med = x.rolling(window, center=True, min_periods=1).median()
    mad = (x - med).abs().rolling(window, center=True, min_periods=1).median()
    thresh = np.maximum(k * 1.4826 * mad, floor)
    bad = (x - med).abs() > thresh
    bad &= x.notna()
    return x.mask(bad), int(bad.sum())


def clean_topic(name: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    st: dict = {"rows_in": int(len(df))}
    df = df.dropna(subset=["t_s"])
    df = df.sort_values("t_s", kind="stable")
    before = len(df)
    df = df.drop_duplicates(subset="t_s", keep="first")
    # near-duplicates: identical rows logged twice with jittered timestamps
    st["duplicates_removed"] = before - len(df)
    vals = df.drop(columns="t_s")
    dup_mask = vals.eq(vals.shift()).all(axis=1) & (df["t_s"].diff() < 0.01)
    st["duplicates_removed"] += int(dup_mask.sum())
    df = df[~dup_mask].reset_index(drop=True)

    st["out_of_range"], st["spikes_removed"] = {}, {}
    for c in df.columns:
        if c == "t_s" or c not in CHANNEL_BY_NAME:
            continue
        ch = CHANNEL_BY_NAME[c]
        oor = (df[c] < ch.lo) | (df[c] > ch.hi)
        if oor.any():
            st["out_of_range"][c] = int(oor.sum())
            df.loc[oor, c] = np.nan
        if c in HAMPEL_CHANNELS and len(df) > 20:
            df[c], n = hampel(df[c], floor=HAMPEL_CHANNELS[c])
            if n:
                st["spikes_removed"][c] = n
    dt = np.diff(df["t_s"].to_numpy())
    if len(dt):
        med = float(np.median(dt))
        gaps = dt > max(3 * med, 0.25)
        st.update(rate_hz=round(1 / med, 2) if med > 0 else None, gaps=int(gaps.sum()),
                  max_gap_s=round(float(dt.max()), 3), jitter_ms=round(float(np.std(dt[~gaps]) * 1e3), 3)
                  if (~gaps).any() else None)
    return df, st


def _interp(t_grid, t, y, max_gap_s, mode):
    ok = ~np.isnan(y)
    t, y = t[ok], y[ok]
    if len(t) < 2:
        return np.full_like(t_grid, np.nan, dtype=float)
    if mode == "ffill":
        idx = np.searchsorted(t, t_grid, side="right") - 1
        out = np.where(idx >= 0, y[np.clip(idx, 0, None)], np.nan)
        age = t_grid - t[np.clip(idx, 0, None)]
        out[(idx < 0) | (age > max_gap_s)] = np.nan
        return out
    out = np.interp(t_grid, t, y, left=np.nan, right=np.nan)
    # distance to nearest neighbours on both sides -> mask long gaps
    j = np.clip(np.searchsorted(t, t_grid), 1, len(t) - 1)
    span = t[j] - t[j - 1]
    out[span > max_gap_s] = np.nan
    return out


def normalize(raw: RawFlight, rate_hz: float = DEFAULT_RATE_HZ, max_gap_s: float = 1.0):
    """Return (wide telemetry DataFrame on a uniform grid, QC report dict)."""
    cleaned, topic_stats = {}, {}
    for name, df in raw.topics.items():
        cleaned[name], topic_stats[name] = clean_topic(name, df)
        topic_stats[name].update(raw.adapter_stats.get(name, {}))
    t_end = max((d["t_s"].max() for d in cleaned.values() if len(d)), default=0.0)
    t_grid = np.round(np.arange(0.0, t_end + 1e-9, 1.0 / rate_hz), 6)
    out = {"t_s": t_grid}
    for name, df in cleaned.items():
        t = df["t_s"].to_numpy(dtype=float)
        for c in df.columns:
            if c == "t_s" or c not in CHANNEL_BY_NAME or c in out:
                continue
            y = df[c].to_numpy(dtype=float, copy=True)
            if c in ANGLE_CHANNELS:  # interpolate on the unwrapped angle
                ok = ~np.isnan(y)
                y[ok] = np.degrees(np.unwrap(np.radians(y[ok])))
                v = _interp(t_grid, t, y, max_gap_s, "linear")
                out[c] = (v + 180) % 360 - 180
            else:
                out[c] = _interp(t_grid, t, y, max_gap_s, CHANNEL_BY_NAME[c].interp)
    wide = pd.DataFrame(out)
    for c in CHANNEL_NAMES:  # fixed column order; absent channels are all-NaN (schema-stable)
        if c not in wide:
            wide[c] = np.nan
    wide = wide[["t_s", *CHANNEL_NAMES]]
    wide.insert(0, "flight_id", raw.flight_id)
    return wide, qc_report(wide, topic_stats, rate_hz)


def _longest_flat_run_s(x: np.ndarray, rate_hz: float) -> float:
    if len(x) < 2:
        return 0.0
    same = np.concatenate([[False], np.diff(x) == 0]) & ~np.isnan(x)
    best = run = 0
    for s in same:
        run = run + 1 if s else 0
        best = max(best, run)
    return best / rate_hz


def qc_report(wide: pd.DataFrame, topic_stats: dict, rate_hz: float) -> dict:
    chans = {}
    present = 0
    for c in CHANNEL_NAMES:
        cov = float(wide[c].notna().mean())
        if cov > 0:
            present += 1
        chans[c] = {"coverage": round(cov, 4)}
        if cov > 0 and c not in ("gnss_fix", "gnss_nsat", "throttle", "gen_kw", "elevator_pos_deg"):
            flat = _longest_flat_run_s(wide[c].to_numpy(), rate_hz)
            if flat > 20:
                chans[c]["longest_flatline_s"] = round(flat, 1)
    present_cov = [v["coverage"] for v in chans.values() if v["coverage"] > 0]
    mean_cov = float(np.mean(present_cov)) if present_cov else 0.0
    n_spikes = sum(sum(s.get("spikes_removed", {}).values()) for s in topic_stats.values())
    n_oor = sum(sum(s.get("out_of_range", {}).values()) for s in topic_stats.values())
    n_gaps = sum(s.get("gaps", 0) for s in topic_stats.values())
    n_nonmono = sum(s.get("non_monotonic", 0) for s in topic_stats.values())
    score = 100.0 * mean_cov - min(10, 0.5 * n_gaps) - min(5, 0.05 * n_spikes) - min(5, 0.05 * n_oor) \
        - min(5, 0.5 * n_nonmono)
    return {
        "schema_version": SCHEMA_VERSION,
        "rate_hz": rate_hz,
        "n_samples": int(len(wide)),
        "channels_present": present,
        "channels_expected": len(CHANNEL_NAMES),
        "mean_coverage": round(mean_cov, 4),
        "gaps": n_gaps,
        "spikes_removed": n_spikes,
        "out_of_range": n_oor,
        "non_monotonic_timestamps": n_nonmono,
        "qc_score": round(max(0.0, score), 1),
        "topics": topic_stats,
        "channels": chans,
    }
