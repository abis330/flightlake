"""Heuristic flight-phase segmentation (fixed-wing / STOL).

Good enough to drive metrics on any log. A more rigorous, *learned* segmenter with
weak supervision and active learning lives in the ``aerolabel`` project.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _merge_short(labels: np.ndarray, min_len: int) -> np.ndarray:
    lab = labels.copy()
    changed = True
    while changed:
        changed = False
        starts = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
        ends = np.r_[starts[1:], len(lab)]
        for s, e in zip(starts, ends):
            if e - s < min_len and len(starts) > 1:
                fill = lab[s - 1] if s > 0 else lab[e]
                if fill != lab[s]:
                    lab[s:e] = fill
                    changed = True
                    break
    return lab


def detect_phases(w: pd.DataFrame, rate_hz: float = 10.0) -> np.ndarray:
    n = len(w)
    out = np.array(["taxi"] * n, dtype=object)
    alt = w["alt_rel_m"].interpolate(limit_direction="both").fillna(0).rolling(int(rate_hz), center=True,
                                                                                min_periods=1).median()
    gs = w["groundspeed_mps"].interpolate(limit_direction="both").fillna(0).to_numpy()
    vz = w["vz_mps"].interpolate(limit_direction="both").fillna(0).rolling(int(5 * rate_hz), center=True,
                                                                           min_periods=1).mean().to_numpy()
    airborne = (alt > 1.0).to_numpy()  # ~weight-off-wheels proxy (1 s median-filtered)
    if not airborne.any():
        out[gs > 5] = "takeoff_roll"
        return out
    air_idx = np.flatnonzero(airborne)
    lo, hi = air_idx[0], air_idx[-1]
    # takeoff roll: contiguous high ground speed run ending at lift-off
    i = lo - 1
    while i >= 0 and gs[i] > 5.0:
        out[i] = "takeoff_roll"
        i -= 1
    j = hi + 1
    while j < n and gs[j] > 4.0:
        out[j] = "landing_rollout"
        j += 1
    seg = np.where(vz[lo:hi + 1] > 1.0, "climb", np.where(vz[lo:hi + 1] < -1.0, "descent", "cruise"))
    out[lo:hi + 1] = seg
    cruise_idx = np.flatnonzero(out == "cruise")
    if len(cruise_idx):
        last_cruise = cruise_idx[-1]
        appr = (np.arange(n) > last_cruise) & (alt.to_numpy() < 160) & airborne
        out[appr] = "approach"
    return _merge_short(out, int(4 * rate_hz))


def phase_segments(t: np.ndarray, phases: np.ndarray) -> list[dict]:
    starts = np.flatnonzero(np.r_[True, phases[1:] != phases[:-1]])
    ends = np.r_[starts[1:], len(phases)] - 1
    return [{"phase": str(phases[s]), "t_start": float(t[s]), "t_end": float(t[e])} for s, e in zip(starts, ends)]
