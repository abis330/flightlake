"""Adapter output contract: a flight as a bag of per-topic tables in canonical units."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class RawFlight:
    flight_id: str
    # topic name -> DataFrame with a float column ``t_s`` (seconds since log start)
    # plus any subset of canonical channel columns (see schema.py), still at native rate.
    topics: dict[str, pd.DataFrame]
    messages: pd.DataFrame  # columns: t_s, level, text
    meta: dict = field(default_factory=dict)
    labels: pd.DataFrame | None = None  # optional ground truth: t_s, phase, fault
    # bookkeeping filled by adapters (e.g. raw row counts, non-monotonic timestamps)
    adapter_stats: dict = field(default_factory=dict)


def quat_to_euler_deg(q0, q1, q2, q3):
    """ZYX (yaw-pitch-roll) Euler angles in degrees from a Hamilton quaternion (w,x,y,z)."""
    q0, q1, q2, q3 = (np.asarray(a, dtype=float) for a in (q0, q1, q2, q3))
    roll = np.degrees(np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (q0 * q2 - q3 * q1), -1, 1)))
    yaw = np.degrees(np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2)))
    return roll, pitch, yaw


def empty_messages() -> pd.DataFrame:
    return pd.DataFrame({"t_s": pd.Series(dtype=float), "level": pd.Series(dtype=str),
                         "text": pd.Series(dtype=str)})
