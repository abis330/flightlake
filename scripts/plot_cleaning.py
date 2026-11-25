"""Figure: a raw firmware topic (jitter, spikes, dropout, duplicates) vs the cleaned 10 Hz channel.

python scripts/plot_cleaning.py --raw data/raw/FLT-00003 --lake data/lake --out docs/cleaning.png
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from flightlake import Lake

BLUE, ORANGE, INK, MUTED = "#257de0", "#f47230", "#1f1f1f", "#8a8a8a"
ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--lake", default="data/lake")
ap.add_argument("--topic", default="local_position")
ap.add_argument("--out", default="docs/cleaning.png")
a = ap.parse_args()

raw_dir = Path(a.raw)
meta = json.loads((raw_dir / "meta.json").read_text())
raw = pd.read_csv(raw_dir / "topics" / f"{a.topic}.csv")
raw["t_s"] = (raw.timestamp_us - meta["boot_time_us"]) / 1e6
raw["alt"] = -raw.z_m
tel = Lake(a.lake).telemetry(meta["flight_id"], ["alt_rel_m"])
qc = Lake(a.lake).summary(meta["flight_id"])["qc"]["topics"][a.topic]

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    ax.tick_params(colors=MUTED)
axes[0].plot(raw.t_s, raw.alt, color=ORANGE, lw=1)
axes[0].plot(tel.t_s, tel.alt_rel_m, color=BLUE, lw=2)
axes[0].set_ylim(-60, tel.alt_rel_m.max() * 1.4)
axes[0].set_title(f"{meta['flight_id']} altitude: raw topic (orange) vs cleaned 10 Hz channel (blue)", loc="left",
                  fontsize=10, color=INK)
axes[0].set_xlabel("time since boot (s)", color=INK)
axes[0].set_ylabel("altitude above home (m)", color=INK)
rows = [("rows in", qc["rows_in"]), ("duplicates removed", qc["duplicates_removed"]),
        ("out-of-range removed", sum(qc.get("out_of_range", {}).values())),
        ("spikes removed (Hampel)", sum(qc.get("spikes_removed", {}).values())),
        ("non-monotonic timestamps", qc.get("non_monotonic", 0)), ("gaps > 3x period", qc.get("gaps", 0)),
        ("timestamp jitter (ms)", qc.get("jitter_ms"))]
axes[1].axis("off")
axes[1].set_title(f"QC report: topic '{a.topic}'", loc="left", fontsize=10, color=INK)
for i, (k, v) in enumerate(rows):
    axes[1].text(0.0, 0.85 - i * 0.12, k, color=MUTED, fontsize=10, transform=axes[1].transAxes)
    axes[1].text(0.95, 0.85 - i * 0.12, str(v), color=INK, fontsize=10, ha="right", transform=axes[1].transAxes)
fig.tight_layout()
fig.savefig(a.out, dpi=150)
print(a.out)
