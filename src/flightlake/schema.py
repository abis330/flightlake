"""Canonical telemetry schema.

Every adapter (simulator, PX4 ULog, ArduPilot/ROS CSV exports, ...) is mapped onto
this single wide, uniformly-sampled schema so downstream ML, analytics and agents
never have to care which autopilot or firmware version produced a log.
"""
from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = "1.0"
DEFAULT_RATE_HZ = 10.0


@dataclass(frozen=True)
class Channel:
    name: str
    unit: str
    lo: float  # physically plausible lower bound (for QC)
    hi: float  # physically plausible upper bound (for QC)
    interp: str = "linear"  # 'linear' for continuous signals, 'ffill' for discrete states
    group: str = "misc"
    desc: str = ""


CHANNELS: list[Channel] = [
    # --- navigation / air data
    Channel("lat_deg", "deg", -90, 90, group="nav", desc="GNSS latitude"),
    Channel("lon_deg", "deg", -180, 180, group="nav", desc="GNSS longitude"),
    Channel("alt_msl_m", "m", -500, 9000, group="nav", desc="GNSS altitude MSL"),
    Channel("alt_rel_m", "m", -50, 6000, group="nav", desc="Altitude above home (estimator)"),
    Channel("vz_mps", "m/s", -40, 40, group="nav", desc="Vertical speed, up positive"),
    Channel("groundspeed_mps", "m/s", 0, 120, group="nav"),
    Channel("airspeed_mps", "m/s", 0, 120, group="airdata", desc="Indicated airspeed"),
    # --- attitude & inertial
    Channel("roll_deg", "deg", -180, 180, group="attitude"),
    Channel("pitch_deg", "deg", -90, 90, group="attitude"),
    Channel("yaw_deg", "deg", -180, 180, group="attitude"),
    Channel("p_dps", "deg/s", -400, 400, group="inertial", desc="Body roll rate"),
    Channel("q_dps", "deg/s", -400, 400, group="inertial", desc="Body pitch rate"),
    Channel("r_dps", "deg/s", -400, 400, group="inertial", desc="Body yaw rate"),
    Channel("ax_mps2", "m/s^2", -80, 80, group="inertial"),
    Channel("ay_mps2", "m/s^2", -80, 80, group="inertial"),
    Channel("az_mps2", "m/s^2", -80, 80, group="inertial"),
    Channel("vibe_mps2", "m/s^2", 0, 60, group="inertial", desc="Accel vibration metric"),
    # --- controls & actuators
    Channel("throttle", "-", 0, 1, group="control"),
    Channel("elevator_cmd_deg", "deg", -35, 35, group="control"),
    Channel("elevator_pos_deg", "deg", -35, 35, group="control"),
    Channel("motor1_rpm", "rpm", 0, 12000, group="propulsion"),
    Channel("motor2_rpm", "rpm", 0, 12000, group="propulsion"),
    Channel("motor3_rpm", "rpm", 0, 12000, group="propulsion"),
    Channel("motor4_rpm", "rpm", 0, 12000, group="propulsion"),
    # --- hybrid-electric power
    Channel("batt_v", "V", 0, 1000, group="power"),
    Channel("batt_a", "A", -500, 1500, group="power", desc="Battery current, discharge positive"),
    Channel("batt_soc", "-", 0, 1, group="power"),
    Channel("batt_temp_c", "degC", -40, 120, group="power"),
    Channel("gen_kw", "kW", 0, 500, group="power", desc="Range-extender generator output"),
    # --- GNSS & estimator health
    Channel("gnss_fix", "-", 0, 6, interp="ffill", group="gnss", desc="0 none, 2 2D, 3 3D"),
    Channel("gnss_nsat", "-", 0, 60, interp="ffill", group="gnss"),
    Channel("gnss_hdop", "-", 0, 100, group="gnss"),
    Channel("ekf_pos_innov_m", "m", 0, 5000, group="estimator", desc="GNSS horizontal pos innovation"),
    Channel("ekf_vel_innov_mps", "m/s", 0, 200, group="estimator"),
    Channel("ekf_hacc_m", "m", 0, 10000, group="estimator", desc="Estimated horizontal accuracy"),
]

CHANNEL_BY_NAME = {c.name: c for c in CHANNELS}
CHANNEL_NAMES = [c.name for c in CHANNELS]

PHASES = [
    "taxi",
    "takeoff_roll",
    "climb",
    "cruise",
    "descent",
    "approach",
    "landing_rollout",
]

FAULT_TYPES = [
    "motor_failure",
    "gnss_jamming",
    "gnss_spoofing",
    "battery_degradation",
    "imu_vibration",
    "pitot_blockage",
    "elevator_stuck",
    "generator_dropout",
]

# Which subsystem "owns" each fault -- used by agents/labelers for routing.
FAULT_SUBSYSTEM = {
    "motor_failure": "propulsion",
    "battery_degradation": "power",
    "generator_dropout": "power",
    "gnss_jamming": "navigation",
    "gnss_spoofing": "navigation",
    "imu_vibration": "sensors",
    "pitot_blockage": "sensors",
    "elevator_stuck": "flight_controls",
}
