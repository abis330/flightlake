"""Hybrid-electric STOL flight simulator with fault injection.

Point-mass longitudinal/lateral model with first-order attitude dynamics, a 4-motor
distributed-electric-propulsion (DEP) system, a battery + range-extender generator,
GNSS/EKF health models, Dryden-like turbulence and closed-loop autopilot.

It is deliberately *not* a 6-DOF model: its purpose is to produce realistic-looking,
physically coherent multi-rate telemetry with *known ground truth* (flight phases,
fault onsets) so that data pipelines, ML models, LLMs and agents can be evaluated
quantitatively. Faults propagate through the closed loop (e.g. a blocked pitot makes
the autopilot add throttle and pitch down), which is what makes fault signatures
non-trivial.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

G = 9.81
RHO = 1.20
MASS = 2200.0  # kg
WING_AREA = 28.0  # m^2
T_MAX = 5000.0  # N, all 4 motors at full throttle
P_MAX_KW = 240.0  # electrical power at full throttle (all motors)
CD0, K_IND = 0.03, 0.05
N_MOTORS = 4

SITES = {  # plausible Indian test sites (lat, lon, field elevation m)
    "BLR-test": (13.1986, 77.7066, 915.0),
    "HYD-test": (17.2403, 78.4294, 545.0),
    "PNQ-test": (18.5821, 73.9197, 560.0),
    "JSA-test": (26.8839, 70.8650, 230.0),
}


@dataclass
class FaultSpec:
    type: str
    t_start: float
    t_end: float | None = None  # None -> persists to end of flight
    params: dict = field(default_factory=dict)

    def active(self, t: float) -> bool:
        return t >= self.t_start and (self.t_end is None or t < self.t_end)


@dataclass
class SimConfig:
    flight_id: str
    seed: int = 0
    site: str = "BLR-test"
    airframe: str = "LAT-STOL-H1"
    firmware: str = "lat-fc 2.3.1"
    date: str = "2026-01-15"
    dt: float = 0.02  # 50 Hz truth
    cruise_alt_m: float = 400.0
    cruise_airspeed: float = 45.0
    cruise_duration_s: float = 420.0
    wind_speed: float = 4.0
    wind_dir_deg: float = 270.0  # direction wind blows FROM
    turbulence: float = 1.0
    runway_heading_deg: float = 270.0
    faults: list[FaultSpec] = field(default_factory=list)


def _clip(x: float, lo: float, hi: float) -> float:
    """Scalar clip (np.clip on Python floats is ~20x slower inside the 50 Hz loop)."""
    return lo if x < lo else hi if x > hi else x


def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class _Gust:
    """First-order Gauss-Markov (Dryden-like) gust component."""

    def __init__(self, rng, sigma, tau, dt):
        self.rng, self.sigma, self.a, self.x = rng, sigma, math.exp(-dt / tau), 0.0
        self.b = sigma * math.sqrt(1 - self.a**2)

    def step(self):
        self.x = self.a * self.x + self.b * self.rng.standard_normal()
        return self.x


def simulate(cfg: SimConfig) -> dict:
    """Run the simulation. Returns dict(truth=DataFrame, messages=list, meta=dict)."""
    rng = np.random.default_rng(cfg.seed)
    dt = cfg.dt
    lat0, lon0, field_elev = SITES[cfg.site]
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))

    wind_to = math.radians(cfg.wind_dir_deg + 180.0)
    wind_e, wind_n = cfg.wind_speed * math.sin(wind_to), cfg.wind_speed * math.cos(wind_to)
    gust_w = _Gust(rng, 0.6 * cfg.turbulence, 2.0, dt)
    gust_phi = _Gust(rng, 1.2 * cfg.turbulence, 1.0, dt)
    gust_u = _Gust(rng, 0.5 * cfg.turbulence, 3.0, dt)

    faults = {f.type: f for f in cfg.faults}

    def fault(name, t):
        f = faults.get(name)
        return f if (f is not None and f.active(t)) else None

    # ---- state
    t = 0.0
    x = y = 0.0  # local east/north (m)
    h = 0.0  # height above field
    V = 0.0  # true airspeed (on ground: ground-relative speed along heading)
    psi = cfg.runway_heading_deg
    gamma = 0.0
    phi = 0.0
    theta = 0.0
    throttle = 0.0
    soc = float(rng.uniform(0.92, 0.99))
    batt_temp = float(rng.uniform(24, 32))
    motor_rpm = np.zeros(N_MOTORS)
    elev_pos = 0.0
    yaw_transient = 0.0
    v_int = 0.0
    spoof_e = spoof_n = 0.0
    dr_err = 0.0  # dead-reckoning error growth under jamming
    pitot_reading = None
    pitot_detected_t = None
    phase = "taxi"
    phase_t0 = 0.0
    taxi_dur = float(rng.uniform(12, 25))
    heading_target = cfg.runway_heading_deg
    next_wp_t = None
    wp_count = 0
    landed = False
    flags: dict[str, bool] = {}
    msgs: list[tuple[float, str, str]] = []
    rows = []

    def msg(level, text):
        msgs.append((t, level, text))

    msg("INFO", f"Boot: {cfg.firmware}, airframe {cfg.airframe}")
    msg("INFO", "Preflight checks passed")
    msg("INFO", "Armed")

    v_ias_meas_prev = 0.0
    max_t = 3600.0
    while t < max_t:
        # ---------------- faults affecting plant
        mf = fault("motor_failure", t)
        health = np.ones(N_MOTORS)
        if mf:
            health[mf.params.get("motor", 2) - 1] = 0.0
            if not flags.get("mf_onset"):
                flags["mf_onset"] = True
                yaw_transient = 7.0 * (1 if mf.params.get("motor", 2) <= 2 else -1)
            if t - mf.t_start > 0.6 and not flags.get("mf_msg"):
                flags["mf_msg"] = True
                msg("CRITICAL", f"ESC {mf.params.get('motor', 2)}: no telemetry / motor stopped")
        thrust_frac = health.mean()

        gd = fault("generator_dropout", t)
        bd = fault("battery_degradation", t)
        r_int = 0.06 * (1 + (bd.params.get("r_factor", 3.0) - 1) if bd else 1.0)

        # ---------------- airspeed measurement (pitot)
        pb = fault("pitot_blockage", t)
        v_true_ias = max(V - 0.0, 0.0)
        if pb and h > 5:
            if pitot_reading is None:
                pitot_reading = v_ias_meas_prev
            pitot_reading = max(0.0, pitot_reading - pb.params.get("decay_mps2", 0.6) * dt)
            v_meas = pitot_reading
            if pitot_detected_t is None and abs(v_meas - V) > 10:
                pitot_detected_t = t + 6.0  # monitor needs persistence
            if pitot_detected_t is not None and t >= pitot_detected_t:
                if not flags.get("pb_msg"):
                    flags["pb_msg"] = True
                    msg("WARNING", "Airspeed sensor failure detected, using synthetic airspeed")
                v_ctrl = V  # synthetic airspeed restores the loop
            else:
                v_ctrl = v_meas
        else:
            v_meas = v_true_ias
            v_ctrl = V
        v_ias_meas_prev = v_meas

        # ---------------- phase logic & guidance
        if phase == "taxi":
            v_target, gamma_cmd, throttle_ff = 4.0, 0.0, 0.12
            if t - phase_t0 > taxi_dur:
                phase, phase_t0 = "takeoff_roll", t
                psi = cfg.runway_heading_deg
                hw0 = -(wind_e * math.sin(math.radians(psi)) + wind_n * math.cos(math.radians(psi)))
                V = max(V + hw0, 0.0)  # switch reference from ground speed to airspeed
                msg("INFO", "Takeoff run started")
        elif phase == "takeoff_roll":
            v_target, gamma_cmd, throttle_ff = 99.0, 0.0, 1.0
            if V >= 21.0:
                phase, phase_t0 = "climb", t
                msg("INFO", "Takeoff detected")
        elif phase == "climb":
            v_target, throttle_ff = 30.0, 0.85
            gamma_cmd = math.radians(6.0) if h < cfg.cruise_alt_m - 30 else math.radians(2.0)
            if v_ctrl < 26:
                gamma_cmd *= max(0.0, (v_ctrl - 22) / 4)  # envelope protection
            if h >= cfg.cruise_alt_m - 5:
                phase, phase_t0 = "cruise", t
                next_wp_t = t + float(rng.uniform(50, 110))
                msg("INFO", f"Reached cruise altitude {cfg.cruise_alt_m:.0f} m")
        elif phase == "cruise":
            v_target, throttle_ff = cfg.cruise_airspeed, 0.34
            gamma_cmd = _clip(0.02 * (cfg.cruise_alt_m - h) / max(V, 10), -0.08, 0.08)
            if t >= next_wp_t:
                wp_count += 1
                heading_target = (heading_target + float(rng.choice([-1, 1]) * rng.uniform(40, 120))) % 360
                next_wp_t = t + float(rng.uniform(50, 110))
                msg("INFO", f"Waypoint {wp_count} reached, new course {heading_target:.0f} deg")
            if t - phase_t0 > cfg.cruise_duration_s:
                phase, phase_t0 = "descent", t
                msg("INFO", "Descent initiated")
        elif phase == "descent":
            v_target, throttle_ff = 36.0, 0.18
            gamma_cmd = math.radians(-4.0)
            heading_target = math.degrees(math.atan2(-x, -y)) % 360
            if h <= 150:
                phase, phase_t0 = "approach", t
                msg("INFO", "Approach: STOL steep glide path engaged")
        elif phase == "approach":
            v_target, throttle_ff = 24.0, 0.3
            gamma_cmd = math.radians(-6.0) if h > 4 else math.radians(-1.0)
            heading_target = cfg.runway_heading_deg
            if h <= 0.05 and t - phase_t0 > 2:
                phase, phase_t0 = "landing_rollout", t
                landed = True
                msg("INFO", "Landing detected")
        elif phase == "landing_rollout":
            v_target, gamma_cmd, throttle_ff = 0.0, 0.0, 0.0
            if V <= 3.0:
                msg("INFO", "Disarmed")
                break

        on_ground = phase in ("taxi", "takeoff_roll", "landing_rollout")

        # ---------------- elevator / pitch loop
        es = fault("elevator_stuck", t)
        gamma_eff_cmd = gamma_cmd
        elev_cmd = -2.0 + 25.0 * (gamma_cmd - gamma) * 3.0 - 0.15 * (V - 30.0) * 0.2
        elev_cmd = _clip(elev_cmd, -25, 25)
        gamma_tau = 1.5
        if es and not on_ground:
            if "es_pos" not in flags:
                flags["es_pos"] = elev_pos
            elev_pos = flags["es_pos"]
            # Elevator frozen: autopilot falls back to thrust-for-pitch with degraded authority.
            # Result: sluggish path tracking + lightly damped phugoid-like oscillation.
            ph = 2 * math.pi * (t - es.t_start) / 32.0
            gamma_eff_cmd = _clip(0.6 * gamma_cmd + math.radians(2.5 * math.sin(ph))
                                  + math.radians(0.08 * (elev_pos - elev_cmd)), -0.12, 0.12)
            gamma_tau = 6.0
            if abs(elev_cmd - elev_pos) > 4 and not flags.get("es_msg"):
                flags.setdefault("es_t", t)
                if t - flags["es_t"] > 3:
                    flags["es_msg"] = True
                    msg("WARNING", "Elevator actuator tracking error")
        else:
            elev_pos += (elev_cmd - elev_pos) * min(1.0, dt / 0.08)

        # ---------------- throttle loop (uses *measured* airspeed -> pitot faults propagate)
        if on_ground and phase != "takeoff_roll":
            throttle_cmd = throttle_ff if phase == "taxi" else 0.0
            if phase == "taxi":
                throttle_cmd = _clip(0.12 + 0.05 * (v_target - V), 0, 0.3)
        elif phase == "takeoff_roll":
            throttle_cmd = 1.0
        else:
            err = v_target - v_ctrl
            v_int = _clip(v_int + err * dt, -40, 40)
            throttle_cmd = throttle_ff / max(thrust_frac, 0.25) + 0.06 * err + 0.004 * v_int
            throttle_cmd = _clip(throttle_cmd, 0.0, 1.0)
            if pb and v_ctrl is not V:  # blocked pitot: autopilot believes it is slow
                gamma_eff_cmd = min(gamma_eff_cmd, math.radians(-2.5))
        throttle += (throttle_cmd - throttle) * min(1.0, dt / 0.3)

        # ---------------- propulsion & longitudinal dynamics
        thrust = T_MAX * throttle * thrust_frac
        q_dyn = 0.5 * RHO * V * V
        if on_ground:
            drag = q_dyn * WING_AREA * 0.06 + 0.03 * MASS * G * (1 if V > 0.1 else 0)
            brake = 0.0 if phase != "landing_rollout" else 0.28 * MASS * G
            V_dot = (thrust - drag - brake) / MASS
            if phase == "taxi":
                V_dot = 0.5 * (v_target - V)
            gamma = 0.0
            h = 0.0
        else:
            CL = MASS * G / max(q_dyn * WING_AREA, 1.0)
            CD = CD0 + K_IND * min(CL, 3.5) ** 2 * 0.25
            CD += {"descent": 0.02, "approach": 0.09}.get(phase, 0.0)  # flaps / high-lift config
            CD += _clip(0.01 * (v_ctrl - v_target), 0.0, 0.08)  # speed-brake / drag devices
            drag = q_dyn * WING_AREA * CD
            V_dot = (thrust - drag - MASS * G * math.sin(gamma)) / MASS + 0.3 * gust_u.step()
            gamma += (gamma_eff_cmd - gamma) * min(1.0, dt / gamma_tau)
        V = max(0.0, V + V_dot * dt)

        # ---------------- lateral dynamics
        if on_ground:
            phi = 0.0
            psi_dot = 0.0
        else:
            phi_cmd = _clip(1.2 * _wrap180(heading_target - psi), -25, 25)
            if mf:
                phi_cmd += 2.5 * (1 if mf.params.get("motor", 2) <= 2 else -1)
            phi += (phi_cmd - phi) * min(1.0, dt / 0.8) + gust_phi.step() * dt * 3
            psi_dot = math.degrees(G * math.tan(math.radians(phi)) / max(V, 10.0))
        yaw_transient *= math.exp(-dt / 2.5)
        psi = (psi + (psi_dot + yaw_transient) * dt) % 360

        # ---------------- kinematics
        w_gust = 0.0 if on_ground else gust_w.step()
        h_dot = V * math.sin(gamma) + w_gust
        if not on_ground:
            h = h + h_dot * dt
            if phase == "approach" and h < 0:
                h = 0.0
        gs_e = V * math.cos(gamma) * math.sin(math.radians(psi)) + (0 if on_ground else wind_e)
        gs_n = V * math.cos(gamma) * math.cos(math.radians(psi)) + (0 if on_ground else wind_n)
        if on_ground and phase != "taxi":
            # V is airspeed during the roll: headwind shortens ground roll (key STOL metric)
            headwind = -(wind_e * math.sin(math.radians(psi)) + wind_n * math.cos(math.radians(psi)))
            gs_along = max(V - headwind, 0.0)
            gs_e = gs_along * math.sin(math.radians(psi))
            gs_n = gs_along * math.cos(math.radians(psi))
        x += gs_e * dt
        y += gs_n * dt
        groundspeed = math.hypot(gs_e, gs_n)

        alpha = 0.0 if on_ground else _clip((MASS * G / max(q_dyn * WING_AREA, 1) - 0.45) / 0.12, -4, 12)
        theta_target = math.degrees(gamma) + alpha * (0 if on_ground else 1) + (3.0 if phase == "landing_rollout" else 0)
        theta_prev = theta
        theta += (theta_target - theta) * min(1.0, dt / 0.6)  # finite pitch dynamics (rotation, flare)
        q_rate = (theta - theta_prev) / dt
        p_rate = 0.0 if on_ground else (phi_cmd - phi) / 0.8

        # ---------------- motors & power
        target_rpm = (900 + 5200 * throttle) * health
        motor_rpm += (target_rpm - motor_rpm) * min(1.0, dt / 0.25)
        p_elec_kw = P_MAX_KW * (throttle**1.5) * thrust_frac + 2.0
        gen_kw = 0.0
        if phase in ("climb", "cruise") and not gd:
            # hybrid energy management: follow load and recharge toward 85% SOC
            gen_kw = _clip(p_elec_kw + 100.0 * (0.85 - soc), 0.0, 70.0)
        if gd and not flags.get("gd_msg") and phase in ("climb", "cruise"):
            flags["gd_msg"] = True
            msg("ERROR", "Generator offline: range extender fault, running on battery")
        ocv = 330.0 + 85.0 * soc - 6.0 * math.exp(-15 * soc)
        if bd:  # cell degradation: resistance rise + OCV depression (capacity fade / weak cell group)
            ocv -= bd.params.get("ocv_drop_v", 10.0) * min(1.0, (t - bd.t_start) / 30.0)
        i_batt = (p_elec_kw - gen_kw) * 1000.0 / max(ocv, 1.0)
        v_batt = ocv - i_batt * r_int
        soc -= (i_batt * v_batt) * dt / (30.0 * 3.6e6)  # 30 kWh pack
        soc = _clip(soc, 0, 1)
        batt_temp += (i_batt**2 * r_int * 0.00004 - (batt_temp - 28.0) * 0.004) * dt
        if v_batt < 350 and not flags.get("lowv"):
            flags["lowv"] = True
            msg("WARNING", f"Battery voltage low: {v_batt:.0f} V")
        if batt_temp > 55 and not flags.get("hott"):
            flags["hott"] = True
            msg("WARNING", f"Battery temperature high: {batt_temp:.0f} C")

        # ---------------- vibration
        vib = 0.4 + 1.1 * throttle
        if mf:
            vib += 0.6
        iv = fault("imu_vibration", t)
        if iv:
            vib *= iv.params.get("factor", 5.0)
            if vib > 4 and not flags.get("iv_msg"):
                flags.setdefault("iv_t", t)
                if t - flags["iv_t"] > 5:
                    flags["iv_msg"] = True
                    msg("WARNING", "High accelerometer vibration detected")

        # ---------------- GNSS & EKF
        gj = fault("gnss_jamming", t)
        gsf = fault("gnss_spoofing", t)
        nsat = int(_clip(round(14 + 2 * math.sin(t / 97.0) + rng.normal(0, 0.4)), 6, 30))
        hdop = 0.75 + 0.05 * rng.standard_normal()
        fix = 3
        pos_innov = abs(rng.normal(0, 0.8)) + 0.3
        vel_innov = abs(rng.normal(0, 0.15)) + 0.05
        hacc = 1.2 + 0.2 * abs(rng.standard_normal())
        if gj:
            nsat, fix, hdop = int(rng.integers(0, 3)), 0, 99.0
            dr_err += (0.4 + 0.2 * rng.random()) * dt
            hacc = 1.2 + 4.0 * dr_err
            pos_innov = float("nan")
            vel_innov = float("nan")
            if not flags.get("gj_msg"):
                flags["gj_msg"] = True
                msg("WARNING", "GNSS: fix lost (no satellites)")
        elif dr_err > 0:
            pos_innov = dr_err + abs(rng.normal(0, 1))  # reacquisition jump
            dr_err *= math.exp(-dt / 3)
            if not flags.get("gj_rec"):
                flags["gj_rec"] = True
                msg("INFO", "GNSS: 3D fix regained")
        if gsf:
            rate = gsf.params.get("drift_mps", 2.0)
            ang = math.radians(gsf.params.get("drift_dir_deg", 45.0))
            spoof_e += rate * math.sin(ang) * dt
            spoof_n += rate * math.cos(ang) * dt
            pos_innov = math.hypot(spoof_e, spoof_n) * 0.6 + abs(rng.normal(0, 0.8))
            vel_innov = rate * 0.8 + abs(rng.normal(0, 0.2))
            nsat = min(nsat + 3, 30)  # spoofers often present strong, many signals
            if pos_innov > 15 and not flags.get("gs_msg"):
                flags["gs_msg"] = True
                msg("WARNING", "EKF: GPS position test ratio high, rejecting GPS")

        lat = lat0 + (y + spoof_n) / m_per_deg_lat
        lon = lon0 + (x + spoof_e) / m_per_deg_lon

        rows.append(
            (
                t, phase, x, y, h, h + field_elev, h_dot, V, v_meas, groundspeed, phi,
                theta, psi, p_rate, q_rate, psi_dot + yaw_transient, V_dot, throttle, elev_cmd, elev_pos,
                *motor_rpm, v_batt, i_batt, soc, batt_temp, gen_kw, fix, nsat, hdop,
                pos_innov, vel_innov, hacc, vib, lat, lon, gamma,
            )
        )
        t += dt

    cols = [
        "t", "phase", "x_e", "y_n", "h", "alt_msl", "h_dot", "v_tas", "v_ias_meas", "groundspeed", "roll",
        "pitch", "yaw", "p", "q", "r", "v_dot", "throttle", "elev_cmd", "elev_pos",
        "rpm1", "rpm2", "rpm3", "rpm4", "batt_v", "batt_a", "soc", "batt_temp", "gen_kw", "gnss_fix",
        "gnss_nsat", "gnss_hdop", "pos_innov", "vel_innov", "hacc", "vibe", "lat", "lon", "gamma",
    ]
    truth = pd.DataFrame(rows, columns=cols)
    truth["fault"] = "none"
    for f in cfg.faults:
        end = f.t_end if f.t_end is not None else truth.t.max() + 1
        truth.loc[(truth.t >= f.t_start) & (truth.t < end), "fault"] = f.type
    meta = {
        "flight_id": cfg.flight_id,
        "airframe": cfg.airframe,
        "firmware": cfg.firmware,
        "site": cfg.site,
        "date": cfg.date,
        "duration_s": float(truth.t.max()),
        "wind_speed": cfg.wind_speed,
        "wind_dir_deg": cfg.wind_dir_deg,
        "landed": landed,
        "faults": [asdict(f) for f in cfg.faults],
        "sim_config": {k: v for k, v in asdict(cfg).items() if k != "faults"},
    }
    return {"truth": truth, "messages": msgs, "meta": meta}
