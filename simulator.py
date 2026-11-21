"""
simulator.py - Tower Brain building simulator (Person 1)

11 floor-zones, each a lumped RC thermal model (thermal mass C, envelope conductance UA),
driven by outdoor temperature, solar gain by facade, people, and the AC controller.
Time step seen by controllers: 15 min. Internally: 1-min substeps (stable, since tau >> 1 min).

Interface for the optimizer (Person 2) and dashboard (Person 3):

    sim = BuildingSim("timetable.csv", scenario="summer", seed=1)
    sim.reset(start_day="Monday")
    while not sim.done:
        setpoints = controller(sim)          # array of 11 values in deg C, np.nan = AC off
        state = sim.step(setpoints)          # advances 15 minutes, returns a dict
    results = sim.history()                  # DataFrame, one row per 15-min step
    kpis = sim.kpis()                        # dict: kWh, peak kW, comfort %, ...

    run(controller, scenario="summer", ...)  # convenience wrapper returning (history, kpis)
    FixedSchedule(...)                       # the baseline controller

All thermal parameters are assumptions (see Params and assumptions.md), not measured data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

N_FLOORS = 11
STEP_MIN = 15
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
COMFORT_MIN, COMFORT_MAX = 22.0, 26.0  # deg C, occupied-comfort range used for KPIs and guardrail

SCENARIOS = {  # outdoor temperature range (deg C) and day-of-year for the sun position
    "summer": dict(tmin=31.0, tmax=43.0, doy=200),     # typical hot July day, Abu Dhabi
    "heatwave": dict(tmin=34.0, tmax=47.0, doy=200),   # +3 to +4 deg C heatwave
    "november": dict(tmin=21.0, tmax=31.0, doy=321),   # mild day (the actual demo month)
}


def to_min(t: str) -> int:
    h, m = str(t).split(":")
    return int(h) * 60 + int(m)


@dataclass
class Params:
    # --- geometry (assumed: one rectangular 11-floor building, 30 m x 14 m plan) ---
    latitude: float = 24.45
    solar_noon_h: float = 12.37          # Abu Dhabi: UTC+4 zone, longitude 54.4 E
    floor_area: float = 420.0            # m2 per floor (3 classrooms + corridors/offices)
    floor_height: float = 3.6            # m
    facade_ns_len: float = 30.0          # N and S facades are the long ones
    facade_ew_len: float = 14.0          # E and W facades
    glazing_ratio: float = 0.40
    # --- envelope ---
    u_wall: float = 0.50                 # W/m2K
    u_glass: float = 2.10                # W/m2K
    shgc: float = 0.30                   # solar heat gain coefficient incl. blinds
    wall_absorptance: float = 0.5
    h_out: float = 17.0                  # W/m2K outside film coefficient (sol-air)
    roof_u: float = 0.40
    roof_absorptance: float = 0.7
    roof_longwave_k: float = 3.0         # sky radiation loss, K
    # lower floors partly shaded by neighbouring buildings/podium (assumption)
    shading: tuple = (0.6, 0.6, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    # --- thermal mass and ventilation ---
    thermal_mass_kj_m2k: float = 120.0   # effective, incl. furniture and part of slab
    vent_ac_on_w_k: float = 150.0        # fresh air + infiltration when AC runs
    vent_per_person_w_k: float = 9.0     # 7.5 L/s per person (demand based, same in all cases)
    infiltration_off_w_k: float = 40.0
    # --- internal gains ---
    internal_w_m2_open: float = 10.0     # lights + equipment while building is open
    person_w: float = 100.0
    class_equip_w: float = 500.0         # projector/PC per class in session
    staff_per_floor: int = 4
    open_start: int = 7 * 60
    open_end: int = 18 * 60
    staff_start: int = 8 * 60
    staff_end: int = 17 * 60
    # --- AC plant ---
    q_max_kw: float = 55.0               # max thermal cooling per floor
    prop_band_k: float = 1.0             # proportional thermostat band
    latent_factor: float = 1.15          # extra load for dehumidification (humid climate)
    fan_w_m2: float = 3.5                # AHU fan/pump power per floor while AC is on


# ----------------------------------------------------------------------------------
# Weather and sun
# ----------------------------------------------------------------------------------
def outdoor_temp(minute: np.ndarray, tmin: float, tmax: float) -> np.ndarray:
    """Asymmetric daily curve: minimum at 05:30, maximum at 15:00."""
    h = minute / 60.0
    out = np.empty_like(h)
    a = h < 5.5
    b = (h >= 5.5) & (h < 15.0)
    c = h >= 15.0
    t = (h[a] + 9.0) / 14.5
    out[a] = tmin + (tmax - tmin) * 0.5 * (1 + np.cos(np.pi * t))
    t = (h[b] - 5.5) / 9.5
    out[b] = tmin + (tmax - tmin) * 0.5 * (1 - np.cos(np.pi * t))
    t = (h[c] - 15.0) / 14.5
    out[c] = tmin + (tmax - tmin) * 0.5 * (1 + np.cos(np.pi * t))
    return out


def solar_vertical(doy: int, p: Params):
    """Clear-sky irradiance (W/m2) on N, E, S, W vertical surfaces and horizontal (GHI), per minute."""
    m = np.arange(1440)
    h = m / 60.0
    decl = math.radians(23.45 * math.sin(math.radians(360.0 / 365.0 * (284 + doy))))
    lat = math.radians(p.latitude)
    omega = np.radians(15.0 * (h - p.solar_noon_h))
    sin_alt = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(omega)
    alt = np.arcsin(np.clip(sin_alt, -1, 1))
    up = sin_alt > 0.02
    cos_az = (np.sin(decl) - sin_alt * np.sin(lat)) / (np.cos(alt) * np.cos(lat) + 1e-9)
    az = np.degrees(np.arccos(np.clip(cos_az, -1, 1)))
    az = np.where(omega > 0, 360.0 - az, az)  # azimuth from north, clockwise
    s = np.where(up, sin_alt, 1.0)
    dni = np.where(up, 1000.0 * np.exp(-0.15 / s), 0.0)
    dhi = np.where(up, 120.0 * np.maximum(sin_alt, 0) ** 0.6, 0.0)
    ghi = dni * np.maximum(sin_alt, 0) + dhi
    surf = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}
    iv = {}
    for k, a_s in surf.items():
        cos_t = np.cos(alt) * np.cos(np.radians(az - a_s))
        iv[k] = np.where(up, dni * np.maximum(cos_t, 0) + 0.5 * dhi + 0.1 * ghi, 0.0)
    return iv, ghi


# ----------------------------------------------------------------------------------
# Attendance model (no-shows, cancellations, late arrivals) - synthetic but learnable
# ----------------------------------------------------------------------------------
def draw_attendance(tt: pd.DataFrame, rng: np.random.Generator, weeks: int = 1,
                    cancel_rate: float = 0.05, start_date: str = "2026-09-07") -> pd.DataFrame:
    rows = []
    d0 = pd.Timestamp(start_date)
    for w in range(weeks):
        for _, r in tt.iterrows():
            start = to_min(r.start_time_24hr)
            mean = 0.90
            if start < 10 * 60:
                mean -= 0.08                                   # early classes
            if r.day == "Thursday" and start >= 14 * 60:
                mean -= 0.12                                   # Thursday afternoon dip
            if r.day == "Friday":
                mean -= 0.10                                   # Friday drop
            if r.class_size >= 45:
                mean -= 0.03                                   # big classes skip more
            p = rng.beta(mean * 30, (1 - mean) * 30)
            cancelled = bool(rng.random() < cancel_rate)
            att = 0 if cancelled else int(rng.binomial(int(r.class_size), p))
            date = d0 + pd.Timedelta(days=7 * w + DAY_NAMES.index(r.day))
            rows.append(dict(date=date.date().isoformat(), week=w + 1, day=r.day,
                             room_id=r.room_id, floor_id=r.floor_id,
                             start_time=r.start_time_24hr, end_time=r.end_time_24hr,
                             scheduled=int(r.class_size), attending=att,
                             cancelled=int(cancelled)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------
# Baseline controller
# ----------------------------------------------------------------------------------
class FixedSchedule:
    """Fixed-schedule AC. All floors on at start_min, off at stop_min, constant setpoint.
    stagger_min > 0 makes floor k start k*stagger_min later (a 'tuned timer')."""

    def __init__(self, start_min=7 * 60, stop_min=18 * 60, setpoint=23.5, stagger_min=0,
                 monday_start_min=4 * 60, name=None):
        self.start, self.stop, self.sp, self.stagger = start_min, stop_min, setpoint, stagger_min
        self.monday_start = monday_start_min  # optional earlier start to recover from the weekend
        self.name = name or "fixed"

    def __call__(self, sim: "BuildingSim") -> np.ndarray:
        sp = np.full(N_FLOORS, np.nan)
        if sim.day_idx >= 5:
            return sp
        start = self.monday_start if (sim.day_idx == 0 and self.monday_start is not None) else self.start
        for f in range(N_FLOORS):
            if start + f * self.stagger <= sim.minute < self.stop:
                sp[f] = self.sp
        return sp


# ----------------------------------------------------------------------------------
# The simulator
# ----------------------------------------------------------------------------------
class BuildingSim:
    def __init__(self, timetable_path="timetable.csv", scenario="summer", params: Params | None = None,
                 seed: int = 0, cancel_rate: float = 0.05):
        self.p = params or Params()
        self.tt = pd.read_csv(timetable_path)
        self.scenario_name = scenario
        self.sc = SCENARIOS[scenario]
        self.seed = seed
        self.cancel_rate = cancel_rate
        p = self.p
        self.C = p.thermal_mass_kj_m2k * 1e3 * p.floor_area  # J/K per floor
        self.kp_w_k = p.q_max_kw * 1e3 / p.prop_band_k
        self._build_envelope()

    # ---- static envelope numbers per floor and solar/weather arrays ----
    def _build_envelope(self):
        p = self.p
        m = np.arange(1440)
        self.tout = outdoor_temp(m, self.sc["tmin"], self.sc["tmax"])
        iv, ghi = solar_vertical(self.sc["doy"], p)
        self.ghi = ghi
        lens = {"N": p.facade_ns_len, "S": p.facade_ns_len, "E": p.facade_ew_len, "W": p.facade_ew_len}
        self.ua_env = np.zeros(N_FLOORS)
        self.g_sol = np.zeros((1440, N_FLOORS))
        for f in range(N_FLOORS):
            ua = 0.0
            g = np.zeros(1440)
            for o, ln in lens.items():
                facade = ln * p.floor_height
                glass = facade * p.glazing_ratio
                wall = facade - glass
                ua += glass * p.u_glass + wall * p.u_wall
                g += glass * p.shgc * p.shading[f] * iv[o]
                g += wall * p.u_wall * p.wall_absorptance * p.shading[f] * iv[o] / p.h_out
            if f == N_FLOORS - 1:  # roof on the top floor
                ua += p.floor_area * p.roof_u
                g += p.floor_area * p.roof_u * (p.roof_absorptance * ghi / p.h_out - p.roof_longwave_k)
            self.ua_env[f] = ua
            self.g_sol[:, f] = g

    # ---- (re)start a run ----
    def reset(self, start_day: str = "Monday", n_days: int = 5):
        self.rng = np.random.default_rng(self.seed)
        self.attendance = draw_attendance(self.tt, self.rng, 1, self.cancel_rate)
        self._build_occupancy()
        self.day_idx = DAY_NAMES.index(start_day)
        self.first_day = self.day_idx
        self.end_day = self.day_idx + n_days
        self.minute = 0
        self.t_in = np.full(N_FLOORS, 25.5)
        # passive heat soak since closing time (18:00 the previous working day): AC off, nobody inside
        soak_min = (54 if start_day == "Monday" else 6) * 60
        zero = np.zeros(N_FLOORS)
        nan = np.full(N_FLOORS, np.nan)
        for k in range(soak_min):
            self._substep((18 * 60 + k) % 1440, nan, zero, zero, False)
        self._rows = []
        return self

    @property
    def done(self) -> bool:
        return self.day_idx >= self.end_day

    # ---- per-day occupancy arrays (minute x floor) from the attendance draw ----
    def _build_occupancy(self):
        p = self.p
        self.occ = np.zeros((7, 1440, N_FLOORS))     # people (students + staff)
        self.ncls = np.zeros((7, 1440, N_FLOORS))    # classes in session
        for _, r in self.attendance.iterrows():
            d = DAY_NAMES.index(r.day)
            f = int(r.floor_id[1:]) - 1
            s, e = to_min(r.start_time), to_min(r.end_time)
            if r.cancelled:
                continue
            mins = np.arange(s - 5, e)
            presence = np.clip((mins - (s - 5)) / 15.0, 0, 1)  # arrivals spread from -5 to +10 min
            self.occ[d, mins, f] += r.attending * presence
            self.ncls[d, s:e, f] += 1
        for d in range(5):
            self.occ[d, p.staff_start:p.staff_end, :] += p.staff_per_floor

    # ---- one minute of physics for all floors ----
    def _substep(self, m, setp, people, ncls, open_flag):
        p = self.p
        on = ~np.isnan(setp)
        vent = np.where(on, p.vent_ac_on_w_k + p.vent_per_person_w_k * people, p.infiltration_off_w_k)
        ua = self.ua_env + vent
        internal = (p.internal_w_m2_open * p.floor_area * float(open_flag)
                    + p.person_w * people + p.class_equip_w * ncls)
        g = self.g_sol[m] + internal
        q_ac = np.where(on, np.clip(self.kp_w_k * (self.t_in - np.where(on, setp, 0.0)), 0, p.q_max_kw * 1e3), 0.0)
        self.t_in = self.t_in + 60.0 / self.C * (ua * (self.tout[m] - self.t_in) + g - q_ac)
        cop = max(2.0, 3.4 - 0.07 * (self.tout[m] - 35.0))
        pw = q_ac * p.latent_factor / cop + np.where(on, p.fan_w_m2 * p.floor_area, 0.0)
        return q_ac, pw

    # ---- advance one 15-minute control step ----
    def step(self, setpoints) -> dict:
        setp = np.asarray(setpoints, dtype=float)
        if setp.shape != (N_FLOORS,):
            raise ValueError("setpoints must have 11 values (np.nan = AC off)")
        p = self.p
        d = self.day_idx
        t0 = self.minute
        pw_sum = np.zeros(N_FLOORS)
        q_sum = np.zeros(N_FLOORS)
        occ_sum = np.zeros(N_FLOORS)
        tin_sum = np.zeros(N_FLOORS)
        pm_total = pm_ok = 0.0
        for k in range(STEP_MIN):
            m = t0 + k
            people = self.occ[d, m] if d < 7 else np.zeros(N_FLOORS)
            ncls = self.ncls[d, m] if d < 7 else np.zeros(N_FLOORS)
            open_flag = d < 5 and p.open_start <= m < p.open_end
            q, pw = self._substep(m, setp, people, ncls, open_flag)
            pw_sum += pw
            q_sum += q
            occ_sum += people
            tin_sum += self.t_in
            ok = (self.t_in >= COMFORT_MIN) & (self.t_in <= COMFORT_MAX)
            pm_total += people.sum()
            pm_ok += (people * ok).sum()
        p_kw = pw_sum / STEP_MIN / 1000.0
        row = dict(day=DAY_NAMES[d % 7], day_idx=d, minute=t0,
                   time=f"{t0 // 60:02d}:{t0 % 60:02d}", t_out=float(self.tout[t0 + STEP_MIN // 2]),
                   total_kw=float(p_kw.sum()), cooling_kw=float(q_sum.sum() / STEP_MIN / 1000.0),
                   people=float(occ_sum.sum() / STEP_MIN), pm_total=pm_total, pm_ok=pm_ok)
        for f in range(N_FLOORS):
            row[f"kw_f{f + 1}"] = float(p_kw[f])
            row[f"t_in_f{f + 1}"] = float(tin_sum[f] / STEP_MIN)
            row[f"occ_f{f + 1}"] = float(occ_sum[f] / STEP_MIN)
            row[f"sp_f{f + 1}"] = float(setp[f]) if not np.isnan(setp[f]) else np.nan
        self._rows.append(row)
        self.minute += STEP_MIN
        if self.minute >= 1440:
            self.minute = 0
            self.day_idx += 1
        return row

    # ---- information for optimizers (timetable only, not the true attendance) ----
    def scheduled_classes(self, horizon_min: int = 120) -> pd.DataFrame:
        """Classes scheduled today that start within the horizon or are in session."""
        day = DAY_NAMES[self.day_idx % 7]
        t = self.tt[self.tt.day == day].copy()
        t["s"] = t.start_time_24hr.map(to_min)
        t["e"] = t.end_time_24hr.map(to_min)
        return t[(t.e > self.minute) & (t.s <= self.minute + horizon_min)].reset_index(drop=True)

    def outdoor_now(self) -> float:
        return float(self.tout[self.minute])

    # ---- results ----
    def history(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)

    def daily_table(self) -> pd.DataFrame:
        """kWh, peak kW and comfort % for each simulated day."""
        h = self.history()
        step_h = STEP_MIN / 60.0
        g = h.groupby(["day_idx", "day"])
        t = pd.DataFrame({
            "kwh": g.total_kw.sum() * step_h,
            "peak_kw": g.total_kw.max(),
            "comfort_pct": 100 * g.pm_ok.sum() / g.pm_total.sum().clip(lower=1e-9),
        }).reset_index().drop(columns="day_idx")
        return t.round(1)

    def kpis(self) -> dict:
        h = self.history()
        step_h = STEP_MIN / 60.0
        per_day = h.groupby("day_idx").total_kw.sum() * step_h
        peak_row = h.loc[h.total_kw.idxmax()]
        area = self.p.floor_area * N_FLOORS
        return dict(
            scenario=self.scenario_name,
            total_kwh=float(h.total_kw.sum() * step_h),
            avg_weekday_kwh=float(per_day.mean()),
            kwh_per_m2_per_day=float(per_day.mean() / area),
            peak_kw_15min=float(h.total_kw.max()),
            peak_day=str(peak_row["day"]),
            peak_time=str(peak_row["time"]),
            peak_w_per_m2=float(h.total_kw.max() * 1000 / area),
            comfort_pct=float(100 * h.pm_ok.sum() / max(h.pm_total.sum(), 1e-9)),
        )


def run(controller, scenario="summer", timetable_path="timetable.csv", start_day="Monday", n_days=5,
        seed=1, cancel_rate=0.05, params: Params | None = None):
    sim = BuildingSim(timetable_path, scenario=scenario, params=params, seed=seed, cancel_rate=cancel_rate)
    sim.reset(start_day=start_day, n_days=n_days)
    while not sim.done:
        sim.step(controller(sim))
    return sim.history(), sim.kpis()


def make_deliverables(out_dir="."):
    """Writes attendance_history.csv, baseline_results.csv and baseline_profile_summer.csv."""
    import os
    tt_path = os.path.join(out_dir, "timetable.csv")
    tt = pd.read_csv(tt_path)
    # 15 weeks of synthetic attendance for the occupancy-prediction model (Person 2)
    hist = draw_attendance(tt, np.random.default_rng(42), weeks=15)
    hist.to_csv(os.path.join(out_dir, "attendance_history.csv"), index=False)

    rows = []
    variants = {
        "fixed_schedule": FixedSchedule(),  # 07:00-18:00, 23.5 C, all floors together
        "tuned_timer": FixedSchedule(7 * 60, 17 * 60 + 30, 24.0, 5, name="tuned_timer"),
    }
    for sc in SCENARIOS:
        for name, ctrl in variants.items():
            if name == "tuned_timer" and sc != "summer":
                continue
            h, k = run(ctrl, scenario=sc, timetable_path=tt_path)
            sim_days = h.groupby(["day_idx", "day"]).total_kw.agg(["sum", "max"]).reset_index()
            tf = sim_days[sim_days.day_idx >= 1]  # Tue-Fri = typical weekday (Monday is weekend recovery)
            rows.append(dict(
                scenario=sc, baseline=name,
                week_kwh=round(k["total_kwh"]),
                typical_weekday_kwh=round(tf["sum"].mean() * STEP_MIN / 60),
                kwh_per_m2_typical_weekday=round(tf["sum"].mean() * STEP_MIN / 60 / (Params().floor_area * N_FLOORS), 3),
                peak_kw=round(k["peak_kw_15min"], 1), peak_w_per_m2=round(k["peak_w_per_m2"], 1),
                peak_day=k["peak_day"], peak_time=k["peak_time"],
                comfort_pct=round(k["comfort_pct"], 1)))
            if sc == "summer" and name == "fixed_schedule":
                cols = ["day", "time", "t_out", "total_kw", "people"] + [f"kw_f{i}" for i in range(1, 12)] \
                       + [f"t_in_f{i}" for i in range(1, 12)]
                h[cols].round(2).to_csv(os.path.join(out_dir, "baseline_profile_summer.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "baseline_results.csv"), index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(make_deliverables().to_string(index=False))
