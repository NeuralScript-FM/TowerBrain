"""
controller.py - Tower Brain rule-based AC controller (Person 3)

Uses PERSON 2's occupancy model. Source order (first one that exists):
  1. LIVE MODEL  - Person 2's trained Random Forest (random_forest_model.joblib, made by `python model.py`).
                   Every 15 min the controller feeds it the building's occupancy history (assumed people-counter
                   / Wi-Fi sensor readings) and gets a 1-hour-ahead forecast per floor, exactly as model.py was trained.
  2. CSV FILE    - data/occupancy_predictions.csv (columns below), if Person 2 exports forecasts for the simulated week.
  3. FALLBACK    - timetable estimate (class_size x 0.82), also used for anything beyond the model's 1-hour horizon.
Every decision-log row says which source it used.

Run:
    python controller.py                          # full before/after, writes outputs/
    python controller.py --model path.joblib      # use a specific Person 2 model file
    python controller.py --predictions path.csv   # use a CSV of forecasts instead
    python controller.py --no-predictions         # force the timetable fallback
    python controller.py --cap-frac 0.70          # peak cap = 70 % of the baseline peak

Person 2 file format (one row per prediction):
    timestamp,target_timestamp,floor_id,predicted_people
    2026-12-14 08:00,2026-12-14 08:15,F3,27.4
  timestamp        = when the prediction was made (controller only uses rows made at or before "now")
  target_timestamp = the time being predicted
  floor_id         = F1..F11
  predicted_people = predicted STUDENTS on that floor. The controller adds the 4 staff per floor
                     (Mon-Fri 08:00-17:00). Set PREDICTION_INCLUDES_STAFF = True if Person 2's
                     number already includes staff.
The simulated week maps Monday = WEEK_START (2026-12-14).

Rules (checked per floor every 15 min):
  1 comfort guardrail   occupied and floor >= 26 C           -> 24 C, never staggered
  5 maintain            occupied and within limit            -> 24 C students / 25.5 C staff only
  2 pre-cool            arrival soon and floor needs pull-down -> 24 C, started early enough to be ready
  3 stagger             pre-cool would exceed the cap        -> delayed, most urgent floor served first
  4 empty-floor setback empty and nobody due soon            -> AC off (setback target 28 C)
Decisions use predictions and temperatures only. True occupancy from the simulator is written to the
log for evaluation but is never used to decide.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from simulator import BuildingSim, FixedSchedule, N_FLOORS, DAY_NAMES, COMFORT_MAX, SCENARIOS

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
TIMETABLE = DATA_DIR / "timetable.csv"
PREDICTIONS_PATH = DATA_DIR / "occupancy_predictions.csv"

WEEK_START = pd.Timestamp("2026-12-14")      # Monday of the simulated week
PREDICTION_INCLUDES_STAFF = False
STEP = 15
MODEL_NOISE_FLOOR = 4.5                      # people; Person 2's model forecasts at or below this are background, not a class


def to_min(t):
    h, m = str(t).split(":")
    return int(h) * 60 + int(m)


def fmt(abs_min):
    return f"{DAY_NAMES[(abs_min // 1440) % 7]} {(abs_min % 1440) // 60:02d}:{abs_min % 60:02d}"


# ------------------------------------------------------------------------------
# Occupancy source: Person 2's model first, timetable fallback second
# ------------------------------------------------------------------------------
class OccupancyPredictor:
    def __init__(self, timetable, predictions_path=None, attendance_ratio=0.82, staff=4):
        self.ratio, self.staff = attendance_ratio, staff
        self.includes_staff = PREDICTION_INCLUDES_STAFF
        self.sessions = {f: [] for f in range(N_FLOORS)}          # fallback: (start, end, expected students)
        self.raw_sessions = {f: [] for f in range(N_FLOORS)}      # (start, end, class_size) for the model's features
        for r in timetable.itertuples():
            if r.day not in DAY_NAMES:
                continue
            f = int(str(r.floor_id)[1:]) - 1
            d = DAY_NAMES.index(r.day) * 1440
            self.raw_sessions[f].append((d + to_min(r.start_time_24hr), d + to_min(r.end_time_24hr), float(r.class_size)))
            self.sessions[f].append((d + to_min(r.start_time_24hr), d + to_min(r.end_time_24hr),
                                     float(r.class_size) * attendance_ratio))
        self.model = {f: {} for f in range(N_FLOORS)}             # model[floor][issue_min][target_min] = people
        self.issue_times = {f: [] for f in range(N_FLOORS)}
        self.n_model_rows = 0
        self.source_file = None
        if predictions_path is not None and Path(predictions_path).exists():
            self._load(Path(predictions_path))

    def _load(self, path):
        df = pd.read_csv(path)
        need = {"timestamp", "target_timestamp", "floor_id", "predicted_people"}
        if not need.issubset(df.columns):
            raise ValueError(f"{path.name} must have columns {sorted(need)}; found {list(df.columns)}")
        im = ((pd.to_datetime(df["timestamp"]) - WEEK_START).dt.total_seconds() // 60).astype(int)
        tm = ((pd.to_datetime(df["target_timestamp"]) - WEEK_START).dt.total_seconds() // 60).astype(int)
        ok = (tm >= 0) & (tm < 7 * 1440)
        for fl, i, t, p in zip(df["floor_id"][ok], im[ok], tm[ok], df["predicted_people"][ok]):
            f = int(str(fl).strip()[1:]) - 1
            if 0 <= f < N_FLOORS:
                self.model[f].setdefault(int(i), {})[int(t)] = float(p)
                self.n_model_rows += 1
        self.issue_times = {f: sorted(self.model[f]) for f in range(N_FLOORS)}
        self.source_file = path.name
        if self.n_model_rows == 0:
            print(f"WARNING: {path.name} has no rows inside the week starting {WEEK_START.date()}; using fallback only.")

    def staff_at(self, m):
        d, md = divmod(m, 1440)
        return self.staff if (d < 5 and 8 * 60 <= md < 17 * 60) else 0

    def _fallback_students(self, f, m):
        return sum(p for s, e, p in self.sessions[f] if s <= m < e)

    def people(self, f, now, target):
        """Predicted people on floor f at minute `target`, using only predictions issued at or before `now`.
        Returns (people, 'model' | 'fallback')."""
        if self.n_model_rows:
            issues = self.issue_times[f]
            k = int(np.searchsorted(issues, now, side="right")) - 1
            for _ in range(8):                                    # newest forecast that covers this target
                if k < 0:
                    break
                table = self.model[f][issues[k]]
                for dt in (0, 7, -7):
                    hit = table.get(target + dt)
                    if hit is not None:
                        if self.includes_staff:
                            # live model was trained with a few "background" people on every floor 07:00-19:00,
                            # so forecasts <= MODEL_NOISE_FLOOR mean "no class": use the known staff roster instead
                            return (hit if hit > MODEL_NOISE_FLOOR else float(self.staff_at(target))), "model"
                        return hit + self.staff_at(target), "model"
                k -= 1
        return self._fallback_students(f, target) + self.staff_at(target), "fallback"


class LiveModelPredictor(OccupancyPredictor):
    """Person 2's Random Forest, called live. Needs the same 17 features model.py trains on."""

    def __init__(self, timetable, model_path, **kw):
        super().__init__(timetable, None, **kw)
        import joblib
        bundle = joblib.load(model_path)
        self.rf, self.features = bundle["model"], bundle["feature_columns"]
        try:
            self.rf.set_params(n_jobs=1)                     # 11 rows per call: threads would only add overhead
        except Exception:
            pass
        self.includes_staff = True                           # model forecasts TOTAL people; staff roster is applied as a floor
        self.source_file = Path(model_path).name
        self.n_model_rows = 1                                # switches the model path on
        self.occ_hist = [[] for _ in range(N_FLOORS)]        # per-floor 15-min occupancy readings so far
        self.seen_rows = 0
        self.n_forecasts = 0

    def _sched(self, f, m):
        return sum(p for s, e, p in self.raw_sessions[f] if s <= m < e)

    def observe(self, sim, now):
        """Read the latest sensor row, then register a 1-hour-ahead forecast for every floor."""
        for row in sim._rows[self.seen_rows:]:
            for f in range(N_FLOORS):
                self.occ_hist[f].append(row[f"occ_f{f + 1}"])
        self.seen_rows = len(sim._rows)
        tgt = now + 60
        X = []
        for f in range(N_FLOORS):
            h = self.occ_hist[f]
            g = lambda k: h[-k] if len(h) >= k else 0.0       # before the week starts: empty building
            d, md = divmod(now, 1440)
            td, tmd = divmod(tgt, 1440)
            X.append({
                "floor_number": f + 1, "hour": md // 60, "minute": md % 60, "day_of_week": d % 7,
                "is_weekend": int(d % 7 >= 5), "future_hour": tmd // 60, "future_minute": tmd % 60,
                "future_day_of_week": td % 7, "current_occupancy": g(1), "occupancy_lag_15m": g(2),
                "occupancy_lag_30m": g(3), "occupancy_lag_60m": g(5), "occupancy_same_time_previous_day": g(97),
                "occupancy_mean_last_hour": float(np.mean(h[-4:])) if len(h) >= 4 else 0.0,
                "scheduled_occupancy": self._sched(f, now), "scheduled_next_hour": self._sched(f, tgt),
                "outdoor_temp_c": float(sim.outdoor_now()),
            })
        y = np.maximum(0, self.rf.predict(pd.DataFrame(X)[self.features]))
        for f in range(N_FLOORS):
            self.model[f][now] = {tgt: float(np.rint(y[f]))}
            self.issue_times[f].append(now)
        self.n_forecasts += N_FLOORS


# ------------------------------------------------------------------------------
# Controller
# ------------------------------------------------------------------------------
class TowerBrainController:
    def __init__(self, predictor, cap_kw=None, normal_sp=24.0, staff_sp=25.5, setback_sp=28.0,
                 comfort_limit=COMFORT_MAX, precool_goal=25.8, horizon_min=720, buffer_min=15,
                 early_window_min=120, occupied_threshold=3.0, student_threshold=4.5, dwell_min=30,
                 safety=1.15):
        self.pred, self.cap_kw = predictor, cap_kw
        self.normal_sp, self.staff_sp, self.setback_sp = normal_sp, staff_sp, setback_sp
        self.comfort_limit, self.goal = comfort_limit, precool_goal
        self.horizon, self.buffer, self.early = horizon_min, buffer_min, early_window_min
        self.occ_thr, self.student_thr, self.dwell, self.safety = occupied_threshold, student_threshold, dwell_min, safety
        self.prev_sp = np.full(N_FLOORS, np.nan)
        self.last_change = np.full(N_FLOORS, -10 ** 9)
        self.log = []
        self.source_counts = {"model": 0, "fallback": 0, "mixed": 0}

    # thermal helpers: same assumed building parameters the simulator uses
    def _cop(self, tout):
        return max(2.0, 3.4 - 0.07 * (tout - 35.0))

    def est_kw(self, sim, t_in, sp):
        p = sim.p
        q = float(np.clip(sim.kp_w_k * (t_in - sp), 0, p.q_max_kw * 1e3))
        return q * p.latent_factor / self._cop(sim.outdoor_now()) / 1e3 + p.fan_w_m2 * p.floor_area / 1e3

    def pulldown_min(self, sim, f, t_in):
        """Minutes for floor f to fall from t_in to the pre-cool goal at full cooling (simple RC estimate)."""
        if t_in <= self.goal:
            return 0
        p = sim.p
        t_mid = 0.5 * (t_in + self.goal)
        leak = max((sim.ua_env[f] + p.vent_ac_on_w_k) * (sim.outdoor_now() - t_mid), 0.0)
        solar = max(sim.g_sol[min(sim.minute, 1439), f], 0.0)
        net = max(p.q_max_kw * 1e3 - leak - solar, 0.15 * p.q_max_kw * 1e3)
        return int(np.ceil(self.safety * (t_in - self.goal) * sim.C / net / 60 / 5) * 5)

    def __call__(self, sim):
        now = sim.day_idx * 1440 + sim.minute
        if hasattr(self.pred, "observe"):
            self.pred.observe(sim, now)                                       # live Person 2 forecast
        info = []
        for f in range(N_FLOORS):
            srcs = set()
            occ, s = self.pred.people(f, now, now)
            srcs.add(s)
            arrival, arr_people = None, 0.0
            if occ < self.occ_thr:
                for dt in range(STEP, self.horizon + 1, STEP):
                    ppl, s = self.pred.people(f, now, now + dt)
                    if ppl >= self.occ_thr:
                        arrival, arr_people = dt, ppl
                        srcs.add(s)
                        break
            tag = "fallback" if srcs == {"fallback"} else "model" if srcs == {"model"} else "mixed"
            info.append(dict(f=f, t_in=float(sim.t_in[f]), occ=occ, arrival=arrival, arr_people=arr_people, tag=tag))

        sp = np.full(N_FLOORS, np.nan)
        rec, cands, running_kw = {}, [], 0.0

        for d in info:
            f, t_in, occ, arrival = d["f"], d["t_in"], d["occ"], d["arrival"]
            if occ >= self.occ_thr:                                           # Rules 1 and 5
                if t_in >= self.comfort_limit:
                    sp[f], act = self.normal_sp, "comfort_override"
                    why = (f"Rule 1: {occ:.0f} people and floor is {t_in:.2f} C (limit {self.comfort_limit:.0f} C): "
                           f"set {self.normal_sp:g} C, exempt from staggering")
                elif occ > self.student_thr:
                    sp[f], act = self.normal_sp, "maintain"
                    why = f"Rule 5: {occ:.0f} people (students in class), floor {t_in:.1f} C within limit: hold {self.normal_sp:g} C"
                else:
                    sp[f], act = self.staff_sp, "maintain"
                    why = f"Rule 5: {occ:.0f} people (only staff present), floor {t_in:.1f} C within limit: hold {self.staff_sp:g} C"
                running_kw += self.est_kw(sim, t_in, sp[f])
                rec[f] = (act, why)
                continue
            if arrival is None:                                               # Rule 4, nobody due
                rec[f] = ("setback", f"Rule 4: floor empty ({occ:.0f} people), nobody predicted within "
                                     f"{self.horizon // 60} h: AC off, setback target {self.setback_sp:g} C, floor is {t_in:.1f} C")
                continue
            who = "staff" if d["arr_people"] <= self.student_thr else "a class"
            need = self.pulldown_min(sim, f, t_in)
            slack = arrival - need - self.buffer
            d.update(need=need, slack=slack, who=who)
            recently_on = self.prev_sp[f] == self.normal_sp and now - self.last_change[f] < self.dwell
            if need == 0 and not recently_on:
                rec[f] = ("setback", f"Rule 4: empty ({occ:.0f} people), {who} arrive in {arrival} min, "
                                     f"floor already at {t_in:.1f} C: AC off, setback {self.setback_sp:g} C")
            elif slack > self.early and not recently_on:
                rec[f] = ("setback", f"Rule 4: empty ({occ:.0f} people), {who} arrive in {arrival} min, "
                                     f"pull-down from {t_in:.1f} C needs ~{need} min: AC off, setback {self.setback_sp:g} C")
            else:
                cands.append(d)

        cap = self.cap_kw if self.cap_kw is not None else float("inf")        # Rules 2 and 3
        locked = [d for d in cands if self.prev_sp[d["f"]] == self.normal_sp and now - self.last_change[d["f"]] < self.dwell]
        rest = sorted((d for d in cands if d not in locked), key=lambda d: d["slack"])
        for d in locked + rest:
            f, t_in = d["f"], d["t_in"]
            kw = self.est_kw(sim, t_in, self.normal_sp)
            lead = f"{d['who']} arrive in {d['arrival']} min, floor {t_in:.1f} C needs ~{d['need']} min to reach {self.goal:g} C"
            urgent = d["slack"] <= 0
            held_off = np.isnan(self.prev_sp[f]) and now - self.last_change[f] < self.dwell and not urgent
            if d in locked:
                run, extra = True, f" | hold: pre-cool started <{self.dwell} min ago, keeping it on"
            elif held_off:
                run, extra = False, ""
            elif urgent or running_kw + kw <= cap:
                run, extra = True, ("" if urgent else " | cap has room, starting early so the pull-down is spread out")
            else:
                run, extra = False, ""
            if run:
                sp[f] = self.normal_sp
                running_kw += kw
                why = f"Rule 2: {lead}: pre-cool at {self.normal_sp:g} C{extra}"
                if urgent and running_kw > cap:
                    why += f" | deadline reached, runs even though the {cap:.0f} kW cap is exceeded"
                rec[f] = ("pre_cool", why)
            elif held_off:
                rec[f] = ("stagger", f"Rule 3: {lead} (slack {d['slack']} min), AC switched <{self.dwell} min ago: hold off")
            else:
                rec[f] = ("stagger", f"Rule 3: {lead.split(',')[0]} (slack {d['slack']} min) but ~{running_kw + kw:.0f} kW "
                                     f"would pass the cap {cap:.0f} kW: pre-cool delayed, most urgent floor served first")

        for d in info:
            f = d["f"]
            act, why = rec[f]
            new = sp[f]
            on = not np.isnan(new)
            if not ((not on and np.isnan(self.prev_sp[f])) or (on and new == self.prev_sp[f])):
                self.last_change[f] = now
            tag = {"model": "[prediction: Person 2 model]", "fallback": "[fallback prediction]",
                   "mixed": "[prediction: Person 2 model + fallback for gaps]"}[d["tag"]]
            self.source_counts[d["tag"]] += 1
            true_occ = float(sim.occ[sim.day_idx, sim.minute, f])          # evaluation only, never used to decide
            self.log.append(dict(
                timestamp=fmt(now), floor_id=f"F{f + 1}",
                predicted_occupancy=round(d["occ"], 1), actual_occupancy=round(true_occ, 1),
                indoor_temperature=round(d["t_in"], 2), outdoor_temperature=round(sim.outdoor_now(), 2),
                previous_setpoint=None if np.isnan(self.prev_sp[f]) else float(self.prev_sp[f]),
                new_setpoint=float(new) if on else (self.setback_sp if act in ("setback", "stagger") else None),
                action=act, reason=f"{why} {tag}", ac_on=int(on),
                estimated_power_kw=round(self.est_kw(sim, d["t_in"], new), 1) if on else 0.0,
                prediction_source=d["tag"]))
        self.prev_sp = sp.copy()
        return sp


# ------------------------------------------------------------------------------
# Experiment
# ------------------------------------------------------------------------------
def one_run(controller, scenario, seed):
    sim = BuildingSim(str(TIMETABLE), scenario=scenario, seed=seed)
    sim.reset(start_day="Monday", n_days=5)
    while not sim.done:
        sim.step(controller(sim))
    k = sim.kpis()
    h = sim.history()
    t = h[[f"t_in_f{i}" for i in range(1, N_FLOORS + 1)]].to_numpy()
    o = h[[f"occ_f{i}" for i in range(1, N_FLOORS + 1)]].to_numpy()
    hot = (t > COMFORT_MAX) & (o >= 1)
    k["violations"] = int(hot.sum())
    k["degree_hours"] = float(((t - COMFORT_MAX).clip(min=0) * (o >= 1)).sum() * STEP / 60)
    k["typical_weekday_kwh"] = float((h.groupby("day_idx").total_kw.sum() * STEP / 60).iloc[1:].mean())
    return h, k


def main():
    ap = argparse.ArgumentParser(description="Tower Brain controller (uses Person 2's predictions when available)")
    ap.add_argument("--model", default=None, help="Person 2's random_forest_model.joblib")
    ap.add_argument("--predictions", default=str(PREDICTIONS_PATH))
    ap.add_argument("--no-predictions", action="store_true")
    ap.add_argument("--cap-frac", type=float, default=0.70, help="peak cap as a fraction of the fixed-schedule peak")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tt = pd.read_csv(TIMETABLE)
    pred_path = None if a.no_predictions else a.predictions
    model_path = None
    if not a.no_predictions:
        for c in ([Path(a.model)] if a.model else []) + [DATA_DIR / "random_forest_model.joblib",
                                                        BASE_DIR / "random_forest_model.joblib",
                                                        OUT_DIR / "random_forest_model.joblib"]:
            if c.exists():
                model_path = c
                break
        if a.model and model_path is None:
            raise FileNotFoundError(f"--model file not found: {a.model}")

    def make_pred():
        return LiveModelPredictor(tt, model_path) if model_path else OccupancyPredictor(tt, pred_path)

    probe = make_pred()
    if model_path:
        print(f"Using Person 2's live model: {model_path}")
    elif probe.n_model_rows:
        print(f"Using Person 2 predictions file: {probe.source_file} ({probe.n_model_rows} rows)")
    else:
        print("No Person 2 model (random_forest_model.joblib) or predictions file found: timetable fallback for everything.\n"
              "  -> run `python model.py`, then copy random_forest_model.joblib into data/ and re-run.")

    rows, results, src, summer_log = [], {}, {}, None
    cols = ["day", "time", "total_kw", "people"] + [f"kw_f{i}" for i in range(1, 12)] + [f"t_in_f{i}" for i in range(1, 12)]
    for sc in SCENARIOS:
        hb, kb = one_run(FixedSchedule(), sc, a.seed)
        ctrl = TowerBrainController(make_pred(), cap_kw=a.cap_frac * kb["peak_kw_15min"])
        ho, ko = one_run(ctrl, sc, a.seed)
        results[sc] = (hb, ho)
        pol = [("Fixed schedule (baseline)", kb), ("Tower Brain", ko)]
        if sc == "summer":
            _, kt = one_run(FixedSchedule(7 * 60, 17 * 60 + 30, 24.0, 5, name="tuned_timer"), sc, a.seed)
            pol.append(("Tuned timer (Person 1)", kt))
        for name, k in pol:
            rows.append(dict(scenario=sc, who=name, week_kwh=round(k["total_kwh"], 1),
                             typical_weekday_kwh=round(k["typical_weekday_kwh"], 1), peak_kw=round(k["peak_kw_15min"], 1),
                             peak_time=f"{k['peak_day']} {k['peak_time']}", comfort_pct=round(k["comfort_pct"], 1),
                             comfort_violations_floor_intervals=k["violations"], degree_hours_above_26=round(k["degree_hours"], 1)))
        if sc == "summer":
            summer_log, src = pd.DataFrame(ctrl.log), dict(ctrl.source_counts)
            summer_kpis = {k: v for k, v in ko.items() if k in ("scenario", "total_kwh", "avg_weekday_kwh", "kwh_per_m2_per_day",
                           "peak_kw_15min", "peak_day", "peak_time", "peak_w_per_m2", "comfort_pct")}
            ho.to_csv(out / "tower_brain_summer_history.csv", index=False)           # read by llm_explainer.py
            pd.DataFrame([summer_kpis]).to_csv(out / "tower_brain_summer_kpis.csv", index=False)
            ho[cols].round(2).to_csv(out / "optimized_results.csv", index=False)
            hb[cols].round(2).to_csv(out / "baseline_replica_profile.csv", index=False)
        elif sc == "heatwave":
            ho[cols].round(2).to_csv(out / "optimized_results_heatwave.csv", index=False)

    pd.DataFrame(rows).to_csv(out / "before_after.csv", index=False)
    summer_log.to_csv(out / "decision_log.csv", index=False)
    d, pick = summer_log, []
    def add(df, n=1):
        pick.extend(df.index[:n].tolist())
    add(d[(d.timestamp == "Monday 00:00") & (d.floor_id == "F1")])
    add(d[(d.timestamp == "Monday 04:15") & (d.action == "pre_cool")], 2)
    add(d[(d.timestamp == "Monday 04:15") & (d.action == "stagger")], 2)
    add(d[d.action == "comfort_override"], 2)
    add(d[(d.action == "pre_cool") & (d.prediction_source == "model") & d.timestamp.str.contains("Tuesday")], 2)
    add(d[(d.action == "maintain") & d.reason.str.contains("students") & (d.prediction_source == "model") & d.timestamp.str.contains("Tuesday")], 2)
    add(d[(d.action == "maintain") & d.reason.str.contains("only staff") & d.timestamp.str.contains("Tuesday 10")])
    add(d[(d.action == "setback") & (d.prediction_source == "model") & d.timestamp.str.contains("Tuesday 1[89]")])
    d.loc[sorted(set(pick))].to_csv(out / "decision_log_sample.csv", index=False)

    md = ["| Scenario | Policy | Week kWh | Typical weekday kWh | Peak kW | Comfort % | Occupied floor-intervals >26 C |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['scenario']} | {r['who']} | {r['week_kwh']:,.0f} | {r['typical_weekday_kwh']:,.0f} | {r['peak_kw']} | "
                  f"{r['comfort_pct']} | {r['comfort_violations_floor_intervals']} |")
    (out / "before_after.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "controller_summary.json").write_text(json.dumps(dict(
        scenario="summer", kpis=summer_kpis, before_after=rows,
        prediction_source=("live_model" if model_path else "csv_file" if probe.n_model_rows else "fallback"),
        prediction_file=probe.source_file, decisions_by_source=src,
        cap_frac=a.cap_frac, seed=a.seed, simulated=True), indent=2), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        hb, ho = results["summer"]
        fig, ax = plt.subplots(figsize=(9, 4))
        for h, lab in [(hb, "Fixed schedule"), (ho, "Tower Brain")]:
            d = h[h.day == "Tuesday"]
            ax.plot(range(len(d)), d.total_kw.to_numpy(), label=lab)
        ax.set_xticks(range(0, 96, 8))
        ax.set_xticklabels([f"{i // 4:02d}:00" for i in range(0, 96, 8)])
        ax.set_ylabel("kW")
        ax.set_title("Tuesday power profile (summer, simulated)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "power_profile_tuesday.png", dpi=120)
    except Exception as e:
        print("plot skipped:", e)

    print("\n" + "\n".join(md))
    print("\nDecisions by prediction source (summer):", src)
    print("Files written to", out)


if __name__ == "__main__":
    main()
