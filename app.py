"""
TOWER BRAIN - Streamlit control-centre dashboard
===========================================================

Place this file in the project root, next to simulator.py and controller.py:

    tower_brain/
        app.py                  <- this file
        simulator.py            (building simulator)
        controller.py           (cooling controller)
        data/timetable.csv, data/random_forest_model.joblib, ...
        outputs/, person2/, docs/

Launch (from that folder):   python -m streamlit run app.py

What this app does
------------------
* It calls the team's REAL code live:  simulator.BuildingSim / FixedSchedule (building simulator),
  controller.TowerBrainController + LiveModelPredictor (cooling controller), which loads the trained
  Random Forest (data/random_forest_model.joblib).  The model is only LOADED, never retrained.
* Each (scenario, heat offset, cancelled classes, seed, prediction mode) run is cached, so changing
  a control re-runs the simulation once (~6 s) and revisiting a setting is instant.
* Nothing in the original member files is modified.  Private simulator hooks used (documented below):
  sim.attendance + sim._build_occupancy() for the cancel-class feature.

Everything shown is SIMULATED.  Cost and CO2 are ESTIMATES from editable sidebar assumptions - the
project files contain no tariff or emission factor.
"""
from __future__ import annotations

import dataclasses
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------------------------
# Page setup (must be the first Streamlit call)
# ----------------------------------------------------------------------------------------------
st.set_page_config(page_title="Tower Brain | AI Cooling Control Centre", page_icon="🏢",
                   layout="wide", initial_sidebar_state="expanded")

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))          # so `import simulator` / `import controller` always work

# ----------------------------------------------------------------------------------------------
# Paths and file checks
# ----------------------------------------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
P2_DIR = BASE_DIR / "person2"
MODEL_PATH = DATA_DIR / "random_forest_model.joblib"
TIMETABLE_PATH = DATA_DIR / "timetable.csv"

REQUIRED_FILES = {
    "simulator.py (building simulator)": BASE_DIR / "simulator.py",
    "controller.py (cooling controller)": BASE_DIR / "controller.py",
    "data/timetable.csv (timetable)": TIMETABLE_PATH,
}
missing_required = [name for name, p in REQUIRED_FILES.items() if not p.exists()]
if missing_required:
    st.error("**Required project files are missing:** " + ", ".join(missing_required) +
             f"\n\nThis app must be placed in the project root (the folder that contains `simulator.py`). "
             f"It is currently in: `{BASE_DIR}`")
    st.stop()

try:
    import simulator as S      # building simulator
    import controller as C     # cooling controller (imports the simulator; loads the occupancy model lazily)
except Exception as exc:       # pragma: no cover - environment problem
    st.error(f"Could not import the team's modules: `{type(exc).__name__}: {exc}`\n\n"
             "Run `pip install -r requirements.txt` in the activated virtual environment.")
    st.stop()

MODEL_PRESENT = MODEL_PATH.exists()
MODEL_SKLEARN_VERSION = "1.9.1"            # version stored inside random_forest_model.joblib
try:
    import sklearn
    SKLEARN_VERSION = sklearn.__version__
except Exception:
    SKLEARN_VERSION = None

# ----------------------------------------------------------------------------------------------
# Constants taken from the team's code (not re-defined)
# ----------------------------------------------------------------------------------------------
N_FLOORS = S.N_FLOORS                                 # 11
FLOORS = list(range(1, N_FLOORS + 1))
STEP_MIN = S.STEP_MIN                                 # 15
STEP_H = STEP_MIN / 60.0
STEPS_PER_DAY = 1440 // STEP_MIN                      # 96
COMFORT_MIN, COMFORT_MAX = S.COMFORT_MIN, S.COMFORT_MAX    # 22 / 26 C
DAYS = S.DAY_NAMES[:5]                                # Monday..Friday (the simulated week has no weekend)
CAP_FRAC = 0.70                                       # controller.py default: cap = 70 % of baseline peak
ANCHOR = pd.Timestamp("2001-01-01")                   # a Monday; only weekday + clock time are ever shown

SCENARIO_TITLES = {"summer": "Summer (hot July day)", "heatwave": "Heatwave (preset)", "november": "November (mild)"}
KW = [f"kw_f{i}" for i in FLOORS]
TI = [f"t_in_f{i}" for i in FLOORS]
OC = [f"occ_f{i}" for i in FLOORS]
SP = [f"sp_f{i}" for i in FLOORS]

COL_BASE, COL_TB, COL_GREY = "#f59e0b", "#14b8a6", "#64748b"

ACTION_LABEL = {"pre_cool": "Pre-cool", "maintain": "Maintain", "comfort_override": "Comfort override",
                "stagger": "Staggered (waiting)", "setback": "Setback (AC off)"}
ACTION_EXPLAIN = {
    "pre_cool": "People are expected soon and the floor is warm, so cooling starts early enough for the floor to be ready on arrival.",
    "maintain": "The floor is occupied and within the comfort limit, so the controller just holds the setpoint "
                "(24 C with students, a relaxed 25.5 C with staff only).",
    "comfort_override": "An occupied floor reached the 26 C comfort limit. The guardrail forces cooling and exempts the floor from staggering.",
    "stagger": "Pre-cooling is needed, but starting it now would push estimated load over the peak cap, "
               "so this floor waits while a more urgent floor is served first.",
    "setback": "The floor is empty and nobody is due soon, so AC is switched off (setback target 28 C) to avoid cooling an empty space.",
}
ACTION_COLOR = {"pre_cool": "#3b82f6", "maintain": "#22c55e", "comfort_override": "#ef4444",
                "stagger": "#f59e0b", "setback": "#64748b"}

PERIOD_OPTIONS = {
    "Full week (Mon-Fri)": ([0, 1, 2, 3, 4], "sum"),
    "Typical weekday (Tue-Fri, per-day average)": ([1, 2, 3, 4], "avg"),
    **{d: ([i], "sum") for i, d in enumerate(DAYS)},
}

# ----------------------------------------------------------------------------------------------
# Styling (dark "control centre" cards; they carry their own colours so they work in light/dark themes)
# ----------------------------------------------------------------------------------------------
CSS = """
<style>
.tb-hero{background:linear-gradient(120deg,#0b1220 0%,#0f2a3d 55%,#0d4a4a 100%);border-radius:16px;padding:22px 28px;
  color:#e2e8f0;margin-bottom:10px;border:1px solid #1e3a4a}
.tb-hero{display:flex;align-items:center;justify-content:space-between;gap:24px}
.tb-hero-txt{flex:1;min-width:0}
.tb-logo{height:190px;width:auto;flex:0 0 auto;mix-blend-mode:screen}
@media (max-width:760px){.tb-logo{display:none}}
.tb-hero h1{margin:0;font-size:2.0rem;letter-spacing:.14em;color:#fff;font-weight:800}
.tb-hero .sub{color:#7dd3c8;font-size:1.0rem;margin-top:2px}
.tb-hero .desc{color:#cbd5e1;font-size:.92rem;margin-top:8px;max-width:980px}
.tb-flow span{display:inline-block;background:#12324a;border:1px solid #1f5f7a;color:#bfe9ff;border-radius:999px;
  padding:2px 12px;margin:8px 6px 0 0;font-size:.78rem}
.tb-sim{background:#3b2a05;border:1px solid #92650b;color:#fde68a;border-radius:10px;padding:8px 14px;font-size:.85rem;margin-bottom:10px}
.tb-mode-ai{background:#06322e;border:1px solid #14b8a6;color:#99f6e4;border-radius:10px;padding:8px 14px;font-size:.85rem;margin-bottom:10px}
.tb-mode-fb{background:#3b0d0d;border:1px solid #ef4444;color:#fecaca;border-radius:10px;padding:8px 14px;font-size:.85rem;margin-bottom:10px}
.tb-kpi{background:#0f172a;border:1px solid #1e293b;border-top:3px solid var(--acc);border-radius:12px;padding:12px 14px;height:100%;color:#e2e8f0}
.tb-kpi .l{font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:#94a3b8}
.tb-kpi .v{font-size:1.65rem;font-weight:700;color:#fff;line-height:1.25}
.tb-kpi .u{font-size:.85rem;color:#94a3b8;font-weight:500}
.tb-kpi .s{font-size:.78rem;color:#cbd5e1;margin-top:2px}
.tb-kpi .tag{font-size:.65rem;color:#fbbf24;border:1px solid #92650b;border-radius:6px;padding:0 5px;margin-left:6px;vertical-align:middle}
.tb-bldg{background:#0b1220;border:1px solid #1e293b;border-radius:14px;padding:10px;color:#e2e8f0}
.tb-roof{height:14px;background:#1e293b;border-radius:8px 8px 0 0;margin:0 18px 6px 18px}
.tb-ground{height:10px;background:#1e293b;border-radius:0 0 8px 8px;margin-top:6px}
.tb-floor{display:flex;align-items:center;gap:10px;background:#111c30;border:1px solid #1e293b;border-left:7px solid var(--tc);
  border-radius:8px;padding:5px 10px;margin:4px 0;font-size:.8rem}
.tb-floor.sel{outline:2px solid #38bdf8;background:#14243d}
.tb-floor .fl{font-weight:800;width:30px;color:#fff}
.tb-chip{border-radius:999px;padding:1px 9px;font-size:.68rem;font-weight:700;color:#06121f;min-width:118px;text-align:center}
.tb-floor .m{color:#cbd5e1;min-width:62px}
.tb-floor .m b{color:#fff}
.tb-bar{flex:1;height:6px;background:#1e293b;border-radius:4px;min-width:40px}
.tb-bar i{display:block;height:6px;border-radius:4px;background:#38bdf8}
.tb-legend{font-size:.72rem;color:#94a3b8;margin-top:6px}
.bd-scene{position:relative;border-radius:14px;border:1px solid #1e293b;padding:14px 0 0 0;overflow:visible;color:#e2e8f0}
.bd-orb{position:absolute;top:14px;right:16px;width:22px;height:22px;border-radius:50%}
.bd-wrap{width:54%;margin-left:3%;position:relative}
.bd-antenna{width:3px;height:26px;background:#475569;margin:0 auto}
.bd-roof{display:flex;justify-content:space-around;align-items:flex-end;height:18px;background:#334155;border-radius:6px 6px 0 0;padding:0 14px}
.bd-unit{width:26px;height:11px;background:#64748b;border-radius:3px 3px 0 0}
.bd-body{background:#1e293b;border-left:3px solid #334155;border-right:3px solid #334155}
.bd-floor{position:relative;display:flex;align-items:center;gap:4px;height:34px;padding:0 6px 0 2px;border-bottom:2px solid #0f172a;cursor:crosshair}
.bd-floor:hover{z-index:20;background:#27384f}
.bd-floor.sel{outline:2px solid #38bdf8;outline-offset:-2px}
.bd-fl{width:24px;font-size:.64rem;font-weight:700;color:#94a3b8;text-align:center}
.bd-win{flex:1;height:20px;border-radius:2px}
.bd-win.lit{box-shadow:0 0 6px rgba(255,255,255,.35)}
.bd-door{flex:1;height:28px;align-self:flex-end;background:#0f172a;border:2px solid #64748b;border-bottom:none;border-radius:3px 3px 0 0}
.bd-ac{width:6px;height:22px;border-radius:3px}
.bd-base{height:10px;background:#334155;margin:0 -6px}
.bd-ground{height:14px;background:#14532d;border-radius:0 0 14px 14px;margin-top:0}
.bd-tip{display:none;position:absolute;top:0;left:calc(100% + 10px);width:72%;min-width:168px;background:#0b1220;border:1px solid #38bdf8;
  border-radius:10px;padding:8px 10px;font-size:.74rem;box-shadow:0 6px 18px rgba(0,0,0,.45);color:#e2e8f0;pointer-events:none}
.bd-tip.up{top:auto;bottom:0}
.bd-floor:hover .bd-tip{display:block}
.bd-tip-h{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:5px;font-size:.85rem}
.bd-chip{border-radius:999px;padding:1px 8px;font-size:.62rem;font-weight:700;color:#06121f;white-space:nowrap}
.bd-r{display:flex;justify-content:space-between;gap:8px;padding:1px 0}
.bd-r span{color:#94a3b8}
.bd-r small{color:#94a3b8;font-weight:400}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------
def fnum(x, d=0):
    return f"{x:,.{d}f}"


def show_df(df, height=None):
    """st.dataframe that works across Streamlit versions (use_container_width was replaced by width='stretch')."""
    kw = {"hide_index": True}
    if height:
        kw["height"] = height
    try:
        new_api = tuple(int(x) for x in st.__version__.split(".")[:2]) >= (1, 49)
    except Exception:
        new_api = False
    if new_api:
        st.dataframe(df, width="stretch", **kw)
    else:
        st.dataframe(df, use_container_width=True, **kw)


def show_line(df, colors, height=320):
    try:
        st.line_chart(df, color=colors, height=height)
    except Exception:
        st.line_chart(df, height=height)


def show_bar(df, colors, height=320):
    try:
        st.bar_chart(df, color=colors, height=height)
    except Exception:
        st.bar_chart(df, height=height)


def kpi_card(label, value, unit="", sub="", accent="#14b8a6", tag=""):
    tag_html = f'<span class="tag">{tag}</span>' if tag else ""
    return (f'<div class="tb-kpi" style="--acc:{accent}"><div class="l">{label}{tag_html}</div>'
            f'<div class="v">{value} <span class="u">{unit}</span></div><div class="s">{sub}</div></div>')


def class_label(r):
    return f"{r.day[:3]} {r.start_time_24hr}-{r.end_time_24hr} | {r.floor_id} | {r.room_id} | {int(r.class_size)} students"


# ----------------------------------------------------------------------------------------------
# Data access (cached)
# ----------------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_timetable() -> pd.DataFrame:
    return pd.read_csv(TIMETABLE_PATH)


@st.cache_data(show_spinner=False)
def load_optional_csv(rel: str):
    p = BASE_DIR / rel
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_optional_json(rel: str):
    p = BASE_DIR / rel
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ----------------------------------------------------------------------------------------------
# Simulation layer: calls the team's real code (cached per parameter set)
# ----------------------------------------------------------------------------------------------
def _register_scenario(tmin: float, tmax: float, doy: int) -> str:
    """BuildingSim looks scenarios up in simulator.SCENARIOS; register a runtime entry (originals untouched)."""
    key = f"app_{tmin:.2f}_{tmax:.2f}_{doy}"
    S.SCENARIOS.setdefault(key, dict(tmin=float(tmin), tmax=float(tmax), doy=int(doy)))
    return key


def _simulate(ctrl, scen_key: str, seed: int, cancels: tuple):
    """Same loop as controller.one_run(), plus a hook to cancel classes between reset() and stepping.

    Cancellation = nobody attends that session (attending=0, cancelled=1).  Done AFTER reset() so every
    other class keeps exactly the same random attendance as the un-cancelled run.
    """
    sim = S.BuildingSim(str(TIMETABLE_PATH), scenario=scen_key, seed=seed)
    sim.reset(start_day="Monday", n_days=5)
    if cancels and not (hasattr(sim, "attendance") and hasattr(sim, "_build_occupancy")):
        raise RuntimeError("Cancel-class needs simulator.BuildingSim.attendance and ._build_occupancy(), which this version "
                           "of simulator.py no longer provides. Check whether simulator.py was changed.")
    for day, room, start in cancels:
        m = (sim.attendance.day == day) & (sim.attendance.room_id == room) & (sim.attendance.start_time == start)
        if int(m.sum()) != 1:
            raise ValueError(f"Class {day} {room} {start} not found in the simulated attendance table.")
        sim.attendance.loc[m, ["attending", "cancelled"]] = [0, 1]
    if cancels:
        sim._build_occupancy()                      # private method of the building simulator (see module docstring)
    while not sim.done:
        sim.step(ctrl(sim))
    h = sim.history()
    k = sim.kpis()
    t, o = h[TI].to_numpy(), h[OC].to_numpy()
    hot = (t > COMFORT_MAX) & (o >= 1)
    k["violations"] = int(hot.sum())
    k["degree_hours"] = float(((t - COMFORT_MAX).clip(min=0) * (o >= 1)).sum() * STEP_MIN / 60)
    return h, k


@st.cache_data(show_spinner=False)
def run_baseline(tmin, tmax, doy, cancels, seed):
    key = _register_scenario(tmin, tmax, doy)
    return _simulate(S.FixedSchedule(), key, seed, cancels)


@st.cache_data(show_spinner=False)
def run_tower_brain(tmin, tmax, doy, cancels, informed, use_model, seed, cap_kw):
    """Tower Brain run.  use_model=True -> the Random Forest occupancy model is called live (AI-integrated).
    use_model=False -> timetable-only prediction (NO AI; fallback) - only ever used when the user chooses it."""
    key = _register_scenario(tmin, tmax, doy)
    tt = load_timetable()
    tt_pred = tt
    if cancels and informed:        # the "registrar" removes cancelled classes from the timetable the predictor sees
        drop = pd.Series(False, index=tt.index)
        for day, room, start in cancels:
            drop |= (tt.day == day) & (tt.room_id == room) & (tt.start_time_24hr == start)
        tt_pred = tt[~drop]
    pred = C.LiveModelPredictor(tt_pred, MODEL_PATH) if use_model else C.OccupancyPredictor(tt_pred, None)
    ctrl = C.TowerBrainController(pred, cap_kw=cap_kw)
    h, k = _simulate(ctrl, key, seed, cancels)
    log = pd.DataFrame(ctrl.log)
    # The controller tags each reason with "[prediction: Person 2 model]"; show a neutral name on the dashboard (display only).
    if "reason" in log.columns:
        log["reason"] = log["reason"].str.replace("Person 2 model", "occupancy model", regex=False)
    return dict(h=h, k=k, log=log, sources=dict(ctrl.source_counts), cap=float(cap_kw),
                n_forecasts=int(getattr(pred, "n_forecasts", 0)))


@st.cache_data(show_spinner=False)
def run_comparison_policy(kind, tmin, tmax, doy, cancels, seed, cap_kw):
    """Extra comparison policies used only on the Model & Validation tab."""
    key = _register_scenario(tmin, tmax, doy)
    if kind == "tuned_timer":                     # the tuned timer, exactly as controller.py defines it
        h, k = _simulate(S.FixedSchedule(7 * 60, 17 * 60 + 30, 24.0, 5, name="tuned_timer"), key, seed, cancels)
    else:                                         # "timetable_only": Tower Brain rules, no AI model
        tt = load_timetable()
        if cancels:
            drop = pd.Series(False, index=tt.index)
            for day, room, start in cancels:
                drop |= (tt.day == day) & (tt.room_id == room) & (tt.start_time_24hr == start)
            tt = tt[~drop]
        ctrl = C.TowerBrainController(C.OccupancyPredictor(tt, None), cap_kw=cap_kw)
        h, k = _simulate(ctrl, key, seed, cancels)
    return h, k


# ----------------------------------------------------------------------------------------------
# KPI maths (all from the simulator's own history columns)
# ----------------------------------------------------------------------------------------------
def with_ts(h: pd.DataFrame) -> pd.DataFrame:
    h = h.copy()
    h["ts"] = ANCHOR + pd.to_timedelta(h["day_idx"] * 1440 + h["minute"], unit="m")
    return h


def period_stats(h: pd.DataFrame, days, mode):
    d = h[h.day_idx.isin(days)]
    kwh = float(d.total_kw.sum() * STEP_H)
    if mode == "avg":
        kwh /= len(days)
    i = d.total_kw.idxmax()
    t, o = d[TI].to_numpy(), d[OC].to_numpy()
    return dict(kwh=kwh, peak=float(d.total_kw.max()), peak_when=f"{h.day[i][:3]} {h.time[i]}",
                comfort=float(100 * d.pm_ok.sum() / max(d.pm_total.sum(), 1e-9)),
                violations=int(((t > COMFORT_MAX) & (o >= 1)).sum()))


def floor_period(h: pd.DataFrame, days, mode):
    d = h[h.day_idx.isin(days)]
    kwh = d[KW].sum().to_numpy() * STEP_H
    if mode == "avg":
        kwh = kwh / len(days)
    t, o = d[TI].to_numpy(), d[OC].to_numpy()
    ok = (t >= COMFORT_MIN) & (t <= COMFORT_MAX)
    comfort = 100 * (o * ok).sum(axis=0) / np.maximum(o.sum(axis=0), 1e-9)
    return kwh, comfort


def class_start_temps(hb, ht, tt, cancels):
    """Floor temperature (15-min average) in the step in which each non-cancelled class starts."""
    rows = []
    cancelled = set(cancels)
    for r in tt.itertuples():
        if (r.day, r.room_id, r.start_time_24hr) in cancelled or r.day not in DAYS:
            continue
        i = DAYS.index(r.day) * STEPS_PER_DAY + S.to_min(r.start_time_24hr) // STEP_MIN
        f = int(r.floor_id[1:])
        rows.append((hb[f"t_in_f{f}"].iloc[i], ht[f"t_in_f{f}"].iloc[i]))
    a = np.array(rows)
    return dict(n=len(a), base_mean=a[:, 0].mean(), tb_mean=a[:, 1].mean(),
                base_hot=int((a[:, 0] > COMFORT_MAX).sum()), tb_hot=int((a[:, 1] > COMFORT_MAX).sum()))


def changed_decisions(log_a: pd.DataFrame, log_b: pd.DataFrame) -> pd.DataFrame:
    m = log_a.merge(log_b, on=["timestamp", "floor_id"], suffixes=("_before", "_after"))
    chg = ((m.action_before != m.action_after) | (m.ac_on_before != m.ac_on_after) |
           (m.new_setpoint_before.fillna(-1) != m.new_setpoint_after.fillna(-1)))
    return m[chg]


# ----------------------------------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------------------------------
# Tower Brain logo, embedded so app.py stays one self-contained file (transparent WebP made from the supplied JPG)
LOGO_WEBP_B64 = "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"
st.markdown(
    '<div class="tb-hero"><div class="tb-hero-txt"><h1>TOWER BRAIN</h1>'
    '<div class="sub">AI-driven cooling optimisation for an 11-floor smart building</div>'
    '<div class="desc">Tower Brain treats the building as 11 floor-zones and, every 15 minutes, uses the timetable, '
    'a learned occupancy forecast and floor temperatures to decide when each floor is pre-cooled, staggered, '
    'held or switched off - while a comfort guardrail protects occupied floors.</div>'
    '<div class="tb-flow"><span>1 Monitor</span><span>2 Analyze</span><span>3 Control &amp; Optimize</span><span>4 Report</span></div></div>'
    '<img class="tb-logo" alt="Tower Brain logo" src="data:image/webp;base64,' + LOGO_WEBP_B64 + '"/></div>',
    unsafe_allow_html=True)
st.markdown('<div class="tb-sim"><b>SIMULATED PROOF OF CONCEPT.</b> Weather, attendance, thermal behaviour and the '
            'occupancy "sensors" are all synthetic - nothing here is measured data from a real building. '
            'Cost and CO<sub>2</sub> are <b>estimates</b> from the editable assumptions in the sidebar.</div>',
            unsafe_allow_html=True)

# ----------------------------------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------------------------------
tt_all = load_timetable()
st.session_state.setdefault("cancels", [])

sb = st.sidebar
sb.markdown("## Control panel")

sb.markdown("**1 - Scenario**")
scen_name = sb.selectbox("Weather scenario", list(SCENARIO_TITLES), format_func=lambda k: SCENARIO_TITLES[k],
                         label_visibility="collapsed")
base_sc = S.SCENARIOS[scen_name]
offset = sb.slider("Heatwave adjustment: extra outdoor heat (degC)", 0.0, 6.0, 0.0, 0.5,
                   help="Adds this many degrees to the whole outdoor-temperature curve of the selected scenario "
                        "and re-runs BOTH the fixed-schedule baseline and Tower Brain under the hotter weather.")
tmin, tmax, doy = base_sc["tmin"] + offset, base_sc["tmax"] + offset, base_sc["doy"]
sb.caption(f"Outdoor range used: **{tmin:.1f} - {tmax:.1f} degC**" + ("  (unmodified preset)" if offset == 0 else f"  (preset +{offset:g} degC)"))

sb.markdown("**2 - Occupancy forecast source**")
mode_opts = ["AI: Random Forest occupancy model (live)"] if MODEL_PRESENT else []
mode_opts.append("NO-AI fallback: timetable only")
mode_sel = sb.radio("Forecast source", mode_opts, label_visibility="collapsed",
                    help="AI mode loads the trained occupancy model (never retrained). The fallback uses the timetable only "
                         "(class size x 0.82) and is shown with a red banner - it is never substituted silently.")
use_model = mode_sel.startswith("AI")
if not MODEL_PRESENT:
    sb.error("`data/random_forest_model.joblib` not found - the AI-integrated mode is unavailable. "
             "Copy the trained model file there. Only the NO-AI timetable fallback can run.")
elif SKLEARN_VERSION and SKLEARN_VERSION != MODEL_SKLEARN_VERSION:
    sb.warning(f"scikit-learn {SKLEARN_VERSION} installed, but the model was saved with {MODEL_SKLEARN_VERSION}. "
               f"If loading fails, run: pip install scikit-learn=={MODEL_SKLEARN_VERSION}")

sb.markdown("**3 - Cancel a class**")
cancel_day = sb.selectbox("Day", DAYS, index=1, key="cancel_day")
day_classes = tt_all[tt_all.day == cancel_day].sort_values(["start_time_24hr", "floor_id", "room_id"])
cls_map = {class_label(r): (r.day, r.room_id, r.start_time_24hr) for r in day_classes.itertuples()}
cls_pick = sb.selectbox("Class", list(cls_map), key="cancel_pick")
informed = sb.checkbox("Timetable system is notified of the cancellation", value=True,
                       help="ON: the cancelled class is also removed from the timetable the predictor sees (e.g. the registrar "
                            "updates the schedule). OFF: reality changes but the predictor still expects the class - shows "
                            "that the model largely echoes the timetable.")
b1, b2 = sb.columns(2)
if b1.button("Cancel class"):
    if cls_map[cls_pick] not in st.session_state["cancels"]:
        st.session_state["cancels"].append(cls_map[cls_pick])
if b2.button("Restore all"):
    st.session_state["cancels"] = []
cancels = tuple(sorted(st.session_state["cancels"]))
if cancels:
    for c in cancels:
        sb.caption("Cancelled: " + next((k for k, v in cls_map_all.items() if v == c), f"{c[0]} {c[1]} {c[2]}")
                   if (cls_map_all := {class_label(r): (r.day, r.room_id, r.start_time_24hr) for r in tt_all.itertuples()}) else "")
else:
    sb.caption("No classes cancelled.")
sb.caption("Tested in the summer scenario: cancelling Tue 16:00 F5/R502, Tue 15:00 F3/R303 or Tue 15:00 F11/R1103 "
           "visibly changes that floor's decisions. Most other single cancellations change very little.")

sb.markdown("**4 - Inspect**")
floor_sel = sb.selectbox("Floor to inspect", FLOORS, index=5, format_func=lambda n: f"F{n}", key="floor_sel")
period_name = sb.selectbox("KPI period", list(PERIOD_OPTIONS), index=0)
p_days, p_mode = PERIOD_OPTIONS[period_name]

sb.markdown("**5 - Estimate assumptions (editable)**")
tariff = sb.number_input("Electricity tariff (AED per kWh)", 0.0, 2.0, 0.20, 0.01, format="%.3f",
                         help="PLACEHOLDER ASSUMPTION. The project files contain no tariff, and this default has NOT been verified. "
                              "Replace it with the tariff your team agrees on (and cite the source) before quoting AED figures.")
co2f = sb.number_input("CO2 emission factor (kg CO2e per kWh)", 0.0, 1.5, 0.40, 0.01, format="%.3f",
                       help="PLACEHOLDER ASSUMPTION. The project files contain no emission factor, and this default has NOT been verified. "
                              "Replace it with the factor your team agrees on (and cite the source) before quoting CO2 figures.")
with sb.expander("Advanced"):
    seed = int(st.number_input("Random seed (attendance draw)", 0, 99, 1, 1,
                               help="Seed 1 reproduces the team's saved results. Other seeds test robustness."))

# ----------------------------------------------------------------------------------------------
# Run (or fetch from cache) the simulations
# ----------------------------------------------------------------------------------------------
try:
    with st.spinner("Running the fixed-schedule baseline simulation..."):
        hb0, kb0 = run_baseline(tmin, tmax, doy, (), seed)
    cap_kw = CAP_FRAC * kb0["peak_kw_15min"]            # same rule as controller.py: 70 % of baseline peak
    with st.spinner("Running Tower Brain (controller + occupancy model, ~6 s the first time)..."):
        tb0 = run_tower_brain(tmin, tmax, doy, (), True, use_model, seed, cap_kw)
        if cancels:
            hb, kb = run_baseline(tmin, tmax, doy, cancels, seed)
            tb = run_tower_brain(tmin, tmax, doy, cancels, informed, use_model, seed, cap_kw)
        else:
            hb, kb, tb = hb0, kb0, tb0
except Exception as exc:
    st.error(f"The simulation could not run: **{type(exc).__name__}: {exc}**")
    if use_model:
        st.info("If this mentions scikit-learn or unpickling, install the version the model was saved with: "
                f"`pip install scikit-learn=={MODEL_SKLEARN_VERSION}`. You can also pick the NO-AI fallback in the sidebar "
                "(clearly labelled; it does not use the occupancy model).")
    st.stop()

ht, log = tb["h"], tb["log"]
hb_ts, ht_ts = with_ts(hb), with_ts(ht)

if use_model:
    st.markdown(f'<div class="tb-mode-ai"><b>AI-INTEGRATED RUN.</b> The Random Forest occupancy model '
                f'(<code>{MODEL_PATH.name}</code>) is called live every 15 minutes for a 1-hour-ahead forecast per floor '
                f'({tb["n_forecasts"]:,} forecasts this run).</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="tb-mode-fb"><b>NO-AI FALLBACK RUN.</b> Occupancy forecasts come from the timetable only '
                '(class size x 0.82). The occupancy model is <b>not</b> used in these results.</div>', unsafe_allow_html=True)
if cancels:
    st.markdown(f'<div class="tb-sim"><b>Scenario modified:</b> {len(cancels)} class(es) cancelled '
                f'(timetable {"notified" if informed else "NOT notified"}). Baseline and Tower Brain both ran under this change.</div>',
                unsafe_allow_html=True)

sb_stats = period_stats(hb, p_days, p_mode)         # baseline
st_stats = period_stats(ht, p_days, p_mode)         # Tower Brain

# ----------------------------------------------------------------------------------------------
# KPI cards
# ----------------------------------------------------------------------------------------------
unit_e = "kWh/day" if p_mode == "avg" else "kWh"
saved = sb_stats["kwh"] - st_stats["kwh"]
saved_pct = 100 * saved / sb_stats["kwh"] if sb_stats["kwh"] else 0.0
peak_red = sb_stats["peak"] - st_stats["peak"]
peak_red_pct = 100 * peak_red / sb_stats["peak"] if sb_stats["peak"] else 0.0
cost_b, cost_t = sb_stats["kwh"] * tariff, st_stats["kwh"] * tariff
co2_b, co2_t = sb_stats["kwh"] * co2f, st_stats["kwh"] * co2f
good, bad = "#22c55e", "#ef4444"

st.markdown(f"**KPIs - {period_name} - {SCENARIO_TITLES[scen_name]}"
            f"{f' +{offset:g} degC' if offset else ''}** &nbsp;<span style='color:#94a3b8;font-size:.8rem'>(simulated)</span>",
            unsafe_allow_html=True)
r1 = st.columns(4)
r1[0].markdown(kpi_card("Tower Brain energy", fnum(st_stats["kwh"]), unit_e,
                        f"Baseline: {fnum(sb_stats['kwh'])} {unit_e}", COL_TB), unsafe_allow_html=True)
r1[1].markdown(kpi_card("Energy saved", fnum(saved), unit_e, "vs fixed-schedule baseline",
                        good if saved >= 0 else bad), unsafe_allow_html=True)
r1[2].markdown(kpi_card("Savings", f"{saved_pct:.1f}", "%", "cooling electricity (compressor + fans)",
                        good if saved >= 0 else bad), unsafe_allow_html=True)
r1[3].markdown(kpi_card("Peak demand", fnum(st_stats["peak"], 1), "kW",
                        f"Baseline {fnum(sb_stats['peak'], 1)} kW | {peak_red:+.1f} kW ({peak_red_pct:+.1f}%)",
                        good if peak_red >= 0 else bad), unsafe_allow_html=True)
r2 = st.columns(3)
r2[0].markdown(kpi_card("Comfort (occupied person-time in 22-26 degC)", f"{st_stats['comfort']:.1f}", "%",
                        f"Baseline {sb_stats['comfort']:.1f}% | occupied floor-intervals >26 degC: "
                        f"{st_stats['violations']} vs {sb_stats['violations']}", COL_TB), unsafe_allow_html=True)
r2[1].markdown(kpi_card("Estimated cost", fnum(cost_t), f"AED{'/day' if p_mode == 'avg' else ''}",
                        f"Baseline {fnum(cost_b)} AED | saves {fnum(cost_b - cost_t)} AED @ {tariff:.3f} AED/kWh",
                        "#a78bfa", tag="ESTIMATE"), unsafe_allow_html=True)
co2_unit = "kg CO2e" + ("/day" if p_mode == "avg" else "")
r2[2].markdown(kpi_card("Estimated CO2", fnum(co2_t), co2_unit,
                        f"Baseline {fnum(co2_b)} | avoided {fnum(co2_b - co2_t)} @ {co2f:.3f} kg/kWh",
                        "#a78bfa", tag="ESTIMATE"), unsafe_allow_html=True)

verb = "less" if saved >= 0 else "more"
st.caption(
    f"Readout (simulated): over **{period_name.lower()}**, Tower Brain used {abs(saved_pct):.1f}% {verb} cooling electricity than the fixed schedule "
    f"and its peak was {abs(peak_red_pct):.1f}% {'lower' if peak_red >= 0 else 'higher'} ({st_stats['peak_when']} vs {sb_stats['peak_when']}). "
    "Results are specific to this synthetic building, timetable and seed. Full-week figures include Monday's heat-soak recovery "
    "(the simulator starts Monday with floors at 40-45 degC); the Tue-Fri view avoids that.")

# ----------------------------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------------------------
tab_cc, tab_energy, tab_floors, tab_log, tab_cancel, tab_model = st.tabs(
    ["Control centre", "Energy & power", "Floor comparison", "Decision log", "Cancel-class impact", "Model, validation & assumptions"])

# ============================ TAB 1: control centre ============================
with tab_cc:
    cL, cR = st.columns([5, 7])
    with cL:
        st.markdown("#### 11-floor building view")
        day_sel = st.select_slider("Day", options=DAYS, value="Tuesday", key="day_sel")
        times = [f"{m // 60:02d}:{m % 60:02d}" for m in range(0, 1440, STEP_MIN)]
        time_sel = st.select_slider("Time of day", options=times, value="09:00", key="time_sel")
        view = st.radio("Show", ["Tower Brain", "Fixed-schedule baseline"], horizontal=True, key="view_sel")
        row_i = DAYS.index(day_sel) * STEPS_PER_DAY + times.index(time_sel)
        h_view = ht if view == "Tower Brain" else hb
        rw = h_view.iloc[row_i]
        stamp = f"{day_sel} {time_sel}"
        lg = log[log.timestamp == stamp].set_index("floor_id") if view == "Tower Brain" else None

        # ---- sky colour follows the time of day ----
        hour = times.index(time_sel) * STEP_MIN / 60.0
        if 7.0 <= hour < 18.0:
            sky, orb = "linear-gradient(#5aa9e6,#cfe8fb)", "background:#fde047;box-shadow:0 0 14px #fde047"
        elif 5.5 <= hour < 7.0 or 18.0 <= hour < 19.5:
            sky, orb = "linear-gradient(#5b6fb0,#f59e6b)", "background:#fb923c;box-shadow:0 0 14px #fb923c"
        else:
            sky, orb = "linear-gradient(#0b1530,#1b2a4a)", "background:#e2e8f0;box-shadow:0 0 10px #e2e8f0"

        N_WIN = 6
        floors_html = []
        for n in reversed(FLOORS):                                  # F11 on top
            t_in, occ, kw, sp = rw[f"t_in_f{n}"], rw[f"occ_f{n}"], rw[f"kw_f{n}"], rw[f"sp_f{n}"]
            if lg is not None and f"F{n}" in lg.index:
                e = lg.loc[f"F{n}"]
                act = e["action"]
                status, chip = ACTION_LABEL[act], ACTION_COLOR[act]
                pred_txt = f' <small>(forecast {e["predicted_occupancy"]:.0f})</small>'
            else:
                on = not np.isnan(sp)
                status, chip = ("AC ON (fixed)" if on else "AC OFF"), ("#22c55e" if on else "#64748b")
                pred_txt = ""
            tcol = "#ef4444" if t_in > COMFORT_MAX else "#f59e0b" if t_in >= 25.0 else "#22c55e" if t_in >= COMFORT_MIN else "#38bdf8"
            sp_txt = "AC off" if np.isnan(sp) else f"{sp:g} &deg;C"
            n_lit = 0 if occ < 0.5 else max(1, round(min(occ / 100, 1) * N_WIN))   # bright windows ~ people on the floor
            wins = []
            for w in range(N_WIN):
                if n == 1 and w == N_WIN // 2:
                    wins.append('<span class="bd-door"></span>')       # ground-floor entrance
                    continue
                lit = w < n_lit
                wins.append(f'<span class="bd-win{" lit" if lit else ""}" style="background:{tcol};opacity:{1.0 if lit else 0.3}"></span>')
            tip = (f'<div class="bd-tip{" up" if n <= 4 else ""}">'
                   f'<div class="bd-tip-h"><b>Floor {n}</b><span class="bd-chip" style="background:{chip}">{status}</span></div>'
                   f'<div class="bd-r"><span>Occupancy</span><b>{occ:.0f} people{pred_txt}</b></div>'
                   f'<div class="bd-r"><span>Temperature</span><b style="color:{tcol}">{t_in:.1f} &deg;C</b></div>'
                   f'<div class="bd-r"><span>Energy usage</span><b>{kw:.1f} kW <small>({kw * STEP_H:.2f} kWh/15 min)</small></b></div>'
                   f'<div class="bd-r"><span>Setpoint (SP)</span><b>{sp_txt}</b></div></div>')
            floors_html.append(
                f'<div class="bd-floor{" sel" if n == floor_sel else ""}"><span class="bd-fl">F{n}</span>'
                + "".join(wins) + f'<span class="bd-ac" style="background:{chip}"></span>' + tip + '</div>')

        st.markdown(
            f'<div class="bd-scene" style="background:{sky}"><div class="bd-orb" style="{orb}"></div>'
            '<div class="bd-wrap"><div class="bd-antenna"></div>'
            '<div class="bd-roof"><span class="bd-unit"></span><span class="bd-unit"></span><span class="bd-unit"></span></div>'
            '<div class="bd-body">' + "".join(floors_html) + '</div><div class="bd-base"></div></div>'
            '<div class="bd-ground"></div></div>'
            '<div class="tb-legend">Hover a floor to see its occupancy, temperature, energy usage and setpoint. '
            'Window tint = indoor temperature: blue &lt;22 | green 22-25 | amber 25-26 | red &gt;26 &deg;C. '
            'Brighter windows = more people. Side strip = AC status (teal/green on, grey off; Tower Brain colours match the action). '
            f'Total now: {rw["total_kw"]:.1f} kW, outdoor {rw["t_out"]:.1f} &deg;C.</div>',
            unsafe_allow_html=True)

        with st.expander("All floors at this time (table, for touch screens)"):
            show_df(pd.DataFrame({
                "Floor": [f"F{n}" for n in reversed(FLOORS)],
                "People": [round(rw[f"occ_f{n}"]) for n in reversed(FLOORS)],
                "Temp (C)": [round(rw[f"t_in_f{n}"], 1) for n in reversed(FLOORS)],
                "Energy (kW)": [round(rw[f"kw_f{n}"], 1) for n in reversed(FLOORS)],
                "SP (C)": [None if np.isnan(rw[f"sp_f{n}"]) else rw[f"sp_f{n}"] for n in reversed(FLOORS)]}))
        if lg is not None and f"F{floor_sel}" in lg.index:
            e = lg.loc[f"F{floor_sel}"]
            st.info(f"**F{floor_sel} at {stamp}: {ACTION_LABEL[e['action']]}.** {e['reason']}")

    with cR:
        st.markdown(f"#### Power demand - {day_sel}")
        d_i = DAYS.index(day_sel)
        pf = pd.DataFrame({"Baseline (fixed schedule)": hb_ts[hb_ts.day_idx == d_i].set_index("ts").total_kw,
                           "Tower Brain": ht_ts[ht_ts.day_idx == d_i].set_index("ts").total_kw})
        pf["Tower Brain peak cap (70% of baseline peak)"] = tb["cap"]
        show_line(pf, [COL_BASE, COL_TB, COL_GREY], 300)
        st.caption("Building cooling electricity (kW, 15-minute averages). The cap is the controller's *soft* target - "
                   "floors at their deadline and occupied floors are never delayed, so the peak can exceed it.")
        st.markdown("#### Cumulative energy - full week")
        ce = pd.DataFrame({"Baseline (fixed schedule)": (hb_ts.set_index("ts").total_kw.cumsum() * STEP_H),
                           "Tower Brain": (ht_ts.set_index("ts").total_kw.cumsum() * STEP_H)})
        show_line(ce, [COL_BASE, COL_TB], 260)
        st.caption("kWh accumulated since Monday 00:00.")

# ============================ TAB 2: energy & power ============================
with tab_energy:
    scope = st.radio("Chart range", ["Selected day", "Full week"], horizontal=True, key="energy_scope")
    sel_day = DAYS.index(st.session_state.get("day_sel", "Tuesday"))
    if scope == "Selected day":
        b_use, t_use = hb_ts[hb_ts.day_idx == sel_day], ht_ts[ht_ts.day_idx == sel_day]
        st.caption(f"Showing {DAYS[sel_day]} (change the day on the Control-centre tab).")
    else:
        b_use, t_use = hb_ts, ht_ts
    st.markdown("#### Power demand (kW)")
    pf2 = pd.DataFrame({"Baseline (fixed schedule)": b_use.set_index("ts").total_kw, "Tower Brain": t_use.set_index("ts").total_kw})
    pf2["Peak cap"] = tb["cap"]
    show_line(pf2, [COL_BASE, COL_TB, COL_GREY], 330)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Energy per day (kWh)")
        dd = pd.DataFrame({"Baseline (fixed schedule)": hb.groupby("day_idx").total_kw.sum() * STEP_H,
                           "Tower Brain": ht.groupby("day_idx").total_kw.sum() * STEP_H})
        dd.index = [i + 1 for i in dd.index]
        dd.index.name = "Day (1=Mon ... 5=Fri)"
        show_bar(dd, [COL_BASE, COL_TB], 300)
        dtab = pd.DataFrame({"Day": DAYS, "Baseline kWh": dd.iloc[:, 0].round(1).to_numpy(), "Tower Brain kWh": dd.iloc[:, 1].round(1).to_numpy()})
        dtab["Saved kWh"] = (dtab["Baseline kWh"] - dtab["Tower Brain kWh"]).round(1)
        dtab["Saved %"] = (100 * dtab["Saved kWh"] / dtab["Baseline kWh"]).round(1)
        show_df(dtab)
    with c2:
        st.markdown("#### Outdoor temperature and occupancy")
        show_line(pd.DataFrame({"Outdoor temperature (degC)": t_use.set_index("ts").t_out}), [COL_BASE], 150)
        show_line(pd.DataFrame({"People in building (simulated counter)": t_use.set_index("ts").people}), ["#38bdf8"], 150)
        st.caption("Occupancy is identical in both runs (same seed), so the comparison is like-for-like.")
    st.markdown("#### Summary for the selected KPI period")
    summ = pd.DataFrame({
        "Metric": [f"Energy ({unit_e})", "Peak demand (kW)", "Peak at", "Comfort (%)", "Occupied floor-intervals >26 degC"],
        "Baseline": [round(sb_stats["kwh"], 1), round(sb_stats["peak"], 1), sb_stats["peak_when"], round(sb_stats["comfort"], 2), sb_stats["violations"]],
        "Tower Brain": [round(st_stats["kwh"], 1), round(st_stats["peak"], 1), st_stats["peak_when"], round(st_stats["comfort"], 2), st_stats["violations"]]})
    show_df(summ)
    st.download_button("Download KPI summary (CSV)", summ.to_csv(index=False).encode("utf-8"), "tower_brain_kpis.csv", "text/csv")

# ============================ TAB 3: floor comparison ============================
with tab_floors:
    fb_kwh, fb_comf = floor_period(hb, p_days, p_mode)
    ft_kwh, ft_comf = floor_period(ht, p_days, p_mode)
    st.markdown(f"#### Cooling energy by floor - {period_name}")
    fe = pd.DataFrame({"Baseline (fixed schedule)": fb_kwh, "Tower Brain": ft_kwh}, index=FLOORS)
    fe.index.name = "Floor (1 = F1 ... 11 = F11)"
    show_bar(fe, [COL_BASE, COL_TB], 300)
    ftab = pd.DataFrame({"Floor": [f"F{n}" for n in FLOORS], f"Baseline {unit_e}": fb_kwh.round(1), f"Tower Brain {unit_e}": ft_kwh.round(1),
                         "Saved": (fb_kwh - ft_kwh).round(1), "Saved %": (100 * (fb_kwh - ft_kwh) / np.maximum(fb_kwh, 1e-9)).round(1),
                         "Comfort baseline %": fb_comf.round(1), "Comfort Tower Brain %": ft_comf.round(1)})
    show_df(ftab)
    st.caption("Per-floor comfort uses the 15-minute average temperature weighted by people on that floor, so it differs slightly "
               "from the simulator's per-minute building KPI.")

    cs = class_start_temps(hb, ht, tt_all, cancels)
    st.markdown("#### Temperature at class start (derived from the simulation)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Classes checked", cs["n"])
    m2.metric("Mean floor temp, baseline", f"{cs['base_mean']:.1f} degC")
    m3.metric("Mean floor temp, Tower Brain", f"{cs['tb_mean']:.1f} degC")
    m4.metric("Class starts above 26 degC", f"{cs['base_hot']} / {cs['tb_hot']}", help="baseline / Tower Brain")
    st.caption("Tower Brain pre-cools toward 25.8 degC, not lower, so floors are typically a little warmer at class start than under the "
               "baseline (23.5 degC setpoint) while still inside the comfort range. This is a deliberate energy trade-off, not extra comfort.")

    st.markdown(f"#### Inspect F{floor_sel}")
    fscope = st.radio("Range", ["Selected day", "Full week"], horizontal=True, key="floor_scope")
    fday = DAYS.index(st.session_state.get("day_sel", "Tuesday"))
    bu, tu = (hb_ts[hb_ts.day_idx == fday], ht_ts[ht_ts.day_idx == fday]) if fscope == "Selected day" else (hb_ts, ht_ts)
    ft = pd.DataFrame({"Baseline (degC)": bu.set_index("ts")[f"t_in_f{floor_sel}"], "Tower Brain (degC)": tu.set_index("ts")[f"t_in_f{floor_sel}"]})
    ft["Comfort min 22"], ft["Comfort max 26"] = COMFORT_MIN, COMFORT_MAX
    st.markdown("Indoor temperature")
    show_line(ft, [COL_BASE, COL_TB, COL_GREY, COL_GREY], 280)
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("Floor power (kW)")
        show_line(pd.DataFrame({"Baseline": bu.set_index("ts")[f"kw_f{floor_sel}"], "Tower Brain": tu.set_index("ts")[f"kw_f{floor_sel}"]}), [COL_BASE, COL_TB], 230)
    with g2:
        st.markdown("People on floor (simulated counter)")
        show_line(pd.DataFrame({"People": tu.set_index("ts")[f"occ_f{floor_sel}"]}), ["#38bdf8"], 230)
    fl_classes = tt_all[(tt_all.floor_id == f"F{floor_sel}") & (tt_all.day == DAYS[fday])].sort_values("start_time_24hr").copy()
    fl_classes["status"] = ["CANCELLED" if (r.day, r.room_id, r.start_time_24hr) in cancels else "scheduled" for r in fl_classes.itertuples()]
    st.markdown(f"Timetable for F{floor_sel} on {DAYS[fday]}")
    show_df(fl_classes[["room_id", "start_time_24hr", "end_time_24hr", "class_size", "status"]].rename(
        columns={"room_id": "Room", "start_time_24hr": "Start", "end_time_24hr": "End", "class_size": "Class size", "status": "Status"}))

# ============================ TAB 4: decision log ============================
with tab_log:
    st.markdown("#### What is the controller doing, and why?")
    st.caption(("Real output of `controller.TowerBrainController` for this run - nothing written by hand. " if use_model else
                "NO-AI FALLBACK run: forecasts come from the timetable only. ") +
               "`Est. kW` is the controller's own estimate (not the simulator's metered power).")
    sc = tb["sources"]
    st.write(f"Decisions by prediction source: **model only {sc.get('model', 0):,}** | **model + timetable for arrivals beyond 1 h {sc.get('mixed', 0):,}** | "
             f"**timetable only {sc.get('fallback', 0):,}**. (The model only looks 1 hour ahead, so pre-cool timing beyond that uses the "
             "timetable by design; the first hour of Monday has no forecast yet.)")
    f1, f2, f3 = st.columns([2, 2, 1])
    fl_f = f1.multiselect("Floors", [f"F{n}" for n in FLOORS], default=[f"F{floor_sel}"])
    ac_f = f2.multiselect("Actions", list(ACTION_LABEL.values()), default=list(ACTION_LABEL.values()))
    chg_only = f3.checkbox("State changes only", value=True, help="Hide repeated identical decisions (same AC state, setpoint and action as the previous step).")
    lg_all = log.copy()
    lg_all["_sp"] = lg_all.new_setpoint.where(lg_all.ac_on == 1)
    key = lg_all.action + "|" + lg_all._sp.fillna(-1).astype(str)
    lg_all["_chg"] = key != key.groupby(lg_all.floor_id).shift()
    view_log = lg_all
    if fl_f:
        view_log = view_log[view_log.floor_id.isin(fl_f)]
    view_log = view_log[view_log.action.map(ACTION_LABEL).isin(ac_f)]
    if chg_only:
        view_log = view_log[view_log._chg]
    disp = pd.DataFrame({
        "Time": view_log.timestamp, "Floor": view_log.floor_id, "Action": view_log.action.map(ACTION_LABEL),
        "AC": np.where(view_log.ac_on == 1, "ON", "OFF"), "Setpoint (degC)": view_log._sp,
        "Predicted people": view_log.predicted_occupancy, "Actual people (sim)": view_log.actual_occupancy,
        "Indoor degC": view_log.indoor_temperature, "Est. kW": view_log.estimated_power_kw,
        "Reason": view_log.reason, "Source": view_log.prediction_source})
    st.write(f"**{len(disp):,}** of {len(log):,} decisions shown.")
    show_df(disp, height=380)
    if len(disp):
        opts = list(range(min(len(disp), 400)))
        pick = st.selectbox("Explain a decision in plain language", opts,
                            format_func=lambda i: f"{disp.iloc[i]['Time']} | {disp.iloc[i]['Floor']} | {disp.iloc[i]['Action']}")
        r = disp.iloc[pick]
        act_key = next(k for k, v in ACTION_LABEL.items() if v == r["Action"])
        st.success(f"**{r['Floor']} at {r['Time']} - {r['Action']}.** {ACTION_EXPLAIN[act_key]}\n\n*Controller's exact reason:* {r['Reason']}")
    st.download_button("Download shown decisions (CSV)", disp.to_csv(index=False).encode("utf-8"), "tower_brain_decisions.csv", "text/csv")
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("Action counts (whole week, all floors)")
        ac_counts = log.action.map(ACTION_LABEL).value_counts().rename_axis("Action").reset_index(name="Decisions")
        show_df(ac_counts)

# ============================ TAB 5: cancel-class impact ============================
with tab_cancel:
    if not cancels:
        st.info("No class is cancelled. Pick a class in the sidebar (**3 - Cancel a class**) and press **Cancel class**. "
                "This tab then compares Tower Brain with and without that cancellation.")
    else:
        st.markdown("#### Effect of the cancellation (Tower Brain, same scenario and seed)")
        d_tb = tb["k"]["total_kwh"] - tb0["k"]["total_kwh"]
        d_base = kb["total_kwh"] - kb0["total_kwh"]
        ch = changed_decisions(tb0["log"], log)
        aff = sorted(set(int(c[1][1:]) // 100 for c in []) | set(int(f[1:]) for f in ch.floor_id.unique()))
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Tower Brain week energy change", f"{d_tb:+.1f} kWh", help="cancelled run minus un-cancelled run")
        k2.metric("Baseline week energy change", f"{d_base:+.1f} kWh")
        k3.metric("Decisions that changed", f"{len(ch):,}")
        k4.metric("Floors affected", ", ".join(f"F{a}" for a in aff) if aff else "none")
        st.caption("A single cancelled class is a small part of a 5-day building total, so building-level changes are small. "
                   "The visible effect is on the affected floor's setpoints, AC on/off state and reasons.")
        cls_floor = sorted({int(tt_all[(tt_all.day == c[0]) & (tt_all.room_id == c[1]) & (tt_all.start_time_24hr == c[2])].floor_id.iloc[0][1:]) for c in cancels})
        fsel = st.selectbox("Floor to compare", cls_floor, format_func=lambda n: f"F{n}", key="cancel_floor")
        cday = DAYS.index(next(c[0] for c in cancels if int(tt_all[(tt_all.day == c[0]) & (tt_all.room_id == c[1])].floor_id.iloc[0][1:]) == fsel))
        h0t, h1t = with_ts(tb0["h"]), ht_ts
        cmp_kw = pd.DataFrame({"Tower Brain, class held": h0t[h0t.day_idx == cday].set_index("ts")[f"kw_f{fsel}"],
                               "Tower Brain, class cancelled": h1t[h1t.day_idx == cday].set_index("ts")[f"kw_f{fsel}"]})
        st.markdown(f"F{fsel} power on {DAYS[cday]} (kW)")
        show_line(cmp_kw, [COL_GREY, COL_TB], 260)
        pa = log[(log.floor_id == f"F{fsel}") & log.timestamp.str.startswith(DAYS[cday])].reset_index(drop=True)
        pa["ts"] = ANCHOR + pd.to_timedelta(np.arange(len(pa)) * STEP_MIN, unit="m")
        st.markdown(f"F{fsel} on {DAYS[cday]}: forecast vs actual people")
        show_line(pa.set_index("ts")[["predicted_occupancy", "actual_occupancy"]].rename(
            columns={"predicted_occupancy": "Predicted (controller's view)", "actual_occupancy": "Actual (simulated)"}), [COL_TB, "#38bdf8"], 230)
        if not informed:
            st.warning("The timetable was NOT notified: the model keeps forecasting the cancelled class (look for predicted people far above actual). "
                       "The model is ~95% driven by the timetable's next-hour schedule, so it cannot detect a cancellation before the class would have started.")
        if len(ch):
            chd = pd.DataFrame({"Time": ch.timestamp, "Floor": ch.floor_id,
                                "Action before": ch.action_before.map(ACTION_LABEL), "Action after": ch.action_after.map(ACTION_LABEL),
                                "Setpoint before": ch.new_setpoint_before.where(ch.ac_on_before == 1),
                                "Setpoint after": ch.new_setpoint_after.where(ch.ac_on_after == 1),
                                "Reason after": ch.reason_after})
            st.markdown("Decisions that changed")
            show_df(chd, height=300)
        else:
            st.info("This cancellation did not change any controller decision (the floor stays occupied by other classes or staff).")

# ============================ TAB 6: model, validation, assumptions ============================
with tab_model:
    st.markdown("#### Occupancy model (Random Forest)")
    acc = load_optional_json("person2/accuracy_report.json")
    if acc:
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("MAE (people)", f"{acc['MAE_people']:.2f}")
        a2.metric("RMSE (people)", f"{acc['RMSE_people']:.2f}")
        a3.metric("R squared", f"{acc['R_squared']:.3f}")
        a4.metric("Test rows", f"{acc['testing_rows']:,}")
        st.caption("Measured on the model's OWN synthetic test data (last 20% of 90 simulated days). It is not accuracy inside the building "
                   "simulator, and not real-world accuracy.")
    else:
        st.info("`accuracy_report.json` not found in the occupancy-model folder.")
    st.info("**What the model contributes (from the controller write-up):** its feature importance is "
            "about 95% the timetable's next-hour schedule, so it largely echoes the timetable. In the summer week the timetable-only "
            "fallback scored 7,604 kWh (-6.2%) with a 173.2 kW peak, slightly better than the model run (7,748 kWh, 180.1 kW). "
            "Use the policy comparison below to reproduce this for any current setting.")
    ic1, ic2 = st.columns(2)
    for col, (rel, cap_txt) in zip((ic1, ic2), (("person2/feature_importance.png", "Feature importance"),
                                                ("person2/actual_vs_predicted.png", "Actual vs predicted, floor F6"))):
        p = BASE_DIR / rel
        if p.exists():
            col.image(str(p), caption=cap_txt)
    fi = load_optional_csv("person2/feature_importance.csv")
    if fi is not None:
        st.caption(f"Top feature: **{fi.iloc[0]['feature']}** ({100 * fi.iloc[0]['importance']:.1f}% of importance) - the model largely echoes the timetable.")

    st.markdown("#### Validation against the team's saved results")
    ba = load_optional_csv("outputs/before_after.csv")
    can_validate = (ba is not None and offset == 0 and not cancels and seed == 1 and use_model)
    if ba is None:
        st.info("`outputs/before_after.csv` not found - nothing to validate against.")
    elif not can_validate:
        st.info("Validation applies to an unmodified preset (heat adjustment 0, no cancelled classes, seed 1, AI mode). "
                "Reset those controls to compare this live run with the team's saved `before_after.csv`.")
    else:
        rows = []
        for who, k_live in (("Fixed schedule (baseline)", kb), ("Tower Brain", tb["k"])):
            ref = ba[(ba.scenario == scen_name) & (ba.who == who)]
            if len(ref):
                r = ref.iloc[0]
                ok = abs(r.week_kwh - k_live["total_kwh"]) < 0.1 and abs(r.peak_kw - k_live["peak_kw_15min"]) < 0.1
                rows.append({"Policy": who, "Saved week kWh": r.week_kwh, "Live week kWh": round(k_live["total_kwh"], 1),
                             "Saved peak kW": r.peak_kw, "Live peak kW": round(k_live["peak_kw_15min"], 1), "Match": "YES" if ok else "NO"})
        if rows:
            show_df(pd.DataFrame(rows))
            (st.success if all(x["Match"] == "YES" for x in rows) else st.warning)(
                "Live results reproduce the saved controller outputs." if all(x["Match"] == "YES" for x in rows)
                else "Live results differ from the saved files (different scikit-learn version or modified model?).")

    st.markdown("#### Does the AI help? Compare policies for the current settings")
    st.caption("Runs two extra simulations (about 10 s) under exactly the current scenario, heat adjustment, cancellations and seed.")
    if st.button("Run policy comparison"):
        with st.spinner("Running timetable-only Tower Brain and the tuned timer..."):
            h_nt, k_nt = run_comparison_policy("timetable_only", tmin, tmax, doy, cancels, seed, tb["cap"])
            h_tt, k_tt = run_comparison_policy("tuned_timer", tmin, tmax, doy, cancels, seed, tb["cap"])
        pol = [("Fixed schedule (baseline)", kb), (("Tower Brain + occupancy model" if use_model else "Tower Brain (timetable-only, NO AI)"), tb["k"]),
               ("Tower Brain, timetable-only (NO AI)", k_nt), ("Tuned timer", k_tt)]
        pt = pd.DataFrame([{"Policy": n, "Week kWh": round(k["total_kwh"], 1), "Peak kW": round(k["peak_kw_15min"], 1),
                            "Comfort %": round(k["comfort_pct"], 2), "Occupied floor-intervals >26 degC": k["violations"],
                            "Saving vs baseline %": round(100 * (1 - k["total_kwh"] / kb["total_kwh"]), 1)} for n, k in pol])
        show_df(pt)
        st.caption("Honest reading: in the team's own tests the timetable-only variant matched or beat the model on energy, and the tuned timer "
                   "uses similar energy but with more comfort violations and no peak reduction. Check the table for the current settings.")

    st.markdown("#### Team's saved before/after table (seed 1, from `controller.py`)")
    if ba is not None:
        show_df(ba.assign(who=ba["who"].str.replace(" (Person 1)", "", regex=False)))
    pimg = OUT_DIR / "power_profile_tuesday.png"
    if pimg.exists():
        st.image(str(pimg), caption="Saved Tuesday power profile (summer, simulated)")

    st.markdown("#### Assumptions and limitations")
    st.markdown(
        "- **Simulated only.** Weather, attendance, building physics and the occupancy counter are synthetic; no real Liwa University data.\n"
        f"- **Comfort** = share of occupied person-time with floor temperature in {COMFORT_MIN:g}-{COMFORT_MAX:g} degC (the simulator's definition).\n"
        "- **Energy** = electrical cooling power (compressor + fans) summed over 15-minute steps. "
        "The controller's per-decision kW is only an estimate.\n"
        f"- **Cost and CO2** are estimates: tariff {tariff:.3f} AED/kWh and factor {co2f:.3f} kg/kWh are *your* sidebar assumptions, not project data.\n"
        "- **Peak cap** (70% of the baseline peak) is soft; the achieved peak can exceed it.\n"
        "- **Occupancy inputs** assume perfect people counters. A real deployment would need smart meters, occupancy sensing, indoor temperature "
        "sensors and building-management integration.\n"
        "- **Model vs simulator** were built separately (different attendance patterns); the controller applies a noise floor and a staff roster "
        "to use the model safely (see the controller write-up).\n"
        "- **One simulated week per scenario**; results will vary with seed and parameters.")
    with st.expander("Building simulation parameters (assumptions, from simulator.Params)"):
        pr = dataclasses.asdict(S.Params())
        show_df(pd.DataFrame({"Parameter": list(pr), "Value": [str(v) for v in pr.values()]}))
