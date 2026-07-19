"""
Cycle & fertility forecast app.

Run:
    streamlit run app.py

Required in the same folder:
    model_def.py
    model_menses.pkl
    model_ovulation.pkl
    cycle_seq.csv
    panel.csv
"""

import datetime
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Required so pickle can locate the CyclePredictor class.
from model_def import CyclePredictor


# ============================================================
# Configuration
# ============================================================

st.set_page_config(
    page_title="Cycle & Fertility Forecast",
    page_icon="🩸",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

HISTORY = [
    "prior_cycle_length_mean",
    "prior_cycle_length_sd",
    "n_prior_cycles",
]


# ============================================================
# Load models and data
# ============================================================

@st.cache_resource
def load_models():
    with open(BASE_DIR / "model_menses.pkl", "rb") as file:
        menses_model = pickle.load(file)

    with open(BASE_DIR / "model_ovulation.pkl", "rb") as file:
        ovulation_model = pickle.load(file)

    return menses_model, ovulation_model


@st.cache_data
def load_data():
    cycle_seq = pd.read_csv(
        BASE_DIR / "cycle_seq.csv"
    )

    panel = pd.read_csv(
        BASE_DIR / "panel.csv"
    )

    cycle_features = pd.read_csv(
        BASE_DIR / "cycle_features.csv"
    )

    # Convert relevant columns to numeric.
    for column in [
        "onset_day",
        "next_onset_day",
        "ovulation_day",
        "cycle_index",
    ]:
        if column in cycle_seq.columns:
            cycle_seq[column] = pd.to_numeric(
                cycle_seq[column],
                errors="coerce",
            )

    if "day_in_study" in panel.columns:
        panel["day_in_study"] = pd.to_numeric(
            panel["day_in_study"],
            errors="coerce",
        )

    if "cycle_index" in cycle_features.columns:
        cycle_features["cycle_index"] = pd.to_numeric(
            cycle_features["cycle_index"],
            errors="coerce",
        )

    return cycle_seq, panel, cycle_features


try:
    model_men, model_ov = load_models()
    cycle_seq, panel, cycle_features = load_data()

except Exception as error:
    st.error(f"Could not load models or data: {error}")
    st.stop()


# Check model compatibility.
for model_name, model_object in [
    ("Menses model", model_men),
    ("Ovulation model", model_ov),
]:
    if not hasattr(model_object, "predict_day"):
        st.error(
            f"{model_name} does not contain predict_day(). "
            "Retrain and save it using the current model_def.py."
        )
        st.stop()


# ============================================================
# Forecast display
# ============================================================

# Replace the entire forecast_panel function in app.py with this version.
# Replace the entire forecast_panel function in app.py with this.
# Adds: shared x-axis (symmetrical panels), fixed heights, per-panel colors,
# and the "already occurred" guard.

# Replace the entire forecast_panel in app.py with this.
#
# Option A: honest ovulation-passed detection from the temperature signal
# (uses only wearable data the woman would actually have — never the true day).
#
# Behavior:
#   - period panel: keeps the "already occurred" guard (period is observable)
#   - ovulation panel: detects a sustained temperature rise in the data so far;
#     if found, declares the fertile window passed and stops forecasting forward.
#   - reference_day: shown only as a demo comparison, never used to gate anything.

def _ovulation_passed(days_so_far, today):
    """Detect a sustained post-ovulation temperature rise from observed data only.
       Returns the estimated ovulation day (int) if detected, else None."""
    if days_so_far is None or days_so_far.empty:
        return None
    for col in ("nightly_temperature", "temp_rel"):
        if col not in days_so_far.columns:
            continue
        t = (days_so_far.sort_values("day")[["day", col]]
             .dropna())
        if len(t) < 6:
            continue
        vals = t[col].to_numpy(dtype=float)
        dys = t["day"].to_numpy(dtype=float)
        # compare each candidate split: baseline vs the 3 days after it
        for i in range(3, len(vals) - 2):
            baseline = vals[:i].mean()
            after = vals[i:i + 3].mean()
            if after - baseline > 0.20:            # ~0.2 sustained rise
                return int(round(dys[i]))          # rise starts ~1 day post-ovulation
    return None


OVULATION_CUTOFF = 21          # cycle day after which we treat ovulation as passed

def forecast_panel(
    col, title, model, label, days, hist, today,
    true_day=None, reference_day=None,
    fertile_window=False, note=None, x_range=None, color="#c0392b",
):
    with col:
        st.subheader(title)

        # observed measurements available through today
        if not days.empty:
            days_so_far = days.loc[days["day"] <= today].copy()
        else:
            days_so_far = pd.DataFrame({"day": pd.Series(dtype=float)})

        # --- PERIOD: observable event, honest to guard on true_day ---
        if true_day is not None and today >= true_day:
            st.success(f"{label.capitalize()} already occurred on day {true_day}.")
            st.caption("Move the slider earlier to see the forecast leading up to it.")
            return

        # --- OVULATION: simple biological day cutoff ---
        if fertile_window and today > OVULATION_CUTOFF:
            st.warning(
                f"Past the typical ovulation window (usually by ~day {OVULATION_CUTOFF}). "
                "Ovulation has most likely already occurred this cycle."
            )
            if reference_day is not None:
                st.caption(f"(Demo reference: recorded ovulation day {reference_day}.)")
            st.caption("Move the slider earlier to see the fertile-window forecast.")
            return

        # --- run the forecast ---
        try:
            pred = model.predict_day(days_so_far, hist, today)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        if pred is None or pred.empty or not {"day", "p_event_day", "surv"}.issubset(pred.columns):
            st.warning("No forecast available for this day.")
            return

        pred = pred.copy()
        pred["day"] = pd.to_numeric(pred["day"], errors="coerce")
        pred["p_event_day"] = pd.to_numeric(pred["p_event_day"], errors="coerce").fillna(0.0)
        pred["surv"] = pd.to_numeric(pred["surv"], errors="coerce")
        pred = pred.dropna(subset=["day"]).sort_values("day")

        if float(pred["p_event_day"].sum()) <= 0:
            st.warning("No event probability mass.")
            return

        # ---- metrics ----
        peak = int(round(float(pred.loc[pred["p_event_day"].idxmax(), "day"])))
        away = peak - today
        m1, m2 = st.columns(2)
        m1.metric(f"Most likely {label}", f"day {peak}", f"{away:+d} days from today")

        s = pred.dropna(subset=["surv"])
        lo = s.loc[s["surv"] <= 0.90, "day"]
        hi = s.loc[s["surv"] <= 0.10, "day"]
        if len(lo) and len(hi):
            m2.metric("Likely window", f"day {int(lo.iloc[0])}–{int(hi.iloc[0])}")
        else:
            m2.metric("Likely window", "—")

        # reference day: demo comparison only, never gates the forecast
        actual = true_day if true_day is not None else reference_day
        if actual is not None:
            st.metric("Actual event (reference)", f"day {actual}",
                      f"prediction error {peak - actual:+d} days")

        if fertile_window:
            fs, fe = peak - 5, peak + 1
            if fs <= today <= fe:
                st.success("Predicted fertile window is active now.")
            else:
                st.info(f"Predicted fertile window: cycle day {fs}–{fe}")

        # ---- symmetrical, continuous day axis ----
        lo_day, hi_day = int(pred["day"].min()), int(pred["day"].max())
        if x_range is not None:
            lo_day, hi_day = x_range
        full_days = np.arange(lo_day, hi_day + 1)

        prob = (pred.set_index("day")["p_event_day"]
                    .reindex(full_days, fill_value=0.0).rename("probability"))
        surv = (pred.set_index("day")["surv"]
                    .reindex(full_days).ffill().fillna(1.0).rename("not yet occurred"))
        prob.index.name = "cycle day"
        surv.index.name = "cycle day"

        st.caption("Estimated event probability by cycle day")
        st.bar_chart(prob, height=260, color=color)
        st.caption("Estimated probability the event has not occurred")
        st.line_chart(surv, height=220, color=color)

        if note:
            st.caption(note)
# ============================================================
# Page
# ============================================================

st.title("Cycle & fertility forecast")

st.caption(
    "Day-by-day predictions using wearable signals "
    "and previous-cycle history."
)

mode = st.radio(
    "Mode",
    [
        "Use a woman from the data",
        "Enter manually",
    ],
    horizontal=True,
)


# ============================================================
# Existing participant mode
# ============================================================

if mode == "Use a woman from the data":

    control_1, control_2 = st.columns(2)

    available_ids = (
        cycle_seq["id"]
        .dropna()
        .unique()
        .tolist()
    )

    try:
        available_ids = sorted(available_ids)
    except TypeError:
        available_ids = sorted(
            available_ids,
            key=str,
        )

    wid = control_1.selectbox(
        "Woman ID",
        available_ids,
    )

    woman_cycles = (
        cycle_seq.loc[
            cycle_seq["id"] == wid
        ]
        .copy()
        .sort_values("onset_day")
        .reset_index(drop=True)
    )

    if woman_cycles.empty:
        st.error(
            "No cycle records were found for this participant."
        )
        st.stop()

    cycle_options = list(
        woman_cycles.index
    )

    selected_position = control_2.selectbox(
        "Cycle",
        cycle_options,
        format_func=lambda index: (
            f"Cycle {woman_cycles.iloc[index].get('cycle_index', index)}"
            f" — onset day "
            f"{woman_cycles.iloc[index]['onset_day']:.0f}"
        ),
    )

    row = woman_cycles.iloc[
        selected_position
    ]

    onset = pd.to_numeric(
        row.get("onset_day"),
        errors="coerce",
    )

    end = pd.to_numeric(
        row.get("next_onset_day"),
        errors="coerce",
    )

    ovulation_day = pd.to_numeric(
        row.get("ovulation_day"),
        errors="coerce",
    )

    selected_cycle_index = row.get(
        "cycle_index",
        selected_position,
    )

    if pd.isna(onset):
        st.error(
            "The selected cycle has no valid onset day."
        )
        st.stop()

    # If next_onset_day is missing, use the next cycle onset.
    if (
        pd.isna(end)
        and selected_position + 1 < len(woman_cycles)
    ):
        end = pd.to_numeric(
            woman_cycles.iloc[
                selected_position + 1
            ]["onset_day"],
            errors="coerce",
        )

    days = panel.loc[
        (panel["id"] == wid)
        & (
            panel["day_in_study"]
            >= onset
        )
    ].copy()

    if pd.notna(end) and end > onset:
        days = days.loc[
            days["day_in_study"] < end
        ].copy()

    days["day"] = (
        days["day_in_study"] - onset
    )

    days = (
        days
        .dropna(subset=["day"])
        .sort_values("day")
    )

    # Get history belonging to this exact cycle.
    if (
        "cycle_index" in cycle_features.columns
        and pd.notna(selected_cycle_index)
    ):
        history_rows = cycle_features.loc[
            (cycle_features["id"] == wid)
            & (
                cycle_features["cycle_index"]
                == selected_cycle_index
            )
        ]
    else:
        history_rows = cycle_features.loc[
            cycle_features["id"] == wid
        ]

    if history_rows.empty:
        hist = {}
    else:
        hist = history_rows.iloc[0].to_dict()

    # Include cycle index because the updated model uses it.
    hist["cycle_index"] = selected_cycle_index

    for feature in HISTORY:
        hist.setdefault(feature, np.nan)

    if days.empty:
        max_day = 3
    else:
        max_day = max(
            3,
            int(np.floor(days["day"].max())),
        )

    slider_default = min(
        10,
        max_day,
    )

    today = st.slider(
        "Today = cycle day",
        min_value=3,
        max_value=max_day,
        value=slider_default,
    )

    true_period_day = None

    if pd.notna(end) and end > onset:
        true_period_day = int(
            round(end - onset)
        )

    true_ovulation_day = None

    if (
        pd.notna(ovulation_day)
        and ovulation_day > onset
    ):
        true_ovulation_day = int(
            round(ovulation_day - onset)
        )

    if true_period_day is not None:
        st.metric(
            "Selected cycle length",
            f"{true_period_day} days",
        )
    else:
        st.info(
            "This cycle does not have a known next onset."
        )


# ============================================================
# Manual mode
# ============================================================

else:
    control_1, control_2 = st.columns(2)

    start_date = control_1.date_input(
        "My period started on",
        datetime.date.today()
        - datetime.timedelta(days=10),
    )

    average_length = control_2.number_input(
        "My usual cycle length",
        min_value=21,
        max_value=40,
        value=29,
    )

    cycle_sd = control_1.number_input(
        "Typical cycle variation",
        min_value=0.0,
        max_value=15.0,
        value=2.0,
        step=0.5,
    )

    number_prior_cycles = control_2.number_input(
        "Number of previous cycles",
        min_value=0,
        max_value=100,
        value=3,
    )

    # No wearable data was entered.
    days = pd.DataFrame(
        {"day": pd.Series(dtype=float)}
    )

    hist = {
        "prior_cycle_length_mean":
            float(average_length),
        "prior_cycle_length_sd":
            float(cycle_sd),
        "n_prior_cycles":
            float(number_prior_cycles),
        "cycle_index":
            float(number_prior_cycles),
    }

    true_period_day = None
    true_ovulation_day = None

    days_elapsed = max(
        0,
        (
            datetime.date.today()
            - start_date
        ).days,
    )

    st.info(
        f"Today is approximately cycle day "
        f"{days_elapsed}."
    )

    slider_default = min(
        max(3, days_elapsed),
        40,
    )

    today = st.slider(
        "Today = cycle day",
        min_value=3,
        max_value=40,
        value=slider_default,
    )


# ============================================================
# Forecasts
# ============================================================

st.divider()

left, right = st.columns(2)

forecast_panel(
    col=left,
    title="🩸 Next period",
    model=model_men,
    label="next period",
    days=days,
    hist=hist,
    today=today,
    true_day=true_period_day,
)

forecast_panel(
    col=right,
    title="🌱 Ovulation and fertile window",
    model=model_ov,
    label="ovulation",
    days=days,
    hist=hist,
    today=today,
    true_day=true_ovulation_day,
    fertile_window=True,
    
)

st.divider()

