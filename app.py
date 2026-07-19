"""
Cycle & fertility forecast app.

Run:  streamlit run app.py
Needs in the same folder:
  model_def.py, model_menses.pkl, model_ovulation.pkl,
  cycle_seq.csv, panel.csv, cycle_features.csv
"""

import datetime
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from model_def import CyclePredictor          # needed to unpickle


st.set_page_config(page_title="Cycle & Fertility Forecast", page_icon="🩸", layout="wide")


@st.cache_resource
def load_models():
    return joblib.load("model_menses.pkl"), joblib.load("model_ovulation.pkl")

@st.cache_data
def load_data():
    return (pd.read_csv("cycle_seq.csv"),
            pd.read_csv("panel.csv"),
            pd.read_csv("cycle_features.csv"))

model_men, model_ov = load_models()
cycle_seq, panel, cycle_features = load_data()


# ---------------------------------------------------------------------------
# one forecast panel (used twice, side by side)
# ---------------------------------------------------------------------------
def forecast_panel(col, title, model, label, days, hist, today, true_len, note=None):
    with col:
        st.subheader(title)
        pred = model.predict_day(days, hist, today)
        pred = pred[pred["day"] >= today]

        if not len(pred):
            st.warning("No forecast available for this day.")
            return

        peak = int(pred.loc[pred["p_event_day"].idxmax(), "day"])
        away = peak - today

        m1, m2 = st.columns(2)
        m1.metric(f"Most likely {label}", f"day {peak}", f"~{away} days away")
        s = pred.sort_values("day")
        cum = s["p_event_day"].cumsum() / s["p_event_day"].sum()
        win = s["day"][(cum >= 0.1) & (cum <= 0.9)]
        if len(win):
            m2.metric("Likely window", f"day {int(win.min())}–{int(win.max())}")
        if true_len is not None and label == "next period":
            st.metric("Actual (this cycle)", f"day {true_len}",
                      f"error {peak - true_len:+d} days")

        st.caption("Probability the event happens each day")
        st.bar_chart(pred.set_index("day")[["p_event_day"]]
                     .rename(columns={"p_event_day": "probability"}))

        st.caption("Chance it hasn't happened yet")
        st.line_chart(pred.set_index("day")[["surv"]]
                      .rename(columns={"surv": "not-yet"}))

        if note:
            st.caption(note)


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------
st.title("Cycle & fertility forecast")
st.caption("Day-by-day prediction from wearable signals + cycle history. "
           "Move the slider to see how the forecast updates as the cycle progresses.")

mode = st.radio("Mode", ["Use a woman from the data", "Enter manually"],
                horizontal=True)

if mode == "Use a woman from the data":
    c1, c2 = st.columns(2)
    wid = c1.selectbox("Woman (id)", sorted(cycle_seq["id"].unique()))
    row = cycle_seq[cycle_seq.id == wid].iloc[0]
    onset, end = row["onset_day"], row["next_onset_day"]
    days = panel[(panel.id == wid) & (panel.day_in_study >= onset) &
                 (panel.day_in_study < end)].copy()
    days["day"] = days["day_in_study"] - onset
    hist = cycle_features[cycle_features.id == wid].iloc[0].to_dict()
    max_day = int(days["day"].max()) if len(days) else 20
    true_len = int(end - onset)
    c2.metric("This cycle's actual length", f"{true_len} days")
    today = st.slider("Today = cycle day", 3, max(4, max_day), min(10, max_day))
else:
    c1, c2 = st.columns(2)
    start_date = c1.date_input("My period started on",
                               datetime.date.today() - datetime.timedelta(days=10))
    avg_len = c2.number_input("My usual cycle length (days)", 21, 40, 29)
    days = pd.DataFrame({"day": []})
    hist = {"prior_cycle_length_mean": avg_len,
            "prior_cycle_length_sd": 2.0, "n_prior_cycles": 3}
    true_len = None
    days_elapsed = (datetime.date.today() - start_date).days
    st.info(f"Today is day {days_elapsed} of your cycle.")
    today = st.slider("Today = cycle day", 3, 40, min(max(3, days_elapsed), 40))

st.divider()

# ---------------------------------------------------------------------------
# two models side by side
# ---------------------------------------------------------------------------
left, right = st.columns(2)
forecast_panel(left,  "🩸 Next period", model_men, "next period",
               days, hist, today, true_len)
forecast_panel(right, "🌱 Fertile window", model_ov, "ovulation",
               days, hist, today, true_len,
               note="Ovulation labels are temperature-estimated, not lab-confirmed.")