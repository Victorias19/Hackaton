"""
Hackathon demo skeleton.
Three tabs, all pre-wired so you only fill in the track-specific logic:
  1) Demo      -> input -> run_core_logic -> output   (swap this when track drops)
  2) Data      -> upload a CSV, preview + summary stats
  3) Visualize -> point charts at any column of the uploaded data

When the track is revealed, mostly you edit run_core_logic() and maybe wire
the uploaded DataFrame into it. Everything else already works.
"""

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

from rag import build_index, retrieve
import joblib

@st.cache_resource
def load_cycle_models():
    return joblib.load("model_menses.pkl"), joblib.load("model_ovulation.pkl")

@st.cache_data
def load_cycle_data():
    return (pd.read_csv("cycle_seq.csv"), pd.read_csv("panel.csv"), pd.read_csv("cycle_features.csv"))

from model_def import CyclePredictor
# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Demo", page_icon="⚡", layout="wide")

# ----------------------------------------------------------------------------
# PROVIDER SWITCH
# Currently set up for Groq via its OpenAI-compatible endpoint — free, no card.
# Key from console.groq.com (starts with gsk_).
# To switch to OpenAI later: set BASE_URL = None, change the model names below,
# and in rag.py flip EMBED_ENABLED back on.
# ----------------------------------------------------------------------------
BASE_URL = "https://api.groq.com/openai/v1"  # Groq
# BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"  # Gemini
# BASE_URL = None  # <- OpenAI


def get_client():
    key = st.secrets.get("OPENAI_API_KEY", None)
    if not key:
        key = st.session_state.get("api_key_input", None)
    if not key:
        return None
    if BASE_URL:
        return OpenAI(api_key=key, base_url=BASE_URL)
    return OpenAI(api_key=key)


# ----------------------------------------------------------------------------
# CORE LOGIC  ---  THIS IS THE ONLY PART YOU SWAP WHEN THE TRACK DROPS
# ----------------------------------------------------------------------------
def run_core_logic(user_input: str, client: OpenAI, context_docs: list[str]) -> str:
    """
    Right now: a simple RAG answer. Replace the body with whatever the challenge
    needs — a classifier, a hybrid ML+domain model, an agent loop, etc.
    You also have st.session_state['data'] (a DataFrame) available if a CSV was
    uploaded, so you can feed real data in here.
    """
    context = "\n\n".join(context_docs) if context_docs else "(no context)"
    system = (
        "You are a helpful assistant. Use the provided context when relevant. "
        "If the context does not contain the answer, say so plainly."
    )
    prompt = f"Context:\n{context}\n\nQuestion: {user_input}"
    resp = client.chat.completions.create(
        model=st.session_state.get("model", "llama-3.3-70b-versatile"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    st.session_state["model"] = st.selectbox(
        "Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        index=0,
    )
    st.text_input(
        "API key (fallback)",
        type="password",
        key="api_key_input",
        help="Only needed if not set in Streamlit secrets.",
    )
    st.caption("Set OPENAI_API_KEY in app secrets for the deployed version.")

    st.divider()
    st.subheader("Knowledge base (text)")
    uploaded_docs = st.file_uploader(
        "Upload text/markdown docs (optional)",
        type=["txt", "md"],
        accept_multiple_files=True,
    )

    st.divider()
    st.subheader("Data (CSV)")
    uploaded_csv = st.file_uploader("Upload a CSV", type=["csv"])

# ----------------------------------------------------------------------------
# Handle uploads
# ----------------------------------------------------------------------------
if "index" not in st.session_state:
    st.session_state["index"] = None
if "data" not in st.session_state:
    st.session_state["data"] = None

if uploaded_docs:
    docs = [f.read().decode("utf-8", errors="ignore") for f in uploaded_docs]
    client = get_client()
    if client:
        with st.spinner("Indexing documents..."):
            st.session_state["index"] = build_index(docs, client)
        st.sidebar.success(f"Indexed {len(docs)} document(s).")

if uploaded_csv is not None:
    try:
        st.session_state["data"] = pd.read_csv(uploaded_csv)
        st.sidebar.success(f"Loaded {st.session_state['data'].shape[0]} rows.")
    except Exception as e:
        st.sidebar.error(f"Couldn't read CSV: {e}")

# ----------------------------------------------------------------------------
# Main — three tabs
# ----------------------------------------------------------------------------
st.title("⚡ Demo")

tab_demo, tab_data, tab_viz, tab_cycle = st.tabs(["Demo", "Data", "Visualize", "Cycle forecast"])
# ---- TAB 1: DEMO -----------------------------------------------------------
with tab_demo:
    st.write("Swap the placeholder logic when the challenge is revealed.")
    user_input = st.text_area("Your input", height=120, placeholder="Ask something...")

    if st.button("Run", type="primary"):
        client = get_client()
        if not client:
            st.error("No API key. Add it in the sidebar or in app secrets.")
        elif not user_input.strip():
            st.warning("Enter some input first.")
        else:
            context_docs = []
            if st.session_state["index"] is not None:
                context_docs = retrieve(
                    user_input, st.session_state["index"], client, k=3
                )
            with st.spinner("Working..."):
                try:
                    output = run_core_logic(user_input, client, context_docs)
                    st.subheader("Output")
                    st.write(output)
                    if context_docs:
                        with st.expander("Retrieved context"):
                            for i, d in enumerate(context_docs, 1):
                                st.markdown(f"**Chunk {i}**")
                                st.caption(d[:500] + ("..." if len(d) > 500 else ""))
                except Exception as e:
                    st.error(f"Something broke: {e}")

# ---- TAB 2: DATA -----------------------------------------------------------
with tab_data:
    df = st.session_state["data"]
    if df is None:
        st.info("Upload a CSV in the sidebar to see it here.")
    else:
        st.subheader("Preview")
        st.dataframe(df.head(50), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing values", int(df.isna().sum().sum()))

        st.subheader("Summary statistics")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

        with st.expander("Column types & missing counts"):
            info = pd.DataFrame(
                {
                    "dtype": df.dtypes.astype(str),
                    "missing": df.isna().sum(),
                    "unique": df.nunique(),
                }
            )
            st.dataframe(info, use_container_width=True)

# ---- TAB 3: VISUALIZE ------------------------------------------------------
with tab_viz:
    df = st.session_state["data"]
    if df is None:
        st.info("Upload a CSV in the sidebar to plot it here.")
    else:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        all_cols = df.columns.tolist()

        chart_type = st.selectbox(
            "Chart type", ["Line", "Bar", "Scatter", "Histogram"]
        )

        try:
            if chart_type == "Line":
                cols = st.multiselect(
                    "Columns to plot (numeric)", numeric_cols,
                    default=numeric_cols[:1],
                )
                if cols:
                    st.line_chart(df[cols])

            elif chart_type == "Bar":
                cols = st.multiselect(
                    "Columns to plot (numeric)", numeric_cols,
                    default=numeric_cols[:1],
                )
                if cols:
                    st.bar_chart(df[cols])

            elif chart_type == "Scatter":
                cx = st.selectbox("X axis", numeric_cols, index=0)
                cy = st.selectbox(
                    "Y axis", numeric_cols,
                    index=min(1, len(numeric_cols) - 1),
                )
                color = st.selectbox("Color by (optional)", ["(none)"] + all_cols)
                kwargs = {"x": cx, "y": cy}
                if color != "(none)":
                    kwargs["color"] = color
                st.scatter_chart(df, **kwargs)

            elif chart_type == "Histogram":
                col = st.selectbox("Column", numeric_cols)
                bins = st.slider("Bins", 5, 60, 20)
                counts, edges = np.histogram(df[col].dropna(), bins=bins)
                hist_df = pd.DataFrame(
                    {"count": counts},
                    index=np.round((edges[:-1] + edges[1:]) / 2, 3),
                )
                st.bar_chart(hist_df)
        except Exception as e:
            st.error(f"Plot error: {e}")
with tab_cycle:
    import datetime
    st.header("Cycle & fertility forecast")
    st.caption("Enter when your period started; the model forecasts the next "
               "period and fertile window day by day from wearable signals + history.")

    model_men, model_ov = load_cycle_models()
    cycle_seq, panel, cycle_features = load_cycle_data()

    mode = st.radio("Mode", ["Use a woman from the data", "Enter manually"],
                    horizontal=True)

    # ---- gather this woman's cycle rows + history ----
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
    else:
        c1, c2 = st.columns(2)
        start_date = c1.date_input("My period started on",
                                   datetime.date.today() - datetime.timedelta(days=10))
        avg_len = c2.number_input("My usual cycle length (days)", 21, 40, 29)
        max_day = 40
        true_len = None
        # no wearable rows -> model falls back to cohort medians (fill_)
        days = pd.DataFrame({"day": []})
        hist = {"prior_cycle_length_mean": avg_len,
                "prior_cycle_length_sd": 2.0, "n_prior_cycles": 3}
        days_elapsed = (datetime.date.today() - start_date).days
        st.info(f"Today is day {days_elapsed} of your cycle.")

    # ---- which forecast ----
    event = st.radio("Forecast", ["Next period", "Fertile window"], horizontal=True)
    model = model_men if event == "Next period" else model_ov
    label = "next period" if event == "Next period" else "ovulation"

    # ---- 'today' = how many days into the cycle ----
    if mode == "Enter manually":
        today = st.slider("Today = cycle day", 3, max_day,
                          min(max(3, days_elapsed), max_day))
    else:
        today = st.slider("Today = cycle day", 3, max(4, max_day),
                          min(10, max_day))

    # ---- predict ----
    pred = model.predict_day(days, hist, today)
    pred = pred[pred["day"] >= today]

    if len(pred):
        peak = int(pred.loc[pred["p_event_day"].idxmax(), "day"])
        away = peak - today

        m1, m2, m3 = st.columns(3)
        m1.metric(f"Most likely {label}", f"day {peak}", f"~{away} days away")
        # 80% window
        s = pred.sort_values("day")
        cum = s["p_event_day"].cumsum() / s["p_event_day"].sum()
        win = s["day"][(cum >= 0.1) & (cum <= 0.9)]
        if len(win):
            m2.metric("Likely window", f"day {int(win.min())}–{int(win.max())}")
        if true_len is not None and event == "Next period":
            m3.metric("Actual", f"day {true_len}", f"error {peak-true_len:+d} d")

        st.subheader("Probability the event happens each day")
        st.bar_chart(pred.set_index("day")[["p_event_day"]]
                     .rename(columns={"p_event_day": "probability"}))

        st.subheader("Chance it hasn't happened yet")
        st.line_chart(pred.set_index("day")[["surv"]]
                      .rename(columns={"surv": "not-yet probability"}))

        with st.expander("forecast table"):
            st.dataframe(pred.reset_index(drop=True), use_container_width=True)
    else:
        st.warning("No forecast available for this day.")

    if event == "Fertile window":
        st.caption("Note: ovulation labels are temperature-estimated, not lab-confirmed.")
