# CycleCast

> **One-line pitch:** A wearable-only forecaster that predicts a woman's next period and fertile window day by day, updating its prediction as each new day of sensor data arrives.

**🔗 Live demo:** [your-streamlit-url]
**🏷️ Track:** [fill in at kickoff]
**👤 Built by:** [your name] — solo

---

## The Problem

Cycle-tracking apps mostly predict by calendar averaging — they assume every cycle is the length of your last few. But real cycles shift with stress, sleep, and illness, and the fertile window moves with them. Wearables already capture the physiological signals (temperature, heart rate, HRV, sleep) that shift across a cycle, yet most tools don't use them to update a forecast in real time. CycleCast does.

## What It Does

You tell it when your period started; it forecasts, for every upcoming day, the probability that your next period begins and the probability you ovulate that day. As more days of wearable data come in, the forecast sharpens — so on day 8 you get a rough window, and by day 14 a tighter one. It shows the most-likely day, the likely window, and the full day-by-day probability curve for both events side by side.

## Approach

- **Method:** Longitudinal survival analysis with a `RandomSurvivalForest` (scikit-survival), trained on *landmark snapshots* — each snapshot is a cycle observed up to day *t*, labelled with the days remaining until the event.
- **Key idea:** Framing it as "given the cycle so far, how many days remain" (rather than "predict from day zero") turns every partial cycle into a training example and makes the model naturally sequential — it re-forecasts each day from the data seen so far.
- **Why this approach:** Survival models handle *censored* cycles (period or ovulation not yet observed) correctly instead of discarding them, and the landmark framing captures within-cycle physiological trends — which a static calendar model can't.
- **Bracelet-only by design:** features are strictly what a wearable can know at prediction time — daily signals plus prior cycle *lengths* derived from past period dates. Leakage-prone quantities (luteal/follicular length, ovulation-derived priors) are never used as inputs.

## Why You Can Trust It

Performance is measured by **grouped 5-fold cross-validation with whole women held out per fold** — so the score reflects prediction for *new users*, not memorised training subjects. Metric is Harrell's concordance index (C-index; 0.5 = random, 1.0 = perfect).

| Model | Train C-index | Cross-val C-index (held-out women) |
|---|---|---|
| **Next period** | 0.861 | **0.822** |
| **Fertile window** | 0.819 | **0.747** |

The small train-vs-CV gap (0.86 → 0.82 for menses) indicates the model generalises rather than overfits. Missing wearable days are handled explicitly — each signal carries `missing` and `coverage` flags — so gaps in real sensor data don't silently corrupt the forecast.

Permutation importance confirms the model uses genuine physiology, not just the calendar: after `day_now` (days elapsed), the top drivers are **resting-HR slope, sleep duration, and temperature slope** — the wearable trends that actually shift across a cycle.

## Demo Guide

1. Open the live demo link above.
2. Pick a woman from the dataset (or enter a cycle-start date manually).
3. Drag the **"today = cycle day"** slider and watch both forecasts update — most-likely day, window, and probability curves for *next period* (left) and *fertile window* (right).
4. In data mode, the period panel also shows the **actual** cycle length and the prediction error, so you can see accuracy directly.

## Tech Stack

- **Frontend/demo:** Streamlit
- **Core:** Python, scikit-survival (RandomSurvivalForest), scikit-learn, pandas, NumPy
- **Deploy:** Streamlit Community Cloud

## Limitations & What's Next

- **Ovulation labels are temperature-estimated, not lab-confirmed** (no progesterone/PdG confirmation in the data), so the fertility model predicts an algorithmic ovulation estimate rather than ground truth. Its lower CV (0.747) reflects that noisier target.
- **Small cohort** (35 women, 63 cycles) — the model is a strong proof of concept, but wider data would tighten the fertile-window prediction and support per-woman personalisation.
- **Next step:** add lab-confirmed ovulation labels and a per-woman random effect (frailty) so the model adapts to each user's baseline rather than the cohort average.

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repo Structure

```
app.py                # Streamlit app — two forecasts side by side
model_def.py          # CyclePredictor class (survival model + features + CV)
model_menses.pkl      # trained next-period model
model_ovulation.pkl   # trained fertile-window model
cycle_seq.csv         # cycle-level data (onsets, ovulation, priors)
panel.csv             # daily wearable signals
cycle_features.csv    # per-cycle history features
validate_models.py    # structural + score validation -> validation.json
requirements.txt      # dependencies
```