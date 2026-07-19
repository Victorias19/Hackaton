"""
Full evaluation: held-out (cross-validated) performance for both models.
Writes evaluation.json with the honest, generalization metrics.

Run:  python evaluate_models.py
Needs: model_def.py, model_menses.pkl, model_ovulation.pkl,
       cycle_seq.csv, panel.csv, cycle_features.csv
"""

import json
import pickle
import datetime
import numpy as np
import pandas as pd
from model_def import CyclePredictor


cycle_seq = pd.read_csv("cycle_seq.csv")
panel = pd.read_csv("panel.csv")
cycle_features = pd.read_csv("cycle_features.csv")


def evaluate(pkl_path, event):
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)

    # rebuild internal training arrays (not pickled), then cross-validate
    model.fit(cycle_seq, panel, cycle_features)
    cv = model.cv()                                  # GroupKFold over women

    res = {
        "event": model.event,
        "n_features": len(model.features_),
        "n_trees": model.rsf.n_estimators,
        "train_c_index": round(float(model.train_c_), 4),
        "cv_c_index_folds": [round(float(s), 4) for s in cv],
        "cv_c_index_mean": round(float(cv.mean()), 4),
        "cv_c_index_std": round(float(cv.std()), 4),
        "overfit_gap": round(float(model.train_c_ - cv.mean()), 4),
    }

    # simple verdict
    m = res["cv_c_index_mean"]
    res["verdict"] = ("strong" if m >= 0.75 else
                      "useful" if m >= 0.65 else
                      "weak" if m >= 0.55 else
                      "near-random")

    print(f"\n{event.upper()}")
    print(f"  train C-index : {res['train_c_index']}")
    print(f"  CV   C-index  : {res['cv_c_index_mean']} ± {res['cv_c_index_std']}  ({res['verdict']})")
    print(f"  overfit gap   : {res['overfit_gap']}")
    return res


report = {
    "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "n_women": int(cycle_seq["id"].nunique()),
    "n_cycles": int(len(cycle_seq)),
    "metric": "Harrell's concordance index (C-index); 0.5=random, 1.0=perfect",
    "validation": "5-fold GroupKFold — whole women held out (tests generalization to new users)",
    "models": {
        "menses": evaluate("model_menses.pkl", "menses"),
        "ovulation": evaluate("model_ovulation.pkl", "ovulation"),
    },
}

with open("evaluation.json", "w") as f:
    json.dump(report, f, indent=2)

print("\nwrote evaluation.json")
