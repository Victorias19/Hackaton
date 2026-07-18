"""
Ready-to-use validation helpers — your "why you can trust it" layer.

Most hackathon teams skip honest evaluation. You won't. When you build a model
in run_core_logic (or anywhere), call one of these to get a clean metrics dict
you can show in the demo and cite in the README.

Requires scikit-learn. If it's not installed:
    pip install scikit-learn
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# CLASSIFICATION
# ---------------------------------------------------------------------------
def evaluate_classifier(model, X, y, cv: int = 5) -> dict:
    """
    Cross-validated classification metrics.
    Returns accuracy / precision / recall / f1 (mean +/- std across folds).
    Usage:
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier()
        metrics = evaluate_classifier(m, X, y)
        st.json(metrics)
    """
    from sklearn.model_selection import cross_validate

    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    res = cross_validate(model, X, y, cv=cv, scoring=scoring)
    out = {}
    for s in scoring:
        arr = res[f"test_{s}"]
        out[s] = {"mean": round(float(arr.mean()), 4),
                  "std": round(float(arr.std()), 4)}
    out["cv_folds"] = cv
    out["n_samples"] = int(len(y))
    return out


def holdout_classifier(model, X, y, test_size: float = 0.2, seed: int = 42) -> dict:
    """Single train/test split with a full classification report."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return {
        "accuracy": round(float(accuracy_score(yte, pred)), 4),
        "f1_macro": round(float(f1_score(yte, pred, average="macro")), 4),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
    }


# ---------------------------------------------------------------------------
# REGRESSION
# ---------------------------------------------------------------------------
def evaluate_regressor(model, X, y, cv: int = 5) -> dict:
    """
    Cross-validated regression metrics: R2, MAE, RMSE (mean +/- std).
    Good for your scientific/QSAR-style tasks.
    """
    from sklearn.model_selection import cross_validate

    scoring = ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"]
    res = cross_validate(model, X, y, cv=cv, scoring=scoring)
    r2 = res["test_r2"]
    mae = -res["test_neg_mean_absolute_error"]
    rmse = -res["test_neg_root_mean_squared_error"]
    return {
        "r2": {"mean": round(float(r2.mean()), 4), "std": round(float(r2.std()), 4)},
        "mae": {"mean": round(float(mae.mean()), 4), "std": round(float(mae.std()), 4)},
        "rmse": {"mean": round(float(rmse.mean()), 4), "std": round(float(rmse.std()), 4)},
        "cv_folds": cv,
        "n_samples": int(len(y)),
    }


# ---------------------------------------------------------------------------
# BASELINE — always compare against a dumb model. Beating a baseline is the
# single most convincing "why trust it" line you can give a judge.
# ---------------------------------------------------------------------------
def baseline_comparison(X, y, task: str = "classification", cv: int = 5) -> dict:
    """
    Score a trivial baseline so you can say "we beat the naive baseline by X".
    task: "classification" (most-frequent) or "regression" (mean predictor).
    """
    from sklearn.model_selection import cross_val_score

    if task == "classification":
        from sklearn.dummy import DummyClassifier
        base = DummyClassifier(strategy="most_frequent")
        score = cross_val_score(base, X, y, cv=cv, scoring="accuracy")
        return {"baseline_accuracy": round(float(score.mean()), 4)}
    else:
        from sklearn.dummy import DummyRegressor
        base = DummyRegressor(strategy="mean")
        score = cross_val_score(base, X, y, cv=cv, scoring="r2")
        return {"baseline_r2": round(float(score.mean()), 4)}


# ---------------------------------------------------------------------------
# UNCERTAINTY — quick bootstrap CI for any point estimate (mean, metric, etc.)
# Your stats edge: show a confidence interval, not just a number.
# ---------------------------------------------------------------------------
def bootstrap_ci(values, n_boot: int = 2000, ci: float = 95, seed: int = 42) -> dict:
    """
    Bootstrap confidence interval for the mean of `values`.
    Returns mean and lower/upper bounds.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    boot_means = [rng.choice(values, size=len(values), replace=True).mean()
                  for _ in range(n_boot)]
    lo = np.percentile(boot_means, (100 - ci) / 2)
    hi = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return {
        "mean": round(float(values.mean()), 4),
        f"ci{int(ci)}_low": round(float(lo), 4),
        f"ci{int(ci)}_high": round(float(hi), 4),
    }
