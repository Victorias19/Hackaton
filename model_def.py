import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv


DAILY = [
    "temp_rel",
    "nightly_temperature",
    "wrist_temp_diff",
    "resting_hr",
    "hrv_rmssd",
    "resp_rate",
    "minutesasleep",
    "efficiency",
    "steps_total",
    "active_minutes",
    "calories_total",
]

HISTORY = [
    "prior_cycle_length_mean",
    "prior_cycle_length_sd",
    "n_prior_cycles",
]


class CyclePredictor:
    def __init__(self, event="menses"):
        self.event = event

        self.end = (
            "next_onset_day"
            if event == "menses"
            else "ovulation_day"
        )

        self.rsf = RandomSurvivalForest(
            n_estimators=300,
            min_samples_leaf=8,
            max_depth=5,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )

    # ---------------------------------------------------------
    # Features
    # ---------------------------------------------------------

    def _feat(self, cd, hist, t):
        sofar = (
            cd[cd["day"] <= t]
            .copy()
            .sort_values("day")
        )

        row = {
            "day_now": float(t),

            # Safe cycle-level information known at cycle start
            "cycle_index": pd.to_numeric(
                hist.get("cycle_index", np.nan),
                errors="coerce",
            ),
        }

        for c in DAILY:
            if c in sofar.columns:
                observed = sofar[
                    ["day", c]
                ].copy()

                observed["day"] = pd.to_numeric(
                    observed["day"],
                    errors="coerce",
                )

                observed[c] = pd.to_numeric(
                    observed[c],
                    errors="coerce",
                )

                observed = (
                    observed.dropna()
                    .sort_values("day")
                )
            else:
                observed = pd.DataFrame(
                    columns=["day", c]
                )

            n = len(observed)

            # Missing-data information
            row[c + "_missing"] = float(n == 0)

            # Percentage of possible days with measurements
            row[c + "_coverage"] = float(
                n / max(t + 1, 1)
            )

            if n == 0:
                row[c + "_mean"] = np.nan
                row[c + "_last"] = np.nan
                row[c + "_slope"] = np.nan
                continue

            measurement_days = observed[
                "day"
            ].to_numpy(dtype=float)

            values = observed[
                c
            ].to_numpy(dtype=float)

            row[c + "_mean"] = float(
                np.mean(values)
            )

            row[c + "_last"] = float(
                values[-1]
            )

            # Use actual observation days for the slope
            if (
                n >= 2
                and np.unique(measurement_days).size >= 2
            ):
                row[c + "_slope"] = float(
                    np.polyfit(
                        measurement_days,
                        values,
                        1,
                    )[0]
                )
            else:
                row[c + "_slope"] = np.nan

        for h in HISTORY:
            value = pd.to_numeric(
                hist.get(h, np.nan),
                errors="coerce",
            )

            row[h] = value
            row[h + "_missing"] = float(
                pd.isna(value)
            )

        return row

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    def fit(
        self,
        cs,
        pn,
        cf,
        censor_day_col=None,
    ):
        seq = cs.copy()
        panel = pn.copy()

        seq["onset_day"] = pd.to_numeric(
            seq["onset_day"],
            errors="coerce",
        )

        seq[self.end] = pd.to_numeric(
            seq[self.end],
            errors="coerce",
        )

        panel["day_in_study"] = pd.to_numeric(
            panel["day_in_study"],
            errors="coerce",
        )

        # Find the following menstrual onset
        seq = (
            seq.sort_values(
                ["id", "onset_day"]
            )
            .copy()
        )

        seq["_following_onset"] = (
            seq.groupby("id")["onset_day"]
            .shift(-1)
        )

        if (
            censor_day_col is not None
            and censor_day_col in seq.columns
        ):
            seq[censor_day_col] = pd.to_numeric(
                seq[censor_day_col],
                errors="coerce",
            )

        H = (
            cf.set_index(
                ["id", "cycle_index"]
            )
            .to_dict("index")
        )

        rows = []
        times = []
        events = []
        groups = []

        for _, cycle in seq.iterrows():
            wid = cycle["id"]

            onset = cycle["onset_day"]
            event_day = cycle[self.end]
            following_onset = cycle[
                "_following_onset"
            ]

            if pd.isna(onset):
                continue

            event_observed = False
            stop_day = np.nan

            # Known event
            if (
                pd.notna(event_day)
                and event_day > onset
            ):
                stop_day = event_day
                event_observed = True

            # Missing next_onset_day, but next cycle onset exists
            elif (
                self.event == "menses"
                and pd.notna(following_onset)
                and following_onset > onset
            ):
                stop_day = following_onset
                event_observed = True

            # Missing event: retain as censored
            else:
                censor_candidates = []

                if (
                    censor_day_col is not None
                    and censor_day_col in cycle.index
                ):
                    explicit_censor = cycle[
                        censor_day_col
                    ]

                    if (
                        pd.notna(explicit_censor)
                        and explicit_censor > onset
                    ):
                        censor_candidates.append(
                            float(explicit_censor)
                        )

                available_days = panel.loc[
                    (panel["id"] == wid)
                    & (
                        panel["day_in_study"]
                        >= onset
                    ),
                    "day_in_study",
                ].dropna()

                # Prevent measurements from the next cycle
                if pd.notna(following_onset):
                    available_days = available_days[
                        available_days
                        < following_onset
                    ]

                    if following_onset > onset:
                        censor_candidates.append(
                            float(following_onset)
                        )

                if not available_days.empty:
                    censor_candidates.append(
                        float(available_days.max())
                    )

                if censor_candidates:
                    stop_day = min(
                        censor_candidates
                    )

            # Cannot use a cycle without event/censoring time
            if (
                pd.isna(stop_day)
                or stop_day <= onset
            ):
                continue

            total = int(
                np.floor(stop_day - onset)
            )

            if total <= 2:
                continue

            if event_observed:
                days = panel[
                    (panel["id"] == wid)
                    & (
                        panel["day_in_study"]
                        >= onset
                    )
                    & (
                        panel["day_in_study"]
                        < stop_day
                    )
                ].copy()
            else:
                days = panel[
                    (panel["id"] == wid)
                    & (
                        panel["day_in_study"]
                        >= onset
                    )
                    & (
                        panel["day_in_study"]
                        <= stop_day
                    )
                ].copy()

            # Keep the cycle even when all sensor data is missing
            if days.empty:
                days = pd.DataFrame(
                    columns=list(panel.columns)
                    + ["day"]
                )
            else:
                days["day"] = (
                    days["day_in_study"]
                    - onset
                )

            cycle_index = cycle.get(
                "cycle_index",
                np.nan,
            )

            history = dict(
                H.get(
                    (wid, cycle_index),
                    {},
                )
            )

            # Add safe cycle-level feature
            history["cycle_index"] = cycle_index

            for t in range(2, total, 2):
                remaining = float(
                    stop_day - (onset + t)
                )

                if remaining <= 0:
                    continue

                rows.append(
                    self._feat(
                        days,
                        history,
                        t,
                    )
                )

                times.append(remaining)
                events.append(event_observed)
                groups.append(wid)

        if not rows:
            raise ValueError(
                "No training observations were created."
            )

        X_raw = pd.DataFrame(rows).replace(
            [np.inf, -np.inf],
            np.nan,
        )

        self.features_ = X_raw.columns.tolist()

        self.fill_ = (
            X_raw.median()
            .reindex(self.features_)
            .fillna(0.0)
        )

        X = X_raw.fillna(self.fill_)

        y = Surv.from_arrays(
            event=np.asarray(
                events,
                dtype=bool,
            ),
            time=np.asarray(
                times,
                dtype=float,
            ),
        )

        self._X_raw = X_raw
        self._X = X
        self._y = y
        self._g = np.asarray(groups)

        self.rsf.fit(X, y)

        self.train_c_ = self.rsf.score(
            X,
            y,
        )

        return self

    # ---------------------------------------------------------
    # Cross-validation
    # ---------------------------------------------------------

    def cv(self, n=5):
        unique_people = pd.Series(
            self._g
        ).nunique()

        n = min(n, unique_people)

        splitter = GroupKFold(
            n_splits=n
        )

        scores = []

        for train, test in splitter.split(
            self._X_raw,
            self._y,
            self._g,
        ):
            X_train_raw = self._X_raw.iloc[
                train
            ]

            X_test_raw = self._X_raw.iloc[
                test
            ]

            # Calculate medians only from training fold
            fold_fill = (
                X_train_raw.median()
                .reindex(self.features_)
                .fillna(0.0)
            )

            X_train = X_train_raw.fillna(
                fold_fill
            )

            X_test = X_test_raw.fillna(
                fold_fill
            )

            model = RandomSurvivalForest(
                n_estimators=300,
                min_samples_leaf=8,
                max_depth=5,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            )

            model.fit(
                X_train,
                self._y[train],
            )

            try:
                score = model.score(
                    X_test,
                    self._y[test],
                )
            except ValueError:
                score = np.nan

            scores.append(score)

        return np.asarray(scores)

    # ---------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------

    def importances(
        self,
        top=12,
        n_repeats=10,
    ):
        result = permutation_importance(
            self.rsf,
            self._X,
            self._y,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1,
        )

        output = pd.DataFrame(
            {
                "importance_mean":
                    result.importances_mean,
                "importance_sd":
                    result.importances_std,
            },
            index=self.features_,
        )

        output.index.name = "feature"

        return (
            output
            .sort_values(
                "importance_mean",
                ascending=False,
            )
            .head(top)
        )

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    def predict_day(
        self,
        cd,
        hist,
        t,
    ):
        features = self._feat(
            cd,
            hist,
            t,
        )

        X = (
            pd.DataFrame([features])
            .reindex(columns=self.features_)
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(self.fill_)
        )

        fn = self.rsf.predict_survival_function(
            X
        )[0]

        remaining = np.asarray(
            fn.x,
            dtype=float,
        )

        survival = np.asarray(
            fn.y,
            dtype=float,
        )

        probability = -np.diff(
            np.r_[1.0, survival]
        )

        return pd.DataFrame(
            {
                "day": t + remaining,
                "p_event_day": np.round(
                    probability,
                    4,
                ),
                "surv": np.round(
                    survival,
                    4,
                ),
            }
        )
