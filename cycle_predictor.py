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
    def __init__(
        self,
        event="menses",
        n_estimators=300,
        min_samples_leaf=8,
        max_depth=5,
        random_state=42,
    ):
        if event not in {"menses", "ovulation"}:
            raise ValueError("event must be 'menses' or 'ovulation'.")

        self.event = event
        self.end = (
            "next_onset_day"
            if event == "menses"
            else "ovulation_day"
        )

        self.random_state = random_state

        self.model_params = {
            "n_estimators": n_estimators,
            "min_samples_leaf": min_samples_leaf,
            "max_depth": max_depth,
            "max_features": "sqrt",
            "random_state": random_state,
            "n_jobs": -1,
        }

        self.rsf = RandomSurvivalForest(
            **self.model_params
        )

    # ---------------------------------------------------------
    # Utility functions
    # ---------------------------------------------------------

    @staticmethod
    def _number(value):
        """
        Convert a value to a finite float.

        Invalid or missing values become NaN.
        """
        value = pd.to_numeric(
            pd.Series([value]),
            errors="coerce",
        ).iloc[0]

        if pd.isna(value) or not np.isfinite(value):
            return np.nan

        return float(value)

    @staticmethod
    def _fill_values(X):
        """
        Calculate median imputation values.

        Completely missing columns receive 0. Missing-indicator
        features tell the model whether the value was originally absent.
        """
        return (
            X.median(numeric_only=True)
            .reindex(X.columns)
            .fillna(0.0)
        )

    def _new_model(self, seed=None):
        """Create a fresh RSF with the original parameters."""

        params = self.model_params.copy()

        if seed is not None:
            params["random_state"] = seed

        return RandomSurvivalForest(**params)

    # ---------------------------------------------------------
    # Landmark features
    # ---------------------------------------------------------

    def _landmark_features(
        self,
        cycle_days,
        history,
        t,
    ):
        """
        Create features from data observed through cycle day t.

        Missing measurements are not dropped. They are represented
        through NaN values plus explicit missingness indicators.
        """

        if "day" not in cycle_days.columns:
            raise ValueError(
                "cycle_days must contain a 'day' column."
            )

        sofar = (
            cycle_days.loc[
                cycle_days["day"] <= t
            ]
            .copy()
            .sort_values("day")
        )

        row = {
            "day_now": float(t),
        }

        for column in DAILY:
            if column in sofar.columns:
                observed = sofar[
                    ["day", column]
                ].copy()

                observed["day"] = pd.to_numeric(
                    observed["day"],
                    errors="coerce",
                )

                observed[column] = pd.to_numeric(
                    observed[column],
                    errors="coerce",
                )

                observed = (
                    observed.dropna()
                    .sort_values("day")
                )
            else:
                observed = pd.DataFrame(
                    columns=["day", column]
                )

            n_observations = len(observed)

            # Explicit information about missingness.
            row[column + "_missing"] = float(
                n_observations == 0
            )

            row[column + "_n_obs"] = float(
                n_observations
            )

            if n_observations == 0:
                row[column + "_mean"] = np.nan
                row[column + "_last"] = np.nan
                row[column + "_slope"] = np.nan
                continue

            days = observed["day"].to_numpy(
                dtype=float
            )

            values = observed[column].to_numpy(
                dtype=float
            )

            row[column + "_mean"] = float(
                np.mean(values)
            )

            row[column + "_last"] = float(
                values[-1]
            )

            # Calculate the slope using actual measurement days.
            if (
                n_observations >= 2
                and np.unique(days).size >= 2
            ):
                row[column + "_slope"] = float(
                    np.polyfit(
                        days,
                        values,
                        deg=1,
                    )[0]
                )
            else:
                row[column + "_slope"] = np.nan

        # Add historical cycle features.
        for column in HISTORY:
            value = self._number(
                history.get(column, np.nan)
            )

            row[column] = value
            row[column + "_missing"] = float(
                pd.isna(value)
            )

        return row

    # ---------------------------------------------------------
    # Fit
    # ---------------------------------------------------------

    def fit(
        self,
        cycle_seq,
        panel,
        cycle_features,
        censor_day_col=None,
        infer_censor_from_panel=True,
    ):
        """
        Fit the landmark survival model.

        Parameters
        ----------
        cycle_seq:
            Cycle-level data.

        panel:
            Daily/nightly observations.

        cycle_features:
            Historical features for each cycle.

        censor_day_col:
            Optional column in cycle_seq containing the final
            observed study day for incomplete cycles.

        infer_censor_from_panel:
            When True, use the final available panel day as the
            censoring day when the event date is missing.
        """

        required_cycle_columns = {
            "id",
            "cycle_index",
            "onset_day",
            self.end,
        }

        required_panel_columns = {
            "id",
            "day_in_study",
        }

        required_history_columns = {
            "id",
            "cycle_index",
        }

        missing = (
            required_cycle_columns
            - set(cycle_seq.columns)
        )

        if missing:
            raise ValueError(
                "cycle_seq is missing columns: "
                f"{sorted(missing)}"
            )

        missing = (
            required_panel_columns
            - set(panel.columns)
        )

        if missing:
            raise ValueError(
                "panel is missing columns: "
                f"{sorted(missing)}"
            )

        missing = (
            required_history_columns
            - set(cycle_features.columns)
        )

        if missing:
            raise ValueError(
                "cycle_features is missing columns: "
                f"{sorted(missing)}"
            )

        seq = cycle_seq.copy()
        pnl = panel.copy()

        seq["onset_day"] = pd.to_numeric(
            seq["onset_day"],
            errors="coerce",
        )

        seq[self.end] = pd.to_numeric(
            seq[self.end],
            errors="coerce",
        )

        pnl["day_in_study"] = pd.to_numeric(
            pnl["day_in_study"],
            errors="coerce",
        )

        # Sort cycles so the next recorded onset can be derived.
        seq = (
            seq.sort_values(
                [
                    "id",
                    "onset_day",
                    "cycle_index",
                ]
            )
            .copy()
        )

        seq["_following_onset"] = (
            seq.groupby("id")["onset_day"]
            .shift(-1)
        )

        if censor_day_col is not None:
            if censor_day_col not in seq.columns:
                raise ValueError(
                    f"{censor_day_col!r} is not "
                    "a column in cycle_seq."
                )

            seq[censor_day_col] = pd.to_numeric(
                seq[censor_day_col],
                errors="coerce",
            )

        if cycle_features.duplicated(
            ["id", "cycle_index"]
        ).any():
            raise ValueError(
                "cycle_features contains duplicate "
                "(id, cycle_index) rows."
            )

        history_lookup = (
            cycle_features
            .set_index(["id", "cycle_index"])
            .to_dict("index")
        )

        rows = []
        times = []
        events = []
        groups = []
        cycle_keys = []

        report = {
            "cycles_total": int(len(seq)),
            "cycles_used": 0,
            "cycles_with_event": 0,
            "cycles_censored": 0,
            "cycles_missing_onset": 0,
            "cycles_without_followup_end": 0,
            "cycles_too_short": 0,
            "landmarks_created": 0,
        }

        for _, cycle in seq.iterrows():
            woman_id = cycle["id"]
            cycle_index = cycle["cycle_index"]

            onset = self._number(
                cycle["onset_day"]
            )

            event_day = self._number(
                cycle[self.end]
            )

            following_onset = self._number(
                cycle["_following_onset"]
            )

            if pd.isna(onset):
                report["cycles_missing_onset"] += 1
                continue

            stop_day = np.nan
            event_observed = False

            # ---------------------------------------------
            # Case 1: explicit event date is known
            # ---------------------------------------------

            if (
                pd.notna(event_day)
                and event_day > onset
            ):
                stop_day = event_day
                event_observed = True

            # ---------------------------------------------
            # Case 2: derive missing menstruation event
            # from the following recorded onset
            # ---------------------------------------------

            elif (
                self.event == "menses"
                and pd.notna(following_onset)
                and following_onset > onset
            ):
                stop_day = following_onset
                event_observed = True

            # ---------------------------------------------
            # Case 3: event is missing, retain as censored
            # ---------------------------------------------

            else:
                censor_candidates = []

                # Explicit censoring date.
                if censor_day_col is not None:
                    explicit_censor = self._number(
                        cycle[censor_day_col]
                    )

                    if (
                        pd.notna(explicit_censor)
                        and explicit_censor > onset
                    ):
                        censor_candidates.append(
                            explicit_censor
                        )

                participant_days = pnl.loc[
                    (pnl["id"] == woman_id)
                    & (
                        pnl["day_in_study"]
                        >= onset
                    ),
                    "day_in_study",
                ].dropna()

                # Prevent daily data from spilling into
                # the following cycle.
                if pd.notna(following_onset):
                    participant_days = (
                        participant_days.loc[
                            participant_days
                            < following_onset
                        ]
                    )

                    if following_onset > onset:
                        censor_candidates.append(
                            following_onset
                        )

                if (
                    infer_censor_from_panel
                    and not participant_days.empty
                ):
                    censor_candidates.append(
                        float(
                            participant_days.max()
                        )
                    )

                if censor_candidates:
                    # Use the earliest valid follow-up boundary.
                    stop_day = min(censor_candidates)
                    event_observed = False

            # A survival target cannot be formed without
            # either an event or censoring time.
            if (
                pd.isna(stop_day)
                or stop_day <= onset
            ):
                report[
                    "cycles_without_followup_end"
                ] += 1
                continue

            total_days = int(
                np.floor(stop_day - onset)
            )

            if total_days <= 2:
                report["cycles_too_short"] += 1
                continue

            # For an observed event, do not include the event day.
            if event_observed:
                cycle_days = pnl.loc[
                    (pnl["id"] == woman_id)
                    & (
                        pnl["day_in_study"]
                        >= onset
                    )
                    & (
                        pnl["day_in_study"]
                        < stop_day
                    )
                ].copy()

            # For a censored cycle, the censor day can be included.
            else:
                cycle_days = pnl.loc[
                    (pnl["id"] == woman_id)
                    & (
                        pnl["day_in_study"]
                        >= onset
                    )
                    & (
                        pnl["day_in_study"]
                        <= stop_day
                    )
                ].copy()

            # Do not discard the cycle when all daily data is missing.
            if cycle_days.empty:
                empty_columns = list(
                    dict.fromkeys(
                        list(pnl.columns) + ["day"]
                    )
                )

                cycle_days = pd.DataFrame(
                    columns=empty_columns
                )
            else:
                cycle_days["day"] = (
                    cycle_days["day_in_study"]
                    - onset
                )

            history = history_lookup.get(
                (woman_id, cycle_index),
                {},
            )

            rows_before_cycle = len(rows)

            for t in range(
                2,
                total_days,
                2,
            ):
                remaining_time = float(
                    stop_day - (onset + t)
                )

                if remaining_time <= 0:
                    continue

                rows.append(
                    self._landmark_features(
                        cycle_days,
                        history,
                        t,
                    )
                )

                times.append(remaining_time)
                events.append(event_observed)
                groups.append(woman_id)

                cycle_keys.append(
                    f"{woman_id!r}::{cycle_index!r}"
                )

            if len(rows) > rows_before_cycle:
                report["cycles_used"] += 1

                if event_observed:
                    report["cycles_with_event"] += 1
                else:
                    report["cycles_censored"] += 1

        if not rows:
            raise ValueError(
                "No training landmarks were created. "
                "Each usable cycle needs an onset day and "
                "either an event day or a censoring endpoint."
            )

        X_raw = (
            pd.DataFrame(rows)
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

        self.features_ = X_raw.columns.tolist()

        # Final-model imputation values.
        self.fill_ = self._fill_values(
            X_raw
        )

        X = X_raw.fillna(
            self.fill_
        )

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

        if np.sum(y["event"]) == 0:
            raise ValueError(
                "There are no observed events. "
                "A model cannot learn event timing from "
                "censored cycles alone."
            )

        self._X_raw = X_raw
        self._X = X
        self._y = y
        self._groups = np.asarray(groups)
        self._cycle_keys = np.asarray(cycle_keys)

        self.rsf = self._new_model()

        self.rsf.fit(
            X,
            y,
        )

        self.train_c_ = float(
            self.rsf.score(X, y)
        )

        report["landmarks_created"] = int(
            len(X)
        )

        report["event_landmarks"] = int(
            np.sum(y["event"])
        )

        report["censored_landmarks"] = int(
            np.sum(~y["event"])
        )

        self.fit_report_ = report

        return self

    # ---------------------------------------------------------
    # Cross-validation
    # ---------------------------------------------------------

    def cross_val_cindex(
        self,
        n_splits=5,
    ):
        """
        Grouped cross-validation by participant.

        Imputation is recalculated inside each training fold,
        avoiding preprocessing leakage.
        """

        number_of_groups = (
            pd.Series(self._groups).nunique()
        )

        if number_of_groups < 2:
            raise ValueError(
                "At least two participants are required "
                "for grouped cross-validation."
            )

        n_splits = min(
            int(n_splits),
            int(number_of_groups),
        )

        if n_splits < 2:
            raise ValueError(
                "n_splits must be at least 2."
            )

        splitter = GroupKFold(
            n_splits=n_splits
        )

        scores = []

        split_iterator = splitter.split(
            self._X_raw,
            self._y,
            self._groups,
        )

        for fold, (train_index, test_index) in enumerate(
            split_iterator,
            start=1,
        ):
            # A training fold needs at least one event.
            if (
                np.sum(
                    self._y[train_index]["event"]
                )
                == 0
            ):
                scores.append(np.nan)
                continue

            X_train_raw = self._X_raw.iloc[
                train_index
            ]

            X_test_raw = self._X_raw.iloc[
                test_index
            ]

            # Learn medians only from the training fold.
            fold_fill = self._fill_values(
                X_train_raw
            )

            X_train = X_train_raw.fillna(
                fold_fill
            )

            X_test = X_test_raw.fillna(
                fold_fill
            )

            model = self._new_model(
                seed=self.random_state + fold
            )

            model.fit(
                X_train,
                self._y[train_index],
            )

            try:
                score = float(
                    model.score(
                        X_test,
                        self._y[test_index],
                    )
                )
            except ValueError:
                # A test fold can have no comparable pairs.
                score = np.nan

            scores.append(score)

        return np.asarray(
            scores,
            dtype=float,
        )

    # ---------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------

    def importances(
        self,
        top=12,
        n_repeats=10,
    ):
        """
        Permutation importance using the fitted data.

        Positive importance means performance decreased when
        the feature was shuffled.
        """

        result = permutation_importance(
            estimator=self.rsf,
            X=self._X,
            y=self._y,
            n_repeats=n_repeats,
            random_state=self.random_state,
            n_jobs=-1,
        )

        importance = pd.DataFrame(
            {
                "importance_mean": (
                    result.importances_mean
                ),
                "importance_sd": (
                    result.importances_std
                ),
            },
            index=self.features_,
        )

        importance.index.name = "feature"

        return (
            importance
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
        cycle_days,
        history,
        t,
    ):
        if "day" not in cycle_days.columns:
            raise ValueError(
                "cycle_days must contain 'day'. "
                "Create it as day_in_study - onset_day, "
                "or use woman_cycle()."
            )

        features = self._landmark_features(
            cycle_days,
            history,
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

        survival_function = (
            self.rsf
            .predict_survival_function(X)[0]
        )

        remaining = np.asarray(
            survival_function.x,
            dtype=float,
        )

        survival = np.asarray(
            survival_function.y,
            dtype=float,
        )

        previous_survival = np.r_[
            1.0,
            survival[:-1],
        ]

        event_probability = np.clip(
            previous_survival - survival,
            0.0,
            1.0,
        )

        return pd.DataFrame(
            {
                "day": t + remaining,
                "remaining_days": remaining,
                "p_event_day": np.round(
                    event_probability,
                    4,
                ),
                "surv": np.round(
                    survival,
                    4,
                ),
            }
        )


def woman_cycle(
    cycle_seq,
    panel,
    cycle_features,
    wid,
    cycle_index=None,
):
    """
    Prepare one woman's cycle for prediction.

    When cycle_index is omitted, the latest cycle is selected.
    """

    person_cycles = cycle_seq.loc[
        cycle_seq["id"] == wid
    ].copy()

    if person_cycles.empty:
        raise ValueError(
            f"No cycles found for id={wid!r}."
        )

    person_cycles["onset_day"] = pd.to_numeric(
        person_cycles["onset_day"],
        errors="coerce",
    )

    person_cycles = (
        person_cycles
        .dropna(subset=["onset_day"])
        .sort_values("onset_day")
    )

    if person_cycles.empty:
        raise ValueError(
            f"No valid onset days found for id={wid!r}."
        )

    if cycle_index is None:
        selected_cycle = person_cycles.iloc[-1]
    else:
        matching_cycles = person_cycles.loc[
            person_cycles["cycle_index"]
            == cycle_index
        ]

        if matching_cycles.empty:
            raise ValueError(
                f"No cycle_index={cycle_index!r} "
                f"found for id={wid!r}."
            )

        selected_cycle = matching_cycles.iloc[0]

    onset = float(
        selected_cycle["onset_day"]
    )

    selected_cycle_index = (
        selected_cycle["cycle_index"]
    )

    stop_candidates = []

    # Use next_onset_day when available.
    if "next_onset_day" in selected_cycle.index:
        known_next_onset = pd.to_numeric(
            pd.Series(
                [
                    selected_cycle[
                        "next_onset_day"
                    ]
                ]
            ),
            errors="coerce",
        ).iloc[0]

        if (
            pd.notna(known_next_onset)
            and known_next_onset > onset
        ):
            stop_candidates.append(
                float(known_next_onset)
            )

    # Also check the following row's onset.
    later_onsets = person_cycles.loc[
        person_cycles["onset_day"] > onset,
        "onset_day",
    ]

    if not later_onsets.empty:
        stop_candidates.append(
            float(later_onsets.min())
        )

    panel_copy = panel.copy()

    panel_copy["day_in_study"] = pd.to_numeric(
        panel_copy["day_in_study"],
        errors="coerce",
    )

    days = panel_copy.loc[
        (panel_copy["id"] == wid)
        & (
            panel_copy["day_in_study"]
            >= onset
        )
    ].copy()

    if stop_candidates:
        stop_day = min(stop_candidates)

        days = days.loc[
            days["day_in_study"] < stop_day
        ]

    days["day"] = (
        days["day_in_study"] - onset
    )

    days = days.sort_values("day")

    history_rows = cycle_features.loc[
        (cycle_features["id"] == wid)
        & (
            cycle_features["cycle_index"]
            == selected_cycle_index
        )
    ]

    if history_rows.empty:
        history = {
            column: np.nan
            for column in HISTORY
        }
    else:
        history = (
            history_rows.iloc[0].to_dict()
        )

    return days, history