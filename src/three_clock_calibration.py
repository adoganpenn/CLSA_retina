"""Leakage-resistant symmetric calibration for retinal and methylation ages.

The functions in this module operate on one participant-level cohort and one
shared fold assignment.  Every clock is recalibrated from its age estimate to
chronological age with the same spline-plus-ridge pipeline.  Calibrators are
fit only on the training participants for a fold and applied to that fold's
held-out participants.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer


DEFAULT_CLOCK_COLUMNS = {
    "Retinal": "retinal_age_input",
    "Horvath": "horvath_age_input",
    "Hannum": "hannum_age_input",
}

METRIC_COLUMNS = (
    "mae",
    "median_absolute_error",
    "rmse",
    "predictive_r2",
    "pearson_r",
    "spearman_rho",
    "mean_error",
    "calibration_intercept",
    "calibration_slope",
    "ccc",
    "loa_lower",
    "loa_upper",
)


@dataclass(frozen=True)
class CalibrationConfig:
    n_splits: int = 5
    n_knots: int = 5
    degree: int = 3
    ridge_alpha: float = 1.0
    seed: int = 20261118

    def validate(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.n_knots < 3:
            raise ValueError("n_knots must be at least 3")
        if self.degree < 1:
            raise ValueError("degree must be positive")
        if self.ridge_alpha < 0:
            raise ValueError("ridge_alpha cannot be negative")


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def deterministic_participant_folds(
    participant_ids: Iterable[object],
    *,
    n_splits: int = 5,
    seed: int = 20261118,
) -> pd.Series:
    """Assign participant-stable, approximately balanced deterministic folds."""

    values = pd.Series(participant_ids, dtype="string")
    if values.isna().any() or values.str.strip().eq("").any():
        raise ValueError("Participant identifiers must be nonmissing and nonempty")
    unique = values.drop_duplicates()
    if len(unique) < n_splits:
        raise ValueError("Fewer unique participants than requested folds")

    def digest(value: str) -> str:
        payload = f"{seed}|{value}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    ordered = sorted(unique.astype(str), key=lambda value: (digest(value), value))
    mapping = {value: index % n_splits for index, value in enumerate(ordered)}
    return values.astype(str).map(mapping).astype(int)


def normalize_fold_ids(
    frame: pd.DataFrame,
    participant_column: str,
    *,
    fold_column: str | None = None,
    config: CalibrationConfig = CalibrationConfig(),
) -> pd.Series:
    """Use a complete existing fold column or construct deterministic folds."""

    config.validate()
    if fold_column and fold_column in frame.columns:
        folds = frame[fold_column]
        if folds.notna().all() and folds.nunique(dropna=True) >= 2:
            normalized = pd.Categorical(folds.astype(str)).codes
            return pd.Series(normalized, index=frame.index, dtype=int)
    generated = deterministic_participant_folds(
        frame[participant_column],
        n_splits=config.n_splits,
        seed=config.seed,
    )
    generated.index = frame.index
    return generated


def prepare_three_clock_frame(
    source: pd.DataFrame,
    *,
    participant_column: str = "participant_id",
    age_column: str = "chronological_age",
    retinal_candidates: Sequence[str] = (
        "retinal_age_raw_oof",
        "retinal_age_oof",
        "retinal_age",
    ),
    horvath_candidates: Sequence[str] = (
        "epigenetic_dnam_age",
        "DNAmAge_COM",
        "horvath_age",
    ),
    hannum_candidates: Sequence[str] = (
        "epigenetic_hannum_age",
        "Hannum_Age_COM",
        "hannum_age",
    ),
    fold_column: str | None = "cv_fold",
    config: CalibrationConfig = CalibrationConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the complete, participant-unique common comparison cohort.

    Returns the analysis frame and a flow table.  The flow table makes all
    exclusions explicit and records whether the retinal input is the raw OOF
    prediction or the already calibrated OOF estimate.
    """

    if participant_column not in source or age_column not in source:
        raise KeyError(f"Required columns missing: {participant_column}, {age_column}")

    def choose(candidates: Sequence[str], label: str) -> str:
        for candidate in candidates:
            if candidate in source.columns:
                return candidate
        raise KeyError(f"No {label} column found. Tried {list(candidates)}")

    retinal = choose(retinal_candidates, "retinal-age")
    horvath = choose(horvath_candidates, "Horvath-age")
    hannum = choose(hannum_candidates, "Hannum-age")
    chosen = {
        "retinal_age_input": retinal,
        "horvath_age_input": horvath,
        "hannum_age_input": hannum,
    }

    keep = [participant_column, age_column, *chosen.values()]
    if fold_column and fold_column in source.columns:
        keep.append(fold_column)
    optional = [
        column
        for column in source.columns
        if column
        in {
            "sex_labeled",
            "sex",
            "racial_background",
            "site",
            "collection_site",
            "geographic_stratum",
            "cataract_any",
            "ICQ_CATRCT_COM",
            "VIS_CATRCT_COM",
            "n_embedded_images",
        }
    ]
    keep.extend(column for column in optional if column not in keep)
    work = source[keep].copy()
    work[participant_column] = work[participant_column].astype("string").str.strip()
    work[age_column] = _numeric(work[age_column])
    for destination, original in chosen.items():
        work[destination] = _numeric(work[original])

    duplicate_mask = work[participant_column].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        raise ValueError(
            "The source is not participant-unique: "
            f"{duplicate_count} rows have duplicated participant identifiers"
        )

    required = [
        participant_column,
        age_column,
        "retinal_age_input",
        "horvath_age_input",
        "hannum_age_input",
    ]
    finite = np.isfinite(work[required[1:]].to_numpy(float)).all(axis=1)
    complete = work[required].notna().all(axis=1) & finite
    analysis = work.loc[complete].copy().reset_index(drop=True)
    if len(analysis) < config.n_splits:
        raise ValueError("Too few complete participants for cross-fitted calibration")
    analysis["calibration_fold"] = normalize_fold_ids(
        analysis,
        participant_column,
        fold_column=fold_column,
        config=config,
    ).to_numpy()
    analysis = analysis.rename(columns={age_column: "chronological_age"})

    flow = pd.DataFrame(
        [
            {"step": "Source participant rows", "n": int(len(work))},
            {
                "step": "Nonmissing chronological age",
                "n": int(work[age_column].notna().sum()),
            },
            {
                "step": "Nonmissing retinal age input",
                "n": int(work["retinal_age_input"].notna().sum()),
            },
            {
                "step": "Nonmissing Horvath age",
                "n": int(work["horvath_age_input"].notna().sum()),
            },
            {
                "step": "Nonmissing Hannum age",
                "n": int(work["hannum_age_input"].notna().sum()),
            },
            {"step": "Complete common cohort", "n": int(len(analysis))},
        ]
    )
    flow["retinal_input_column"] = retinal
    flow["horvath_input_column"] = horvath
    flow["hannum_input_column"] = hannum
    flow["retinal_input_is_raw_oof"] = retinal == "retinal_age_raw_oof"
    return analysis, flow


def make_calibrator(config: CalibrationConfig) -> Pipeline:
    config.validate()
    return Pipeline(
        [
            (
                "spline",
                SplineTransformer(
                    n_knots=config.n_knots,
                    degree=config.degree,
                    include_bias=False,
                    extrapolation="linear",
                ),
            ),
            ("ridge", Ridge(alpha=config.ridge_alpha)),
        ]
    )


def cross_fitted_calibrate(
    frame: pd.DataFrame,
    prediction_column: str,
    *,
    target_column: str = "chronological_age",
    fold_column: str = "calibration_fold",
    config: CalibrationConfig = CalibrationConfig(),
) -> tuple[pd.Series, pd.DataFrame]:
    """Apply the same training-fold-only calibrator to every held-out fold."""

    required = [prediction_column, target_column, fold_column]
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError(f"Missing calibration columns: {missing}")
    x = _numeric(frame[prediction_column]).to_numpy().reshape(-1, 1)
    y = _numeric(frame[target_column]).to_numpy()
    folds = frame[fold_column].to_numpy()
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Calibration inputs must be finite and complete")

    calibrated = np.full(len(frame), np.nan, dtype=float)
    audits: list[dict[str, object]] = []
    unique_folds = pd.unique(folds)
    if len(unique_folds) < 2:
        raise ValueError("At least two calibration folds are required")
    for fold in unique_folds:
        test = folds == fold
        train = ~test
        if train.sum() < max(config.n_knots + 2, 10) or test.sum() == 0:
            raise ValueError(f"Fold {fold!r} is too small for calibration")
        model = make_calibrator(config)
        model.fit(x[train], y[train])
        calibrated[test] = model.predict(x[test])
        audits.append(
            {
                "prediction_column": prediction_column,
                "held_out_fold": str(fold),
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                "training_folds": "|".join(
                    sorted({str(value) for value in folds[train]})
                ),
                "test_input_min": float(x[test].min()),
                "test_input_max": float(x[test].max()),
                "train_input_min": float(x[train].min()),
                "train_input_max": float(x[train].max()),
                "n_knots": config.n_knots,
                "degree": config.degree,
                "ridge_alpha": config.ridge_alpha,
                "extrapolation": "linear",
            }
        )
    if not np.isfinite(calibrated).all():
        raise RuntimeError("Some held-out rows did not receive calibrated predictions")
    return pd.Series(calibrated, index=frame.index), pd.DataFrame(audits)


def lin_ccc(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    covariance = float(np.mean((observed - observed.mean()) * (predicted - predicted.mean())))
    denominator = (
        float(np.var(observed, ddof=0))
        + float(np.var(predicted, ddof=0))
        + float((observed.mean() - predicted.mean()) ** 2)
    )
    return float(2 * covariance / denominator) if denominator > 0 else np.nan


def regression_metrics(observed: Iterable[float], predicted: Iterable[float]) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid]
    predicted = predicted[valid]
    if len(observed) < 3:
        raise ValueError("At least three complete observations are required")
    error = predicted - observed
    slope, intercept = np.polyfit(observed, predicted, 1)
    pearson = stats.pearsonr(observed, predicted).statistic
    spearman = stats.spearmanr(observed, predicted).statistic
    error_sd = float(np.std(error, ddof=1))
    return {
        "n": int(len(observed)),
        "mae": float(mean_absolute_error(observed, predicted)),
        "median_absolute_error": float(np.median(np.abs(error))),
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "predictive_r2": float(r2_score(observed, predicted)),
        "pearson_r": float(pearson),
        "spearman_rho": float(spearman),
        "mean_error": float(np.mean(error)),
        "error_sd": error_sd,
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
        "ccc": lin_ccc(observed, predicted),
        "loa_lower": float(np.mean(error) - 1.96 * error_sd),
        "loa_upper": float(np.mean(error) + 1.96 * error_sd),
    }


def metric_table(
    frame: pd.DataFrame,
    clock_columns: Mapping[str, str],
    *,
    target_column: str = "chronological_age",
    stage: str,
) -> pd.DataFrame:
    rows = []
    for clock, column in clock_columns.items():
        rows.append(
            {
                "stage": stage,
                "clock": clock,
                "prediction_column": column,
                **regression_metrics(frame[target_column], frame[column]),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_metric_intervals(
    frame: pd.DataFrame,
    clock_columns: Mapping[str, str],
    *,
    target_column: str = "chronological_age",
    stage: str,
    repetitions: int = 2000,
    seed: int = 20261118,
) -> pd.DataFrame:
    if repetitions < 100:
        raise ValueError("At least 100 bootstrap repetitions are required")
    rng = np.random.default_rng(seed)
    observed = frame[target_column].to_numpy(float)
    point = metric_table(frame, clock_columns, target_column=target_column, stage=stage)
    boot = {clock: {metric: [] for metric in METRIC_COLUMNS} for clock in clock_columns}
    for _ in range(repetitions):
        indices = rng.integers(0, len(frame), len(frame))
        sampled_observed = observed[indices]
        if np.unique(sampled_observed).size < 3:
            continue
        for clock, column in clock_columns.items():
            values = regression_metrics(sampled_observed, frame[column].to_numpy(float)[indices])
            for metric in METRIC_COLUMNS:
                boot[clock][metric].append(values[metric])

    rows = []
    for clock in clock_columns:
        point_row = point.loc[point["clock"].eq(clock)].iloc[0]
        for metric in METRIC_COLUMNS:
            values = np.asarray(boot[clock][metric], dtype=float)
            rows.append(
                {
                    "stage": stage,
                    "clock": clock,
                    "metric": metric,
                    "estimate": float(point_row[metric]),
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "bootstrap_repetitions": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def _signflip_pvalue(differences: np.ndarray, repetitions: int, rng: np.random.Generator) -> float:
    observed = abs(float(np.mean(differences)))
    exceed = 1
    for _ in range(repetitions):
        signs = rng.choice((-1.0, 1.0), size=len(differences))
        exceed += abs(float(np.mean(differences * signs))) >= observed
    return float(exceed / (repetitions + 1))


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def paired_clock_comparisons(
    frame: pd.DataFrame,
    clock_columns: Mapping[str, str],
    *,
    target_column: str = "chronological_age",
    stage: str,
    repetitions: int = 10000,
    seed: int = 20261118,
) -> pd.DataFrame:
    """Compare clocks on identical participants using paired resampling.

    Positive MAE and RMSE differences mean the first named clock has greater
    error. Positive R2 and CCC differences mean the first clock performs better.
    """

    if repetitions < 100:
        raise ValueError("At least 100 paired repetitions are required")
    y = frame[target_column].to_numpy(float)
    rng = np.random.default_rng(seed)
    rows = []

    def comparison_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
        error = predicted - observed
        return {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "predictive_r2": float(r2_score(observed, predicted)),
            "ccc": lin_ccc(observed, predicted),
        }

    for first, second in itertools.combinations(clock_columns, 2):
        first_prediction = frame[clock_columns[first]].to_numpy(float)
        second_prediction = frame[clock_columns[second]].to_numpy(float)
        first_abs = np.abs(first_prediction - y)
        second_abs = np.abs(second_prediction - y)
        paired_abs_difference = first_abs - second_abs
        point_first = comparison_metrics(y, first_prediction)
        point_second = comparison_metrics(y, second_prediction)
        bootstrap = {metric: [] for metric in ("mae", "rmse", "predictive_r2", "ccc")}
        for _ in range(repetitions):
            indices = rng.integers(0, len(frame), len(frame))
            sampled_y = y[indices]
            if np.unique(sampled_y).size < 3:
                continue
            first_metrics = comparison_metrics(sampled_y, first_prediction[indices])
            second_metrics = comparison_metrics(sampled_y, second_prediction[indices])
            for metric in bootstrap:
                bootstrap[metric].append(first_metrics[metric] - second_metrics[metric])
        row = {
            "stage": stage,
            "first_clock": first,
            "second_clock": second,
            "n": int(len(frame)),
            "mean_paired_absolute_error_difference": float(paired_abs_difference.mean()),
            "median_paired_absolute_error_difference": float(np.median(paired_abs_difference)),
            "first_better_participant_fraction": float(np.mean(first_abs < second_abs)),
            "ties_fraction": float(np.mean(first_abs == second_abs)),
            "paired_signflip_p": _signflip_pvalue(
                paired_abs_difference,
                repetitions,
                np.random.default_rng(seed + 7919 * (len(rows) + 1)),
            ),
        }
        for metric, values in bootstrap.items():
            array = np.asarray(values, dtype=float)
            row[f"delta_{metric}"] = float(point_first[metric] - point_second[metric])
            row[f"delta_{metric}_ci_low"] = float(np.quantile(array, 0.025))
            row[f"delta_{metric}_ci_high"] = float(np.quantile(array, 0.975))
        rows.append(row)
    result = pd.DataFrame(rows)
    result["paired_signflip_q_bh"] = benjamini_hochberg(result["paired_signflip_p"])
    result["mae_difference_direction"] = np.where(
        result["delta_mae"] < 0,
        "first_clock_lower_mae",
        np.where(result["delta_mae"] > 0, "second_clock_lower_mae", "tie"),
    )
    return result


def age_bin_performance(
    frame: pd.DataFrame,
    clock_columns: Mapping[str, str],
    *,
    target_column: str = "chronological_age",
    minimum_n: int = 30,
) -> pd.DataFrame:
    lower = int(np.floor(frame[target_column].min() / 5) * 5)
    upper = int(np.ceil(frame[target_column].max() / 5) * 5 + 5)
    bins = np.arange(lower, upper, 5)
    work = frame.copy()
    work["age_bin_5y"] = pd.cut(work[target_column], bins=bins, right=False)
    rows = []
    for age_bin, subset in work.groupby("age_bin_5y", observed=True):
        if len(subset) < minimum_n:
            continue
        for clock, column in clock_columns.items():
            rows.append(
                {
                    "age_bin_5y": str(age_bin),
                    "clock": clock,
                    **regression_metrics(subset[target_column], subset[column]),
                }
            )
    return pd.DataFrame(rows)


def subgroup_performance(
    frame: pd.DataFrame,
    clock_columns: Mapping[str, str],
    subgroup_columns: Sequence[str],
    *,
    target_column: str = "chronological_age",
    minimum_n: int = 30,
) -> pd.DataFrame:
    rows = []
    for subgroup_column in subgroup_columns:
        if subgroup_column not in frame.columns:
            continue
        for level, subset in frame.dropna(subset=[subgroup_column]).groupby(
            subgroup_column, observed=True
        ):
            if len(subset) < minimum_n:
                continue
            for clock, prediction_column in clock_columns.items():
                rows.append(
                    {
                        "subgroup_variable": subgroup_column,
                        "subgroup_level": str(level),
                        "clock": clock,
                        **regression_metrics(subset[target_column], subset[prediction_column]),
                    }
                )
    return pd.DataFrame(rows)


def calibration_curve_table(
    frame: pd.DataFrame,
    clock_columns: Mapping[str, str],
    *,
    target_column: str = "chronological_age",
    minimum_n: int = 30,
) -> pd.DataFrame:
    age_bins = age_bin_performance(
        frame,
        clock_columns,
        target_column=target_column,
        minimum_n=minimum_n,
    )
    if age_bins.empty:
        return age_bins
    return age_bins[
        [
            "age_bin_5y",
            "clock",
            "n",
            "mae",
            "rmse",
            "mean_error",
            "calibration_intercept",
            "calibration_slope",
        ]
    ].copy()
