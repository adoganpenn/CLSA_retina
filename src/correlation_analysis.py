"""Leakage-aware retinal-age and RETFound comorbidity analyses.

The functions in this module are deliberately independent of Databricks.  The
``notebooks/correlation.py`` notebook handles Spark I/O and uses these helpers
for participant-level inference and grouped cross-validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values, preserving missing values."""

    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(values[valid])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def summarize_retinal_age_strata(
    frame: pd.DataFrame,
    variables: Sequence[str],
    *,
    participant_column: str = "participant_id",
) -> pd.DataFrame:
    """Summarize participant-visit retinal age and gap within SAP strata."""

    required = {
        participant_column,
        "retinal_age",
        "retinal_age_gap",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Retinal-age frame is missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for variable in variables:
        if variable not in frame.columns:
            rows.append(
                {
                    "stratifier": variable,
                    "level": None,
                    "analysis_status": "column_missing",
                }
            )
            continue
        working = frame.loc[
            frame[variable].notna() & frame["retinal_age"].notna()
        ].copy()
        if working.empty:
            rows.append(
                {
                    "stratifier": variable,
                    "level": None,
                    "analysis_status": "no_observed_records",
                }
            )
            continue
        for level, group in working.groupby(variable, dropna=False, observed=True):
            retinal_age = pd.to_numeric(group["retinal_age"], errors="coerce")
            gap = pd.to_numeric(group["retinal_age_gap"], errors="coerce")
            rows.append(
                {
                    "stratifier": variable,
                    "level": str(level),
                    "n_participant_visits": int(len(group)),
                    "n_participants": int(group[participant_column].nunique()),
                    "retinal_age_mean": float(retinal_age.mean()),
                    "retinal_age_sd": float(retinal_age.std(ddof=1)),
                    "retinal_age_median": float(retinal_age.median()),
                    "retinal_age_q1": float(retinal_age.quantile(0.25)),
                    "retinal_age_q3": float(retinal_age.quantile(0.75)),
                    "retinal_age_gap_mean": float(gap.mean()),
                    "retinal_age_gap_sd": float(gap.std(ddof=1)),
                    "retinal_age_gap_median": float(gap.median()),
                    "retinal_age_gap_q1": float(gap.quantile(0.25)),
                    "retinal_age_gap_q3": float(gap.quantile(0.75)),
                    "analysis_status": "ok",
                }
            )
    return pd.DataFrame(rows)


def _numeric_design(series: pd.Series, name: str) -> pd.DataFrame:
    values = pd.to_numeric(series, errors="coerce")
    return pd.DataFrame({name: values}, index=series.index)


def _categorical_design(
    series: pd.Series,
    name: str,
    *,
    drop_first: bool = True,
) -> tuple[pd.DataFrame, str | None]:
    clean = series.astype("string")
    levels = sorted(clean.dropna().unique().tolist())
    if len(levels) < 2:
        return pd.DataFrame(index=series.index), levels[0] if levels else None
    categorical = pd.Categorical(clean, categories=levels)
    design = pd.get_dummies(
        categorical,
        prefix=name,
        prefix_sep="=",
        drop_first=drop_first,
        dtype=float,
    )
    design.index = series.index
    return design, levels[0] if drop_first else None


def fit_retinal_age_associations(
    frame: pd.DataFrame,
    exposures: Mapping[str, str],
    *,
    outcome: str = "retinal_age_gap",
    participant_column: str = "participant_id",
    age_column: str = "age_at_fundus_years",
    sex_column: str = "sex_at_birth",
    visit_column: str = "visit",
    minimum_records: int = 100,
) -> pd.DataFrame:
    """Fit one adjusted retinal-age-gap model per SAP exposure.

    Models adjust for chronological age, sex, and visit.  Standard errors are
    clustered by participant so BL and F1 observations from the same person are
    not treated as independent.  Categorical exposures are reported relative to
    their lexicographically first observed level.
    """

    try:
        import statsmodels.api as sm
    except ImportError as exc:  # pragma: no cover - exercised on Databricks
        raise ImportError(
            "statsmodels is required; install requirements-retfound.txt first."
        ) from exc

    required = {
        outcome,
        participant_column,
        age_column,
        sex_column,
        visit_column,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Association frame is missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for exposure, exposure_type in exposures.items():
        if exposure not in frame.columns:
            rows.append(
                {
                    "exposure": exposure,
                    "analysis_status": "column_missing",
                }
            )
            continue
        columns = [
            outcome,
            participant_column,
            age_column,
            sex_column,
            visit_column,
            exposure,
        ]
        working = frame.loc[:, list(dict.fromkeys(columns))].copy()
        working[outcome] = pd.to_numeric(working[outcome], errors="coerce")
        working[age_column] = pd.to_numeric(working[age_column], errors="coerce")
        complete = [outcome, participant_column, age_column, sex_column, visit_column]
        complete.append(exposure)
        working = working.dropna(subset=list(dict.fromkeys(complete)))
        if len(working) < minimum_records:
            rows.append(
                {
                    "exposure": exposure,
                    "n_participant_visits": int(len(working)),
                    "n_participants": int(working[participant_column].nunique()),
                    "analysis_status": "insufficient_records",
                }
            )
            continue

        design_parts = [_numeric_design(working[age_column], age_column)]
        if exposure != sex_column:
            sex_design, _ = _categorical_design(
                working[sex_column], sex_column
            )
            design_parts.append(sex_design)
        if exposure != visit_column:
            visit_design, _ = _categorical_design(
                working[visit_column], visit_column
            )
            design_parts.append(visit_design)

        if exposure_type == "continuous":
            exposure_design = _numeric_design(working[exposure], exposure)
            reference = None
        elif exposure_type == "categorical":
            exposure_design, reference = _categorical_design(
                working[exposure], exposure
            )
        else:
            raise ValueError(
                f"Unsupported exposure type for {exposure}: {exposure_type}"
            )
        if exposure_design.shape[1] == 0:
            rows.append(
                {
                    "exposure": exposure,
                    "n_participant_visits": int(len(working)),
                    "n_participants": int(working[participant_column].nunique()),
                    "analysis_status": "no_exposure_variation",
                }
            )
            continue

        design_parts.append(exposure_design)
        design = pd.concat(design_parts, axis=1)
        design = design.loc[:, ~design.columns.duplicated()]
        design = sm.add_constant(design.astype(float), has_constant="add")
        outcome_values = working.loc[design.index, outcome].astype(float)
        groups = working.loc[design.index, participant_column].astype(str)
        try:
            fitted = sm.OLS(outcome_values, design).fit(
                cov_type="cluster",
                cov_kwds={"groups": groups, "use_correction": True},
            )
            covariance = "participant_clustered"
        except Exception:
            fitted = sm.OLS(outcome_values, design).fit(cov_type="HC3")
            covariance = "HC3_fallback"

        for term in exposure_design.columns:
            if term not in fitted.params:
                continue
            confidence = fitted.conf_int().loc[term]
            level = term.split("=", 1)[1] if "=" in term else None
            rows.append(
                {
                    "exposure": exposure,
                    "exposure_type": exposure_type,
                    "term": term,
                    "level": level,
                    "reference_level": reference,
                    "n_participant_visits": int(len(working)),
                    "n_participants": int(groups.nunique()),
                    "beta_years": float(fitted.params[term]),
                    "standard_error": float(fitted.bse[term]),
                    "ci95_low": float(confidence.iloc[0]),
                    "ci95_high": float(confidence.iloc[1]),
                    "p_value": float(fitted.pvalues[term]),
                    "covariance": covariance,
                    "adjustment": "chronological_age + sex + visit",
                    "analysis_status": "ok",
                }
            )

    result = pd.DataFrame(rows)
    if "p_value" in result.columns:
        result["p_value_fdr_bh"] = benjamini_hochberg(result["p_value"])
    return result


@dataclass(frozen=True)
class ComorbidityModelConfig:
    n_splits: int = 5
    pca_components: int = 64
    minimum_positive_records: int = 50
    minimum_negative_records: int = 50
    random_seed: int = 20260727
    maximum_iterations: int = 2000


def _binary_target(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric.isin([0, 1]))


def _classification_metrics(
    observed: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        roc_auc_score,
    )

    observed = np.asarray(observed, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predicted = (probability >= 0.5).astype(int)
    true_positive = int(((observed == 1) & (predicted == 1)).sum())
    false_negative = int(((observed == 1) & (predicted == 0)).sum())
    true_negative = int(((observed == 0) & (predicted == 0)).sum())
    false_positive = int(((observed == 0) & (predicted == 1)).sum())
    sensitivity = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    return {
        "auroc": float(roc_auc_score(observed, probability)),
        "average_precision": float(
            average_precision_score(observed, probability)
        ),
        "balanced_accuracy_at_0_5": float(
            balanced_accuracy_score(observed, predicted)
        ),
        "sensitivity_at_0_5": float(sensitivity),
        "specificity_at_0_5": float(specificity),
        "brier_score": float(brier_score_loss(observed, probability)),
    }


def cross_validate_comorbidity_models(
    frame: pd.DataFrame,
    outcomes: Sequence[str],
    *,
    embedding_column: str = "embedding",
    participant_column: str = "participant_id",
    visit_column: str = "visit",
    age_column: str = "age_at_fundus_years",
    sex_column: str = "sex_at_birth",
    expected_embedding_dim: int = 1024,
    config: ComorbidityModelConfig = ComorbidityModelConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate clinical, RETFound, and combined disease classifiers.

    PCA/scaling is fit only on each outer training fold and shared across
    disease outcomes for efficiency.  GroupKFold keeps every visit from a
    participant in one fold.  Returned probabilities are out-of-fold only.
    """

    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    required = {
        embedding_column,
        participant_column,
        visit_column,
        age_column,
        sex_column,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Model frame is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Model frame is empty.")

    vectors = np.vstack(
        frame[embedding_column].map(
            lambda value: np.asarray(value, dtype=np.float32)
        )
    )
    if vectors.shape[1] != expected_embedding_dim:
        raise ValueError(
            f"Expected {expected_embedding_dim} RETFound features; "
            f"found {vectors.shape[1]}."
        )
    if not np.isfinite(vectors).all():
        raise ValueError("RETFound vectors contain NaN or infinity.")

    groups = frame[participant_column].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(config.n_splits, len(unique_groups))
    if n_splits < 2:
        raise ValueError("At least two participants are required for grouped CV.")

    clinical = frame[[age_column, sex_column, visit_column]].copy()
    clinical[age_column] = pd.to_numeric(clinical[age_column], errors="coerce")
    clinical[sex_column] = clinical[sex_column].astype("string")
    clinical[visit_column] = clinical[visit_column].astype("string")
    eligible_outcomes: set[str] = set()
    for outcome in outcomes:
        if outcome not in frame.columns:
            continue
        target = _binary_target(frame[outcome])
        if (
            int((target == 1).sum()) >= config.minimum_positive_records
            and int((target == 0).sum()) >= config.minimum_negative_records
        ):
            eligible_outcomes.add(outcome)
    fold_assignments = np.full(len(frame), -1, dtype=int)
    predictions: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_index, test_index) in enumerate(
        splitter.split(vectors, groups=groups)
    ):
        print(
            f"Preparing grouped fold {fold + 1}/{n_splits}: "
            f"train={len(train_index):,}, test={len(test_index):,}"
        )
        train_groups = set(groups[train_index])
        test_groups = set(groups[test_index])
        if train_groups & test_groups:
            raise RuntimeError("Participant leakage detected in grouped folds.")
        fold_assignments[test_index] = fold

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(vectors[train_index])
        test_scaled = scaler.transform(vectors[test_index])
        components = min(
            config.pca_components,
            expected_embedding_dim,
            max(1, len(train_index) - 1),
        )
        pca = PCA(
            n_components=components,
            svd_solver="randomized",
            random_state=config.random_seed + fold,
        )
        train_embedding = pca.fit_transform(train_scaled)
        test_embedding = pca.transform(test_scaled)

        clinical_transformer = ColumnTransformer(
            [
                (
                    "age",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    [age_column],
                ),
                (
                    "categorical",
                    Pipeline(
                        [
                            (
                                "impute",
                                SimpleImputer(strategy="most_frequent"),
                            ),
                            (
                                "one_hot",
                                OneHotEncoder(
                                    handle_unknown="ignore",
                                    sparse_output=False,
                                ),
                            ),
                        ]
                    ),
                    [sex_column, visit_column],
                ),
            ],
            remainder="drop",
        )
        train_clinical = clinical_transformer.fit_transform(
            clinical.iloc[train_index]
        )
        test_clinical = clinical_transformer.transform(
            clinical.iloc[test_index]
        )

        feature_sets = {
            "clinical": (train_clinical, test_clinical),
            "retfound_embedding": (train_embedding, test_embedding),
            "combined": (
                np.hstack([train_embedding, train_clinical]),
                np.hstack([test_embedding, test_clinical]),
            ),
        }
        for outcome in outcomes:
            if outcome not in eligible_outcomes:
                continue
            target = _binary_target(frame[outcome]).to_numpy()
            train_observed = np.isfinite(target[train_index])
            test_observed = np.isfinite(target[test_index])
            y_train = target[train_index][train_observed].astype(int)
            y_test = target[test_index][test_observed].astype(int)
            if (
                len(np.unique(y_train)) < 2
                or len(np.unique(y_test)) < 2
                or int((y_train == 1).sum()) < 2
                or int((y_train == 0).sum()) < 2
            ):
                continue
            for model_family, (train_features, test_features) in feature_sets.items():
                model = LogisticRegression(
                    class_weight="balanced",
                    max_iter=config.maximum_iterations,
                    random_state=config.random_seed + fold,
                    solver="lbfgs",
                )
                model.fit(train_features[train_observed], y_train)
                probability = model.predict_proba(
                    test_features[test_observed]
                )[:, 1]
                metrics = _classification_metrics(y_test, probability)
                metric_rows.append(
                    {
                        "outcome": outcome,
                        "model_family": model_family,
                        "evaluation_scope": "fold",
                        "fold": fold,
                        "n_records": int(len(y_test)),
                        "n_positive": int((y_test == 1).sum()),
                        "prevalence": float(y_test.mean()),
                        "pca_components": int(components),
                        **metrics,
                    }
                )
                observed_indices = test_index[test_observed]
                for row_index, observed, score in zip(
                    observed_indices,
                    y_test,
                    probability,
                ):
                    predictions.append(
                        {
                            "participant_id": str(
                                frame.iloc[row_index][participant_column]
                            ),
                            "visit": str(frame.iloc[row_index][visit_column]),
                            "outcome": outcome,
                            "model_family": model_family,
                            "fold": fold,
                            "observed": int(observed),
                            "predicted_probability": float(score),
                        }
                    )
        print(f"Completed grouped fold {fold + 1}/{n_splits}.")

    if (fold_assignments < 0).any():
        raise RuntimeError("At least one record did not receive an outer fold.")

    prediction_frame = pd.DataFrame(
        predictions,
        columns=[
            "participant_id",
            "visit",
            "outcome",
            "model_family",
            "fold",
            "observed",
            "predicted_probability",
        ],
    )
    for (outcome, model_family), group in prediction_frame.groupby(
        ["outcome", "model_family"], observed=True
    ):
        observed = group["observed"].to_numpy(dtype=int)
        probability = group["predicted_probability"].to_numpy(dtype=float)
        if len(np.unique(observed)) < 2:
            continue
        metric_rows.append(
            {
                "outcome": outcome,
                "model_family": model_family,
                "evaluation_scope": "pooled_oof",
                "fold": -1,
                "n_records": int(len(group)),
                "n_positive": int((observed == 1).sum()),
                "prevalence": float(observed.mean()),
                "pca_components": int(
                    min(config.pca_components, expected_embedding_dim)
                ),
                **_classification_metrics(observed, probability),
            }
        )

    availability_rows = []
    for outcome in outcomes:
        if outcome not in frame.columns:
            availability_rows.append(
                {"outcome": outcome, "analysis_status": "column_missing"}
            )
            continue
        target = _binary_target(frame[outcome])
        positive = int((target == 1).sum())
        negative = int((target == 0).sum())
        if positive < config.minimum_positive_records:
            status = "insufficient_positive_records"
        elif negative < config.minimum_negative_records:
            status = "insufficient_negative_records"
        elif not (
            (prediction_frame.get("outcome", pd.Series(dtype=str)) == outcome).any()
        ):
            status = "no_evaluable_grouped_folds"
        else:
            status = "modeled"
        availability_rows.append(
            {
                "outcome": outcome,
                "n_observed": int(target.notna().sum()),
                "n_positive": positive,
                "n_negative": negative,
                "prevalence": float(target.mean()) if target.notna().any() else None,
                "analysis_status": status,
            }
        )

    metrics_frame = pd.DataFrame(metric_rows)
    availability_frame = pd.DataFrame(availability_rows)
    return metrics_frame, prediction_frame, availability_frame
