"""Reusable helpers for a resumable two-group retinal-image analysis.

The Databricks notebook is the orchestration layer.  This module keeps the
parts that need unit-testable invariants outside notebook state: participant
matching, batch validation, participant-level aggregation, and grouped model
evaluation.  No function logs participant identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


def require_columns(frame: Any, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def slugify(value: object, default: str = "analysis") -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()
    return text or default


def stable_image_key(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:24]


def batch_ranges(total: int, batch_size: int) -> list[tuple[int, int]]:
    if total < 0:
        raise ValueError("total cannot be negative")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    return [
        (start, min(start + batch_size, total))
        for start in range(0, total, batch_size)
    ]


def consolidate_batch_parquets(
    paths: Sequence[str | Path],
    *,
    key_column: str,
    expected_keys: Iterable[object] | None = None,
    required_columns: Sequence[str] = (),
) -> Any:
    """Load completed batch Parquets and verify identity, schema, and grain."""
    import pandas as pd

    frames = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file() or path.stat().st_size < 1:
            raise FileNotFoundError(f"Completed batch is absent or empty: {path}")
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame(columns=[key_column, *required_columns])
    output = pd.concat(frames, ignore_index=True, sort=False)
    require_columns(output, [key_column, *required_columns], "Batch output")
    output[key_column] = output[key_column].astype(str)
    if output[key_column].duplicated().any():
        raise ValueError(f"Batch output is not unique by {key_column}")
    if expected_keys is not None:
        expected = {str(value) for value in expected_keys}
        observed = set(output[key_column])
        if observed != expected:
            raise ValueError(
                "Completed batch keys do not equal the expected cohort: "
                f"missing={len(expected - observed)}, unexpected={len(observed - expected)}"
            )
    output.attrs = {}
    return output


@dataclass(frozen=True)
class MatchConfig:
    ratio: int = 1
    caliper_years: float = 2.0
    exact_columns: tuple[str, ...] = ()
    case_label: int = 1
    control_label: int = 0


def match_participants(
    participants: Any,
    config: MatchConfig = MatchConfig(),
    *,
    id_column: str = "participant_id",
    age_column: str = "age",
    label_column: str = "group_label",
) -> tuple[Any, Any, Any]:
    """Deterministic nearest-age matching without control reuse.

    Cases with the fewest available candidates are matched first.  This is a
    transparent, reproducible greedy algorithm rather than an optimal-matching
    claim.  The returned pair and membership tables are private artifacts.
    """
    import numpy as np
    import pandas as pd

    if config.ratio < 1:
        raise ValueError("Matching ratio must be at least 1")
    if config.caliper_years < 0:
        raise ValueError("Age caliper cannot be negative")
    require_columns(
        participants,
        [id_column, age_column, label_column, *config.exact_columns],
        "Participant table",
    )
    work = participants.copy().reset_index(drop=True)
    work[id_column] = work[id_column].astype(str)
    work[age_column] = pd.to_numeric(work[age_column], errors="coerce")
    work[label_column] = pd.to_numeric(work[label_column], errors="coerce")
    if work[id_column].duplicated().any():
        raise ValueError("Matching requires one row per participant")
    if work[[id_column, age_column, label_column]].isna().any().any():
        raise ValueError("Matching IDs, ages, and labels cannot be missing")
    observed_labels = set(work[label_column].astype(int))
    expected_labels = {config.case_label, config.control_label}
    if observed_labels != expected_labels:
        raise ValueError(
            f"Matching requires labels {expected_labels}, found {observed_labels}"
        )
    cases = work[work[label_column].astype(int) == config.case_label].copy()
    controls = work[work[label_column].astype(int) == config.control_label].copy()
    available = set(controls.index)

    def eligible(case: Any, indices: set[int]) -> Any:
        candidate = controls.loc[sorted(indices)].copy()
        for column in config.exact_columns:
            case_value = case[column]
            if pd.isna(case_value):
                return candidate.iloc[0:0].copy()
            candidate = candidate[
                candidate[column].astype(str) == str(case_value)
            ].copy()
        candidate["_age_difference"] = candidate[age_column] - float(case[age_column])
        candidate["_absolute_age_difference"] = candidate["_age_difference"].abs()
        return candidate[
            candidate["_absolute_age_difference"] <= config.caliper_years
        ].sort_values(
            ["_absolute_age_difference", id_column], kind="stable"
        )

    difficulty = []
    for case_index, case in cases.iterrows():
        difficulty.append(
            (
                len(eligible(case, available)),
                float(case[age_column]),
                str(case[id_column]),
                case_index,
            )
        )

    pair_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    for sequence, (_, _, _, case_index) in enumerate(sorted(difficulty), start=1):
        case = cases.loc[case_index]
        candidate = eligible(case, available)
        if len(candidate) < config.ratio:
            audit_rows.append(
                {
                    "case_id": str(case[id_column]),
                    "case_age": float(case[age_column]),
                    "matched": False,
                    "reason": "insufficient_controls_within_caliper_or_exact_stratum",
                    "eligible_controls_remaining": int(len(candidate)),
                }
            )
            continue
        selected = candidate.head(config.ratio)
        match_set_id = f"match_{sequence:08d}"
        membership_rows.append(
            {
                "match_set_id": match_set_id,
                id_column: str(case[id_column]),
                label_column: int(config.case_label),
                "match_role": "case",
                "match_rank": 0,
            }
        )
        for rank, (control_index, control) in enumerate(selected.iterrows(), start=1):
            available.remove(control_index)
            pair_rows.append(
                {
                    "match_set_id": match_set_id,
                    "case_id": str(case[id_column]),
                    "control_id": str(control[id_column]),
                    "match_rank": int(rank),
                    "case_age": float(case[age_column]),
                    "control_age": float(control[age_column]),
                    "age_difference_years": float(control["_age_difference"]),
                    "absolute_age_difference_years": float(
                        control["_absolute_age_difference"]
                    ),
                }
            )
            membership_rows.append(
                {
                    "match_set_id": match_set_id,
                    id_column: str(control[id_column]),
                    label_column: int(config.control_label),
                    "match_role": "control",
                    "match_rank": int(rank),
                }
            )
        audit_rows.append(
            {
                "case_id": str(case[id_column]),
                "case_age": float(case[age_column]),
                "matched": True,
                "reason": None,
                "eligible_controls_remaining": int(len(candidate)),
            }
        )
    pair_columns = [
        "match_set_id",
        "case_id",
        "control_id",
        "match_rank",
        "case_age",
        "control_age",
        "age_difference_years",
        "absolute_age_difference_years",
    ]
    audit_columns = [
        "case_id",
        "case_age",
        "matched",
        "reason",
        "eligible_controls_remaining",
    ]
    membership_columns = [
        "match_set_id",
        id_column,
        label_column,
        "match_role",
        "match_rank",
    ]
    pairs = pd.DataFrame(pair_rows, columns=pair_columns)
    audit = pd.DataFrame(audit_rows, columns=audit_columns)
    membership = pd.DataFrame(membership_rows, columns=membership_columns)
    for frame in (pairs, audit, membership):
        frame.attrs = {}
    return pairs, audit, membership


def matching_balance(
    before: Any,
    after: Any,
    *,
    age_column: str = "age",
    label_column: str = "group_label",
) -> Any:
    """Return age balance before and after matching without identifiers."""
    import numpy as np
    import pandas as pd

    rows = []
    for stage, frame in (("before", before), ("after", after)):
        require_columns(frame, [age_column, label_column], f"{stage} matching frame")
        age = pd.to_numeric(frame[age_column], errors="coerce")
        label = pd.to_numeric(frame[label_column], errors="coerce")
        group_0 = age[label == 0].dropna().to_numpy(float)
        group_1 = age[label == 1].dropna().to_numpy(float)
        pooled_sd = math.sqrt(
            ((group_0.var(ddof=1) + group_1.var(ddof=1)) / 2)
        ) if len(group_0) > 1 and len(group_1) > 1 else math.nan
        rows.append(
            {
                "stage": stage,
                "group_a_n": int(len(group_0)),
                "group_b_n": int(len(group_1)),
                "group_a_age_mean": float(np.mean(group_0)) if len(group_0) else math.nan,
                "group_b_age_mean": float(np.mean(group_1)) if len(group_1) else math.nan,
                "age_standardized_mean_difference": (
                    float((np.mean(group_1) - np.mean(group_0)) / pooled_sd)
                    if pooled_sd and np.isfinite(pooled_sd)
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def aggregate_participant_embeddings(
    embeddings: Any,
    *,
    id_column: str = "participant_id",
    embedding_column: str = "embedding",
    expected_dim: int = 1024,
    carry_columns: Sequence[str] = (
        "group_label",
        "age",
        "match_set_id",
    ),
) -> Any:
    """Mean-pool image embeddings after validating participant invariants."""
    import numpy as np
    import pandas as pd

    require_columns(
        embeddings,
        [id_column, embedding_column, *carry_columns],
        "Image embeddings",
    )
    work = embeddings.copy()
    work[id_column] = work[id_column].astype(str)
    vectors = []
    for value in work[embedding_column]:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if vector.size != expected_dim or not np.isfinite(vector).all():
            raise ValueError("An image contains an invalid RETFound embedding")
        vectors.append(vector)
    work["_validated_embedding"] = vectors
    rows = []
    for participant_id, group in work.groupby(id_column, sort=True):
        row: dict[str, Any] = {
            id_column: str(participant_id),
            embedding_column: np.mean(
                np.stack(group["_validated_embedding"].to_list()), axis=0
            ).astype(np.float32),
            "n_embedded_images": int(len(group)),
        }
        for column in carry_columns:
            values = group[column].dropna().astype(str).unique()
            if len(values) != 1:
                raise ValueError(
                    f"Participant {column} must be invariant across images"
                )
            original = group[column].dropna().iloc[0]
            row[column] = original
        rows.append(row)
    output = pd.DataFrame(rows)
    output.attrs = {}
    return output


def _fit_scaled_logistic(matrix: Any, label: Any, c_value: float) -> dict[str, Any]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(matrix)
    scaled = scaler.transform(matrix)
    estimator = LogisticRegression(
        C=float(c_value),
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=20260819,
    ).fit(scaled, label)
    coefficients = estimator.coef_.reshape(-1).astype(np.float64) / np.maximum(
        scaler.scale_, 1e-12
    )
    intercept = float(estimator.intercept_[0] - coefficients @ scaler.mean_)
    return {
        "model_type": "scaled_l2_logistic_regression",
        "c_value": float(c_value),
        "embedding_dim": int(matrix.shape[1]),
        "coefficients": coefficients,
        "intercept": intercept,
        "scaler_mean": scaler.mean_.astype(np.float64),
        "scaler_scale": scaler.scale_.astype(np.float64),
        "estimator": estimator,
    }


def fit_grouped_oof_classifier(
    frame: Any,
    *,
    folds: int = 5,
    inner_folds: int = 4,
    c_grid: Sequence[float] = (0.001, 0.01, 0.1, 1.0),
    expected_dim: int = 1024,
    embedding_column: str = "embedding",
    label_column: str = "group_label",
    id_column: str = "participant_id",
    group_column: str = "match_set_id",
    random_state: int = 20260819,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    """Nested CV that keeps every matched set entirely within one fold."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    if folds < 3 or inner_folds < 2:
        raise ValueError("Use at least 3 outer and 2 inner folds")
    if not c_grid or any(float(value) <= 0 for value in c_grid):
        raise ValueError("c_grid must contain positive values")
    require_columns(
        frame,
        [id_column, label_column, group_column, embedding_column],
        "Participant classifier frame",
    )
    work = frame.copy().reset_index(drop=True)
    work[id_column] = work[id_column].astype(str)
    work[group_column] = work[group_column].astype(str)
    if work[id_column].duplicated().any():
        raise ValueError("Classifier frame must be unique by participant")
    label = pd.to_numeric(work[label_column], errors="coerce").astype(int).to_numpy()
    if set(label) != {0, 1}:
        raise ValueError("Classifier requires both binary groups")
    matrix = np.stack(
        [np.asarray(value, dtype=np.float64).reshape(-1) for value in work[embedding_column]]
    )
    if matrix.shape[1] != expected_dim or not np.isfinite(matrix).all():
        raise ValueError("Classifier embeddings have the wrong dimension or nonfinite values")
    groups = work[group_column].to_numpy()
    unique_groups = np.unique(groups)
    usable_folds = min(int(folds), len(unique_groups))
    if usable_folds < 3:
        raise ValueError("At least three matched sets are required for grouped CV")
    outer = StratifiedGroupKFold(
        n_splits=usable_folds, shuffle=True, random_state=random_state
    )
    oof_logit = np.full(len(work), np.nan, dtype=float)
    oof_probability = np.full(len(work), np.nan, dtype=float)
    fold_assignment = np.full(len(work), -1, dtype=int)
    heads: list[dict[str, Any]] = []
    for fold, (development, evaluation) in enumerate(
        outer.split(matrix, label, groups)
    ):
        if set(groups[development]) & set(groups[evaluation]):
            raise RuntimeError("A matched set leaked across outer folds")
        development_label = label[development]
        development_groups = groups[development]
        usable_inner = min(inner_folds, len(np.unique(development_groups)))
        if usable_inner < 2:
            raise ValueError("Not enough development matched sets for inner CV")
        inner = StratifiedGroupKFold(
            n_splits=usable_inner,
            shuffle=True,
            random_state=random_state + fold + 1,
        )
        scores = []
        for c_value in sorted(float(value) for value in c_grid):
            aucs = []
            for inner_train, inner_test in inner.split(
                matrix[development], development_label, development_groups
            ):
                if len(np.unique(development_label[inner_test])) < 2:
                    continue
                head = _fit_scaled_logistic(
                    matrix[development][inner_train],
                    development_label[inner_train],
                    c_value,
                )
                logit = (
                    matrix[development][inner_test] @ head["coefficients"]
                    + head["intercept"]
                )
                aucs.append(roc_auc_score(development_label[inner_test], logit))
            if aucs:
                scores.append((float(np.mean(aucs)), c_value))
        if not scores:
            raise ValueError("Grouped inner CV could not produce a two-class validation fold")
        best_auc = max(score for score, _ in scores)
        best_c = min(value for score, value in scores if np.isclose(score, best_auc))
        head = _fit_scaled_logistic(matrix[development], development_label, best_c)
        head.update(
            {
                "fold": int(fold),
                "inner_mean_auc": float(best_auc),
                "n_development_participants": int(len(development)),
                "n_evaluation_participants": int(len(evaluation)),
                "grouped_by": group_column,
            }
        )
        logit = matrix[evaluation] @ head["coefficients"] + head["intercept"]
        probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -40, 40)))
        oof_logit[evaluation] = logit
        oof_probability[evaluation] = probability
        fold_assignment[evaluation] = fold
        heads.append(head)
    if np.isnan(oof_probability).any() or np.any(fold_assignment < 0):
        raise RuntimeError("Grouped OOF classifier did not score every participant")
    output = work[[id_column, label_column, group_column]].copy()
    output["fold"] = fold_assignment
    output["classifier_logit_oof"] = oof_logit
    output["group_b_probability_oof"] = oof_probability
    selected_c = min(
        sorted(float(value) for value in c_grid),
        key=lambda value: (
            -sum(np.isclose(head["c_value"], value) for head in heads),
            value,
        ),
    )
    final_head = _fit_scaled_logistic(matrix, label, selected_c)
    final_head.update(
        {
            "model_name": "RETFound_two_group_linear",
            "frozen": True,
            "selected_from_outer_folds": True,
            "outer_folds": int(usable_folds),
            "participant_level_training": True,
            "grouped_by": group_column,
        }
    )
    for result in (output,):
        result.attrs = {}
    return output, heads, final_head


def prediction_metrics(
    label: Any,
    probability: Any,
    *,
    bootstrap_repetitions: int = 2000,
    random_state: int = 20260819,
) -> dict[str, Any]:
    """Participant-level discrimination and calibration with stratified CIs."""
    import numpy as np
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        roc_auc_score,
    )

    label = np.asarray(label, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if label.shape != probability.shape or set(label) != {0, 1}:
        raise ValueError("Metrics require aligned binary labels and probabilities")
    rng = np.random.default_rng(random_state)
    class_indices = [np.flatnonzero(label == value) for value in (0, 1)]
    aucs = []
    average_precisions = []
    for _ in range(int(bootstrap_repetitions)):
        sampled = np.concatenate(
            [rng.choice(index, len(index), replace=True) for index in class_indices]
        )
        aucs.append(roc_auc_score(label[sampled], probability[sampled]))
        average_precisions.append(
            average_precision_score(label[sampled], probability[sampled])
        )
    prediction = probability >= 0.5
    tn, fp, fn, tp = confusion_matrix(label, prediction, labels=[0, 1]).ravel()
    return {
        "n_participants": int(len(label)),
        "n_group_b": int(label.sum()),
        "n_group_a": int((label == 0).sum()),
        "auroc": float(roc_auc_score(label, probability)),
        "auroc_95_ci_low": float(np.quantile(aucs, 0.025)),
        "auroc_95_ci_high": float(np.quantile(aucs, 0.975)),
        "average_precision": float(average_precision_score(label, probability)),
        "average_precision_95_ci_low": float(np.quantile(average_precisions, 0.025)),
        "average_precision_95_ci_high": float(np.quantile(average_precisions, 0.975)),
        "brier_score": float(brier_score_loss(label, probability)),
        "sensitivity_at_0_5": float(tp / max(tp + fn, 1)),
        "specificity_at_0_5": float(tn / max(tn + fp, 1)),
    }


def matched_set_permutation_inference(
    frame: Any,
    metric_columns: Sequence[str],
    *,
    label_column: str = "group_label",
    set_column: str = "match_set_id",
    permutations: int = 5000,
    bootstrap_repetitions: int = 2000,
    random_state: int = 20260819,
) -> Any:
    """Matched-set label permutation with max-|T| family-wise control."""
    import numpy as np
    import pandas as pd

    require_columns(
        frame,
        [label_column, set_column, *metric_columns],
        "Matched inference frame",
    )
    work = frame.copy().reset_index(drop=True)
    labels = pd.to_numeric(work[label_column], errors="coerce").to_numpy(int)
    matrix = work[list(metric_columns)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    if set(labels) != {0, 1} or not np.isfinite(matrix).all():
        raise ValueError("Matched inference requires finite metrics and both groups")
    set_indices = [
        index.to_numpy(int)
        for _, index in work.groupby(set_column, sort=True).groups.items()
    ]
    if len(set_indices) < 2:
        raise ValueError("Matched inference requires at least two matched sets")
    for indices in set_indices:
        if len(np.unique(labels[indices])) != 2:
            raise ValueError("Every matched set must contain both groups")

    def statistic(current_labels: Any) -> tuple[Any, Any]:
        group_a = matrix[current_labels == 0]
        group_b = matrix[current_labels == 1]
        difference = group_b.mean(axis=0) - group_a.mean(axis=0)
        standard_error = np.sqrt(
            group_a.var(axis=0, ddof=1) / len(group_a)
            + group_b.var(axis=0, ddof=1) / len(group_b)
        )
        return difference, difference / np.maximum(standard_error, 1e-12)

    observed_difference, observed_t = statistic(labels)
    rng = np.random.default_rng(random_state)
    null_t = np.empty((int(permutations), len(metric_columns)), dtype=float)
    for repeat in range(int(permutations)):
        permuted = labels.copy()
        for indices in set_indices:
            permuted[indices] = rng.permutation(permuted[indices])
        _, null_t[repeat] = statistic(permuted)
    maximum_null = np.max(np.abs(null_t), axis=1)

    bootstrap = np.empty(
        (int(bootstrap_repetitions), len(metric_columns)), dtype=float
    )
    for repeat in range(int(bootstrap_repetitions)):
        sampled_sets = rng.choice(len(set_indices), len(set_indices), replace=True)
        selected = np.concatenate([set_indices[index] for index in sampled_sets])
        sampled_labels = labels[selected]
        sampled_matrix = matrix[selected]
        bootstrap[repeat] = (
            sampled_matrix[sampled_labels == 1].mean(axis=0)
            - sampled_matrix[sampled_labels == 0].mean(axis=0)
        )
    rows = []
    for column_index, metric in enumerate(metric_columns):
        raw_p = (
            1
            + np.sum(
                np.abs(null_t[:, column_index])
                >= abs(observed_t[column_index])
            )
        ) / (len(null_t) + 1)
        adjusted_p = (
            1 + np.sum(maximum_null >= abs(observed_t[column_index]))
        ) / (len(maximum_null) + 1)
        rows.append(
            {
                "metric": metric,
                "group_b_minus_group_a": float(observed_difference[column_index]),
                "bootstrap_95_ci_low": float(
                    np.quantile(bootstrap[:, column_index], 0.025)
                ),
                "bootstrap_95_ci_high": float(
                    np.quantile(bootstrap[:, column_index], 0.975)
                ),
                "permutation_p_value": float(raw_p),
                "max_t_adjusted_p_value": float(adjusted_p),
                "matched_sets": int(len(set_indices)),
            }
        )
    return pd.DataFrame(rows)
