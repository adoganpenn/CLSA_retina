"""Participant-level glaucoma classification and spatial validation helpers.

The primary classifier is deliberately linear on frozen RETFound embeddings so
its logit can be decomposed exactly into patch contributions. All model
selection and evaluation are performed at participant level.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence


def require_columns(frame: Any, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def match_controls_ratio(
    cases: Any,
    controls: Any,
    *,
    ratio: int = 2,
    caliper_years: float = 1.0,
    exact_columns: Sequence[str] = ("sex_normalized", "visit"),
    id_column: str = "participant_id",
    age_column: str = "age",
) -> Any:
    """Deterministically match up to ``ratio`` controls per case without reuse."""
    import pandas as pd

    if ratio < 1:
        raise ValueError("ratio must be at least 1")
    if caliper_years < 0:
        raise ValueError("caliper_years cannot be negative")
    require_columns(cases, [id_column, age_column], "Cases")
    require_columns(controls, [id_column, age_column], "Controls")
    case_frame = cases.copy().reset_index(drop=True)
    control_frame = controls.copy().reset_index(drop=True)
    case_frame[age_column] = pd.to_numeric(case_frame[age_column], errors="coerce")
    control_frame[age_column] = pd.to_numeric(
        control_frame[age_column], errors="coerce"
    )
    case_frame = case_frame.dropna(subset=[age_column])
    control_frame = control_frame.dropna(subset=[age_column])
    if case_frame[id_column].astype(str).duplicated().any():
        raise ValueError("Cases must contain one row per participant")
    if control_frame[id_column].astype(str).duplicated().any():
        raise ValueError("Controls must contain one row per participant")

    available = set(control_frame.index)
    case_order = []
    for index, case in case_frame.iterrows():
        candidates = control_frame.copy()
        for column in exact_columns:
            if column in case_frame.columns and column in control_frame.columns:
                candidates = candidates.loc[
                    candidates[column].astype(str) == str(case[column])
                ]
        candidates = candidates.loc[
            (candidates[age_column] - float(case[age_column])).abs()
            <= caliper_years
        ]
        case_order.append((len(candidates), float(case[age_column]), str(case[id_column]), index))

    rows = []
    for _, _, _, case_index in sorted(case_order):
        case = case_frame.loc[case_index]
        candidates = control_frame.loc[sorted(available)].copy()
        for column in exact_columns:
            if column in case_frame.columns and column in control_frame.columns:
                candidates = candidates.loc[
                    candidates[column].astype(str) == str(case[column])
                ].copy()
        candidates["_age_distance"] = (
            candidates[age_column] - float(case[age_column])
        ).abs()
        candidates = candidates.loc[
            candidates["_age_distance"] <= caliper_years
        ].sort_values(["_age_distance", id_column], kind="stable")
        for match_rank, (control_index, control) in enumerate(
            candidates.head(ratio).iterrows(), start=1
        ):
            available.remove(control_index)
            rows.append(
                {
                    "match_set_id": f"set_{case_index:07d}",
                    "case_id": str(case[id_column]),
                    "control_id": str(control[id_column]),
                    "control_rank": match_rank,
                    "case_age": float(case[age_column]),
                    "control_age": float(control[age_column]),
                    "absolute_age_difference": float(control["_age_distance"]),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "match_set_id",
            "case_id",
            "control_id",
            "control_rank",
            "case_age",
            "control_age",
            "absolute_age_difference",
        ],
    )


def _validate_classifier_frame(
    frame: Any,
    *,
    expected_dim: int,
    embedding_column: str,
    label_column: str,
    id_column: str,
) -> tuple[Any, Any, Any]:
    import numpy as np
    import pandas as pd

    require_columns(
        frame,
        [id_column, label_column, embedding_column],
        "Classifier frame",
    )
    work = frame.copy().reset_index(drop=True)
    work[id_column] = work[id_column].astype(str)
    work[label_column] = pd.to_numeric(work[label_column], errors="coerce")
    if work[id_column].duplicated().any():
        raise ValueError("Classifier frame must contain one row per participant")
    if work[label_column].isna().any() or set(work[label_column].astype(int)) != {0, 1}:
        raise ValueError("Classifier label must contain both binary classes")
    matrix = []
    for vector in work[embedding_column]:
        array = np.asarray(vector, dtype=np.float64).reshape(-1)
        if array.size != expected_dim or not np.isfinite(array).all():
            raise ValueError("Classifier contains an invalid embedding")
        matrix.append(array)
    return work, np.stack(matrix), work[label_column].astype(int).to_numpy()


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
        random_state=20260811,
    ).fit(scaled, label)
    scaled_coefficients = estimator.coef_.reshape(-1).astype(np.float64)
    coefficients = scaled_coefficients / np.maximum(scaler.scale_, 1e-12)
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


def predict_linear_head(frame: Any, head: Mapping[str, Any], *, embedding_column: str = "embedding") -> Any:
    """Return logits and probabilities from an effective linear head."""
    import numpy as np
    import pandas as pd

    coefficients = np.asarray(head["coefficients"], dtype=np.float64).reshape(-1)
    matrix = np.stack(
        [np.asarray(value, dtype=np.float64).reshape(-1) for value in frame[embedding_column]]
    )
    if matrix.shape[1] != coefficients.size:
        raise ValueError("Embedding and classifier dimensions do not match")
    logit = matrix @ coefficients + float(head["intercept"])
    probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -40, 40)))
    return pd.DataFrame(
        {
            "classifier_logit": logit,
            "glaucoma_probability": probability,
        },
        index=frame.index,
    )


def fit_oof_linear_classifier(
    frame: Any,
    *,
    folds: int = 5,
    inner_folds: int = 4,
    c_grid: Sequence[float] = (0.001, 0.01, 0.1, 1.0),
    expected_dim: int = 1024,
    embedding_column: str = "embedding",
    label_column: str = "glaucoma_label",
    id_column: str = "participant_id",
    random_state: int = 20260811,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    """Nested participant-level CV for a linear RETFound glaucoma head."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    if folds < 3 or inner_folds < 2:
        raise ValueError("Use at least 3 outer and 2 inner folds")
    if not c_grid or any(float(value) <= 0 for value in c_grid):
        raise ValueError("c_grid must contain positive values")
    work, matrix, label = _validate_classifier_frame(
        frame,
        expected_dim=expected_dim,
        embedding_column=embedding_column,
        label_column=label_column,
        id_column=id_column,
    )
    if int(np.bincount(label).min()) < folds:
        raise ValueError("Each class must support every outer fold")
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    oof_logit = np.full(len(work), np.nan, dtype=float)
    oof_probability = np.full(len(work), np.nan, dtype=float)
    fold_assignment = np.full(len(work), -1, dtype=int)
    heads = []
    for fold, (development, evaluation) in enumerate(outer.split(matrix, label)):
        development_label = label[development]
        usable_inner_folds = min(inner_folds, int(np.bincount(development_label).min()))
        inner = StratifiedKFold(
            n_splits=usable_inner_folds,
            shuffle=True,
            random_state=random_state + fold + 1,
        )
        scores = []
        for c_value in sorted(float(value) for value in c_grid):
            aucs = []
            for inner_train, inner_test in inner.split(
                matrix[development], development_label
            ):
                head = _fit_scaled_logistic(
                    matrix[development][inner_train],
                    development_label[inner_train],
                    c_value,
                )
                logit = (
                    matrix[development][inner_test] @ head["coefficients"]
                    + head["intercept"]
                )
                aucs.append(
                    roc_auc_score(development_label[inner_test], logit)
                )
            scores.append((float(np.mean(aucs)), c_value))
        # Prefer the smaller C when mean AUC ties, reducing model complexity.
        best_auc = max(value[0] for value in scores)
        best_c = min(value[1] for value in scores if np.isclose(value[0], best_auc))
        head = _fit_scaled_logistic(
            matrix[development],
            development_label,
            best_c,
        )
        head.update(
            {
                "fold": fold,
                "inner_mean_auc": best_auc,
                "n_development_participants": int(len(development)),
                "n_evaluation_participants": int(len(evaluation)),
            }
        )
        logit = matrix[evaluation] @ head["coefficients"] + head["intercept"]
        probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -40, 40)))
        oof_logit[evaluation] = logit
        oof_probability[evaluation] = probability
        fold_assignment[evaluation] = fold
        heads.append(head)
    if np.isnan(oof_probability).any() or np.any(fold_assignment < 0):
        raise RuntimeError("OOF classifier did not score every participant")
    output = work[[id_column, label_column]].copy()
    output["fold"] = fold_assignment
    output["classifier_logit_oof"] = oof_logit
    output["glaucoma_probability_oof"] = oof_probability
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
            "model_name": "CLSA_RETFound_glaucoma_linear",
            "frozen": True,
            "selected_from_outer_folds": True,
            "outer_folds": folds,
            "participant_level_training": True,
        }
    )
    return output, heads, final_head


def classifier_metrics(
    label: Any,
    probability: Any,
    *,
    bootstrap_repetitions: int = 2000,
    threshold: float = 0.5,
    random_state: int = 20260811,
) -> dict[str, Any]:
    """Participant-level discrimination, calibration, and stratified bootstrap CIs."""
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
    prediction = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(label, prediction, labels=[0, 1]).ravel()
    rng = np.random.default_rng(random_state)
    class_indices = [np.flatnonzero(label == value) for value in (0, 1)]
    aucs = np.empty(bootstrap_repetitions, dtype=float)
    average_precisions = np.empty(bootstrap_repetitions, dtype=float)
    for repeat in range(bootstrap_repetitions):
        sampled = np.concatenate(
            [rng.choice(index, len(index), replace=True) for index in class_indices]
        )
        aucs[repeat] = roc_auc_score(label[sampled], probability[sampled])
        average_precisions[repeat] = average_precision_score(
            label[sampled], probability[sampled]
        )
    return {
        "n_participants": int(len(label)),
        "n_glaucoma": int(label.sum()),
        "n_controls": int((label == 0).sum()),
        "auroc": float(roc_auc_score(label, probability)),
        "auroc_95_ci_low": float(np.quantile(aucs, 0.025)),
        "auroc_95_ci_high": float(np.quantile(aucs, 0.975)),
        "average_precision": float(average_precision_score(label, probability)),
        "average_precision_95_ci_low": float(
            np.quantile(average_precisions, 0.025)
        ),
        "average_precision_95_ci_high": float(
            np.quantile(average_precisions, 0.975)
        ),
        "brier_score": float(brier_score_loss(label, probability)),
        "threshold": float(threshold),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "positive_predictive_value": float(tp / max(tp + fp, 1)),
        "negative_predictive_value": float(tn / max(tn + fn, 1)),
    }


def paired_auc_difference(
    reference: Any,
    comparator: Any,
    *,
    reference_probability: str,
    comparator_probability: str,
    id_column: str = "participant_id",
    label_column: str = "glaucoma_label",
    bootstrap_repetitions: int = 2000,
    random_state: int = 20260811,
) -> dict[str, Any]:
    """Paired stratified-bootstrap AUROC difference on identical participants."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    require_columns(
        reference,
        [id_column, label_column, reference_probability],
        "Reference predictions",
    )
    require_columns(
        comparator,
        [id_column, label_column, comparator_probability],
        "Comparator predictions",
    )
    left = reference[
        [id_column, label_column, reference_probability]
    ].copy().rename(columns={reference_probability: "_reference_score"})
    right = comparator[
        [id_column, label_column, comparator_probability]
    ].copy().rename(columns={comparator_probability: "_comparator_score"})
    left[id_column] = left[id_column].astype(str)
    right[id_column] = right[id_column].astype(str)
    merged = left.merge(
        right,
        on=[id_column, label_column],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("Paired AUROC comparison requires identical participants")
    label = merged[label_column].astype(int).to_numpy()
    reference_score = merged["_reference_score"].astype(float).to_numpy()
    comparator_score = merged["_comparator_score"].astype(float).to_numpy()
    if set(label) != {0, 1}:
        raise ValueError("Paired AUROC comparison requires both classes")
    rng = np.random.default_rng(random_state)
    class_indices = [np.flatnonzero(label == value) for value in (0, 1)]
    differences = np.empty(bootstrap_repetitions, dtype=float)
    for repeat in range(bootstrap_repetitions):
        sampled = np.concatenate(
            [rng.choice(index, len(index), replace=True) for index in class_indices]
        )
        differences[repeat] = roc_auc_score(
            label[sampled], reference_score[sampled]
        ) - roc_auc_score(label[sampled], comparator_score[sampled])
    observed = roc_auc_score(label, reference_score) - roc_auc_score(
        label, comparator_score
    )
    return {
        "n_participants": int(len(merged)),
        "auroc_difference": float(observed),
        "auroc_difference_95_ci_low": float(np.quantile(differences, 0.025)),
        "auroc_difference_95_ci_high": float(np.quantile(differences, 0.975)),
    }


def optic_nerve_masks(
    shape: tuple[int, int],
    *,
    center_x_fraction: float,
    center_y_fraction: float,
    radius_fraction: float,
    peripapillary_multiplier: float = 2.0,
) -> dict[str, Any]:
    """Create optic-disc and peripapillary masks from independent coordinates."""
    import numpy as np

    height, width = shape
    values = (center_x_fraction, center_y_fraction, radius_fraction)
    if not all(np.isfinite(values)):
        raise ValueError("Optic-nerve coordinates must be finite")
    if not 0 <= center_x_fraction <= 1 or not 0 <= center_y_fraction <= 1:
        raise ValueError("Optic-nerve center fractions must lie in [0, 1]")
    if not 0 < radius_fraction <= 0.25:
        raise ValueError("Optic-disc radius fraction must lie in (0, 0.25]")
    yy, xx = np.indices(shape)
    center_x = center_x_fraction * (width - 1)
    center_y = center_y_fraction * (height - 1)
    radius = radius_fraction * min(height, width)
    distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    disc = distance <= radius
    peripapillary_total = distance <= peripapillary_multiplier * radius
    return {
        "optic_disc": disc,
        "peripapillary_annulus": peripapillary_total & ~disc,
        "optic_disc_plus_peripapillary": peripapillary_total,
        "disc_center_x_px": float(center_x),
        "disc_center_y_px": float(center_y),
        "disc_radius_px": float(radius),
    }


def regional_attribution_metrics(
    attribution_grid: Any,
    retina_mask: Any,
    region_masks: Mapping[str, Any],
) -> dict[str, float]:
    """Measure positive/absolute glaucoma evidence and enrichment by anatomy."""
    import numpy as np
    from scipy import ndimage

    grid = np.asarray(attribution_grid, dtype=float)
    retina = np.asarray(retina_mask, dtype=bool)
    resized = ndimage.zoom(
        grid,
        (retina.shape[0] / grid.shape[0], retina.shape[1] / grid.shape[1]),
        order=1,
    )[: retina.shape[0], : retina.shape[1]]
    if resized.shape != retina.shape or not retina.any():
        raise ValueError("Attribution and retinal mask could not be aligned")
    positive = np.maximum(resized, 0) * retina
    absolute = np.abs(resized) * retina
    positive_total = float(positive.sum())
    absolute_total = float(absolute.sum())
    result: dict[str, float] = {}
    for name, raw_mask in region_masks.items():
        if name.endswith("_px"):
            continue
        mask = np.asarray(raw_mask, dtype=bool) & retina
        area_fraction = float(mask.sum() / retina.sum())
        positive_fraction = (
            float(positive[mask].sum() / positive_total)
            if positive_total > 0
            else 0.0
        )
        absolute_fraction = (
            float(absolute[mask].sum() / absolute_total)
            if absolute_total > 0
            else 0.0
        )
        result[f"{name}_area_fraction"] = area_fraction
        result[f"{name}_positive_mass_fraction"] = positive_fraction
        result[f"{name}_absolute_mass_fraction"] = absolute_fraction
        result[f"{name}_positive_enrichment"] = (
            positive_fraction / area_fraction if area_fraction > 0 else math.nan
        )
        result[f"{name}_absolute_enrichment"] = (
            absolute_fraction / area_fraction if area_fraction > 0 else math.nan
        )
        result[f"{name}_signed_mean"] = (
            float(resized[mask].mean()) if mask.any() else math.nan
        )
    peak_y, peak_x = np.unravel_index(np.argmax(positive), positive.shape)
    center_x = float(region_masks["disc_center_x_px"])
    center_y = float(region_masks["disc_center_y_px"])
    radius = max(float(region_masks["disc_radius_px"]), 1e-6)
    result["positive_peak_distance_from_disc_radii"] = float(
        np.sqrt((peak_x - center_x) ** 2 + (peak_y - center_y) ** 2) / radius
    )
    result["positive_peak_inside_disc"] = float(
        np.asarray(region_masks["optic_disc"], dtype=bool)[peak_y, peak_x]
    )
    return result


def sample_equal_area_control_masks(
    retina_mask: Any,
    excluded_mask: Any,
    *,
    target_area: int,
    n_masks: int = 20,
    random_state: int = 20260811,
) -> list[Any]:
    """Sample circular retinal control regions approximating a target area."""
    import numpy as np

    retina = np.asarray(retina_mask, dtype=bool)
    excluded = np.asarray(excluded_mask, dtype=bool)
    if retina.shape != excluded.shape or target_area < 1:
        raise ValueError("Invalid equal-area mask inputs")
    radius = math.sqrt(target_area / math.pi)
    yy, xx = np.indices(retina.shape)
    eligible = np.argwhere(retina & ~excluded)
    rng = np.random.default_rng(random_state)
    masks = []
    if not len(eligible):
        return masks
    order = rng.permutation(len(eligible))
    for position in order:
        center_y, center_x = eligible[position]
        mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
        mask &= retina & ~excluded
        if mask.sum() >= 0.80 * target_area:
            masks.append(mask)
        if len(masks) >= n_masks:
            break
    return masks
