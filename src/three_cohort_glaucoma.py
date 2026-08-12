"""Participant-level helpers for the CLSA/Zeiss glaucoma triangulation study.

The same-camera CLSA glaucoma-versus-healthy comparison is the primary disease
analysis. Cross-source harmonization is deliberately implemented as a
sensitivity analysis under an additive source-effect assumption; it cannot
identify a source-by-disease interaction without healthy Zeiss controls.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence


def require_columns(frame: Any, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def canonical_sex(value: Any) -> str:
    """Map common released/string sex codes without inventing missing values."""
    import pandas as pd

    if pd.isna(value):
        return "MISSING"
    text = str(value).strip().upper()
    if text in {"F", "FEMALE", "2", "2.0", "WOMAN"}:
        return "F"
    if text in {"M", "MALE", "1", "1.0", "MAN"}:
        return "M"
    if text in {"", "NAN", "NONE", "NULL", "8", "9", "-8", "-9"}:
        return "MISSING"
    return f"OTHER:{text}"


def validate_embedding_frame(
    frame: Any,
    label: str,
    expected_dim: int = 1024,
) -> Any:
    """Validate participant records and return normalized NumPy embeddings."""
    import numpy as np
    import pandas as pd

    require_columns(frame, ["participant_id", "age", "embedding"], label)
    work = frame.copy()
    work["participant_id"] = work["participant_id"].astype(str)
    work["age"] = pd.to_numeric(work["age"], errors="coerce")
    if work["participant_id"].str.strip().isin(["", "nan", "None"]).any():
        raise ValueError(f"{label} contains missing participant identifiers")
    if work["age"].isna().any() or (~work["age"].between(0, 120)).any():
        raise ValueError(f"{label} contains invalid chronological ages")
    vectors = []
    for vector in work["embedding"]:
        array = np.asarray(vector, dtype=np.float64).reshape(-1)
        if array.size != expected_dim:
            raise ValueError(
                f"{label} contains an embedding of dimension {array.size}; "
                f"expected {expected_dim}"
            )
        if not np.isfinite(array).all() or np.linalg.norm(array) <= 0:
            raise ValueError(f"{label} contains a nonfinite or zero embedding")
        vectors.append(array)
    work["embedding"] = vectors
    if "sex" in work.columns:
        normalized = work["sex"].map(canonical_sex)
        if "sex_normalized" in work.columns:
            existing = work["sex_normalized"].map(canonical_sex)
            normalized = normalized.where(normalized != "MISSING", existing)
        work["sex_normalized"] = normalized
    elif "sex_normalized" in work.columns:
        work["sex_normalized"] = work["sex_normalized"].map(canonical_sex)
    else:
        work["sex_normalized"] = "MISSING"
    return work.reset_index(drop=True)


def select_representative_visit(
    frame: Any,
    *,
    preferred_visits: Sequence[str] = ("BL", "F1"),
) -> Any:
    """Keep one deterministic participant-visit without mixing visit ages."""
    require_columns(frame, ["participant_id"], "Participant frame")
    work = frame.copy()
    if "visit" not in work.columns:
        if work["participant_id"].duplicated().any():
            raise ValueError("Participant frame has duplicate IDs but no visit column")
        return work.reset_index(drop=True)
    order = {str(value).upper(): index for index, value in enumerate(preferred_visits)}
    work["_visit_order"] = (
        work["visit"].astype(str).str.upper().map(order).fillna(len(order))
    )
    work = work.sort_values(
        ["participant_id", "_visit_order", "visit"], kind="stable"
    ).drop_duplicates("participant_id", keep="first")
    return work.drop(columns="_visit_order").reset_index(drop=True)


def aggregate_embedding_rows(
    frame: Any,
    *,
    group_columns: Sequence[str] = ("participant_id", "visit"),
    metadata_columns: Sequence[str] = (),
) -> Any:
    """Average image embeddings while retaining one auditable record per group."""
    import numpy as np
    import pandas as pd

    require_columns(frame, [*group_columns, "embedding", "age"], "Image embeddings")
    rows = []
    for keys, group in frame.groupby(list(group_columns), sort=True, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, key_values))
        matrix = np.stack(group["embedding"].map(lambda value: np.asarray(value, dtype=float)))
        row["embedding"] = matrix.mean(axis=0)
        row["age"] = float(pd.to_numeric(group["age"], errors="coerce").median())
        row["n_images"] = int(len(group))
        for column in metadata_columns:
            if column not in group.columns:
                continue
            observed = group[column].dropna()
            row[column] = observed.iloc[0] if len(observed) else None
        rows.append(row)
    return pd.DataFrame(rows)


def _categorical_columns(series: Any, prefix: str) -> tuple[Any, list[str], list[str]]:
    import numpy as np
    values = series.fillna("MISSING").astype(str)
    categories = sorted(values.unique())
    columns = []
    names = []
    for category in categories[1:]:
        columns.append((values == category).to_numpy(float))
        names.append(f"{prefix}={category}")
    matrix = np.column_stack(columns) if columns else np.empty((len(values), 0))
    return matrix, names, categories


def _regression_design(
    frame: Any,
    *,
    exposure_column: str,
    numeric_covariates: Sequence[str],
    categorical_covariates: Sequence[str],
) -> tuple[Any, list[str]]:
    import numpy as np
    import pandas as pd

    work = frame
    parts = [np.ones((len(work), 1)), pd.to_numeric(work[exposure_column]).to_numpy(float)[:, None]]
    names = ["intercept", exposure_column]
    for column in numeric_covariates:
        if column not in work.columns:
            continue
        values = pd.to_numeric(work[column], errors="coerce")
        median = float(values.median()) if values.notna().any() else 0.0
        missing = values.isna().to_numpy(float)
        filled = values.fillna(median).to_numpy(float)
        scale = float(np.std(filled))
        if scale <= 1e-12:
            continue
        centered = (filled - float(np.mean(filled))) / scale
        parts.append(centered[:, None])
        names.append(column)
        if column == "age":
            parts.append((centered**2)[:, None])
            names.append("age_squared")
        if missing.any():
            parts.append(missing[:, None])
            names.append(f"{column}_missing")
    for column in categorical_covariates:
        if column not in work.columns:
            continue
        matrix, category_names, _ = _categorical_columns(work[column], column)
        if matrix.shape[1]:
            parts.append(matrix)
            names.extend(category_names)
    return np.column_stack(parts), names


def adjusted_group_effect(
    frame: Any,
    *,
    outcome_column: str,
    exposure_column: str = "exposed",
    numeric_covariates: Sequence[str] = ("age",),
    categorical_covariates: Sequence[str] = ("sex_normalized",),
) -> dict[str, Any]:
    """Estimate an adjusted exposure contrast with HC3 robust uncertainty."""
    import numpy as np
    import pandas as pd
    from scipy import stats

    require_columns(frame, [outcome_column, exposure_column], "Regression frame")
    work = frame.copy()
    work[outcome_column] = pd.to_numeric(work[outcome_column], errors="coerce")
    work[exposure_column] = pd.to_numeric(work[exposure_column], errors="coerce")
    work = work.dropna(subset=[outcome_column, exposure_column])
    if len(work) < 20 or work[exposure_column].nunique() != 2:
        raise ValueError("Adjusted comparison requires at least 20 records and two groups")
    design, design_names = _regression_design(
        work,
        exposure_column=exposure_column,
        numeric_covariates=numeric_covariates,
        categorical_covariates=categorical_covariates,
    )
    outcome = work[outcome_column].to_numpy(float)
    inverse = np.linalg.pinv(design.T @ design)
    coefficients = inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ inverse) * design, axis=1)
    scaled = residuals / np.maximum(1 - leverage, 1e-8)
    covariance = inverse @ (design.T @ ((scaled**2)[:, None] * design)) @ inverse
    standard_error = float(np.sqrt(max(covariance[1, 1], 0)))
    estimate = float(coefficients[1])
    z_score = estimate / standard_error if standard_error > 0 else math.nan
    p_value = float(2 * stats.norm.sf(abs(z_score))) if np.isfinite(z_score) else math.nan
    return {
        "n": int(len(work)),
        "n_exposed": int((work[exposure_column] == 1).sum()),
        "n_reference": int((work[exposure_column] == 0).sum()),
        "adjusted_difference": estimate,
        "hc3_standard_error": standard_error,
        "ci_95_low": estimate - 1.96 * standard_error,
        "ci_95_high": estimate + 1.96 * standard_error,
        "p_value": p_value,
        "design_columns": design_names,
    }


def greedy_match(
    exposed: Any,
    reference: Any,
    *,
    caliper_years: float,
    exact_columns: Sequence[str] = ("sex_normalized",),
    id_column: str = "participant_id",
    age_column: str = "age",
) -> Any:
    """Deterministic no-replacement 1:1 matching with privacy-safe pair IDs."""
    import pandas as pd

    require_columns(exposed, [id_column, age_column], "Exposed group")
    require_columns(reference, [id_column, age_column], "Reference group")
    if caliper_years < 0:
        raise ValueError("caliper_years cannot be negative")
    cases = exposed.copy().reset_index(drop=True)
    controls = reference.copy().reset_index(drop=True)
    cases[age_column] = pd.to_numeric(cases[age_column], errors="coerce")
    controls[age_column] = pd.to_numeric(controls[age_column], errors="coerce")
    cases = cases.dropna(subset=[age_column])
    controls = controls.dropna(subset=[age_column])
    if cases[id_column].duplicated().any() or controls[id_column].duplicated().any():
        raise ValueError("Matching inputs must contain one row per participant")
    available = set(controls.index)
    rows = []
    for _, case in cases.sort_values([age_column, id_column], kind="stable").iterrows():
        candidates = controls.loc[sorted(available)].copy()
        for column in exact_columns:
            if column in cases.columns and column in controls.columns:
                candidates = candidates.loc[
                    candidates[column].astype(str) == str(case[column])
                ].copy()
        if candidates.empty:
            continue
        candidates["_distance"] = (candidates[age_column] - case[age_column]).abs()
        candidates = candidates.sort_values(["_distance", id_column], kind="stable")
        control = candidates.iloc[0]
        if float(control["_distance"]) > caliper_years:
            continue
        available.remove(control.name)
        rows.append(
            {
                "pair_id": f"pair_{len(rows):07d}",
                "exposed_id": str(case[id_column]),
                "reference_id": str(control[id_column]),
                "exposed_age": float(case[age_column]),
                "reference_age": float(control[age_column]),
                "absolute_age_difference": float(control["_distance"]),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "pair_id",
            "exposed_id",
            "reference_id",
            "exposed_age",
            "reference_age",
            "absolute_age_difference",
        ],
    )


def paired_outcome_effect(
    pairs: Any,
    exposed: Any,
    reference: Any,
    *,
    outcome_column: str,
    bootstrap_repetitions: int = 5000,
    random_state: int = 20260811,
) -> tuple[dict[str, Any], Any]:
    """Calculate exposed-minus-reference differences and participant bootstrap CI."""
    import numpy as np

    if pairs.empty:
        raise ValueError("No matched pairs are available")
    exposed_values = exposed.set_index("participant_id")[outcome_column]
    reference_values = reference.set_index("participant_id")[outcome_column]
    analysis = pairs.copy()
    analysis["exposed_outcome"] = analysis["exposed_id"].map(exposed_values)
    analysis["reference_outcome"] = analysis["reference_id"].map(reference_values)
    analysis = analysis.dropna(subset=["exposed_outcome", "reference_outcome"])
    analysis["paired_difference"] = (
        analysis["exposed_outcome"] - analysis["reference_outcome"]
    )
    values = analysis["paired_difference"].to_numpy(float)
    if not len(values):
        raise ValueError("Matched pairs have no complete outcomes")
    rng = np.random.default_rng(random_state)
    means = np.empty(bootstrap_repetitions, dtype=float)
    for index in range(bootstrap_repetitions):
        means[index] = np.mean(rng.choice(values, size=len(values), replace=True))
    return (
        {
            "n_pairs": int(len(values)),
            "mean_difference": float(np.mean(values)),
            "median_difference": float(np.median(values)),
            "bootstrap_95_ci_low": float(np.quantile(means, 0.025)),
            "bootstrap_95_ci_high": float(np.quantile(means, 0.975)),
            "mean_absolute_age_difference": float(
                analysis["absolute_age_difference"].mean()
            ),
        },
        analysis,
    )


def _harmonization_design(
    frame: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Any, list[str], dict[str, Any]]:
    import numpy as np
    import pandas as pd

    require_columns(frame, ["age", "glaucoma", "source", "sex_normalized"], "Harmonization frame")
    work = frame.copy()
    age = pd.to_numeric(work["age"], errors="coerce")
    if age.isna().any():
        raise ValueError("Harmonization frame contains missing age")
    if metadata is None:
        age_center = float(age.mean())
        age_scale = float(age.std(ddof=0)) or 1.0
        sex_by_source = [
            set(group["sex_normalized"].astype(str))
            for _, group in work.groupby("source")
        ]
        shared_sex = set.intersection(*sex_by_source) if sex_by_source else set()
        # A sex term is estimable across source only when at least two categories
        # occur in every source. Otherwise it is omitted and the notebook reports
        # an age-only cross-source adjustment.
        sex_categories = sorted(shared_sex) if len(shared_sex) >= 2 else []
    else:
        age_center = float(metadata["age_center"])
        age_scale = float(metadata["age_scale"])
        sex_categories = list(metadata["sex_categories"])
    age_z = (age.to_numpy(float) - age_center) / age_scale
    parts = [
        np.ones((len(work), 1)),
        age_z[:, None],
        (age_z**2)[:, None],
        pd.to_numeric(work["glaucoma"]).to_numpy(float)[:, None],
        (work["source"].astype(str) == "Zeiss").to_numpy(float)[:, None],
    ]
    names = ["intercept", "age", "age_squared", "glaucoma", "source_zeiss"]
    sex = work["sex_normalized"].astype(str)
    for category in sex_categories[1:]:
        parts.append((sex == category).to_numpy(float)[:, None])
        names.append(f"sex={category}")
    return (
        np.column_stack(parts),
        names,
        {
            "age_center": age_center,
            "age_scale": age_scale,
            "sex_categories": sex_categories,
        },
    )


def fit_additive_source_harmonizer(
    frame: Any,
    *,
    ridge: float = 1e-6,
    expected_dim: int = 1024,
) -> dict[str, Any]:
    """Fit a balanced, ComBat-like additive location/scale source correction.

    The triangular design must contain CLSA healthy, CLSA glaucoma, and Zeiss
    glaucoma records. The source-by-glaucoma interaction is intentionally not
    estimable and is reported as a limitation by the notebook.
    """
    import numpy as np

    work = validate_embedding_frame(frame, "Three-cohort harmonization", expected_dim)
    required_cells = {("CLSA", 0), ("CLSA", 1), ("Zeiss", 1)}
    observed_cells = set(
        zip(work["source"].astype(str), work["glaucoma"].astype(int))
    )
    missing_cells = required_cells - observed_cells
    if missing_cells:
        raise ValueError(f"Harmonization design is missing cells: {sorted(missing_cells)}")
    design, names, metadata = _harmonization_design(work)
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("Harmonization design matrix is rank deficient")
    matrix = np.stack(work["embedding"].to_numpy()).astype(np.float64)
    cell = work["source"].astype(str) + "|" + work["glaucoma"].astype(str)
    cell_counts = cell.value_counts()
    weights = cell.map(lambda value: 1.0 / float(cell_counts[value])).to_numpy(float)
    weights *= len(weights) / weights.sum()
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_matrix = matrix * np.sqrt(weights)[:, None]
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_matrix,
    )
    fitted = design @ coefficients
    residual = matrix - fitted
    source_index = names.index("source_zeiss")
    residual_scale = {}
    for source in ("CLSA", "Zeiss"):
        selected = work["source"].astype(str).to_numpy() == source
        scale = np.std(residual[selected], axis=0, ddof=1)
        residual_scale[source] = np.maximum(scale, 1e-6)
    return {
        "method": "balanced_additive_location_scale",
        "expected_dim": expected_dim,
        "design_columns": names,
        "design_metadata": metadata,
        "coefficients": coefficients,
        "source_shift": coefficients[source_index],
        "residual_scale_clsa": residual_scale["CLSA"],
        "residual_scale_zeiss": residual_scale["Zeiss"],
        "ridge": ridge,
        "assumption": "additive source effect; source-by-glaucoma interaction not identifiable",
    }


def apply_source_harmonizer(
    frame: Any,
    bundle: Mapping[str, Any],
    *,
    mode: str = "location_scale",
) -> Any:
    """Apply the fitted source correction, leaving CLSA vectors unchanged."""
    import numpy as np

    if mode not in {"location", "location_scale"}:
        raise ValueError("mode must be location or location_scale")
    work = validate_embedding_frame(
        frame, "Harmonization application", int(bundle["expected_dim"])
    )
    design, names, _ = _harmonization_design(
        work, metadata=bundle["design_metadata"]
    )
    if names != list(bundle["design_columns"]):
        raise ValueError("Harmonization design columns changed during application")
    matrix = np.stack(work["embedding"].to_numpy()).astype(np.float64)
    zeiss = (work["source"].astype(str) == "Zeiss").to_numpy()
    corrected = matrix.copy()
    source_shift = np.asarray(bundle["source_shift"], dtype=float)
    if mode == "location":
        corrected[zeiss] -= source_shift
    else:
        coefficients = np.asarray(bundle["coefficients"], dtype=float)
        full_fitted = design @ coefficients
        reference_design = design.copy()
        reference_design[:, names.index("source_zeiss")] = 0.0
        biological_fitted = reference_design @ coefficients
        residual = matrix - full_fitted
        ratio = np.asarray(bundle["residual_scale_clsa"]) / np.asarray(
            bundle["residual_scale_zeiss"]
        )
        corrected[zeiss] = biological_fitted[zeiss] + residual[zeiss] * ratio
    output = work.copy()
    output["embedding"] = [row.astype(np.float32) for row in corrected]
    output["harmonization_mode"] = mode
    return output


def crossfit_source_harmonizer(
    frame: Any,
    *,
    folds: int = 5,
    mode: str = "location_scale",
    ridge: float = 1e-6,
    expected_dim: int = 1024,
    random_state: int = 20260811,
) -> tuple[Any, list[dict[str, Any]]]:
    """Transform every participant with a harmonizer that did not see them.

    Folds are stratified across the three observed source-by-glaucoma cells.
    This mirrors held-out cross-device evaluation: each affine correction is
    learned on development participants and applied only to evaluation
    participants. CLSA embeddings remain unchanged by the underlying
    harmonizer, while every Zeiss embedding is genuinely out of fit.
    """
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    if folds < 2:
        raise ValueError("folds must be at least 2")
    work = validate_embedding_frame(frame, "Cross-fit harmonization", expected_dim)
    work = work.reset_index(drop=True).copy()
    cell = work["source"].astype(str) + "|" + work["glaucoma"].astype(str)
    minimum_cell = int(cell.value_counts().min())
    if minimum_cell < folds:
        raise ValueError(
            f"Cross-fit harmonization requires at least {folds} records in "
            f"every source-by-glaucoma cell; smallest cell has {minimum_cell}."
        )
    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )
    corrected: list[Any | None] = [None] * len(work)
    fold_assignment = np.full(len(work), -1, dtype=int)
    bundles: list[dict[str, Any]] = []
    placeholder = np.zeros(len(work), dtype=np.int8)
    for fold, (development, evaluation) in enumerate(
        splitter.split(placeholder, cell.to_numpy())
    ):
        bundle = fit_additive_source_harmonizer(
            work.iloc[development].copy(),
            ridge=ridge,
            expected_dim=expected_dim,
        )
        transformed = apply_source_harmonizer(
            work.iloc[evaluation].copy(),
            bundle,
            mode=mode,
        )
        for position, embedding in zip(
            evaluation,
            transformed["embedding"].to_numpy(),
        ):
            corrected[int(position)] = embedding
            fold_assignment[int(position)] = fold
        bundles.append(bundle)
    if any(value is None for value in corrected) or np.any(fold_assignment < 0):
        raise RuntimeError("Cross-fit harmonization did not transform every record")
    output = work.copy()
    output["embedding"] = corrected
    output["harmonization_fold"] = fold_assignment
    output["harmonization_cross_fitted"] = True
    output["harmonization_mode"] = mode
    return output, bundles


def embedding_shift_summary(reference: Any, target: Any) -> dict[str, Any]:
    """Summarize featurewise standardized differences between two domains."""
    import numpy as np

    reference_matrix = np.stack(reference["embedding"].to_numpy()).astype(float)
    target_matrix = np.stack(target["embedding"].to_numpy()).astype(float)
    pooled = np.sqrt(
        (reference_matrix.var(axis=0, ddof=1) + target_matrix.var(axis=0, ddof=1)) / 2
    )
    smd = np.divide(
        target_matrix.mean(axis=0) - reference_matrix.mean(axis=0),
        pooled,
        out=np.zeros(reference_matrix.shape[1]),
        where=pooled > 1e-12,
    )
    absolute = np.abs(smd)
    return {
        "median_absolute_feature_smd": float(np.median(absolute)),
        "p90_absolute_feature_smd": float(np.quantile(absolute, 0.90)),
        "features_absolute_smd_gt_0_1": int(np.sum(absolute > 0.1)),
        "features_absolute_smd_gt_0_2": int(np.sum(absolute > 0.2)),
        "features_absolute_smd_gt_0_5": int(np.sum(absolute > 0.5)),
    }


def cross_validated_domain_auc(
    reference: Any,
    target: Any,
    *,
    max_per_domain: int = 5000,
    folds: int = 5,
    random_state: int = 20260811,
) -> dict[str, float]:
    """Measure residual linear source information on held-out participants."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(random_state)

    def sample(frame: Any) -> Any:
        if len(frame) <= max_per_domain:
            return frame
        return frame.iloc[rng.choice(len(frame), max_per_domain, replace=False)]

    reference = sample(reference)
    target = sample(target)
    matrix = np.vstack(
        [np.stack(reference["embedding"]), np.stack(target["embedding"])]
    ).astype(np.float32)
    label = np.r_[np.zeros(len(reference)), np.ones(len(target))]
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    aucs = []
    for train, test in splitter.split(matrix, label):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                penalty="l2",
                solver="liblinear",
                max_iter=2000,
                random_state=random_state,
            ),
        )
        model.fit(matrix[train], label[train])
        probability = model.predict_proba(matrix[test])[:, 1]
        aucs.append(roc_auc_score(label[test], probability))
    signed_mean = float(np.mean(aucs))
    # A source classifier with AUC 0.34 retains exactly as much directional
    # information as one with AUC 0.66 after reversing its labels. Reporting
    # max(AUC, 1-AUC) prevents inverse separation from being called success.
    effective_mean = max(signed_mean, 1.0 - signed_mean)
    return {
        "domain_auc_mean": effective_mean,
        "domain_auc_signed_mean": signed_mean,
        "domain_auc_sd": float(np.std(aucs, ddof=1)),
        "n_reference": int(len(reference)),
        "n_target": int(len(target)),
    }


def nested_harmonization_domain_auc(
    frame: Any,
    *,
    mode: str,
    folds: int = 5,
    ridge: float = 1e-6,
    expected_dim: int = 1024,
    max_per_source: int = 5000,
    random_state: int = 20260811,
) -> dict[str, Any]:
    """Evaluate residual source information on untouched outer-fold records.

    The harmonizer and source classifier are both fitted on development
    participants. AUC is calculated only on evaluation participants. This is
    stricter than fitting a correction once and then cross-validating a source
    classifier over the already-corrected complete dataset.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    work = validate_embedding_frame(frame, "Nested domain evaluation", expected_dim)
    work = work.reset_index(drop=True).copy()
    cell = work["source"].astype(str) + "|" + work["glaucoma"].astype(str)
    if int(cell.value_counts().min()) < folds:
        raise ValueError("Every source-by-glaucoma cell must support all outer folds")
    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )
    rng = np.random.default_rng(random_state)
    signed_aucs = []
    fold_rows = []

    def balanced_glaucoma_sample(frame: Any) -> Any:
        glaucoma = frame[frame["glaucoma"].astype(int) == 1].copy()
        groups = []
        counts = glaucoma["source"].astype(str).value_counts()
        if set(counts.index) != {"CLSA", "Zeiss"}:
            raise ValueError("Each domain fold requires CLSA and Zeiss glaucoma")
        sample_size = min(int(counts.min()), int(max_per_source))
        for source in ("CLSA", "Zeiss"):
            group = glaucoma[glaucoma["source"].astype(str) == source]
            if len(group) > sample_size:
                positions = rng.choice(len(group), sample_size, replace=False)
                group = group.iloc[positions]
            groups.append(group)
        import pandas as pd

        return pd.concat(groups, ignore_index=True)

    placeholder = np.zeros(len(work), dtype=np.int8)
    for fold, (development, evaluation) in enumerate(
        splitter.split(placeholder, cell.to_numpy())
    ):
        development_frame = work.iloc[development].copy()
        evaluation_frame = work.iloc[evaluation].copy()
        bundle = fit_additive_source_harmonizer(
            development_frame,
            ridge=ridge,
            expected_dim=expected_dim,
        )
        development_corrected = apply_source_harmonizer(
            development_frame,
            bundle,
            mode=mode,
        )
        evaluation_corrected = apply_source_harmonizer(
            evaluation_frame,
            bundle,
            mode=mode,
        )
        train = balanced_glaucoma_sample(development_corrected)
        test = balanced_glaucoma_sample(evaluation_corrected)
        x_train = np.stack(train["embedding"].to_numpy()).astype(np.float32)
        x_test = np.stack(test["embedding"].to_numpy()).astype(np.float32)
        y_train = (train["source"].astype(str) == "Zeiss").to_numpy(int)
        y_test = (test["source"].astype(str) == "Zeiss").to_numpy(int)
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                penalty="l2",
                solver="liblinear",
                max_iter=2000,
                random_state=random_state + fold,
            ),
        )
        classifier.fit(x_train, y_train)
        probability = classifier.predict_proba(x_test)[:, 1]
        auc = float(roc_auc_score(y_test, probability))
        signed_aucs.append(auc)
        fold_rows.append(
            {
                "fold": fold,
                "signed_auc": auc,
                "effective_auc": max(auc, 1.0 - auc),
                "n_development": int(len(train)),
                "n_evaluation": int(len(test)),
            }
        )
    signed_mean = float(np.mean(signed_aucs))
    return {
        "domain_auc_mean": max(signed_mean, 1.0 - signed_mean),
        "domain_auc_signed_mean": signed_mean,
        "domain_auc_sd": float(np.std(signed_aucs, ddof=1)),
        "fold_results": fold_rows,
        "evaluation_design": "outer-fold harmonizer and classifier evaluation",
    }
