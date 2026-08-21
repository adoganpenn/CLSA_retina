"""Participant-level fairness utilities for RETFound retinal-age analyses.

The functions in this module are independent of Databricks and Spark so the
matching and estimands can be unit tested locally.  They deliberately evaluate
out-of-fold predictions at one record per participant; two eyes must never be
treated as two independent people.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def _require_columns(frame: Any, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def binary_indicator(value: Any) -> bool | None:
    """Decode common CLSA yes/no releases without treating sentinels as no."""
    import pandas as pd

    if value is None or pd.isna(value):
        return None
    text = str(value).strip().upper()
    if text in {"1", "1.0", "11", "Y", "YES", "TRUE", "T"}:
        return True
    if text in {"0", "0.0", "2", "2.0", "N", "NO", "FALSE", "F"}:
        return False
    # CLSA missing/refused/don't-know values are intentionally not guessed.
    return None


def classify_racial_background(
    frame: Any,
    indicator_to_label: Mapping[str, str],
    *,
    output_column: str = "racial_background",
) -> Any:
    """Classify released multiple-response cultural/racial indicators.

    A participant selecting more than one released group is retained as
    ``Multiple groups`` rather than forced into a single category.  The result
    describes self-reported released categories, not genetic ancestry.
    """
    import pandas as pd

    _require_columns(frame, indicator_to_label, "Racial-background source")
    output = frame.copy()
    decoded = {
        column: output[column].map(binary_indicator)
        for column in indicator_to_label
    }
    decoded_frame = pd.DataFrame(decoded, index=output.index)

    labels: list[str | None] = []
    details: list[str | None] = []
    counts: list[int] = []
    statuses: list[str] = []
    for index in output.index:
        selected = [
            label
            for column, label in indicator_to_label.items()
            if not pd.isna(decoded_frame.at[index, column])
            and bool(decoded_frame.at[index, column])
        ]
        observed = sum(
            not pd.isna(decoded_frame.at[index, column])
            for column in indicator_to_label
        )
        counts.append(len(selected))
        if len(selected) == 1:
            labels.append(selected[0])
            details.append(selected[0])
            statuses.append("single_released_group")
        elif len(selected) > 1:
            labels.append("Multiple groups")
            details.append(" | ".join(sorted(selected)))
            statuses.append("multiple_released_groups")
        elif observed:
            labels.append(None)
            details.append(None)
            statuses.append("no_released_group_selected")
        else:
            labels.append(None)
            details.append(None)
            statuses.append("all_indicators_missing_or_unknown")

    output[output_column] = pd.Series(labels, index=output.index, dtype="string")
    output[f"{output_column}_detail"] = pd.Series(
        details, index=output.index, dtype="string"
    )
    output[f"{output_column}_selection_count"] = counts
    output[f"{output_column}_status"] = statuses
    output[f"{output_column}_analysis_eligible"] = output[output_column].notna()
    return output


def pool_age_predictions_to_participants(
    frame: Any,
    *,
    participant_column: str = "participant_id",
    visit_column: str = "visit",
    age_column: str = "age",
    prediction_column: str = "retinal_age_prediction_oof",
    visit_priority: Sequence[str] = ("BL", "F1"),
    carry_columns: Sequence[str] = (),
) -> Any:
    """Average eyes/images within visits, then retain one visit per person."""
    import numpy as np
    import pandas as pd

    required = [participant_column, visit_column, age_column, prediction_column]
    _require_columns(frame, required, "Image-level age predictions")
    work = frame.copy()
    work[participant_column] = work[participant_column].astype(str)
    work[visit_column] = (
        work[visit_column]
        .astype("string")
        .str.upper()
        .replace({"FUP1": "F1"})
    )
    work[age_column] = pd.to_numeric(work[age_column], errors="coerce")
    work[prediction_column] = pd.to_numeric(
        work[prediction_column], errors="coerce"
    )
    work = work.dropna(subset=[age_column, prediction_column, visit_column])

    def stable_value(series: Any) -> Any:
        nonmissing = series.dropna()
        if nonmissing.empty:
            return None
        modes = nonmissing.mode(dropna=True)
        return modes.iloc[0] if not modes.empty else nonmissing.iloc[0]

    aggregations: dict[str, Any] = {
        age_column: "median",
        prediction_column: "mean",
    }
    for column in carry_columns:
        if column in work.columns and column not in aggregations:
            aggregations[column] = stable_value
    visit = (
        work.groupby([participant_column, visit_column], as_index=False)
        .agg(**{
            age_column: (age_column, "median"),
            prediction_column: (prediction_column, "mean"),
            "n_images": (prediction_column, "size"),
            "age_within_visit_range": (
                age_column,
                lambda values: float(np.nanmax(values) - np.nanmin(values)),
            ),
            **{
                column: (column, stable_value)
                for column in carry_columns
                if column in work.columns
            },
        })
    )
    priority = {str(value).upper(): index for index, value in enumerate(visit_priority)}
    visit["_visit_priority"] = visit[visit_column].map(priority).fillna(
        len(priority)
    )
    visit = visit.sort_values(
        [participant_column, "_visit_priority", "n_images", visit_column],
        ascending=[True, True, False, True],
        kind="stable",
    )
    participant = visit.drop_duplicates(participant_column, keep="first").copy()
    participant = participant.drop(columns="_visit_priority").reset_index(drop=True)
    participant["retinal_age_gap_oof"] = (
        participant[prediction_column] - participant[age_column]
    )
    participant["absolute_error_oof"] = participant[
        "retinal_age_gap_oof"
    ].abs()
    return participant


def _metric_values(age: Any, prediction: Any) -> dict[str, float]:
    import numpy as np

    age = np.asarray(age, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    gap = prediction - age
    if len(age) > 1 and float(np.std(age, ddof=1)) > 0:
        slope, intercept = np.polyfit(age, prediction, 1)
        correlation = float(np.corrcoef(age, prediction)[0, 1])
    else:
        slope = intercept = correlation = float("nan")
    return {
        "mae": float(np.mean(np.abs(gap))),
        "rmse": float(np.sqrt(np.mean(gap**2))),
        "mean_gap": float(np.mean(gap)),
        "median_gap": float(np.median(gap)),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "correlation": correlation,
        "overprediction_rate": float(np.mean(gap > 0)),
    }


def fairness_metric_table(
    frame: Any,
    group_column: str,
    *,
    age_column: str = "age",
    prediction_column: str = "retinal_age_prediction_oof",
    participant_column: str = "participant_id",
    bootstrap_repetitions: int = 1000,
    random_state: int = 20260821,
) -> Any:
    """Return participant-level age performance with bootstrap confidence intervals."""
    import numpy as np
    import pandas as pd

    _require_columns(
        frame,
        [participant_column, group_column, age_column, prediction_column],
        "Fairness evaluation frame",
    )
    if bootstrap_repetitions < 0:
        raise ValueError("bootstrap_repetitions cannot be negative")
    work = frame.dropna(subset=[group_column, age_column, prediction_column]).copy()
    if work[participant_column].astype(str).duplicated().any():
        raise ValueError("Fairness metrics require one row per participant")
    rng = np.random.default_rng(random_state)
    metric_names = list(_metric_values([1, 2], [1, 2]))
    rows: list[dict[str, Any]] = []
    groups = [("Overall", work)] + list(work.groupby(group_column, dropna=False))
    for group, subset in groups:
        subset = subset.reset_index(drop=True)
        point = _metric_values(subset[age_column], subset[prediction_column])
        bootstrap = {name: [] for name in metric_names}
        if bootstrap_repetitions and len(subset) > 1:
            for _ in range(bootstrap_repetitions):
                sampled = subset.iloc[rng.integers(0, len(subset), len(subset))]
                values = _metric_values(
                    sampled[age_column], sampled[prediction_column]
                )
                for name in metric_names:
                    if np.isfinite(values[name]):
                        bootstrap[name].append(values[name])
        row: dict[str, Any] = {
            group_column: str(group),
            "n_participants": int(len(subset)),
            "age_mean": float(pd.to_numeric(subset[age_column]).mean()),
            "age_sd": float(pd.to_numeric(subset[age_column]).std(ddof=1)),
            **point,
        }
        for name in metric_names:
            samples = np.asarray(bootstrap[name], dtype=float)
            row[f"{name}_ci_low"] = (
                float(np.quantile(samples, 0.025)) if samples.size else np.nan
            )
            row[f"{name}_ci_high"] = (
                float(np.quantile(samples, 0.975)) if samples.size else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _gower_distance(
    target: Any,
    references: Any,
    covariates: Sequence[str],
    *,
    numeric_covariates: set[str],
) -> Any:
    import numpy as np
    import pandas as pd

    total = np.zeros(len(references), dtype=float)
    denominators = np.zeros(len(references), dtype=float)
    for column in covariates:
        target_value = target.get(column)
        reference_values = references[column]
        target_missing = pd.isna(target_value)
        reference_missing = reference_values.isna().to_numpy()
        both_missing = reference_missing & target_missing
        one_missing = reference_missing ^ target_missing
        observed = ~(both_missing | one_missing)
        distance = np.zeros(len(references), dtype=float)
        distance[one_missing] = 1.0
        if column in numeric_covariates and not target_missing:
            values = pd.to_numeric(reference_values, errors="coerce").to_numpy(float)
            finite = values[np.isfinite(values)]
            scale = float(np.ptp(finite)) if finite.size else 1.0
            scale = scale if scale > 0 else 1.0
            distance[observed] = np.minimum(
                np.abs(values[observed] - float(target_value)) / scale, 1.0
            )
        elif not target_missing:
            distance[observed] = (
                reference_values.astype(str).to_numpy()[observed]
                != str(target_value)
            ).astype(float)
        informative = ~both_missing
        total += distance
        denominators += informative.astype(float)
    return total / np.maximum(denominators, 1.0)


def match_group_to_reference(
    frame: Any,
    target_group: str,
    *,
    group_column: str = "racial_background",
    reference_group: str = "White",
    participant_column: str = "participant_id",
    age_column: str = "age",
    age_caliper_years: float = 1.0,
    ratio: int = 2,
    exact_columns: Sequence[str] = ("sex_at_birth",),
    distance_columns: Sequence[str] = (),
    numeric_distance_columns: Sequence[str] = (),
) -> tuple[Any, Any, Any]:
    """Difficulty-first nearest-neighbour matching without reference reuse.

    Each target receives up to ``ratio`` references and at least one. Matching
    is run separately for each target racial-background group, so a White
    reference can legitimately appear in another group's separate estimand.
    """
    import numpy as np
    import pandas as pd

    if ratio < 1:
        raise ValueError("ratio must be at least 1")
    if age_caliper_years <= 0:
        raise ValueError("age_caliper_years must be positive")
    columns = [participant_column, group_column, age_column]
    columns += list(exact_columns) + list(distance_columns)
    _require_columns(frame, columns, "Matching frame")
    work = frame.copy()
    work[participant_column] = work[participant_column].astype(str)
    if work[participant_column].duplicated().any():
        raise ValueError("Matching requires one row per participant")
    work[age_column] = pd.to_numeric(work[age_column], errors="coerce")
    targets = work[work[group_column].astype(str) == str(target_group)].copy()
    references = work[
        work[group_column].astype(str) == str(reference_group)
    ].copy()
    targets = targets.dropna(subset=[age_column])
    references = references.dropna(subset=[age_column])
    if targets.empty or references.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def candidate_mask(target: Any, pool: Any) -> Any:
        mask = (pool[age_column] - float(target[age_column])).abs().le(
            age_caliper_years
        )
        for column in exact_columns:
            target_value = target[column]
            if pd.isna(target_value):
                mask &= pool[column].isna()
            else:
                mask &= pool[column].astype(str).eq(str(target_value))
        return mask

    target_order: list[tuple[int, float, str, int]] = []
    for index, target in targets.iterrows():
        count = int(candidate_mask(target, references).sum())
        target_order.append(
            (count, float(target[age_column]), str(target[participant_column]), index)
        )
    target_order.sort()

    available = references.copy()
    pair_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    numeric = set(numeric_distance_columns)
    for sequence, (_, _, _, index) in enumerate(target_order, 1):
        target = targets.loc[index]
        candidates = available[candidate_mask(target, available)].copy()
        available_before = len(candidates)
        if candidates.empty:
            audit_rows.append(
                {
                    "target_group": target_group,
                    "target_participant_id": str(target[participant_column]),
                    "matched": False,
                    "matched_reference_count": 0,
                    "eligible_references_at_match": 0,
                    "reason": "no_reference_within_constraints",
                }
            )
            continue
        candidates["age_distance_years"] = (
            candidates[age_column] - float(target[age_column])
        ).abs()
        if distance_columns:
            candidates["covariate_distance"] = _gower_distance(
                target,
                candidates,
                list(distance_columns),
                numeric_covariates=numeric,
            )
        else:
            candidates["covariate_distance"] = 0.0
        candidates = candidates.sort_values(
            ["covariate_distance", "age_distance_years", participant_column],
            kind="stable",
        )
        selected = candidates.head(ratio)
        match_set_id = f"{target_group.replace(' ', '_')}_{sequence:07d}"
        membership_rows.append(
            {
                "match_set_id": match_set_id,
                "participant_id": str(target[participant_column]),
                "match_role": "target",
                "target_group": target_group,
            }
        )
        for rank, (_, reference) in enumerate(selected.iterrows(), 1):
            reference_id = str(reference[participant_column])
            pair_rows.append(
                {
                    "match_set_id": match_set_id,
                    "match_rank": rank,
                    "target_group": target_group,
                    "target_participant_id": str(target[participant_column]),
                    "reference_participant_id": reference_id,
                    "target_age": float(target[age_column]),
                    "reference_age": float(reference[age_column]),
                    "absolute_age_difference_years": float(
                        reference["age_distance_years"]
                    ),
                    "covariate_distance": float(reference["covariate_distance"]),
                }
            )
            membership_rows.append(
                {
                    "match_set_id": match_set_id,
                    "participant_id": reference_id,
                    "match_role": "reference",
                    "target_group": target_group,
                }
            )
        selected_ids = selected[participant_column].astype(str)
        available = available[
            ~available[participant_column].astype(str).isin(selected_ids)
        ]
        audit_rows.append(
            {
                "target_group": target_group,
                "target_participant_id": str(target[participant_column]),
                "matched": True,
                "matched_reference_count": int(len(selected)),
                "eligible_references_at_match": int(available_before),
                "reason": None if len(selected) == ratio else "partial_ratio_match",
            }
        )
    return (
        pd.DataFrame(pair_rows),
        pd.DataFrame(audit_rows),
        pd.DataFrame(membership_rows),
    )


def standardized_mean_differences(
    target: Any,
    reference: Any,
    covariates: Sequence[str],
) -> Any:
    """Return maximum absolute SMD per numeric or categorical covariate."""
    import numpy as np
    import pandas as pd

    _require_columns(target, covariates, "Target balance frame")
    _require_columns(reference, covariates, "Reference balance frame")
    rows: list[dict[str, Any]] = []
    for column in covariates:
        left = target[column]
        right = reference[column]
        left_numeric = pd.to_numeric(left, errors="coerce")
        right_numeric = pd.to_numeric(right, errors="coerce")
        is_numeric = (
            left_numeric.notna().sum() == left.notna().sum()
            and right_numeric.notna().sum() == right.notna().sum()
        )
        if is_numeric:
            pooled = np.sqrt(
                (left_numeric.var(ddof=1) + right_numeric.var(ddof=1)) / 2
            )
            smd = (
                (left_numeric.mean() - right_numeric.mean()) / pooled
                if np.isfinite(pooled) and pooled > 0
                else 0.0
            )
        else:
            categories = sorted(
                set(left.dropna().astype(str)) | set(right.dropna().astype(str))
            )
            category_smds = []
            for category in categories:
                p_left = float(left.astype(str).eq(category).mean())
                p_right = float(right.astype(str).eq(category).mean())
                pooled = np.sqrt(
                    (p_left * (1 - p_left) + p_right * (1 - p_right)) / 2
                )
                category_smds.append(
                    (p_left - p_right) / pooled if pooled > 0 else 0.0
                )
            smd = max(category_smds, key=abs) if category_smds else 0.0
        rows.append({"covariate": column, "smd": float(smd), "abs_smd": abs(float(smd))})
    return pd.DataFrame(rows)


def matched_outcome_contrasts(
    frame: Any,
    membership: Any,
    outcomes: Sequence[str] = ("retinal_age_gap_oof", "absolute_error_oof"),
    *,
    participant_column: str = "participant_id",
    bootstrap_repetitions: int = 5000,
    random_state: int = 20260821,
) -> Any:
    """Compare each target with the within-set mean of its matched references."""
    import numpy as np
    import pandas as pd

    _require_columns(frame, [participant_column, *outcomes], "Outcome frame")
    _require_columns(
        membership,
        ["match_set_id", "participant_id", "match_role", "target_group"],
        "Match membership",
    )
    if membership.empty:
        return pd.DataFrame()
    work = membership.merge(
        frame[[participant_column, *outcomes]],
        left_on="participant_id",
        right_on=participant_column,
        how="left",
        validate="many_to_one",
    )
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []
    for target_group, group_membership in work.groupby("target_group"):
        for outcome in outcomes:
            differences = []
            for _, match_set in group_membership.groupby("match_set_id"):
                target = pd.to_numeric(
                    match_set.loc[match_set["match_role"] == "target", outcome],
                    errors="coerce",
                ).dropna()
                references = pd.to_numeric(
                    match_set.loc[
                        match_set["match_role"] == "reference", outcome
                    ],
                    errors="coerce",
                ).dropna()
                if len(target) == 1 and len(references):
                    differences.append(float(target.iloc[0] - references.mean()))
            values = np.asarray(differences, dtype=float)
            if not values.size:
                continue
            if bootstrap_repetitions and values.size > 1:
                samples = np.asarray(
                    [
                        rng.choice(values, size=len(values), replace=True).mean()
                        for _ in range(bootstrap_repetitions)
                    ]
                )
                low, high = np.quantile(samples, [0.025, 0.975])
                # Centered-bootstrap two-sided probability; conservative at 1/B.
                p_value = 2 * min(float(np.mean(samples <= 0)), float(np.mean(samples >= 0)))
                p_value = min(max(p_value, 1 / bootstrap_repetitions), 1.0)
            else:
                low = high = p_value = np.nan
            rows.append(
                {
                    "target_group": target_group,
                    "outcome": outcome,
                    "n_matched_sets": int(len(values)),
                    "target_minus_reference": float(values.mean()),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "p_value": float(p_value),
                }
            )
    return pd.DataFrame(rows)


def benjamini_hochberg(values: Sequence[float]) -> Any:
    """Benjamini-Hochberg adjusted p-values preserving missing entries."""
    import numpy as np

    p_values = np.asarray(values, dtype=float)
    adjusted = np.full(len(p_values), np.nan)
    finite = np.flatnonzero(np.isfinite(p_values))
    if not len(finite):
        return adjusted
    order = finite[np.argsort(p_values[finite])]
    ranked = p_values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted
