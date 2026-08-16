"""Participant-level questionnaire and aging-biomarker analyses.

The functions are independent of Databricks I/O.  Notebook 10 assembles the
matched CLSA cohort and uses these helpers for multiplicity-controlled,
participant-level inference.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def require_columns(
    frame: pd.DataFrame, columns: Sequence[str], label: str
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Return Benjamini--Hochberg adjusted p-values."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(values[valid])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def _pooled_standard_deviation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or len(right) < 2:
        return math.nan
    denominator = len(left) + len(right) - 2
    if denominator < 1:
        return math.nan
    variance = (
        (len(left) - 1) * np.var(left, ddof=1)
        + (len(right) - 1) * np.var(right, ddof=1)
    ) / denominator
    return float(np.sqrt(max(variance, 0.0)))


def _adjusted_group_model(
    frame: pd.DataFrame,
    *,
    outcome: str,
    outcome_type: str,
    group_column: str,
    covariates: Sequence[str],
    categorical_covariates: Sequence[str],
    cluster_column: str | None,
) -> dict[str, Any]:
    """Fit one adjusted group model with matched-set clustered uncertainty."""
    import statsmodels.api as sm

    columns = [outcome, group_column, *covariates]
    if cluster_column:
        columns.append(cluster_column)
    columns = list(dict.fromkeys(columns))
    work = frame[columns].copy()
    work[outcome] = pd.to_numeric(work[outcome], errors="coerce")
    work[group_column] = pd.to_numeric(work[group_column], errors="coerce")
    for covariate in covariates:
        if covariate not in categorical_covariates:
            work[covariate] = pd.to_numeric(work[covariate], errors="coerce")
    work = work.dropna(subset=columns)
    if len(work) < 20 or work[group_column].nunique() != 2:
        return {"adjusted_status": "insufficient_complete_records"}
    if outcome_type == "binary" and set(work[outcome].unique()) != {0, 1}:
        return {"adjusted_status": "binary_outcome_lacks_both_levels"}

    design = pd.DataFrame(
        {group_column: work[group_column].astype(float)}, index=work.index
    )
    for covariate in covariates:
        if covariate in categorical_covariates:
            encoded = pd.get_dummies(
                work[covariate].astype("string"),
                prefix=covariate,
                drop_first=True,
                dtype=float,
            )
            design = pd.concat([design, encoded], axis=1)
        else:
            design[covariate] = work[covariate].astype(float)
    design = design.loc[:, design.nunique(dropna=False) > 1]
    if group_column not in design.columns:
        return {"adjusted_status": "group_column_removed_from_design"}
    design = sm.add_constant(design.astype(float), has_constant="add")
    outcome_values = work.loc[design.index, outcome].astype(float)
    try:
        if outcome_type == "binary":
            model = sm.GLM(
                outcome_values,
                design,
                family=sm.families.Binomial(),
            )
        elif outcome_type == "continuous":
            model = sm.OLS(outcome_values, design)
        else:
            raise ValueError(f"Unsupported adjusted outcome type: {outcome_type}")
        if cluster_column and work.loc[design.index, cluster_column].nunique() >= 10:
            fitted = model.fit(
                cov_type="cluster",
                cov_kwds={
                    "groups": work.loc[design.index, cluster_column].astype(str),
                    "use_correction": True,
                },
            )
            covariance = "matched_set_cluster_robust"
        else:
            fitted = model.fit(cov_type="HC3")
            covariance = "HC3"
        coefficient = float(fitted.params[group_column])
        low, high = np.asarray(fitted.conf_int().loc[group_column], dtype=float)
        result = {
            "adjusted_status": "ok",
            "adjusted_n": int(len(work)),
            "adjusted_n_clusters": (
                int(work[cluster_column].nunique()) if cluster_column else None
            ),
            "adjusted_covariance": covariance,
            "adjusted_coefficient": coefficient,
            "adjusted_ci_low": float(low),
            "adjusted_ci_high": float(high),
            "adjusted_p_value": float(fitted.pvalues[group_column]),
            "adjusted_effect_scale": (
                "log_odds_ratio" if outcome_type == "binary" else "mean_difference"
            ),
        }
        if outcome_type == "binary":
            result.update(
                {
                    "adjusted_odds_ratio": float(np.exp(coefficient)),
                    "adjusted_odds_ratio_ci_low": float(np.exp(low)),
                    "adjusted_odds_ratio_ci_high": float(np.exp(high)),
                }
            )
        return result
    except Exception as error:
        return {
            "adjusted_status": f"model_failed:{type(error).__name__}",
        }


def compare_questionnaire_groups(
    frame: pd.DataFrame,
    variable_specifications: Mapping[str, Mapping[str, str]],
    *,
    group_column: str = "glaucoma_label",
    participant_column: str = "participant_id",
    covariates: Sequence[str] = ("age", "sex_at_birth", "visit"),
    categorical_covariates: Sequence[str] = ("sex_at_birth", "visit"),
    cluster_column: str | None = "match_set_id",
    minimum_per_group: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare questionnaire answers between two participant groups.

    Binary and continuous outcomes receive adjusted models. Multi-level
    categorical outcomes receive an omnibus chi-square test. FDR correction is
    applied once across the prespecified questionnaire-variable family.
    """
    from scipy import stats

    require_columns(
        frame,
        [participant_column, group_column],
        "Questionnaire comparison frame",
    )
    if frame[participant_column].astype(str).duplicated().any():
        raise ValueError("Questionnaire comparison requires one row per participant")
    group = pd.to_numeric(frame[group_column], errors="coerce")
    if set(group.dropna().unique()) != {0, 1}:
        raise ValueError("Questionnaire comparison requires binary groups 0 and 1")

    rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for variable, specification in variable_specifications.items():
        label = specification.get("label", variable)
        variable_type = specification.get("type", "categorical")
        if variable not in frame.columns:
            rows.append(
                {
                    "variable": variable,
                    "label": label,
                    "variable_type": variable_type,
                    "analysis_status": "column_missing",
                }
            )
            continue
        observed = frame[variable].notna()
        missing_table = pd.crosstab(
            group, observed.astype(int)
        ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        missing_p = (
            float(stats.fisher_exact(missing_table.to_numpy()).pvalue)
            if missing_table.to_numpy().sum() > 0
            else math.nan
        )
        missing_rows.append(
            {
                "variable": variable,
                "label": label,
                "healthy_n": int((group == 0).sum()),
                "glaucoma_n": int((group == 1).sum()),
                "healthy_missing_n": int(((group == 0) & ~observed).sum()),
                "glaucoma_missing_n": int(((group == 1) & ~observed).sum()),
                "healthy_missing_fraction": float(
                    (~observed[group == 0]).mean()
                ),
                "glaucoma_missing_fraction": float(
                    (~observed[group == 1]).mean()
                ),
                "missingness_fisher_p": missing_p,
            }
        )

        work = frame.loc[observed].copy()
        values = work[variable]
        group_work = pd.to_numeric(work[group_column], errors="coerce")
        n_healthy = int((group_work == 0).sum())
        n_glaucoma = int((group_work == 1).sum())
        base = {
            "variable": variable,
            "label": label,
            "variable_type": variable_type,
            "n_healthy": n_healthy,
            "n_glaucoma": n_glaucoma,
        }
        if min(n_healthy, n_glaucoma) < minimum_per_group:
            rows.append({**base, "analysis_status": "insufficient_group_records"})
            continue

        if variable_type in {"binary", "continuous"}:
            numeric = pd.to_numeric(values, errors="coerce")
            work = work.loc[numeric.notna()].copy()
            work[variable] = numeric[numeric.notna()].astype(float)
            healthy = work.loc[
                pd.to_numeric(work[group_column], errors="coerce") == 0,
                variable,
            ].to_numpy(float)
            glaucoma = work.loc[
                pd.to_numeric(work[group_column], errors="coerce") == 1,
                variable,
            ].to_numpy(float)
            base.update(
                {
                    "n_healthy": int(len(healthy)),
                    "n_glaucoma": int(len(glaucoma)),
                    "healthy_mean": float(np.mean(healthy)),
                    "glaucoma_mean": float(np.mean(glaucoma)),
                    "raw_difference_glaucoma_minus_healthy": float(
                        np.mean(glaucoma) - np.mean(healthy)
                    ),
                }
            )
            if min(len(healthy), len(glaucoma)) < minimum_per_group:
                rows.append(
                    {**base, "analysis_status": "insufficient_numeric_records"}
                )
                continue
            if variable_type == "binary":
                if not set(np.unique(np.concatenate([healthy, glaucoma]))).issubset(
                    {0.0, 1.0}
                ):
                    rows.append({**base, "analysis_status": "invalid_binary_codes"})
                    continue
                table = pd.crosstab(
                    pd.to_numeric(work[group_column], errors="coerce"),
                    work[variable].astype(int),
                ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
                unadjusted_p = float(stats.fisher_exact(table.to_numpy()).pvalue)
                base.update(
                    {
                        "healthy_prevalence": float(np.mean(healthy)),
                        "glaucoma_prevalence": float(np.mean(glaucoma)),
                        "unadjusted_test": "fisher_exact",
                        "unadjusted_p_value": unadjusted_p,
                    }
                )
            else:
                unadjusted_p = float(
                    stats.ttest_ind(
                        glaucoma, healthy, equal_var=False, nan_policy="omit"
                    ).pvalue
                )
                mann_whitney_p = float(
                    stats.mannwhitneyu(
                        glaucoma, healthy, alternative="two-sided"
                    ).pvalue
                )
                pooled_sd = _pooled_standard_deviation(healthy, glaucoma)
                base.update(
                    {
                        "healthy_sd": float(np.std(healthy, ddof=1)),
                        "glaucoma_sd": float(np.std(glaucoma, ddof=1)),
                        "standardized_mean_difference": (
                            float((np.mean(glaucoma) - np.mean(healthy)) / pooled_sd)
                            if np.isfinite(pooled_sd) and pooled_sd > 0
                            else math.nan
                        ),
                        "unadjusted_test": "welch_t",
                        "unadjusted_p_value": unadjusted_p,
                        "mann_whitney_p_value": mann_whitney_p,
                    }
                )
            adjusted = _adjusted_group_model(
                work,
                outcome=variable,
                outcome_type=variable_type,
                group_column=group_column,
                covariates=[column for column in covariates if column in work.columns],
                categorical_covariates=categorical_covariates,
                cluster_column=(
                    cluster_column if cluster_column in work.columns else None
                ),
            )
            primary_p = adjusted.get("adjusted_p_value", unadjusted_p)
            rows.append(
                {
                    **base,
                    **adjusted,
                    "primary_p_value": primary_p,
                    "analysis_status": "ok",
                }
            )
        elif variable_type == "categorical":
            clean = values.astype("string")
            table = pd.crosstab(group_work, clean)
            table = table.loc[table.sum(axis=1) > 0, table.sum(axis=0) > 0]
            if table.shape[0] != 2 or table.shape[1] < 2:
                rows.append({**base, "analysis_status": "no_category_variation"})
                continue
            chi2, p_value, _, _ = stats.chi2_contingency(table.to_numpy())
            total = float(table.to_numpy().sum())
            cramer_v = math.sqrt(
                float(chi2) / (total * min(table.shape[0] - 1, table.shape[1] - 1))
            )
            rows.append(
                {
                    **base,
                    "n_levels": int(table.shape[1]),
                    "levels": " | ".join(map(str, table.columns)),
                    "unadjusted_test": "chi_square_omnibus",
                    "unadjusted_p_value": float(p_value),
                    "cramers_v": float(cramer_v),
                    "primary_p_value": float(p_value),
                    "adjusted_status": "not_fit_for_multilevel_categorical",
                    "analysis_status": "ok",
                }
            )
        else:
            raise ValueError(f"Unsupported variable type for {variable}: {variable_type}")

    results = pd.DataFrame(rows)
    if "primary_p_value" in results.columns:
        results["fdr_q_value"] = benjamini_hochberg(
            pd.to_numeric(results["primary_p_value"], errors="coerce")
        )
        results["significant_fdr_0_05"] = results["fdr_q_value"] < 0.05
    missingness = pd.DataFrame(missing_rows)
    if not missingness.empty:
        missingness["missingness_fdr_q_value"] = benjamini_hochberg(
            missingness["missingness_fisher_p"]
        )
    return results, missingness


def questionnaire_group_descriptives(
    frame: pd.DataFrame,
    variable_specifications: Mapping[str, Mapping[str, str]],
    *,
    group_column: str = "glaucoma_label",
) -> pd.DataFrame:
    """Return long-form, identifier-free questionnaire descriptives."""
    require_columns(frame, [group_column], "Questionnaire descriptive frame")
    rows: list[dict[str, Any]] = []
    for variable, specification in variable_specifications.items():
        if variable not in frame.columns:
            continue
        variable_type = specification.get("type", "categorical")
        label = specification.get("label", variable)
        for group_value, group in frame.groupby(group_column, dropna=False):
            observed = group[variable].dropna()
            base = {
                "variable": variable,
                "label": label,
                "variable_type": variable_type,
                "glaucoma_label": int(group_value),
                "participants_total": int(len(group)),
                "participants_observed": int(len(observed)),
                "participants_missing": int(group[variable].isna().sum()),
            }
            if variable_type == "continuous":
                numeric = pd.to_numeric(observed, errors="coerce").dropna()
                rows.append(
                    {
                        **base,
                        "level": None,
                        "level_count": None,
                        "level_fraction_among_observed": None,
                        "mean": float(numeric.mean()) if len(numeric) else math.nan,
                        "standard_deviation": (
                            float(numeric.std(ddof=1)) if len(numeric) > 1 else math.nan
                        ),
                        "median": float(numeric.median()) if len(numeric) else math.nan,
                        "q1": float(numeric.quantile(0.25)) if len(numeric) else math.nan,
                        "q3": float(numeric.quantile(0.75)) if len(numeric) else math.nan,
                    }
                )
            else:
                counts = observed.astype("string").value_counts(dropna=False)
                for level, count in counts.items():
                    rows.append(
                        {
                            **base,
                            "level": str(level),
                            "level_count": int(count),
                            "level_fraction_among_observed": (
                                float(count / len(observed)) if len(observed) else math.nan
                            ),
                            "mean": None,
                            "standard_deviation": None,
                            "median": None,
                            "q1": None,
                            "q3": None,
                        }
                    )
    return pd.DataFrame(rows)


def lins_concordance(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute Lin's concordance correlation coefficient."""
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3:
        return math.nan
    covariance = float(np.cov(x, y, ddof=1)[0, 1])
    denominator = float(
        np.var(x, ddof=1)
        + np.var(y, ddof=1)
        + (np.mean(x) - np.mean(y)) ** 2
    )
    return float(2 * covariance / denominator) if denominator > 0 else math.nan


def age_measure_agreement(
    frame: pd.DataFrame,
    *,
    retinal_age_column: str,
    comparator_columns: Mapping[str, str],
    group_column: str = "glaucoma_label",
    minimum_pairs: int = 10,
) -> pd.DataFrame:
    """Compare retinal age with chronological and epigenetic clock ages."""
    from scipy import stats

    require_columns(frame, [retinal_age_column, group_column], "Age agreement frame")
    rows: list[dict[str, Any]] = []
    strata: list[tuple[str, pd.DataFrame]] = [("all", frame)]
    for group_value, label in ((0, "healthy"), (1, "glaucoma")):
        strata.append(
            (
                label,
                frame[
                    pd.to_numeric(frame[group_column], errors="coerce")
                    == group_value
                ],
            )
        )
    for stratum, subset in strata:
        for comparator, comparator_label in comparator_columns.items():
            if comparator not in subset.columns:
                rows.append(
                    {
                        "stratum": stratum,
                        "comparator": comparator,
                        "comparator_label": comparator_label,
                        "analysis_status": "column_missing",
                    }
                )
                continue
            paired = subset[[retinal_age_column, comparator]].apply(
                pd.to_numeric, errors="coerce"
            ).dropna()
            if len(paired) < minimum_pairs:
                rows.append(
                    {
                        "stratum": stratum,
                        "comparator": comparator,
                        "comparator_label": comparator_label,
                        "n_pairs": int(len(paired)),
                        "analysis_status": "insufficient_pairs",
                    }
                )
                continue
            retinal = paired[retinal_age_column].to_numpy(float)
            other = paired[comparator].to_numpy(float)
            difference = retinal - other
            standard_error = float(stats.sem(difference))
            critical = float(stats.t.ppf(0.975, len(difference) - 1))
            pearson = stats.pearsonr(retinal, other)
            spearman = stats.spearmanr(retinal, other)
            wilcoxon_p = (
                float(stats.wilcoxon(difference).pvalue)
                if not np.allclose(difference, 0)
                else 1.0
            )
            rows.append(
                {
                    "stratum": stratum,
                    "comparator": comparator,
                    "comparator_label": comparator_label,
                    "n_pairs": int(len(paired)),
                    "retinal_age_mean": float(np.mean(retinal)),
                    "comparator_mean": float(np.mean(other)),
                    "mean_difference_retinal_minus_comparator": float(
                        np.mean(difference)
                    ),
                    "difference_95_ci_low": float(
                        np.mean(difference) - critical * standard_error
                    ),
                    "difference_95_ci_high": float(
                        np.mean(difference) + critical * standard_error
                    ),
                    "paired_t_p_value": float(stats.ttest_rel(retinal, other).pvalue),
                    "wilcoxon_p_value": wilcoxon_p,
                    "pearson_r": float(pearson.statistic),
                    "pearson_p_value": float(pearson.pvalue),
                    "spearman_rho": float(spearman.statistic),
                    "spearman_p_value": float(spearman.pvalue),
                    "lins_concordance": lins_concordance(retinal, other),
                    "mean_absolute_difference": float(np.mean(np.abs(difference))),
                    "root_mean_squared_difference": float(
                        np.sqrt(np.mean(np.square(difference)))
                    ),
                    "analysis_status": "ok",
                }
            )
    result = pd.DataFrame(rows)
    if "paired_t_p_value" in result.columns:
        result["paired_difference_fdr_q_value"] = benjamini_hochberg(
            pd.to_numeric(result["paired_t_p_value"], errors="coerce")
        )
    return result


def correlate_age_accelerations(
    frame: pd.DataFrame,
    *,
    retinal_gap_column: str,
    epigenetic_acceleration_columns: Mapping[str, str],
    group_column: str = "glaucoma_label",
    minimum_pairs: int = 10,
) -> pd.DataFrame:
    """Correlate retinal-age gap with released epigenetic accelerations."""
    from scipy import stats

    require_columns(frame, [retinal_gap_column, group_column], "Age-gap frame")
    rows: list[dict[str, Any]] = []
    strata: list[tuple[str, pd.DataFrame]] = [("all", frame)]
    for group_value, label in ((0, "healthy"), (1, "glaucoma")):
        strata.append(
            (
                label,
                frame[
                    pd.to_numeric(frame[group_column], errors="coerce")
                    == group_value
                ],
            )
        )
    for stratum, subset in strata:
        for column, label in epigenetic_acceleration_columns.items():
            if column not in subset.columns:
                rows.append(
                    {
                        "stratum": stratum,
                        "epigenetic_measure": column,
                        "label": label,
                        "analysis_status": "column_missing",
                    }
                )
                continue
            paired = subset[[retinal_gap_column, column]].apply(
                pd.to_numeric, errors="coerce"
            ).dropna()
            if len(paired) < minimum_pairs:
                rows.append(
                    {
                        "stratum": stratum,
                        "epigenetic_measure": column,
                        "label": label,
                        "n_pairs": int(len(paired)),
                        "analysis_status": "insufficient_pairs",
                    }
                )
                continue
            retinal = paired[retinal_gap_column].to_numpy(float)
            epigenetic = paired[column].to_numpy(float)
            pearson = stats.pearsonr(retinal, epigenetic)
            spearman = stats.spearmanr(retinal, epigenetic)
            rows.append(
                {
                    "stratum": stratum,
                    "epigenetic_measure": column,
                    "label": label,
                    "n_pairs": int(len(paired)),
                    "pearson_r": float(pearson.statistic),
                    "pearson_p_value": float(pearson.pvalue),
                    "spearman_rho": float(spearman.statistic),
                    "spearman_p_value": float(spearman.pvalue),
                    "analysis_status": "ok",
                }
            )
    result = pd.DataFrame(rows)
    if "pearson_p_value" in result.columns:
        result["pearson_fdr_q_value"] = benjamini_hochberg(
            pd.to_numeric(result["pearson_p_value"], errors="coerce")
        )
    return result
