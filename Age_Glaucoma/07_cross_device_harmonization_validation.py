# Databricks notebook source
# MAGIC %md
# MAGIC # Cross-device harmonization validation
# MAGIC
# MAGIC This notebook refines the exploratory Zeiss transport analysis from
# MAGIC notebook 06 using practices from multi-device retinal-imaging studies:
# MAGIC
# MAGIC - participant-held-out fitting of every source correction;
# MAGIC - balanced source-by-disease development cells;
# MAGIC - comparison of location-only and location-plus-scale correction;
# MAGIC - age/sex-matched source diagnostics;
# MAGIC - a nested held-out source-classification test;
# MAGIC - prespecified acceptance gates before a Zeiss result is called
# MAGIC   supportive.
# MAGIC
# MAGIC It reuses the stored participant-level RETFound embeddings and the
# MAGIC frozen `CLSA_healthy` age head. **No images are re-embedded.** The
# MAGIC within-CLSA glaucoma comparison from notebook 06 remains primary.
# MAGIC
# MAGIC Relevant precedents include cross-device foundation-embedding
# MAGIC translation (ICML FMSD 2026 preliminary work), paired-camera image
# MAGIC translation (TVST 2023), feature alignment for glaucoma domain
# MAGIC generalization (Biomedical Optics Express 2022), and standardized
# MAGIC disc-centered evaluation across 13 fundus datasets (npj Digital
# MAGIC Medicine 2023). Unlike the paired-device studies, CLSA and Zeiss do not
# MAGIC contain the same eyes imaged on both devices; a nonlinear device
# MAGIC translator is therefore not identifiable and is intentionally omitted.

# COMMAND ----------
from pathlib import Path
import importlib
import json
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# COMMAND ----------
dbutils.widgets.text(
    "repo_root",
    "/Workspace/Users/ad0038@pennmedicine.upenn.edu/CLSA/CLSA_retina",
)
dbutils.widgets.text(
    "age_glaucoma_output_root",
    "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/derived/"
    "clsa_retinal_aging/Age_Glaucoma",
)
dbutils.widgets.text("three_cohort_root", "")
dbutils.widgets.text("healthy_model_path", "")
dbutils.widgets.text("output_root", "")
dbutils.widgets.text("crossfit_folds", "5")
dbutils.widgets.text("age_caliper_years", "1.0")
dbutils.widgets.text("bootstrap_repetitions", "5000")
dbutils.widgets.text("harmonization_ridge", "0.000001")
dbutils.widgets.text("domain_classifier_max_per_source", "5000")
dbutils.widgets.text("maximum_acceptable_domain_auc", "0.60")
dbutils.widgets.text("maximum_acceptable_p90_feature_smd", "0.10")

# COMMAND ----------
repo_root = Path(dbutils.widgets.get("repo_root").strip())
age_glaucoma_root = Path(
    dbutils.widgets.get("age_glaucoma_output_root").strip()
)


def configured_path(widget_name, default):
    value = dbutils.widgets.get(widget_name).strip()
    return Path(value) if value else Path(default)


three_cohort_root = configured_path(
    "three_cohort_root",
    age_glaucoma_root / "11_three_cohort_glaucoma",
)
cohort_root = three_cohort_root / "01_cohort"
healthy_model_path = configured_path(
    "healthy_model_path",
    age_glaucoma_root / "06_CLSA_healthy_model" / "CLSA_healthy.joblib",
)
output_root = configured_path(
    "output_root",
    age_glaucoma_root / "12_cross_device_harmonization_validation",
)
crossfit_folds = int(dbutils.widgets.get("crossfit_folds"))
age_caliper_years = float(dbutils.widgets.get("age_caliper_years"))
bootstrap_repetitions = int(dbutils.widgets.get("bootstrap_repetitions"))
harmonization_ridge = float(dbutils.widgets.get("harmonization_ridge"))
domain_classifier_max_per_source = int(
    dbutils.widgets.get("domain_classifier_max_per_source")
)
maximum_acceptable_domain_auc = float(
    dbutils.widgets.get("maximum_acceptable_domain_auc")
)
maximum_acceptable_p90_feature_smd = float(
    dbutils.widgets.get("maximum_acceptable_p90_feature_smd")
)

if crossfit_folds < 3:
    raise ValueError("crossfit_folds must be at least 3")
if age_caliper_years < 0:
    raise ValueError("age_caliper_years cannot be negative")
if bootstrap_repetitions < 500:
    raise ValueError("bootstrap_repetitions must be at least 500")
if not 0.5 <= maximum_acceptable_domain_auc <= 1.0:
    raise ValueError("maximum_acceptable_domain_auc must be between 0.5 and 1")
if maximum_acceptable_p90_feature_smd <= 0:
    raise ValueError("maximum_acceptable_p90_feature_smd must be positive")

module_root = repo_root / "src"
if not module_root.exists():
    raise FileNotFoundError(f"Repository source directory not found: {module_root}")
if str(module_root) not in sys.path:
    sys.path.insert(0, str(module_root))

import fundus_retfound_pipeline as _fundus_pipeline  # noqa: E402
import three_cohort_glaucoma as _three_cohort  # noqa: E402

_fundus_pipeline = importlib.reload(_fundus_pipeline)
_three_cohort = importlib.reload(_three_cohort)

from fundus_retfound_pipeline import (  # noqa: E402
    load_age_head,
    predict_retinal_age,
    write_frame,
    write_json,
)
from three_cohort_glaucoma import (  # noqa: E402
    adjusted_group_effect,
    canonical_sex,
    cross_validated_domain_auc,
    crossfit_source_harmonizer,
    embedding_shift_summary,
    greedy_match,
    nested_harmonization_domain_auc,
    paired_outcome_effect,
    validate_embedding_frame,
)

print("Loaded fundus helper:", _fundus_pipeline.__file__)
print("Loaded harmonization helper:", _three_cohort.__file__)
print("RETFound inference performed by this notebook: false")

# COMMAND ----------
input_paths = {
    "CLSA healthy participants": cohort_root / "clsa_healthy_participants.parquet",
    "CLSA glaucoma participants": cohort_root / "clsa_glaucoma_only_participants.parquet",
    "Zeiss glaucoma participants": cohort_root / "zeiss_glaucoma_participants.parquet",
    "notebook 06 results": three_cohort_root / "THREE_COHORT_GLAUCOMA_RESULTS.csv",
    "frozen healthy age model": healthy_model_path,
}
missing = [f"{name}: {path}" for name, path in input_paths.items() if not path.exists()]
if missing:
    raise FileNotFoundError("Required notebook 06 outputs are missing:\n- " + "\n- ".join(missing))

participant_output_root = output_root / "01_crossfit_participants"
diagnostic_root = output_root / "02_domain_diagnostics"
comparison_root = output_root / "03_transport_estimates"
figure_root = output_root / "04_figures"
model_root = output_root / "05_harmonizers"
for path in (
    participant_output_root,
    diagnostic_root,
    comparison_root,
    figure_root,
    model_root,
):
    path.mkdir(parents=True, exist_ok=True)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Load the locked notebook 06 cohorts
# MAGIC
# MAGIC The strict healthy model and the same-source CLSA glaucoma estimate are
# MAGIC not refitted. Harmonization is restricted to the cross-device
# MAGIC transport analysis.

# COMMAND ----------
healthy = pd.read_parquet(input_paths["CLSA healthy participants"])
clsa_glaucoma = pd.read_parquet(input_paths["CLSA glaucoma participants"])
zeiss_glaucoma = pd.read_parquet(input_paths["Zeiss glaucoma participants"])


def prepare_cohort(frame, cohort, source, glaucoma):
    frame = frame.copy()
    frame.attrs = {}
    frame["cohort"] = cohort
    frame["source"] = source
    frame["glaucoma"] = glaucoma
    if "sex_normalized" not in frame.columns:
        frame["sex_normalized"] = frame.get(
            "sex", pd.Series(index=frame.index, dtype=object)
        ).map(canonical_sex)
    else:
        frame["sex_normalized"] = frame["sex_normalized"].map(canonical_sex)
    frame["age"] = pd.to_numeric(frame["age"], errors="coerce")
    frame["age_squared"] = frame["age"] ** 2
    frame = validate_embedding_frame(frame, cohort)
    frame["age_squared"] = frame["age"] ** 2
    if frame["participant_id"].astype(str).duplicated().any():
        raise ValueError(f"{cohort} is not unique at participant level")
    return frame


healthy = prepare_cohort(healthy, "CLSA healthy", "CLSA", 0)
clsa_glaucoma = prepare_cohort(
    clsa_glaucoma,
    "CLSA glaucoma-only",
    "CLSA",
    1,
)
zeiss_glaucoma = prepare_cohort(
    zeiss_glaucoma,
    "Zeiss glaucoma",
    "Zeiss",
    1,
)

age_bundle = load_age_head(str(healthy_model_path))
if age_bundle.get("model_name") != "CLSA_healthy" or not age_bundle.get("frozen", False):
    raise ValueError("The configured age model is not the frozen CLSA_healthy model")

combined = pd.concat([healthy, clsa_glaucoma, zeiss_glaucoma], ignore_index=True)
combined = validate_embedding_frame(combined, "Three-cohort participant input")

cohort_audit = pd.DataFrame(
    [
        {
            "cohort": cohort,
            "participants": int(len(frame)),
            "mean_age": float(frame["age"].mean()),
            "sd_age": float(frame["age"].std()),
            "sex_observed_fraction": float(
                (frame["sex_normalized"] != "MISSING").mean()
            ),
        }
        for cohort, frame in (
            ("CLSA healthy", healthy),
            ("CLSA glaucoma-only", clsa_glaucoma),
            ("Zeiss glaucoma", zeiss_glaucoma),
        )
    ]
)
display(cohort_audit.round(3))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Lock age/sex-matched cross-device diagnostic participants
# MAGIC
# MAGIC Exactly the same matched participants are used to compare raw and
# MAGIC corrected feature distributions. This prevents an apparently improved
# MAGIC result from being caused by changing the diagnostic cohort.

# COMMAND ----------
def shared_sex_matching_columns(exposed, reference):
    observed_exposed = exposed["sex_normalized"] != "MISSING"
    observed_reference = reference["sex_normalized"] != "MISSING"
    shared = set(exposed.loc[observed_exposed, "sex_normalized"]) & set(
        reference.loc[observed_reference, "sex_normalized"]
    )
    if observed_exposed.mean() >= 0.80 and observed_reference.mean() >= 0.80 and len(shared) >= 2:
        return ["sex_normalized"]
    return []


source_exact_columns = shared_sex_matching_columns(
    zeiss_glaucoma,
    clsa_glaucoma,
)
source_pairs = greedy_match(
    zeiss_glaucoma,
    clsa_glaucoma,
    caliper_years=age_caliper_years,
    exact_columns=source_exact_columns,
)
if source_pairs.empty:
    raise ValueError("No Zeiss-to-CLSA glaucoma source-diagnostic matches were found")

write_frame(source_pairs, diagnostic_root / "source_diagnostic_pairs_private.parquet")
print("Source diagnostic pairs:", len(source_pairs))
print("Mean absolute age difference:", source_pairs["absolute_age_difference"].mean())
print("Exact matching columns:", source_exact_columns or "age only")


def select_locked_source_pairs(frame):
    clsa_ids = set(source_pairs["reference_id"].astype(str))
    zeiss_ids = set(source_pairs["exposed_id"].astype(str))
    clsa = frame[
        (frame["source"] == "CLSA")
        & (frame["glaucoma"] == 1)
        & (frame["participant_id"].astype(str).isin(clsa_ids))
    ].copy()
    zeiss = frame[
        (frame["source"] == "Zeiss")
        & (frame["glaucoma"] == 1)
        & (frame["participant_id"].astype(str).isin(zeiss_ids))
    ].copy()
    if len(clsa) != len(source_pairs) or len(zeiss) != len(source_pairs):
        raise ValueError("Locked domain pair IDs did not resolve one-to-one")
    return clsa, zeiss

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Cross-fit two prespecified harmonization variants
# MAGIC
# MAGIC For each fold, the source correction is estimated using only the other
# MAGIC participants. The held-out participant is then transformed. This avoids
# MAGIC fitting and evaluating a scanner correction on the same record.

# COMMAND ----------
modes = ("location", "location_scale")
stage_frames = {"raw": combined.copy()}
bundle_paths = {}
for mode in modes:
    print(f"Cross-fitting {mode} harmonization across {crossfit_folds} folds")
    transformed, bundles = crossfit_source_harmonizer(
        combined,
        folds=crossfit_folds,
        mode=mode,
        ridge=harmonization_ridge,
        random_state=20260811,
    )
    bundle_path = model_root / f"crossfit_{mode}_harmonizers.joblib"
    joblib.dump(bundles, bundle_path)
    bundle_paths[mode] = str(bundle_path)

    # Only Zeiss embeddings changed. Reapply the locked healthy age head to
    # those corrected vectors; retain the CLSA OOF/application predictions.
    zeiss = transformed[transformed["source"] == "Zeiss"].copy()
    zeiss = zeiss.drop(
        columns=[
            column
            for column in (
                "retinal_age_prediction",
                "retinal_age_gap",
                "absolute_error",
                "retinal_age_raw",
            )
            if column in zeiss.columns
        ]
    )
    predictions = predict_retinal_age(zeiss, age_bundle)
    prediction_columns = [
        "participant_id",
        "retinal_age_prediction",
        "retinal_age_gap",
        "absolute_error",
    ]
    zeiss = zeiss.merge(
        predictions[prediction_columns],
        on="participant_id",
        how="inner",
        validate="one_to_one",
    )
    clsa = transformed[transformed["source"] == "CLSA"].copy()
    transformed = pd.concat([clsa, zeiss], ignore_index=True)
    stage_frames[mode] = transformed
    write_frame(
        transformed,
        participant_output_root / f"participants_crossfit_{mode}.parquet",
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Source-removal diagnostics and acceptance gates
# MAGIC
# MAGIC A correction passes only if both its matched cross-fitted source AUC and
# MAGIC its stricter nested outer-fold AUC are at most 0.60, while the 90th
# MAGIC percentile absolute feature SMD is at most 0.10. These thresholds are
# MAGIC prespecified pragmatic gates, not universal clinical standards.

# COMMAND ----------
diagnostic_rows = []
nested_fold_rows = []
for stage, frame in stage_frames.items():
    clsa_locked, zeiss_locked = select_locked_source_pairs(frame)
    shift = embedding_shift_summary(clsa_locked, zeiss_locked)
    matched_auc = cross_validated_domain_auc(
        clsa_locked,
        zeiss_locked,
        max_per_domain=domain_classifier_max_per_source,
    )
    if stage == "raw":
        nested_auc = matched_auc.copy()
        nested_auc["evaluation_design"] = "raw matched participant CV"
        nested_auc["fold_results"] = []
    else:
        nested_auc = nested_harmonization_domain_auc(
            combined,
            mode=stage,
            folds=crossfit_folds,
            ridge=harmonization_ridge,
            max_per_source=domain_classifier_max_per_source,
        )
        for row in nested_auc["fold_results"]:
            nested_fold_rows.append({"stage": stage, **row})
    passed = (
        matched_auc["domain_auc_mean"] <= maximum_acceptable_domain_auc
        and nested_auc["domain_auc_mean"] <= maximum_acceptable_domain_auc
        and shift["p90_absolute_feature_smd"]
        <= maximum_acceptable_p90_feature_smd
    )
    diagnostic_rows.append(
        {
            "stage": stage,
            **shift,
            "matched_domain_auc_effective": matched_auc["domain_auc_mean"],
            "matched_domain_auc_signed": matched_auc["domain_auc_signed_mean"],
            "nested_domain_auc_effective": nested_auc["domain_auc_mean"],
            "nested_domain_auc_signed": nested_auc["domain_auc_signed_mean"],
            "nested_domain_auc_sd": nested_auc["domain_auc_sd"],
            "passes_source_removal_gate": bool(passed),
        }
    )

domain_diagnostics = pd.DataFrame(diagnostic_rows)
nested_fold_diagnostics = pd.DataFrame(nested_fold_rows)
write_frame(domain_diagnostics, diagnostic_root / "cross_device_diagnostics.csv")
write_frame(nested_fold_diagnostics, diagnostic_root / "nested_domain_auc_folds.csv")
display(domain_diagnostics.round(4))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Re-estimate the transported retinal-age contrasts
# MAGIC
# MAGIC Adjusted regression and no-replacement participant matching are repeated
# MAGIC for the raw and both cross-fitted corrections. Agreement across these
# MAGIC methods is required before describing Zeiss as external support.

# COMMAND ----------
def transport_comparison(stage, exposed, reference, comparison):
    exact_columns = shared_sex_matching_columns(exposed, reference)
    categorical = exact_columns.copy()
    regression = pd.concat(
        [
            reference.assign(exposed=0),
            exposed.assign(exposed=1),
        ],
        ignore_index=True,
    )
    regression["age_squared"] = regression["age"] ** 2
    adjusted = adjusted_group_effect(
        regression,
        outcome_column="retinal_age_gap",
        numeric_covariates=("age", "age_squared"),
        categorical_covariates=categorical,
    )
    pairs = greedy_match(
        exposed,
        reference,
        caliper_years=age_caliper_years,
        exact_columns=exact_columns,
    )
    matched, _ = paired_outcome_effect(
        pairs,
        exposed,
        reference,
        outcome_column="retinal_age_gap",
        bootstrap_repetitions=bootstrap_repetitions,
    )
    return [
        {
            "stage": stage,
            "comparison": comparison,
            "method": "adjusted_hc3_regression",
            "estimate": adjusted["adjusted_difference"],
            "ci_95_low": adjusted["ci_95_low"],
            "ci_95_high": adjusted["ci_95_high"],
            "p_value": adjusted["p_value"],
            "n": adjusted["n"],
            "n_pairs": np.nan,
            "mean_absolute_age_difference": np.nan,
            "adjustment": " + ".join(adjusted["design_columns"]),
        },
        {
            "stage": stage,
            "comparison": comparison,
            "method": "matched_participant_bootstrap",
            "estimate": matched["mean_difference"],
            "ci_95_low": matched["bootstrap_95_ci_low"],
            "ci_95_high": matched["bootstrap_95_ci_high"],
            "p_value": np.nan,
            "n": matched["n_pairs"] * 2,
            "n_pairs": matched["n_pairs"],
            "mean_absolute_age_difference": matched[
                "mean_absolute_age_difference"
            ],
            "adjustment": (
                "1:1 age"
                + ("/sex" if exact_columns else "")
                + f" matching within ±{age_caliper_years:g} years"
            ),
        },
    ]


transport_rows = []
for stage, frame in stage_frames.items():
    stage_healthy = frame[frame["cohort"] == "CLSA healthy"].copy()
    stage_clsa_glaucoma = frame[
        frame["cohort"] == "CLSA glaucoma-only"
    ].copy()
    stage_zeiss = frame[frame["cohort"] == "Zeiss glaucoma"].copy()
    transport_rows.extend(
        transport_comparison(
            stage,
            stage_zeiss,
            stage_clsa_glaucoma,
            "zeiss_glaucoma_vs_clsa_glaucoma",
        )
    )
    transport_rows.extend(
        transport_comparison(
            stage,
            stage_zeiss,
            stage_healthy,
            "zeiss_glaucoma_vs_clsa_healthy",
        )
    )

transport_results = pd.DataFrame(transport_rows)
write_frame(transport_results, comparison_root / "transport_estimates.csv")
display(transport_results.round(4))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Method concordance and publication decision

# COMMAND ----------
notebook06_results = pd.read_csv(input_paths["notebook 06 results"])
primary = notebook06_results[
    (notebook06_results["comparison"] == "clsa_glaucoma_vs_clsa_healthy_primary")
    & (notebook06_results["method"] == "matched_participant_bootstrap")
]
if len(primary) != 1:
    raise ValueError("Could not resolve the locked primary CLSA result")
primary_estimate = float(primary.iloc[0]["estimate"])

decision_rows = []
for mode in modes:
    diagnostic = domain_diagnostics.set_index("stage").loc[mode]
    estimates = transport_results[
        (transport_results["stage"] == mode)
        & (
            transport_results["comparison"]
            == "zeiss_glaucoma_vs_clsa_healthy"
        )
    ]
    signs = np.sign(estimates["estimate"].to_numpy(float))
    directionally_consistent = bool(
        len(signs) == 2
        and np.all(signs == np.sign(primary_estimate))
    )
    supportive = bool(
        diagnostic["passes_source_removal_gate"]
        and directionally_consistent
    )
    decision_rows.append(
        {
            "method": mode,
            "passes_source_removal_gate": bool(
                diagnostic["passes_source_removal_gate"]
            ),
            "directionally_consistent_with_primary_clsa": directionally_consistent,
            "eligible_as_supportive_external_analysis": supportive,
            "nested_domain_auc_effective": float(
                diagnostic["nested_domain_auc_effective"]
            ),
            "p90_absolute_feature_smd": float(
                diagnostic["p90_absolute_feature_smd"]
            ),
        }
    )
method_decisions = pd.DataFrame(decision_rows)
write_frame(method_decisions, output_root / "HARMONIZATION_METHOD_DECISIONS.csv")
display(method_decisions)

eligible = method_decisions[
    method_decisions["eligible_as_supportive_external_analysis"]
].sort_values(
    ["nested_domain_auc_effective", "p90_absolute_feature_smd"]
)
selected_method = "none" if eligible.empty else str(eligible.iloc[0]["method"])

# COMMAND ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
plot_diagnostics = domain_diagnostics.set_index("stage").loc[
    ["raw", "location", "location_scale"]
]
axes[0].bar(
    plot_diagnostics.index,
    plot_diagnostics["nested_domain_auc_effective"],
    color=["#dc2626", "#2563eb", "#7c3aed"],
)
axes[0].axhline(maximum_acceptable_domain_auc, color="black", linestyle="--")
axes[0].axhline(0.5, color="gray", linestyle=":")
axes[0].set_ylim(0.45, 1.02)
axes[0].set_ylabel("Direction-invariant held-out source AUC")
axes[0].set_title("Residual device information")

axes[1].bar(
    plot_diagnostics.index,
    plot_diagnostics["p90_absolute_feature_smd"],
    color=["#dc2626", "#2563eb", "#7c3aed"],
)
axes[1].axhline(
    maximum_acceptable_p90_feature_smd,
    color="black",
    linestyle="--",
)
axes[1].set_ylabel("90th percentile |feature SMD|")
axes[1].set_title("Matched embedding balance")
fig.tight_layout()
fig.savefig(figure_root / "harmonization_acceptance_diagnostics.png", dpi=200)
display(fig)
plt.close(fig)

forest = transport_results[
    transport_results["comparison"] == "zeiss_glaucoma_vs_clsa_healthy"
].copy()
forest["label"] = forest["stage"] + " | " + forest["method"]
forest = forest.reset_index(drop=True)
fig, axis = plt.subplots(figsize=(10, 6))
y = np.arange(len(forest))
axis.errorbar(
    forest["estimate"],
    y,
    xerr=np.vstack(
        [
            forest["estimate"] - forest["ci_95_low"],
            forest["ci_95_high"] - forest["estimate"],
        ]
    ),
    fmt="o",
    color="#1d4ed8",
    capsize=3,
)
axis.axvline(0, color="black", linewidth=1)
axis.axvline(primary_estimate, color="#059669", linestyle="--", label="Primary CLSA effect")
axis.set_yticks(y, forest["label"])
axis.set_xlabel("Retinal-age-gap difference (years)")
axis.set_title("Zeiss transport estimates across correction methods")
axis.legend()
fig.tight_layout()
fig.savefig(figure_root / "transport_method_forest.png", dpi=200)
display(fig)
plt.close(fig)

# COMMAND ----------
summary = {
    "analysis": "cross_device_harmonization_validation",
    "retfound_embeddings_recalculated": False,
    "primary_within_clsa_estimate_years": primary_estimate,
    "crossfit_folds": crossfit_folds,
    "methods_compared": ["raw", *modes],
    "selected_external_support_method": selected_method,
    "external_result_is_confirmatory": selected_method != "none",
    "acceptance_thresholds": {
        "maximum_effective_domain_auc": maximum_acceptable_domain_auc,
        "maximum_p90_absolute_feature_smd": maximum_acceptable_p90_feature_smd,
    },
    "harmonizer_paths": bundle_paths,
    "literature_alignment": [
        "participant-held-out correction fitting",
        "balanced source-by-disease cells",
        "matched domain diagnostics",
        "nested held-out device classification",
        "multiple prespecified correction variants",
    ],
    "nonidentifiable_components": [
        "source-by-glaucoma interaction because healthy Zeiss controls are absent",
        "nonlinear cross-device translation because paired-device images are absent",
        "glaucoma severity differences because CLSA severity is unavailable",
    ],
    "interpretation_rule": (
        "The Zeiss analysis is supportive only if a cross-fitted method passes "
        "both source-removal gates and its adjusted and matched estimates are "
        "directionally consistent with the locked within-CLSA result."
    ),
}
write_json(summary, output_root / "CROSS_DEVICE_HARMONIZATION_SUMMARY.json")
print(json.dumps(summary, indent=2))
print("Notebook 07 complete")
