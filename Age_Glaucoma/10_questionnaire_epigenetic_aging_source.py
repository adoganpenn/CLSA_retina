# Databricks notebook source
# MAGIC %md
# MAGIC # Matched CLSA glaucoma: questionnaire, retinal age, and epigenetic aging
# MAGIC
# MAGIC This notebook joins the complete participant-level cohort saved by
# MAGIC notebook 08 to the anatomy audit from notebook 09 and the visit-matched
# MAGIC SAP questionnaire table. It answers three prespecified questions:
# MAGIC
# MAGIC 1. Which questionnaire measures differ between matched healthy and
# MAGIC    glaucoma-only CLSA participants?
# MAGIC 2. Do RETFound retinal-age and retinal-age-gap measures differ by group?
# MAGIC 3. Among baseline participants with released methylation phenotypes, how
# MAGIC    do retinal age, chronological age, Horvath DNAm age, Hannum age, and
# MAGIC    the released epigenetic-acceleration measures relate?
# MAGIC
# MAGIC The six epigenetic fields are released baseline-derived phenotypes; this
# MAGIC notebook does not reconstruct clocks from raw DNA. Epigenetic analyses
# MAGIC are restricted to baseline fundus visits to preserve temporal alignment.
# MAGIC Questionnaire inference is participant-level, clusters uncertainty by
# MAGIC the notebook 08 matched set, and controls false discovery rate separately
# MAGIC for the prespecified derived/composite SAP outcomes and the secondary raw
# MAGIC component questions. Age and sex remain model covariates; analytic weight
# MAGIC and sampling stratum remain survey-design metadata rather than outcomes.
# MAGIC The CLSA release lacks a documented PSU, so these are matched-cohort
# MAGIC analyses—not full complex-survey estimates.

# COMMAND ----------
# MAGIC %pip install -q "statsmodels~=0.14.2" "scipy>=1.11,<2"

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
from pathlib import Path
import importlib
import json
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

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
dbutils.widgets.text("notebook08_root", "")
dbutils.widgets.text("notebook09_root", "")
dbutils.widgets.text(
    "sap_questionnaire_path",
    "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/derived/"
    "clsa_retinal_aging/sap_questionnaire_visit",
)
dbutils.widgets.text(
    "epigenetic_path",
    "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/derived/"
    "clsa_retinal_aging/sap_epigenetic_baseline",
)
dbutils.widgets.text("three_cohort_root", "")
dbutils.widgets.text("output_root", "")
dbutils.widgets.dropdown("require_complete_notebook09", "true", ["true", "false"])
dbutils.widgets.text("minimum_per_group", "10")
dbutils.widgets.text("fdr_alpha", "0.05")

# COMMAND ----------
repo_root = Path(dbutils.widgets.get("repo_root").strip())
age_glaucoma_root = Path(
    dbutils.widgets.get("age_glaucoma_output_root").strip()
)


def configured_path(widget_name, default):
    value = dbutils.widgets.get(widget_name).strip()
    return Path(value) if value else Path(default)


notebook08_root = configured_path(
    "notebook08_root",
    age_glaucoma_root / "13_glaucoma_classifier_spatial_validation",
)
notebook09_root = configured_path(
    "notebook09_root",
    age_glaucoma_root / "14_clsa_anatomic_explainability",
)
three_cohort_root = configured_path(
    "three_cohort_root",
    age_glaucoma_root / "11_three_cohort_glaucoma",
)
output_root = configured_path(
    "output_root",
    age_glaucoma_root / "15_questionnaire_epigenetic_aging",
)
sap_questionnaire_path = Path(
    dbutils.widgets.get("sap_questionnaire_path").strip()
)
epigenetic_path = Path(dbutils.widgets.get("epigenetic_path").strip())
require_complete_notebook09 = (
    dbutils.widgets.get("require_complete_notebook09") == "true"
)
minimum_per_group = int(dbutils.widgets.get("minimum_per_group"))
fdr_alpha = float(dbutils.widgets.get("fdr_alpha"))
if minimum_per_group < 5:
    raise ValueError("minimum_per_group must be at least 5")
if not 0 < fdr_alpha < 1:
    raise ValueError("fdr_alpha must lie in (0, 1)")

module_root = repo_root / "src"
if not module_root.exists():
    raise FileNotFoundError(f"Repository source directory not found: {module_root}")
if str(module_root) not in sys.path:
    sys.path.insert(0, str(module_root))

import questionnaire_epigenetic_analysis as _questionnaire_analysis  # noqa: E402

_questionnaire_analysis = importlib.reload(_questionnaire_analysis)
from fundus_retfound_pipeline import write_frame, write_json  # noqa: E402
from questionnaire_epigenetic_analysis import (  # noqa: E402
    age_measure_agreement,
    compare_questionnaire_groups,
    correlate_age_accelerations,
    questionnaire_group_descriptives,
)

print("Loaded questionnaire helper:", _questionnaire_analysis.__file__)

# COMMAND ----------
private_root = output_root / "01_private_participant_data"
statistics_root = output_root / "02_statistics"
figure_root = output_root / "03_figures"
for path in (private_root, statistics_root, figure_root):
    path.mkdir(parents=True, exist_ok=True)

cohort_path = (
    notebook08_root
    / "01_participant_classifier"
    / "CLSA_glaucoma_participant_oof_predictions.parquet"
)
match_path = (
    notebook08_root
    / "01_participant_classifier"
    / "matched_control_sets_private.parquet"
)
image_anatomy_path = (
    notebook09_root
    / "01_image_anatomy"
    / "clsa_anatomic_explainability_private.parquet"
)
participant_anatomy_path = (
    notebook09_root
    / "03_statistics"
    / "participant_anatomic_metrics_private.parquet"
)
required_paths = {
    "notebook 08 participant cohort": cohort_path,
    "notebook 08 matched sets": match_path,
    "notebook 09 image anatomy": image_anatomy_path,
    "notebook 09 participant anatomy": participant_anatomy_path,
    "SAP questionnaire Delta table": sap_questionnaire_path,
    "baseline epigenetic Delta table": epigenetic_path,
}
missing_paths = [
    f"{label}: {path}"
    for label, path in required_paths.items()
    if not path.exists()
]
if missing_paths:
    raise FileNotFoundError(
        "Required completed outputs are missing:\n- " + "\n- ".join(missing_paths)
    )


def normalize_visit_pandas(series):
    normalized = series.astype("string").str.upper()
    return normalized.replace({"FUP1": "F1"})


def write_aggregate(frame, filename):
    path = statistics_root / filename
    write_frame(frame.replace({np.nan: None}), path)
    return path

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Reconstruct the exact matched notebook 08 participant cohort

# COMMAND ----------
cohort_raw = pd.read_parquet(cohort_path)
cohort_raw.attrs = {}
required_cohort_columns = {
    "participant_id",
    "visit",
    "age",
    "glaucoma_label",
    "glaucoma_probability_oof",
    "classifier_logit_oof",
}
missing_cohort_columns = required_cohort_columns - set(cohort_raw.columns)
if missing_cohort_columns:
    raise ValueError(
        "Notebook 08 cohort is missing required columns: "
        f"{sorted(missing_cohort_columns)}"
    )
cohort_raw["participant_id"] = cohort_raw["participant_id"].astype(str)
cohort_raw["visit"] = normalize_visit_pandas(cohort_raw["visit"])
if cohort_raw["participant_id"].duplicated().any():
    raise ValueError("Notebook 08 cohort is not unique by participant")
if set(pd.to_numeric(cohort_raw["glaucoma_label"]).unique()) != {0, 1}:
    raise ValueError("Notebook 08 cohort does not contain both CLSA groups")

retinal_age_column = next(
    (
        column
        for column in (
            "retinal_age_prediction",
            "retinal_age_prediction_oof",
            "retinal_age",
        )
        if column in cohort_raw.columns
    ),
    None,
)
retinal_gap_column = next(
    (
        column
        for column in (
            "retinal_age_gap",
            "retinal_age_gap_oof",
        )
        if column in cohort_raw.columns
    ),
    None,
)

if retinal_age_column is None or retinal_gap_column is None:
    cohort_data_root = three_cohort_root / "01_cohort"
    prediction_sources = [
        cohort_data_root / "clsa_healthy_participants.parquet",
        cohort_data_root / "clsa_glaucoma_only_participants.parquet",
    ]
    if not all(path.exists() for path in prediction_sources):
        raise FileNotFoundError(
            "Retinal-age columns are absent from notebook 08 and notebook 06 "
            "participant prediction tables were not found."
        )
    retinal_source = pd.concat(
        [pd.read_parquet(path) for path in prediction_sources],
        ignore_index=True,
        sort=False,
    )
    retinal_source["participant_id"] = retinal_source[
        "participant_id"
    ].astype(str)
    retinal_source["visit"] = normalize_visit_pandas(retinal_source["visit"])
    retinal_source = retinal_source[
        [
            "participant_id",
            "visit",
            "retinal_age_prediction",
            "retinal_age_gap",
        ]
    ].drop_duplicates(["participant_id", "visit"])
    cohort_raw = cohort_raw.merge(
        retinal_source,
        on=["participant_id", "visit"],
        how="left",
        validate="one_to_one",
    )
    retinal_age_column = "retinal_age_prediction"
    retinal_gap_column = "retinal_age_gap"

cohort_columns = [
    "participant_id",
    "visit",
    "age",
    "glaucoma_label",
    "glaucoma_probability_oof",
    "classifier_logit_oof",
]
for optional in ("fold", "sex", "sex_normalized"):
    if optional in cohort_raw.columns:
        cohort_columns.append(optional)
cohort = cohort_raw[cohort_columns].copy()
cohort["retinal_age"] = pd.to_numeric(
    cohort_raw[retinal_age_column], errors="coerce"
)
cohort["retinal_age_gap"] = pd.to_numeric(
    cohort_raw[retinal_gap_column], errors="coerce"
)
cohort["retinal_age_prediction_mode"] = np.where(
    pd.to_numeric(cohort["glaucoma_label"], errors="coerce") == 0,
    "CLSA_healthy_grouped_out_of_fold",
    "CLSA_healthy_frozen_model_application",
)

matches = pd.read_parquet(match_path)
matches.attrs = {}
required_match_columns = {"match_set_id", "case_id", "control_id"}
if not required_match_columns.issubset(matches.columns):
    raise ValueError(
        "Notebook 08 match table is missing: "
        f"{sorted(required_match_columns - set(matches.columns))}"
    )
case_map = matches[["case_id", "match_set_id"]].drop_duplicates().rename(
    columns={"case_id": "participant_id"}
)
control_map = matches[["control_id", "match_set_id"]].drop_duplicates().rename(
    columns={"control_id": "participant_id"}
)
match_map = pd.concat([case_map, control_map], ignore_index=True)
match_map["participant_id"] = match_map["participant_id"].astype(str)
if match_map["participant_id"].duplicated().any():
    raise ValueError("A notebook 08 participant maps to multiple matched sets")
cohort = cohort.merge(
    match_map,
    on="participant_id",
    how="left",
    validate="one_to_one",
)
if cohort["match_set_id"].isna().any():
    raise ValueError("Some notebook 08 participants lack a matched-set assignment")

cohort_counts = (
    cohort.groupby("glaucoma_label", as_index=False)
    .agg(
        participants=("participant_id", "nunique"),
        mean_age=("age", "mean"),
        mean_retinal_age=("retinal_age", "mean"),
        mean_retinal_age_gap=("retinal_age_gap", "mean"),
    )
)
display(cohort_counts.round(3))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Verify notebook 09 coverage and attach participant anatomy metrics

# COMMAND ----------
image_anatomy = pd.read_parquet(image_anatomy_path)
image_anatomy["participant_id"] = image_anatomy["participant_id"].astype(str)
notebook09_participants = image_anatomy["participant_id"].nunique()
expected_participants = cohort["participant_id"].nunique()
print(
    "Notebook 09 participant coverage: "
    f"{notebook09_participants:,}/{expected_participants:,}"
)
if require_complete_notebook09 and notebook09_participants != expected_participants:
    raise RuntimeError(
        "Notebook 09 is incomplete for the saved notebook 08 cohort: "
        f"{notebook09_participants:,}/{expected_participants:,} participants. "
        "Finish notebook 09 with maximum_images=0 before notebook 10."
    )

participant_anatomy = pd.read_parquet(participant_anatomy_path)
participant_anatomy["participant_id"] = participant_anatomy[
    "participant_id"
].astype(str)
if participant_anatomy["participant_id"].duplicated().any():
    raise ValueError("Notebook 09 participant anatomy is not unique")
participant_anatomy_columns = [
    column
    for column in participant_anatomy.columns
    if column not in {"glaucoma_label"}
]
cohort = cohort.merge(
    participant_anatomy[participant_anatomy_columns],
    on="participant_id",
    how="left",
    validate="one_to_one",
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Load only matched participants from questionnaire and epigenetic tables
# MAGIC
# MAGIC Questionnaire answers are joined on participant **and exact image visit**.
# MAGIC Baseline methylation outputs are joined by participant, then set to missing
# MAGIC for non-baseline image visits in the inferential dataset.

# COMMAND ----------
PRIMARY_SAP_SPECIFICATIONS = {
    "visual_impairment_self_report": {
        "type": "binary",
        "label": "Self-reported visual impairment",
    },
    "depression_cesd10": {"type": "binary", "label": "CES-D10 depression"},
    "married_or_partnered": {"type": "binary", "label": "Married/partnered"},
    "diabetes": {"type": "binary", "label": "Diabetes"},
    "hypertension": {"type": "binary", "label": "Hypertension"},
    "heart_disease": {"type": "binary", "label": "Heart disease"},
    "stroke": {"type": "binary", "label": "Stroke"},
    "arthritis_any": {"type": "binary", "label": "Arthritis"},
    "osteoporosis": {"type": "binary", "label": "Osteoporosis"},
    "asthma_or_copd": {"type": "binary", "label": "Asthma or COPD"},
    "cancer": {"type": "binary", "label": "Cancer"},
    "low_back_pain": {"type": "binary", "label": "Low back pain"},
    "hearing_noise": {"type": "binary", "label": "Noise-related hearing issue"},
    "hearing_aid": {"type": "binary", "label": "Hearing-aid use"},
    "social_any_at_least_weekly": {
        "type": "binary",
        "label": "Any social activity at least weekly",
    },
    "visual_acuity_better_eye": {
        "type": "continuous",
        "label": "Better-eye acuity score",
    },
    "multimorbidity_selected_count": {
        "type": "continuous",
        "label": "Selected-condition count",
    },
    "ethnicity_spirometry": {
        "type": "categorical",
        "label": "Race/ethnicity (released spirometry output)",
    },
    "education_level_sap_harmonized": {
        "type": "categorical",
        "label": "Education category",
    },
    "household_income_band": {
        "type": "categorical",
        "label": "Household-income band",
    },
    "smoking_status": {"type": "categorical", "label": "Smoking status"},
    "adl_class": {"type": "categorical", "label": "ADL class"},
    "self_rated_healthy_aging": {
        "type": "categorical",
        "label": "Self-rated healthy aging",
    },
}

# Raw and component questions are tested separately so correlated components do
# not enlarge the confirmatory FDR family containing their derived composites.
SECONDARY_SAP_COMPONENT_SPECIFICATIONS = {
    "self_reported_vision": {
        "type": "categorical",
        "label": "Self-reported vision category",
    },
    "visual_acuity_left": {
        "type": "continuous",
        "label": "Left-eye acuity score",
    },
    "visual_acuity_right": {
        "type": "continuous",
        "label": "Right-eye acuity score",
    },
    "visual_impairment_acuity": {
        "type": "binary",
        "label": "Acuity-defined visual impairment (logMAR gate)",
    },
    "cesd10_score": {"type": "continuous", "label": "CES-D10 score"},
    "education_level_sap": {
        "type": "categorical",
        "label": "Original SAP education response",
    },
    "marital_status": {
        "type": "categorical",
        "label": "Original marital-status response",
    },
    "oa_hand": {"type": "binary", "label": "Hand osteoarthritis"},
    "oa_hip": {"type": "binary", "label": "Hip osteoarthritis"},
    "oa_knee": {"type": "binary", "label": "Knee osteoarthritis"},
    "asthma": {"type": "binary", "label": "Asthma"},
    "copd": {"type": "binary", "label": "COPD"},
    "social_outside_household_at_least_weekly": {
        "type": "binary",
        "label": "Outside-household activity weekly",
    },
    "social_religious_at_least_weekly": {
        "type": "binary",
        "label": "Religious activity weekly",
    },
    "social_education_culture_at_least_weekly": {
        "type": "binary",
        "label": "Educational/cultural activity weekly",
    },
    "social_club_at_least_weekly": {
        "type": "binary",
        "label": "Club activity weekly",
    },
    "social_association_at_least_weekly": {
        "type": "binary",
        "label": "Association activity weekly",
    },
    "social_other_at_least_weekly": {
        "type": "binary",
        "label": "Other social activity weekly",
    },
    "social_outside_household": {
        "type": "categorical",
        "label": "Outside-household activity frequency",
    },
    "social_religious": {
        "type": "categorical",
        "label": "Religious activity frequency",
    },
    "social_education_culture": {
        "type": "categorical",
        "label": "Educational/cultural activity frequency",
    },
    "social_club": {
        "type": "categorical",
        "label": "Club activity frequency",
    },
    "social_association": {
        "type": "categorical",
        "label": "Association activity frequency",
    },
    "social_other": {
        "type": "categorical",
        "label": "Other social activity frequency",
    },
}

QUESTIONNAIRE_SPECIFICATIONS = {
    **PRIMARY_SAP_SPECIFICATIONS,
    **SECONDARY_SAP_COMPONENT_SPECIFICATIONS,
}

EPIGENETIC_COLUMNS = [
    "epigenetic_dnam_age",
    "epigenetic_age_acceleration_difference",
    "epigenetic_age_acceleration_residual",
    "epigenetic_ieaa",
    "epigenetic_eeaa",
    "epigenetic_hannum_age",
]

cohort_id_rows = [(value,) for value in cohort["participant_id"].astype(str)]
cohort_ids_spark = spark.createDataFrame(
    cohort_id_rows, schema="participant_id string"
).dropDuplicates()

questionnaire_spark = spark.read.format("delta").load(
    str(sap_questionnaire_path)
)
questionnaire_spark = (
    questionnaire_spark.withColumn(
        "participant_id", F.trim(F.col("participant_id").cast("string"))
    )
    .withColumn(
        "visit",
        F.when(F.upper(F.col("visit")).isin("F1", "FUP1"), F.lit("F1"))
        .when(F.upper(F.col("visit")) == "BL", F.lit("BL"))
        .otherwise(F.lit(None).cast("string")),
    )
    .join(cohort_ids_spark, "participant_id", "inner")
)
questionnaire_columns = [
    "participant_id",
    "visit",
    "age_at_fundus_years",
    "sex_at_birth",
    "analytic_weight",
    "sampling_strata",
    *QUESTIONNAIRE_SPECIFICATIONS,
]
available_questionnaire_columns = [
    column for column in questionnaire_columns if column in questionnaire_spark.columns
]
questionnaire_spark = questionnaire_spark.select(
    *available_questionnaire_columns
)
if (
    questionnaire_spark.groupBy("participant_id", "visit")
    .count()
    .filter(F.col("count") > 1)
    .limit(1)
    .count()
):
    raise ValueError("Questionnaire table is not unique by participant and visit")
questionnaire = questionnaire_spark.toPandas()
questionnaire.attrs = {}
for missing_column in set(questionnaire_columns) - set(questionnaire.columns):
    questionnaire[missing_column] = np.nan

epigenetic_spark = (
    spark.read.format("delta")
    .load(str(epigenetic_path))
    .withColumn(
        "participant_id", F.trim(F.col("participant_id").cast("string"))
    )
    .join(cohort_ids_spark, "participant_id", "inner")
)
required_epigenetic_columns = {
    "participant_id",
    "chronological_age_at_baseline",
    *EPIGENETIC_COLUMNS,
}
missing_epigenetic_columns = required_epigenetic_columns - set(
    epigenetic_spark.columns
)
if missing_epigenetic_columns:
    raise ValueError(
        "Baseline epigenetic table is missing: "
        f"{sorted(missing_epigenetic_columns)}"
    )
epigenetic_spark = epigenetic_spark.select(
    "participant_id",
    F.col("chronological_age_at_baseline").alias(
        "epigenetic_chronological_age_at_baseline"
    ),
    *EPIGENETIC_COLUMNS,
    "epigenetic_measures_available_count",
    "epigenetic_complete_six_measure_panel",
    "epigenetic_difference_qc_status",
    "epigenetic_clock_range_qc_status",
)
if (
    epigenetic_spark.groupBy("participant_id")
    .count()
    .filter(F.col("count") > 1)
    .limit(1)
    .count()
):
    raise ValueError("Baseline epigenetic table is not unique by participant")
epigenetic = epigenetic_spark.toPandas()
epigenetic.attrs = {}

master = cohort.merge(
    questionnaire,
    on=["participant_id", "visit"],
    how="left",
    validate="one_to_one",
).merge(
    epigenetic,
    on="participant_id",
    how="left",
    validate="one_to_one",
)
if master["participant_id"].duplicated().any() or len(master) != len(cohort):
    raise RuntimeError("Questionnaire/epigenetic joins changed participant grain")
master["age"] = pd.to_numeric(master["age"], errors="coerce")
master["chronological_age"] = master["age"]
master["questionnaire_age_difference"] = master["age"] - pd.to_numeric(
    master.get("age_at_fundus_years", np.nan), errors="coerce"
)
for column in EPIGENETIC_COLUMNS:
    master[column] = pd.to_numeric(master[column], errors="coerce")
    master.loc[master["visit"] != "BL", column] = np.nan
master.loc[
    master["visit"] != "BL", "epigenetic_chronological_age_at_baseline"
] = np.nan
for column in (
    "epigenetic_measures_available_count",
    "epigenetic_complete_six_measure_panel",
    "epigenetic_difference_qc_status",
    "epigenetic_clock_range_qc_status",
):
    master.loc[master["visit"] != "BL", column] = np.nan
master["retinal_minus_chronological_age"] = (
    master["retinal_age"] - master["chronological_age"]
)
master["retinal_minus_dnam_age"] = (
    master["retinal_age"] - master["epigenetic_dnam_age"]
)
master["retinal_minus_hannum_age"] = (
    master["retinal_age"] - master["epigenetic_hannum_age"]
)
write_frame(
    master,
    private_root / "matched_questionnaire_epigenetic_master_private.parquet",
)

coverage_rows = []
for label, subset in (
    ("all", master),
    ("healthy", master[master["glaucoma_label"] == 0]),
    ("glaucoma", master[master["glaucoma_label"] == 1]),
):
    coverage_rows.append(
        {
            "group": label,
            "participants": int(subset["participant_id"].nunique()),
            "exact_visit_questionnaire": int(subset["sex_at_birth"].notna().sum()),
            "retinal_age": int(subset["retinal_age"].notna().sum()),
            "baseline_fundus_visit": int((subset["visit"] == "BL").sum()),
            "horvath_dnam_age": int(subset["epigenetic_dnam_age"].notna().sum()),
            "hannum_age": int(subset["epigenetic_hannum_age"].notna().sum()),
            "complete_six_epigenetic_measures": int(
                (
                    pd.to_numeric(
                        subset["epigenetic_complete_six_measure_panel"],
                        errors="coerce",
                    )
                    == 1
                ).sum()
            ),
        }
    )
coverage = pd.DataFrame(coverage_rows)
write_aggregate(coverage, "cohort_linkage_coverage.csv")
display(coverage)

sap_coverage_rows = []
for analysis_family, specifications in (
    ("primary_derived_or_composite", PRIMARY_SAP_SPECIFICATIONS),
    ("secondary_raw_or_component", SECONDARY_SAP_COMPONENT_SPECIFICATIONS),
):
    for variable, specification in specifications.items():
        observed_n = int(master[variable].notna().sum())
        sap_coverage_rows.append(
            {
                "variable": variable,
                "label": specification["label"],
                "analysis_family": analysis_family,
                "role": "analyzed_outcome",
                "observed_participants": observed_n,
                "coverage_fraction": observed_n / len(master),
                "coverage_status": (
                    "available" if observed_n else "no_observed_values"
                ),
                "note": (
                    "Acuity threshold remains disabled until the released "
                    "score is confirmed to be logMAR."
                    if variable == "visual_impairment_acuity"
                    else ""
                ),
            }
        )
for variable, label, role, note in (
    (
        "age_at_fundus_years",
        "Age at fundus visit",
        "model_covariate",
        "Used for visit alignment and adjustment; not tested as a questionnaire outcome.",
    ),
    (
        "sex_at_birth",
        "Sex at birth",
        "model_covariate",
        "Used for adjustment; not tested as a questionnaire outcome.",
    ),
    (
        "analytic_weight",
        "CLSA analytic weight",
        "survey_design",
        "Retained for sensitivity analyses; not an outcome.",
    ),
    (
        "sampling_strata",
        "CLSA sampling stratum",
        "survey_design",
        "Retained as design metadata; not an outcome.",
    ),
):
    observed_n = int(master[variable].notna().sum())
    sap_coverage_rows.append(
        {
            "variable": variable,
            "label": label,
            "analysis_family": "covariate_or_design",
            "role": role,
            "observed_participants": observed_n,
            "coverage_fraction": observed_n / len(master),
            "coverage_status": "available" if observed_n else "no_observed_values",
            "note": note,
        }
    )
for variable, label, note in (
    (
        "frailty",
        "Frailty",
        "Not derived because the SAP row does not define a frailty construct or source variable.",
    ),
    (
        "multimorbidity_32_condition_count",
        "Official 32-condition multimorbidity count",
        "Not derived because the SAP does not provide the complete source-variable list; the selected-condition count is labeled explicitly.",
    ),
    (
        "survey_psu",
        "Survey primary sampling unit",
        "No documented PSU field was identified; participant ID is not treated as a PSU.",
    ),
):
    sap_coverage_rows.append(
        {
            "variable": variable,
            "label": label,
            "analysis_family": "not_derived",
            "role": "unavailable",
            "observed_participants": 0,
            "coverage_fraction": 0.0,
            "coverage_status": "not_validly_derivable",
            "note": note,
        }
    )
sap_coverage = pd.DataFrame(sap_coverage_rows)
write_aggregate(sap_coverage, "sap_questionnaire_coverage_audit.csv")
display(sap_coverage)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Multiplicity-controlled questionnaire comparisons
# MAGIC
# MAGIC Binary and continuous measures use models adjusted for age, sex, and
# MAGIC visit with matched-set clustered uncertainty. Multi-level categorical
# MAGIC measures use omnibus chi-square tests. Benjamini--Hochberg correction is
# MAGIC applied separately to (1) the primary derived/composite SAP outcomes and
# MAGIC (2) the secondary raw/component questions. Missingness is analyzed within
# MAGIC the same two families. This avoids counting a composite and each of its
# MAGIC source questions as independent confirmatory hypotheses.

# COMMAND ----------
primary_questionnaire_results, primary_questionnaire_missingness = (
    compare_questionnaire_groups(
        master,
        PRIMARY_SAP_SPECIFICATIONS,
        group_column="glaucoma_label",
        participant_column="participant_id",
        covariates=("age", "sex_at_birth", "visit"),
        categorical_covariates=("sex_at_birth", "visit"),
        cluster_column="match_set_id",
        minimum_per_group=minimum_per_group,
    )
)
secondary_questionnaire_results, secondary_questionnaire_missingness = (
    compare_questionnaire_groups(
        master,
        SECONDARY_SAP_COMPONENT_SPECIFICATIONS,
        group_column="glaucoma_label",
        participant_column="participant_id",
        covariates=("age", "sex_at_birth", "visit"),
        categorical_covariates=("sex_at_birth", "visit"),
        cluster_column="match_set_id",
        minimum_per_group=minimum_per_group,
    )
)
primary_questionnaire_results["analysis_family"] = "primary_derived_or_composite"
secondary_questionnaire_results["analysis_family"] = "secondary_raw_or_component"
primary_questionnaire_missingness["analysis_family"] = (
    "primary_derived_or_composite"
)
secondary_questionnaire_missingness["analysis_family"] = (
    "secondary_raw_or_component"
)
questionnaire_results = pd.concat(
    [primary_questionnaire_results, secondary_questionnaire_results],
    ignore_index=True,
    sort=False,
)
questionnaire_missingness = pd.concat(
    [primary_questionnaire_missingness, secondary_questionnaire_missingness],
    ignore_index=True,
    sort=False,
)
questionnaire_descriptives = questionnaire_group_descriptives(
    master,
    QUESTIONNAIRE_SPECIFICATIONS,
    group_column="glaucoma_label",
)
family_lookup = {
    variable: "primary_derived_or_composite"
    for variable in PRIMARY_SAP_SPECIFICATIONS
}
family_lookup.update(
    {
        variable: "secondary_raw_or_component"
        for variable in SECONDARY_SAP_COMPONENT_SPECIFICATIONS
    }
)
questionnaire_descriptives["analysis_family"] = questionnaire_descriptives[
    "variable"
].map(family_lookup)
for column in (
    "fdr_q_value",
    "primary_p_value",
    "adjusted_odds_ratio",
    "adjusted_odds_ratio_ci_low",
    "adjusted_odds_ratio_ci_high",
):
    for result_frame in (
        primary_questionnaire_results,
        secondary_questionnaire_results,
        questionnaire_results,
    ):
        if column not in result_frame.columns:
            result_frame[column] = np.nan
write_aggregate(questionnaire_results, "questionnaire_group_comparisons.csv")
write_aggregate(
    primary_questionnaire_results,
    "questionnaire_primary_group_comparisons.csv",
)
write_aggregate(
    secondary_questionnaire_results,
    "questionnaire_secondary_component_comparisons.csv",
)
write_aggregate(questionnaire_missingness, "questionnaire_missingness_comparisons.csv")
write_aggregate(questionnaire_descriptives, "questionnaire_group_descriptives.csv")
significant_questionnaire = questionnaire_results[
    questionnaire_results.get("fdr_q_value", pd.Series(dtype=float)) < fdr_alpha
].copy()
display(
    questionnaire_results.sort_values(
        ["fdr_q_value", "primary_p_value"], na_position="last"
    ).round(5)
)
print(
    f"Questionnaire measures significant at FDR q<{fdr_alpha}: "
    f"{len(significant_questionnaire)} total"
)
print(
    significant_questionnaire.groupby("analysis_family").size().rename(
        "significant_measures"
    )
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Retinal-age and epigenetic-aging group comparisons
# MAGIC
# MAGIC Clock ages (`DNAmAge` and Hannum age) are compared in years and adjusted
# MAGIC for chronological age. The released residual, IEAA, EEAA, and
# MAGIC acceleration-difference measures are analyzed on their released scales.
# MAGIC The universal prespecified epigenetic acceleration outcome is the released
# MAGIC `AgeAccelerationResidual_COM` field.

# COMMAND ----------
AGING_VARIABLE_SPECIFICATIONS = {
    "retinal_age": {"type": "continuous", "label": "RETFound retinal age"},
    "retinal_age_gap": {
        "type": "continuous",
        "label": "RETFound retinal-age gap",
    },
    "epigenetic_dnam_age": {
        "type": "continuous",
        "label": "Horvath DNAm age",
    },
    "epigenetic_hannum_age": {
        "type": "continuous",
        "label": "Hannum epigenetic age",
    },
    "epigenetic_age_acceleration_difference": {
        "type": "continuous",
        "label": "Released DNAm age-acceleration difference",
    },
    "epigenetic_age_acceleration_residual": {
        "type": "continuous",
        "label": "Epigenetic age-acceleration residual",
    },
    "epigenetic_ieaa": {"type": "continuous", "label": "IEAA"},
    "epigenetic_eeaa": {"type": "continuous", "label": "EEAA"},
    "retinal_minus_chronological_age": {
        "type": "continuous",
        "label": "Retinal minus chronological age",
    },
    "retinal_minus_dnam_age": {
        "type": "continuous",
        "label": "Retinal minus Horvath DNAm age",
    },
    "retinal_minus_hannum_age": {
        "type": "continuous",
        "label": "Retinal minus Hannum age",
    },
}
aging_group_results, aging_missingness = compare_questionnaire_groups(
    master,
    AGING_VARIABLE_SPECIFICATIONS,
    group_column="glaucoma_label",
    participant_column="participant_id",
    covariates=("age", "sex_at_birth", "visit"),
    categorical_covariates=("sex_at_birth", "visit"),
    cluster_column="match_set_id",
    minimum_per_group=minimum_per_group,
)
for column in ("fdr_q_value", "primary_p_value"):
    if column not in aging_group_results.columns:
        aging_group_results[column] = np.nan
write_aggregate(aging_group_results, "aging_biomarker_group_comparisons.csv")
write_aggregate(aging_missingness, "aging_biomarker_missingness.csv")
display(
    aging_group_results.sort_values(
        ["fdr_q_value", "primary_p_value"], na_position="last"
    ).round(5)
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Within-participant agreement among aging measures
# MAGIC
# MAGIC A paired mean difference tests calibration/interchangeability, while
# MAGIC Pearson, Spearman, and Lin concordance quantify association and agreement.
# MAGIC A significant mean difference does not imply that one clock is more
# MAGIC biologically valid than another.

# COMMAND ----------
age_agreement = age_measure_agreement(
    master,
    retinal_age_column="retinal_age",
    comparator_columns={
        "chronological_age": "Chronological age",
        "epigenetic_dnam_age": "Horvath DNAm age",
        "epigenetic_hannum_age": "Hannum epigenetic age",
    },
    group_column="glaucoma_label",
    minimum_pairs=minimum_per_group,
)
write_aggregate(age_agreement, "retinal_vs_chronological_epigenetic_agreement.csv")
display(age_agreement.round(5))

age_gap_correlations = correlate_age_accelerations(
    master,
    retinal_gap_column="retinal_age_gap",
    epigenetic_acceleration_columns={
        "epigenetic_age_acceleration_difference": (
            "Released DNAm age-acceleration difference"
        ),
        "epigenetic_age_acceleration_residual": (
            "Epigenetic age-acceleration residual"
        ),
        "epigenetic_ieaa": "IEAA",
        "epigenetic_eeaa": "EEAA",
    },
    group_column="glaucoma_label",
    minimum_pairs=minimum_per_group,
)
write_aggregate(age_gap_correlations, "retinal_epigenetic_gap_correlations.csv")
display(age_gap_correlations.round(5))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Publication-oriented aggregate figures
# MAGIC
# MAGIC Figures contain no participant identifiers. Identity lines and
# MAGIC Bland--Altman limits are descriptive; inferential results remain in the
# MAGIC participant-level tables above.

# COMMAND ----------
binary_forest = primary_questionnaire_results[
    (primary_questionnaire_results["variable_type"] == "binary")
    & (primary_questionnaire_results["adjusted_status"] == "ok")
    & primary_questionnaire_results["adjusted_odds_ratio"].notna()
    & np.isfinite(primary_questionnaire_results["adjusted_odds_ratio"])
    & np.isfinite(primary_questionnaire_results["adjusted_odds_ratio_ci_low"])
    & np.isfinite(primary_questionnaire_results["adjusted_odds_ratio_ci_high"])
].sort_values("adjusted_odds_ratio")
if not binary_forest.empty:
    figure_height = max(6, 0.35 * len(binary_forest))
    fig, axis = plt.subplots(figsize=(9, figure_height))
    y = np.arange(len(binary_forest))
    odds = binary_forest["adjusted_odds_ratio"].to_numpy(float)
    low = binary_forest["adjusted_odds_ratio_ci_low"].to_numpy(float)
    high = binary_forest["adjusted_odds_ratio_ci_high"].to_numpy(float)
    colors = np.where(
        binary_forest["fdr_q_value"].to_numpy(float) < fdr_alpha,
        "#C44E52",
        "#4C72B0",
    )
    for index in range(len(binary_forest)):
        axis.errorbar(
            odds[index],
            y[index],
            xerr=[[odds[index] - low[index]], [high[index] - odds[index]]],
            fmt="o",
            color=colors[index],
            capsize=3,
        )
    axis.axvline(1, color="black", linestyle="--", linewidth=1)
    axis.set_xscale("log")
    axis.set_yticks(y)
    axis.set_yticklabels(binary_forest["label"], fontsize=8)
    axis.set_xlabel("Adjusted odds ratio: glaucoma versus healthy")
    axis.set_title("Primary matched CLSA SAP comparisons")
    fig.tight_layout()
    fig.savefig(
        figure_root / "questionnaire_binary_adjusted_odds_ratios.png",
        dpi=220,
        bbox_inches="tight",
    )
    display(fig)
    plt.close(fig)

# COMMAND ----------
clock_pairs = [
    ("epigenetic_dnam_age", "Horvath DNAm age"),
    ("epigenetic_hannum_age", "Hannum age"),
]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = {0: "#4C78A8", 1: "#E45756"}
labels = {0: "Healthy", 1: "Glaucoma"}
for axis, (clock, clock_label) in zip(axes, clock_pairs):
    paired = master[["retinal_age", clock, "glaucoma_label"]].dropna()
    for group_value in (0, 1):
        group = paired[paired["glaucoma_label"] == group_value]
        axis.scatter(
            group[clock],
            group["retinal_age"],
            s=24,
            alpha=0.65,
            color=colors[group_value],
            label=labels[group_value],
        )
    if len(paired):
        lower = float(min(paired[clock].min(), paired["retinal_age"].min()))
        upper = float(max(paired[clock].max(), paired["retinal_age"].max()))
        axis.plot([lower, upper], [lower, upper], "k--", linewidth=1)
    axis.set_xlabel(f"{clock_label} (years)")
    axis.set_ylabel("RETFound retinal age (years)")
    axis.set_title(f"Retinal age versus {clock_label}")
    axis.legend(frameon=False)
fig.tight_layout()
fig.savefig(
    figure_root / "retinal_vs_epigenetic_clock_scatter.png",
    dpi=220,
    bbox_inches="tight",
)
display(fig)
plt.close(fig)

# COMMAND ----------
bland_altman_pairs = [
    ("chronological_age", "Chronological age"),
    ("epigenetic_dnam_age", "Horvath DNAm age"),
    ("epigenetic_hannum_age", "Hannum age"),
]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for axis, (comparator, comparator_label) in zip(axes, bland_altman_pairs):
    paired = master[["retinal_age", comparator, "glaucoma_label"]].dropna()
    mean_age = (paired["retinal_age"] + paired[comparator]) / 2
    difference = paired["retinal_age"] - paired[comparator]
    for group_value in (0, 1):
        selected = paired["glaucoma_label"] == group_value
        axis.scatter(
            mean_age[selected],
            difference[selected],
            s=20,
            alpha=0.55,
            color=colors[group_value],
            label=labels[group_value],
        )
    if len(difference) >= 2:
        difference_mean = float(difference.mean())
        difference_sd = float(difference.std(ddof=1))
        axis.axhline(difference_mean, color="black", linewidth=1.5)
        axis.axhline(
            difference_mean + 1.96 * difference_sd,
            color="black",
            linestyle="--",
            linewidth=1,
        )
        axis.axhline(
            difference_mean - 1.96 * difference_sd,
            color="black",
            linestyle="--",
            linewidth=1,
        )
    axis.set_xlabel("Mean of two age measures (years)")
    axis.set_ylabel("Retinal age minus comparator (years)")
    axis.set_title(comparator_label)
axes[0].legend(frameon=False)
fig.suptitle("Bland–Altman assessment of RETFound retinal age", y=1.02)
fig.tight_layout()
fig.savefig(
    figure_root / "retinal_age_bland_altman.png",
    dpi=220,
    bbox_inches="tight",
)
display(fig)
plt.close(fig)

# COMMAND ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
gap_groups = [
    master.loc[master["glaucoma_label"] == value, "retinal_age_gap"].dropna()
    for value in (0, 1)
]
axes[0].boxplot(gap_groups, labels=["Healthy", "Glaucoma"], showfliers=False)
axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
axes[0].set_ylabel("RETFound retinal-age gap (years)")
axes[0].set_title("Retinal-age gap by matched group")

correlation_plot = age_gap_correlations[
    age_gap_correlations["analysis_status"] == "ok"
].pivot(
    index="epigenetic_measure",
    columns="stratum",
    values="pearson_r",
)
if not correlation_plot.empty:
    image = axes[1].imshow(
        correlation_plot.to_numpy(float),
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto",
    )
    axes[1].set_xticks(np.arange(len(correlation_plot.columns)))
    axes[1].set_xticklabels(correlation_plot.columns)
    axes[1].set_yticks(np.arange(len(correlation_plot.index)))
    axes[1].set_yticklabels(
        [value.replace("epigenetic_", "").replace("_", " ") for value in correlation_plot.index],
        fontsize=8,
    )
    axes[1].set_title("Retinal-gap versus epigenetic-acceleration correlation")
    fig.colorbar(image, ax=axes[1], fraction=0.046, label="Pearson r")
else:
    axes[1].text(0.5, 0.5, "Insufficient paired epigenetic data", ha="center")
    axes[1].axis("off")
fig.tight_layout()
fig.savefig(
    figure_root / "retinal_and_epigenetic_age_acceleration.png",
    dpi=220,
    bbox_inches="tight",
)
display(fig)
plt.close(fig)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Reproducible interpretation summary

# COMMAND ----------
significant_aging = aging_group_results[
    aging_group_results.get("fdr_q_value", pd.Series(dtype=float)) < fdr_alpha
]
summary = {
    "analysis": "matched_CLSA_glaucoma_questionnaire_epigenetic_aging",
    "n_participants": int(master["participant_id"].nunique()),
    "n_healthy": int((master["glaucoma_label"] == 0).sum()),
    "n_glaucoma": int((master["glaucoma_label"] == 1).sum()),
    "n_match_sets": int(master["match_set_id"].nunique()),
    "notebook09_participant_coverage": {
        "observed": int(notebook09_participants),
        "expected": int(expected_participants),
    },
    "questionnaire_linkage": {
        "exact_participant_visit": True,
        "n_linked": int(master["sex_at_birth"].notna().sum()),
        "inference": "age/sex/visit adjusted with matched-set clustered covariance",
        "survey_design_note": (
            "CLSA analytic weights retained for sensitivity work; no documented "
            "PSU was available, so full complex-survey inference was not claimed"
        ),
    },
    "sap_questionnaire_scope": {
        "primary_derived_or_composite": list(PRIMARY_SAP_SPECIFICATIONS),
        "secondary_raw_or_component": list(
            SECONDARY_SAP_COMPONENT_SPECIFICATIONS
        ),
        "covariates_not_outcomes": ["age_at_fundus_years", "sex_at_birth"],
        "survey_design_not_outcomes": ["analytic_weight", "sampling_strata"],
        "not_validly_derivable": [
            "frailty",
            "multimorbidity_32_condition_count",
            "survey_psu",
        ],
    },
    "epigenetic_scope": {
        "measurement_visit": "BL",
        "n_horvath_dnam_age": int(master["epigenetic_dnam_age"].notna().sum()),
        "n_hannum_age": int(master["epigenetic_hannum_age"].notna().sum()),
        "raw_DNA_recalculated": False,
        "primary_acceleration_measure": "epigenetic_age_acceleration_residual",
    },
    "retinal_age_prediction_modes": {
        "healthy": "CLSA_healthy grouped out-of-fold",
        "glaucoma": "CLSA_healthy frozen-model application",
    },
    "multiplicity": {
        "method": "Benjamini-Hochberg",
        "alpha": fdr_alpha,
        "questionnaire_fdr_scopes": [
            "primary_derived_or_composite",
            "secondary_raw_or_component",
        ],
        "significant_primary_questionnaire_variables": (
            significant_questionnaire.loc[
                significant_questionnaire["analysis_family"]
                == "primary_derived_or_composite",
                "variable",
            ].tolist()
        ),
        "significant_secondary_questionnaire_variables": (
            significant_questionnaire.loc[
                significant_questionnaire["analysis_family"]
                == "secondary_raw_or_component",
                "variable",
            ].tolist()
        ),
        "significant_aging_variables": significant_aging["variable"].tolist(),
    },
    "interpretation_limits": [
        "CLSA glaucoma is physician-diagnosed self-report, not adjudicated severity.",
        "Retinal and epigenetic clocks are differently trained estimators and are not interchangeable measures.",
        "Epigenetic analyses are a smaller baseline-only subset and require explicit missingness review.",
        "Questionnaire comparisons are observational and do not establish causal glaucoma risk factors.",
        "Anatomic attribution metrics from notebook 09 are attached for secondary analyses but are not treated as questionnaire outcomes.",
    ],
    "outputs": {
        "private_master": str(
            private_root / "matched_questionnaire_epigenetic_master_private.parquet"
        ),
        "questionnaire_results": str(
            statistics_root / "questionnaire_group_comparisons.csv"
        ),
        "questionnaire_descriptives": str(
            statistics_root / "questionnaire_group_descriptives.csv"
        ),
        "sap_coverage_audit": str(
            statistics_root / "sap_questionnaire_coverage_audit.csv"
        ),
        "questionnaire_primary_results": str(
            statistics_root / "questionnaire_primary_group_comparisons.csv"
        ),
        "questionnaire_secondary_results": str(
            statistics_root
            / "questionnaire_secondary_component_comparisons.csv"
        ),
        "aging_group_results": str(
            statistics_root / "aging_biomarker_group_comparisons.csv"
        ),
        "age_agreement": str(
            statistics_root
            / "retinal_vs_chronological_epigenetic_agreement.csv"
        ),
        "age_gap_correlations": str(
            statistics_root / "retinal_epigenetic_gap_correlations.csv"
        ),
    },
}
write_json(summary, output_root / "QUESTIONNAIRE_EPIGENETIC_AGING_SUMMARY.json")
print(json.dumps(summary, indent=2, default=str))
print("Notebook 10 complete")
