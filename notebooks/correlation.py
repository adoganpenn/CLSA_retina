# Databricks notebook source
# MAGIC %md
# MAGIC # Correlation: SAP phenotypes, retinal age, and RETFound vectors
# MAGIC
# MAGIC This notebook has two prespecified sections:
# MAGIC
# MAGIC 1. Analyze retinal age and retinal-age gap across the principal SAP
# MAGIC    measures, with participant-clustered adjusted regressions.
# MAGIC 2. Test whether the 1,024-dimensional RETFound representation predicts
# MAGIC    selected comorbidities using participant-disjoint cross-validation.
# MAGIC
# MAGIC Both eyes are averaged to one participant-visit record before any
# MAGIC inference. BL and F1 records from the same participant are assigned to
# MAGIC the same model fold. The comorbidity models are exploratory prediction
# MAGIC analyses, not diagnostic models or evidence of causality.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Environment
# MAGIC
# MAGIC Run notebook 02's environment setup first. If this is a fresh runtime:
# MAGIC
# MAGIC ```python
# MAGIC %pip install -r /Workspace/Users/ad0038@pennmedicine.upenn.edu/CLSA/CLSA_retina/requirements-retfound.txt
# MAGIC dbutils.library.restartPython()
# MAGIC ```

# COMMAND ----------
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

# COMMAND ----------
# Reproducible configuration is fixed in the next cell.

# COMMAND ----------
from pathlib import Path

repo_root = "/Workspace/Users/ad0038@pennmedicine.upenn.edu/CLSA/CLSA_retina"
derived_root = (
    "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/derived/"
    "clsa_retinal_aging"
)
sap_path = f"{derived_root}/sap_questionnaire_visit"
embedding_path = (
    f"{derived_root}/fundus_retfound/02_embeddings/retfound_embeddings_delta"
)
prediction_path_requested = ""
output_root = Path(f"{derived_root}/correlation")
should_run_retinal_age = True
should_run_models = True
expected_embedding_dim = 1024
n_splits = 5
pca_components = 64
minimum_outcome_positives = 50
minimum_outcome_negatives = 50
random_seed = 20260727

if expected_embedding_dim != 1024:
    raise ValueError("This analysis expects the 1,024-dimensional RETFound vector.")
if n_splits < 2:
    raise ValueError("n_splits must be at least 2.")
if pca_components < 1:
    raise ValueError("pca_components must be at least 1.")

module_root = Path(repo_root) / "src"
if str(module_root) not in sys.path:
    sys.path.insert(0, str(module_root))

from correlation_analysis import (  # noqa: E402
    ComorbidityModelConfig,
    cross_validate_comorbidity_models,
    fit_retinal_age_associations,
    summarize_retinal_age_strata,
)

# COMMAND ----------
def databricks_path_exists(path: str) -> bool:
    try:
        dbutils.fs.ls(path)
        return True
    except Exception:
        return False


def require_columns(frame, columns, label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def normalize_visit(column):
    return (
        F.when(F.upper(column).isin("F1", "FUP1"), F.lit("F1"))
        .when(F.upper(column) == "BL", F.lit("BL"))
        .otherwise(F.lit(None).cast("string"))
    )


def write_delta(frame, path: str, partition_by=()) -> None:
    writer = (
        frame.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
    )
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.save(path)
    print("Wrote:", path)


def write_pandas_outputs(frame: pd.DataFrame, stem: str) -> None:
    if frame.empty:
        print(f"No rows to write for {stem}.")
        return
    clean = frame.replace({np.nan: None})
    csv_path = output_root / f"{stem}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(csv_path, index=False)
    write_delta(spark.createDataFrame(clean), str(output_root / stem))


for required_path, label in (
    (sap_path, "SAP participant-visit table"),
    (embedding_path, "RETFound embedding table"),
):
    if not databricks_path_exists(required_path):
        raise FileNotFoundError(f"{label} was not found: {required_path}")

output_root.mkdir(parents=True, exist_ok=True)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Load and validate SAP phenotypes

# COMMAND ----------
sap = spark.read.format("delta").load(sap_path)
require_columns(
    sap,
    ["participant_id", "visit", "age_at_fundus_years", "sex_at_birth"],
    "SAP participant-visit table",
)
sap = (
    sap.withColumn("participant_id", F.col("participant_id").cast("string"))
    .withColumn("visit", normalize_visit(F.col("visit")))
    .filter(F.col("participant_id").isNotNull() & F.col("visit").isNotNull())
)
duplicate_sap = sap.groupBy("participant_id", "visit").count().filter(
    F.col("count") > 1
)
if duplicate_sap.limit(1).count():
    raise ValueError("SAP input is not unique on participant_id and visit.")

COMORBIDITY_OUTCOMES = [
    "diabetes",
    "hypertension",
    "heart_disease",
    "stroke",
    "oa_hand",
    "oa_hip",
    "oa_knee",
    "osteoporosis",
    "asthma",
    "copd",
    "cancer",
    "low_back_pain",
    "arthritis_any",
    "asthma_or_copd",
    "depression_cesd10",
]

print("SAP participant-visits:", sap.count())
display(sap.groupBy("visit").agg(F.count("*").alias("participant_visits")))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Retinal-age stratification and association analysis
# MAGIC
# MAGIC The preferred result is an out-of-fold retinal-age prediction when the
# MAGIC age head was trained in CLSA. If an externally trained, locked age head
# MAGIC was applied, its prediction file can be supplied explicitly. In either
# MAGIC case, the analysis averages both eyes and recomputes retinal-age gap as
# MAGIC mean predicted retinal age minus visit-matched chronological age.

# COMMAND ----------
def resolve_prediction_path(requested: str) -> tuple[str, str]:
    if requested:
        if not databricks_path_exists(requested):
            raise FileNotFoundError(
                f"Requested retinal-age prediction path was not found: {requested}"
            )
        return requested, "explicit"
    age_root = str(Path(embedding_path).parent.parent / "03_age_model")
    candidates = [
        (
            f"{age_root}/retfound_age_predictions_oof.parquet",
            "CLSA out-of-fold age-head predictions",
        ),
        (
            f"{age_root}/retinal_age_predictions.parquet",
            "externally supplied age-head predictions",
        ),
    ]
    for candidate, label in candidates:
        if databricks_path_exists(candidate):
            return candidate, label
    raise FileNotFoundError(
        "No retinal-age predictions were found. In notebook 02, either supply "
        "existing_age_model or intentionally run train_age_head=true. Checked: "
        + ", ".join(path for path, _ in candidates)
    )


retinal_age_analysis = None
if should_run_retinal_age:
    prediction_path, prediction_provenance = resolve_prediction_path(
        prediction_path_requested
    )
    predictions_raw = spark.read.parquet(prediction_path)
    prediction_column = next(
        (
            name
            for name in (
                "retinal_age_prediction_oof",
                "retinal_age_prediction",
                "retinal_age",
            )
            if name in predictions_raw.columns
        ),
        None,
    )
    raw_prediction_column = next(
        (
            name
            for name in (
                "retinal_age_raw_oof",
                "retinal_age_raw",
            )
            if name in predictions_raw.columns
        ),
        None,
    )
    if prediction_column is None:
        raise ValueError(
            "Prediction file has no recognized retinal-age prediction column."
        )

    embedding_keys = (
        spark.read.format("delta")
        .load(embedding_path)
        .select("image_path", "participant_id", "visit")
        .dropDuplicates(["image_path"])
    )
    predictions = predictions_raw
    if "participant_id" not in predictions.columns or "visit" not in predictions.columns:
        require_columns(predictions, ["image_path"], "Retinal-age predictions")
        predictions = predictions.join(embedding_keys, "image_path", "left")
    require_columns(
        predictions,
        ["participant_id", "visit", prediction_column],
        "Retinal-age predictions",
    )
    prediction_selections = [
        F.col("participant_id").cast("string").alias("participant_id"),
        normalize_visit(F.col("visit")).alias("visit"),
        F.col(prediction_column).cast("double").alias("retinal_age_image"),
    ]
    if raw_prediction_column:
        prediction_selections.append(
            F.col(raw_prediction_column)
            .cast("double")
            .alias("retinal_age_raw_image")
        )
    else:
        prediction_selections.append(
            F.lit(None).cast("double").alias("retinal_age_raw_image")
        )
    if "image_path" in predictions.columns:
        prediction_selections.append(F.col("image_path"))
    else:
        prediction_selections.append(F.lit(None).cast("string").alias("image_path"))

    predictions = predictions.select(*prediction_selections).filter(
        F.col("participant_id").isNotNull()
        & F.col("visit").isNotNull()
        & F.col("retinal_age_image").isNotNull()
    )
    retinal_age_visit = predictions.groupBy("participant_id", "visit").agg(
        F.avg("retinal_age_image").alias("retinal_age"),
        F.avg("retinal_age_raw_image").alias("retinal_age_raw"),
        F.count("*").alias("n_retinal_age_images"),
        F.countDistinct("image_path").alias("n_distinct_retinal_age_images"),
    )
    retinal_age_analysis = (
        retinal_age_visit.join(sap, ["participant_id", "visit"], "inner")
        .withColumn(
            "retinal_age_gap",
            F.col("retinal_age") - F.col("age_at_fundus_years"),
        )
        .withColumn(
            "age_band",
            F.when(F.col("age_at_fundus_years") < 55, F.lit("45-54"))
            .when(F.col("age_at_fundus_years") < 65, F.lit("55-64"))
            .when(F.col("age_at_fundus_years") < 75, F.lit("65-74"))
            .otherwise(F.lit("75+")),
        )
        .withColumn(
            "multimorbidity_selected_group",
            F.when(
                F.col("multimorbidity_selected_count").isNull(),
                F.lit(None).cast("string"),
            )
            .when(F.col("multimorbidity_selected_count") == 0, F.lit("0"))
            .when(F.col("multimorbidity_selected_count") == 1, F.lit("1"))
            .when(F.col("multimorbidity_selected_count") == 2, F.lit("2"))
            .otherwise(F.lit("3+")),
        )
        .withColumn("prediction_provenance", F.lit(prediction_provenance))
        .withColumn("prediction_path", F.lit(prediction_path))
    )
    if not retinal_age_analysis.limit(1).count():
        raise ValueError(
            "Retinal-age predictions did not match SAP participant visits."
        )
    write_delta(
        retinal_age_analysis,
        str(output_root / "01_retinal_age/analysis_dataset"),
        partition_by=("visit",),
    )

    STRATIFIERS = [
        "visit",
        "age_band",
        "sex_at_birth",
        "ethnicity_spirometry",
        "self_reported_vision",
        "visual_impairment_self_report",
        "visual_impairment_acuity",
        "smoking_status",
        "education_level_sap_harmonized",
        "household_income_band",
        "married_or_partnered",
        "multimorbidity_selected_group",
        "hearing_noise",
        "hearing_aid",
        "social_outside_household_at_least_weekly",
        "social_religious_at_least_weekly",
        "social_education_culture_at_least_weekly",
        "social_club_at_least_weekly",
        "social_association_at_least_weekly",
        "social_other_at_least_weekly",
        "social_any_at_least_weekly",
        "adl_class",
        "self_rated_healthy_aging",
        *COMORBIDITY_OUTCOMES,
    ]
    CONTINUOUS_EXPOSURES = [
        "visual_acuity_left",
        "visual_acuity_right",
        "visual_acuity_both",
        "visual_acuity_better_eye",
        "cesd10_score",
        "multimorbidity_selected_count",
        "epigenetic_dnam_age",
        "epigenetic_age_acceleration_difference",
        "epigenetic_age_acceleration_residual",
        "epigenetic_ieaa",
        "epigenetic_eeaa",
        "epigenetic_hannum_age",
        "frailty",
    ]
    CATEGORICAL_EXPOSURES = [
        variable for variable in STRATIFIERS if variable not in {"visit", "age_band"}
    ]
    analysis_columns = list(
        dict.fromkeys(
            [
                "participant_id",
                "visit",
                "age_at_fundus_years",
                "sex_at_birth",
                "retinal_age",
                "retinal_age_gap",
                *STRATIFIERS,
                *CONTINUOUS_EXPOSURES,
            ]
        )
    )
    available_analysis_columns = [
        column for column in analysis_columns if column in retinal_age_analysis.columns
    ]
    retinal_age_pandas = retinal_age_analysis.select(
        *available_analysis_columns
    ).toPandas()
    strata = summarize_retinal_age_strata(
        retinal_age_pandas,
        STRATIFIERS,
    )
    exposures = {
        **{name: "continuous" for name in CONTINUOUS_EXPOSURES},
        **{name: "categorical" for name in CATEGORICAL_EXPOSURES},
    }
    associations = fit_retinal_age_associations(
        retinal_age_pandas,
        exposures,
        minimum_records=100,
    )
    write_pandas_outputs(strata, "01_retinal_age/stratified_summary")
    write_pandas_outputs(associations, "01_retinal_age/adjusted_associations")
    display(spark.createDataFrame(strata.replace({np.nan: None})))
    associations_display = spark.createDataFrame(
        associations.replace({np.nan: None})
    )
    if "p_value_fdr_bh" in associations_display.columns:
        associations_display = associations_display.orderBy(
            F.col("p_value_fdr_bh").asc_nulls_last()
        )
    display(associations_display)
    print("Retinal-age prediction source:", prediction_path)
else:
    print("Retinal-age section disabled.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Predict comorbidities from RETFound vectors
# MAGIC
# MAGIC Models are evaluated with out-of-fold probabilities only. Three model
# MAGIC families are compared:
# MAGIC
# MAGIC - `clinical`: chronological age, sex, and visit only
# MAGIC - `retfound_embedding`: fold-fitted PCA of the 1,024 RETFound features
# MAGIC - `combined`: RETFound features plus age, sex, and visit
# MAGIC
# MAGIC Comparing `retfound_embedding` and `combined` with `clinical` shows
# MAGIC whether the image representation adds predictive information beyond a
# MAGIC small demographic baseline. PCA, scaling, and logistic regression are
# MAGIC refit inside every participant-grouped outer fold.

# COMMAND ----------
if should_run_models:
    embeddings = spark.read.format("delta").load(embedding_path)
    require_columns(
        embeddings,
        ["participant_id", "visit", "embedding", "embedding_dim"],
        "RETFound embeddings",
    )
    embeddings = (
        embeddings.withColumn(
            "participant_id", F.col("participant_id").cast("string")
        )
        .withColumn("visit", normalize_visit(F.col("visit")))
        .filter(
            F.col("participant_id").isNotNull()
            & F.col("visit").isNotNull()
            & F.col("embedding").isNotNull()
            & (F.col("embedding_dim") == expected_embedding_dim)
            & (F.size("embedding") == expected_embedding_dim)
        )
    )
    embedding_sum_expression = (
        "aggregate(collect_list(embedding), "
        f"array_repeat(CAST(0.0 AS FLOAT), {expected_embedding_dim}), "
        "(acc, x) -> zip_with(acc, x, "
        "(a, b) -> CAST(a + b AS FLOAT)))"
    )
    participant_visit_embeddings = (
        embeddings.groupBy("participant_id", "visit")
        .agg(
            F.count("*").alias("n_embedding_images"),
            F.expr(embedding_sum_expression).alias("embedding_sum"),
        )
        .withColumn(
            "embedding",
            F.expr(
                "transform(embedding_sum, x -> "
                "CAST(x / n_embedding_images AS FLOAT))"
            ),
        )
        .drop("embedding_sum")
        .withColumn("embedding_dim", F.lit(expected_embedding_dim))
        .join(sap, ["participant_id", "visit"], "inner")
        .filter(F.col("age_at_fundus_years").isNotNull())
    )
    if not participant_visit_embeddings.limit(1).count():
        raise ValueError("No RETFound vectors matched the SAP participant visits.")
    write_delta(
        participant_visit_embeddings,
        str(output_root / "02_comorbidity/participant_visit_embeddings"),
        partition_by=("visit",),
    )

    model_columns = [
        "participant_id",
        "visit",
        "age_at_fundus_years",
        "sex_at_birth",
        "embedding",
        *[
            outcome
            for outcome in COMORBIDITY_OUTCOMES
            if outcome in participant_visit_embeddings.columns
        ],
    ]
    model_frame = participant_visit_embeddings.select(*model_columns).toPandas()
    model_frame["embedding"] = model_frame["embedding"].map(
        lambda value: np.asarray(value, dtype=np.float32)
    )
    metrics, oof_predictions, outcome_availability = (
        cross_validate_comorbidity_models(
            model_frame,
            COMORBIDITY_OUTCOMES,
            expected_embedding_dim=expected_embedding_dim,
            config=ComorbidityModelConfig(
                n_splits=n_splits,
                pca_components=pca_components,
                minimum_positive_records=minimum_outcome_positives,
                minimum_negative_records=minimum_outcome_negatives,
                random_seed=random_seed,
            ),
        )
    )
    write_pandas_outputs(metrics, "02_comorbidity/model_metrics")
    write_pandas_outputs(oof_predictions, "02_comorbidity/oof_predictions")
    write_pandas_outputs(
        outcome_availability,
        "02_comorbidity/outcome_availability",
    )
    if not metrics.empty:
        pooled = metrics.loc[metrics["evaluation_scope"] == "pooled_oof"]
        display(
            spark.createDataFrame(pooled.replace({np.nan: None})).orderBy(
                "outcome", "model_family"
            )
        )
    display(spark.createDataFrame(outcome_availability.replace({np.nan: None})))
else:
    print("Comorbidity prediction section disabled.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Interpretation and durable outputs
# MAGIC
# MAGIC Retinal-age coefficients are differences in retinal-age gap (years),
# MAGIC adjusted for chronological age, sex, and visit. The BH-FDR column is the
# MAGIC primary multiplicity-adjusted result. Survey weights are not applied
# MAGIC because notebook 03 documents that a valid released PSU/longitudinal
# MAGIC design specification is unavailable.
# MAGIC
# MAGIC Comorbidity AUROC and average precision must be read from the
# MAGIC `pooled_oof` rows. Average precision should always be interpreted beside
# MAGIC prevalence. Use the clinical comparator to assess incremental value;
# MAGIC do not interpret prediction as causality or use these models clinically
# MAGIC without external validation and calibration.
# MAGIC
# MAGIC Outputs under `output_root`:
# MAGIC
# MAGIC - `01_retinal_age/analysis_dataset`
# MAGIC - `01_retinal_age/stratified_summary`
# MAGIC - `01_retinal_age/adjusted_associations`
# MAGIC - `02_comorbidity/participant_visit_embeddings`
# MAGIC - `02_comorbidity/model_metrics`
# MAGIC - `02_comorbidity/oof_predictions`
# MAGIC - `02_comorbidity/outcome_availability`

# COMMAND ----------
run_metadata = {
    "sap_path": sap_path,
    "embedding_path": embedding_path,
    "retinal_age_prediction_path_requested": prediction_path_requested,
    "output_root": str(output_root),
    "expected_embedding_dim": expected_embedding_dim,
    "n_splits": n_splits,
    "pca_components": pca_components,
    "minimum_outcome_positives": minimum_outcome_positives,
    "minimum_outcome_negatives": minimum_outcome_negatives,
    "random_seed": random_seed,
    "participant_visit_aggregation": "mean across retinal images",
    "cross_validation_group": "participant_id",
    "survey_weighting": "not applied; valid PSU/longitudinal design unavailable",
}
metadata_path = output_root / "correlation_run_metadata.json"
metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
print("Run metadata:", metadata_path)
