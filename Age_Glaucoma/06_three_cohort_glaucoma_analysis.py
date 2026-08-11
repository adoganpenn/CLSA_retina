# Databricks notebook source
# MAGIC %md
# MAGIC # Three-cohort glaucoma retinal-age analysis
# MAGIC
# MAGIC This notebook triangulates three participant-level cohorts:
# MAGIC
# MAGIC 1. screen-negative CLSA healthy controls;
# MAGIC 2. CLSA participants reporting physician-diagnosed glaucoma with a
# MAGIC    complete negative screen for the other prespecified ocular diseases;
# MAGIC 3. the Zeiss glaucoma-source cohort.
# MAGIC
# MAGIC The **primary disease comparison** is CLSA glaucoma versus CLSA healthy,
# MAGIC because both groups share the CLSA acquisition and RETFound pipeline.
# MAGIC Zeiss comparisons are transportability analyses. The harmonized Zeiss
# MAGIC results are sensitivity analyses under an additive source-effect
# MAGIC assumption; without healthy Zeiss controls, a source-by-glaucoma
# MAGIC interaction cannot be estimated.
# MAGIC
# MAGIC “Glaucoma-only” means no concurrent released positive screen for retinal
# MAGIC detachment, cataract, macular degeneration, or (at F1) diabetic
# MAGIC retinopathy. Systemic conditions are adjusted rather than automatically
# MAGIC excluded; an optional widget can require no selected systemic
# MAGIC multimorbidity.
# MAGIC
# MAGIC This notebook **never recalculates RETFound embeddings**. It first loads
# MAGIC the reusable CLSA Delta cache. If that cache is absent, it reads the
# MAGIC already-completed notebook 02 Parquet batches and consolidates only the
# MAGIC stable vector columns into the cache. Missing vectors stop the analysis
# MAGIC and direct the user back to notebook 02; no model inference is started.

# COMMAND ----------
from pathlib import Path
import hashlib
import importlib
import json
import sys

import joblib
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
dbutils.widgets.text("clsa_embedding_cache_path", "")
dbutils.widgets.text("clsa_quality_path", "")
dbutils.widgets.text("ocular_screen_path", "")
dbutils.widgets.text("sap_path", "")
dbutils.widgets.text("healthy_images_path", "")
dbutils.widgets.text("healthy_oof_path", "")
dbutils.widgets.text("healthy_model_path", "")
dbutils.widgets.text("zeiss_embeddings_path", "")
dbutils.widgets.dropdown(
    "strict_never_glaucoma_controls", "true", ["true", "false"]
)
dbutils.widgets.dropdown(
    "exclude_age_model_training_overlap", "true", ["true", "false"]
)
dbutils.widgets.dropdown(
    "exclude_selected_systemic_comorbidity", "false", ["false", "true"]
)
dbutils.widgets.dropdown(
    "require_complete_glaucoma_embeddings", "true", ["true", "false"]
)
dbutils.widgets.text("age_caliper_years", "1.0")
dbutils.widgets.text("bootstrap_repetitions", "5000")
dbutils.widgets.dropdown(
    "harmonization_mode", "location_scale", ["location_scale", "location"]
)
dbutils.widgets.text("harmonization_ridge", "0.000001")
dbutils.widgets.text("domain_classifier_max_per_source", "5000")

# COMMAND ----------
repo_root = Path(dbutils.widgets.get("repo_root").strip())
age_glaucoma_root = Path(
    dbutils.widgets.get("age_glaucoma_output_root").strip()
)
derived_root = age_glaucoma_root.parent


def configured_path(widget_name, default):
    value = dbutils.widgets.get(widget_name).strip()
    return value or str(default)


clsa_embedding_cache_path = configured_path(
    "clsa_embedding_cache_path",
    age_glaucoma_root / "00_inputs" / "clsa_embeddings_delta",
)
clsa_quality_path = configured_path(
    "clsa_quality_path",
    derived_root / "fundus_retfound" / "01_quality" / "fundus_quality_manifest.parquet",
)
ocular_screen_path = configured_path(
    "ocular_screen_path",
    age_glaucoma_root / "03_clsa_controls" / "ocular_screen_delta",
)
sap_path = configured_path(
    "sap_path",
    derived_root / "sap_questionnaire_visit",
)
healthy_images_path = configured_path(
    "healthy_images_path",
    age_glaucoma_root / "03_clsa_controls" / "eligible_images_delta",
)
healthy_oof_path = configured_path(
    "healthy_oof_path",
    age_glaucoma_root
    / "07_retinal_age_inference"
    / "CLSA_healthy_participant_visit_oof.parquet",
)
healthy_model_path = configured_path(
    "healthy_model_path",
    age_glaucoma_root / "06_CLSA_healthy_model" / "CLSA_healthy.joblib",
)
zeiss_embeddings_path = configured_path(
    "zeiss_embeddings_path",
    age_glaucoma_root
    / "01_zeiss_source_cohort"
    / "zeiss_embedded_images.parquet",
)

strict_never_glaucoma_controls = (
    dbutils.widgets.get("strict_never_glaucoma_controls") == "true"
)
exclude_age_model_training_overlap = (
    dbutils.widgets.get("exclude_age_model_training_overlap") == "true"
)
exclude_selected_systemic_comorbidity = (
    dbutils.widgets.get("exclude_selected_systemic_comorbidity") == "true"
)
require_complete_glaucoma_embeddings = (
    dbutils.widgets.get("require_complete_glaucoma_embeddings") == "true"
)
age_caliper_years = float(dbutils.widgets.get("age_caliper_years"))
bootstrap_repetitions = int(dbutils.widgets.get("bootstrap_repetitions"))
harmonization_mode = dbutils.widgets.get("harmonization_mode")
harmonization_ridge = float(dbutils.widgets.get("harmonization_ridge"))
domain_classifier_max_per_source = int(
    dbutils.widgets.get("domain_classifier_max_per_source")
)

if age_caliper_years < 0:
    raise ValueError("age_caliper_years cannot be negative")
if bootstrap_repetitions < 500:
    raise ValueError("bootstrap_repetitions must be at least 500")
if harmonization_ridge < 0:
    raise ValueError("harmonization_ridge cannot be negative")
if domain_classifier_max_per_source < 100:
    raise ValueError("domain_classifier_max_per_source must be at least 100")

module_root = repo_root / "src"
if not module_root.exists():
    raise FileNotFoundError(f"Repository source directory not found: {module_root}")
if str(module_root) not in sys.path:
    sys.path.insert(0, str(module_root))

import fundus_retfound_pipeline as _fundus_pipeline  # noqa: E402
import three_cohort_glaucoma as _three_cohort  # noqa: E402

_fundus_pipeline = importlib.reload(_fundus_pipeline)
_three_cohort = importlib.reload(_three_cohort)

from age_glaucoma_model import prediction_summary  # noqa: E402
from fundus_retfound_pipeline import (  # noqa: E402
    load_age_head,
    predict_retinal_age,
    write_frame,
    write_json,
)
from three_cohort_glaucoma import (  # noqa: E402
    adjusted_group_effect,
    aggregate_embedding_rows,
    apply_source_harmonizer,
    canonical_sex,
    cross_validated_domain_auc,
    embedding_shift_summary,
    fit_additive_source_harmonizer,
    greedy_match,
    paired_outcome_effect,
    select_representative_visit,
    validate_embedding_frame,
)

print("Loaded fundus pipeline:", _fundus_pipeline.__file__)
print("Loaded three-cohort helper:", _three_cohort.__file__)

# COMMAND ----------
analysis_root = age_glaucoma_root / "11_three_cohort_glaucoma"
cohort_root = analysis_root / "01_cohort"
prediction_root = analysis_root / "02_predictions"
comparison_root = analysis_root / "03_comparisons"
harmonization_root = analysis_root / "04_harmonization"
figure_root = analysis_root / "05_figures"
for path in (
    cohort_root,
    prediction_root,
    comparison_root,
    harmonization_root,
    figure_root,
):
    path.mkdir(parents=True, exist_ok=True)


def databricks_path_exists(path):
    if Path(path).exists():
        return True
    try:
        dbutils.fs.ls(str(path))
        return True
    except Exception:
        return False


required_paths = {
    "CLSA quality manifest": clsa_quality_path,
    "CLSA ocular screen": ocular_screen_path,
    "SAP participant-visit table": sap_path,
    "CLSA healthy images": healthy_images_path,
    "CLSA healthy OOF predictions": healthy_oof_path,
    "CLSA_healthy frozen model": healthy_model_path,
    "Zeiss embeddings": zeiss_embeddings_path,
}
missing_paths = [
    f"{label}: {path}"
    for label, path in required_paths.items()
    if not databricks_path_exists(path)
]
if missing_paths:
    raise FileNotFoundError(
        "Required prior-stage outputs are missing:\n- " + "\n- ".join(missing_paths)
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Derive a visit-matched CLSA glaucoma-only ocular phenotype
# MAGIC
# MAGIC The diagnosis is the released self-report that a doctor diagnosed
# MAGIC glaucoma. It is not chart adjudication and does not encode glaucoma
# MAGIC subtype or severity. All other prespecified ocular fields must be
# MAGIC observed and negative at the same visit as the fundus image.

# COMMAND ----------
screen = spark.read.format("delta").load(ocular_screen_path)
required_screen_columns = {
    "participant_id",
    "visit",
    "glaucoma",
    "retinal_detachment",
    "cataract",
    "macular_degeneration",
    "diabetic_retinopathy",
    "age_at_fundus_years",
    "sex_at_birth",
}
missing_screen_columns = required_screen_columns - set(screen.columns)
if missing_screen_columns:
    raise ValueError(
        "Ocular screen is missing required columns: "
        f"{sorted(missing_screen_columns)}"
    )

screen = (
    screen.withColumn("participant_id", F.trim(F.col("participant_id").cast("string")))
    .withColumn(
        "visit",
        F.when(F.upper(F.col("visit")).isin("F1", "FUP1"), F.lit("F1"))
        .when(F.upper(F.col("visit")) == "BL", F.lit("BL"))
        .otherwise(F.lit(None).cast("string")),
    )
)

sap = spark.read.format("delta").load(sap_path)
sap_optional_columns = [
    column
    for column in (
        "diabetes",
        "hypertension",
        "smoking_status",
        "ethnicity_spirometry",
        "multimorbidity_selected_count",
    )
    if column in sap.columns
]
if sap_optional_columns:
    sap_extra = sap.select(
        F.trim(F.col("participant_id").cast("string")).alias("participant_id"),
        F.when(F.upper(F.col("visit")).isin("F1", "FUP1"), F.lit("F1"))
        .when(F.upper(F.col("visit")) == "BL", F.lit("BL"))
        .otherwise(F.lit(None).cast("string"))
        .alias("visit"),
        *[F.col(column) for column in sap_optional_columns],
    ).dropDuplicates(["participant_id", "visit"])
    screen = screen.join(sap_extra, ["participant_id", "visit"], "left")

other_ocular_fields = [
    "retinal_detachment",
    "cataract",
    "macular_degeneration",
    "diabetic_retinopathy",
]
other_observed = sum(
    F.when(F.col(column).isNotNull(), F.lit(1)).otherwise(F.lit(0))
    for column in other_ocular_fields
)
other_positive = sum(
    F.when(F.col(column) == 1, F.lit(1)).otherwise(F.lit(0))
    for column in other_ocular_fields
)
other_required = F.when(F.col("visit") == "BL", F.lit(3)).otherwise(F.lit(4))

screen = (
    screen.withColumn("other_ocular_observed", other_observed)
    .withColumn("other_ocular_required", other_required)
    .withColumn("other_ocular_positive_count", other_positive)
    .withColumn(
        "glaucoma_only_ocular",
        (F.col("glaucoma") == 1)
        & (F.col("other_ocular_observed") == F.col("other_ocular_required"))
        & (F.col("other_ocular_positive_count") == 0)
        & F.col("age_at_fundus_years").isNotNull(),
    )
)

# At BL, diabetic retinopathy is structurally absent and is not required.
screen = screen.withColumn(
    "glaucoma_only_ocular",
    F.when(
        F.col("visit") == "BL",
        (F.col("glaucoma") == 1)
        & F.col("retinal_detachment").isNotNull()
        & F.col("cataract").isNotNull()
        & F.col("macular_degeneration").isNotNull()
        & (F.col("retinal_detachment") == 0)
        & (F.col("cataract") == 0)
        & (F.col("macular_degeneration") == 0)
        & F.col("age_at_fundus_years").isNotNull(),
    ).otherwise(F.col("glaucoma_only_ocular")),
)

if exclude_selected_systemic_comorbidity:
    if "multimorbidity_selected_count" not in screen.columns:
        raise ValueError(
            "exclude_selected_systemic_comorbidity=true requires "
            "multimorbidity_selected_count in sap_questionnaire_visit"
        )
    screen = screen.withColumn(
        "glaucoma_only_eligible",
        F.col("glaucoma_only_ocular")
        & (F.col("multimorbidity_selected_count") == 0),
    )
else:
    screen = screen.withColumn(
        "glaucoma_only_eligible", F.col("glaucoma_only_ocular")
    )

screen_audit = (
    screen.groupBy("visit")
    .agg(
        F.countDistinct(
            F.when(F.col("glaucoma") == 1, F.col("participant_id"))
        ).alias("glaucoma_positive_participants"),
        F.countDistinct(
            F.when(F.col("glaucoma_only_ocular"), F.col("participant_id"))
        ).alias("glaucoma_only_ocular_participants"),
        F.countDistinct(
            F.when(F.col("glaucoma_only_eligible"), F.col("participant_id"))
        ).alias("glaucoma_only_eligible_participants"),
    )
    .orderBy("visit")
)
display(screen_audit)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Materialize glaucoma-only CLSA RETFound vectors
# MAGIC
# MAGIC Notebook 02 already embedded every quality-passing CLSA image. Reusing
# MAGIC those exact vectors avoids model-version drift and unnecessary GPU work.
# MAGIC This section verifies that every eligible quality-passing glaucoma image
# MAGIC has a corresponding vector and saves a glaucoma-only Delta table.

# COMMAND ----------
def stable_embedding_projection(frame):
    """Keep only fields with a stable schema across completed embedding batches."""
    required = {"image_path", "participant_id", "visit", "embedding"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "CLSA embedding records are missing stable columns: "
            f"{sorted(missing)}"
        )
    selections = [
        F.col("image_path").cast("string").alias("image_path"),
        F.trim(F.col("participant_id").cast("string")).alias("participant_id"),
        F.col("visit").cast("string").alias("visit"),
        F.col("embedding").cast("array<float>").alias("embedding"),
    ]
    if "eye" in frame.columns:
        selections.append(F.col("eye").cast("string").alias("eye"))
    return frame.select(*selections)


def load_existing_clsa_embeddings(cache_path):
    """Load prior RETFound results; never run image preprocessing or inference."""
    cache_path = str(cache_path).rstrip("/")
    completed_root = str(
        derived_root / "fundus_retfound" / "02_embeddings"
    )
    candidates = [
        (cache_path, "reusable_delta_cache"),
        (
            f"{completed_root}/retfound_embeddings_delta",
            "completed_delta",
        ),
        (
            f"{completed_root}/retfound_embeddings.parquet",
            "completed_parquet",
        ),
    ]
    checked = []
    for candidate, mode in candidates:
        if candidate in checked:
            continue
        checked.append(candidate)
        if not databricks_path_exists(candidate):
            continue
        if databricks_path_exists(f"{candidate}/_delta_log"):
            loaded = spark.read.format("delta").load(candidate)
        else:
            loaded = spark.read.parquet(candidate)
        return stable_embedding_projection(loaded), candidate, mode

    # The completed notebook 02 run may contain only resumable 500-image
    # batches. Reading them is reuse, not RETFound inference. Projecting the
    # stable columns avoids the historical age int/double Parquet mismatch.
    batch_glob = (
        f"{completed_root}/batches/batch_*/retfound_embeddings.parquet"
    )
    checked.append(batch_glob)
    try:
        loaded = stable_embedding_projection(spark.read.parquet(batch_glob))
        if not loaded.columns:
            raise ValueError("Embedding batch glob resolved without a schema")
    except Exception as exc:
        raise FileNotFoundError(
            "No previously calculated CLSA RETFound vectors were found. "
            "This notebook will not recalculate them. Checked:\n- "
            + "\n- ".join(checked)
            + "\nFinish or resume notebook 02, or set clsa_embedding_cache_path "
            "to an existing Delta/Parquet vector dataset."
        ) from exc

    duplicate_paths = (
        loaded.groupBy("image_path").count().filter(F.col("count") > 1)
    )
    if duplicate_paths.limit(1).count():
        raise ValueError(
            "Completed RETFound batches contain duplicate image paths. "
            "Resolve stale/overlapping batches before analysis."
        )
    (
        loaded.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(cache_path)
    )
    return (
        spark.read.format("delta").load(cache_path),
        batch_glob,
        "completed_batches_cached_as_delta",
    )


clsa_embeddings, resolved_embedding_path, embedding_source_mode = (
    load_existing_clsa_embeddings(clsa_embedding_cache_path)
)
print("CLSA embedding source mode:", embedding_source_mode)
print("CLSA embedding source:", resolved_embedding_path)
print("RETFound inference performed by this notebook: false")
quality = spark.read.parquet(clsa_quality_path)
for label, frame, required in (
    (
        "CLSA embedding cache",
        clsa_embeddings,
        {"image_path", "participant_id", "visit", "embedding"},
    ),
    (
        "CLSA quality manifest",
        quality,
        {"image_path", "participant_id", "visit", "quality_pass"},
    ),
):
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")

clsa_embeddings = (
    clsa_embeddings.select(
        F.col("image_path").cast("string").alias("image_path"),
        F.trim(F.col("participant_id").cast("string")).alias("participant_id"),
        F.when(F.upper(F.col("visit")).isin("F1", "FUP1"), F.lit("F1"))
        .when(F.upper(F.col("visit")) == "BL", F.lit("BL"))
        .otherwise(F.lit(None).cast("string"))
        .alias("visit"),
        F.col("embedding").cast("array<float>").alias("embedding"),
        *(
            [F.col("eye").cast("string").alias("eye")]
            if "eye" in clsa_embeddings.columns
            else []
        ),
    )
    .dropDuplicates(["image_path"])
)
quality_columns = [
    column
    for column in (
        "image_path",
        "participant_id",
        "visit",
        "quality_pass",
        "retina_fraction",
        "brightness_mean",
        "contrast_std",
        "gradient_energy",
    )
    if column in quality.columns
]
quality_pass = (
    quality.select(*quality_columns)
    .dropDuplicates(["image_path"])
    .filter(F.col("quality_pass").cast("boolean"))
)
if "participant_id" in quality_pass.columns:
    quality_pass = quality_pass.withColumn(
        "participant_id", F.trim(F.col("participant_id").cast("string"))
    )
if "visit" in quality_pass.columns:
    quality_pass = quality_pass.withColumn(
        "visit",
        F.when(F.upper(F.col("visit")).isin("F1", "FUP1"), F.lit("F1"))
        .when(F.upper(F.col("visit")) == "BL", F.lit("BL"))
        .otherwise(F.lit(None).cast("string")),
    )
glaucoma_keys = screen.filter(F.col("glaucoma_only_eligible")).dropDuplicates(
    ["participant_id", "visit"]
)
quality_with_keys = quality_pass.join(
    glaucoma_keys,
    ["participant_id", "visit"],
    "inner",
)
quality_for_embedding_join = quality_pass.drop(
    *[
        column
        for column in ("participant_id", "visit")
        if column in quality_pass.columns
    ]
)

missing_vector_images = quality_with_keys.select("image_path").join(
    clsa_embeddings.select("image_path"), "image_path", "left_anti"
)
missing_vector_count = missing_vector_images.count()
if missing_vector_count:
    message = (
        f"{missing_vector_count:,} eligible quality-passing CLSA glaucoma images "
        "lack RETFound vectors. Resume notebook 02 before this analysis."
    )
    if require_complete_glaucoma_embeddings:
        raise RuntimeError(message)
    print("WARNING:", message, "They will be excluded.")

screen_metadata_columns = [
    "participant_id",
    "visit",
    "age_at_fundus_years",
    "sex_at_birth",
    "glaucoma",
    "retinal_detachment",
    "cataract",
    "macular_degeneration",
    "diabetic_retinopathy",
    *sap_optional_columns,
]
glaucoma_images_spark = (
    clsa_embeddings.join(quality_for_embedding_join, "image_path", "inner")
    .join(
        glaucoma_keys.select(*screen_metadata_columns),
        ["participant_id", "visit"],
        "inner",
    )
)
if glaucoma_images_spark.groupBy("image_path").count().filter(F.col("count") > 1).limit(1).count():
    raise ValueError("Glaucoma vector materialization created duplicate image paths")

glaucoma_vector_path = cohort_root / "clsa_glaucoma_only_embeddings_delta"
(
    glaucoma_images_spark.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("visit")
    .save(str(glaucoma_vector_path))
)
display(
    glaucoma_images_spark.groupBy("visit").agg(
        F.count("*").alias("images"),
        F.countDistinct("participant_id").alias("participants"),
        F.avg("age_at_fundus_years").alias("mean_age"),
    )
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Build participant-level embedding and prediction tables
# MAGIC
# MAGIC The frozen `CLSA_healthy` age head is loaded from notebook 03. Healthy
# MAGIC CLSA outcomes remain grouped out-of-fold predictions; glaucoma outcomes
# MAGIC use the frozen head. By default, glaucoma participants who contributed a
# MAGIC different, previously screen-negative visit to age-model training are
# MAGIC excluded so the glaucoma application is participant-level held out.

# COMMAND ----------
age_bundle = load_age_head(healthy_model_path)
if age_bundle.get("model_name") != "CLSA_healthy":
    raise ValueError("Configured age model is not the frozen CLSA_healthy model")
if not age_bundle.get("frozen", False):
    raise ValueError("Configured CLSA_healthy model is not marked frozen")


def spark_participant_visit_mean(frame, age_column, metadata_columns):
    embedding_sum = F.aggregate(
        F.collect_list(F.col("embedding").cast("array<double>")),
        F.array_repeat(F.lit(0.0), 1024),
        lambda accumulator, vector: F.zip_with(
            accumulator,
            vector,
            lambda left, right: left + right,
        ),
    )
    aggregations = [
        embedding_sum.alias("embedding_sum"),
        F.count("*").alias("n_images"),
        F.avg(F.col(age_column).cast("double")).alias("age"),
    ]
    for column in metadata_columns:
        if column in frame.columns:
            aggregations.append(F.first(column, ignorenulls=True).alias(column))
    return (
        frame.groupBy("participant_id", "visit")
        .agg(*aggregations)
        .withColumn(
            "embedding",
            F.expr(
                "transform(embedding_sum, value -> "
                "cast(value / n_images as float))"
            ),
        )
        .drop("embedding_sum")
    )


metadata_columns = [
    "sex_at_birth",
    *sap_optional_columns,
]
glaucoma_participant_spark = spark_participant_visit_mean(
    glaucoma_images_spark,
    "age_at_fundus_years",
    metadata_columns,
)

healthy_images_spark = spark.read.format("delta").load(healthy_images_path)
healthy_images_spark = (
    healthy_images_spark.withColumn(
        "participant_id", F.trim(F.col("participant_id").cast("string"))
    )
    .withColumn(
        "visit",
        F.when(F.upper(F.col("visit")).isin("F1", "FUP1"), F.lit("F1"))
        .when(F.upper(F.col("visit")) == "BL", F.lit("BL"))
        .otherwise(F.lit(None).cast("string")),
    )
)
existing_metadata = [
    column
    for column in ("sex_at_birth", *sap_optional_columns)
    if column in healthy_images_spark.columns
]
if existing_metadata:
    healthy_images_spark = healthy_images_spark.drop(*existing_metadata)
healthy_images_spark = healthy_images_spark.join(
        screen.select(
            "participant_id",
            "visit",
            "sex_at_birth",
            *sap_optional_columns,
        ).dropDuplicates(["participant_id", "visit"]),
        ["participant_id", "visit"],
        "left",
    )
if strict_never_glaucoma_controls:
    ever_glaucoma = screen.filter(F.col("glaucoma") == 1).select(
        "participant_id"
    ).distinct()
    healthy_images_spark = healthy_images_spark.join(
        ever_glaucoma,
        "participant_id",
        "left_anti",
    )

if exclude_selected_systemic_comorbidity:
    healthy_images_spark = healthy_images_spark.filter(
        F.col("multimorbidity_selected_count") == 0
    )

healthy_participant_spark = spark_participant_visit_mean(
    healthy_images_spark,
    "age_at_fundus_years",
    metadata_columns,
)

glaucoma_participant = glaucoma_participant_spark.toPandas().rename(
    columns={"sex_at_birth": "sex"}
)
healthy_participant = healthy_participant_spark.toPandas().rename(
    columns={"sex_at_birth": "sex"}
)
glaucoma_participant.attrs = {}
healthy_participant.attrs = {}

glaucoma_participant = validate_embedding_frame(
    glaucoma_participant,
    "CLSA glaucoma-only participant visits",
)
healthy_participant = validate_embedding_frame(
    healthy_participant,
    "CLSA healthy participant visits",
)

healthy_oof = pd.read_parquet(healthy_oof_path)
healthy_oof["participant_id"] = healthy_oof["participant_id"].astype(str)
healthy_oof["visit"] = healthy_oof["visit"].astype(str).str.upper()
age_model_training_ids = set(healthy_oof["participant_id"])
training_overlap = glaucoma_participant["participant_id"].isin(
    age_model_training_ids
)
print(
    "CLSA glaucoma participants appearing in age-model training at another "
    f"screen-negative visit: {int(training_overlap.sum()):,}"
)
if exclude_age_model_training_overlap:
    glaucoma_participant = glaucoma_participant.loc[
        ~training_overlap
    ].reset_index(drop=True)
if glaucoma_participant.empty:
    raise ValueError(
        "No CLSA glaucoma-only participants remain after age-model training "
        "overlap exclusion. Review the overlap count before changing the widget."
    )
healthy_participant = healthy_participant.merge(
    healthy_oof[
        [
            "participant_id",
            "visit",
            "retinal_age_prediction",
            "retinal_age_gap",
            "absolute_error",
        ]
    ],
    on=["participant_id", "visit"],
    how="inner",
    validate="one_to_one",
)
glaucoma_predictions = predict_retinal_age(glaucoma_participant, age_bundle)
glaucoma_participant = glaucoma_participant.merge(
    glaucoma_predictions[
        [
            "participant_id",
            "visit",
            "retinal_age_prediction",
            "retinal_age_gap",
            "absolute_error",
        ]
    ],
    on=["participant_id", "visit"],
    how="inner",
    validate="one_to_one",
)

zeiss_images = pd.read_parquet(zeiss_embeddings_path).rename(
    columns={"patient_id": "participant_id", "dcm_path": "image_path"}
)
zeiss_images.attrs = {}
zeiss_images = validate_embedding_frame(
    zeiss_images,
    "Zeiss glaucoma image embeddings",
)
zeiss_metadata = [column for column in ("sex", "race") if column in zeiss_images.columns]
zeiss_participant = aggregate_embedding_rows(
    zeiss_images,
    group_columns=("participant_id",),
    metadata_columns=zeiss_metadata,
)
zeiss_participant = validate_embedding_frame(
    zeiss_participant,
    "Zeiss glaucoma participants",
)
zeiss_predictions = predict_retinal_age(zeiss_participant, age_bundle)
zeiss_participant = zeiss_participant.merge(
    zeiss_predictions[
        [
            "participant_id",
            "retinal_age_prediction",
            "retinal_age_gap",
            "absolute_error",
        ]
    ],
    on="participant_id",
    how="inner",
    validate="one_to_one",
)

healthy_participant = select_representative_visit(healthy_participant)
glaucoma_participant = select_representative_visit(glaucoma_participant)

for frame, cohort, source, glaucoma in (
    (healthy_participant, "CLSA healthy", "CLSA", 0),
    (glaucoma_participant, "CLSA glaucoma-only", "CLSA", 1),
    (zeiss_participant, "Zeiss glaucoma", "Zeiss", 1),
):
    frame["cohort"] = cohort
    frame["source"] = source
    frame["glaucoma"] = glaucoma
    frame["sex_normalized"] = frame.get(
        "sex", pd.Series(index=frame.index, dtype=object)
    ).map(canonical_sex)

write_frame(healthy_participant, cohort_root / "clsa_healthy_participants.parquet")
write_frame(glaucoma_participant, cohort_root / "clsa_glaucoma_only_participants.parquet")
write_frame(zeiss_participant, cohort_root / "zeiss_glaucoma_participants.parquet")

cohort_summary = pd.DataFrame(
    [
        prediction_summary(healthy_participant, "CLSA healthy (OOF)"),
        prediction_summary(glaucoma_participant, "CLSA glaucoma-only"),
        prediction_summary(zeiss_participant, "Zeiss glaucoma"),
    ]
)
write_frame(cohort_summary, prediction_root / "three_cohort_prediction_summary.csv")
display(cohort_summary.round(3))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Primary and transported raw comparisons
# MAGIC
# MAGIC Each contrast reports both an age/sex-adjusted HC3 regression and a
# MAGIC deterministic, no-replacement participant match. The CLSA comparison
# MAGIC additionally adjusts for visit and available systemic covariates.

# COMMAND ----------
systemic_numeric = [
    column for column in ("diabetes", "hypertension")
    if column in healthy_participant.columns and column in glaucoma_participant.columns
]
systemic_categorical = [
    column for column in ("smoking_status",)
    if column in healthy_participant.columns and column in glaucoma_participant.columns
]


def run_comparison(
    name,
    exposed,
    reference,
    *,
    outcome_column="retinal_age_gap",
    within_clsa=False,
    output_directory=comparison_root,
):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(
        [
            exposed.assign(exposed=1),
            reference.assign(exposed=0),
        ],
        ignore_index=True,
    )
    def usable_sex(frame):
        values = frame["sex_normalized"].astype(str)
        return set(values[values != "MISSING"])

    shared_observed_sex = usable_sex(exposed) & usable_sex(reference)
    sex_adjustment_available = (
        len(shared_observed_sex) >= 2
        and (exposed["sex_normalized"] != "MISSING").mean() >= 0.80
        and (reference["sex_normalized"] != "MISSING").mean() >= 0.80
    )
    numeric = ["age", *(systemic_numeric if within_clsa else [])]
    categorical = [
        *(["sex_normalized"] if sex_adjustment_available else []),
        *(["visit", *systemic_categorical] if within_clsa else []),
    ]
    adjusted = adjusted_group_effect(
        combined,
        outcome_column=outcome_column,
        numeric_covariates=numeric,
        categorical_covariates=categorical,
    )
    exact = [
        *(["sex_normalized"] if sex_adjustment_available else []),
        *(["visit"] if within_clsa else []),
    ]
    pairs = greedy_match(
        exposed,
        reference,
        caliper_years=age_caliper_years,
        exact_columns=exact,
    )
    if pairs.empty:
        raise RuntimeError(f"No matches were found for {name}")
    paired, paired_rows = paired_outcome_effect(
        pairs,
        exposed,
        reference,
        outcome_column=outcome_column,
        bootstrap_repetitions=bootstrap_repetitions,
    )
    write_frame(pairs, output_directory / f"{name}_match_pairs.parquet")
    write_frame(
        paired_rows,
        output_directory / f"{name}_matched_outcomes.parquet",
    )
    rows = [
        {
            "comparison": name,
            "method": "adjusted_hc3_regression",
            "estimate": adjusted["adjusted_difference"],
            "ci_95_low": adjusted["ci_95_low"],
            "ci_95_high": adjusted["ci_95_high"],
            "p_value": adjusted["p_value"],
            "n": adjusted["n"],
            "n_pairs": np.nan,
            "mean_absolute_age_difference": np.nan,
            "adjustment": " + ".join(adjusted["design_columns"][2:]),
        },
        {
            "comparison": name,
            "method": "matched_participant_bootstrap",
            "estimate": paired["mean_difference"],
            "ci_95_low": paired["bootstrap_95_ci_low"],
            "ci_95_high": paired["bootstrap_95_ci_high"],
            "p_value": np.nan,
            "n": 2 * paired["n_pairs"],
            "n_pairs": paired["n_pairs"],
            "mean_absolute_age_difference": paired[
                "mean_absolute_age_difference"
            ],
            "adjustment": "1:1 age"
            + ("/sex" if sex_adjustment_available else "")
            + ("/visit" if within_clsa else "")
            + f" matching within ±{age_caliper_years:g} years",
        },
    ]
    return pd.DataFrame(rows)


raw_results = pd.concat(
    [
        run_comparison(
            "clsa_glaucoma_vs_clsa_healthy_primary",
            glaucoma_participant,
            healthy_participant,
            within_clsa=True,
        ),
        run_comparison(
            "zeiss_glaucoma_vs_clsa_glaucoma_raw",
            zeiss_participant,
            glaucoma_participant,
        ),
        run_comparison(
            "zeiss_glaucoma_vs_clsa_healthy_raw",
            zeiss_participant,
            healthy_participant,
        ),
    ],
    ignore_index=True,
)
write_frame(raw_results, comparison_root / "raw_comparison_results.csv")
display(raw_results.round(4))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Additive source harmonization and residual-domain diagnostics
# MAGIC
# MAGIC The harmonizer is fitted to participant-level embeddings with balanced
# MAGIC weights for the three observed design cells. It preserves modeled age,
# MAGIC sex, and glaucoma effects while removing an additive Zeiss location
# MAGIC effect; the optional location-scale mode also aligns residual feature
# MAGIC scales. This is analogous to feature-level scanner harmonization, not a
# MAGIC randomized correction.
# MAGIC
# MAGIC Because no healthy Zeiss cell exists, the disease-by-source interaction
# MAGIC remains unidentifiable. Results are considered more credible only when
# MAGIC they agree with the same-camera CLSA effect and residual source
# MAGIC classification falls substantially.

# COMMAND ----------
harmonization_input = pd.concat(
    [healthy_participant, glaucoma_participant, zeiss_participant],
    ignore_index=True,
)
harmonization_bundle = fit_additive_source_harmonizer(
    harmonization_input,
    ridge=harmonization_ridge,
)
joblib.dump(
    harmonization_bundle,
    harmonization_root / "additive_source_harmonizer.joblib",
)
harmonized = apply_source_harmonizer(
    harmonization_input,
    harmonization_bundle,
    mode=harmonization_mode,
)

zeiss_harmonized = harmonized[harmonized["cohort"] == "Zeiss glaucoma"].copy()
clsa_glaucoma_harmonized = harmonized[
    harmonized["cohort"] == "CLSA glaucoma-only"
].copy()
clsa_healthy_harmonized = harmonized[
    harmonized["cohort"] == "CLSA healthy"
].copy()

# CLSA vectors are unchanged. Preserve their primary/OOF prediction columns.
zeiss_harmonized = zeiss_harmonized.drop(
    columns=[
        column
        for column in (
            "retinal_age_prediction",
            "retinal_age_gap",
            "absolute_error",
            "retinal_age_raw",
        )
        if column in zeiss_harmonized.columns
    ]
)
zeiss_harmonized_predictions = predict_retinal_age(
    zeiss_harmonized,
    age_bundle,
)
zeiss_harmonized = zeiss_harmonized.merge(
    zeiss_harmonized_predictions[
        [
            "participant_id",
            "retinal_age_prediction",
            "retinal_age_gap",
            "absolute_error",
        ]
    ],
    on="participant_id",
    how="inner",
    validate="one_to_one",
)
write_frame(
    zeiss_harmonized,
    harmonization_root / "zeiss_glaucoma_harmonized_participants.parquet",
)

source_pairs = greedy_match(
    zeiss_participant,
    glaucoma_participant,
    caliper_years=age_caliper_years,
    exact_columns=(
        ("sex_normalized",)
        if (
            (zeiss_participant["sex_normalized"] != "MISSING").mean() >= 0.80
            and (glaucoma_participant["sex_normalized"] != "MISSING").mean()
            >= 0.80
            and len(
                set(
                    zeiss_participant.loc[
                        zeiss_participant["sex_normalized"] != "MISSING",
                        "sex_normalized",
                    ]
                )
                & set(
                    glaucoma_participant.loc[
                        glaucoma_participant["sex_normalized"] != "MISSING",
                        "sex_normalized",
                    ]
                )
            )
            >= 2
        )
        else ()
    ),
)
if source_pairs.empty:
    raise RuntimeError("No age/sex-matched glaucoma participants for domain diagnostics")
clsa_source_ids = set(source_pairs["reference_id"])
zeiss_source_ids = set(source_pairs["exposed_id"])
domain_clsa = glaucoma_participant[
    glaucoma_participant["participant_id"].isin(clsa_source_ids)
].copy()
domain_zeiss_raw = zeiss_participant[
    zeiss_participant["participant_id"].isin(zeiss_source_ids)
].copy()
domain_zeiss_harmonized = zeiss_harmonized[
    zeiss_harmonized["participant_id"].isin(zeiss_source_ids)
].copy()

domain_diagnostics = []
for stage, target in (
    ("raw", domain_zeiss_raw),
    ("harmonized", domain_zeiss_harmonized),
):
    diagnostics = {
        "stage": stage,
        **embedding_shift_summary(domain_clsa, target),
        **cross_validated_domain_auc(
            domain_clsa,
            target,
            max_per_domain=domain_classifier_max_per_source,
        ),
    }
    domain_diagnostics.append(diagnostics)
domain_diagnostics = pd.DataFrame(domain_diagnostics)
write_frame(
    domain_diagnostics,
    harmonization_root / "residual_domain_diagnostics.csv",
)
display(domain_diagnostics.round(4))

harmonized_results = pd.concat(
    [
        run_comparison(
            "zeiss_glaucoma_vs_clsa_glaucoma_harmonized",
            zeiss_harmonized,
            glaucoma_participant,
            output_directory=harmonization_root,
        ),
        run_comparison(
            "zeiss_glaucoma_vs_clsa_healthy_harmonized",
            zeiss_harmonized,
            healthy_participant,
            output_directory=harmonization_root,
        ),
    ],
    ignore_index=True,
)
write_frame(
    harmonized_results,
    harmonization_root / "harmonized_comparison_results.csv",
)
display(harmonized_results.round(4))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Publication-oriented figures and triangulation table

# COMMAND ----------
all_results = pd.concat(
    [
        raw_results.assign(analysis_stage="raw"),
        harmonized_results.assign(analysis_stage="harmonized_sensitivity"),
    ],
    ignore_index=True,
)
write_frame(all_results, analysis_root / "THREE_COHORT_GLAUCOMA_RESULTS.csv")

plt.rcParams.update({"font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
for frame, label, color in (
    (healthy_participant, "CLSA healthy OOF", "#2563eb"),
    (glaucoma_participant, "CLSA glaucoma-only", "#059669"),
    (zeiss_participant, "Zeiss glaucoma raw", "#dc2626"),
    (zeiss_harmonized, "Zeiss glaucoma harmonized", "#7c3aed"),
):
    axes[0].hist(
        frame["retinal_age_gap"],
        bins=45,
        density=True,
        histtype="step",
        linewidth=1.8,
        label=label,
        color=color,
    )
    axes[1].scatter(
        frame["age"],
        frame["retinal_age_gap"],
        s=6,
        alpha=0.16,
        label=label,
        color=color,
    )
axes[0].axvline(0, color="black", linewidth=0.8)
axes[0].set(
    title="Participant retinal-age gap",
    xlabel="Predicted − chronological age (years)",
    ylabel="Density",
)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set(
    title="Gap over chronological age",
    xlabel="Chronological age (years)",
    ylabel="Retinal-age gap (years)",
)

diagnostic_plot = domain_diagnostics.set_index("stage")
axes[2].bar(
    ["Raw", "Harmonized"],
    diagnostic_plot.loc[["raw", "harmonized"], "domain_auc_mean"],
    color=["#dc2626", "#7c3aed"],
)
axes[2].axhline(0.5, color="black", linestyle="--", linewidth=1)
axes[2].set(
    title="Held-out CLSA-versus-Zeiss classifier",
    ylabel="Cross-validated domain ROC AUC",
    ylim=(0.45, 1.02),
)
axes[0].legend(frameon=False, fontsize=8)
axes[1].legend(frameon=False, fontsize=8)
for suffix in ("png", "pdf"):
    fig.savefig(
        figure_root / f"figure_06_01_three_cohort_and_domain_diagnostics.{suffix}",
        dpi=220 if suffix == "png" else None,
        bbox_inches="tight",
    )
plt.show()
plt.close(fig)

plot_results = all_results.copy()
plot_results["label"] = (
    plot_results["comparison"].str.replace("_", " ", regex=False)
    + " | "
    + plot_results["method"].str.replace("_", " ", regex=False)
)
plot_results = plot_results.sort_values(["analysis_stage", "comparison", "method"])
fig, axis = plt.subplots(
    figsize=(10, max(5, 0.48 * len(plot_results))),
    constrained_layout=True,
)
y = np.arange(len(plot_results))
axis.errorbar(
    plot_results["estimate"],
    y,
    xerr=np.vstack(
        [
            plot_results["estimate"] - plot_results["ci_95_low"],
            plot_results["ci_95_high"] - plot_results["estimate"],
        ]
    ),
    fmt="o",
    color="#1f2937",
    ecolor="#64748b",
    capsize=3,
)
axis.axvline(0, color="black", linewidth=0.8)
axis.set_yticks(y, plot_results["label"])
axis.set_xlabel("Exposed − reference retinal-age gap (years)")
axis.set_title("Primary, transported, and harmonized-sensitivity estimates")
axis.invert_yaxis()
for suffix in ("png", "pdf"):
    fig.savefig(
        figure_root / f"figure_06_02_effect_forest.{suffix}",
        dpi=220 if suffix == "png" else None,
        bbox_inches="tight",
    )
plt.show()
plt.close(fig)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Reproducible summary and interpretation guardrails

# COMMAND ----------
primary_row = raw_results[
    (raw_results["comparison"] == "clsa_glaucoma_vs_clsa_healthy_primary")
    & (raw_results["method"] == "matched_participant_bootstrap")
].iloc[0]
post_auc = float(
    domain_diagnostics.loc[
        domain_diagnostics["stage"] == "harmonized", "domain_auc_mean"
    ].iloc[0]
)

run_summary = {
    "analysis": "three_cohort_glaucoma_retinal_age_triangulation",
    "primary_comparison": "CLSA glaucoma-only versus CLSA healthy",
    "primary_reason": "same acquisition source and RETFound pipeline",
    "glaucoma_definition": (
        "released physician-diagnosed self-report with complete negative "
        "same-visit screen for other prespecified ocular disease"
    ),
    "strict_never_glaucoma_controls": strict_never_glaucoma_controls,
    "exclude_age_model_training_overlap": exclude_age_model_training_overlap,
    "exclude_selected_systemic_comorbidity": exclude_selected_systemic_comorbidity,
    "n_clsa_healthy_participants": int(len(healthy_participant)),
    "n_clsa_glaucoma_only_participants": int(len(glaucoma_participant)),
    "n_zeiss_glaucoma_participants": int(len(zeiss_participant)),
    "primary_matched_age_gap_difference": float(primary_row["estimate"]),
    "primary_matched_bootstrap_95_ci": [
        float(primary_row["ci_95_low"]),
        float(primary_row["ci_95_high"]),
    ],
    "harmonization_method": harmonization_bundle["method"],
    "harmonization_mode": harmonization_mode,
    "harmonization_assumption": harmonization_bundle["assumption"],
    "post_harmonization_domain_auc": post_auc,
    "residual_domain_signal_flag_auc_gt_0_60": bool(post_auc > 0.60),
    "age_model_path": healthy_model_path,
    "age_model_sha256": hashlib.sha256(Path(healthy_model_path).read_bytes()).hexdigest(),
    "outputs": {
        "glaucoma_vectors": str(glaucoma_vector_path),
        "all_results": str(analysis_root / "THREE_COHORT_GLAUCOMA_RESULTS.csv"),
        "domain_diagnostics": str(
            harmonization_root / "residual_domain_diagnostics.csv"
        ),
    },
    "interpretation": (
        "The within-CLSA estimate is the primary glaucoma association. Zeiss "
        "results are supportive only if raw and harmonized estimates are "
        "directionally consistent with the CLSA estimate and residual domain "
        "classification is acceptably reduced."
    ),
    "limitations": [
        "CLSA glaucoma is self-reported physician diagnosis rather than chart adjudication.",
        "No glaucoma subtype, severity, visual-field, or OCT criterion is required.",
        "Healthy Zeiss controls are absent, so source-by-disease interaction is not identifiable.",
        "Harmonization assumes an additive source effect and may remove biological cohort differences.",
        "Systemic confounders unavailable in Zeiss cannot be balanced across all three cohorts.",
        "A cross-sectional retinal-age gap is an association and not evidence of accelerated aging causality.",
    ],
}
write_json(run_summary, analysis_root / "THREE_COHORT_GLAUCOMA_SUMMARY.json")
print(json.dumps(run_summary, indent=2))
print("Notebook 06 complete")
