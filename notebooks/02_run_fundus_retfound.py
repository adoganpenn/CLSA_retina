# Databricks notebook source
# MAGIC %md
# MAGIC # CLSA fundus quality, RETFound, retinal age, and explainability
# MAGIC
# MAGIC This notebook is intentionally a thin caller around
# MAGIC `src/fundus_retfound_pipeline.py`. Run and debug one stage at a time.
# MAGIC Use a GPU cluster for RETFound embedding and explanation stages.

# COMMAND ----------
# MAGIC %md
# MAGIC ## One-time environment setup
# MAGIC
# MAGIC Run this in a fresh GPU cluster, then restart Python:
# MAGIC
# MAGIC ```python
# MAGIC %pip install -r /Workspace/Repos/<user>/<repo>/requirements-retfound.txt
# MAGIC dbutils.library.restartPython()
# MAGIC ```
# MAGIC
# MAGIC Do not paste Hugging Face tokens into this notebook. The checkpoint is
# MAGIC gated; accept its terms first, then use a Databricks secret.

# COMMAND ----------
import os
from pathlib import Path
import sys

from pyspark.sql import functions as F

# COMMAND ----------
dbutils.widgets.text("repo_root", "/Workspace/Repos/<user>/<repo>")
dbutils.widgets.text(
    "source_path",
    "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/derived/"
    "clsa_retinal_aging/participant_analysis_master",
)
dbutils.widgets.dropdown(
    "source_format", "delta", ["delta", "parquet", "csv"]
)
dbutils.widgets.text(
    "output_root",
    "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/derived/"
    "clsa_retinal_aging/fundus_retfound",
)
dbutils.widgets.text("retfound_repo", "")
dbutils.widgets.text("checkpoint_path", "")
dbutils.widgets.dropdown("allow_downloads", "false", ["false", "true"])
dbutils.widgets.text("hf_secret_scope", "clsa")
dbutils.widgets.text("hf_secret_key", "huggingface-token")
dbutils.widgets.dropdown("device", "cuda", ["cuda", "auto", "cpu"])
dbutils.widgets.text("batch_size", "16")
dbutils.widgets.text("max_images", "0")
dbutils.widgets.dropdown("save_preprocessed", "false", ["false", "true"])
dbutils.widgets.dropdown("force_embeddings", "false", ["false", "true"])
dbutils.widgets.dropdown("train_age_head", "false", ["false", "true"])
dbutils.widgets.text("existing_age_model", "")
dbutils.widgets.dropdown("run_explainability", "true", ["true", "false"])
dbutils.widgets.text("n_explain", "8")
dbutils.widgets.dropdown(
    "explainability_method", "exact", ["exact", "occlusion"]
)

# COMMAND ----------
repo_root = dbutils.widgets.get("repo_root").rstrip("/")
source_path = dbutils.widgets.get("source_path").strip()
source_format = dbutils.widgets.get("source_format")
output_root = Path(dbutils.widgets.get("output_root").strip())
retfound_repo = dbutils.widgets.get("retfound_repo").strip() or None
checkpoint_path = dbutils.widgets.get("checkpoint_path").strip() or None
allow_downloads = dbutils.widgets.get("allow_downloads") == "true"
device_requested = dbutils.widgets.get("device")
batch_size = int(dbutils.widgets.get("batch_size"))
max_images = int(dbutils.widgets.get("max_images"))
save_preprocessed = dbutils.widgets.get("save_preprocessed") == "true"
force_embeddings = dbutils.widgets.get("force_embeddings") == "true"
should_train_age_head = dbutils.widgets.get("train_age_head") == "true"
existing_age_model = dbutils.widgets.get("existing_age_model").strip() or None
should_explain = dbutils.widgets.get("run_explainability") == "true"
n_explain = int(dbutils.widgets.get("n_explain"))
explainability_method = dbutils.widgets.get("explainability_method")

sys.path.insert(0, f"{repo_root}/src")

from fundus_retfound_pipeline import (  # noqa: E402
    AgeModelConfig,
    ExplainabilityConfig,
    QualityConfig,
    RETFoundConfig,
    extract_retfound_embeddings,
    load_age_head,
    load_retfound_model,
    predict_retinal_age,
    run_explainability,
    run_quality_pipeline,
    train_age_head,
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Build an image-level manifest from the CLSA Delta output

# COMMAND ----------
if source_format == "delta":
    source = spark.read.format("delta").load(source_path)
elif source_format == "parquet":
    source = spark.read.parquet(source_path)
else:
    source = spark.read.option("header", "true").csv(source_path)

if "fundus_images" in source.columns:
    # participant_analysis_master: explode the image array without duplicating
    # image bytes. Age comes from the harmonized questionnaire.
    manifest_spark = (
        source.select(
            F.col("participant_id").cast("string").alias("participant_id"),
            F.col("age_years").cast("double").alias("age"),
            F.col("sex_at_birth").cast("string").alias("sex"),
            F.explode("fundus_images").alias("fundus"),
        )
        .select(
            "participant_id",
            "age",
            "sex",
            F.col("fundus.image_path").alias("image_path"),
            F.col("fundus.eye_parsed").alias("eye"),
            F.col("fundus.filename").alias("filename"),
        )
    )
elif "image_path" in source.columns:
    participant_column = (
        "participant_id"
        if "participant_id" in source.columns
        else "participant_id_parsed"
    )
    selections = [
        F.col(participant_column).cast("string").alias("participant_id"),
        F.col("image_path"),
        F.col("eye_parsed").alias("eye")
        if "eye_parsed" in source.columns
        else F.lit(None).cast("string").alias("eye"),
    ]
    if "age" in source.columns:
        selections.append(F.col("age").cast("double").alias("age"))
    elif "age_years" in source.columns:
        selections.append(F.col("age_years").cast("double").alias("age"))
    manifest_spark = source.select(*selections)
else:
    raise ValueError(
        "Source must contain fundus_images or an image_path column."
    )

manifest_spark = manifest_spark.filter(
    F.col("image_path").isNotNull() & F.col("participant_id").isNotNull()
)
if max_images > 0:
    manifest_spark = manifest_spark.limit(max_images)

display(
    manifest_spark.agg(
        F.count("*").alias("images"),
        F.countDistinct("participant_id").alias("participants"),
        F.sum(F.col("age").isNotNull().cast("int")).alias("images_with_age")
        if "age" in manifest_spark.columns
        else F.lit(0).alias("images_with_age"),
    )
)
manifest = manifest_spark.toPandas()
manifest.head()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Technical image quality and AutoMorph-style preprocessing

# COMMAND ----------
quality_config = QualityConfig(save_preprocessed=save_preprocessed)
quality = run_quality_pipeline(
    manifest,
    output_root / "01_quality",
    quality_config,
)
display(
    spark.createDataFrame(
        quality.groupby(["quality_pass", "quality_reasons"], dropna=False)
        .size()
        .reset_index(name="images")
    ).orderBy(F.desc("images"))
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Load RETFound and extract/cache embeddings

# COMMAND ----------
if allow_downloads and not checkpoint_path:
    # The secret is placed only in the current process environment. The module
    # never prints it or stores it in output metadata.
    os.environ["HF_TOKEN"] = dbutils.secrets.get(
        scope=dbutils.widgets.get("hf_secret_scope"),
        key=dbutils.widgets.get("hf_secret_key"),
    )

retfound_config = RETFoundConfig(
    repo_path=retfound_repo,
    checkpoint_path=checkpoint_path,
    allow_downloads=allow_downloads,
    device=device_requested,
    batch_size=batch_size,
)
model, device, resolved_repo, resolved_checkpoint = load_retfound_model(
    retfound_config
)
print("device:", device)
print("RETFound repo:", resolved_repo)
print("checkpoint:", resolved_checkpoint)

embeddings = extract_retfound_embeddings(
    quality,
    output_root / "02_embeddings",
    retfound_config,
    quality_config,
    model=model,
    device=device,
    checkpoint_path=resolved_checkpoint,
    force=force_embeddings,
)
print("embedding rows:", len(embeddings))
print("embedding dimension:", embeddings["embedding_dim"].unique())

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Train or apply the retinal-age head
# MAGIC
# MAGIC For CLSA inference, normally supply an externally trained and locked
# MAGIC age-head artifact. Set `train_age_head=true` only when this dataset is
# MAGIC intentionally being used as age-head training data.

# COMMAND ----------
age_bundle = None
predictions = None
if existing_age_model:
    age_bundle = load_age_head(existing_age_model)
    predictions = predict_retinal_age(
        embeddings,
        age_bundle,
        output_root / "03_age_model" / "retinal_age_predictions.parquet",
    )
elif should_train_age_head:
    if "age" not in embeddings.columns or embeddings["age"].notna().sum() == 0:
        raise ValueError("Age-head training was requested but age is unavailable.")
    predictions, age_bundle = train_age_head(
        embeddings,
        output_root / "03_age_model",
        AgeModelConfig(calibration="intercept"),
    )
else:
    print(
        "No age head was trained or applied. Set existing_age_model for CLSA "
        "inference, or train_age_head=true for an intentional training run."
    )

if predictions is not None:
    display(spark.createDataFrame(predictions.dropna(axis=1, how="all").head(1000)))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Explain the locked age model

# COMMAND ----------
if should_explain:
    if age_bundle is None:
        raise ValueError(
            "Explainability requires a trained or supplied age-head bundle."
        )
    explanations = run_explainability(
        embeddings,
        age_bundle,
        output_root / "04_explainability",
        retfound_config,
        quality_config,
        ExplainabilityConfig(
            n_images=n_explain,
            method=explainability_method,
        ),
        model=model,
        device=device,
    )
    display(spark.createDataFrame(explanations))
