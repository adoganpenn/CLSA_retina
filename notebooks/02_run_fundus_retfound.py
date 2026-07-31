# Databricks notebook source
# MAGIC %md
# MAGIC # CLSA fundus quality, RETFound, retinal age, and explainability
# MAGIC
# MAGIC This notebook applies the configuration validated by
# MAGIC `notebooks/smoketest.ipynb`: 256 px fundus preprocessing, 224 px
# MAGIC RETFound input, per-image/channel standardization, CUDA inference,
# MAGIC 1,024-dimensional vector validation, and durable Parquet + Delta output.
# MAGIC Run and debug one stage at a time on GPU compute.

# COMMAND ----------
# MAGIC %md
# MAGIC ## One-time environment setup
# MAGIC
# MAGIC Run this once on a fresh GPU cluster, then restart Python:
# MAGIC
# MAGIC ```python
# MAGIC %pip install -r /Workspace/Users/ad0038@pennmedicine.upenn.edu/CLSA/CLSA_retina/requirements-retfound.txt
# MAGIC dbutils.library.restartPython()
# MAGIC ```
# MAGIC
# MAGIC The gated Hugging Face token is read from a temporary password-style
# MAGIC widget, placed in the process environment only while the model loads,
# MAGIC and removed before embedding begins. It is never written to an output.

# COMMAND ----------
# MAGIC %pip install -r /Workspace/Users/ad0038@pennmedicine.upenn.edu/CLSA/CLSA_retina/requirements-retfound.txt

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from pyspark.sql import functions as F

# COMMAND ----------
dbutils.widgets.text(
    "repo_root",
    "/Workspace/Users/ad0038@pennmedicine.upenn.edu/CLSA/CLSA_retina",
)
dbutils.widgets.text(
    "source_path",
    "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/derived/"
    "clsa_retinal_aging/fundus_image_manifest",
)
dbutils.widgets.dropdown(
    "source_format", "delta", ["delta", "parquet", "csv"]
)
dbutils.widgets.text(
    "output_root",
    "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/derived/"
    "clsa_retinal_aging/fundus_retfound_smoke",
)
dbutils.widgets.text("retfound_repo", "")
dbutils.widgets.text("checkpoint_path", "")
dbutils.widgets.dropdown("allow_downloads", "true", ["true", "false"])
dbutils.widgets.text("hf_token", "", "Hugging Face token (temporary)")
dbutils.widgets.dropdown("device", "cuda", ["cuda", "auto", "cpu"])
dbutils.widgets.text("batch_size", "2")
dbutils.widgets.text("max_images", "8")
dbutils.widgets.dropdown("save_preprocessed", "true", ["true", "false"])
dbutils.widgets.dropdown("force_embeddings", "true", ["true", "false"])
dbutils.widgets.dropdown("train_age_head", "false", ["false", "true"])
dbutils.widgets.text("existing_age_model", "")
dbutils.widgets.dropdown("run_explainability", "false", ["false", "true"])
dbutils.widgets.text("n_explain", "8")
dbutils.widgets.dropdown(
    "explainability_method", "exact", ["exact", "occlusion"]
)

# COMMAND ----------
# MAGIC %md
# MAGIC The defaults perform a safe eight-image validation using the same
# MAGIC settings as the successful smoke test. For the full cohort, change
# MAGIC `output_root` to `.../fundus_retfound`, set `max_images=0`, use
# MAGIC `batch_size=8` initially on a Tesla T4, and retain
# MAGIC `run_explainability=false` until an age-head artifact is available.

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
expected_embedding_dim = 1024


def databricks_path_exists(path: str) -> bool:
    try:
        dbutils.fs.ls(path)
        return True
    except Exception:
        return False


if source_format == "delta" and not databricks_path_exists(source_path):
    derived_root = (
        "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/derived/"
        "clsa_retinal_aging"
    )
    smoke_candidates = [
        f"{derived_root}/fundus_smoke_manifest",
        (
            f"{derived_root}/fundus_retfound_smoke/00_input/"
            "fundus_smoke_manifest"
        ),
    ]
    fallback = next(
        (
            candidate
            for candidate in smoke_candidates
            if databricks_path_exists(candidate)
        ),
        None,
    )
    if fallback:
        print(
            "Configured full image manifest is unavailable; using smoke "
            f"manifest: {fallback}"
        )
        source_path = fallback
        dbutils.widgets.set("source_path", fallback)
    else:
        raise FileNotFoundError(
            "No fundus source manifest exists. Either set source_path to a "
            "valid smoke manifest or run notebook 01 with "
            "extract_fundus_archives=true to create fundus_image_manifest. "
            f"Configured path: {source_path}"
        )

module_path = Path(repo_root) / "src" / "fundus_retfound_pipeline.py"
if not module_path.exists():
    raise FileNotFoundError(f"Pipeline module was not found: {module_path}")
if str(module_path.parent) not in sys.path:
    sys.path.insert(0, str(module_path.parent))

from fundus_retfound_pipeline import (  # noqa: E402
    AgeModelConfig,
    ExplainabilityConfig,
    QualityConfig,
    RETFoundConfig,
    extract_retfound_embeddings,
    load_age_head,
    load_retfound_model,
    prepare_model_input,
    predict_retinal_age,
    run_explainability,
    run_quality_pipeline,
    train_age_head,
)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
if device_requested == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested, but this compute has no CUDA GPU.")

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
            F.col("fundus.visit").alias("visit"),
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
    if "visit" in source.columns:
        selections.append(F.col("visit").cast("string").alias("visit"))
    if "filename" in source.columns:
        selections.append(F.col("filename").cast("string").alias("filename"))
    if "age" in source.columns:
        selections.append(F.col("age").cast("double").alias("age"))
    elif "age_years" in source.columns:
        selections.append(F.col("age_years").cast("double").alias("age"))
    if "sex" in source.columns:
        selections.append(F.col("sex").cast("string").alias("sex"))
    elif "sex_at_birth" in source.columns:
        selections.append(
            F.col("sex_at_birth").cast("string").alias("sex")
        )
    manifest_spark = source.select(*selections)
else:
    raise ValueError(
        "Source must contain fundus_images or an image_path column."
    )

manifest_spark = manifest_spark.filter(
    F.col("image_path").isNotNull() & F.col("participant_id").isNotNull()
)
manifest_spark = manifest_spark.orderBy(
    "participant_id",
    "eye",
    "image_path",
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
if manifest.empty:
    raise ValueError(f"No usable images were found in {source_path}.")
display(manifest_spark.limit(20))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Technical image quality and AutoMorph-style preprocessing

# COMMAND ----------
quality_config = QualityConfig(
    output_size=256,
    model_input_size=224,
    save_preprocessed=save_preprocessed,
)
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

quality_preview_columns = [
    column
    for column in (
        "participant_id",
        "visit",
        "eye",
        "quality_pass",
        "quality_reasons",
        "original_width",
        "original_height",
        "retina_fraction",
        "brightness_mean",
        "contrast_std",
        "gradient_energy",
        "processed_image_path",
    )
    if column in quality.columns
]
display(quality[quality_preview_columns].head(100))

passing_quality = quality[quality["quality_pass"].fillna(False)].copy()
if passing_quality.empty:
    raise RuntimeError(
        "No images passed technical quality control. Review 01_quality before "
        "changing any thresholds."
    )

sample_input = prepare_model_input(
    passing_quality.iloc[0]["image_path"],
    quality_config,
)
if sample_input.shape != (224, 224, 3):
    raise ValueError(f"Unexpected RETFound input shape: {sample_input.shape}")
if not np.isfinite(sample_input).all():
    raise ValueError("RETFound model input contains NaN or infinity.")
print("RETFound input shape:", sample_input.shape)
print("Channel means:", sample_input.mean(axis=(0, 1)))
print("Channel standard deviations:", sample_input.std(axis=(0, 1)))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Load RETFound and extract/cache embeddings

# COMMAND ----------
retfound_config = RETFoundConfig(
    repo_path=retfound_repo,
    checkpoint_path=checkpoint_path,
    allow_downloads=allow_downloads,
    device=device_requested,
    batch_size=batch_size,
)

temporary_hf_token = dbutils.widgets.get("hf_token").strip()
if allow_downloads and not checkpoint_path and not temporary_hf_token:
    raise ValueError(
        "Enter the temporary Hugging Face token, or provide checkpoint_path."
    )
if temporary_hf_token:
    os.environ["HF_TOKEN"] = temporary_hf_token
try:
    model, device, resolved_repo, resolved_checkpoint = load_retfound_model(
        retfound_config
    )
finally:
    os.environ.pop("HF_TOKEN", None)
    temporary_hf_token = ""

if device_requested == "cuda" and device != "cuda":
    raise RuntimeError(f"Expected CUDA, but RETFound resolved device={device!r}.")
print("Device:", device)
print("RETFound repository:", resolved_repo)
print("Checkpoint:", resolved_checkpoint)

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

embedding_matrix = np.stack(embeddings["embedding"].to_numpy()).astype(
    np.float32
)
expected_shape = (len(embeddings), expected_embedding_dim)
if embedding_matrix.shape != expected_shape:
    raise ValueError(
        f"Expected embedding matrix {expected_shape}, got {embedding_matrix.shape}."
    )
if not np.isfinite(embedding_matrix).all():
    raise ValueError("RETFound vectors contain NaN or infinity.")

norms = np.linalg.norm(embedding_matrix, axis=1)
if not np.all(norms > 0):
    raise ValueError("At least one RETFound vector has zero length.")

quality_paths = set(passing_quality["image_path"].astype(str))
embedded_paths = set(embeddings["image_path"].astype(str))
failure_path = output_root / "02_embeddings" / "retfound_embedding_failures.csv"
failure_paths: set[str] = set()
if failure_path.exists() and failure_path.stat().st_size > 0:
    failures = pd.read_csv(failure_path)
    if "image_path" in failures.columns:
        failure_paths = set(failures["image_path"].dropna().astype(str))
unaccounted_paths = quality_paths - embedded_paths - failure_paths
cached_extra_paths = embedded_paths - quality_paths
if unaccounted_paths or cached_extra_paths:
    raise RuntimeError(
        "The embedding cache does not match this manifest. Use a new "
        "output_root or set force_embeddings=true. "
        f"Unaccounted={len(unaccounted_paths)}, extra={len(cached_extra_paths)}."
    )

print("Final embedding matrix shape:", embedding_matrix.shape)
print("Vector dtype:", embedding_matrix.dtype)
print("Vector L2 norm range:", float(norms.min()), float(norms.max()))

# Store a Spark-native copy for downstream Databricks SQL and Delta joins.
vector_records = []
for row_number, (_, record) in enumerate(embeddings.iterrows()):
    vector_records.append(
        {
            "participant_id": str(record["participant_id"]),
            "visit": str(record.get("visit", "")),
            "eye": str(record.get("eye", "")),
            "image_path": str(record["image_path"]),
            "embedding_dim": int(record["embedding_dim"]),
            "retfound_model": str(record["retfound_model"]),
            "retfound_checkpoint_sha256": str(
                record["retfound_checkpoint_sha256"]
            ),
            "embedding": [
                float(value) for value in embedding_matrix[row_number]
            ],
        }
    )

vectors_spark = spark.createDataFrame(vector_records)
vectors_delta_path = str(
    output_root / "02_embeddings" / "retfound_embeddings_delta"
)
(
    vectors_spark.write.format("delta")
    .mode("overwrite")
    .save(vectors_delta_path)
)

preview_rows = [
    {
        "participant_id": record["participant_id"],
        "visit": record["visit"],
        "eye": record["eye"],
        "embedding_dim": record["embedding_dim"],
        "l2_norm": float(norms[row_number]),
        "first_8_values": record["embedding"][:8],
    }
    for row_number, record in enumerate(vector_records)
]
display(spark.createDataFrame(preview_rows))
print("Parquet vectors:", output_root / "02_embeddings/retfound_embeddings.parquet")
print("Delta vectors:", vectors_delta_path)

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

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Remove temporary credentials

# COMMAND ----------
os.environ.pop("HF_TOKEN", None)
try:
    dbutils.widgets.remove("hf_token")
except Exception:
    pass
print("Temporary Hugging Face token widget removed")
