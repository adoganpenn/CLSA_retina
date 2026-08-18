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
dbutils.widgets.text("hf_token", "", "Hugging Face token (temporary)")

# COMMAND ----------
# MAGIC %md
# MAGIC The defaults perform a safe eight-image validation using the same
# MAGIC settings as the successful smoke test. Set `run_all_images=true` to use
# MAGIC every available image in the full manifest. This includes all BL and F1
# MAGIC images independently; the visits are not sampled, paired, or balanced.
# MAGIC The flag overrides `max_images`, changes the default smoke output folder
# MAGIC to `.../fundus_retfound`, and refuses to fall back to a smoke manifest.
# MAGIC Full runs are checkpointed in 500-image pipeline batches by default.
# MAGIC `batch_size` remains the GPU minibatch size inside each saved pipeline
# MAGIC batch. Use `batch_size=8` initially on a Tesla T4 and retain
# MAGIC `run_explainability=false` until an age-head artifact is available.

# COMMAND ----------
from pathlib import Path

repo_root = "/Workspace/Users/ad0038@pennmedicine.upenn.edu/CLSA/CLSA_retina"
derived_root = (
    "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/derived/"
    "clsa_retinal_aging"
)
source_path = f"{derived_root}/fundus_image_manifest"
source_format = "delta"
age_source_path = f"{derived_root}/sap_fundus_image_analysis"
attach_visit_matched_age = True
require_nonzero_age_coverage = True
exclude_images_without_age = True
output_root = Path(f"{derived_root}/fundus_retfound_smoke")
retfound_repo = None
checkpoint_path = None
allow_downloads = True
device_requested = "cuda"
batch_size = 2
max_images = 8
run_all_images = False
pipeline_batch_size = 500
resume_batches = True
save_preprocessed = True
force_embeddings = True
should_train_age_head = False
existing_age_model = None
should_explain = False
n_explain = 8
explainability_method = "exact"
expected_embedding_dim = 1024

if pipeline_batch_size < 1:
    raise ValueError("pipeline_batch_size must be at least 1.")

if run_all_images:
    max_images = 0
    print("run_all_images=true overrides max_images to 0 for this run.")
    if output_root.name == "fundus_retfound_smoke":
        output_root = output_root.with_name("fundus_retfound")
        print("Full run output_root:", output_root)
    if save_preprocessed:
        print(
            "Full-run notice: save_preprocessed=true will create one additional "
            "JPEG per quality-passing image. Set it to false if only vectors "
            "and quality metrics are required."
        )
    if batch_size < 4:
        print(
            "Full-run notice: batch_size is below 4. This is safe but slow; "
            "batch_size=8 is the recommended starting point for a Tesla T4."
        )


def databricks_path_exists(path: str) -> bool:
    try:
        dbutils.fs.ls(path)
        return True
    except Exception:
        return False


if source_format == "delta" and not databricks_path_exists(source_path):
    if run_all_images:
        raise FileNotFoundError(
            "run_all_images=true requires the full fundus image manifest; "
            "smoke-manifest fallback is disabled. Missing source_path: "
            f"{source_path}"
        )
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
    read_embedding_failure_paths,
    run_explainability,
    run_quality_pipeline,
    train_age_head,
    write_frame,
    write_json,
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
    # image bytes. Its participant-level age may be from a different visit, so
    # it is deliberately not attached to BL and F1 images here. The visit-aware
    # age join below supplies AGE_NMBR_COM/AGE_NMBR_COF1 instead.
    manifest_spark = (
        source.select(
            F.col("participant_id").cast("string").alias("participant_id"),
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
            F.lit(None).cast("double").alias("age"),
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
    if "age_at_fundus_years" in source.columns:
        selections.append(
            F.col("age_at_fundus_years").cast("double").alias("age")
        )
    elif "age" in source.columns:
        selections.append(F.col("age").cast("double").alias("age"))
    elif "age_years" in source.columns:
        selections.append(F.col("age_years").cast("double").alias("age"))
    if "age_at_fundus_source_variable" in source.columns:
        selections.append(
            F.col("age_at_fundus_source_variable")
            .cast("string")
            .alias("age_source_variable")
        )
    elif "age_source_variable" in source.columns:
        selections.append(
            F.col("age_source_variable")
            .cast("string")
            .alias("age_source_variable")
        )
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

if "visit" in manifest_spark.columns:
    manifest_spark = manifest_spark.withColumn(
        "visit",
        F.when(F.upper(F.col("visit")).isin("F1", "FUP1"), F.lit("F1"))
        .when(F.upper(F.col("visit")) == "BL", F.lit("BL"))
        .otherwise(F.upper(F.col("visit"))),
    )

age_already_available = (
    "age" in manifest_spark.columns
    and manifest_spark.filter(F.col("age").isNotNull()).limit(1).count() > 0
)
age_source_is_input = (
    source_format == "delta"
    and source_path.rstrip("/") == age_source_path.rstrip("/")
)

if attach_visit_matched_age and not (
    age_already_available and age_source_is_input
):
    if not age_source_path:
        raise ValueError(
            "attach_visit_matched_age=true requires age_source_path."
        )
    if not databricks_path_exists(age_source_path):
        raise FileNotFoundError(
            "Visit-matched age source was not found. Run notebook 03 first; "
            "it extracts AGE_NMBR_COM for BL and AGE_NMBR_COF1 for F1 and "
            "writes sap_fundus_image_analysis. Missing path: "
            f"{age_source_path}"
        )

    age_source = spark.read.format("delta").load(age_source_path)
    age_column = next(
        (
            column
            for column in (
                "age_at_fundus_years",
                "age_years",
                "age",
            )
            if column in age_source.columns
        ),
        None,
    )
    if age_column is None:
        raise ValueError(
            "Age source must contain age_at_fundus_years, age_years, or age. "
            f"Available columns: {age_source.columns}"
        )

    source_variable_column = next(
        (
            column
            for column in (
                "age_at_fundus_source_variable",
                "age_source_variable",
            )
            if column in age_source.columns
        ),
        None,
    )
    sex_column = next(
        (
            column
            for column in (
                "sex_at_birth",
                "baseline_sex_at_birth",
                "sex",
            )
            if column in age_source.columns
        ),
        None,
    )

    if "image_path" in age_source.columns:
        age_join_keys = ["image_path"]
        age_lookup = age_source.select(
            F.col("image_path").cast("string").alias("image_path"),
            F.col(age_column).cast("double").alias("linked_age"),
            F.col(source_variable_column)
            .cast("string")
            .alias("linked_age_source_variable")
            if source_variable_column
            else F.lit(None).cast("string").alias(
                "linked_age_source_variable"
            ),
            F.col(sex_column).cast("string").alias("linked_sex")
            if sex_column
            else F.lit(None).cast("string").alias("linked_sex"),
        )
    else:
        required_age_keys = {"participant_id", "visit"}
        missing_age_keys = required_age_keys - set(age_source.columns)
        if missing_age_keys:
            raise ValueError(
                "Age source without image_path must contain participant_id "
                f"and visit; missing {sorted(missing_age_keys)}."
            )
        age_join_keys = ["participant_id", "visit"]
        age_lookup = age_source.select(
            F.col("participant_id").cast("string").alias("participant_id"),
            F.when(
                F.upper(F.col("visit")).isin("F1", "FUP1"),
                F.lit("F1"),
            )
            .when(F.upper(F.col("visit")) == "BL", F.lit("BL"))
            .otherwise(F.upper(F.col("visit")))
            .alias("visit"),
            F.col(age_column).cast("double").alias("linked_age"),
            F.col(source_variable_column)
            .cast("string")
            .alias("linked_age_source_variable")
            if source_variable_column
            else F.lit(None).cast("string").alias(
                "linked_age_source_variable"
            ),
            F.col(sex_column).cast("string").alias("linked_sex")
            if sex_column
            else F.lit(None).cast("string").alias("linked_sex"),
        )

    duplicate_age_keys = (
        age_lookup.groupBy(*age_join_keys)
        .count()
        .filter(F.col("count") != 1)
    )
    if duplicate_age_keys.limit(1).count():
        raise ValueError(
            f"Age source is not unique on {age_join_keys}: {age_source_path}"
        )

    manifest_spark = manifest_spark.join(
        age_lookup,
        age_join_keys,
        "left",
    )
    if "age" in manifest_spark.columns:
        manifest_spark = manifest_spark.withColumn(
            "age",
            F.coalesce(F.col("age"), F.col("linked_age")),
        )
    else:
        manifest_spark = manifest_spark.withColumnRenamed(
            "linked_age",
            "age",
        )
    if "sex" in manifest_spark.columns:
        manifest_spark = manifest_spark.withColumn(
            "sex",
            F.coalesce(F.col("sex"), F.col("linked_sex")),
        )
    else:
        manifest_spark = manifest_spark.withColumnRenamed(
            "linked_sex",
            "sex",
        )
    if "age_source_variable" in manifest_spark.columns:
        manifest_spark = manifest_spark.withColumn(
            "age_source_variable",
            F.coalesce(
                F.col("age_source_variable"),
                F.col("linked_age_source_variable"),
            ),
        )
    else:
        manifest_spark = manifest_spark.withColumnRenamed(
            "linked_age_source_variable",
            "age_source_variable",
        )
    manifest_spark = manifest_spark.drop(
        "linked_age",
        "linked_sex",
        "linked_age_source_variable",
    )
    print("Attached visit-matched age from:", age_source_path)

if "age" not in manifest_spark.columns:
    manifest_spark = manifest_spark.withColumn(
        "age",
        F.lit(None).cast("double"),
    )
if "age_source_variable" not in manifest_spark.columns:
    manifest_spark = manifest_spark.withColumn(
        "age_source_variable",
        F.when(F.col("visit") == "BL", F.lit("AGE_NMBR_COM"))
        .when(F.col("visit") == "F1", F.lit("AGE_NMBR_COF1"))
        .otherwise(F.lit(None).cast("string")),
    )
manifest_spark = manifest_spark.withColumn(
    "age_link_status",
    F.when(F.col("age").isNotNull(), F.lit("visit_age_matched")).otherwise(
        F.lit("visit_age_missing")
    ),
)

manifest_spark = manifest_spark.filter(
    F.col("image_path").isNotNull() & F.col("participant_id").isNotNull()
).dropDuplicates(["image_path"])
display(
    manifest_spark.groupBy(
        "visit",
        "age_source_variable",
        "age_link_status",
    )
    .agg(
        F.count("*").alias("images"),
        F.countDistinct("participant_id").alias("participants"),
    )
    .orderBy("visit", "age_link_status")
)
if (
    require_nonzero_age_coverage
    and not manifest_spark.filter(F.col("age").isNotNull()).limit(1).count()
):
    raise RuntimeError(
        "No images have visit-matched age. Run notebook 03 to build "
        "sap_fundus_image_analysis, confirm age_source_path, and rerun 02. "
        "BL requires AGE_NMBR_COM; F1 requires AGE_NMBR_COF1."
    )
if exclude_images_without_age:
    excluded_without_age = manifest_spark.filter(
        F.col("age").isNull()
    ).count()
    print(
        "Images excluded because visit-matched age is unavailable:",
        excluded_without_age,
    )
    manifest_spark = manifest_spark.filter(F.col("age").isNotNull())

manifest_spark = manifest_spark.orderBy(
    "participant_id",
    "eye",
    "image_path",
)
if max_images > 0:
    manifest_spark = manifest_spark.limit(max_images)

if run_all_images:
    print(
        "run_all_images=true: processing every eligible distinct image path; "
        "BL and F1 counts are intentionally allowed to differ."
    )
    if "visit" not in manifest_spark.columns:
        raise ValueError(
            "The full manifest must contain a visit column identifying BL/F1."
        )
    display(
        manifest_spark.groupBy("visit")
        .agg(
            F.count("*").alias("images"),
            F.countDistinct("participant_id").alias("participants"),
        )
        .orderBy("visit")
    )

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


def refresh_cached_metadata(cached, current):
    """Keep derived results but replace source metadata by image path."""
    metadata_columns = [
        column for column in current.columns if column != "image_path"
    ]
    derived = cached.drop(columns=metadata_columns, errors="ignore")
    metadata = current[["image_path", *metadata_columns]].copy()
    return derived.merge(
        metadata,
        on="image_path",
        how="left",
        validate="one_to_one",
        sort=False,
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Technical image quality and AutoMorph-style preprocessing

# COMMAND ----------
quality_config = QualityConfig(
    output_size=256,
    model_input_size=224,
    save_preprocessed=save_preprocessed,
)
quality_output_root = output_root / "01_quality"
if run_all_images:
    quality_batch_frames = []
    quality_batches_root = quality_output_root / "batches"
    quality_batches_root.mkdir(parents=True, exist_ok=True)
    n_quality_batches = (
        len(manifest) + pipeline_batch_size - 1
    ) // pipeline_batch_size

    for batch_number, start in enumerate(
        range(0, len(manifest), pipeline_batch_size),
        start=1,
    ):
        stop = min(start + pipeline_batch_size, len(manifest))
        manifest_batch = manifest.iloc[start:stop].copy()
        batch_root = quality_batches_root / (
            f"batch_{start:09d}_{stop:09d}"
        )
        batch_cache = batch_root / "fundus_quality_manifest.parquet"
        quality_batch = None

        if resume_batches and batch_cache.exists():
            try:
                cached = pd.read_parquet(batch_cache)
                expected_paths = set(
                    manifest_batch["image_path"].astype(str)
                )
                cached_paths = set(cached["image_path"].astype(str))
                if (
                    len(cached) == len(manifest_batch)
                    and cached_paths == expected_paths
                ):
                    quality_batch = refresh_cached_metadata(
                        cached,
                        manifest_batch,
                    )
                    print(
                        f"[quality {batch_number}/{n_quality_batches}] "
                        f"resumed {len(cached):,} images from {batch_cache}",
                        flush=True,
                    )
                else:
                    print(
                        f"[quality {batch_number}/{n_quality_batches}] "
                        "cache does not match the current manifest; recomputing.",
                        flush=True,
                    )
            except Exception as exc:
                print(
                    f"[quality {batch_number}/{n_quality_batches}] "
                    f"cache is unreadable ({type(exc).__name__}); recomputing.",
                    flush=True,
                )

        if quality_batch is None:
            print(
                f"[quality {batch_number}/{n_quality_batches}] processing "
                f"manifest rows {start:,}:{stop:,}",
                flush=True,
            )
            quality_batch = run_quality_pipeline(
                manifest_batch,
                batch_root,
                quality_config,
            )
            print(
                f"[quality {batch_number}/{n_quality_batches}] saved "
                f"{len(quality_batch):,} rows to {batch_cache}",
                flush=True,
            )
        quality_batch_frames.append(quality_batch)

    quality = (
        pd.concat(quality_batch_frames, ignore_index=True)
        .drop_duplicates(subset=["image_path"], keep="last")
    )
    if len(quality) != len(manifest):
        raise RuntimeError(
            "Checkpointed quality rows do not match the full manifest: "
            f"quality={len(quality):,}, manifest={len(manifest):,}."
        )
    write_frame(
        quality,
        quality_output_root / "fundus_quality_manifest.parquet",
    )
    quality.drop(columns=["embedding"], errors="ignore").to_csv(
        quality_output_root / "fundus_quality_manifest.csv",
        index=False,
    )
    write_json(
        {
            "n_images": int(len(quality)),
            "n_pass": int(quality["quality_pass"].sum()),
            "n_fail": int((~quality["quality_pass"]).sum()),
            "pass_rate": float(quality["quality_pass"].mean()),
            "pipeline_batch_size": pipeline_batch_size,
            "n_batches": n_quality_batches,
        },
        quality_output_root / "fundus_quality_summary.json",
    )
else:
    quality = run_quality_pipeline(
        manifest,
        quality_output_root,
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

embedding_output_root = output_root / "02_embeddings"
if run_all_images:
    embedding_batch_frames = []
    embedding_failure_frames = []
    embedding_batches_root = embedding_output_root / "batches"
    embedding_batches_root.mkdir(parents=True, exist_ok=True)
    passing_for_batches = passing_quality.reset_index(drop=True)
    n_embedding_batches = (
        len(passing_for_batches) + pipeline_batch_size - 1
    ) // pipeline_batch_size

    for batch_number, start in enumerate(
        range(0, len(passing_for_batches), pipeline_batch_size),
        start=1,
    ):
        stop = min(start + pipeline_batch_size, len(passing_for_batches))
        quality_batch = passing_for_batches.iloc[start:stop].copy()
        batch_root = embedding_batches_root / (
            f"batch_{start:09d}_{stop:09d}"
        )
        batch_cache = batch_root / "retfound_embeddings.parquet"
        batch_failures_path = (
            batch_root / "retfound_embedding_failures.csv"
        )
        embedding_batch = None
        expected_paths = set(quality_batch["image_path"].astype(str))

        if resume_batches and batch_cache.exists():
            try:
                cached = pd.read_parquet(batch_cache)
                cached_paths = set(cached["image_path"].astype(str))
                cached_failure_paths = read_embedding_failure_paths(
                    batch_failures_path
                )
                accounted_paths = cached_paths | cached_failure_paths
                if (
                    accounted_paths == expected_paths
                    and not (cached_paths & cached_failure_paths)
                ):
                    embedding_batch = refresh_cached_metadata(
                        cached,
                        quality_batch,
                    )
                    print(
                        f"[RETFound {batch_number}/{n_embedding_batches}] "
                        f"resumed {len(cached):,} vectors and "
                        f"{len(cached_failure_paths):,} failures.",
                        flush=True,
                    )
                else:
                    print(
                        f"[RETFound {batch_number}/{n_embedding_batches}] "
                        "cache is incomplete or stale; recomputing this batch.",
                        flush=True,
                    )
            except Exception as exc:
                print(
                    f"[RETFound {batch_number}/{n_embedding_batches}] "
                    f"cache is unreadable ({type(exc).__name__}); recomputing.",
                    flush=True,
                )

        if embedding_batch is None:
            print(
                f"[RETFound {batch_number}/{n_embedding_batches}] processing "
                f"quality-passing rows {start:,}:{stop:,}",
                flush=True,
            )
            embedding_batch = extract_retfound_embeddings(
                quality_batch,
                batch_root,
                retfound_config,
                quality_config,
                model=model,
                device=device,
                checkpoint_path=resolved_checkpoint,
                force=True,
            )
            print(
                f"[RETFound {batch_number}/{n_embedding_batches}] saved "
                f"{len(embedding_batch):,} vectors to {batch_cache}",
                flush=True,
            )

        embedding_batch_frames.append(embedding_batch)
        if batch_failures_path.exists():
            try:
                embedding_failure_frames.append(
                    pd.read_csv(batch_failures_path)
                )
            except pd.errors.EmptyDataError:
                pass

    embeddings = (
        pd.concat(embedding_batch_frames, ignore_index=True)
        .drop_duplicates(subset=["image_path"], keep="last")
    )
    failures = (
        pd.concat(embedding_failure_frames, ignore_index=True)
        if embedding_failure_frames
        else pd.DataFrame(columns=["image_path", "error"])
    )
    failures = failures.drop_duplicates(subset=["image_path"], keep="last")
    write_frame(
        embeddings,
        embedding_output_root / "retfound_embeddings.parquet",
    )
    failures.to_csv(
        embedding_output_root / "retfound_embedding_failures.csv",
        index=False,
    )
    write_json(
        {
            "n_input_quality_passing": int(len(passing_for_batches)),
            "n_embedded": int(len(embeddings)),
            "n_failed": int(len(failures)),
            "embedding_dim": int(embeddings["embedding_dim"].iloc[0]),
            "pipeline_batch_size": pipeline_batch_size,
            "n_batches": n_embedding_batches,
            "device": device,
            "checkpoint": str(resolved_checkpoint),
        },
        embedding_output_root / "retfound_embedding_metadata.json",
    )
else:
    embeddings = extract_retfound_embeddings(
        quality,
        embedding_output_root,
        retfound_config,
        quality_config,
        model=model,
        device=device,
        checkpoint_path=resolved_checkpoint,
        force=force_embeddings,
    )
print("embedding rows:", len(embeddings))
print("embedding dimension:", embeddings["embedding_dim"].unique())

expected_shape = (len(embeddings), expected_embedding_dim)
if run_all_images:
    # Validate one saved vector at a time to avoid allocating an additional
    # approximately 0.5 GiB dense matrix on the driver for the full cohort.
    min_norm = float("inf")
    max_norm = 0.0
    preview_norms = []
    for vector_number, vector in enumerate(embeddings["embedding"]):
        array = np.asarray(vector, dtype=np.float32)
        if array.shape != (expected_embedding_dim,):
            raise ValueError(
                f"Vector {vector_number} has shape {array.shape}; expected "
                f"({expected_embedding_dim},)."
            )
        if not np.isfinite(array).all():
            raise ValueError(
                f"RETFound vector {vector_number} contains NaN or infinity."
            )
        norm = float(np.linalg.norm(array))
        if norm <= 0:
            raise ValueError(
                f"RETFound vector {vector_number} has zero length."
            )
        min_norm = min(min_norm, norm)
        max_norm = max(max_norm, norm)
        if vector_number < 20:
            preview_norms.append(norm)
else:
    embedding_matrix = np.stack(embeddings["embedding"].to_numpy()).astype(
        np.float32
    )
    if embedding_matrix.shape != expected_shape:
        raise ValueError(
            f"Expected embedding matrix {expected_shape}, got "
            f"{embedding_matrix.shape}."
        )
    if not np.isfinite(embedding_matrix).all():
        raise ValueError("RETFound vectors contain NaN or infinity.")
    norms = np.linalg.norm(embedding_matrix, axis=1)
    if not np.all(norms > 0):
        raise ValueError("At least one RETFound vector has zero length.")
    min_norm = float(norms.min())
    max_norm = float(norms.max())
    preview_norms = [float(value) for value in norms[:20]]

quality_paths = set(passing_quality["image_path"].astype(str))
embedded_paths = set(embeddings["image_path"].astype(str))
failure_path = output_root / "02_embeddings" / "retfound_embedding_failures.csv"
failure_paths = read_embedding_failure_paths(failure_path)
unaccounted_paths = quality_paths - embedded_paths - failure_paths
cached_extra_paths = embedded_paths - quality_paths
if unaccounted_paths or cached_extra_paths:
    raise RuntimeError(
        "The embedding cache does not match this manifest. Use a new "
        "output_root, or set resume_batches=false for a clean batch rerun. "
        f"Unaccounted={len(unaccounted_paths)}, extra={len(cached_extra_paths)}."
    )

print("Final embedding matrix shape:", expected_shape)
print("Vector dtype: float32")
print("Vector L2 norm range:", min_norm, max_norm)

# Load the durable Parquet result directly into Spark. Avoid converting every
# 1,024-element vector to a Python list on the driver during a full run.
embedding_parquet_path = str(
    embedding_output_root / "retfound_embeddings.parquet"
)
embeddings_spark = spark.read.parquet(embedding_parquet_path)
vectors_spark = embeddings_spark.select(
    F.col("participant_id").cast("string").alias("participant_id"),
    F.col("visit").cast("string").alias("visit")
    if "visit" in embeddings_spark.columns
    else F.lit("").alias("visit"),
    F.col("eye").cast("string").alias("eye")
    if "eye" in embeddings_spark.columns
    else F.lit("").alias("eye"),
    F.col("image_path").cast("string").alias("image_path"),
    F.col("embedding_dim").cast("int").alias("embedding_dim"),
    F.col("retfound_model").cast("string").alias("retfound_model"),
    F.col("retfound_checkpoint_sha256")
    .cast("string")
    .alias("retfound_checkpoint_sha256"),
    F.col("embedding").cast("array<float>").alias("embedding"),
)
vectors_delta_path = str(
    output_root / "02_embeddings" / "retfound_embeddings_delta"
)
(
    vectors_spark.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(vectors_delta_path)
)

preview_rows = []
for row_number, (_, record) in enumerate(embeddings.head(20).iterrows()):
    preview_rows.append(
        {
            "participant_id": str(record["participant_id"]),
            "visit": str(record.get("visit", "")),
            "eye": str(record.get("eye", "")),
            "embedding_dim": int(record["embedding_dim"]),
            "l2_norm": preview_norms[row_number],
            "first_8_values": [
                float(value) for value in record["embedding"][:8]
            ],
        }
    )
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
