# Databricks notebook source
# MAGIC %md
# MAGIC # Live RETFound vectorizer for completed quality batches
# MAGIC
# MAGIC Run this notebook in a second Databricks notebook while notebook 02 is
# MAGIC still producing technical-quality batches. It watches
# MAGIC `01_quality/batches`, consumes only complete batch Parquets, retains only
# MAGIC rows with `quality_pass=True`, and immediately checkpoints their
# MAGIC 1,024-element RETFound vectors.
# MAGIC
# MAGIC A quality file is considered complete only when it is readable, its
# MAGIC directory name has the expected `batch_START_STOP` form, it contains
# MAGIC exactly `STOP - START` rows, and every `image_path` is nonmissing and
# MAGIC unique. Thus the batch currently being written is never consumed.
# MAGIC
# MAGIC Stop/restart is safe. Each quality batch has an independent RETFound
# MAGIC output and completion marker. The notebook polls until all expected
# MAGIC quality batches have been vectorized.

# COMMAND ----------
# MAGIC %md
# MAGIC On a fresh GPU cluster, first run and restart Python:
# MAGIC
# MAGIC ```python
# MAGIC %pip install -r /Workspace/Users/ad0038@pennmedicine.upenn.edu/CLSA/CLSA_retina/requirements-retfound.txt
# MAGIC dbutils.library.restartPython()
# MAGIC ```

# COMMAND ----------
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from pyspark.sql import functions as F

# COMMAND ----------
dbutils.widgets.text("hf_token", "", "Hugging Face token (temporary)")

# COMMAND ----------
from pathlib import Path

repo_root = "/Workspace/Users/ad0038@pennmedicine.upenn.edu/CLSA/CLSA_retina"
derived_root = (
    "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/derived/"
    "clsa_retinal_aging/fundus_retfound"
)
quality_batches_root = Path(f"{derived_root}/01_quality/batches")
embedding_output_root = Path(f"{derived_root}/02_embeddings")
quality_pass_columns = ["quality_pass"]
expected_quality_batches = 213
poll_seconds = 30
heartbeat_seconds = 5 * 60
retfound_repo = None
checkpoint_path = None
allow_downloads = True
device_requested = "cuda"
gpu_batch_size = 8
resume_batches = True
expected_embedding_dim = 1024

if expected_quality_batches < 1:
    raise ValueError("expected_quality_batches must be at least 1.")
if poll_seconds < 5 or poll_seconds > 300:
    raise ValueError("poll_seconds must be between 5 and 300.")
if heartbeat_seconds < 60:
    raise ValueError("heartbeat_minutes must be at least 1.")
if gpu_batch_size < 1:
    raise ValueError("gpu_batch_size must be at least 1.")
if not quality_pass_columns:
    raise ValueError("At least one explicit quality pass column is required.")
if not quality_batches_root.exists():
    raise FileNotFoundError(
        f"Quality batch directory does not exist: {quality_batches_root}"
    )

module_root = Path(repo_root) / "src"
if str(module_root) not in sys.path:
    sys.path.insert(0, str(module_root))

from fundus_retfound_pipeline import (  # noqa: E402
    QualityConfig,
    RETFoundConfig,
    extract_retfound_embeddings,
    load_retfound_model,
    read_embedding_failure_paths,
    write_frame,
    write_json,
)
from quality_passed_retfound import (  # noqa: E402
    parse_quality_batch_name,
    read_completed_quality_batch,
    select_quality_passed,
)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
if device_requested == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested, but this compute has no CUDA GPU.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Configure the producer/consumer handoff
# MAGIC
# MAGIC For the run shown in the quality log, keep:
# MAGIC
# MAGIC ```text
# MAGIC expected_quality_batches=213
# MAGIC quality_pass_columns=quality_pass
# MAGIC poll_seconds=30
# MAGIC gpu_batch_size=8
# MAGIC resume_batches=true
# MAGIC ```
# MAGIC
# MAGIC If a later AutoMorph step adds a separate validated pass flag to every
# MAGIC batch, use `quality_pass,automorph_quality_pass`. The image must then
# MAGIC explicitly pass both flags. Missing or unfamiliar flag values stop the
# MAGIC run instead of being interpreted as passes.

# COMMAND ----------
retfound_config = RETFoundConfig(
    repo_path=retfound_repo,
    checkpoint_path=checkpoint_path,
    allow_downloads=allow_downloads,
    device=device_requested,
    batch_size=gpu_batch_size,
)
quality_config = QualityConfig(
    output_size=256,
    model_input_size=224,
    save_preprocessed=False,
)
embedding_batches_root = embedding_output_root / "batches"
embedding_batches_root.mkdir(parents=True, exist_ok=True)
progress_path = embedding_output_root / "retfound_live_progress.json"

model = None
resolved_device = None
resolved_repo = None
resolved_checkpoint = None


def ensure_retfound_loaded() -> None:
    global model, resolved_device, resolved_repo, resolved_checkpoint
    if model is not None:
        return
    temporary_hf_token = dbutils.widgets.get("hf_token").strip()
    if allow_downloads and not checkpoint_path and not temporary_hf_token:
        raise ValueError(
            "Enter the temporary Hugging Face token, or provide checkpoint_path."
        )
    if temporary_hf_token:
        os.environ["HF_TOKEN"] = temporary_hf_token
    try:
        model, resolved_device, resolved_repo, resolved_checkpoint = (
            load_retfound_model(retfound_config)
        )
    finally:
        os.environ.pop("HF_TOKEN", None)
        temporary_hf_token = ""
    if device_requested == "cuda" and resolved_device != "cuda":
        raise RuntimeError(
            "Expected CUDA, but RETFound resolved device="
            f"{resolved_device!r}."
        )
    print("Device:", resolved_device)
    print("RETFound repository:", resolved_repo)
    print("Checkpoint:", resolved_checkpoint)


def image_path_digest(paths) -> str:
    digest = hashlib.sha256()
    for path in sorted(str(value) for value in paths):
        digest.update(path.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def cached_batch_is_complete(
    batch_root: Path,
    expected_paths: set[str],
    source_digest: str,
) -> bool:
    completion_path = batch_root / "quality_handoff_complete.json"
    completion = read_json(completion_path)
    if not completion or completion.get("quality_image_paths_sha256") != source_digest:
        return False
    cache_path = batch_root / "retfound_embeddings.parquet"
    failures_path = batch_root / "retfound_embedding_failures.csv"
    embedded_paths = set()
    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path, columns=["image_path"])
            embedded_paths = set(cached["image_path"].astype(str))
        except Exception:
            return False
    failure_paths = read_embedding_failure_paths(failures_path)
    return (
        embedded_paths | failure_paths == expected_paths
        and not embedded_paths & failure_paths
    )


def process_quality_batch(batch_name: str, quality_batch) -> dict:
    passing, audit = select_quality_passed(
        quality_batch,
        quality_pass_columns,
    )
    passing = passing.sort_values("image_path", kind="stable").reset_index(
        drop=True
    )
    expected_paths = set(passing["image_path"].astype(str))
    source_digest = image_path_digest(expected_paths)
    batch_root = embedding_batches_root / batch_name
    batch_root.mkdir(parents=True, exist_ok=True)
    completion_path = batch_root / "quality_handoff_complete.json"

    if resume_batches and cached_batch_is_complete(
        batch_root,
        expected_paths,
        source_digest,
    ):
        completion = read_json(completion_path)
        print(
            f"[live RETFound] resumed {batch_name}: "
            f"quality={len(audit):,}, passed={len(passing):,}, "
            f"embedded={completion['n_embedded']:,}, "
            f"failed={completion['n_failed']:,}",
            flush=True,
        )
        return completion

    if passing.empty:
        embeddings = pd.DataFrame()
        failures = pd.DataFrame(columns=["image_path", "error"])
    else:
        ensure_retfound_loaded()
        print(
            f"[live RETFound] processing {batch_name}: "
            f"{len(passing):,}/{len(audit):,} images passed QC.",
            flush=True,
        )
        embeddings = extract_retfound_embeddings(
            passing,
            batch_root,
            retfound_config,
            quality_config,
            model=model,
            device=resolved_device,
            checkpoint_path=resolved_checkpoint,
            force=True,
        )
        failures_path = batch_root / "retfound_embedding_failures.csv"
        if failures_path.exists() and failures_path.stat().st_size:
            try:
                failures = pd.read_csv(failures_path)
            except pd.errors.EmptyDataError:
                failures = pd.DataFrame(columns=["image_path", "error"])
        else:
            failures = pd.DataFrame(columns=["image_path", "error"])

    embedded_paths = (
        set(embeddings["image_path"].astype(str))
        if not embeddings.empty
        else set()
    )
    failure_paths = set(failures["image_path"].dropna().astype(str))
    if (
        embedded_paths | failure_paths != expected_paths
        or embedded_paths & failure_paths
    ):
        raise RuntimeError(
            f"Batch {batch_name} does not exactly account for its "
            "quality-passing images."
        )
    if not embeddings.empty:
        dimensions = sorted(
            embeddings["embedding_dim"].dropna().astype(int).unique()
        )
        if dimensions != [expected_embedding_dim]:
            raise ValueError(
                f"Batch {batch_name} produced dimensions {dimensions}; "
                f"expected [{expected_embedding_dim}]."
            )

    completion = {
        "quality_batch": batch_name,
        "quality_source_path": str(
            quality_batches_root
            / batch_name
            / "fundus_quality_manifest.parquet"
        ),
        "quality_pass_columns": quality_pass_columns,
        "quality_rows": int(len(audit)),
        "quality_passing_rows": int(len(passing)),
        "quality_image_paths_sha256": source_digest,
        "n_embedded": int(len(embedded_paths)),
        "n_failed": int(len(failure_paths)),
        "embedding_dim": expected_embedding_dim,
        "checkpoint": (
            str(resolved_checkpoint) if resolved_checkpoint else None
        ),
    }
    write_json(completion, completion_path)
    print(
        f"[live RETFound] completed {batch_name}: "
        f"embedded={len(embedded_paths):,}, failed={len(failure_paths):,}",
        flush=True,
    )
    return completion


def complete_contiguous_producer_sequence(batch_names) -> bool:
    if len(batch_names) != expected_quality_batches:
        return False
    intervals = sorted(parse_quality_batch_name(name) for name in batch_names)
    if intervals[0][0] != 0:
        return False
    return all(
        current_start == previous_stop
        for (_, previous_stop), (current_start, _) in zip(
            intervals,
            intervals[1:],
        )
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## Live consumer loop
# MAGIC
# MAGIC This cell remains active while the producer runs. A heartbeat is printed
# MAGIC every five minutes by default. Canceling this cell is safe; rerunning it
# MAGIC validates and resumes all completed outputs.

# COMMAND ----------
handled_signatures = {}
batch_results = {}
last_heartbeat = 0.0
started_at = time.time()

print(
    "Watching for",
    expected_quality_batches,
    "completed quality batches under",
    quality_batches_root,
    flush=True,
)

while True:
    candidate_paths = sorted(
        quality_batches_root.glob(
            "batch_*/fundus_quality_manifest.parquet"
        )
    )
    for quality_path in candidate_paths:
        batch_name = quality_path.parent.name
        try:
            parse_quality_batch_name(batch_name)
            stat = quality_path.stat()
        except (ValueError, FileNotFoundError):
            continue
        signature = (int(stat.st_size), int(stat.st_mtime_ns))
        if handled_signatures.get(batch_name) == signature:
            continue

        quality_batch = read_completed_quality_batch(quality_path)
        if quality_batch is None:
            continue
        result = process_quality_batch(batch_name, quality_batch)
        handled_signatures[batch_name] = signature
        batch_results[batch_name] = result
        write_json(
            {
                "expected_quality_batches": expected_quality_batches,
                "completed_vector_batches": len(batch_results),
                "latest_completed_batch": batch_name,
                "quality_pass_columns": quality_pass_columns,
                "elapsed_hours": (time.time() - started_at) / 3600.0,
            },
            progress_path,
        )

    now = time.time()
    if now - last_heartbeat >= heartbeat_seconds:
        print(
            "[live RETFound heartbeat] "
            f"vectorized {len(batch_results)}/{expected_quality_batches} "
            f"complete quality batches; discovered {len(candidate_paths)} "
            "batch paths.",
            flush=True,
        )
        last_heartbeat = now

    if len(batch_results) > expected_quality_batches:
        raise RuntimeError(
            "More completed quality batches were discovered than expected. "
            "This output directory may contain stale batches from another run."
        )
    if complete_contiguous_producer_sequence(batch_results):
        print(
            "All expected quality batches have durable RETFound handoff "
            "markers.",
            flush=True,
        )
        break
    time.sleep(poll_seconds)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Consolidate completed vectors
# MAGIC
# MAGIC This runs only after all 213 expected quality batches have been handled.
# MAGIC Per-batch Parquets remain the recovery source if consolidation is
# MAGIC interrupted.

# COMMAND ----------
embedding_frames = []
failure_frames = []
for batch_name in sorted(batch_results):
    batch_root = embedding_batches_root / batch_name
    embedding_path = batch_root / "retfound_embeddings.parquet"
    failure_path = batch_root / "retfound_embedding_failures.csv"
    if embedding_path.exists():
        embedding_frames.append(pd.read_parquet(embedding_path))
    if failure_path.exists() and failure_path.stat().st_size:
        try:
            failure_frames.append(pd.read_csv(failure_path))
        except pd.errors.EmptyDataError:
            pass

if not embedding_frames:
    raise RuntimeError("No quality-passing images produced RETFound vectors.")
embeddings = pd.concat(embedding_frames, ignore_index=True)
failures = (
    pd.concat(failure_frames, ignore_index=True)
    if failure_frames
    else pd.DataFrame(columns=["image_path", "error"])
)
if embeddings["image_path"].astype(str).duplicated().any():
    raise RuntimeError("Duplicate image paths exist across RETFound batches.")
if not failures.empty and failures["image_path"].astype(str).duplicated().any():
    raise RuntimeError("Duplicate failure paths exist across RETFound batches.")
embedded_paths = set(embeddings["image_path"].astype(str))
failure_paths = set(failures["image_path"].dropna().astype(str))
if embedded_paths & failure_paths:
    raise RuntimeError("An image appears in both embedded and failed outputs.")

embedding_output_root.mkdir(parents=True, exist_ok=True)
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
        "producer_consumer_mode": "live_completed_quality_batches",
        "quality_batches_root": str(quality_batches_root),
        "quality_pass_columns": quality_pass_columns,
        "expected_quality_batches": expected_quality_batches,
        "completed_quality_batches": len(batch_results),
        "n_input_quality_passing": int(len(embedded_paths | failure_paths)),
        "n_embedded": int(len(embedded_paths)),
        "n_failed": int(len(failure_paths)),
        "embedding_dim": expected_embedding_dim,
        "gpu_batch_size": gpu_batch_size,
        "device": resolved_device,
        "checkpoint": (
            str(resolved_checkpoint) if resolved_checkpoint else None
        ),
    },
    embedding_output_root / "retfound_embedding_metadata.json",
)

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
    embedding_output_root / "retfound_embeddings_delta"
)
(
    vectors_spark.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(vectors_delta_path)
)

display(
    vectors_spark.groupBy("visit").agg(
        F.count("*").alias("embedded_images"),
        F.countDistinct("participant_id").alias("participants"),
    )
)
print("RETFound vectors:", f"{len(embeddings):,}")
print("Embedding failures:", f"{len(failures):,}")
print("Parquet vectors:", embedding_parquet_path)
print("Delta vectors:", vectors_delta_path)
