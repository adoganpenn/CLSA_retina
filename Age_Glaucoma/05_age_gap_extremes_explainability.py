# Databricks notebook source
# MAGIC %md
# MAGIC # Extreme retinal-age-gap explainability and physiology-proxy analysis
# MAGIC
# MAGIC This notebook analyzes the participant-level top and bottom age-gap
# MAGIC deciles in the matched CLSA and Zeiss cohorts. The primary analysis uses
# MAGIC one deterministic image per participant, calculates deciles within each
# MAGIC source, verifies exact reproduction of every stored RETFound vector, and
# MAGIC compares signed patch contributions with participant-level permutation
# MAGIC inference.
# MAGIC
# MAGIC Image-derived vessel, optic-disc, central-retina, peripheral-retina,
# MAGIC border, and high-gradient masks are exploratory **physiology proxies**.
# MAGIC They are not validated segmentations or diagnoses. Their purpose is to
# MAGIC distinguish plausible anatomic attention from camera borders, brightness,
# MAGIC and texture artifacts.

# COMMAND ----------
# MAGIC %pip install "timm==1.0.28" pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
from pathlib import Path
import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage, stats

# COMMAND ----------
dbutils.widgets.text("hf_token", "", "Hugging Face token (temporary)")

# COMMAND ----------
from pathlib import Path

repo_root = Path(
    "/Workspace/Users/ad0038@pennmedicine.upenn.edu/CLSA/CLSA_retina"
)
output_root = Path(
    "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/derived/"
    "clsa_retinal_aging/Age_Glaucoma"
)
retfound_repo = None
checkpoint_path = None
checkpoint_cache_path = Path(
    "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/derived/"
    "clsa_retinal_aging/model_checkpoints/RETFound_mae_natureCFP.pth"
)
allow_downloads = False
allow_repo_clone = True
hf_token = dbutils.widgets.get("hf_token").strip()
device_requested = "cuda"
extreme_quantile = 0.10
selection_metric = "raw_gap"
pipeline_batch_size = 25
resume_batches = True
max_participants_per_group = 0
n_permutations = 2000
age_match_caliper_years = 3.0
embedding_cosine_threshold = 0.99
align_left_eyes = True
n_examples_per_group = 2

if not math.isclose(extreme_quantile, 0.10, rel_tol=0, abs_tol=1e-12):
    raise ValueError(
        "This prespecified notebook is labeled and tested for top/bottom 10%; "
        "extreme_quantile must remain 0.10."
    )
if selection_metric not in {"raw_gap", "age_adjusted_gap"}:
    raise ValueError("selection_metric must be raw_gap or age_adjusted_gap")
if pipeline_batch_size < 1:
    raise ValueError("pipeline_batch_size must be positive")
if max_participants_per_group < 0:
    raise ValueError("max_participants_per_group cannot be negative")
if n_permutations < 100:
    raise ValueError("n_permutations must be at least 100")
if age_match_caliper_years < 0:
    raise ValueError("age_match_caliper_years cannot be negative")
if not 0 <= embedding_cosine_threshold <= 1:
    raise ValueError("embedding_cosine_threshold must be between 0 and 1")
if n_examples_per_group < 1:
    raise ValueError("n_examples_per_group must be positive")
if device_requested == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested, but this compute has no CUDA GPU")

module_root = repo_root / "src"
if not module_root.exists():
    raise FileNotFoundError(f"Repository source directory is missing: {module_root}")
if str(module_root) not in sys.path:
    sys.path.insert(0, str(module_root))

import fundus_retfound_pipeline as _fundus_pipeline  # noqa: E402
import age_glaucoma_model as _age_glaucoma_model  # noqa: E402
import age_gap_extremes as _extreme_module  # noqa: E402

_fundus_pipeline = importlib.reload(_fundus_pipeline)
_age_glaucoma_model = importlib.reload(_age_glaucoma_model)
_extreme_module = importlib.reload(_extreme_module)
if not getattr(_fundus_pipeline, "PARQUET_RUNTIME_ATTRS_SAFE", False):
    raise RuntimeError("The loaded fundus pipeline has the old Parquet writer")
if not hasattr(_age_glaucoma_model, "load_zeiss_retfound_model"):
    raise RuntimeError("The loaded age/glaucoma module lacks the Zeiss encoder")
if not hasattr(_extreme_module, "permutation_patch_comparison"):
    raise RuntimeError("The loaded age-gap-extremes helper is stale")

from age_gap_extremes import (  # noqa: E402
    attribution_physiology_metrics,
    benjamini_hochberg,
    build_participant_extremes,
    fundus_physiology_proxies,
    paired_permutation_patch_comparison,
    permutation_patch_comparison,
)
from age_glaucoma_model import (  # noqa: E402
    attribution_group_statistics,
    exact_patch_map_from_array,
    load_zeiss_retfound_model,
    prepare_zeiss_dicom_input,
)
from fundus_retfound_pipeline import (  # noqa: E402
    QualityConfig,
    RETFoundConfig,
    load_age_head,
    load_retfound_model,
    prepare_model_input,
    preprocess_fundus,
    sha256_file,
    write_frame,
    write_json,
)
print("Loaded fundus pipeline:", _fundus_pipeline.__file__)
print("Loaded age-gap helper:", _extreme_module.__file__)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Assemble participant-level extreme deciles
# MAGIC
# MAGIC Group membership is defined from the participant mean age gap, not from
# MAGIC individual eyes. Decile thresholds are calculated separately within CLSA
# MAGIC and Zeiss to prevent acquisition source from defining the extremes.

# COMMAND ----------
model_path = output_root / "06_CLSA_healthy_model" / "CLSA_healthy.joblib"
comparison_root = output_root / "08_matched_comparison"
zeiss_images_path = comparison_root / "zeiss_matched_images_for_explainability.parquet"
clsa_images_path = comparison_root / "clsa_matched_images_for_explainability.parquet"
analysis_root = output_root / "10_age_gap_extremes"
cohort_root = analysis_root / "01_cohort"
attribution_root = analysis_root / "02_attributions"
statistics_root = analysis_root / "03_statistics"
figure_root = analysis_root / "04_figures"
for path in (cohort_root, attribution_root, statistics_root, figure_root):
    path.mkdir(parents=True, exist_ok=True)
for path in (model_path, zeiss_images_path, clsa_images_path):
    if not path.exists():
        raise FileNotFoundError(f"Required notebook-03 output is missing: {path}")

zeiss_raw = pd.read_parquet(zeiss_images_path)
clsa_raw = pd.read_parquet(clsa_images_path)
zeiss_raw.attrs = {}
clsa_raw.attrs = {}


def normalized_image_table(frame, cohort, gap_column, prediction_column):
    required = {"participant_id", "image_path", "age", "embedding", gap_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{cohort} input is missing columns: {sorted(missing)}")
    output = pd.DataFrame(
        {
            "cohort": cohort,
            "participant_id": frame["participant_id"].astype(str),
            "image_path": frame["image_path"].astype(str),
            "age": pd.to_numeric(frame["age"], errors="coerce"),
            "age_gap": pd.to_numeric(frame[gap_column], errors="coerce"),
            "retinal_age_prediction": pd.to_numeric(
                frame[prediction_column], errors="coerce"
            ),
            "embedding": frame["embedding"],
        }
    )
    for target, candidates in {
        "eye": ("eye", "laterality"),
        "sex": ("sex",),
        "visit": ("visit",),
        "retina_fraction": ("retina_fraction", "automorph_retina_fraction"),
        "brightness_mean": ("brightness_mean",),
        "contrast_std": ("contrast_std",),
        "gradient_energy": ("gradient_energy",),
        "quality_pass": ("quality_pass",),
    }.items():
        source = next((name for name in candidates if name in frame.columns), None)
        output[target] = frame[source] if source else np.nan
    return output


zeiss_images = normalized_image_table(
    zeiss_raw,
    "Zeiss glaucoma",
    "retinal_age_gap",
    "retinal_age_prediction",
)
clsa_images = normalized_image_table(
    clsa_raw,
    "CLSA healthy",
    "retinal_age_gap_oof",
    "retinal_age_prediction_oof",
)
all_images = pd.concat([zeiss_images, clsa_images], ignore_index=True)
participant_cohort, representative_images, extreme_thresholds = (
    build_participant_extremes(
        all_images,
        quantile=extreme_quantile,
        selection_metric=selection_metric,
    )
)

# Calculate the alternate definition for a prespecified sensitivity audit.
alternate_metric = (
    "age_adjusted_gap" if selection_metric == "raw_gap" else "raw_gap"
)
alternate_participants, _, alternate_thresholds = build_participant_extremes(
    all_images,
    quantile=extreme_quantile,
    selection_metric=alternate_metric,
)
membership_audit = participant_cohort[
    ["cohort", "participant_id", "extreme_group"]
].merge(
    alternate_participants[
        ["cohort", "participant_id", "extreme_group"]
    ].rename(columns={"extreme_group": "alternate_extreme_group"}),
    on=["cohort", "participant_id"],
    how="inner",
    validate="one_to_one",
)
membership_summary = (
    membership_audit.assign(
        same_assignment=lambda value: (
            value["extreme_group"] == value["alternate_extreme_group"]
        )
    )
    .groupby("cohort", as_index=False)
    .agg(
        participants=("participant_id", "size"),
        same_assignment=("same_assignment", "sum"),
    )
)
membership_summary["assignment_agreement"] = (
    membership_summary["same_assignment"] / membership_summary["participants"]
)
definition_overlap_rows = []
for cohort in sorted(membership_audit["cohort"].unique()):
    cohort_audit = membership_audit[membership_audit["cohort"] == cohort]
    for group_name in ("bottom_10_percent", "top_10_percent"):
        primary_ids = set(
            cohort_audit.loc[
                cohort_audit["extreme_group"] == group_name, "participant_id"
            ].astype(str)
        )
        alternate_ids = set(
            cohort_audit.loc[
                cohort_audit["alternate_extreme_group"] == group_name,
                "participant_id",
            ].astype(str)
        )
        union = primary_ids | alternate_ids
        intersection = primary_ids & alternate_ids
        definition_overlap_rows.append(
            {
                "cohort": cohort,
                "extreme_group": group_name,
                "primary_participants": len(primary_ids),
                "alternate_participants": len(alternate_ids),
                "overlap_participants": len(intersection),
                "jaccard_overlap": len(intersection) / len(union) if union else np.nan,
            }
        )
definition_overlap = pd.DataFrame(definition_overlap_rows)

extreme_images = representative_images[
    representative_images["extreme_group"].isin(
        ["top_10_percent", "bottom_10_percent"]
    )
].copy()
if max_participants_per_group:
    limited = []
    for _, group in extreme_images.groupby(["cohort", "extreme_group"], sort=True):
        group = group.sort_values("participant_age_gap", kind="stable")
        if len(group) > max_participants_per_group:
            positions = np.linspace(
                0, len(group) - 1, max_participants_per_group
            ).round().astype(int)
            group = group.iloc[positions]
        limited.append(group)
    extreme_images = pd.concat(limited, ignore_index=True)

extreme_images["analysis_image_id"] = [
    hashlib.sha256(f"{cohort}|{path}".encode()).hexdigest()[:24]
    for cohort, path in zip(extreme_images["cohort"], extreme_images["image_path"])
]
extreme_images = extreme_images.sort_values(
    ["cohort", "extreme_group", "participant_id", "image_path"], kind="stable"
).reset_index(drop=True)
if extreme_images.duplicated(["cohort", "participant_id"]).any():
    raise RuntimeError("Participant duplication remains after representative-image selection")

write_frame(participant_cohort, cohort_root / "participant_age_gap_cohort.parquet")
write_frame(extreme_images, cohort_root / "extreme_representative_images.parquet")
write_frame(extreme_thresholds, cohort_root / "extreme_thresholds.csv")
write_frame(alternate_thresholds, cohort_root / "alternate_thresholds.csv")
write_frame(membership_summary, cohort_root / "definition_sensitivity.csv")
write_frame(
    definition_overlap,
    cohort_root / "raw_vs_age_adjusted_extreme_overlap.csv",
)

cohort_counts = (
    extreme_images.groupby(["cohort", "extreme_group"], as_index=False)
    .agg(images=("image_path", "size"), participants=("participant_id", "nunique"))
)
display(extreme_thresholds)
display(cohort_counts)
display(membership_summary)
display(definition_overlap)

# Baseline balance is reported before any attribution analysis.
balance_rows = []
for (cohort, group_name), group in extreme_images.groupby(
    ["cohort", "extreme_group"], sort=True
):
    row = {
        "cohort": cohort,
        "extreme_group": group_name,
        "participants": int(group["participant_id"].nunique()),
        "age_mean": float(group["age"].mean()),
        "age_sd": float(group["age"].std(ddof=1)),
        "age_gap_mean": float(group["participant_age_gap"].mean()),
        "age_gap_sd": float(group["participant_age_gap"].std(ddof=1)),
        "female_fraction": np.nan,
    }
    if group["sex"].notna().any():
        sex = group["sex"].astype(str).str.upper().str.strip()
        row["female_fraction"] = float(sex.isin(["F", "FEMALE"]).mean())
    balance_rows.append(row)
balance_summary = pd.DataFrame(balance_rows)
write_frame(balance_summary, cohort_root / "extreme_baseline_balance.csv")
display(balance_summary)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Load the frozen age head and both validated source encoders

# COMMAND ----------
age_bundle = load_age_head(model_path)
if checkpoint_path:
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"checkpoint_path is not a file: {checkpoint_path}")
elif checkpoint_cache_path and checkpoint_cache_path.is_file():
    checkpoint_path = str(checkpoint_cache_path)
    print("Using persistent checkpoint:", checkpoint_path)
elif not allow_downloads:
    raise FileNotFoundError(
        "No checkpoint is available. Retain the persistent cache path or set "
        "allow_downloads=true and enter the authorized temporary hf_token."
    )
elif not hf_token:
    raise ValueError("Enter an authorized hf_token for the gated RETFound checkpoint")

retfound_config = RETFoundConfig(
    repo_path=retfound_repo,
    checkpoint_path=checkpoint_path,
    allow_downloads=allow_downloads,
    allow_repo_clone=allow_repo_clone,
    device=device_requested,
    batch_size=1,
)
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
try:
    clsa_model, device, resolved_repo, resolved_checkpoint = load_retfound_model(
        retfound_config
    )
finally:
    os.environ.pop("HF_TOKEN", None)
    hf_token = ""

try:
    retfound_repo_commit = subprocess.check_output(
        ["git", "-C", str(resolved_repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
except (OSError, subprocess.CalledProcessError):
    retfound_repo_commit = "unknown"
print("RETFound source:", resolved_repo)
print("RETFound source commit:", retfound_repo_commit)

if (
    checkpoint_cache_path
    and Path(resolved_checkpoint) != checkpoint_cache_path
    and not checkpoint_cache_path.exists()
):
    checkpoint_cache_path.parent.mkdir(parents=True, exist_ok=True)
    partial_checkpoint = checkpoint_cache_path.with_suffix(".pth.partial")
    shutil.copy2(resolved_checkpoint, partial_checkpoint)
    if sha256_file(partial_checkpoint) != sha256_file(resolved_checkpoint):
        partial_checkpoint.unlink(missing_ok=True)
        raise RuntimeError("Persistent checkpoint hash verification failed")
    partial_checkpoint.replace(checkpoint_cache_path)
    resolved_checkpoint = checkpoint_cache_path

zeiss_model = load_zeiss_retfound_model(resolved_checkpoint, device)
quality_config = QualityConfig(
    output_size=256,
    model_input_size=224,
    save_preprocessed=False,
)
checkpoint_hash = sha256_file(resolved_checkpoint)
analysis_signature = hashlib.sha256(
    json.dumps(
        {
            "checkpoint_sha256": checkpoint_hash,
            "retfound_repo_commit": retfound_repo_commit,
            "age_model_sha256": sha256_file(model_path),
            "helper_sha256": sha256_file(module_root / "age_gap_extremes.py"),
            "selection_metric": selection_metric,
            "extreme_quantile": extreme_quantile,
            "align_left_eyes": align_left_eyes,
            "embedding_cosine_threshold": embedding_cosine_threshold,
        },
        sort_keys=True,
    ).encode()
).hexdigest()
print("Device:", device)
print("Checkpoint SHA-256:", checkpoint_hash)
print("Analysis signature:", analysis_signature[:16] + "...")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Resumable exact attributions and image-derived physiology proxies
# MAGIC
# MAGIC Each batch is durable and independently resumable. Stored embeddings must
# MAGIC reproduce with cosine ≥ the configured threshold and additive patch
# MAGIC contributions must reconstruct the frozen age-head prediction.

# COMMAND ----------
batch_root = attribution_root / "batches"
batch_root.mkdir(parents=True, exist_ok=True)


def is_left_eye(record):
    value = str(record.get("eye", "")).strip().upper()
    return value in {"L", "LEFT", "OS"}


def source_input_and_display(record):
    if record["cohort"] == "Zeiss glaucoma":
        model_array, display_image = prepare_zeiss_dicom_input(record["image_path"])
        return model_array, display_image.convert("RGB"), zeiss_model
    model_array = prepare_model_input(record["image_path"], quality_config)
    display_image = preprocess_fundus(record["image_path"], quality_config).image
    return model_array, display_image.convert("RGB"), clsa_model


def atomic_save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir="/local_disk0/tmp") as temporary_directory:
        local_path = Path(temporary_directory) / path.name
        np.savez_compressed(local_path, **arrays)
        partial_path = path.with_suffix(path.suffix + ".partial")
        shutil.copy2(local_path, partial_path)
        partial_path.replace(path)


def read_completed_batch(manifest_path, maps_path, expected_ids):
    if not (manifest_path.exists() and maps_path.exists()):
        return None
    try:
        manifest = pd.read_parquet(manifest_path)
        if set(manifest["analysis_image_id"].astype(str)) != set(expected_ids):
            return None
        if (
            "analysis_signature" not in manifest.columns
            or set(manifest["analysis_signature"].astype(str))
            != {analysis_signature}
        ):
            return None
        with np.load(maps_path) as archive:
            maps = archive["variable_maps"].copy()
        successful = manifest["map_index"].notna()
        if int(successful.sum()) != len(maps):
            return None
        return manifest, maps
    except Exception:
        return None


all_batch_manifests = []
map_lookup = {}
n_batches = math.ceil(len(extreme_images) / pipeline_batch_size)
for batch_number, start in enumerate(
    range(0, len(extreme_images), pipeline_batch_size), start=1
):
    stop = min(start + pipeline_batch_size, len(extreme_images))
    batch = extreme_images.iloc[start:stop].copy()
    batch_directory = batch_root / f"batch_{start:06d}_{stop:06d}"
    manifest_path = batch_directory / "attribution_manifest.parquet"
    maps_path = batch_directory / "variable_maps.npz"
    expected_ids = batch["analysis_image_id"].astype(str).tolist()
    completed = (
        read_completed_batch(manifest_path, maps_path, expected_ids)
        if resume_batches
        else None
    )
    if completed is not None:
        manifest, maps = completed
        print(
            f"[extremes {batch_number}/{n_batches}] resumed "
            f"{len(manifest):,} participants",
            flush=True,
        )
    else:
        batch_directory.mkdir(parents=True, exist_ok=True)
        rows = []
        map_arrays = []
        for _, record in batch.iterrows():
            base = record.to_dict()
            output = {
                key: base.get(key)
                for key in (
                    "analysis_image_id",
                    "cohort",
                    "participant_id",
                    "image_path",
                    "eye",
                    "sex",
                    "age",
                    "age_gap",
                    "participant_age_gap",
                    "age_adjusted_gap",
                    "extreme_group",
                    "retina_fraction",
                    "brightness_mean",
                    "contrast_std",
                    "gradient_energy",
                )
            }
            output["analysis_signature"] = analysis_signature
            output.update(
                {
                    "map_index": None,
                    "embedding_cosine": np.nan,
                    "embedding_max_absolute_difference": np.nan,
                    "reconstruction_error": np.nan,
                    "valid_for_analysis": False,
                    "error": None,
                }
            )
            try:
                input_array, display_image, source_model = source_input_and_display(base)
                result = exact_patch_map_from_array(
                    model=source_model,
                    age_model=age_bundle,
                    input_array=input_array,
                    device=device,
                    stored_embedding=base["embedding"],
                )
                variable_grid = np.asarray(result["variable_grid"], dtype=np.float64)
                if align_left_eyes and is_left_eye(base):
                    variable_grid = np.fliplr(variable_grid)
                    display_image = display_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                proxy_maps = fundus_physiology_proxies(np.asarray(display_image))
                physiology = attribution_physiology_metrics(variable_grid, proxy_maps)
                valid = bool(
                    result["reconstruction_error"] <= 1e-4
                    and np.isfinite(result["embedding_cosine"])
                    and result["embedding_cosine"] >= embedding_cosine_threshold
                )
                output.update(physiology)
                output.update(
                    {
                        "map_index": len(map_arrays),
                        "embedding_cosine": result["embedding_cosine"],
                        "embedding_max_absolute_difference": result[
                            "embedding_max_absolute_difference"
                        ],
                        "reconstruction_error": result["reconstruction_error"],
                        "valid_for_analysis": valid,
                    }
                )
                map_arrays.append(variable_grid)
            except Exception as exc:
                output["error"] = f"{type(exc).__name__}: {str(exc)}"[:500]
            rows.append(output)
        manifest = pd.DataFrame(rows)
        maps = (
            np.stack(map_arrays)
            if map_arrays
            else np.empty((0, 14, 14), dtype=np.float64)
        )
        atomic_save_npz(maps_path, variable_maps=maps)
        write_frame(manifest, manifest_path)
        print(
            f"[extremes {batch_number}/{n_batches}] saved "
            f"{len(manifest):,} participants; valid="
            f"{int(manifest['valid_for_analysis'].sum()):,}",
            flush=True,
        )
    for _, record in manifest.loc[manifest["map_index"].notna()].iterrows():
        map_lookup[str(record["analysis_image_id"])] = maps[int(record["map_index"])]
    manifest["map_batch_path"] = str(maps_path)
    all_batch_manifests.append(manifest)

attribution_manifest = pd.concat(all_batch_manifests, ignore_index=True)
write_frame(
    attribution_manifest,
    attribution_root / "extreme_attribution_manifest.parquet",
)
validation_summary = (
    attribution_manifest.groupby(
        ["cohort", "extreme_group", "valid_for_analysis"], as_index=False
    )
    .agg(
        participants=("participant_id", "nunique"),
        processing_errors=("error", lambda value: int(value.notna().sum())),
        cosine_min=("embedding_cosine", "min"),
        cosine_median=("embedding_cosine", "median"),
        reconstruction_error_max=("reconstruction_error", "max"),
    )
)
write_frame(validation_summary, attribution_root / "validation_summary.csv")
display(validation_summary)

valid_manifest = attribution_manifest[
    attribution_manifest["valid_for_analysis"]
].copy()
valid_counts = valid_manifest.groupby(["cohort", "extreme_group"])[
    "participant_id"
].nunique()
for cohort in ("CLSA healthy", "Zeiss glaucoma"):
    for group_name in ("top_10_percent", "bottom_10_percent"):
        count = int(valid_counts.get((cohort, group_name), 0))
        if count < 5:
            raise RuntimeError(
                f"Only {count} valid {cohort} {group_name} participants; "
                "at least five are required. Inspect validation_summary.csv."
            )

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Participant-level patch inference and multiplicity control
# MAGIC
# MAGIC Primary patch inference uses label permutation and reports unadjusted,
# MAGIC Benjamini–Hochberg FDR, and strong max-|T| family-wise-error p-values over
# MAGIC all 196 patches. Analyses are conducted within source; cross-source spatial
# MAGIC concordance is reported separately.

# COMMAND ----------
def normalized_maps_for(cohort, group_name, participant_ids=None):
    selected = valid_manifest[
        (valid_manifest["cohort"] == cohort)
        & (valid_manifest["extreme_group"] == group_name)
    ].copy()
    if participant_ids is not None:
        selected = selected[selected["participant_id"].astype(str).isin(participant_ids)]
    maps = np.stack(
        [map_lookup[str(value)] for value in selected["analysis_image_id"]]
    )
    return attribution_group_statistics(maps)["normalized_maps"], selected


def normalized_maps_in_participant_order(cohort, group_name, participant_ids):
    selected = valid_manifest[
        (valid_manifest["cohort"] == cohort)
        & (valid_manifest["extreme_group"] == group_name)
    ].copy()
    selected["participant_id"] = selected["participant_id"].astype(str)
    selected = selected.set_index("participant_id", verify_integrity=True)
    missing = [value for value in participant_ids if value not in selected.index]
    if missing:
        raise RuntimeError(
            f"Matched sensitivity is missing {len(missing)} attribution maps"
        )
    maps = np.stack(
        [map_lookup[str(selected.loc[value, "analysis_image_id"])] for value in participant_ids]
    )
    return attribution_group_statistics(maps)["normalized_maps"]


patch_rows = []
effect_results = {}
for cohort in ("CLSA healthy", "Zeiss glaucoma"):
    top_maps, _ = normalized_maps_for(cohort, "top_10_percent")
    bottom_maps, _ = normalized_maps_for(cohort, "bottom_10_percent")
    result = permutation_patch_comparison(
        top_maps,
        bottom_maps,
        n_permutations=n_permutations,
        random_state=20260808,
    )
    result["top_mean"] = np.mean(top_maps, axis=0)
    result["bottom_mean"] = np.mean(bottom_maps, axis=0)
    effect_results[cohort] = result
    height, width = result["mean_difference"].shape
    for row in range(height):
        for column in range(width):
            patch_rows.append(
                {
                    "analysis": "primary_all_extremes",
                    "cohort": cohort,
                    "patch_row": row,
                    "patch_column": column,
                    "retinal_y_fraction": (row + 0.5) / height,
                    "retinal_x_fraction": (column + 0.5) / width,
                    "top_minus_bottom_mean_contribution": float(
                        result["mean_difference"][row, column]
                    ),
                    "welch_t": float(result["welch_t"][row, column]),
                    "p_unadjusted": float(result["p_unadjusted"][row, column]),
                    "p_fdr": float(result["p_fdr"][row, column]),
                    "p_fwer": float(result["p_fwer"][row, column]),
                    "n_top": result["n_top"],
                    "n_bottom": result["n_bottom"],
                    "n_permutations": result["n_permutations"],
                }
            )
patch_statistics = pd.DataFrame(patch_rows)
write_frame(patch_statistics, statistics_root / "patchwise_primary_statistics.parquet")
display(
    patch_statistics.sort_values(["p_fwer", "p_fdr", "p_unadjusted"]).head(20)
)

clsa_effect = effect_results["CLSA healthy"]["mean_difference"].reshape(-1)
zeiss_effect = effect_results["Zeiss glaucoma"]["mean_difference"].reshape(-1)
spatial_concordance = {
    "spearman_effect_map": float(stats.spearmanr(clsa_effect, zeiss_effect).statistic),
    "pearson_effect_map": float(np.corrcoef(clsa_effect, zeiss_effect)[0, 1]),
    "sign_concordance": float(np.mean(np.sign(clsa_effect) == np.sign(zeiss_effect))),
    "clsa_fwer_significant_patches": int(
        np.sum(effect_results["CLSA healthy"]["p_fwer"] < 0.05)
    ),
    "zeiss_fwer_significant_patches": int(
        np.sum(effect_results["Zeiss glaucoma"]["p_fwer"] < 0.05)
    ),
    "same_direction_fwer_significant_in_both": int(
        np.sum(
            (effect_results["CLSA healthy"]["p_fwer"] < 0.05)
            & (effect_results["Zeiss glaucoma"]["p_fwer"] < 0.05)
            & (
                np.sign(effect_results["CLSA healthy"]["mean_difference"])
                == np.sign(effect_results["Zeiss glaucoma"]["mean_difference"])
            )
        )
    ),
    "same_direction_fdr_significant_in_both": int(
        np.sum(
            (effect_results["CLSA healthy"]["p_fdr"] < 0.05)
            & (effect_results["Zeiss glaucoma"]["p_fdr"] < 0.05)
            & (
                np.sign(effect_results["CLSA healthy"]["mean_difference"])
                == np.sign(effect_results["Zeiss glaucoma"]["mean_difference"])
            )
        )
    ),
}
write_json(spatial_concordance, statistics_root / "cross_cohort_spatial_concordance.json")
print(json.dumps(spatial_concordance, indent=2))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Age/sex-matched sensitivity analysis
# MAGIC
# MAGIC Top and bottom participants are greedily matched without replacement on
# MAGIC sex when available and chronological age within the prespecified caliper.
# MAGIC This does not replace the primary extreme-decile analysis; it tests whether
# MAGIC the spatial result persists after reducing age imbalance.

# COMMAND ----------
def match_extremes_on_age_and_sex(frame, caliper):
    pair_columns = [
        "cohort",
        "pair_id",
        "top_participant_id",
        "bottom_participant_id",
        "top_age",
        "bottom_age",
        "absolute_age_difference",
    ]
    pairs = []
    for cohort, cohort_frame in frame.groupby("cohort", sort=True):
        top = cohort_frame[cohort_frame["extreme_group"] == "top_10_percent"].copy()
        bottom = cohort_frame[
            cohort_frame["extreme_group"] == "bottom_10_percent"
        ].copy()
        available = set(bottom.index)
        top = top.sort_values(["age", "participant_id"], kind="stable")
        for _, top_row in top.iterrows():
            candidates = bottom.loc[sorted(available)].copy()
            if candidates.empty:
                break
            if pd.notna(top_row.get("sex")) and candidates["sex"].notna().any():
                same_sex = candidates[
                    candidates["sex"].astype(str).str.upper().str.strip()
                    == str(top_row["sex"]).upper().strip()
                ]
                candidates = same_sex
                if candidates.empty:
                    continue
            candidates["age_distance"] = (candidates["age"] - top_row["age"]).abs()
            candidates = candidates.sort_values(
                ["age_distance", "participant_id"], kind="stable"
            )
            bottom_row = candidates.iloc[0]
            if float(bottom_row["age_distance"]) > caliper:
                continue
            available.remove(bottom_row.name)
            pairs.append(
                {
                    "cohort": cohort,
                    "pair_id": f"{cohort[:1]}_{len(pairs):06d}",
                    "top_participant_id": str(top_row["participant_id"]),
                    "bottom_participant_id": str(bottom_row["participant_id"]),
                    "top_age": float(top_row["age"]),
                    "bottom_age": float(bottom_row["age"]),
                    "absolute_age_difference": float(bottom_row["age_distance"]),
                }
            )
    return pd.DataFrame(pairs, columns=pair_columns)


age_match_pairs = match_extremes_on_age_and_sex(
    valid_manifest, age_match_caliper_years
)
write_frame(age_match_pairs, statistics_root / "age_sex_matched_extreme_pairs.parquet")
matched_patch_rows = []
matched_effect_results = {}
for cohort in ("CLSA healthy", "Zeiss glaucoma"):
    cohort_pairs = age_match_pairs[age_match_pairs["cohort"] == cohort]
    if len(cohort_pairs) < 5:
        continue
    top_ids = cohort_pairs["top_participant_id"].astype(str).tolist()
    bottom_ids = cohort_pairs["bottom_participant_id"].astype(str).tolist()
    top_maps = normalized_maps_in_participant_order(
        cohort, "top_10_percent", top_ids
    )
    bottom_maps = normalized_maps_in_participant_order(
        cohort, "bottom_10_percent", bottom_ids
    )
    result = paired_permutation_patch_comparison(
        top_maps,
        bottom_maps,
        n_permutations=n_permutations,
        random_state=20260809,
    )
    matched_effect_results[cohort] = result
    for row in range(result["mean_difference"].shape[0]):
        for column in range(result["mean_difference"].shape[1]):
            matched_patch_rows.append(
                {
                    "analysis": "age_sex_matched_sensitivity",
                    "cohort": cohort,
                    "patch_row": row,
                    "patch_column": column,
                    "top_minus_bottom_mean_contribution": float(
                        result["mean_difference"][row, column]
                    ),
                    "paired_t": float(result["paired_t"][row, column]),
                    "p_unadjusted": float(result["p_unadjusted"][row, column]),
                    "p_fdr": float(result["p_fdr"][row, column]),
                    "p_fwer": float(result["p_fwer"][row, column]),
                    "n_pairs": int(len(cohort_pairs)),
                }
            )
matched_patch_statistics = pd.DataFrame(
    matched_patch_rows,
    columns=[
        "analysis",
        "cohort",
        "patch_row",
        "patch_column",
        "top_minus_bottom_mean_contribution",
        "paired_t",
        "p_unadjusted",
        "p_fdr",
        "p_fwer",
        "n_pairs",
    ],
)
write_frame(
    matched_patch_statistics,
    statistics_root / "patchwise_age_matched_sensitivity.parquet",
)
age_match_summary = (
    age_match_pairs.groupby("cohort", as_index=False)
    .agg(
        matched_pairs=("pair_id", "size"),
        median_age_difference=("absolute_age_difference", "median"),
        maximum_age_difference=("absolute_age_difference", "max"),
    )
    if not age_match_pairs.empty
    else pd.DataFrame(
        columns=[
            "cohort",
            "matched_pairs",
            "median_age_difference",
            "maximum_age_difference",
        ]
    )
)
write_frame(age_match_summary, statistics_root / "age_match_summary.csv")
display(age_match_summary)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Does attribution overlap plausible physiology or image artifacts?
# MAGIC
# MAGIC Attribution magnitude is compared with image-derived proxy masks. Primary
# MAGIC reporting includes anatomy-oriented proxies and negative-control artifact
# MAGIC proxies (retinal border, high gradient, luminance, and background).

# COMMAND ----------
physiology_columns = [
    column
    for column in valid_manifest.columns
    if column.endswith("_enrichment")
    or column.endswith("_spearman")
    or column == "background_attribution_fraction"
]


def bootstrap_difference(top, bottom, repetitions=2000, seed=20260810):
    top = np.asarray(top, dtype=float)
    bottom = np.asarray(bottom, dtype=float)
    top = top[np.isfinite(top)]
    bottom = bottom[np.isfinite(bottom)]
    if len(top) < 2 or len(bottom) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    differences = np.empty(repetitions)
    for index in range(repetitions):
        differences[index] = (
            rng.choice(top, len(top), replace=True).mean()
            - rng.choice(bottom, len(bottom), replace=True).mean()
        )
    return (
        float(top.mean() - bottom.mean()),
        float(np.quantile(differences, 0.025)),
        float(np.quantile(differences, 0.975)),
    )


physiology_rows = []
for cohort in ("CLSA healthy", "Zeiss glaucoma"):
    cohort_frame = valid_manifest[valid_manifest["cohort"] == cohort]
    top_frame = cohort_frame[cohort_frame["extreme_group"] == "top_10_percent"]
    bottom_frame = cohort_frame[
        cohort_frame["extreme_group"] == "bottom_10_percent"
    ]
    for metric in physiology_columns:
        top = pd.to_numeric(top_frame[metric], errors="coerce").dropna().to_numpy()
        bottom = pd.to_numeric(bottom_frame[metric], errors="coerce").dropna().to_numpy()
        if len(top) < 2 or len(bottom) < 2:
            continue
        test = stats.mannwhitneyu(top, bottom, alternative="two-sided")
        difference, ci_low, ci_high = bootstrap_difference(top, bottom)
        rank_biserial = 2 * float(test.statistic) / (len(top) * len(bottom)) - 1
        physiology_rows.append(
            {
                "cohort": cohort,
                "metric": metric,
                "n_top": len(top),
                "n_bottom": len(bottom),
                "top_mean": float(np.mean(top)),
                "bottom_mean": float(np.mean(bottom)),
                "top_minus_bottom_mean": difference,
                "bootstrap_95_ci_low": ci_low,
                "bootstrap_95_ci_high": ci_high,
                "rank_biserial_effect": rank_biserial,
                "p_unadjusted": float(test.pvalue),
            }
        )
physiology_statistics = pd.DataFrame(physiology_rows)
if not physiology_statistics.empty:
    physiology_statistics["p_fdr"] = np.nan
    for _, indices in physiology_statistics.groupby("cohort").groups.items():
        physiology_statistics.loc[indices, "p_fdr"] = benjamini_hochberg(
            physiology_statistics.loc[indices, "p_unadjusted"]
        )
write_frame(
    physiology_statistics,
    statistics_root / "physiology_proxy_group_comparison.csv",
)
display(physiology_statistics.sort_values(["p_fdr", "p_unadjusted"]).head(30))


def hc3_extreme_effect(frame, outcome):
    work = frame[["extreme_group", "age", "sex", outcome]].copy()
    work[outcome] = pd.to_numeric(work[outcome], errors="coerce")
    work["age"] = pd.to_numeric(work["age"], errors="coerce")
    work = work.dropna(subset=[outcome, "age", "extreme_group"])
    if len(work) < 20:
        return None
    top = (work["extreme_group"] == "top_10_percent").to_numpy(float)
    centered_age = work["age"].to_numpy(float) - float(work["age"].mean())
    design_columns = [
        np.ones(len(work)),
        top,
        centered_age,
        centered_age**2,
    ]
    if work["sex"].notna().any():
        sex = work["sex"].fillna("missing").astype(str).str.upper().str.strip()
        for category in sorted(sex.unique())[1:]:
            design_columns.append((sex == category).to_numpy(float))
    design = np.column_stack(design_columns)
    outcome_values = work[outcome].to_numpy(float)
    inverse = np.linalg.pinv(design.T @ design)
    coefficients = inverse @ design.T @ outcome_values
    residuals = outcome_values - design @ coefficients
    leverage = np.sum((design @ inverse) * design, axis=1)
    scaled_residuals = residuals / np.maximum(1 - leverage, 1e-6)
    meat = design.T @ ((scaled_residuals**2)[:, None] * design)
    covariance = inverse @ meat @ inverse
    standard_error = float(np.sqrt(max(covariance[1, 1], 0)))
    estimate = float(coefficients[1])
    z_score = estimate / standard_error if standard_error > 0 else np.nan
    p_value = float(2 * stats.norm.sf(abs(z_score))) if np.isfinite(z_score) else np.nan
    return {
        "n_participants": int(len(work)),
        "adjusted_top_minus_bottom": estimate,
        "hc3_standard_error": standard_error,
        "ci_95_low": estimate - 1.96 * standard_error,
        "ci_95_high": estimate + 1.96 * standard_error,
        "p_unadjusted": p_value,
    }


adjusted_rows = []
for cohort in ("CLSA healthy", "Zeiss glaucoma"):
    cohort_frame = valid_manifest[valid_manifest["cohort"] == cohort]
    for metric in physiology_columns:
        result = hc3_extreme_effect(cohort_frame, metric)
        if result is not None:
            adjusted_rows.append({"cohort": cohort, "metric": metric, **result})
physiology_adjusted = pd.DataFrame(adjusted_rows)
if not physiology_adjusted.empty:
    physiology_adjusted["p_fdr"] = np.nan
    for _, indices in physiology_adjusted.groupby("cohort").groups.items():
        physiology_adjusted.loc[indices, "p_fdr"] = benjamini_hochberg(
            physiology_adjusted.loc[indices, "p_unadjusted"]
        )
write_frame(
    physiology_adjusted,
    statistics_root / "physiology_proxy_age_sex_adjusted_hc3.csv",
)
display(physiology_adjusted.sort_values(["p_fdr", "p_unadjusted"]).head(30))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Publication figures

# COMMAND ----------
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Figure 1: distributions and cohort-specific thresholds.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
for axis, cohort in zip(axes, ("CLSA healthy", "Zeiss glaucoma")):
    values = participant_cohort.loc[
        participant_cohort["cohort"] == cohort, "age_gap"
    ]
    threshold = extreme_thresholds[extreme_thresholds["cohort"] == cohort].iloc[0]
    axis.hist(values, bins=40, color="#4c78a8", alpha=0.80)
    axis.axvline(threshold["bottom_threshold"], color="#2a9d8f", linestyle="--")
    axis.axvline(threshold["top_threshold"], color="#d1495b", linestyle="--")
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_title(cohort)
    axis.set_xlabel("Participant retinal-age gap (years)")
    axis.set_ylabel("Participants")
for suffix in ("png", "pdf"):
    fig.savefig(figure_root / f"figure_05_01_age_gap_distributions.{suffix}", bbox_inches="tight")
plt.show()
plt.close(fig)

# Figure 2: mean maps, difference, and max-T significance.
fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), constrained_layout=True)
for row_index, cohort in enumerate(("CLSA healthy", "Zeiss glaucoma")):
    result = effect_results[cohort]
    grids = [
        (result["bottom_mean"], "Bottom decile mean"),
        (result["top_mean"], "Top decile mean"),
        (result["mean_difference"], "Top − bottom"),
        (-np.log10(np.maximum(result["p_fwer"], 1e-6)), "−log10 max-T FWER P"),
    ]
    for column_index, (grid, title) in enumerate(grids):
        axis = axes[row_index, column_index]
        if column_index < 3:
            limit = float(np.max(np.abs(grid))) or 1
            image = axis.imshow(grid, cmap="coolwarm", vmin=-limit, vmax=limit)
        else:
            image = axis.imshow(grid, cmap="viridis", vmin=0)
            significant = result["p_fwer"] < 0.05
            if significant.any():
                axis.contour(significant.astype(float), levels=[0.5], colors="white")
        axis.set_title(title)
        if column_index == 0:
            axis.set_ylabel(cohort)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
for suffix in ("png", "pdf"):
    fig.savefig(figure_root / f"figure_05_02_patchwise_extremes.{suffix}", bbox_inches="tight")
plt.show()
plt.close(fig)

# Figure 3: physiology-proxy enrichments.
plot_metrics = [
    "vessel_proxy_enrichment",
    "optic_disc_proxy_enrichment",
    "central_proxy_enrichment",
    "peripheral_proxy_enrichment",
    "border_proxy_enrichment",
    "high_gradient_proxy_enrichment",
]
fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
colors = {"bottom_10_percent": "#2a9d8f", "top_10_percent": "#d1495b"}
for axis, metric in zip(axes.ravel(), plot_metrics):
    positions = []
    data = []
    labels = []
    position = 1
    for cohort in ("CLSA healthy", "Zeiss glaucoma"):
        for group_name in ("bottom_10_percent", "top_10_percent"):
            values = pd.to_numeric(
                valid_manifest.loc[
                    (valid_manifest["cohort"] == cohort)
                    & (valid_manifest["extreme_group"] == group_name),
                    metric,
                ],
                errors="coerce",
            ).dropna()
            data.append(values)
            positions.append(position)
            labels.append(f"{cohort.split()[0]}\n{group_name.split('_')[0]}")
            position += 1
        position += 0.5
    box = axis.boxplot(data, positions=positions, widths=0.65, patch_artist=True, showfliers=False)
    for patch, group_name in zip(box["boxes"], ["bottom_10_percent", "top_10_percent"] * 2):
        patch.set_facecolor(colors[group_name])
        patch.set_alpha(0.7)
    axis.axhline(1, color="gray", linestyle="--", linewidth=0.8)
    axis.set_xticks(positions, labels, fontsize=8)
    axis.set_title(metric.replace("_proxy_enrichment", "").replace("_", " ").title())
    axis.set_ylabel("Attribution enrichment")
for suffix in ("png", "pdf"):
    fig.savefig(figure_root / f"figure_05_03_physiology_overlap.{suffix}", bbox_inches="tight")
plt.show()
plt.close(fig)


def resized_signed_map(grid, size):
    resized = ndimage.zoom(
        grid,
        (size[1] / grid.shape[0], size[0] / grid.shape[1]),
        order=1,
    )
    return resized[: size[1], : size[0]]


def anatomy_overlay(image, proxies):
    rgb = np.asarray(image, dtype=float) / 255.0
    overlay = rgb.copy()
    masks = [
        (proxies["vessel_proxy"], np.array([1.0, 0.1, 0.1])),
        (proxies["optic_disc_proxy"], np.array([0.0, 0.9, 1.0])),
        (proxies["central_proxy"], np.array([1.0, 0.9, 0.0])),
    ]
    for mask, color in masks:
        edge = mask ^ ndimage.binary_erosion(mask)
        overlay[edge] = color
    return np.clip(overlay, 0, 1)


# Figure 4: deterministic, evenly spaced examples; no participant identifiers.
example_rows = []
for _, group in valid_manifest.groupby(["cohort", "extreme_group"], sort=True):
    group = group.sort_values("participant_age_gap", kind="stable")
    positions = np.linspace(0, len(group) - 1, min(n_examples_per_group, len(group))).round().astype(int)
    example_rows.extend(group.iloc[positions].to_dict(orient="records"))
n_rows = len(example_rows)
fig, axes = plt.subplots(n_rows, 4, figsize=(13, max(3, 2.8 * n_rows)), constrained_layout=True)
axes = np.atleast_2d(axes)
for row_index, record in enumerate(example_rows):
    _, display_image, _ = source_input_and_display(record)
    if align_left_eyes and is_left_eye(record):
        display_image = display_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    grid = map_lookup[str(record["analysis_image_id"])]
    heat = resized_signed_map(grid, display_image.size)
    limit = float(np.percentile(np.abs(heat), 98)) or 1
    proxies = fundus_physiology_proxies(np.asarray(display_image))
    axes[row_index, 0].imshow(display_image)
    axes[row_index, 1].imshow(heat, cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[row_index, 2].imshow(display_image)
    axes[row_index, 2].imshow(heat, cmap="coolwarm", vmin=-limit, vmax=limit, alpha=0.45)
    axes[row_index, 3].imshow(anatomy_overlay(display_image, proxies))
    axes[row_index, 0].set_ylabel(
        f"{record['cohort']}\n{record['extreme_group'].split('_')[0]}\n"
        f"age {float(record['age']):.1f}, gap {float(record['participant_age_gap']):+.1f}",
        fontsize=8,
    )
    for column_index, title in enumerate(("Fundus", "Signed attribution", "Overlay", "Proxy contours")):
        if row_index == 0:
            axes[row_index, column_index].set_title(title)
        axes[row_index, column_index].set_xticks([])
        axes[row_index, column_index].set_yticks([])
for suffix in ("png", "pdf"):
    fig.savefig(figure_root / f"figure_05_04_image_heatmap_physiology.{suffix}", bbox_inches="tight")
plt.show()
plt.close(fig)

# Figure 5: cross-source patch-effect concordance.
fig, axis = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
axis.scatter(clsa_effect, zeiss_effect, alpha=0.65, s=20)
axis.axhline(0, color="gray", linewidth=0.8)
axis.axvline(0, color="gray", linewidth=0.8)
axis.set_xlabel("CLSA top − bottom patch contribution")
axis.set_ylabel("Zeiss top − bottom patch contribution")
axis.set_title(f"Cross-source spatial concordance\nSpearman r={spatial_concordance['spearman_effect_map']:.2f}")
for suffix in ("png", "pdf"):
    fig.savefig(figure_root / f"figure_05_05_cross_source_concordance.{suffix}", bbox_inches="tight")
plt.show()
plt.close(fig)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Reproducible publication summary and interpretation guardrails

# COMMAND ----------
run_summary = {
    "analysis": "participant_level_extreme_retinal_age_gap_explainability",
    "selection_metric": selection_metric,
    "extreme_quantile": extreme_quantile,
    "one_image_per_participant": True,
    "within_source_thresholds": True,
    "n_selected_images": int(len(extreme_images)),
    "n_valid_attributions": int(len(valid_manifest)),
    "n_permutations": n_permutations,
    "multiple_testing": "patchwise BH-FDR and max-|T| FWER",
    "age_sex_matching_caliper_years": age_match_caliper_years,
    "checkpoint_sha256": checkpoint_hash,
    "retfound_repo": str(resolved_repo),
    "retfound_repo_commit": retfound_repo_commit,
    "embedding_cosine_threshold": embedding_cosine_threshold,
    "left_eyes_horizontally_aligned": align_left_eyes,
    "cross_cohort_spatial_concordance": spatial_concordance,
    "physiology_proxy_warning": (
        "Vessel, optic-disc, central, peripheral, border, and gradient maps are "
        "image-derived exploratory proxies, not validated clinical segmentations."
    ),
    "limitations": [
        "Extreme-group analyses are descriptive and do not establish causality.",
        "The CLSA age head was trained in screen-negative CLSA and transported to Zeiss.",
        "Disease status and acquisition source remain confounded across cohorts.",
        "Age-gap extremes can reflect regression to the mean; raw and age-adjusted definitions and age/sex-matched sensitivity outputs are reported.",
        "Attribution localization describes the fitted model, not a causal retinal lesion.",
        "Image-derived physiology proxies require validation against expert or automated segmentations before anatomic claims.",
    ],
}
write_json(run_summary, analysis_root / "age_gap_extremes_summary.json")
print(json.dumps(run_summary, indent=2))

os.environ.pop("HF_TOKEN", None)
try:
    dbutils.widgets.remove("hf_token")
except Exception:
    pass
print("Notebook 05 complete; temporary Hugging Face token removed")
