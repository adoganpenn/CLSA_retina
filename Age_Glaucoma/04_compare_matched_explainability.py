# Databricks notebook source
# MAGIC %md
# MAGIC # Source-aware explainability for matched CLSA and Zeiss subgroups
# MAGIC
# MAGIC This GPU notebook loads the frozen `CLSA_healthy` linear age head and the
# MAGIC matched image/vector outputs from notebook 03. It reproduces each source's
# MAGIC original RETFound input path:
# MAGIC
# MAGIC - CLSA: the shared CLSA fundus crop and per-image channel normalization.
# MAGIC - Zeiss: the attached DICOM/AutoMorph crop and ImageNet normalization.
# MAGIC
# MAGIC Before accepting an attribution, it compares the reproduced feature with
# MAGIC the stored 1,024-element embedding. Group maps use one participant-level
# MAGIC map per match set, horizontally align left eyes, and report robust spatial
# MAGIC outlier frequencies. The age head is never retrained here.

# COMMAND ----------
# MAGIC %md
# MAGIC On a fresh GPU cluster, run once and restart Python:
# MAGIC
# MAGIC ```python
# MAGIC %pip install -r /Workspace/Users/ad0038@pennmedicine.upenn.edu/CLSA/CLSA_retina/requirements-retfound.txt
# MAGIC %pip install pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg
# MAGIC dbutils.library.restartPython()
# MAGIC ```

# COMMAND ----------
from pathlib import Path
import importlib
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

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
dbutils.widgets.text("retfound_repo", "")
dbutils.widgets.text("checkpoint_path", "")
dbutils.widgets.dropdown("allow_downloads", "true", ["true", "false"])
dbutils.widgets.text("hf_token", "", "Hugging Face token (temporary)")
dbutils.widgets.dropdown("device", "cuda", ["cuda", "auto", "cpu"])
dbutils.widgets.text("n_match_sets", "40")
dbutils.widgets.text("n_outlier_match_sets", "20")
dbutils.widgets.text("embedding_cosine_threshold", "0.99")
dbutils.widgets.text("spatial_outlier_z", "3.5")
dbutils.widgets.dropdown("align_left_eyes", "true", ["true", "false"])

# COMMAND ----------
repo_root = Path(dbutils.widgets.get("repo_root").strip())
output_root = Path(dbutils.widgets.get("age_glaucoma_output_root").strip())
retfound_repo = dbutils.widgets.get("retfound_repo").strip() or None
checkpoint_path = dbutils.widgets.get("checkpoint_path").strip() or None
allow_downloads = dbutils.widgets.get("allow_downloads") == "true"
hf_token = dbutils.widgets.get("hf_token").strip()
device_requested = dbutils.widgets.get("device")
n_match_sets = int(dbutils.widgets.get("n_match_sets"))
n_outlier_match_sets = int(dbutils.widgets.get("n_outlier_match_sets"))
embedding_cosine_threshold = float(
    dbutils.widgets.get("embedding_cosine_threshold")
)
spatial_outlier_z = float(dbutils.widgets.get("spatial_outlier_z"))
align_left_eyes = dbutils.widgets.get("align_left_eyes") == "true"

if n_match_sets < 5:
    raise ValueError("n_match_sets must be at least 5")
if n_outlier_match_sets < 0 or n_outlier_match_sets > n_match_sets:
    raise ValueError("n_outlier_match_sets must be between 0 and n_match_sets")
if not 0 <= embedding_cosine_threshold <= 1:
    raise ValueError("embedding_cosine_threshold must be between 0 and 1")
if spatial_outlier_z <= 0:
    raise ValueError("spatial_outlier_z must be positive")
if device_requested == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested, but this compute has no CUDA GPU")

if hf_token:
    import os

    os.environ["HF_TOKEN"] = hf_token

module_root = repo_root / "src"
if str(module_root) not in sys.path:
    sys.path.insert(0, str(module_root))

import fundus_retfound_pipeline as _fundus_pipeline  # noqa: E402

# Databricks persists imported modules across notebook reruns. Reload the Git
# version and refuse to continue if it predates the PlanMetrics-safe writer.
_fundus_pipeline = importlib.reload(_fundus_pipeline)
if not getattr(_fundus_pipeline, "PARQUET_RUNTIME_ATTRS_SAFE", False):
    raise RuntimeError(
        "The loaded fundus_retfound_pipeline has the old Parquet writer. "
        "Pull the latest Git revision and restart Python. Loaded module: "
        f"{_fundus_pipeline.__file__}"
    )
print("Reloaded PlanMetrics-safe fundus pipeline:", _fundus_pipeline.__file__)

from age_glaucoma_model import (  # noqa: E402
    attribution_group_statistics,
    exact_patch_map_from_array,
    prepare_zeiss_dicom_input,
)
from fundus_retfound_pipeline import (  # noqa: E402
    QualityConfig,
    RETFoundConfig,
    load_age_head,
    load_retfound_model,
    prepare_model_input,
    write_frame,
)

# COMMAND ----------
model_path = output_root / "06_CLSA_healthy_model" / "CLSA_healthy.joblib"
comparison_root = output_root / "08_matched_comparison"
pair_level_path = comparison_root / "matched_pair_level_analysis.parquet"
zeiss_images_path = (
    comparison_root / "zeiss_matched_images_for_explainability.parquet"
)
clsa_images_path = (
    comparison_root / "clsa_matched_images_for_explainability.parquet"
)
explain_root = output_root / "09_matched_explainability"
map_root = explain_root / "individual_maps"
figure_root = explain_root / "figures"
for path in (explain_root, map_root, figure_root):
    path.mkdir(parents=True, exist_ok=True)
for path in (model_path, pair_level_path, zeiss_images_path, clsa_images_path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required notebook-03 output is missing: {path}"
        )

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Select matched sets across typical and outlying age-gap differences

# COMMAND ----------
pair_level = pd.read_parquet(pair_level_path)
pair_level.attrs = {}
pair_level["absolute_paired_gap_difference"] = pair_level[
    "paired_age_gap_difference"
].abs()
n_available = int(pair_level["match_set_id"].nunique())
n_select = min(n_match_sets, n_available)
n_outlier = min(n_outlier_match_sets, n_select)

outlier_sets = (
    pair_level.sort_values("absolute_paired_gap_difference", ascending=False)
    .head(n_outlier)["match_set_id"]
    .tolist()
)
remaining = pair_level[~pair_level["match_set_id"].isin(outlier_sets)].copy()
n_representative = n_select - len(outlier_sets)
if n_representative and len(remaining):
    ordered = remaining.sort_values("paired_age_gap_difference")
    positions = np.linspace(0, len(ordered) - 1, n_representative).round().astype(int)
    representative_sets = ordered.iloc[positions]["match_set_id"].tolist()
else:
    representative_sets = []
selected_match_sets = list(dict.fromkeys(outlier_sets + representative_sets))
selection_summary = pd.DataFrame(
    [
        {
            "available_match_sets": n_available,
            "selected_match_sets": len(selected_match_sets),
            "selected_outlier_sets": len(outlier_sets),
            "selected_representative_sets": len(representative_sets),
        }
    ]
)
display(selection_summary)

# COMMAND ----------
zeiss_images = pd.read_parquet(zeiss_images_path)
clsa_images = pd.read_parquet(clsa_images_path)
zeiss_images.attrs = {}
clsa_images.attrs = {}
zeiss_images["cohort"] = "Zeiss glaucoma"
clsa_images["cohort"] = "CLSA healthy"


def representative_images(frame, participant_column):
    work = frame[frame["match_set_id"].isin(selected_match_sets)].copy()
    eye_column = "eye" if "eye" in work.columns else (
        "laterality" if "laterality" in work.columns else None
    )
    if eye_column:
        eye = work[eye_column].astype(str).str.upper().str.strip()
        work["_eye_priority"] = np.where(
            eye.isin(["R", "RIGHT", "OD"]), 0, 1
        )
    else:
        work["_eye_priority"] = 1
    work = work.sort_values(
        ["match_set_id", participant_column, "_eye_priority", "image_path"],
        kind="stable",
    )
    return work.drop_duplicates(
        ["match_set_id", participant_column], keep="first"
    ).drop(columns=["_eye_priority"])


zeiss_selected = representative_images(zeiss_images, "participant_id")
clsa_selected = representative_images(clsa_images, "participant_id")
print(
    f"Selected images: Zeiss={len(zeiss_selected):,}; "
    f"CLSA={len(clsa_selected):,}"
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Load the frozen head and the identical RETFound encoder checkpoint

# COMMAND ----------
age_bundle = load_age_head(model_path)
retfound_config = RETFoundConfig(
    repo_path=retfound_repo,
    checkpoint_path=checkpoint_path,
    allow_downloads=allow_downloads,
    device=device_requested,
    batch_size=1,
)
model, device, resolved_repo, resolved_checkpoint = load_retfound_model(
    retfound_config
)
quality_config = QualityConfig(
    output_size=256,
    model_input_size=224,
    save_preprocessed=False,
)
print("Device:", device)
print("RETFound repository:", resolved_repo)
print("Checkpoint:", resolved_checkpoint)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Compute exact source-specific patch contributions

# COMMAND ----------
def is_left_eye(record):
    for column in ("eye", "laterality"):
        if column in record and pd.notna(record[column]):
            return str(record[column]).strip().upper() in {"L", "LEFT", "OS"}
    return False


records = []
map_arrays = []
combined = pd.concat([zeiss_selected, clsa_selected], ignore_index=True, sort=False)
for index, (_, record) in enumerate(combined.iterrows(), start=1):
    source = str(record["cohort"])
    image_path = str(record["image_path"])
    try:
        if source == "Zeiss glaucoma":
            input_array, _ = prepare_zeiss_dicom_input(image_path)
        else:
            input_array = prepare_model_input(image_path, quality_config)
        result = exact_patch_map_from_array(
            model=model,
            age_model=age_bundle,
            input_array=input_array,
            device=device,
            stored_embedding=record["embedding"],
        )
        variable_grid = np.asarray(result["variable_grid"], dtype=np.float64)
        if align_left_eyes and is_left_eye(record):
            variable_grid = np.fliplr(variable_grid)
        valid = bool(
            result["reconstruction_error"] <= 1e-4
            and np.isfinite(result["embedding_cosine"])
            and result["embedding_cosine"] >= embedding_cosine_threshold
        )
        map_path = map_root / f"map_{index:04d}.npz"
        np.savez_compressed(
            map_path,
            variable_grid=variable_grid,
            additive_grid=np.asarray(result["grid"], dtype=np.float64),
        )
        map_arrays.append(variable_grid)
        records.append(
            {
                "map_index": index - 1,
                "cohort": source,
                "match_set_id": record["match_set_id"],
                "participant_id": record["participant_id"],
                "image_path": image_path,
                "eye": record.get("eye", record.get("laterality")),
                "prediction_from_reproduced_feature": result["prediction"],
                "embedding_cosine": result["embedding_cosine"],
                "embedding_max_absolute_difference": result[
                    "embedding_max_absolute_difference"
                ],
                "reconstruction_error": result["reconstruction_error"],
                "valid_for_group_comparison": valid,
                "map_path": str(map_path),
                "error": None,
            }
        )
    except Exception as exc:
        records.append(
            {
                "map_index": None,
                "cohort": source,
                "match_set_id": record["match_set_id"],
                "participant_id": record["participant_id"],
                "image_path": image_path,
                "eye": record.get("eye", record.get("laterality")),
                "prediction_from_reproduced_feature": np.nan,
                "embedding_cosine": np.nan,
                "embedding_max_absolute_difference": np.nan,
                "reconstruction_error": np.nan,
                "valid_for_group_comparison": False,
                "map_path": None,
                "error": f"{type(exc).__name__}: {str(exc)}"[:500],
            }
        )
    if index % 10 == 0 or index == len(combined):
        print(f"Explained {index:,}/{len(combined):,} images")

explanation_manifest = pd.DataFrame(records)
write_frame(
    explanation_manifest,
    explain_root / "source_specific_explainability_manifest.parquet",
)
quality_summary = (
    explanation_manifest.groupby(["cohort", "valid_for_group_comparison"])
    .size()
    .rename("images")
    .reset_index()
)
display(quality_summary)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Aggregate one attribution map per match set and cohort

# COMMAND ----------
valid_manifest = explanation_manifest[
    explanation_manifest["valid_for_group_comparison"]
].copy()
if valid_manifest.empty:
    raise RuntimeError(
        "No source-specific attributions reproduced their stored embeddings. "
        "Inspect the manifest before interpreting any spatial comparison."
    )

valid_map_lookup = {
    int(row["map_index"]): np.load(row["map_path"])["variable_grid"]
    for _, row in valid_manifest.iterrows()
}
set_maps = []
set_records = []
for (cohort, match_set_id), group in valid_manifest.groupby(
    ["cohort", "match_set_id"], sort=True
):
    maps = np.stack(
        [valid_map_lookup[int(value)] for value in group["map_index"]]
    )
    set_maps.append(np.mean(maps, axis=0))
    set_records.append(
        {
            "cohort": cohort,
            "match_set_id": match_set_id,
            "set_map_index": len(set_maps) - 1,
            "n_participant_images": int(len(maps)),
        }
    )
set_manifest = pd.DataFrame(set_records)
complete_sets = set(
    set_manifest.groupby("match_set_id")["cohort"].nunique().loc[lambda value: value == 2].index
)
set_manifest = set_manifest[set_manifest["match_set_id"].isin(complete_sets)].copy()
if len(complete_sets) < 5:
    raise RuntimeError(
        f"Only {len(complete_sets)} match sets have valid maps for both cohorts; "
        "at least 5 are required. Inspect embedding reproduction failures."
    )

cohort_maps = {}
cohort_stats = {}
for cohort in ("CLSA healthy", "Zeiss glaucoma"):
    indices = set_manifest.loc[
        set_manifest["cohort"] == cohort, "set_map_index"
    ].astype(int)
    maps = np.stack([set_maps[index] for index in indices])
    cohort_maps[cohort] = maps
    cohort_stats[cohort] = attribution_group_statistics(
        maps, outlier_z=spatial_outlier_z
    )

clsa_mean = cohort_stats["CLSA healthy"]["mean_map"]
zeiss_mean = cohort_stats["Zeiss glaucoma"]["mean_map"]
mean_difference = zeiss_mean - clsa_mean
clsa_sd = cohort_stats["CLSA healthy"]["sd_map"]
zeiss_sd = cohort_stats["Zeiss glaucoma"]["sd_map"]
pooled_sd = np.sqrt((clsa_sd**2 + zeiss_sd**2) / 2)
effect_map = np.divide(
    mean_difference,
    pooled_sd,
    out=np.zeros_like(mean_difference),
    where=pooled_sd > 1e-8,
)
outlier_frequency_difference = (
    cohort_stats["Zeiss glaucoma"]["outlier_frequency"]
    - cohort_stats["CLSA healthy"]["outlier_frequency"]
)

# COMMAND ----------
patch_rows = []
height, width = mean_difference.shape
for row in range(height):
    for column in range(width):
        patch_rows.append(
            {
                "patch_row": row,
                "patch_column": column,
                "retinal_y_fraction": (row + 0.5) / height,
                "retinal_x_fraction": (column + 0.5) / width,
                "clsa_mean_normalized_contribution": float(clsa_mean[row, column]),
                "zeiss_mean_normalized_contribution": float(zeiss_mean[row, column]),
                "zeiss_minus_clsa_contribution": float(mean_difference[row, column]),
                "standardized_effect": float(effect_map[row, column]),
                "clsa_outlier_frequency": float(
                    cohort_stats["CLSA healthy"]["outlier_frequency"][row, column]
                ),
                "zeiss_outlier_frequency": float(
                    cohort_stats["Zeiss glaucoma"]["outlier_frequency"][row, column]
                ),
                "zeiss_minus_clsa_outlier_frequency": float(
                    outlier_frequency_difference[row, column]
                ),
            }
        )
patch_comparison = pd.DataFrame(patch_rows)
patch_comparison["absolute_standardized_effect"] = patch_comparison[
    "standardized_effect"
].abs()
patch_comparison["absolute_outlier_frequency_difference"] = patch_comparison[
    "zeiss_minus_clsa_outlier_frequency"
].abs()
write_frame(
    patch_comparison,
    explain_root / "patch_location_comparison.csv",
)
display(
    patch_comparison.sort_values(
        ["absolute_standardized_effect", "absolute_outlier_frequency_difference"],
        ascending=False,
    ).head(20)
)

# COMMAND ----------
individual_outlier_rows = []
for cohort in ("CLSA healthy", "Zeiss glaucoma"):
    cohort_manifest = set_manifest[set_manifest["cohort"] == cohort].reset_index(drop=True)
    robust_z = cohort_stats[cohort]["robust_z"]
    for index, (_, record) in enumerate(cohort_manifest.iterrows()):
        absolute = np.abs(robust_z[index])
        row, column = np.unravel_index(np.argmax(absolute), absolute.shape)
        individual_outlier_rows.append(
            {
                "cohort": cohort,
                "match_set_id": record["match_set_id"],
                "maximum_absolute_spatial_z": float(absolute[row, column]),
                "outlier_patch_row": int(row),
                "outlier_patch_column": int(column),
                "retinal_y_fraction": float((row + 0.5) / height),
                "retinal_x_fraction": float((column + 0.5) / width),
                "is_spatial_outlier": bool(absolute[row, column] >= spatial_outlier_z),
            }
        )
individual_outliers = pd.DataFrame(individual_outlier_rows)
write_frame(
    individual_outliers,
    explain_root / "match_set_spatial_outliers.parquet",
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Group comparison figure

# COMMAND ----------
fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
maps_to_plot = [
    (clsa_mean, "CLSA mean normalized contribution", "coolwarm"),
    (zeiss_mean, "Zeiss mean normalized contribution", "coolwarm"),
    (mean_difference, "Zeiss − CLSA contribution", "coolwarm"),
    (
        cohort_stats["CLSA healthy"]["outlier_frequency"],
        "CLSA spatial outlier frequency",
        "magma",
    ),
    (
        cohort_stats["Zeiss glaucoma"]["outlier_frequency"],
        "Zeiss spatial outlier frequency",
        "magma",
    ),
    (
        outlier_frequency_difference,
        "Zeiss − CLSA outlier frequency",
        "coolwarm",
    ),
]
for axis, (grid, title, cmap) in zip(axes.ravel(), maps_to_plot):
    if cmap == "coolwarm":
        limit = float(np.max(np.abs(grid))) or 1.0
        image = axis.imshow(grid, cmap=cmap, vmin=-limit, vmax=limit)
    else:
        image = axis.imshow(grid, cmap=cmap, vmin=0, vmax=max(float(np.max(grid)), 1e-6))
    axis.set_title(title)
    axis.set_xlabel("Temporal ↔ nasal after left-eye alignment")
    axis.set_ylabel("Superior ↔ inferior")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
fig.savefig(
    figure_root / "matched_group_attribution_comparison.png",
    dpi=240,
    bbox_inches="tight",
)
plt.show()
plt.close(fig)

explain_summary = {
    "model_path": str(model_path),
    "checkpoint": str(resolved_checkpoint),
    "selected_match_sets": len(selected_match_sets),
    "complete_valid_match_sets": len(complete_sets),
    "embedding_cosine_threshold": embedding_cosine_threshold,
    "spatial_outlier_z": spatial_outlier_z,
    "left_eyes_horizontally_aligned": align_left_eyes,
    "source_specific_preprocessing": {
        "CLSA": "CLSA fundus crop plus per-image channel normalization",
        "Zeiss": "attached AutoMorph DICOM crop plus ImageNet normalization",
    },
    "important_limitation": (
        "Spatial differences can still reflect camera/source acquisition because "
        "disease status and data source are confounded."
    ),
}
(explain_root / "matched_explainability_summary.json").write_text(
    json.dumps(explain_summary, indent=2), encoding="utf-8"
)
print(json.dumps(explain_summary, indent=2))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Interpretation guardrail
# MAGIC
# MAGIC A patch difference is a model-attribution difference, not a localized
# MAGIC causal glaucoma lesion. Because CLSA and Zeiss come from different
# MAGIC acquisition systems, interpret group maps as hypothesis-generating until
# MAGIC healthy Zeiss controls or CLSA glaucoma cases are available.

# COMMAND ----------
import os

os.environ.pop("HF_TOKEN", None)
try:
    dbutils.widgets.remove("hf_token")
except Exception:
    pass
print("Temporary Hugging Face token widget removed")
