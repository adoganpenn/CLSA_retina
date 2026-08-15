# Databricks notebook source
# MAGIC %md
# MAGIC # CLSA glaucoma classifier and optic-nerve spatial validation
# MAGIC
# MAGIC This notebook answers a different question from the retinal-age model:
# MAGIC **when a RETFound classifier assigns a glaucoma score, which retinal
# MAGIC regions drive that score?**
# MAGIC
# MAGIC The primary design is deliberately conservative:
# MAGIC
# MAGIC 1. use strict healthy and glaucoma-only CLSA participants from notebook 06;
# MAGIC 2. age/sex/visit-match controls without replacement;
# MAGIC 3. train a nested-CV linear glaucoma head on frozen RETFound embeddings;
# MAGIC 4. evaluate only participant-held-out predictions;
# MAGIC 5. decompose each held-out glaucoma logit exactly into RETFound patches;
# MAGIC 6. quantify optic-disc/peripapillary enrichment and perform targeted
# MAGIC    optic-nerve versus equal-area random occlusion.
# MAGIC
# MAGIC CLSA glaucoma is a released self-report of physician diagnosis, not
# MAGIC clinical adjudication. The resulting model is therefore a **weak-label
# MAGIC research classifier**, not a clinical diagnostic device. Validated
# MAGIC optic-disc coordinates are required before making an anatomic claim;
# MAGIC the automatic bright-disc proxy is explicitly exploratory.
# MAGIC
# MAGIC The gated Hugging Face token is read from a temporary text widget,
# MAGIC placed in `HF_TOKEN` only while the RETFound checkpoint loads, and then
# MAGIC removed from the process environment. Do not print or commit the token.

# COMMAND ----------
# MAGIC %pip install -q "timm>=1.0,<2" "huggingface_hub>=0.24" "pydicom>=2.4,<4"

# COMMAND ----------
from pathlib import Path
import hashlib
import importlib
import json
import math
import os
import sys
import time
import uuid

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
dbutils.widgets.text("three_cohort_root", "")
dbutils.widgets.text("healthy_images_path", "")
dbutils.widgets.text("glaucoma_images_path", "")
dbutils.widgets.text("zeiss_images_path", "")
dbutils.widgets.text("optic_disc_annotations_path", "")
dbutils.widgets.text("output_root", "")
dbutils.widgets.text("control_ratio", "2")
dbutils.widgets.text("age_caliper_years", "1.0")
dbutils.widgets.text("outer_folds", "5")
dbutils.widgets.text("inner_folds", "4")
dbutils.widgets.text("bootstrap_repetitions", "2000")
dbutils.widgets.dropdown("run_explainability", "true", ["true", "false"])
dbutils.widgets.dropdown("explain_all_images", "false", ["false", "true"])
dbutils.widgets.text("n_explain_participants_per_group", "40")
dbutils.widgets.text("max_explain_images_per_participant", "2")
dbutils.widgets.text("explain_batch_size", "50")
dbutils.widgets.dropdown("resume_explainability", "true", ["true", "false"])
dbutils.widgets.dropdown(
    "allow_exploratory_disc_proxy", "true", ["true", "false"]
)
dbutils.widgets.dropdown(
    "require_validated_disc_for_claim", "true", ["true", "false"]
)
dbutils.widgets.text("retfound_repo", "")
dbutils.widgets.text("checkpoint_path", "")
dbutils.widgets.dropdown("allow_repo_clone", "true", ["true", "false"])
dbutils.widgets.dropdown("allow_downloads", "true", ["true", "false"])
dbutils.widgets.text("hf_token", "", "Hugging Face token (temporary)")
dbutils.widgets.dropdown("device", "auto", ["auto", "cuda", "cpu"])
dbutils.widgets.text("embedding_cosine_threshold", "0.999")
dbutils.widgets.text("clsa_logit_replay_tolerance", "0.001")
dbutils.widgets.text("zeiss_probability_replay_tolerance", "0.025")

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
healthy_images_path = configured_path(
    "healthy_images_path",
    age_glaucoma_root / "03_clsa_controls" / "eligible_images_delta",
)
glaucoma_images_path = configured_path(
    "glaucoma_images_path",
    cohort_root / "clsa_glaucoma_only_embeddings_delta",
)
zeiss_images_path = configured_path(
    "zeiss_images_path",
    age_glaucoma_root
    / "01_zeiss_source_cohort"
    / "zeiss_embedded_images.parquet",
)
annotation_text = dbutils.widgets.get("optic_disc_annotations_path").strip()
optic_disc_annotations_path = Path(annotation_text) if annotation_text else None
output_root = configured_path(
    "output_root",
    age_glaucoma_root / "13_glaucoma_classifier_spatial_validation",
)
control_ratio = int(dbutils.widgets.get("control_ratio"))
age_caliper_years = float(dbutils.widgets.get("age_caliper_years"))
outer_folds = int(dbutils.widgets.get("outer_folds"))
inner_folds = int(dbutils.widgets.get("inner_folds"))
bootstrap_repetitions = int(dbutils.widgets.get("bootstrap_repetitions"))
run_explainability_flag = dbutils.widgets.get("run_explainability") == "true"
explain_all_images = dbutils.widgets.get("explain_all_images") == "true"
n_explain_participants_per_group = int(
    dbutils.widgets.get("n_explain_participants_per_group")
)
max_explain_images_per_participant = int(
    dbutils.widgets.get("max_explain_images_per_participant")
)
explain_batch_size = int(dbutils.widgets.get("explain_batch_size"))
resume_explainability = dbutils.widgets.get("resume_explainability") == "true"
allow_exploratory_disc_proxy = (
    dbutils.widgets.get("allow_exploratory_disc_proxy") == "true"
)
require_validated_disc_for_claim = (
    dbutils.widgets.get("require_validated_disc_for_claim") == "true"
)
retfound_repo = dbutils.widgets.get("retfound_repo").strip() or None
checkpoint_path = dbutils.widgets.get("checkpoint_path").strip() or None
allow_repo_clone = dbutils.widgets.get("allow_repo_clone") == "true"
allow_downloads = dbutils.widgets.get("allow_downloads") == "true"
device_requested = dbutils.widgets.get("device")
embedding_cosine_threshold = float(
    dbutils.widgets.get("embedding_cosine_threshold")
)
clsa_logit_replay_tolerance = float(
    dbutils.widgets.get("clsa_logit_replay_tolerance")
)
zeiss_probability_replay_tolerance = float(
    dbutils.widgets.get("zeiss_probability_replay_tolerance")
)

if control_ratio < 1 or control_ratio > 5:
    raise ValueError("control_ratio must be between 1 and 5")
if age_caliper_years < 0:
    raise ValueError("age_caliper_years cannot be negative")
if outer_folds < 3 or inner_folds < 2:
    raise ValueError("Use at least 3 outer and 2 inner folds")
if bootstrap_repetitions < 500:
    raise ValueError("bootstrap_repetitions must be at least 500")
if (
    n_explain_participants_per_group < 1
    or max_explain_images_per_participant < 1
    or explain_batch_size < 1
):
    raise ValueError("Explainability sample and batch sizes must be positive")
if not 0.9 <= embedding_cosine_threshold <= 1.0:
    raise ValueError("embedding_cosine_threshold must lie in [0.9, 1.0]")
if clsa_logit_replay_tolerance <= 0:
    raise ValueError("clsa_logit_replay_tolerance must be positive")
if not 0 < zeiss_probability_replay_tolerance <= 0.10:
    raise ValueError(
        "zeiss_probability_replay_tolerance must lie in (0, 0.10]"
    )

module_root = repo_root / "src"
if not module_root.exists():
    raise FileNotFoundError(f"Repository source directory not found: {module_root}")
if str(module_root) not in sys.path:
    sys.path.insert(0, str(module_root))

import age_gap_extremes as _age_gap_extremes  # noqa: E402
import age_glaucoma_model as _age_glaucoma_model  # noqa: E402
import fundus_retfound_pipeline as _fundus_pipeline  # noqa: E402
import glaucoma_classifier_spatial as _glaucoma_spatial  # noqa: E402
import three_cohort_glaucoma as _three_cohort  # noqa: E402

_age_gap_extremes = importlib.reload(_age_gap_extremes)
_age_glaucoma_model = importlib.reload(_age_glaucoma_model)
_fundus_pipeline = importlib.reload(_fundus_pipeline)
_glaucoma_spatial = importlib.reload(_glaucoma_spatial)
_three_cohort = importlib.reload(_three_cohort)

from age_gap_extremes import fundus_physiology_proxies  # noqa: E402
from age_glaucoma_model import (  # noqa: E402
    exact_linear_patch_map_from_array,
    load_zeiss_retfound_model,
    prepare_zeiss_dicom_input,
)
from fundus_retfound_pipeline import (  # noqa: E402
    QualityConfig,
    RETFoundConfig,
    linear_head_score_from_array,
    load_retfound_model,
    prepare_model_input,
    preprocess_fundus,
    write_frame,
    write_json,
)
from glaucoma_classifier_spatial import (  # noqa: E402
    classifier_metrics,
    fit_oof_linear_classifier,
    match_controls_ratio,
    optic_nerve_masks,
    paired_auc_difference,
    predict_linear_head,
    regional_attribution_metrics,
    sample_equal_area_control_masks,
)
from three_cohort_glaucoma import canonical_sex, validate_embedding_frame  # noqa: E402

print("Loaded RETFound helper:", _fundus_pipeline.__file__)
print("Loaded glaucoma classifier helper:", _glaucoma_spatial.__file__)

# COMMAND ----------
participant_root = output_root / "01_participant_classifier"
image_root = output_root / "02_image_predictions"
attribution_root = output_root / "03_patch_attributions"
anatomy_root = output_root / "04_anatomy"
statistics_root = output_root / "05_statistics"
figure_root = output_root / "06_figures"
for path in (
    participant_root,
    image_root,
    attribution_root,
    anatomy_root,
    statistics_root,
    figure_root,
):
    path.mkdir(parents=True, exist_ok=True)

required_paths = {
    "CLSA healthy participants": cohort_root / "clsa_healthy_participants.parquet",
    "CLSA glaucoma participants": cohort_root / "clsa_glaucoma_only_participants.parquet",
    "Zeiss glaucoma participants": cohort_root / "zeiss_glaucoma_participants.parquet",
    "CLSA healthy image vectors": healthy_images_path,
    "CLSA glaucoma image vectors": glaucoma_images_path,
    "Zeiss glaucoma image vectors": zeiss_images_path,
}
missing = [f"{name}: {path}" for name, path in required_paths.items() if not path.exists()]
if missing:
    raise FileNotFoundError("Required prior-stage outputs are missing:\n- " + "\n- ".join(missing))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Build a confounder-balanced participant cohort
# MAGIC
# MAGIC The classifier is trained on one mean RETFound embedding per selected
# MAGIC participant visit. Controls are matched without replacement. Both eyes
# MAGIC therefore remain on the same side of every model split.

# COMMAND ----------
healthy = pd.read_parquet(required_paths["CLSA healthy participants"])
glaucoma = pd.read_parquet(required_paths["CLSA glaucoma participants"])
for frame, label, cohort in (
    (healthy, 0, "CLSA healthy"),
    (glaucoma, 1, "CLSA glaucoma-only"),
):
    frame.attrs = {}
    frame["glaucoma_label"] = label
    frame["cohort"] = cohort
    if "sex_normalized" not in frame.columns:
        frame["sex_normalized"] = frame.get(
            "sex", pd.Series(index=frame.index, dtype=object)
        ).map(canonical_sex)
    frame["participant_id"] = frame["participant_id"].astype(str)
    frame["visit"] = frame.get(
        "visit", pd.Series("UNKNOWN", index=frame.index)
    ).astype(str).str.upper()

healthy = validate_embedding_frame(healthy, "CLSA healthy participants")
glaucoma = validate_embedding_frame(glaucoma, "CLSA glaucoma participants")
healthy["glaucoma_label"] = 0
glaucoma["glaucoma_label"] = 1
healthy["cohort"] = "CLSA healthy"
glaucoma["cohort"] = "CLSA glaucoma-only"

exact_columns = [
    column
    for column in ("sex_normalized", "visit")
    if column in healthy.columns and column in glaucoma.columns
]
matches = match_controls_ratio(
    glaucoma,
    healthy,
    ratio=control_ratio,
    caliper_years=age_caliper_years,
    exact_columns=exact_columns,
)
if matches.empty:
    raise ValueError("No matched healthy controls were found for CLSA glaucoma")
matched_case_ids = set(matches["case_id"].astype(str))
matched_control_ids = set(matches["control_id"].astype(str))
matched_glaucoma = glaucoma[
    glaucoma["participant_id"].isin(matched_case_ids)
].copy()
matched_healthy = healthy[
    healthy["participant_id"].isin(matched_control_ids)
].copy()
classifier_cohort = pd.concat(
    [matched_healthy, matched_glaucoma],
    ignore_index=True,
)
classifier_cohort = classifier_cohort.sort_values(
    ["glaucoma_label", "participant_id"],
    kind="stable",
).reset_index(drop=True)

write_frame(matches, participant_root / "matched_control_sets_private.parquet")
write_frame(
    classifier_cohort.drop(columns=["embedding"]),
    participant_root / "classifier_cohort_metadata_private.parquet",
)
cohort_balance = classifier_cohort.groupby("glaucoma_label").agg(
    participants=("participant_id", "nunique"),
    mean_age=("age", "mean"),
    sd_age=("age", "std"),
)
display(cohort_balance.round(3))
print("Matched control records:", len(matches))
print("Matched glaucoma participants:", len(matched_glaucoma))
print("Mean absolute age difference:", matches["absolute_age_difference"].mean())
print("Exact matching fields:", exact_columns)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Nested participant-level RETFound glaucoma classifier
# MAGIC
# MAGIC The encoder remains frozen. The outer folds produce held-out participant
# MAGIC probabilities; regularization strength is selected only within each
# MAGIC development fold. The final all-participant head is saved for future
# MAGIC deployment but is not used for the primary performance estimate or
# MAGIC attribution analysis.

# COMMAND ----------
oof_predictions, fold_heads, final_head = fit_oof_linear_classifier(
    classifier_cohort,
    folds=outer_folds,
    inner_folds=inner_folds,
    c_grid=(0.001, 0.01, 0.1, 1.0),
)
participant_predictions = classifier_cohort.drop(columns=["embedding"]).merge(
    oof_predictions,
    on=["participant_id", "glaucoma_label"],
    how="inner",
    validate="one_to_one",
)
if len(participant_predictions) != len(classifier_cohort):
    raise RuntimeError("Participant OOF predictions did not merge one-to-one")

performance = classifier_metrics(
    participant_predictions["glaucoma_label"],
    participant_predictions["glaucoma_probability_oof"],
    bootstrap_repetitions=bootstrap_repetitions,
)
joblib.dump(fold_heads, participant_root / "CLSA_glaucoma_oof_fold_heads.joblib")
joblib.dump(final_head, participant_root / "CLSA_glaucoma_final_head.joblib")
write_frame(
    participant_predictions,
    participant_root / "CLSA_glaucoma_participant_oof_predictions.parquet",
)
write_json(performance, participant_root / "CLSA_glaucoma_performance.json")
display(pd.DataFrame([performance]).round(4))

# The locked all-CLSA head is transported to Zeiss only as a positive-cohort
# application. With no healthy Zeiss arm, these scores cannot estimate Zeiss
# specificity or cross-device AUROC.
zeiss_participants = pd.read_parquet(
    required_paths["Zeiss glaucoma participants"]
)
zeiss_participants.attrs = {}
zeiss_participants = validate_embedding_frame(
    zeiss_participants,
    "Zeiss glaucoma participants",
)
zeiss_participant_scores = predict_linear_head(
    zeiss_participants,
    final_head,
)
zeiss_participant_predictions = zeiss_participants.drop(
    columns=["embedding"]
).copy()
zeiss_participant_predictions["glaucoma_label"] = 1
zeiss_participant_predictions["fold"] = -1
zeiss_participant_predictions["source"] = "Zeiss"
zeiss_participant_predictions["classifier_logit_external"] = (
    zeiss_participant_scores["classifier_logit"]
)
zeiss_participant_predictions["glaucoma_probability_external"] = (
    zeiss_participant_scores["glaucoma_probability"]
)
write_frame(
    zeiss_participant_predictions,
    participant_root / "Zeiss_glaucoma_external_predictions_private.parquet",
)
print("Zeiss glaucoma participants externally scored:", len(zeiss_participants))
print(
    "Mean locked-head Zeiss glaucoma probability:",
    float(
        zeiss_participant_predictions[
            "glaucoma_probability_external"
        ].mean()
    ),
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Demographic shortcut baseline
# MAGIC
# MAGIC A useful retinal classifier must outperform a model that sees only age,
# MAGIC sex, and visit. Because the primary cohort is matched on these fields,
# MAGIC this is also a check that matching worked.

# COMMAND ----------
demographic_columns = ["age"]
demographic = pd.DataFrame(index=classifier_cohort.index)
demographic["age"] = classifier_cohort["age"].astype(float)
for column in ("sex_normalized", "visit"):
    if column in classifier_cohort.columns:
        for category in sorted(classifier_cohort[column].astype(str).unique())[1:]:
            name = f"{column}={category}"
            demographic[name] = (
                classifier_cohort[column].astype(str) == category
            ).astype(float)
            demographic_columns.append(name)
demographic_frame = classifier_cohort[
    ["participant_id", "glaucoma_label"]
].copy()
demographic_frame["embedding"] = [
    row.astype(np.float64)
    for row in demographic[demographic_columns].to_numpy(float)
]
demographic_oof, _, _ = fit_oof_linear_classifier(
    demographic_frame,
    folds=outer_folds,
    inner_folds=inner_folds,
    c_grid=(0.001, 0.01, 0.1, 1.0),
    expected_dim=len(demographic_columns),
)
demographic_performance = classifier_metrics(
    demographic_oof["glaucoma_label"],
    demographic_oof["glaucoma_probability_oof"],
    bootstrap_repetitions=bootstrap_repetitions,
)
model_comparison = pd.DataFrame(
    [
        {"model": "RETFound linear head", **performance},
        {"model": "age/sex/visit baseline", **demographic_performance},
    ]
)
write_frame(model_comparison, participant_root / "classifier_baseline_comparison.csv")
display(model_comparison.round(4))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Attach the held-out head to every selected image
# MAGIC
# MAGIC Existing image vectors are loaded; RETFound inference is not repeated.
# MAGIC The participant mean of the image logits must reconstruct the stored OOF
# MAGIC participant logit before any image is eligible for explainability.

# COMMAND ----------
key_rows = [
    (
        str(row.participant_id),
        str(row.visit).upper(),
        int(row.glaucoma_label),
        int(row.fold),
        float(row.age),
    )
    for row in participant_predictions.itertuples()
]
key_schema = (
    "participant_id string, visit string, glaucoma_label integer, fold integer, "
    "participant_age double"
)
participant_keys = spark.createDataFrame(key_rows, schema=key_schema)


def normalize_image_frame(frame):
    return (
        frame.withColumn(
            "participant_id",
            F.trim(F.col("participant_id").cast("string")),
        )
        .withColumn(
            "visit",
            F.when(F.upper(F.col("visit")).isin("F1", "FUP1"), F.lit("F1"))
            .when(F.upper(F.col("visit")) == "BL", F.lit("BL"))
            .otherwise(F.upper(F.col("visit"))),
        )
    )


healthy_images_spark = normalize_image_frame(
    spark.read.format("delta").load(str(healthy_images_path))
)
glaucoma_images_spark = normalize_image_frame(
    spark.read.format("delta").load(str(glaucoma_images_path))
)
image_required = {"participant_id", "visit", "image_path", "embedding"}
for label, frame in (
    ("Healthy images", healthy_images_spark),
    ("Glaucoma images", glaucoma_images_spark),
):
    missing_columns = image_required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{label} are missing: {sorted(missing_columns)}")

healthy_keys = participant_keys.filter(F.col("glaucoma_label") == 0)
glaucoma_keys = participant_keys.filter(F.col("glaucoma_label") == 1)
healthy_images_spark = healthy_images_spark.join(
    F.broadcast(healthy_keys),
    ["participant_id", "visit"],
    "inner",
)
glaucoma_images_spark = glaucoma_images_spark.join(
    F.broadcast(glaucoma_keys),
    ["participant_id", "visit"],
    "inner",
)
common_columns = [
    column
    for column in (
        "participant_id",
        "visit",
        "glaucoma_label",
        "fold",
        "participant_age",
        "image_path",
        "eye",
        "filename",
        "embedding",
        "retina_fraction",
        "brightness_mean",
        "contrast_std",
        "gradient_energy",
        "dark_fraction",
        "bright_fraction",
    )
    if column in healthy_images_spark.columns
    and column in glaucoma_images_spark.columns
]
image_frame = healthy_images_spark.select(*common_columns).unionByName(
    glaucoma_images_spark.select(*common_columns)
).toPandas()
image_frame.attrs = {}
image_frame = image_frame.rename(columns={"participant_age": "age"})
image_frame = validate_embedding_frame(image_frame, "Matched CLSA image vectors")
if "eye" not in image_frame.columns:
    image_frame["eye"] = "UNKNOWN"
image_frame["glaucoma_label"] = pd.to_numeric(
    image_frame["glaucoma_label"]
).astype(int)
image_frame["fold"] = pd.to_numeric(image_frame["fold"]).astype(int)
image_frame["source"] = "CLSA"

scored_images = []
for fold, head in enumerate(fold_heads):
    selected = image_frame[image_frame["fold"] == fold].copy()
    predictions = predict_linear_head(selected, head)
    selected["classifier_logit_oof"] = predictions["classifier_logit"]
    selected["glaucoma_probability_oof"] = predictions["glaucoma_probability"]
    scored_images.append(selected)
image_predictions = pd.concat(scored_images, ignore_index=True)

reconstructed = (
    image_predictions.groupby("participant_id", as_index=False)
    .agg(reconstructed_participant_logit=("classifier_logit_oof", "mean"))
    .merge(
        participant_predictions[
            ["participant_id", "classifier_logit_oof"]
        ],
        on="participant_id",
        how="inner",
        validate="one_to_one",
    )
)
reconstructed["absolute_reconstruction_error"] = (
    reconstructed["reconstructed_participant_logit"]
    - reconstructed["classifier_logit_oof"]
).abs()
maximum_reconstruction_error = float(
    reconstructed["absolute_reconstruction_error"].max()
)
if maximum_reconstruction_error > 1e-3:
    raise RuntimeError(
        "Image logits do not reconstruct participant logits; maximum error "
        f"was {maximum_reconstruction_error:.6g}."
    )
write_frame(image_predictions, image_root / "CLSA_glaucoma_image_oof_predictions.parquet")
print("Image vectors loaded:", len(image_predictions))
print("Maximum participant-logit reconstruction error:", maximum_reconstruction_error)
print("RETFound embedding inference repeated in this section: false")

zeiss_images = pd.read_parquet(
    required_paths["Zeiss glaucoma image vectors"]
).rename(columns={"patient_id": "participant_id", "dcm_path": "image_path"})
zeiss_images.attrs = {}
zeiss_images = validate_embedding_frame(
    zeiss_images,
    "Zeiss glaucoma image vectors",
)
if "image_path" not in zeiss_images.columns:
    raise ValueError("Zeiss glaucoma image vectors lack image_path/dcm_path")
if "eye" not in zeiss_images.columns:
    zeiss_images["eye"] = "UNKNOWN"
zeiss_images["visit"] = "ZEISS"
zeiss_images["glaucoma_label"] = 1
zeiss_images["fold"] = -1
zeiss_images["source"] = "Zeiss"
zeiss_image_scores = predict_linear_head(zeiss_images, final_head)
zeiss_images["classifier_logit_oof"] = zeiss_image_scores["classifier_logit"]
zeiss_images["glaucoma_probability_oof"] = zeiss_image_scores[
    "glaucoma_probability"
]
zeiss_reconstruction = (
    zeiss_images.groupby("participant_id", as_index=False)
    .agg(reconstructed_participant_logit=("classifier_logit_oof", "mean"))
    .merge(
        zeiss_participant_predictions[
            ["participant_id", "classifier_logit_external"]
        ],
        on="participant_id",
        how="inner",
        validate="one_to_one",
    )
)
zeiss_reconstruction["absolute_reconstruction_error"] = (
    zeiss_reconstruction["reconstructed_participant_logit"]
    - zeiss_reconstruction["classifier_logit_external"]
).abs()
maximum_zeiss_reconstruction_error = float(
    zeiss_reconstruction["absolute_reconstruction_error"].max()
)
if maximum_zeiss_reconstruction_error > 1e-3:
    raise RuntimeError(
        "Zeiss image logits do not reconstruct the participant logits; maximum "
        f"error was {maximum_zeiss_reconstruction_error:.6g}."
    )
write_frame(
    zeiss_images,
    image_root / "Zeiss_glaucoma_image_external_predictions_private.parquet",
)
print("Zeiss image vectors externally scored:", len(zeiss_images))
print(
    "Maximum Zeiss participant-logit reconstruction error:",
    maximum_zeiss_reconstruction_error,
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Image-quality shortcut baseline

# COMMAND ----------
quality_columns = [
    column
    for column in (
        "retina_fraction",
        "brightness_mean",
        "contrast_std",
        "gradient_energy",
        "dark_fraction",
        "bright_fraction",
    )
    if column in image_predictions.columns
]
quality_performance = None
if quality_columns:
    participant_quality = image_predictions.groupby(
        ["participant_id", "glaucoma_label"], as_index=False
    )[quality_columns].mean()
    quality_numeric = participant_quality[quality_columns].apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    quality_columns = [
        column for column in quality_columns if quality_numeric[column].notna().any()
    ]
if quality_columns:
    quality_numeric = quality_numeric[quality_columns]
    quality_numeric = quality_numeric.fillna(quality_numeric.median())
    participant_quality["embedding"] = [
        row.astype(np.float64)
        for row in quality_numeric.to_numpy(float)
    ]
    quality_oof, _, _ = fit_oof_linear_classifier(
        participant_quality,
        folds=outer_folds,
        inner_folds=inner_folds,
        c_grid=(0.001, 0.01, 0.1, 1.0),
        expected_dim=len(quality_columns),
    )
    quality_performance = classifier_metrics(
        quality_oof["glaucoma_label"],
        quality_oof["glaucoma_probability_oof"],
        bootstrap_repetitions=bootstrap_repetitions,
    )
    model_comparison = pd.concat(
        [
            model_comparison,
            pd.DataFrame(
                [{"model": "image-quality baseline", **quality_performance}]
            ),
        ],
        ignore_index=True,
    )
    write_frame(
        model_comparison,
        participant_root / "classifier_baseline_comparison.csv",
    )
display(model_comparison.round(4))

demographic_shortcut_test = paired_auc_difference(
    participant_predictions,
    demographic_oof,
    reference_probability="glaucoma_probability_oof",
    comparator_probability="glaucoma_probability_oof",
    bootstrap_repetitions=bootstrap_repetitions,
    random_state=20260821,
)
quality_shortcut_test = None
if quality_columns:
    quality_shortcut_test = paired_auc_difference(
        participant_predictions,
        quality_oof,
        reference_probability="glaucoma_probability_oof",
        comparator_probability="glaucoma_probability_oof",
        bootstrap_repetitions=bootstrap_repetitions,
        random_state=20260822,
    )
shortcut_tests = {
    "retfound_minus_demographic_auroc": demographic_shortcut_test,
    "retfound_minus_quality_auroc": quality_shortcut_test,
}
shortcut_baselines_outperformed = bool(
    demographic_shortcut_test["auroc_difference_95_ci_low"] > 0
    and (
        quality_shortcut_test is None
        or quality_shortcut_test["auroc_difference_95_ci_low"] > 0
    )
)
write_json(shortcut_tests, participant_root / "shortcut_baseline_tests.json")
display(pd.DataFrame(shortcut_tests).T)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Lock the held-out explainability sample
# MAGIC
# MAGIC Sampling occurs randomly at participant level, stratified by CLSA label,
# MAGIC with a separate random Zeiss sample. This avoids selecting spatial maps
# MAGIC because they produced an extreme score. All available eyes from each
# MAGIC selected participant are then explained; success/error labels are
# MAGIC retained for descriptive overlays.

# COMMAND ----------
participant_predictions["classification_group"] = np.select(
    [
        (participant_predictions["glaucoma_label"] == 1)
        & (participant_predictions["glaucoma_probability_oof"] >= 0.5),
        (participant_predictions["glaucoma_label"] == 1)
        & (participant_predictions["glaucoma_probability_oof"] < 0.5),
        (participant_predictions["glaucoma_label"] == 0)
        & (participant_predictions["glaucoma_probability_oof"] >= 0.5),
        (participant_predictions["glaucoma_label"] == 0)
        & (participant_predictions["glaucoma_probability_oof"] < 0.5),
    ],
    ["true_positive", "false_negative", "false_positive", "true_negative"],
    default="unclassified",
)
participant_predictions["source"] = "CLSA"
if explain_all_images:
    selected_clsa_participants = participant_predictions.copy()
else:
    samples = []
    for label, group_frame in participant_predictions.groupby(
        "glaucoma_label"
    ):
        samples.append(
            group_frame.sample(
                n=min(n_explain_participants_per_group, len(group_frame)),
                random_state=20260811 + int(label),
            )
        )
    selected_clsa_participants = pd.concat(samples, ignore_index=True)

zeiss_for_sampling = zeiss_participant_predictions.rename(
    columns={
        "classifier_logit_external": "classifier_logit_oof",
        "glaucoma_probability_external": "glaucoma_probability_oof",
    }
).copy()
if explain_all_images:
    selected_zeiss_participants = zeiss_for_sampling.copy()
    selected_zeiss_participants["classification_group"] = "external_zeiss"
else:
    selected_zeiss_participants = zeiss_for_sampling.sample(
        n=min(n_explain_participants_per_group, len(zeiss_for_sampling)),
        random_state=20260813,
    ).copy()
    selected_zeiss_participants[
        "classification_group"
    ] = "external_zeiss"

selected_participants = pd.concat(
    [selected_clsa_participants, selected_zeiss_participants],
    ignore_index=True,
    sort=False,
)
selected_clsa_ids = set(
    selected_clsa_participants["participant_id"].astype(str)
)
clsa_explain_images = image_predictions[
    image_predictions["participant_id"].astype(str).isin(selected_clsa_ids)
].merge(
    selected_clsa_participants[
        ["participant_id", "source", "classification_group"]
    ],
    on=["participant_id", "source"],
    how="inner",
    validate="many_to_one",
)
selected_zeiss_ids = set(
    selected_zeiss_participants["participant_id"].astype(str)
)
zeiss_explain_images = zeiss_images[
    zeiss_images["participant_id"].astype(str).isin(selected_zeiss_ids)
].merge(
    selected_zeiss_participants[
        ["participant_id", "source", "classification_group"]
    ],
    on=["participant_id", "source"],
    how="inner",
    validate="many_to_one",
)
explain_images = pd.concat(
    [clsa_explain_images, zeiss_explain_images],
    ignore_index=True,
    sort=False,
)
# A Zeiss participant can have many repeated DICOM acquisitions. Select the
# image nearest that participant's mean classifier logit within each eye, then
# cap the participant total. This avoids cherry-picking extreme predictions and
# prevents a small participant sample from expanding into tens of thousands of
# explanation runs.
explain_images["eye"] = explain_images["eye"].fillna("UNKNOWN").astype(str)
explain_images["_participant_mean_logit"] = explain_images.groupby(
    ["source", "participant_id"]
)["classifier_logit_oof"].transform("mean")
explain_images["_representative_distance"] = (
    explain_images["classifier_logit_oof"]
    - explain_images["_participant_mean_logit"]
).abs()
explain_images = (
    explain_images.sort_values(
        [
            "source",
            "participant_id",
            "eye",
            "_representative_distance",
            "image_path",
        ],
        kind="stable",
    )
    .drop_duplicates(["source", "participant_id", "eye"], keep="first")
    .groupby(["source", "participant_id"], sort=False, group_keys=False)
    .head(max_explain_images_per_participant)
    .drop(columns=["_participant_mean_logit", "_representative_distance"])
    .reset_index(drop=True)
)
write_frame(
    selected_participants,
    attribution_root / "selected_participants_private.parquet",
)
annotation_template = explain_images[
    ["image_path", "participant_id", "source", "visit", "eye"]
].copy()
annotation_template["center_x_fraction"] = np.nan
annotation_template["center_y_fraction"] = np.nan
annotation_template["radius_fraction"] = np.nan
annotation_template["annotation_source"] = ""
annotation_template["validated"] = False
write_frame(annotation_template, anatomy_root / "optic_disc_annotation_template_private.csv")
print("Selected participants:", len(selected_participants))
print("Selected images:", len(explain_images))
print(
    "Maximum representative images per participant:",
    max_explain_images_per_participant,
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Load independent optic-disc annotations
# MAGIC
# MAGIC Required normalized fields are `image_path`, `center_x_fraction`,
# MAGIC `center_y_fraction`, `radius_fraction`, and `validated`. If annotations
# MAGIC are absent, the notebook may run the automatic bright-disc proxy, but
# MAGIC the final summary will keep `anatomic_claim_ready=false`.

# COMMAND ----------
annotation_columns = {
    "image_path",
    "center_x_fraction",
    "center_y_fraction",
    "radius_fraction",
    "validated",
}
if optic_disc_annotations_path is not None:
    if not optic_disc_annotations_path.exists():
        raise FileNotFoundError(
            f"Optic-disc annotation file not found: {optic_disc_annotations_path}"
        )
    if optic_disc_annotations_path.suffix.lower() == ".csv":
        disc_annotations = pd.read_csv(optic_disc_annotations_path)
    else:
        disc_annotations = pd.read_parquet(optic_disc_annotations_path)
    missing_annotation_columns = annotation_columns - set(disc_annotations.columns)
    if missing_annotation_columns:
        raise ValueError(
            "Optic-disc annotations are missing: "
            f"{sorted(missing_annotation_columns)}"
        )
    disc_annotations = disc_annotations.drop_duplicates("image_path")
    disc_annotations["validated"] = (
        disc_annotations["validated"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )
else:
    disc_annotations = pd.DataFrame(columns=sorted(annotation_columns))

explain_images = explain_images.merge(
    disc_annotations,
    on="image_path",
    how="left",
    validate="one_to_one",
)
explain_images["validated"] = explain_images["validated"].fillna(False).astype(bool)
validated_annotation_coverage = float(explain_images["validated"].mean())
if not allow_exploratory_disc_proxy and validated_annotation_coverage < 1.0:
    raise ValueError(
        "Some selected images lack validated optic-disc annotations and "
        "allow_exploratory_disc_proxy=false. Complete the annotation template."
    )
print("Validated optic-disc annotation coverage:", validated_annotation_coverage)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Resumable exact patch decomposition and targeted occlusion

# COMMAND ----------
quality_config = QualityConfig(
    output_size=256,
    model_input_size=224,
    save_preprocessed=False,
)
retfound_config = RETFoundConfig(
    repo_path=retfound_repo,
    checkpoint_path=checkpoint_path,
    allow_downloads=allow_downloads,
    allow_repo_clone=allow_repo_clone,
    device=device_requested,
    batch_size=1,
)


def stable_image_key(image_path):
    return hashlib.sha256(str(image_path).encode("utf-8")).hexdigest()[:20]


def publish_local_artifact(local_path, volume_path, retries=4):
    """Copy a completed local file into a Volume without seek-based writes."""
    local_path = Path(local_path)
    volume_path = Path(volume_path)
    if not local_path.is_file() or local_path.stat().st_size < 1:
        raise OSError(f"Local artifact is absent or empty: {local_path}")
    volume_path.parent.mkdir(parents=True, exist_ok=True)
    source_uri = f"file:{local_path.resolve()}"
    expected_bytes = int(local_path.stat().st_size)
    last_error = None
    for attempt in range(1, retries + 1):
        partial_path = volume_path.with_name(
            f".{volume_path.name}.{uuid.uuid4().hex}.partial"
        )
        try:
            dbutils.fs.cp(source_uri, str(partial_path), False)
            partial_bytes = int(partial_path.stat().st_size)
            if partial_bytes != expected_bytes:
                raise OSError(
                    "Volume staging copy has the wrong size: "
                    f"{partial_bytes} != {expected_bytes}"
                )
            if volume_path.exists():
                dbutils.fs.rm(str(volume_path), False)
            dbutils.fs.mv(str(partial_path), str(volume_path), False)
            final_bytes = int(volume_path.stat().st_size)
            if final_bytes != expected_bytes:
                raise OSError(
                    "Published Volume artifact has the wrong size: "
                    f"{final_bytes} != {expected_bytes}"
                )
            return volume_path
        except Exception as error:
            last_error = error
            try:
                dbutils.fs.rm(str(partial_path), False)
            except Exception:
                pass
            if attempt < retries:
                delay = 2 ** (attempt - 1)
                print(
                    f"Volume artifact copy attempt {attempt}/{retries} failed; "
                    f"retrying in {delay}s: {type(error).__name__}: {error}"
                )
                time.sleep(delay)
    raise OSError(
        f"Could not publish local artifact to {volume_path} after {retries} attempts"
    ) from last_error


def proxy_disc_coordinates(proxies):
    mask = np.asarray(proxies["optic_disc_proxy"], dtype=bool)
    yy, xx = np.where(mask)
    if not len(xx):
        raise ValueError("Automatic optic-disc proxy is empty")
    height, width = mask.shape
    return {
        "center_x_fraction": float(xx.mean() / max(width - 1, 1)),
        "center_y_fraction": float(yy.mean() / max(height - 1, 1)),
        "radius_fraction": float(
            math.sqrt(mask.sum() / math.pi) / min(height, width)
        ),
    }


def resize_boolean_mask(mask, shape):
    from PIL import Image

    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
    nearest = getattr(Image.Resampling, "NEAREST", Image.NEAREST)
    resized = image.resize((shape[1], shape[0]), resample=nearest)
    return np.asarray(resized) > 0


def probability_from_logit(value):
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -40, 40))))


def targeted_occlusion(
    model,
    device,
    head,
    input_array,
    disc_coordinates,
    proxies,
    random_state,
):
    array = np.asarray(input_array, dtype=np.float32)
    shape = array.shape[:2]
    masks = optic_nerve_masks(shape, **disc_coordinates)
    retina = resize_boolean_mask(proxies["retina"], shape)

    def score(mask=None, retain_only=False):
        altered = array.copy()
        if mask is not None:
            if retain_only:
                altered[~mask] = 0.0
            else:
                altered[mask] = 0.0
        return linear_head_score_from_array(
            model,
            head["coefficients"],
            head["intercept"],
            altered,
            device,
        )

    baseline = score()
    disc = masks["optic_disc"] & retina
    combined = masks["optic_disc_plus_peripapillary"] & retina
    disc_masked = score(disc)
    combined_masked = score(combined)
    disc_only = score(combined, retain_only=True)
    disc_removed = combined_masked
    controls = sample_equal_area_control_masks(
        retina,
        combined,
        target_area=int(max(combined.sum(), 1)),
        n_masks=20,
        random_state=random_state,
    )
    random_drops = np.asarray(
        [baseline - score(mask) for mask in controls], dtype=float
    )
    return {
        "baseline_logit_replay": baseline,
        "optic_disc_occlusion_logit_drop": baseline - disc_masked,
        "disc_plus_peripapillary_occlusion_logit_drop": baseline
        - combined_masked,
        "equal_area_random_occlusion_logit_drop_mean": float(
            random_drops.mean()
        )
        if len(random_drops)
        else np.nan,
        "equal_area_random_occlusion_logit_drop_median": float(
            np.median(random_drops)
        )
        if len(random_drops)
        else np.nan,
        "disc_specific_occlusion_drop": float(
            baseline - combined_masked - np.median(random_drops)
        )
        if len(random_drops)
        else np.nan,
        "disc_only_logit": disc_only,
        "disc_removed_logit": disc_removed,
        "disc_only_probability": probability_from_logit(disc_only),
        "disc_removed_probability": probability_from_logit(disc_removed),
        "n_equal_area_random_masks": int(len(random_drops)),
    }


def save_overlay(processed_rgb, grid, masks, record, output_path):
    from scipy import ndimage

    resized = ndimage.zoom(
        np.asarray(grid),
        (
            processed_rgb.shape[0] / np.asarray(grid).shape[0],
            processed_rgb.shape[1] / np.asarray(grid).shape[1],
        ),
        order=1,
    )[: processed_rgb.shape[0], : processed_rgb.shape[1]]
    limit = max(float(np.quantile(np.abs(resized), 0.99)), 1e-8)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(processed_rgb)
    axes[0].set_title("Processed fundus")
    axes[1].imshow(resized, cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[1].set_title("Exact glaucoma-logit contribution")
    axes[2].imshow(processed_rgb)
    axes[2].imshow(
        resized,
        cmap="coolwarm",
        alpha=0.45,
        vmin=-limit,
        vmax=limit,
    )
    for mask, color in (
        (masks["optic_disc"], "cyan"),
        (masks["optic_disc_plus_peripapillary"], "yellow"),
    ):
        axes[2].contour(mask, levels=[0.5], colors=[color], linewidths=1)
    axes[2].set_title(
        f"{record['classification_group']} | p={record['glaucoma_probability_oof']:.2f}"
    )
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


if run_explainability_flag:
    temporary_hf_token = dbutils.widgets.get("hf_token").strip()
    if allow_downloads and not checkpoint_path and not temporary_hf_token:
        raise ValueError(
            "Enter the temporary Hugging Face token in the hf_token widget, "
            "or provide checkpoint_path."
        )
    if temporary_hf_token:
        os.environ["HF_TOKEN"] = temporary_hf_token
    try:
        try:
            clsa_model, device, resolved_repo, resolved_checkpoint = (
                load_retfound_model(retfound_config)
            )
            zeiss_model = load_zeiss_retfound_model(
                resolved_checkpoint, device
            )
        except Exception as error:
            message = str(error).lower()
            if any(
                marker in message
                for marker in (
                    "gated repo",
                    "gatedrepo",
                    "401 client",
                    "unauthorized",
                    "forbidden",
                )
            ):
                raise RuntimeError(
                    "Hugging Face rejected the RETFound checkpoint request. "
                    "The hf_token must belong to an account that has accepted "
                    "access for YukunZhou/RETFound_mae_natureCFP. No token was "
                    "saved."
                ) from None
            raise
    finally:
        os.environ.pop("HF_TOKEN", None)
        temporary_hf_token = ""
    print("Explainability device:", device)
    print("RETFound repository:", resolved_repo)
    print("RETFound checkpoint:", resolved_checkpoint)

    batch_root = attribution_root / "batches"
    batch_root.mkdir(parents=True, exist_ok=True)
    local_artifact_root = Path(
        f"/local_disk0/tmp/clsa_glaucoma_attribution_{os.getpid()}"
    )
    local_artifact_root.mkdir(parents=True, exist_ok=True)
    print("Local explainability artifact staging:", local_artifact_root)
    ordered = explain_images.sort_values(
        ["classification_group", "participant_id", "image_path"],
        kind="stable",
    ).reset_index(drop=True)
    batch_manifests = []
    n_batches = math.ceil(len(ordered) / explain_batch_size)
    for batch_index, start in enumerate(
        range(0, len(ordered), explain_batch_size), start=1
    ):
        stop = min(start + explain_batch_size, len(ordered))
        batch_directory = batch_root / f"batch_{start:07d}_{stop:07d}"
        batch_directory.mkdir(parents=True, exist_ok=True)
        manifest_path = batch_directory / "glaucoma_attribution_manifest.parquet"
        if resume_explainability and manifest_path.exists():
            existing_manifest = pd.read_parquet(manifest_path)
            required_resume_columns = {
                "image_key",
                "source",
                "stored_logit_replay_error",
                "stored_probability_replay_error",
                "replay_validation_metric",
                "stored_embedding_cosine",
                "attribution_npz_path",
                "overlay_path",
            }
            expected_batch = ordered.iloc[start:stop]
            if not allow_exploratory_disc_proxy:
                expected_batch = expected_batch[
                    expected_batch["validated"].astype(bool)
                ]
            expected_keys = {
                stable_image_key(path)
                for path in expected_batch["image_path"].astype(str)
            }
            manifest_keys_match = (
                "image_key" in existing_manifest.columns
                and set(existing_manifest["image_key"].astype(str))
                == expected_keys
            )
            resume_artifacts_valid = False
            if required_resume_columns.issubset(existing_manifest.columns):
                artifact_paths = [
                    Path(path)
                    for column in ("attribution_npz_path", "overlay_path")
                    for path in existing_manifest[column].dropna().astype(str)
                ]
                resume_artifacts_valid = bool(artifact_paths) and all(
                    path.is_file() and path.stat().st_size > 0
                    for path in artifact_paths
                )
            if (
                required_resume_columns.issubset(existing_manifest.columns)
                and manifest_keys_match
                and resume_artifacts_valid
            ):
                print(
                    f"[explain {batch_index}/{n_batches}] resume "
                    f"{manifest_path}"
                )
                batch_manifests.append(existing_manifest)
                continue
            print(
                f"[explain {batch_index}/{n_batches}] recomputing incomplete, "
                f"legacy, or input-mismatched batch: {manifest_path}"
            )
        print(f"[explain {batch_index}/{n_batches}] images {start:,}:{stop:,}")
        rows = []
        for local_index, record in ordered.iloc[start:stop].iterrows():
            head = (
                final_head
                if str(record["source"]) == "Zeiss"
                else fold_heads[int(record["fold"])]
            )
            if str(record["source"]) == "Zeiss":
                input_array, display_image = prepare_zeiss_dicom_input(
                    record["image_path"]
                )
                source_model = zeiss_model
                processed_rgb = np.asarray(display_image.convert("RGB"))
            else:
                input_array = prepare_model_input(
                    record["image_path"], quality_config
                )
                source_model = clsa_model
                processed = preprocess_fundus(
                    record["image_path"], quality_config
                )
                processed_rgb = np.asarray(processed.image.convert("RGB"))
            attribution = exact_linear_patch_map_from_array(
                model=source_model,
                coefficients=head["coefficients"],
                intercept=head["intercept"],
                input_array=input_array,
                device=device,
                stored_embedding=record["embedding"],
            )
            stored_logit = float(record["classifier_logit_oof"])
            replayed_logit = float(attribution["prediction"])
            replay_error = abs(replayed_logit - stored_logit)
            stored_probability = probability_from_logit(stored_logit)
            replayed_probability = probability_from_logit(replayed_logit)
            replay_probability_error = abs(
                replayed_probability - stored_probability
            )
            if str(record["source"]) == "Zeiss":
                replay_validation_metric = "probability_absolute_difference"
                replay_validation_value = replay_probability_error
                replay_validation_tolerance = (
                    zeiss_probability_replay_tolerance
                )
            else:
                replay_validation_metric = "logit_absolute_difference"
                replay_validation_value = replay_error
                replay_validation_tolerance = clsa_logit_replay_tolerance
            if (
                attribution["reconstruction_error"] > 1e-4
                or replay_validation_value > replay_validation_tolerance
                or not np.isfinite(attribution["embedding_cosine"])
                or attribution["embedding_cosine"]
                < embedding_cosine_threshold
            ):
                raise RuntimeError(
                    "Exact glaucoma attribution failed reconstruction for "
                    f"{record['image_path']}: patch={attribution['reconstruction_error']}, "
                    f"logit_difference={replay_error}, "
                    f"probability_difference={replay_probability_error}, "
                    f"validation={replay_validation_metric} "
                    f"{replay_validation_value}>{replay_validation_tolerance}, "
                    f"cosine={attribution['embedding_cosine']}"
                )
            proxies = fundus_physiology_proxies(processed_rgb)
            if bool(record["validated"]):
                disc_coordinates = {
                    "center_x_fraction": float(record["center_x_fraction"]),
                    "center_y_fraction": float(record["center_y_fraction"]),
                    "radius_fraction": float(record["radius_fraction"]),
                }
                anatomy_source = str(
                    record.get("annotation_source", "validated_annotation")
                )
                anatomy_validated = True
            else:
                if not allow_exploratory_disc_proxy:
                    continue
                disc_coordinates = proxy_disc_coordinates(proxies)
                anatomy_source = "exploratory_bright_disc_proxy"
                anatomy_validated = False
            masks = optic_nerve_masks(
                processed_rgb.shape[:2],
                **disc_coordinates,
            )
            regional = regional_attribution_metrics(
                attribution["variable_grid"],
                proxies["retina"],
                masks,
            )
            occlusion = targeted_occlusion(
                source_model,
                device,
                head,
                input_array,
                disc_coordinates,
                proxies,
                random_state=20260811 + int(local_index),
            )
            key = stable_image_key(record["image_path"])
            local_npz_path = (
                local_artifact_root
                / f"{key}_exact_glaucoma_attribution.npz"
            )
            np.savez_compressed(
                local_npz_path,
                grid=attribution["grid"],
                variable_grid=attribution["variable_grid"],
            )
            with np.load(local_npz_path) as saved_attribution:
                if not {"grid", "variable_grid"}.issubset(
                    saved_attribution.files
                ):
                    raise OSError(
                        f"Locally staged attribution NPZ is invalid: {local_npz_path}"
                    )
            published_npz_path = publish_local_artifact(
                local_npz_path,
                batch_directory / local_npz_path.name,
            )
            local_overlay_path = (
                local_artifact_root / f"{key}_glaucoma_overlay.png"
            )
            save_overlay(
                processed_rgb,
                attribution["variable_grid"],
                masks,
                record,
                local_overlay_path,
            )
            published_overlay_path = publish_local_artifact(
                local_overlay_path,
                batch_directory / local_overlay_path.name,
            )
            local_npz_path.unlink(missing_ok=True)
            local_overlay_path.unlink(missing_ok=True)
            rows.append(
                {
                    "image_key": key,
                    "image_path": record["image_path"],
                    "participant_id": str(record["participant_id"]),
                    "source": str(record["source"]),
                    "visit": record["visit"],
                    "eye": record.get("eye"),
                    "glaucoma_label": int(record["glaucoma_label"]),
                    "fold": int(record["fold"]),
                    "classification_group": record["classification_group"],
                    "attribution_npz_path": str(published_npz_path),
                    "overlay_path": str(published_overlay_path),
                    "glaucoma_probability_oof": float(
                        record["glaucoma_probability_oof"]
                    ),
                    "classifier_logit_oof": stored_logit,
                    "replayed_classifier_logit": replayed_logit,
                    "stored_glaucoma_probability": stored_probability,
                    "replayed_glaucoma_probability": replayed_probability,
                    "patch_reconstruction_error": float(
                        attribution["reconstruction_error"]
                    ),
                    "stored_logit_replay_error": replay_error,
                    "stored_probability_replay_error": (
                        replay_probability_error
                    ),
                    "replay_validation_metric": replay_validation_metric,
                    "replay_validation_tolerance": (
                        replay_validation_tolerance
                    ),
                    "stored_embedding_cosine": float(
                        attribution["embedding_cosine"]
                    ),
                    "stored_embedding_max_absolute_difference": float(
                        attribution["embedding_max_absolute_difference"]
                    ),
                    "anatomy_source": anatomy_source,
                    "anatomy_validated": anatomy_validated,
                    **disc_coordinates,
                    **regional,
                    **occlusion,
                }
            )
        batch_manifest = pd.DataFrame(rows)
        write_frame(batch_manifest, manifest_path)
        batch_manifests.append(batch_manifest)
        print(
            f"[explain {batch_index}/{n_batches}] saved {len(batch_manifest)} rows"
        )
    attribution_manifest = pd.concat(batch_manifests, ignore_index=True)
    write_frame(
        attribution_manifest,
        attribution_root / "glaucoma_attribution_manifest_private.parquet",
    )
else:
    attribution_manifest = pd.DataFrame()
    resolved_repo = None
    resolved_checkpoint = None
    print("Explainability disabled; classifier outputs are complete.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. Participant-level spatial and occlusion inference

# COMMAND ----------
spatial_summary = pd.DataFrame()
anatomic_claim_ready = False
clsa_glaucoma_validated_coverage = None
if not attribution_manifest.empty:
    metric_columns = [
        "optic_disc_positive_enrichment",
        "optic_disc_plus_peripapillary_positive_enrichment",
        "positive_peak_distance_from_disc_radii",
        "disc_plus_peripapillary_occlusion_logit_drop",
        "equal_area_random_occlusion_logit_drop_median",
        "disc_specific_occlusion_drop",
        "disc_only_probability",
        "disc_removed_probability",
    ]
    participant_spatial = attribution_manifest.groupby(
        [
            "participant_id",
            "source",
            "glaucoma_label",
            "classification_group",
            "anatomy_validated",
        ],
        as_index=False,
    )[metric_columns].mean()
    write_frame(
        participant_spatial,
        statistics_root / "participant_spatial_metrics_private.parquet",
    )

    rng = np.random.default_rng(20260811)
    summary_rows = []
    for (source, label), group in participant_spatial.groupby(
        ["source", "glaucoma_label"]
    ):
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            if not len(values):
                continue
            means = np.asarray(
                [
                    rng.choice(values, len(values), replace=True).mean()
                    for _ in range(bootstrap_repetitions)
                ]
            )
            summary_rows.append(
                {
                    "source": str(source),
                    "glaucoma_label": int(label),
                    "metric": metric,
                    "n_participants": int(len(values)),
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "bootstrap_95_ci_low": float(np.quantile(means, 0.025)),
                    "bootstrap_95_ci_high": float(np.quantile(means, 0.975)),
                }
            )
    spatial_summary = pd.DataFrame(summary_rows)
    write_frame(spatial_summary, statistics_root / "spatial_metric_summary.csv")
    display(spatial_summary.round(4))

    from scipy import stats

    glaucoma_spatial = participant_spatial[
        (participant_spatial["source"] == "CLSA")
        & (participant_spatial["glaucoma_label"] == 1)
    ].copy()
    control_spatial = participant_spatial[
        (participant_spatial["source"] == "CLSA")
        & (participant_spatial["glaucoma_label"] == 0)
    ].copy()
    validated_fraction = float(glaucoma_spatial["anatomy_validated"].mean())
    clsa_glaucoma_validated_coverage = validated_fraction
    if require_validated_disc_for_claim:
        glaucoma_claim_frame = glaucoma_spatial[
            glaucoma_spatial["anatomy_validated"]
        ].copy()
        control_claim_frame = control_spatial[
            control_spatial["anatomy_validated"]
        ].copy()
    else:
        glaucoma_claim_frame = glaucoma_spatial.copy()
        control_claim_frame = control_spatial.copy()
    paired_drop = glaucoma_claim_frame[
        "disc_specific_occlusion_drop"
    ].dropna().to_numpy(float)
    enrichment = glaucoma_claim_frame[
        "optic_disc_plus_peripapillary_positive_enrichment"
    ].dropna().to_numpy(float)
    zeiss_spatial = participant_spatial[
        participant_spatial["source"] == "Zeiss"
    ].copy()

    def bootstrap_mean_interval(values):
        values = np.asarray(values, dtype=float)
        if not len(values):
            return None, None, None
        means = np.asarray(
            [
                rng.choice(values, len(values), replace=True).mean()
                for _ in range(bootstrap_repetitions)
            ]
        )
        return (
            float(values.mean()),
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        )

    def bootstrap_difference_interval(case_values, control_values):
        case_values = np.asarray(case_values, dtype=float)
        control_values = np.asarray(control_values, dtype=float)
        if not len(case_values) or not len(control_values):
            return None, None, None
        differences = np.asarray(
            [
                rng.choice(case_values, len(case_values), replace=True).mean()
                - rng.choice(
                    control_values, len(control_values), replace=True
                ).mean()
                for _ in range(bootstrap_repetitions)
            ]
        )
        observed = float(case_values.mean() - control_values.mean())
        return (
            observed,
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        )

    drop_mean, drop_ci_low, drop_ci_high = bootstrap_mean_interval(paired_drop)
    enrichment_mean, enrichment_ci_low, enrichment_ci_high = (
        bootstrap_mean_interval(enrichment)
    )
    control_drop = control_claim_frame[
        "disc_specific_occlusion_drop"
    ].dropna().to_numpy(float)
    control_enrichment = control_claim_frame[
        "optic_disc_plus_peripapillary_positive_enrichment"
    ].dropna().to_numpy(float)
    drop_difference, drop_difference_low, drop_difference_high = (
        bootstrap_difference_interval(paired_drop, control_drop)
    )
    enrichment_difference, enrichment_difference_low, enrichment_difference_high = (
        bootstrap_difference_interval(enrichment, control_enrichment)
    )
    spatial_tests = {
        "analysis_population": (
            "glaucoma participants with validated anatomy"
            if require_validated_disc_for_claim
            else "glaucoma participants with available anatomy"
        ),
        "n_occlusion_participants": int(len(paired_drop)),
        "n_enrichment_participants": int(len(enrichment)),
        "mean_disc_specific_logit_drop": drop_mean,
        "disc_specific_logit_drop_95_ci": [drop_ci_low, drop_ci_high],
        "mean_optic_nerve_positive_enrichment": enrichment_mean,
        "optic_nerve_positive_enrichment_95_ci": [
            enrichment_ci_low,
            enrichment_ci_high,
        ],
        "clsa_glaucoma_minus_healthy_disc_specific_drop": drop_difference,
        "clsa_glaucoma_minus_healthy_disc_specific_drop_95_ci": [
            drop_difference_low,
            drop_difference_high,
        ],
        "clsa_glaucoma_minus_healthy_optic_nerve_enrichment": (
            enrichment_difference
        ),
        "clsa_glaucoma_minus_healthy_optic_nerve_enrichment_95_ci": [
            enrichment_difference_low,
            enrichment_difference_high,
        ],
        "external_zeiss_n_participants": int(len(zeiss_spatial)),
        "external_zeiss_mean_disc_specific_logit_drop": float(
            zeiss_spatial["disc_specific_occlusion_drop"].mean()
        )
        if len(zeiss_spatial)
        else None,
        "external_zeiss_mean_optic_nerve_positive_enrichment": float(
            zeiss_spatial[
                "optic_disc_plus_peripapillary_positive_enrichment"
            ].mean()
        )
        if len(zeiss_spatial)
        else None,
        "wilcoxon_p_value": float(
            stats.wilcoxon(paired_drop, alternative="greater").pvalue
        )
        if len(paired_drop) and not np.allclose(paired_drop, 0)
        else None,
    }
    write_json(spatial_tests, statistics_root / "optic_nerve_spatial_tests.json")

    classifier_signal_present = performance["auroc_95_ci_low"] > 0.5
    validated_requirement_met = (
        validated_fraction >= 0.80
        if require_validated_disc_for_claim
        else True
    )
    causal_occlusion_present = bool(
        len(paired_drop)
        and drop_ci_low is not None
        and drop_ci_low > 0
        and spatial_tests["wilcoxon_p_value"] is not None
        and spatial_tests["wilcoxon_p_value"] < 0.05
    )
    optic_nerve_enrichment_present = bool(
        len(enrichment)
        and enrichment_ci_low is not None
        and enrichment_ci_low > 1.0
    )
    glaucoma_control_spatial_difference_present = bool(
        drop_difference_low is not None
        and drop_difference_low > 0
        and enrichment_difference_low is not None
        and enrichment_difference_low > 0
    )
    anatomic_claim_ready = bool(
        classifier_signal_present
        and shortcut_baselines_outperformed
        and validated_requirement_met
        and causal_occlusion_present
        and optic_nerve_enrichment_present
        and glaucoma_control_spatial_difference_present
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    participant_spatial.boxplot(
        column="optic_disc_plus_peripapillary_positive_enrichment",
        by="classification_group",
        ax=axes[0],
        rot=30,
    )
    axes[0].axhline(1.0, color="black", linestyle="--")
    axes[0].set_title("Optic-nerve attribution enrichment")
    axes[0].set_ylabel("Positive attribution / area")
    participant_spatial.boxplot(
        column="disc_specific_occlusion_drop",
        by="classification_group",
        ax=axes[1],
        rot=30,
    )
    axes[1].axhline(0.0, color="black", linestyle="--")
    axes[1].set_title("Optic-nerve drop minus random-region drop")
    axes[1].set_ylabel("Change in glaucoma logit")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(figure_root / "optic_nerve_spatial_validation.png", dpi=200)
    display(fig)
    plt.close(fig)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10. Reproducible interpretation gate

# COMMAND ----------
summary = {
    "analysis": "CLSA_RETFound_glaucoma_classifier_spatial_validation",
    "glaucoma_label": (
        "released physician-diagnosed self-report with negative same-visit "
        "screen for other prespecified ocular disease"
    ),
    "classifier_role": "weak-label research classifier; not a clinical device",
    "n_classifier_participants": int(len(participant_predictions)),
    "n_glaucoma_participants": int(
        participant_predictions["glaucoma_label"].sum()
    ),
    "n_healthy_participants": int(
        (participant_predictions["glaucoma_label"] == 0).sum()
    ),
    "n_external_zeiss_glaucoma_participants": int(
        len(zeiss_participant_predictions)
    ),
    "mean_external_zeiss_glaucoma_probability": float(
        zeiss_participant_predictions[
            "glaucoma_probability_external"
        ].mean()
    ),
    "control_ratio_requested": control_ratio,
    "age_caliper_years": age_caliper_years,
    "participant_oof_performance": performance,
    "demographic_baseline_performance": demographic_performance,
    "quality_baseline_performance": quality_performance,
    "shortcut_baseline_tests": shortcut_tests,
    "shortcut_baselines_outperformed": shortcut_baselines_outperformed,
    "retfound_embeddings_recalculated_for_training": False,
    "source_specific_explainability_preprocessing": {
        "CLSA": "CLSA fundus crop/resize plus ImageNet normalization",
        "Zeiss": "established Zeiss DICOM/AutoMorph crop plus ImageNet normalization",
    },
    "explainability_replay_contract": {
        "minimum_embedding_cosine": embedding_cosine_threshold,
        "maximum_clsa_logit_absolute_difference": (
            clsa_logit_replay_tolerance
        ),
        "maximum_zeiss_probability_absolute_difference": (
            zeiss_probability_replay_tolerance
        ),
        "maximum_images_per_participant": (
            max_explain_images_per_participant
        ),
        "note": (
            "Zeiss uses probability-scale replay because tiny source-encoder "
            "floating-point differences can be amplified on the logit scale; "
            "exact patch additivity and embedding cosine remain mandatory."
        ),
    },
    "explainability_ran": run_explainability_flag,
    "n_explained_images": int(len(attribution_manifest)),
    "validated_disc_annotation_coverage": validated_annotation_coverage,
    "clsa_glaucoma_validated_disc_annotation_coverage": (
        clsa_glaucoma_validated_coverage
    ),
    "anatomic_claim_ready": anatomic_claim_ready,
    "anatomic_claim_requirements": [
        "participant-held-out classifier AUROC lower confidence bound above 0.5",
        "paired-bootstrap RETFound AUROC advantage over demographic and available image-quality baselines",
        "at least 80% validated disc coordinates when required",
        "participant-level optic-nerve occlusion drop greater than equal-area random occlusion",
        "participant-level optic-nerve positive-attribution enrichment lower confidence bound above 1",
        "CLSA glaucoma-minus-healthy spatial enrichment and occlusion-difference lower confidence bounds above 0",
    ],
    "interpretation": (
        "Exact patch maps describe contributions to the held-out linear glaucoma "
        "logit. Anatomic dependence requires agreement with targeted occlusion; "
        "automatic bright-disc proxies remain exploratory."
    ),
    "limitations": [
        "CLSA glaucoma is self-reported rather than clinically adjudicated.",
        "CLSA does not supply glaucoma subtype, visual-field MD, RNFL, or severity.",
        "Attribution localization alone is not causal evidence; occlusion is required.",
        "Zeiss requires same-device controls for complete external classifier validation.",
        "Zeiss positive-cohort scores and maps test transport/localization only; they do not estimate specificity.",
        "Lens status and pseudophakia should be added when available.",
    ],
    "outputs": {
        "participant_predictions": str(
            participant_root / "CLSA_glaucoma_participant_oof_predictions.parquet"
        ),
        "fold_heads": str(
            participant_root / "CLSA_glaucoma_oof_fold_heads.joblib"
        ),
        "image_predictions": str(
            image_root / "CLSA_glaucoma_image_oof_predictions.parquet"
        ),
        "zeiss_external_predictions": str(
            participant_root
            / "Zeiss_glaucoma_external_predictions_private.parquet"
        ),
        "attribution_manifest": str(
            attribution_root / "glaucoma_attribution_manifest_private.parquet"
        ),
        "annotation_template": str(
            anatomy_root / "optic_disc_annotation_template_private.csv"
        ),
    },
}
write_json(summary, output_root / "GLAUCOMA_SPATIAL_VALIDATION_SUMMARY.json")
print(json.dumps(summary, indent=2, default=str))
print("Notebook 08 complete")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 11. Remove temporary credentials

# COMMAND ----------
os.environ.pop("HF_TOKEN", None)
try:
    dbutils.widgets.remove("hf_token")
except Exception:
    pass
print("Temporary Hugging Face token widget removed")
