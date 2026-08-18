# Databricks notebook source
# MAGIC %md
# MAGIC # Train `CLSA_healthy`, apply it to Zeiss glaucoma, and compare matched cohorts
# MAGIC
# MAGIC This notebook implements the primary retinal-age design:
# MAGIC
# MAGIC 1. Train the existing RETFound Ridge age head using every eligible,
# MAGIC    screen-negative CLSA image with participant-grouped cross-validation.
# MAGIC 2. Freeze and save the deployable model as `CLSA_healthy.joblib`.
# MAGIC 3. Apply that frozen head to the already-computed Zeiss RETFound vectors.
# MAGIC 4. Aggregate predictions at participant level, plot retinal-age gap, and
# MAGIC    construct a participant-level age-only matched comparison.
# MAGIC 5. Save matched image/vector inputs for the separate GPU explainability notebook.
# MAGIC
# MAGIC CLSA comparison predictions use grouped out-of-fold predictions, not
# MAGIC optimistic in-sample predictions. The final frozen model is used only for
# MAGIC external Zeiss inference. All eyes and visits from one CLSA participant
# MAGIC remain in the same fold.

# COMMAND ----------
from pathlib import Path
import hashlib
import importlib
import inspect
import json
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

# COMMAND ----------
# Reproducible configuration is fixed in the next cell.

# COMMAND ----------
from pathlib import Path

repo_root = Path(
    "/Workspace/Users/ad0038@pennmedicine.upenn.edu/CLSA/CLSA_retina"
)
output_root = Path(
    "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/derived/"
    "clsa_retinal_aging/Age_Glaucoma"
)
force_retrain = False
ridge_alpha = 10.0
cv_folds = 5
support_min_age = 45.0
support_max_age = 85.0
age_caliper_years = 1.0
match_ratio = 1
bootstrap_repetitions = 5000

if ridge_alpha < 0:
    raise ValueError("ridge_alpha cannot be negative")
if cv_folds < 2:
    raise ValueError("cv_folds must be at least 2")
if support_max_age <= support_min_age:
    raise ValueError("common_support_max_age must exceed common_support_min_age")
if age_caliper_years < 0:
    raise ValueError("age_caliper_years cannot be negative")
if match_ratio < 1:
    raise ValueError("match_ratio must be at least 1")
if bootstrap_repetitions < 100:
    raise ValueError("bootstrap_repetitions must be at least 100")

module_root = repo_root / "src"
if not module_root.exists():
    raise FileNotFoundError(f"Repository source directory not found: {module_root}")
if str(module_root) not in sys.path:
    sys.path.insert(0, str(module_root))

import fundus_retfound_pipeline as _fundus_pipeline  # noqa: E402

# Databricks keeps imported modules alive across notebook reruns. Force the
# repository version to reload so a prior in-memory train_age_head cannot retain
# the metadata-serialization bug after Git pull.
_fundus_pipeline = importlib.reload(_fundus_pipeline)
if "write_metadata" not in inspect.signature(
    _fundus_pipeline.train_age_head
).parameters:
    raise RuntimeError(
        "The loaded fundus_retfound_pipeline is stale and lacks the "
        "write_metadata safety switch. Confirm repo_root, pull Git, and restart Python. "
        f"Loaded module: {_fundus_pipeline.__file__}"
    )
print("Reloaded fundus pipeline:", _fundus_pipeline.__file__)
print("Age-head metadata writing is disabled inside training for this workflow.")

from age_glaucoma_cohort import greedy_age_match  # noqa: E402
from age_glaucoma_model import (  # noqa: E402
    aggregate_clsa_oof_predictions,
    aggregate_zeiss_predictions,
    bootstrap_mean_ci,
    embedding_dataset_signature,
    prediction_summary,
    standardized_mean_difference,
    validate_embedding_frame,
)
from fundus_retfound_pipeline import (  # noqa: E402
    AgeModelConfig,
    load_age_head,
    predict_retinal_age,
    train_age_head,
    write_frame,
)

# COMMAND ----------
clsa_eligible_path = output_root / "03_clsa_controls" / "eligible_images_delta"
zeiss_embeddings_path = (
    output_root / "01_zeiss_source_cohort" / "zeiss_embedded_images.parquet"
)
model_root = output_root / "06_CLSA_healthy_model"
inference_root = output_root / "07_retinal_age_inference"
comparison_root = output_root / "08_matched_comparison"
figure_root = inference_root / "figures"
for path in (model_root, inference_root, comparison_root, figure_root):
    path.mkdir(parents=True, exist_ok=True)

if not clsa_eligible_path.exists():
    raise FileNotFoundError(f"CLSA eligible image table is missing: {clsa_eligible_path}")
if not zeiss_embeddings_path.exists():
    raise FileNotFoundError(f"Zeiss embedding table is missing: {zeiss_embeddings_path}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Assemble and validate every eligible screen-negative CLSA embedding

# COMMAND ----------
clsa_spark = spark.read.format("delta").load(str(clsa_eligible_path))
required_clsa_columns = {
    "image_path",
    "participant_id",
    "visit",
    "embedding",
    "age_at_fundus_years",
}
missing_clsa_columns = required_clsa_columns - set(clsa_spark.columns)
if missing_clsa_columns:
    raise ValueError(
        f"CLSA eligible images lack columns: {sorted(missing_clsa_columns)}"
    )

clsa_selections = [
    F.col("image_path").cast("string").alias("image_path"),
    F.col("participant_id").cast("string").alias("participant_id"),
    F.upper(F.col("visit")).alias("visit"),
    F.col("age_at_fundus_years").cast("double").alias("age"),
    F.col("embedding").cast("array<float>").alias("embedding"),
]
if "eye" in clsa_spark.columns:
    clsa_selections.append(F.col("eye").cast("string").alias("eye"))
elif "eye_parsed" in clsa_spark.columns:
    clsa_selections.append(F.col("eye_parsed").cast("string").alias("eye"))
else:
    clsa_selections.append(F.lit(None).cast("string").alias("eye"))
if "sex_at_birth" in clsa_spark.columns:
    clsa_selections.append(F.col("sex_at_birth").cast("string").alias("sex"))
else:
    clsa_selections.append(F.lit(None).cast("string").alias("sex"))

clsa_training = clsa_spark.select(*clsa_selections).toPandas()
clsa_training = validate_embedding_frame(
    clsa_training,
    "Screen-negative CLSA training embeddings",
    expected_dim=1024,
)
# Databricks may attach Spark Connect PlanMetrics to this Pandas DataFrame.
# These runtime-only attrs are not variables and cannot be written by PyArrow.
clsa_training.attrs = {}
training_signature = embedding_dataset_signature(clsa_training)
print(
    f"CLSA healthy training input: {len(clsa_training):,} images from "
    f"{clsa_training['participant_id'].nunique():,} participants; "
    f"signature={training_signature[:16]}..."
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Grouped-CV training and frozen `CLSA_healthy` artifact
# MAGIC
# MAGIC `train_age_head` uses `GroupKFold(participant_id)`. The deployable Ridge
# MAGIC estimator is fitted on all eligible CLSA images only after the grouped
# MAGIC out-of-fold predictions and calibration have been generated.

# COMMAND ----------
frozen_model_path = model_root / "CLSA_healthy.joblib"
frozen_metadata_path = model_root / "CLSA_healthy_metadata.json"
oof_prediction_path = model_root / "CLSA_healthy_oof_predictions.parquet"
partial_model_path = (
    model_root / "training_artifacts" / "retfound_age_head.joblib"
)
partial_oof_path = (
    model_root / "training_artifacts" / "retfound_age_predictions_oof.parquet"
)

can_resume = (
    not force_retrain
    and frozen_model_path.exists()
    and frozen_metadata_path.exists()
    and oof_prediction_path.exists()
)
if can_resume:
    frozen_metadata = json.loads(frozen_metadata_path.read_text(encoding="utf-8"))
    if frozen_metadata.get("training_dataset_signature") != training_signature:
        raise RuntimeError(
            "The saved CLSA_healthy model was trained on a different eligible "
            "dataset signature. Set force_retrain=true to intentionally replace it."
        )
    age_bundle = load_age_head(frozen_model_path)
    clsa_oof = pd.read_parquet(oof_prediction_path)
    print("Resumed frozen model:", frozen_model_path)
else:
    recovered_partial_training = (
        not force_retrain and partial_model_path.exists() and partial_oof_path.exists()
    )
    if recovered_partial_training:
        age_bundle = load_age_head(partial_model_path)
        clsa_oof = pd.read_parquet(partial_oof_path)
        required_oof_columns = {
            "image_path",
            "participant_id",
            "retinal_age_prediction_oof",
            "retinal_age_gap_oof",
        }
        missing_oof_columns = required_oof_columns - set(clsa_oof.columns)
        if missing_oof_columns:
            raise RuntimeError(
                "Partial grouped-CV output is incomplete: "
                f"{sorted(missing_oof_columns)}. Set force_retrain=true."
            )
        if set(clsa_oof["image_path"].astype(str)) != set(
            clsa_training["image_path"].astype(str)
        ):
            raise RuntimeError(
                "Partial grouped-CV output does not match the current training "
                "images. Set force_retrain=true."
            )
        if int(age_bundle["embedding_dim"]) != 1024:
            raise RuntimeError("Recovered partial model is not a 1,024-feature head")
        print(
            "Recovered the completed grouped-CV model and OOF predictions from "
            "the prior metadata-serialization failure; Ridge fitting was not repeated."
        )
    else:
        clsa_oof, age_bundle = train_age_head(
            clsa_training,
            model_root / "training_artifacts",
            AgeModelConfig(
                alpha=ridge_alpha,
                max_splits=cv_folds,
                calibration="intercept",
                random_state=20260807,
            ),
            write_metadata=False,
        )
    age_bundle.update(
        {
            "model_name": "CLSA_healthy",
            "frozen": True,
            "training_cohort": "CLSA_screen_negative_ocular_controls",
            "training_dataset_signature": training_signature,
            "grouping_unit": "participant_id",
            "comparison_prediction_mode": "grouped_out_of_fold",
        }
    )
    joblib.dump(age_bundle, frozen_model_path)
    write_frame(clsa_oof, oof_prediction_path)
    model_sha256 = hashlib.sha256(frozen_model_path.read_bytes()).hexdigest()
    frozen_metadata = {
        "model_name": "CLSA_healthy",
        "frozen": True,
        "model_path": str(frozen_model_path),
        "model_sha256": model_sha256,
        "training_dataset_signature": training_signature,
        "n_training_images": int(len(clsa_training)),
        "n_training_participants": int(clsa_training["participant_id"].nunique()),
        "embedding_dim": int(age_bundle["embedding_dim"]),
        "ridge_alpha": ridge_alpha,
        "grouped_cv_folds": int(min(cv_folds, clsa_training["participant_id"].nunique())),
        "calibration": age_bundle["calibration"],
        "training_cohort": "CLSA_screen_negative_ocular_controls",
    }
    frozen_metadata_path.write_text(
        json.dumps(frozen_metadata, indent=2), encoding="utf-8"
    )
    print("Trained and froze:", frozen_model_path)

if set(clsa_oof["participant_id"].astype(str)) != set(
    clsa_training["participant_id"].astype(str)
):
    raise RuntimeError("OOF predictions do not cover every training participant")

clsa_participant_visit = aggregate_clsa_oof_predictions(clsa_oof)
write_frame(
    clsa_participant_visit,
    inference_root / "CLSA_healthy_participant_visit_oof.parquet",
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Frozen-model inference on existing Zeiss embeddings

# COMMAND ----------
zeiss_embeddings = pd.read_parquet(zeiss_embeddings_path)
zeiss_embeddings = zeiss_embeddings.rename(
    columns={"patient_id": "participant_id", "dcm_path": "image_path"}
)
zeiss_embeddings = validate_embedding_frame(
    zeiss_embeddings,
    "Zeiss glaucoma-source embeddings",
    expected_dim=1024,
)
zeiss_predictions = predict_retinal_age(
    zeiss_embeddings,
    age_bundle,
    inference_root / "Zeiss_CLSA_healthy_predictions.parquet",
)
zeiss_participants = aggregate_zeiss_predictions(zeiss_predictions)
write_frame(
    zeiss_participants,
    inference_root / "Zeiss_participant_retinal_age.parquet",
)
print(
    f"Frozen-model Zeiss inference: {len(zeiss_predictions):,} images from "
    f"{len(zeiss_participants):,} patients"
)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Embedding-domain-shift audit
# MAGIC
# MAGIC Age matching cannot remove camera/preprocessing shift. This diagnostic
# MAGIC compares the 1,024 feature distributions before interpreting age gaps.

# COMMAND ----------
clsa_matrix = np.stack(clsa_training["embedding"].to_numpy()).astype(np.float32)
zeiss_matrix = np.stack(zeiss_embeddings["embedding"].to_numpy()).astype(np.float32)
clsa_mean = clsa_matrix.mean(axis=0)
zeiss_mean = zeiss_matrix.mean(axis=0)
clsa_variance = clsa_matrix.var(axis=0, ddof=1)
zeiss_variance = zeiss_matrix.var(axis=0, ddof=1)
pooled_sd = np.sqrt((clsa_variance + zeiss_variance) / 2)
feature_smd = np.divide(
    zeiss_mean - clsa_mean,
    pooled_sd,
    out=np.zeros_like(zeiss_mean),
    where=pooled_sd > 1e-8,
)
centroid_denominator = float(np.linalg.norm(clsa_mean) * np.linalg.norm(zeiss_mean))
centroid_cosine = (
    float(clsa_mean @ zeiss_mean / centroid_denominator)
    if centroid_denominator
    else np.nan
)
domain_shift_summary = pd.DataFrame(
    [
        {
            "n_features": int(len(feature_smd)),
            "centroid_cosine_similarity": centroid_cosine,
            "median_absolute_feature_smd": float(np.median(np.abs(feature_smd))),
            "p90_absolute_feature_smd": float(np.quantile(np.abs(feature_smd), 0.90)),
            "features_absolute_smd_gt_0_1": int(np.sum(np.abs(feature_smd) > 0.1)),
            "features_absolute_smd_gt_0_2": int(np.sum(np.abs(feature_smd) > 0.2)),
            "features_absolute_smd_gt_0_5": int(np.sum(np.abs(feature_smd) > 0.5)),
            "zeiss_predictions_below_zero": int(
                (zeiss_predictions["retinal_age_prediction"] < 0).sum()
            ),
            "zeiss_predictions_above_120": int(
                (zeiss_predictions["retinal_age_prediction"] > 120).sum()
            ),
        }
    ]
)
write_frame(
    domain_shift_summary,
    inference_root / "embedding_domain_shift_summary.csv",
)
display(domain_shift_summary.round(4))
del clsa_matrix, zeiss_matrix

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Full-cohort retinal-age-gap summaries and figures

# COMMAND ----------
full_summary = pd.DataFrame(
    [
        prediction_summary(clsa_participant_visit, "CLSA healthy (OOF)"),
        prediction_summary(zeiss_participants, "Zeiss glaucoma (frozen model)"),
    ]
)
write_frame(full_summary, inference_root / "full_cohort_summary.csv")
display(full_summary.round(3))

fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
for frame, label, color in (
    (clsa_participant_visit, "CLSA healthy (OOF)", "#2563eb"),
    (zeiss_participants, "Zeiss glaucoma", "#dc2626"),
):
    axes[0].hist(
        frame["retinal_age_gap"],
        bins=50,
        density=True,
        histtype="step",
        linewidth=2,
        label=label,
        color=color,
    )
    axes[1].scatter(
        frame["age"],
        frame["retinal_age_prediction"],
        s=6,
        alpha=0.18,
        label=label,
        color=color,
    )
    axes[2].scatter(
        frame["age"],
        frame["retinal_age_gap"],
        s=6,
        alpha=0.18,
        label=label,
        color=color,
    )
axes[0].axvline(0, color="black", linewidth=1)
axes[0].set(title="Retinal-age gap", xlabel="Predicted − chronological age (years)", ylabel="Density")
axes[1].plot([support_min_age, support_max_age], [support_min_age, support_max_age], "k--", linewidth=1)
axes[1].set(title="Predicted versus chronological age", xlabel="Chronological age", ylabel="Predicted retinal age")
axes[2].axhline(0, color="black", linewidth=1)
axes[2].set(title="Age gap over chronological age", xlabel="Chronological age", ylabel="Retinal-age gap")
for axis in axes:
    axis.legend(frameon=False)
fig.savefig(figure_root / "full_cohort_retinal_age_gap.png", dpi=220, bbox_inches="tight")
plt.show()
plt.close(fig)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Participant-level age-only matching on common support
# MAGIC
# MAGIC Matching is deliberately age-only: no raw sex-code equality is applied.
# MAGIC CLSA controls are participant-visits so the visit age remains aligned to
# MAGIC its images, but a CLSA participant can be used only once across BL/F1.

# COMMAND ----------
zeiss_match_pool = zeiss_participants[
    zeiss_participants["age"].between(support_min_age, support_max_age, inclusive="both")
].copy()
clsa_match_pool = clsa_participant_visit[
    clsa_participant_visit["age"].between(support_min_age, support_max_age, inclusive="both")
].copy()
clsa_match_pool = clsa_match_pool.rename(columns={"age": "age_at_fundus_years"})

match_pairs, match_audit = greedy_age_match(
    cases=zeiss_match_pool,
    controls=clsa_match_pool,
    ratio=match_ratio,
    caliper_years=age_caliper_years,
    exact_sex=False,
    case_id="participant_id",
    case_age="age",
    control_id="participant_id",
    control_age="age_at_fundus_years",
)
write_frame(match_audit, comparison_root / "age_match_audit.parquet")
if match_pairs.empty:
    raise RuntimeError(
        "Age-only matching unexpectedly produced zero pairs despite common-support "
        "filtering. Review age_match_audit.parquet and confirm the configuration "
        "cell was rerun."
    )
write_frame(match_pairs, comparison_root / "age_match_pairs.parquet")
print(
    f"Matched {match_pairs['zeiss_patient_id'].nunique():,}/"
    f"{len(zeiss_match_pool):,} common-support Zeiss patients to "
    f"{match_pairs['clsa_participant_id'].nunique():,} CLSA participants "
    f"at {match_ratio}:1 within ±{age_caliper_years:g} years."
)

# COMMAND ----------
zeiss_matched = match_pairs.merge(
    zeiss_participants,
    left_on="zeiss_patient_id",
    right_on="participant_id",
    how="left",
    validate="many_to_one",
).rename(
    columns={
        "retinal_age_prediction": "zeiss_retinal_age_prediction",
        "retinal_age_gap": "zeiss_retinal_age_gap",
    }
)
clsa_matched = match_pairs.merge(
    clsa_participant_visit,
    left_on=["clsa_participant_id", "clsa_visit"],
    right_on=["participant_id", "visit"],
    how="left",
    validate="many_to_one",
).rename(
    columns={
        "retinal_age_prediction": "clsa_retinal_age_prediction",
        "retinal_age_gap": "clsa_retinal_age_gap",
    }
)
matched_analysis = zeiss_matched[
    [
        "match_set_id",
        "zeiss_patient_id",
        "zeiss_age_years",
        "zeiss_retinal_age_prediction",
        "zeiss_retinal_age_gap",
        "clsa_participant_id",
        "clsa_visit",
        "clsa_age_years",
        "age_difference_years",
        "absolute_age_difference_years",
    ]
].merge(
    clsa_matched[
        [
            "match_set_id",
            "clsa_participant_id",
            "clsa_visit",
            "clsa_retinal_age_prediction",
            "clsa_retinal_age_gap",
        ]
    ],
    on=["match_set_id", "clsa_participant_id", "clsa_visit"],
    how="left",
    validate="one_to_one",
)
write_frame(matched_analysis, comparison_root / "matched_participant_analysis.parquet")

pair_level = (
    matched_analysis.groupby("match_set_id", as_index=False)
    .agg(
        zeiss_age=("zeiss_age_years", "first"),
        clsa_age=("clsa_age_years", "mean"),
        zeiss_age_gap=("zeiss_retinal_age_gap", "first"),
        clsa_age_gap=("clsa_retinal_age_gap", "mean"),
        mean_absolute_age_difference=("absolute_age_difference_years", "mean"),
    )
)
pair_level["paired_age_gap_difference"] = (
    pair_level["zeiss_age_gap"] - pair_level["clsa_age_gap"]
)
write_frame(pair_level, comparison_root / "matched_pair_level_analysis.parquet")

mean_difference, ci_low, ci_high = bootstrap_mean_ci(
    pair_level["paired_age_gap_difference"],
    n_bootstrap=bootstrap_repetitions,
    random_state=20260807,
)
balance_summary = pd.DataFrame(
    [
        {
            "n_match_sets": int(len(pair_level)),
            "match_ratio": match_ratio,
            "common_support_min_age": support_min_age,
            "common_support_max_age": support_max_age,
            "caliper_years": age_caliper_years,
            "mean_absolute_age_difference": float(
                matched_analysis["absolute_age_difference_years"].mean()
            ),
            "age_smd_before_matching": standardized_mean_difference(
                zeiss_match_pool["age"], clsa_match_pool["age_at_fundus_years"]
            ),
            "age_smd_after_matching": standardized_mean_difference(
                pair_level["zeiss_age"], pair_level["clsa_age"]
            ),
            "mean_paired_age_gap_difference_zeiss_minus_clsa": mean_difference,
            "bootstrap_95_ci_low": ci_low,
            "bootstrap_95_ci_high": ci_high,
        }
    ]
)
write_frame(balance_summary, comparison_root / "matched_comparison_summary.csv")
display(balance_summary.round(4))

# COMMAND ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
axes[0].hist(pair_level["clsa_age_gap"], bins=40, alpha=0.55, label="CLSA healthy", color="#2563eb")
axes[0].hist(pair_level["zeiss_age_gap"], bins=40, alpha=0.55, label="Zeiss glaucoma", color="#dc2626")
axes[0].axvline(0, color="black", linewidth=1)
axes[0].set(title="Age-matched retinal-age gap", xlabel="Retinal-age gap (years)", ylabel="Match sets")
axes[0].legend(frameon=False)
axes[1].hist(pair_level["paired_age_gap_difference"], bins=40, color="#7c3aed", alpha=0.8)
axes[1].axvline(0, color="black", linewidth=1)
axes[1].axvline(mean_difference, color="#dc2626", linewidth=2, label=f"Mean={mean_difference:.2f}")
axes[1].set(title="Within-match difference", xlabel="Zeiss gap − CLSA gap (years)", ylabel="Match sets")
axes[1].legend(frameon=False)
fig.savefig(comparison_root / "matched_retinal_age_gap.png", dpi=220, bbox_inches="tight")
plt.show()
plt.close(fig)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Save matched image/vector inputs for source-aware explainability

# COMMAND ----------
zeiss_prediction_columns = zeiss_predictions[
    ["image_path", "retinal_age_prediction", "retinal_age_gap", "absolute_error"]
]
zeiss_explain_images = zeiss_embeddings.merge(
    zeiss_prediction_columns,
    on="image_path",
    how="inner",
    validate="one_to_one",
).merge(
    match_pairs[["match_set_id", "zeiss_patient_id"]].drop_duplicates(),
    left_on="participant_id",
    right_on="zeiss_patient_id",
    how="inner",
    validate="many_to_one",
)
clsa_prediction_columns = clsa_oof[
    [
        "image_path",
        "retinal_age_prediction_oof",
        "retinal_age_gap_oof",
        "absolute_error_oof",
    ]
]
clsa_explain_images = clsa_training.merge(
    clsa_prediction_columns,
    on="image_path",
    how="inner",
    validate="one_to_one",
).merge(
    match_pairs[["match_set_id", "clsa_participant_id", "clsa_visit"]],
    left_on=["participant_id", "visit"],
    right_on=["clsa_participant_id", "clsa_visit"],
    how="inner",
    validate="many_to_one",
)
write_frame(
    zeiss_explain_images,
    comparison_root / "zeiss_matched_images_for_explainability.parquet",
)
write_frame(
    clsa_explain_images,
    comparison_root / "clsa_matched_images_for_explainability.parquet",
)

run_summary = {
    "model_name": "CLSA_healthy",
    "model_path": str(frozen_model_path),
    "training_dataset_signature": training_signature,
    "n_clsa_training_images": int(len(clsa_training)),
    "n_clsa_training_participants": int(clsa_training["participant_id"].nunique()),
    "n_zeiss_inference_images": int(len(zeiss_predictions)),
    "n_zeiss_inference_participants": int(len(zeiss_participants)),
    "embedding_centroid_cosine_similarity": centroid_cosine,
    "median_absolute_embedding_feature_smd": float(
        domain_shift_summary.iloc[0]["median_absolute_feature_smd"]
    ),
    "n_matched_zeiss_participants": int(match_pairs["zeiss_patient_id"].nunique()),
    "n_matched_clsa_participants": int(match_pairs["clsa_participant_id"].nunique()),
    "age_caliper_years": age_caliper_years,
    "match_ratio": match_ratio,
    "common_support": [support_min_age, support_max_age],
    "age_smd_after_matching": float(balance_summary.iloc[0]["age_smd_after_matching"]),
    "paired_gap_difference_mean": mean_difference,
    "paired_gap_difference_bootstrap_95_ci": [ci_low, ci_high],
    "zeiss_diagnosis_basis": "user_asserted_glaucoma_source_cohort",
    "important_limitation": (
        "Disease status remains confounded with image source/acquisition system. "
        "Matched comparisons are exploratory until source-crossed controls exist."
    ),
}
(output_root / "CLSA_HEALTHY_AGE_GLAUCOMA_SUMMARY.json").write_text(
    json.dumps(run_summary, indent=2), encoding="utf-8"
)
print(json.dumps(run_summary, indent=2))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Next step
# MAGIC
# MAGIC Run `04_compare_matched_explainability.py` on a GPU cluster. It loads the
# MAGIC frozen `CLSA_healthy` model and these matched image/vector Parquets; it
# MAGIC does not retrain the age head.
