# Databricks notebook source
# MAGIC %md
# MAGIC # Symmetric cross-fitted calibration of retinal, Horvath, and Hannum age
# MAGIC
# MAGIC This standalone analysis addresses the like-for-like calibration concern in the
# MAGIC manuscript review. It uses one complete participant cohort, one common fold
# MAGIC assignment, and the same five-knot spline-plus-ridge calibration pipeline for all
# MAGIC three age estimates. Every reported calibrated prediction is held out from the
# MAGIC calibrator that produced it.
# MAGIC
# MAGIC The notebook reports raw and symmetrically calibrated metrics, participant-bootstrap
# MAGIC confidence intervals, paired clock comparisons, five-year age-bin performance,
# MAGIC available subgroup/site diagnostics, a cohort-flow table, two publication figures,
# MAGIC and a manuscript-ready results snippet. It does not export participant identifiers
# MAGIC outside the governed private output directory.
# MAGIC
# MAGIC Cross-validation and statistical-reporting implementation was audited with procedural
# MAGIC guidance from Kassis T, Agarwal V, He Y, Patel D, and Brueckner AM. *Scientific Agent
# MAGIC Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065 (2026).
# MAGIC https://doi.org/10.48550/arXiv.2609.00065

# COMMAND ----------

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT_DEFAULT = Path(
    "/Workspace/Users/ad0038@pennmedicine.upenn.edu/CLSA/CLSA_retina"
)
EPIGENETICS_ROOT_DEFAULT = Path(
    "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/derived/"
    "clsa_retinal_aging/Age_Glaucoma/16_algorithm_fairness/08_epigenetics"
)
INPUT_DEFAULT = (
    EPIGENETICS_ROOT_DEFAULT
    / "01_private"
    / "baseline_three_age_master_private.parquet"
)
RETINAL_OOF_DEFAULT = (
    EPIGENETICS_ROOT_DEFAULT
    / "02_models"
    / "retfound_chronological_age"
    / "retfound_chronological_age_oof.parquet"
)
OUTPUT_DEFAULT = EPIGENETICS_ROOT_DEFAULT / "06_symmetric_calibration"


def widget(name: str, default: str, label: str) -> str:
    try:
        dbutils.widgets.text(name, default, label)  # noqa: F821
        return dbutils.widgets.get(name).strip()  # noqa: F821
    except NameError:
        return default


repo_root = Path(widget("repo_root", str(REPO_ROOT_DEFAULT), "Repository root"))
input_path = Path(widget("input_path", str(INPUT_DEFAULT), "Three-clock input parquet"))
retinal_oof_path = Path(
    widget("retinal_oof_path", str(RETINAL_OOF_DEFAULT), "Retinal OOF parquet")
)
output_root = Path(widget("output_root", str(OUTPUT_DEFAULT), "Output root"))
bootstrap_repetitions = int(widget("bootstrap_repetitions", "2000", "Bootstrap repetitions"))
paired_repetitions = int(widget("paired_repetitions", "10000", "Paired sign-flip/bootstrap repetitions"))
random_seed = int(widget("random_seed", "20261118", "Random seed"))
minimum_subgroup_n = int(widget("minimum_subgroup_n", "30", "Minimum subgroup n"))

if not input_path.is_file():
    fallback = (
        EPIGENETICS_ROOT_DEFAULT
        / "01_private"
        / "patient_three_clock_phenotypes_private.parquet"
    )
    if fallback.is_file():
        input_path = fallback
    else:
        raise FileNotFoundError(
            f"Three-clock input not found: {input_path}. Also tried {fallback}. "
            "Run the main epigenetics notebook through the three-clock cohort cell."
        )

source_root = repo_root / "src"
if not source_root.is_dir():
    raise FileNotFoundError(f"Repository src directory not found: {source_root}")
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

import three_clock_calibration as calibration  # noqa: E402

calibration = importlib.reload(calibration)

config = calibration.CalibrationConfig(
    n_splits=5,
    n_knots=5,
    degree=3,
    ridge_alpha=1.0,
    seed=random_seed,
)

private_root = output_root / "01_private"
statistics_root = output_root / "02_statistics"
figure_root = output_root / "03_figures"
for directory in (private_root, statistics_root, figure_root):
    directory.mkdir(parents=True, exist_ok=True)

print("Input:", input_path)
print("Output:", output_root)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Assemble the common three-clock cohort
# MAGIC
# MAGIC The preferred retinal input is the upstream raw participant-held-out retinal-age
# MAGIC prediction. If an older three-clock parquet lacks that column, this cell merges it
# MAGIC from the chronological-head OOF artifact. Only if neither contains the raw estimate
# MAGIC does the analysis use the already calibrated retinal OOF estimate, and that fallback
# MAGIC is recorded prominently in the flow table and manifest.

# COMMAND ----------


def normalize_identifier(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    return values.str.replace(r"\.0$", "", regex=True)


source = pd.read_parquet(input_path)
if "participant_id" not in source.columns:
    raise KeyError("Input parquet does not contain participant_id")
source["participant_id"] = normalize_identifier(source["participant_id"])

raw_retinal_merge_status = "already_present"
if "retinal_age_raw_oof" not in source.columns:
    if retinal_oof_path.is_file():
        retinal_oof = pd.read_parquet(retinal_oof_path)
        if "participant_id" not in retinal_oof.columns:
            raise KeyError(f"Retinal OOF file lacks participant_id: {retinal_oof_path}")
        retinal_oof["participant_id"] = normalize_identifier(retinal_oof["participant_id"])
        if "visit" in retinal_oof.columns:
            visit = retinal_oof["visit"].astype("string").str.upper().str.strip()
            retinal_oof = retinal_oof.loc[visit.eq("BL")].copy()
        raw_candidates = [
            column
            for column in ("retinal_age_raw_oof", "retinal_age_prediction_raw_oof")
            if column in retinal_oof.columns
        ]
        if raw_candidates:
            raw_column = raw_candidates[0]
            raw_retinal = retinal_oof[["participant_id", raw_column]].rename(
                columns={raw_column: "retinal_age_raw_oof"}
            )
            if raw_retinal["participant_id"].duplicated().any():
                raise ValueError("Baseline raw retinal OOF predictions are not participant-unique")
            source = source.merge(
                raw_retinal,
                on="participant_id",
                how="left",
                validate="one_to_one",
            )
            raw_retinal_merge_status = f"merged_from:{retinal_oof_path}"
        else:
            raw_retinal_merge_status = "raw_column_absent_in_auxiliary_oof"
    else:
        raw_retinal_merge_status = "auxiliary_oof_not_found"

analysis, cohort_flow = calibration.prepare_three_clock_frame(
    source,
    participant_column="participant_id",
    age_column="chronological_age",
    fold_column="cv_fold",
    config=config,
)
cohort_flow["raw_retinal_merge_status"] = raw_retinal_merge_status
cohort_flow.to_csv(statistics_root / "three_clock_calibration_cohort_flow.csv", index=False)

retinal_is_raw = bool(cohort_flow["retinal_input_is_raw_oof"].iloc[0])
if not retinal_is_raw:
    print(
        "WARNING: raw retinal OOF predictions were unavailable. The same held-out "
        "calibration is still applied to all clocks, but the retinal input already "
        "contains the upstream retinal-head calibration. Rerun the main epigenetics "
        "notebook after updating it to persist retinal_age_raw_oof for the strictest comparison."
    )

print(cohort_flow.to_string(index=False))
print("Common complete cohort:", len(analysis))
print("Fold counts:")
print(analysis["calibration_fold"].value_counts().sort_index().to_string())

# COMMAND ----------
# MAGIC %md
# MAGIC ## Cross-fit the identical calibrator for all three clocks

# COMMAND ----------

raw_columns = dict(calibration.DEFAULT_CLOCK_COLUMNS)
calibrated_columns = {}
fold_audits = []
for clock, input_column in raw_columns.items():
    output_column = f"{clock.lower()}_age_calibrated_cf"
    predictions, audit = calibration.cross_fitted_calibrate(
        analysis,
        input_column,
        target_column="chronological_age",
        fold_column="calibration_fold",
        config=config,
    )
    analysis[output_column] = predictions
    audit["clock"] = clock
    audit["output_column"] = output_column
    fold_audits.append(audit)
    calibrated_columns[clock] = output_column

fold_audit = pd.concat(fold_audits, ignore_index=True)
if fold_audit.apply(
    lambda row: str(row["held_out_fold"]) in str(row["training_folds"]).split("|"),
    axis=1,
).any():
    raise RuntimeError("Leakage audit failed: a held-out fold appears in its training folds")
fold_audit.to_csv(statistics_root / "three_clock_calibration_fold_audit.csv", index=False)

private_columns = [
    "participant_id",
    "chronological_age",
    "calibration_fold",
    *raw_columns.values(),
    *calibrated_columns.values(),
]
private_columns.extend(
    column
    for column in analysis.columns
    if column not in private_columns
    and column
    in {
        "sex_labeled",
        "sex",
        "racial_background",
        "site",
        "collection_site",
        "geographic_stratum",
        "cataract_any",
        "ICQ_CATRCT_COM",
        "VIS_CATRCT_COM",
        "n_embedded_images",
    }
)
analysis[private_columns].to_parquet(
    private_root / "three_clock_crossfitted_calibrated_predictions_private.parquet",
    index=False,
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Metrics, uncertainty, and paired comparisons
# MAGIC
# MAGIC `predictive_r2` is explicitly defined as `1 - SSE/SST`; it is not Pearson
# MAGIC correlation squared. All bootstrap samples resample the common participant rows,
# MAGIC preserving the pairing of the three clocks.

# COMMAND ----------

raw_metrics = calibration.metric_table(
    analysis,
    raw_columns,
    stage="As observed before symmetric calibration",
)
calibrated_metrics = calibration.metric_table(
    analysis,
    calibrated_columns,
    stage="Symmetric participant-cross-fitted calibration",
)
metrics = pd.concat([raw_metrics, calibrated_metrics], ignore_index=True)
metrics.to_csv(statistics_root / "three_clock_calibration_metrics.csv", index=False)

raw_intervals = calibration.bootstrap_metric_intervals(
    analysis,
    raw_columns,
    stage="As observed before symmetric calibration",
    repetitions=bootstrap_repetitions,
    seed=random_seed,
)
calibrated_intervals = calibration.bootstrap_metric_intervals(
    analysis,
    calibrated_columns,
    stage="Symmetric participant-cross-fitted calibration",
    repetitions=bootstrap_repetitions,
    seed=random_seed + 1,
)
metric_intervals = pd.concat([raw_intervals, calibrated_intervals], ignore_index=True)
metric_intervals.to_csv(
    statistics_root / "three_clock_calibration_metric_intervals.csv", index=False
)

raw_comparisons = calibration.paired_clock_comparisons(
    analysis,
    raw_columns,
    stage="As observed before symmetric calibration",
    repetitions=paired_repetitions,
    seed=random_seed,
)
calibrated_comparisons = calibration.paired_clock_comparisons(
    analysis,
    calibrated_columns,
    stage="Symmetric participant-cross-fitted calibration",
    repetitions=paired_repetitions,
    seed=random_seed + 1,
)
paired_comparisons = pd.concat(
    [raw_comparisons, calibrated_comparisons], ignore_index=True
)
paired_comparisons.to_csv(
    statistics_root / "three_clock_paired_performance_comparisons.csv", index=False
)

try:
    display(metrics.round(4))  # noqa: F821
    display(
        paired_comparisons[
            [
                "stage",
                "first_clock",
                "second_clock",
                "delta_mae",
                "delta_mae_ci_low",
                "delta_mae_ci_high",
                "paired_signflip_p",
                "paired_signflip_q_bh",
            ]
        ].round(5)
    )  # noqa: F821
except NameError:
    print(metrics.round(4).to_string(index=False))
    print(paired_comparisons.round(5).to_string(index=False))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Age-bin, subgroup, and selection reporting

# COMMAND ----------

age_bins = calibration.age_bin_performance(
    analysis,
    calibrated_columns,
    minimum_n=minimum_subgroup_n,
)
age_bins.to_csv(statistics_root / "three_clock_calibrated_age_bin_performance.csv", index=False)

subgroup_candidates = [
    "sex_labeled",
    "sex",
    "racial_background",
    "site",
    "collection_site",
    "geographic_stratum",
    "cataract_any",
    "ICQ_CATRCT_COM",
    "VIS_CATRCT_COM",
]
subgroups = calibration.subgroup_performance(
    analysis,
    calibrated_columns,
    subgroup_candidates,
    minimum_n=minimum_subgroup_n,
)
subgroups.to_csv(
    statistics_root / "three_clock_calibrated_subgroup_performance.csv", index=False
)

calibration_curve = calibration.calibration_curve_table(
    analysis,
    calibrated_columns,
    minimum_n=minimum_subgroup_n,
)
calibration_curve.to_csv(
    statistics_root / "three_clock_calibration_curve_by_age.csv", index=False
)

characteristic_rows = []
age = analysis["chronological_age"]
for statistic, value in {
    "n": len(age),
    "mean": age.mean(),
    "sd": age.std(ddof=1),
    "median": age.median(),
    "q1": age.quantile(0.25),
    "q3": age.quantile(0.75),
    "minimum": age.min(),
    "maximum": age.max(),
}.items():
    characteristic_rows.append(
        {"variable": "chronological_age", "level_or_statistic": statistic, "value": value}
    )
for column in subgroup_candidates:
    if column not in analysis.columns:
        continue
    counts = analysis[column].astype("string").fillna("Missing").value_counts(dropna=False)
    for level, count in counts.items():
        characteristic_rows.append(
            {
                "variable": column,
                "level_or_statistic": str(level),
                "value": int(count),
                "percent": 100 * int(count) / len(analysis),
            }
        )
pd.DataFrame(characteristic_rows).to_csv(
    statistics_root / "three_clock_common_cohort_characteristics.csv", index=False
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Publication figures

# COMMAND ----------

colors = {"Retinal": "#2F5597", "Horvath": "#D97706", "Hannum": "#737B8C"}

figure, axes = plt.subplots(2, 3, figsize=(13.5, 8.5), layout="constrained")
for row, (stage, columns) in enumerate(
    [
        ("Before symmetric calibration", raw_columns),
        ("Symmetric cross-fitted calibration", calibrated_columns),
    ]
):
    for column_index, (clock, prediction_column) in enumerate(columns.items()):
        axis = axes[row, column_index]
        x = analysis["chronological_age"].to_numpy(float)
        y = analysis[prediction_column].to_numpy(float)
        limits = [min(x.min(), y.min()), max(x.max(), y.max())]
        axis.hexbin(x, y, gridsize=38, mincnt=1, cmap="viridis")
        axis.plot(limits, limits, linestyle="--", color="#555555", linewidth=1.2)
        metric = metrics.loc[
            metrics["stage"].eq(
                "As observed before symmetric calibration"
                if row == 0
                else "Symmetric participant-cross-fitted calibration"
            )
            & metrics["clock"].eq(clock)
        ].iloc[0]
        axis.text(
            0.03,
            0.97,
            f"MAE {metric.mae:.2f} y\nR² {metric.predictive_r2:.3f}\nSlope {metric.calibration_slope:.3f}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "none"},
        )
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_title(f"{clock}\n{stage}")
        axis.set_xlabel("Chronological age (years)")
        axis.set_ylabel("Estimated age (years)")
figure.savefig(
    figure_root / "figure_three_clock_symmetric_calibration_scatter.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
plt.close(figure)

calibrated_stage = "Symmetric participant-cross-fitted calibration"
mae_intervals = metric_intervals[
    metric_intervals["stage"].eq(calibrated_stage)
    & metric_intervals["metric"].eq("mae")
].copy()
calibrated_summary = calibrated_metrics.set_index("clock")
calibrated_pairs = calibrated_comparisons.copy()

figure, axes = plt.subplots(1, 3, figsize=(14, 4.7), layout="constrained")
positions = np.arange(len(mae_intervals))
axes[0].errorbar(
    mae_intervals["estimate"],
    positions,
    xerr=np.vstack(
        [
            mae_intervals["estimate"] - mae_intervals["ci_low"],
            mae_intervals["ci_high"] - mae_intervals["estimate"],
        ]
    ),
    fmt="o",
    color="#222222",
    capsize=4,
)
axes[0].set_yticks(positions, mae_intervals["clock"])
axes[0].invert_yaxis()
axes[0].set_xlabel("MAE (years; participant-bootstrap 95% CI)")
axes[0].set_title("Calibrated absolute error")

pair_labels = calibrated_pairs["first_clock"] + " − " + calibrated_pairs["second_clock"]
pair_positions = np.arange(len(calibrated_pairs))
axes[1].errorbar(
    calibrated_pairs["delta_mae"],
    pair_positions,
    xerr=np.vstack(
        [
            calibrated_pairs["delta_mae"] - calibrated_pairs["delta_mae_ci_low"],
            calibrated_pairs["delta_mae_ci_high"] - calibrated_pairs["delta_mae"],
        ]
    ),
    fmt="o",
    color="#2F5597",
    capsize=4,
)
axes[1].axvline(0, color="#555555", linestyle="--", linewidth=1)
axes[1].set_yticks(pair_positions, pair_labels)
axes[1].invert_yaxis()
axes[1].set_xlabel("Paired MAE difference (years; 95% CI)")
axes[1].set_title("Paired clock comparisons")

clock_order = list(calibrated_columns)
width = 0.36
x = np.arange(len(clock_order))
axes[2].bar(
    x - width / 2,
    [calibrated_summary.loc[clock, "calibration_slope"] for clock in clock_order],
    width,
    label="Calibration slope",
    color=[colors[clock] for clock in clock_order],
)
axes[2].bar(
    x + width / 2,
    [calibrated_summary.loc[clock, "mean_error"] for clock in clock_order],
    width,
    label="Mean error (years)",
    color="#B8C2CC",
)
axes[2].axhline(0, color="#555555", linewidth=0.8)
axes[2].axhline(1, color="#555555", linewidth=0.8, linestyle=":")
axes[2].set_xticks(x, clock_order)
axes[2].set_title("Residual calibration")
axes[2].legend(frameon=False, fontsize=9)
figure.savefig(
    figure_root / "figure_three_clock_symmetric_calibration_summary.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
plt.close(figure)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Reproducibility manifest and manuscript-ready reporting text

# COMMAND ----------


def format_ci(row: pd.Series) -> str:
    return f"{row.estimate:.2f} (95% CI {row.ci_low:.2f} to {row.ci_high:.2f})"


mae_lookup = mae_intervals.set_index("clock")
metric_lookup = calibrated_metrics.set_index("clock")
report_lines = [
    "## Symmetric cross-fitted age-calibration comparison",
    "",
    (
        f"In the complete three-clock cohort (n = {len(analysis):,}), the same five-knot "
        "cubic-spline plus ridge calibrator was fitted within the training participants of "
        "five common folds and applied only to held-out participants for retinal, Horvath, "
        "and Hannum age estimates."
    ),
    "",
]
for clock in calibrated_columns:
    report_lines.append(
        f"- {clock}: MAE {format_ci(mae_lookup.loc[clock])}; RMSE "
        f"{metric_lookup.loc[clock, 'rmse']:.2f} years; predictive R² "
        f"{metric_lookup.loc[clock, 'predictive_r2']:.3f}; Pearson r "
        f"{metric_lookup.loc[clock, 'pearson_r']:.3f}; calibration slope "
        f"{metric_lookup.loc[clock, 'calibration_slope']:.3f}; mean error "
        f"{metric_lookup.loc[clock, 'mean_error']:.2f} years; CCC "
        f"{metric_lookup.loc[clock, 'ccc']:.3f}."
    )
report_lines.extend(
    [
        "",
        "Paired participant-bootstrap intervals and two-sided sign-flip tests are in "
        "`three_clock_paired_performance_comparisons.csv`. Positive MAE differences mean "
        "the first named clock had greater absolute error.",
        "",
        (
            "Interpretation limit: applying a common held-out calibration removes asymmetry "
            "in the post-hoc age mapping, but the retinal predictor was still trained in CLSA "
            "whereas Horvath and Hannum are fixed published clocks. The result therefore does "
            "not compare modality information content independently of model-training provenance."
        ),
    ]
)
report_path = statistics_root / "three_clock_symmetric_calibration_results.md"
report_path.write_text("\n".join(report_lines), encoding="utf-8")

manifest = {
    "analysis": "symmetric_three_clock_crossfitted_calibration",
    "input_path": str(input_path),
    "retinal_oof_path": str(retinal_oof_path),
    "retinal_raw_merge_status": raw_retinal_merge_status,
    "retinal_input_is_raw_oof": retinal_is_raw,
    "n_common_participants": int(len(analysis)),
    "folds": int(analysis["calibration_fold"].nunique()),
    "calibrator": {
        "spline_knots": config.n_knots,
        "spline_degree": config.degree,
        "spline_extrapolation": "linear",
        "ridge_alpha": config.ridge_alpha,
    },
    "bootstrap_repetitions": bootstrap_repetitions,
    "paired_repetitions": paired_repetitions,
    "random_seed": random_seed,
    "predictive_r2_definition": "1 - SSE/SST on held-out participant predictions",
    "pairing_unit": "participant",
    "shared_folds_across_clocks": True,
    "calibration_cross_fitted": True,
    "upstream_training_provenance_is_symmetric": False,
    "interpretation_limit": (
        "The post-hoc calibration layer is symmetric, but the retinal estimator was trained "
        "in CLSA and the Horvath/Hannum clocks were externally defined."
    ),
    "outputs": {
        "private_predictions": str(
            private_root / "three_clock_crossfitted_calibrated_predictions_private.parquet"
        ),
        "metrics": str(statistics_root / "three_clock_calibration_metrics.csv"),
        "metric_intervals": str(
            statistics_root / "three_clock_calibration_metric_intervals.csv"
        ),
        "paired_comparisons": str(
            statistics_root / "three_clock_paired_performance_comparisons.csv"
        ),
        "fold_audit": str(statistics_root / "three_clock_calibration_fold_audit.csv"),
        "reporting_text": str(report_path),
    },
}
(statistics_root / "three_clock_calibration_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
)

print(report_path.read_text(encoding="utf-8"))
print("Completed symmetric calibration analysis:", output_root)
