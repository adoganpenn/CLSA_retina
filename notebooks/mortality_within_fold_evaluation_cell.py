# Final cell: correct within-fold mortality discrimination; NO MODEL REFITTING.
# Primary estimate = equally weighted mean of the held-out fold C-indices.
# Sensitivity estimate = concordance weighted by within-fold comparable pairs.
# Bootstrap resamples the SAME participants for all models within each fold.
from pathlib import Path
import json
import shutil
import tempfile
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

wf_root = globals().get("cm_export_root")
if wf_root is None:
    if "output_root" not in globals():
        raise RuntimeError("Define output_root or run the combined-model cell first.")
    wf_root = Path(output_root) / "combined_retinal_dnam_ages"
wf_root = Path(wf_root)
if "cm_oof" in globals():
    wf_predictions = cm_oof.copy()
else:
    wf_input_path = wf_root / "oof_predictions.csv"
    if not wf_input_path.is_file():
        raise FileNotFoundError(f"Run combined-model cell first; missing {wf_input_path}")
    wf_predictions = pd.read_csv(wf_input_path, dtype={"participant_id": "string"})

wf_names = ["Age + sex", "Age + sex + retinal age", "Age + sex + DNAm ages",
            "Age + sex + retinal + DNAm ages"]
wf_columns = ["participant_id", "model", "fold", "followup_years", "event", "log_risk"]
wf_missing = set(wf_columns) - set(wf_predictions.columns)
if wf_missing:
    raise ValueError(f"OOF predictions missing columns: {sorted(wf_missing)}")
wf_predictions = wf_predictions[wf_columns].copy()
if wf_predictions.isna().any().any():
    raise ValueError("Missing values in required OOF columns; do not drop participants.")
wf_predictions["participant_id"] = wf_predictions["participant_id"].astype("string")
for wf_column in ["fold", "followup_years", "event", "log_risk"]:
    wf_predictions[wf_column] = pd.to_numeric(wf_predictions[wf_column], errors="raise")
if not np.isfinite(wf_predictions[["fold", "followup_years", "event", "log_risk"]]).all().all():
    raise ValueError("Non-finite OOF quantities.")
if (not wf_predictions["event"].isin([0, 1]).all()
        or not wf_predictions["followup_years"].gt(0).all()
        or not wf_predictions["fold"].eq(wf_predictions["fold"].astype(int)).all()):
    raise ValueError("Invalid survival times, event flags, or fold identifiers.")
if set(wf_predictions["model"]) != set(wf_names):
    raise ValueError(f"Expected exactly the four combined-model comparators; found {sorted(set(wf_predictions['model']))}")
if wf_predictions.duplicated(["participant_id", "model"]).any():
    raise ValueError("OOF predictions must have one row per participant per model.")
wf_groups = [wf_predictions.loc[wf_predictions["model"].eq(name)]
             .sort_values("participant_id", kind="stable").reset_index(drop=True)
             for name in wf_names]
wf_reference = wf_groups[0]
for wf_group in wf_groups[1:]:
    if not wf_group[["participant_id", "fold", "followup_years", "event"]].equals(
            wf_reference[["participant_id", "fold", "followup_years", "event"]]):
        raise ValueError("Models differ in participant coverage, folds, or outcomes; paired comparison invalid.")
wf_folds = sorted(wf_reference["fold"].unique())
if len(wf_folds) < 2:
    raise ValueError("Need at least two held-out folds.")
wf_seed = int(globals().get("random_seed", 20260914)) + 291
wf_repetitions = int(globals().get("bootstrap_repetitions", 1000))
if wf_repetitions < 20:
    raise ValueError("Need at least 20 bootstrap repetitions (1000 recommended for reporting).")

# Precompute comparable pairs, never comparing people assigned to different folds.
# Equal event/censor times are comparable; two equal event times are not.
# Validate all pair-count C-indices against the public lifelines implementation.
wf_cache, wf_fold_rows = [], []
for wf_fold in wf_folds:
    wf_mask = wf_reference["fold"].eq(wf_fold).to_numpy()
    wf_data = wf_reference.loc[wf_mask]
    wf_time = wf_data["followup_years"].to_numpy(float)
    wf_event = wf_data["event"].to_numpy(int)
    wf_risks = np.stack([g.loc[wf_mask, "log_risk"].to_numpy(float) for g in wf_groups])
    wf_i, wf_j = np.triu_indices(len(wf_data), k=1)
    wf_eligible = (((wf_time[wf_i] < wf_time[wf_j]) & (wf_event[wf_i] == 1))
                   | ((wf_time[wf_j] < wf_time[wf_i]) & (wf_event[wf_j] == 1))
                   | ((wf_time[wf_i] == wf_time[wf_j]) & (wf_event[wf_i] != wf_event[wf_j])))
    wf_i, wf_j = wf_i[wf_eligible], wf_j[wf_eligible]
    if not len(wf_i):
        raise ValueError(f"Fold {wf_fold} has no comparable survival pairs.")
    wf_i_first = ((wf_time[wf_i] < wf_time[wf_j])
                  | ((wf_time[wf_i] == wf_time[wf_j]) & (wf_event[wf_i] == 1)))
    wf_difference = (wf_risks[:, wf_i] - wf_risks[:, wf_j]) * np.where(wf_i_first, 1, -1)
    wf_credit = (wf_difference > 0).astype(float) + 0.5 * (wf_difference == 0)
    wf_fold_c = wf_credit.mean(axis=1)
    for wf_m, wf_name in enumerate(wf_names):
        wf_check = concordance_index(wf_time, -wf_risks[wf_m], wf_event)
        if not np.isclose(wf_fold_c[wf_m], wf_check, atol=1e-12, rtol=0):
            raise AssertionError(f"Comparable-pair implementation differs from lifelines: {wf_name}, fold {wf_fold}")
        wf_fold_rows.append({"fold": int(wf_fold), "model": wf_name,
            "participants": len(wf_data), "deaths": int(wf_event.sum()),
            "comparable_pairs": len(wf_i), "c_index": float(wf_fold_c[wf_m])})
    wf_cache.append((len(wf_data), wf_i, wf_j, wf_credit))
wf_fold_results = pd.DataFrame(wf_fold_rows)
wf_fold_matrix = np.stack([credit.mean(axis=1) for _, _, _, credit in wf_cache])
wf_point_mean = wf_fold_matrix.mean(axis=0)
wf_pair_counts = np.array([len(i) for _, i, _, _ in wf_cache], dtype=float)
wf_point_weighted = np.average(wf_fold_matrix, axis=0, weights=wf_pair_counts)

wf_rng = np.random.default_rng(wf_seed)
wf_boot_mean, wf_boot_weighted = [], []
for wf_iteration in range(wf_repetitions):
    wf_draw_c, wf_draw_counts = [], []
    for wf_n, wf_i, wf_j, wf_credit in wf_cache:
        # Multiplicities reproduce an ordinary participant bootstrap, including
        # repeated records, without rebuilding quadratic comparison matrices.
        for wf_attempt in range(100):
            wf_multiplicity = np.bincount(wf_rng.integers(0, wf_n, wf_n), minlength=wf_n)
            wf_weights = wf_multiplicity[wf_i] * wf_multiplicity[wf_j]
            wf_denominator = float(wf_weights.sum())
            if wf_denominator > 0:
                break
        else:
            raise RuntimeError("Repeated bootstrap draws have no comparable pairs; cohort too sparse.")
        wf_draw_c.append((wf_credit * wf_weights[None, :]).sum(axis=1) / wf_denominator)
        wf_draw_counts.append(wf_denominator)
    wf_boot_mean.append(np.mean(wf_draw_c, axis=0))
    wf_boot_weighted.append(np.average(wf_draw_c, axis=0, weights=wf_draw_counts))
wf_boot_mean, wf_boot_weighted = np.asarray(wf_boot_mean), np.asarray(wf_boot_weighted)
wf_methods = {
    "Mean within-fold C-index (primary)": (wf_point_mean, wf_boot_mean),
    "Comparable-pair-weighted within-fold C-index (sensitivity)": (wf_point_weighted, wf_boot_weighted),
}
wf_performance_rows, wf_contrast_rows = [], []
wf_contrasts = [(1, 0), (2, 0), (3, 0), (3, 1), (3, 2)]
for wf_method, (wf_points, wf_boot_values) in wf_methods.items():
    wf_bounds = np.quantile(wf_boot_values, [0.025, 0.975], axis=0)
    for wf_m, wf_name in enumerate(wf_names):
        wf_performance_rows.append({"method": wf_method, "model": wf_name,
            "participants": len(wf_reference), "deaths": int(wf_reference["event"].sum()),
            "c_index": wf_points[wf_m], "c_index_lower_95": wf_bounds[0, wf_m],
            "c_index_upper_95": wf_bounds[1, wf_m]})
    for wf_new, wf_base in wf_contrasts:
        wf_ci = np.quantile(wf_boot_values[:, wf_new] - wf_boot_values[:, wf_base], [0.025, 0.975])
        wf_contrast_rows.append({"method": wf_method, "model": wf_names[wf_new],
            "reference": wf_names[wf_base], "delta_c_index": wf_points[wf_new] - wf_points[wf_base],
            "delta_c_lower_95": wf_ci[0], "delta_c_upper_95": wf_ci[1]})
wf_performance = pd.DataFrame(wf_performance_rows)
wf_incremental = pd.DataFrame(wf_contrast_rows)
wf_primary_performance = wf_performance.loc[wf_performance["method"].eq(next(iter(wf_methods)))]
wf_primary_incremental = wf_incremental.loc[wf_incremental["method"].eq(next(iter(wf_methods)))]
print("\nCorrected mean within-fold discrimination:\n", wf_primary_performance.round(4).to_string(index=False))
print("\nCorrected paired within-fold increments:\n", wf_primary_incremental.round(4).to_string(index=False))
print("\nComparable-pair-weighted sensitivity:\n", wf_performance.iloc[4:].round(4).to_string(index=False))

wf_summary_lines = [
    f"Mortality prediction in the common retinal-DNAm subset: {len(wf_reference):,} participants and {int(wf_reference['event'].sum()):,} deaths.",
    f"Combined-model mean within-fold C-index: {wf_point_mean[3]:.3f}; age + sex: {wf_point_mean[0]:.3f}; retinal: {wf_point_mean[1]:.3f}; DNAm: {wf_point_mean[2]:.3f}.",
]
for wf_new, wf_base in [(3, 0), (3, 1), (3, 2)]:
    wf_row = wf_primary_incremental.loc[wf_primary_incremental["model"].eq(wf_names[wf_new])
                                       & wf_primary_incremental["reference"].eq(wf_names[wf_base])].iloc[0]
    wf_interpretation = ("positive increment; pointwise CI excludes zero" if wf_row["delta_c_lower_95"] > 0
        else "negative increment; pointwise CI excludes zero" if wf_row["delta_c_upper_95"] < 0
        else "uncertain increment; CI includes zero")
    wf_summary_lines.append(f"Combined versus {wf_names[wf_base]}: delta C = {wf_row['delta_c_index']:.4f} "
        f"(95% CI {wf_row['delta_c_lower_95']:.4f} to {wf_row['delta_c_upper_95']:.4f}); {wf_interpretation}.")
wf_summary_lines.extend([
    "No comparisons cross validation folds; no models were refitted.",
    "Intervals are exploratory and pointwise (not multiplicity-adjusted). Bootstrap conditions on the existing fitted models and fold assignments; training/refitting uncertainty is not included.",
    "These results do not establish clinical utility, causality, equivalence, or external validity.",
    "Use these corrected discrimination estimates instead of the earlier pooled log-partial-hazard comparisons. The previous calibration results are unchanged.",
])
wf_summary = "\n".join(wf_summary_lines)
print("\nHIGH-LEVEL SUMMARY\n" + wf_summary)
wf_root.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix="clsa_within_fold_") as wf_stage:
    wf_stage = Path(wf_stage)
    for wf_stem, wf_table in {"corrected_within_fold_discrimination": wf_performance,
        "corrected_paired_within_fold_increment": wf_incremental,
        "corrected_fold_performance": wf_fold_results}.items():
        wf_local = wf_stage / f"{wf_stem}.csv"
        wf_table.to_csv(wf_local, index=False)
        shutil.copyfile(wf_local, wf_root / wf_local.name)
    wf_local = wf_stage / "corrected_high_level_summary.txt"
    wf_local.write_text(wf_summary + "\n", encoding="utf-8")
    shutil.copyfile(wf_local, wf_root / wf_local.name)
    wf_metadata = {"evaluation": "Paired fold-stratified participant bootstrap of fixed OOF predictions",
        "primary": "Equal-weight mean of within-fold C-indices", "sensitivity": "Comparable-pair-weighted within-fold concordance",
        "folds": [int(f) for f in wf_folds], "random_seed": wf_seed, "bootstrap_repetitions": wf_repetitions,
        "cross_fold_pairs_used": False, "model_refitting": False, "multiplicity_adjusted": False,
        "training_uncertainty_included": False, "pair_counts_verified_against_lifelines": True,
        "supersedes": "Earlier pooled log-partial-hazard discrimination; not calibration",
        "source": str(wf_root / "oof_predictions.csv"),
        "source_used": "in-memory cm_oof" if "cm_oof" in globals() else "persisted CSV"}
    wf_local = wf_stage / "corrected_evaluation_metadata.json"
    wf_local.write_text(json.dumps(wf_metadata, indent=2), encoding="utf-8")
    shutil.copyfile(wf_local, wf_root / wf_local.name)
print("\nCorrected tables and high-level summary saved to:", wf_root)
