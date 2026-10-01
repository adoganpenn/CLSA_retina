# Final cell: combined retinal + measured DNAm ages for prospective mortality.
# "Combined/multimodal" Cox model, NOT a random-effects mixed model: one row/person.
# Uses completed upstream cohorts and their existing retinal-index survival times.
# All four models use the identical complete-case cohort and nested validation folds.
from pathlib import Path
import json
import shutil
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold

if "retinal_cohort" not in globals() or "epigenetic_cohort" not in globals():
    raise RuntimeError("Run baseline cohort construction first (cell 10).")
if "output_root" not in globals():
    raise RuntimeError("Run the configuration cell first to define output_root.")

cm_seed = int(globals().get("random_seed", 20260914))
cm_bootstraps = int(globals().get("bootstrap_repetitions", 1000))
cm_outer_folds = int(globals().get("outer_folds", 5))
cm_inner_folds = int(globals().get("inner_folds", 3))
cm_penalties = sorted(set(float(x) for x in globals().get(
    "ridge_penalizer_grid", (0.01, 0.1, 1.0))))
cm_horizons = tuple(float(x) for x in globals().get(
    "evaluation_horizons_years", (5.0, 10.0)))
if not cm_penalties or min(cm_penalties) <= 0:
    raise ValueError("Use strictly positive ridge penalties for correlated age predictors.")
if cm_bootstraps < 20 or not cm_horizons:
    raise ValueError("Need at least 20 bootstrap draws and one evaluation horizon.")

cm_clock_labels = {
    "epigenetic_dnam_age": "Horvath DNAm age",
    "epigenetic_hannum_age": "Hannum DNAm age",
}
cm_required = ["participant_id", "followup_years", "event",
               "age_at_fundus_years", "sex_female"]
for cm_label, cm_source in [("retinal", retinal_cohort),
                           ("epigenetic", epigenetic_cohort)]:
    cm_missing = set(cm_required) - set(cm_source.columns)
    if cm_missing:
        raise ValueError(f"{cm_label} cohort missing columns: {sorted(cm_missing)}")
    if cm_source["participant_id"].isna().any() or cm_source["participant_id"].duplicated().any():
        raise ValueError(f"{cm_label} cohort must have one valid ID per participant.")

# DNAm comes from the existing released-data cohort, not a guessed new source.
cm_clocks = [c for c in cm_clock_labels if c in epigenetic_cohort.columns
             and pd.to_numeric(epigenetic_cohort[c], errors="coerce").notna().any()]
if not cm_clocks:
    raise ValueError("Neither released Horvath nor Hannum DNAm age is available.")
print("Measured DNAm ages included:", [cm_clock_labels[c] for c in cm_clocks])
print("Unavailable absolute clocks:", [cm_clock_labels[c] for c in cm_clock_labels
                                      if c not in cm_clocks])
cm_retina = retinal_cohort[cm_required].copy()
if "retinal_age" in retinal_cohort.columns:
    cm_retina["retinal_age"] = pd.to_numeric(retinal_cohort["retinal_age"], errors="coerce")
elif "retinal_age_gap" in retinal_cohort.columns:
    # Exact inverse of the upstream definition; never refits an age model.
    cm_retina["retinal_age"] = (pd.to_numeric(retinal_cohort["retinal_age_gap"], errors="coerce")
                               + pd.to_numeric(retinal_cohort["age_at_fundus_years"], errors="coerce"))
else:
    raise ValueError("Existing retinal_age/retinal_age_gap is missing; do not substitute embeddings.")
if not cm_retina["retinal_age"].notna().any():
    raise ValueError("No existing retinal ages; run the upstream age-prediction linkage.")
cm_frame = cm_retina.merge(epigenetic_cohort[["participant_id", *cm_clocks]],
                          on="participant_id", how="inner", validate="one_to_one")
cm_linked_n = len(cm_frame)
cm_numeric = [*cm_required[1:], "retinal_age", *cm_clocks]
for cm_column in cm_numeric:
    cm_frame[cm_column] = pd.to_numeric(cm_frame[cm_column], errors="coerce")
cm_frame = cm_frame.replace([np.inf, -np.inf], np.nan).dropna(subset=cm_numeric)
cm_frame = cm_frame.loc[cm_frame["followup_years"].gt(0)
                        & cm_frame["event"].isin([0, 1])
                        & cm_frame["sex_female"].isin([0, 1])].copy()
cm_frame = cm_frame.sort_values("participant_id", kind="stable").reset_index(drop=True)
cm_frame["event"] = cm_frame["event"].astype(int)
cm_events = int(cm_frame["event"].sum())
cm_minimum_events = max(30, int(globals().get("minimum_events", 30)))
if cm_events < cm_minimum_events:
    raise ValueError(f"Combined complete-case cohort has {cm_events} deaths; need >= {cm_minimum_events}.")
cm_outer_folds = min(cm_outer_folds, cm_events, len(cm_frame) - cm_events)
if cm_outer_folds < 2 or cm_inner_folds < 2:
    raise ValueError("Insufficient event/non-event participants for nested cross-validation.")
cm_cohort_summary = pd.DataFrame([{
    "linked_participants": cm_linked_n, "complete_case_participants": len(cm_frame),
    "excluded_incomplete_or_invalid": cm_linked_n - len(cm_frame), "deaths": cm_events,
    "median_followup_years": float(cm_frame["followup_years"].median()),
}])
print(cm_cohort_summary.to_string(index=False))
print("Time origin: existing retinal baseline/index date; one identical outcome per model.")

cm_models = {
    "Age + sex": [],
    "Age + sex + retinal age": ["retinal_age"],
    "Age + sex + DNAm ages": cm_clocks,
    "Age + sex + retinal + DNAm ages": ["retinal_age", *cm_clocks],
}

def cm_design(train, test, additions):
    # All transformations estimated on the current training fold only.
    age_mean = train["age_at_fundus_years"].mean()
    age_sd = train["age_at_fundus_years"].std(ddof=0)
    if not np.isfinite(age_sd) or age_sd <= 0:
        raise ValueError("Chronological age is constant in a training fold.")
    train_age = (train["age_at_fundus_years"] - age_mean) / age_sd
    test_age = (test["age_at_fundus_years"] - age_mean) / age_sd
    x_train = pd.DataFrame({"age_z": train_age, "age_z_squared": train_age**2,
                            "sex_female": train["sex_female"]}, index=train.index)
    x_test = pd.DataFrame({"age_z": test_age, "age_z_squared": test_age**2,
                           "sex_female": test["sex_female"]}, index=test.index)
    for column in additions:
        mean, sd = train[column].mean(), train[column].std(ddof=0)
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"{column} is constant in a training fold.")
        x_train[column + "_z"] = (train[column] - mean) / sd
        x_test[column + "_z"] = (test[column] - mean) / sd
    keep = [c for c in x_train if x_train[c].nunique() > 1]
    return x_train[keep], x_test[keep]

def cm_fit(train, test, additions, penalty):
    x_train, x_test = cm_design(train, test, additions)
    fit_data = x_train.assign(followup_years=train["followup_years"], event=train["event"])
    model = CoxPHFitter(penalizer=penalty, l1_ratio=0.0)
    model.fit(fit_data, duration_col="followup_years", event_col="event", show_progress=False)
    risk = model.predict_log_partial_hazard(x_test).to_numpy(dtype=float)
    return model, fit_data, x_test, risk

def cm_c_index(frame, risk):
    return float(concordance_index(frame["followup_years"], -np.asarray(risk), frame["event"]))

cm_tuning_rows, cm_prediction_parts, cm_fold_rows = [], [], []
cm_splitter = StratifiedKFold(n_splits=cm_outer_folds, shuffle=True, random_state=cm_seed)
for cm_fold, (cm_train_ids, cm_test_ids) in enumerate(
        cm_splitter.split(cm_frame, cm_frame["event"]), start=1):
    cm_train, cm_test = cm_frame.iloc[cm_train_ids], cm_frame.iloc[cm_test_ids]
    cm_inner_count = min(cm_inner_folds, int(cm_train["event"].sum()),
                         int(cm_train["event"].eq(0).sum()))
    if cm_inner_count < 2:
        raise ValueError(f"Outer fold {cm_fold}: insufficient events for inner validation.")
    cm_inner = list(StratifiedKFold(n_splits=cm_inner_count, shuffle=True,
                   random_state=cm_seed + cm_fold).split(cm_train, cm_train["event"]))
    for cm_name, cm_additions in cm_models.items():
        cm_candidates = []
        for cm_penalty in cm_penalties:
            cm_scores = []
            for cm_itr, cm_iva in cm_inner:
                cm_a, cm_b = cm_train.iloc[cm_itr], cm_train.iloc[cm_iva]
                _, _, _, cm_risk = cm_fit(cm_a, cm_b, cm_additions, cm_penalty)
                cm_scores.append(cm_c_index(cm_b, cm_risk))
            cm_score = float(np.mean(cm_scores))
            cm_tuning_rows.append({"fold": cm_fold, "model": cm_name,
                                   "penalizer": cm_penalty, "mean_inner_c_index": cm_score})
            cm_candidates.append((cm_score, cm_penalty))
        # Fixed deterministic tie break: prefer more shrinkage for identical scores.
        cm_best_penalty = max(cm_candidates, key=lambda item: (item[0], item[1]))[1]
        cm_fit_model, _, cm_x_test, cm_risk = cm_fit(cm_train, cm_test, cm_additions, cm_best_penalty)
        cm_prediction = cm_test[cm_required[:3]].copy()
        cm_prediction["log_risk"] = cm_risk
        cm_prediction["fold"], cm_prediction["model"] = cm_fold, cm_name
        for cm_horizon in cm_horizons:
            cm_prediction[f"predicted_survival_{cm_horizon:g}y"] = (
                cm_fit_model.predict_survival_function(cm_x_test, times=[cm_horizon])
                .iloc[0].to_numpy(dtype=float))
        cm_prediction_parts.append(cm_prediction)
        cm_fold_rows.append({"fold": cm_fold, "model": cm_name,
                             "participants": len(cm_test), "deaths": int(cm_test["event"].sum()),
                             "penalizer": cm_best_penalty, "c_index": cm_c_index(cm_test, cm_risk)})
    print(f"Combined mortality: outer fold {cm_fold}/{cm_outer_folds} completed.")

cm_oof = pd.concat(cm_prediction_parts, ignore_index=True)
cm_tuning = pd.DataFrame(cm_tuning_rows)
cm_fold_performance = pd.DataFrame(cm_fold_rows)
cm_names = list(cm_models)
cm_aligned = [cm_oof.loc[cm_oof["model"].eq(name)].sort_values("participant_id")
              .reset_index(drop=True) for name in cm_names]
for cm_group in cm_aligned:
    if not cm_group["participant_id"].equals(cm_frame["participant_id"]):
        raise AssertionError("OOF participant coverage/order differs across models.")
    if not np.isfinite(cm_group["log_risk"]).all():
        raise AssertionError("Non-finite held-out predictions.")
cm_point_c = np.array([cm_c_index(cm_group, cm_group["log_risk"]) for cm_group in cm_aligned])
cm_rng, cm_boot_c = np.random.default_rng(cm_seed + 117), []
for cm_iteration in range(cm_bootstraps):
    cm_indices = cm_rng.integers(0, len(cm_frame), len(cm_frame))
    try:
        cm_values = [cm_c_index(g.iloc[cm_indices], g["log_risk"].iloc[cm_indices]) for g in cm_aligned]
        cm_boot_c.append(cm_values)
    except ZeroDivisionError:
        continue
if len(cm_boot_c) < max(20, int(0.9 * cm_bootstraps)):
    raise RuntimeError("Too few evaluable paired bootstrap samples.")
cm_boot_c = np.asarray(cm_boot_c)
cm_c_ci = np.quantile(cm_boot_c, [0.025, 0.975], axis=0)
cm_performance = pd.DataFrame([{
    "model": name, "participants": len(cm_frame), "deaths": cm_events,
    "c_index": cm_point_c[j], "c_index_lower_95": cm_c_ci[0, j],
    "c_index_upper_95": cm_c_ci[1, j],
    "mean_within_fold_c_index": cm_fold_performance.loc[
        cm_fold_performance["model"].eq(name), "c_index"].mean(),
} for j, name in enumerate(cm_names)])
cm_contrasts = [(1, 0), (2, 0), (3, 0), (3, 1), (3, 2)]
cm_delta_rows = []
for cm_new, cm_reference in cm_contrasts:
    cm_ci = np.quantile(cm_boot_c[:, cm_new] - cm_boot_c[:, cm_reference], [0.025, 0.975])
    cm_delta_rows.append({"model": cm_names[cm_new], "reference": cm_names[cm_reference],
                          "delta_c_index": cm_point_c[cm_new] - cm_point_c[cm_reference],
                          "delta_c_lower_95": cm_ci[0], "delta_c_upper_95": cm_ci[1]})
cm_incremental = pd.DataFrame(cm_delta_rows)

# Censoring-aware calibration. Do not treat censored participants as surviving forever.
cm_calibration_rows = []
for cm_name, cm_group in zip(cm_names, cm_aligned):
    for cm_horizon in cm_horizons:
        if cm_group["followup_years"].max() < cm_horizon:
            print(f"Not estimable: {cm_horizon:g}-year calibration beyond available follow-up.")
            continue
        cm_group = cm_group.copy()
        cm_group["predicted_risk"] = 1 - cm_group[f"predicted_survival_{cm_horizon:g}y"]
        cm_group["risk_group"] = pd.qcut(cm_group["predicted_risk"], 5, labels=False, duplicates="drop")
        for cm_bin, cm_bin_data in cm_group.groupby("risk_group", observed=True):
            cm_km = KaplanMeierFitter().fit(cm_bin_data["followup_years"], cm_bin_data["event"])
            cm_position = max(0, np.searchsorted(cm_km.confidence_interval_.index,
                                               cm_horizon, side="right") - 1)
            cm_bounds = cm_km.confidence_interval_.iloc[cm_position]
            cm_at_risk = int(cm_bin_data["followup_years"].ge(cm_horizon).sum())
            cm_calibration_rows.append({"model": cm_name, "horizon_years": cm_horizon,
                "risk_group": int(cm_bin) + 1, "participants": len(cm_bin_data),
                "deaths_by_horizon": int((cm_bin_data["event"].eq(1)
                                           & cm_bin_data["followup_years"].le(cm_horizon)).sum()),
                "at_risk_at_horizon": cm_at_risk,
                "mean_predicted_risk": cm_bin_data["predicted_risk"].mean(),
                "observed_km_risk": 1 - float(cm_km.predict(cm_horizon)) if cm_at_risk else np.nan,
                "observed_lower_95": 1 - float(cm_bounds.iloc[1]) if cm_at_risk else np.nan,
                "observed_upper_95": 1 - float(cm_bounds.iloc[0]) if cm_at_risk else np.nan,
                "sparse_tail_flag": cm_at_risk < 20})
cm_calibration = pd.DataFrame(cm_calibration_rows)

# Descriptive full-cohort penalized fit for PH diagnostics only, NOT performance.
cm_full_penalty = float(cm_fold_performance.loc[
    cm_fold_performance["model"].eq(cm_names[3]), "penalizer"].max())
cm_full_model, cm_full_data, _, _ = cm_fit(cm_frame, cm_frame, cm_models[cm_names[3]], cm_full_penalty)
cm_ph = proportional_hazard_test(cm_full_model, cm_full_data, time_transform="rank").summary
cm_ph = cm_ph.rename_axis("term").reset_index().rename(columns={"-log2(p)": "minus_log2_p"})
cm_ph["model"], cm_ph["penalizer"] = cm_names[3], cm_full_penalty
print("\nHeld-out discrimination (common cohort):\n", cm_performance.round(4).to_string(index=False))
print("\nPaired incremental discrimination:\n", cm_incremental.round(4).to_string(index=False))
print("\nDescriptive PH diagnostics, penalized combined fit:\n", cm_ph.round(4).to_string(index=False))
print("Bootstrap CIs condition on fitted OOF predictions; they do not include refitting uncertainty.")
print("Exploratory internal validation only. No claim of clinical utility or external validation.")

cm_export_root = Path(output_root) / "combined_retinal_dnam_ages"
cm_export_root.mkdir(parents=True, exist_ok=True)
# Stage locally, then stream-copy: avoids random-access writes on /Volumes.
cm_output_paths = []
with tempfile.TemporaryDirectory(prefix="clsa_combined_mortality_") as cm_stage:
    cm_stage = Path(cm_stage)
    for cm_stem, cm_table in {
        "cohort_summary": cm_cohort_summary, "oof_predictions": cm_oof,
        "nested_cv_tuning": cm_tuning, "fold_performance": cm_fold_performance,
        "discrimination": cm_performance, "paired_incremental_c_index": cm_incremental,
        "calibration": cm_calibration, "proportional_hazards_diagnostics": cm_ph,
    }.items():
        cm_local = cm_stage / f"{cm_stem}.csv"
        cm_table.to_csv(cm_local, index=False)
        cm_target = cm_export_root / cm_local.name
        shutil.copyfile(cm_local, cm_target)
        cm_output_paths.append(str(cm_target))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        cm_fig, cm_axes = plt.subplots(1, 3, figsize=(16, 5.3), layout="constrained")
        cm_short_names = ["Age + sex", "+ retinal age", "+ DNAm ages", "+ both modalities"]
        cm_colors = ["#6B7280", "#2F5597", "#D97706", "#15803D"]
        for cm_j in range(4):
            cm_axes[0].errorbar(cm_point_c[cm_j], cm_j, xerr=[[cm_point_c[cm_j] - cm_c_ci[0, cm_j]],
                [cm_c_ci[1, cm_j] - cm_point_c[cm_j]]], fmt="o", color=cm_colors[cm_j], capsize=3)
        cm_axes[0].set(yticks=range(4), yticklabels=cm_short_names,
                       xlabel="Held-out Harrell C-index", title="A  Common-cohort prediction")
        cm_axes[0].invert_yaxis()
        cm_plot_delta = cm_incremental.iloc[[2, 3, 4]]
        cm_values = cm_plot_delta["delta_c_index"].to_numpy()
        cm_axes[1].errorbar(cm_values, range(3), xerr=np.stack([
            cm_values - cm_plot_delta["delta_c_lower_95"].to_numpy(),
            cm_plot_delta["delta_c_upper_95"].to_numpy() - cm_values]), fmt="o", color=cm_colors[3], capsize=3)
        cm_axes[1].axvline(0, color="#111827", linestyle="--", linewidth=0.8)
        cm_axes[1].set(yticks=range(3), yticklabels=["vs age + sex", "vs retinal age", "vs DNAm ages"],
                       xlabel="Combined-model increment in C-index", title="B  Does combining help?")
        cm_axes[1].invert_yaxis()
        cm_usable_horizons = (cm_calibration.loc[
            cm_calibration["observed_km_risk"].notna(), "horizon_years"].unique()
            if not cm_calibration.empty else [])
        if len(cm_usable_horizons):
            cm_plot_horizon = max(cm_usable_horizons)
            cm_max_risk = 0.0
            for cm_j, cm_name in enumerate(cm_names):
                cm_cal = cm_calibration.loc[cm_calibration["model"].eq(cm_name)
                    & cm_calibration["horizon_years"].eq(cm_plot_horizon)].dropna(subset=["observed_km_risk"])
                cm_axes[2].plot(cm_cal["mean_predicted_risk"], cm_cal["observed_km_risk"], "o-",
                               label=cm_short_names[cm_j], color=cm_colors[cm_j], linewidth=1)
                if len(cm_cal):
                    cm_max_risk = max(cm_max_risk, cm_cal["mean_predicted_risk"].max(),
                                      cm_cal["observed_km_risk"].max())
            cm_limit = min(1.0, max(0.1, float(cm_max_risk) * 1.15))
            cm_axes[2].plot([0, cm_limit], [0, cm_limit], "--", color="#111827", linewidth=0.8)
            cm_axes[2].set(xlim=(0, cm_limit), ylim=(0, cm_limit), xlabel="Predicted mortality risk",
                           ylabel="Observed Kaplan-Meier risk", title=f"C  {cm_plot_horizon:g}-year calibration")
            cm_axes[2].legend(fontsize=8, loc="upper left")
        else:
            cm_axes[2].text(0.5, 0.5, "Calibration horizon not estimable", ha="center", transform=cm_axes[2].transAxes)
            cm_axes[2].set_axis_off()
        cm_fig.suptitle(f"Combined retinal-DNAm mortality model: n = {len(cm_frame):,}; deaths = {cm_events:,}", fontsize=12)
        for cm_extension in ("png", "pdf"):
            cm_local = cm_stage / f"combined_mortality_summary.{cm_extension}"
            cm_fig.savefig(cm_local, dpi=300, facecolor="white")
            cm_target = cm_export_root / cm_local.name
            shutil.copyfile(cm_local, cm_target)
            cm_output_paths.append(str(cm_target))
        plt.show()
        plt.close(cm_fig)
    cm_metadata = {
        "analysis": "Exploratory combined retinal and DNAm age mortality prediction",
        "model_type": "Ridge-penalized Cox; not random-effects mixed model",
        "absolute_dnam_ages": cm_clocks, "retinal_predictor": "existing retinal_age",
        "retinal_prediction_path": str(globals().get("prediction_path", "upstream in-memory cohort")),
        "retinal_prediction_provenance": str(globals().get("prediction_provenance", "upstream cohort")),
        "epigenetic_source": globals().get("epigenetic_source_metadata", {}),
        "time_origin": "Existing retinal baseline/index survival outcome for all models",
        "adjustment": ["chronological age (linear and quadratic)", "sex"],
        "participants": len(cm_frame), "deaths": cm_events, "models": cm_models,
        "outer_folds": cm_outer_folds, "inner_folds_requested": cm_inner_folds,
        "ridge_penalties": cm_penalties, "random_seed": cm_seed,
        "bootstrap_draws_requested": cm_bootstraps, "bootstrap_draws_evaluable": len(cm_boot_c),
        "bootstrap_scope": "Paired participant resampling of fixed OOF predictions; no refitting",
        "complete_case": True, "external_validation": False, "raw_dates_exported": False,
        "calibration_sparse_tail_threshold": 20, "output_paths": cm_output_paths,
        "references": ["https://www.bmj.com/content/385/bmj-2023-078378",
                       "https://lifelines.readthedocs.io/en/stable/fitters/regression/CoxPHFitter.html"],
    }
    cm_local = cm_stage / "combined_model_metadata.json"
    cm_local.write_text(json.dumps(cm_metadata, indent=2, default=str), encoding="utf-8")
    shutil.copyfile(cm_local, cm_export_root / cm_local.name)
print("\nCombined model tables and PNG/PDF saved to:", cm_export_root)
