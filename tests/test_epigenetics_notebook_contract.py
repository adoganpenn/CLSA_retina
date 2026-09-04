import ast
import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).parents[1]
    / "Age_Glaucoma"
    / "Algorithm Fairness"
    / "epigenetics.ipynb"
)


def _source():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    return notebook, source


def test_all_code_cells_parse():
    notebook, _ = _source()
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])))


def test_reuses_complete_quality_passing_vector_rollup():
    _, source = _source()
    assert "participant_visit_embeddings.parquet" in source
    assert "n_embedded_images" in source
    assert "vectors are recalculated here" in source


def test_reads_only_supplied_epigenetic_variables_in_chunks():
    _, source = _source()
    for variable in (
        "DNAmAge_COM",
        "AgeAccelerationDifference_COM",
        "AgeAccelerationResidual_COM",
        "IEAA_COM",
        "EEAA_COM",
        "Hannum_Age_COM",
    ):
        assert variable in source
    assert "chunksize=baseline_chunk_size" in source
    assert "baseline_epigenetic_phenotypes_private.parquet" in source


def test_models_are_grouped_oof_and_resumable():
    _, source = _source()
    assert "GroupKFold" in source
    assert "train_age_head(" in source
    assert "write_metadata=False" in source
    assert "training_signature" in source
    assert "participant_grouped_oof" in source
    assert "retfound_chronological_age" in source
    assert "retfound_epigenetic_dnam_age" in source
    assert "retfound_epigenetic_hannum_age" in source


def test_three_way_primary_analysis_is_not_circular():
    _, source = _source()
    assert "retinal_age_oof" in source
    assert "primary_retinal_age_model_target" in source
    assert '"chronological_age"' in source
    assert "three_age_paired_distance_tests.csv" in source
    assert "supports_closer_after_fdr" in source
    assert "wilcoxon_one_sided_p" in source
    assert "signflip_one_sided_p" in source


def test_demographic_and_age_range_analyses_are_present():
    _, source = _source()
    assert "racial_background" in source
    assert "ethnicity_spirometry_labeled" in source
    assert "sex_labeled" in source
    assert "incremental_r2" in source
    assert "demographic_incremental_models.csv" in source
    assert "demographic_adjusted_coefficients.csv" in source
    assert "retfound_heads_demographic_performance.csv" in source
    assert "figure_4d_demographic_incremental_r2.png" in source
    assert "Central 80%" in source
    assert "Lower 10% age tail" in source
    assert "Upper 10% age tail" in source
    assert "chronological_model_5y_age_bin_performance.csv" in source


def test_demographic_formula_inputs_are_statsmodels_safe():
    _, source = _source()
    assert "formula_categoricals" in source
    assert "pd.Categorical(" in source
    assert '"error_message": str(error)[:500]' in source
    assert "Adjusted demographic models failed" in source
    assert "demographic_model_failures.csv" in source


def test_publication_figure_layout_is_not_clipped():
    _, source = _source()
    assert 'context="notebook"' in source
    assert 'layout="constrained"' in source
    assert "pad_inches=0.25" in source
    assert 'xlabel="Mean absolute age difference (years; 95% CI)"' in source
    assert "figure_5b_three_age_pairwise_projections.png" in source
    assert "set_box_aspect" in source


def test_age_tail_findings_are_calibration_guarded():
    _, source = _source()
    assert "retinal_model_mean_error" in source
    assert "tail_calibration_warning" in source
    assert "publication_support_flag" in source
    assert "calibration_sensitive_tail_findings" in source


def test_standard_and_three_dimensional_figures_are_saved():
    _, source = _source()
    for filename in (
        "figure_1_oof_model_performance.png",
        "figure_2_chronological_age_operating_range.png",
        "figure_3_epigenetic_vs_chronological_age.png",
        "figure_5_three_age_3d.png",
        "figure_6_three_age_absolute_distances.png",
        "figure_7_patient_three_clock_phenotypes.png",
        "figure_8_clock_comorbidity_association_forest.png",
        "figure_9_comorbidity_incremental_auc.png",
        "figure_10_longitudinal_retinal_aging.png",
    ):
        assert filename in source


def test_patient_level_clock_phenotypes_are_cross_fitted():
    _, source = _source()
    assert "cross_fitted_spline_residual" in source
    assert "SplineTransformer" in source
    assert "z_retinal_acceleration" in source
    assert "z_epigenetic_mean_acceleration" in source
    assert "shared_acceleration_mean" in source
    assert "retina_specific_discordance" in source
    assert "three_clock_dispersion" in source
    assert "three_clock_pca_loadings.csv" in source
    assert "patient_three_clock_phenotypes_private.parquet" in source
    assert "extreme_discordance_explainability_cohort_private.parquet" in source


def test_comorbidity_models_are_mutually_adjusted_and_multiplicity_corrected():
    _, source = _source()
    assert "available_binary_comorbidities" in source
    assert "mutually_adjusted_clocks" in source
    assert "orthogonal_three_clock_components" in source
    assert "direct_clock_discordance" in source
    assert "z_three_clock_dispersion" in source
    assert "fit.t_test(" in source
    assert "- z_epigenetic_mean_acceleration = 0" in source
    assert "patient_clock_comorbidity_associations.csv" in source
    assert "multimorbidity_clock_associations.csv" in source
    assert "benjamini_hochberg" in source


def test_incremental_prediction_is_out_of_fold():
    _, source = _source()
    assert "StratifiedKFold" in source
    assert "cross_val_predict" in source
    assert 'method="predict_proba"' in source
    assert "delta_auc_vs_base" in source
    assert "brier_improvement_vs_base" in source
    assert "comorbidity_oof_predictions_private.parquet" in source


def test_longitudinal_analysis_controls_baseline_retinal_acceleration():
    _, source = _source()
    assert "retinal_acceleration_bl" in source
    assert "retinal_acceleration_f1" in source
    assert "followup_years" in source
    assert "baseline_epigenetic_to_followup_retinal_associations.csv" in source
    assert "reliability_capability.json" in source
    assert "eye_level_reliability_available" in source
