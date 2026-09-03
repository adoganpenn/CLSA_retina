import ast
import json
from pathlib import Path


REPO_NOTEBOOK = (
    Path(__file__).parents[1]
    / "Age_Glaucoma"
    / "Algorithm Fairness"
    / "01_retfound_age_fairness.ipynb"
)
LOCAL_NOTEBOOK = Path(__file__).with_name("01_retfound_age_fairness.ipynb")
NOTEBOOK = LOCAL_NOTEBOOK if LOCAL_NOTEBOOK.exists() else REPO_NOTEBOOK


def _source():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return notebook, "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )


def test_notebook_code_cells_parse():
    notebook, _ = _source()
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if source.lstrip().startswith("%"):
            continue
        ast.parse(source)


def test_notebook_uses_completed_quality_and_embedding_batches():
    _, source = _source()
    assert "fundus_quality_manifest.parquet" in source
    assert 'glob("batch_*/retfound_embeddings.parquet")' in source
    assert "participant_visit_rollup.parquet" in source
    assert "resume_completed_outputs = True" in source


def test_full_manifest_is_denominator_and_missing_images_are_processed():
    _, source = _source()
    assert 'full_image_manifest_path = derived_root / "fundus_image_manifest"' in source
    assert "process_missing_full_manifest_images = True" in source
    assert "retry_prior_embedding_failures = True" in source
    assert "missing_quality_paths" in source
    assert "run_quality_pipeline(" in source
    assert "missing_embedding_paths" in source
    assert "extract_retfound_embeddings(" in source
    assert "full_image_coverage.parquet" in source


def test_model_is_grouped_oof_and_race_blind():
    _, source = _source()
    assert "train_age_head(" in source
    assert '"race_used_as_model_input": False' in source
    assert "retinal_age_prediction_oof" in source
    assert "CLSA_full_cohort_age_head.joblib" in source
    assert 'training.duplicated(["participant_id", "visit"])' in source
    assert "pool_age_predictions_to_participants(" in source
    assert "n_training_images_represented" in source


def test_race_source_and_interpretation_are_explicit():
    _, source = _source()
    assert "SDC_CULT_BL_COM" in source
    assert '"Black"' in source
    assert "African genetic ancestry" in source
    assert "should not be relabeled `African`" in source
    assert "Multiple groups" in source
    assert "inherently biased" in source
    assert "raw_any_selection" in source


def test_primary_and_sensitivity_matching_are_present():
    _, source = _source()
    assert "primary_age_caliper_years = 1.0" in source
    assert "sensitivity_age_caliper_years = 2.0" in source
    assert "match_group_to_reference(" in source
    assert "standardized_mean_differences(" in source
    assert "p_value_fdr" in source


def test_private_and_aggregate_outputs_are_separated():
    _, source = _source()
    assert 'private_root = output_root / "01_private"' in source
    assert 'statistics_root = output_root / "04_statistics"' in source
    assert "race_match_pairs_private.parquet" in source
    assert "demographic_age_performance.parquet" in source
    assert "RUN_README.md" in source


def test_all_available_specific_groups_use_primary_matched_white_cohorts():
    _, source = _source()
    assert "race_matched_all_available_age_heads" in source
    assert 'caliper_analysis"].astype(str).eq("primary_1y")' in source
    assert 'excluded_nonspecific_groups = {"White", "Multiple groups", "Other"}' in source
    assert "minimum_all_available_n = minimum_inference_group_n" in source
    assert "common_age_band_counts" not in source
    assert "race_matched_all_available_image_selection_private.parquet" in source


def test_all_available_heads_are_cross_fitted_by_participant_and_power_audited():
    _, source = _source()
    assert "KFold(" in source
    assert "population_oof_prediction" in source
    assert "target_head_minus_matched_white_head_mae" in source
    assert "mde_80_percent_power_years" in source
    assert "target_minus_white_calibration_slope" in source
    assert "slope_mde_80_percent_power" in source


def test_secondary_demographic_figure_uses_dictionary_labels():
    _, source = _source()
    assert "dictionary-labeled secondary demographic fairness panels" in source
    assert '"F": "Female", "2": "Female"' in source
    assert '"M": "Male", "1": "Male"' in source
    assert "Less than secondary school graduation" in source
    assert "Secondary graduation or some post-secondary" in source
    assert "Post-secondary degree/diploma" in source
    assert "Less than $20,000" in source
    assert "$150,000 or more" in source
    assert "secondary_demographic_age_performance_labeled.parquet" in source
    assert "np.maximum(point - np.minimum(ci_low, point), 0.0)" in source
