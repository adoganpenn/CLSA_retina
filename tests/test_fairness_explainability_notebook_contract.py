import json
from pathlib import Path


LOCAL_NOTEBOOK = Path(__file__).with_name("02_age_model_anatomic_explainability.ipynb")
REPO_NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "Age_Glaucoma"
    / "Algorithm Fairness"
    / "02_age_model_anatomic_explainability.ipynb"
)
NOTEBOOK = LOCAL_NOTEBOOK if LOCAL_NOTEBOOK.exists() else REPO_NOTEBOOK


def source_text():
    notebook = json.loads(NOTEBOOK.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_notebook_uses_all_available_matched_selection_and_completed_models():
    source = source_text()
    assert "race_matched_all_available_image_selection_private.parquet" in source
    assert "comparison_group" in source
    assert "population_role" in source
    assert "CLSA_full_cohort_age_head.joblib" in source
    assert "race_matched_all_available_age_heads" in source
    assert "matched_white" in source


def test_notebook_is_batched_resumable_and_exactly_paired():
    source = source_text()
    assert "segmentation_batch_size = 4" in source
    assert "resume_batches = True" in source
    assert "exact_multihead_patch_contributions" in source
    assert "expected_pairs" in source
    assert "paired_model_inference" in source
    assert "record_key" in source
    assert "target_head_vs_matched_white_head_on_target_images.csv" in source
    assert "target_vs_matched_white_population_anatomic_inference.csv" in source


def test_notebook_preserves_anatomic_claim_boundaries():
    source = source_text()
    assert "FR-U-Net pixel segmentation" in source
    assert "not segmentations" in source
    assert "latent features, not named physiology" in source
    assert "disc_fovea_affine_matrix" in source


def test_notebook_does_not_hardcode_hugging_face_secret():
    source = source_text()
    assert "dbutils.widgets.text(\"hf_token\"" in source
    assert "hf_ekdo" not in source
