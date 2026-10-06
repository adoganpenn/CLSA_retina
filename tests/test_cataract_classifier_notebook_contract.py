"""Static contract checks for the standalone Databricks cataract analysis."""

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "05_retfound_cataract_classifier_and_questionnaire_sensitivity.py"
)


def source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_script_is_valid_python_and_databricks_source():
    text = source()
    compile(text, str(SCRIPT), "exec")
    assert text.startswith("# Databricks notebook source")
    assert "dbutils.widgets.text" in text


def test_classifier_label_is_exact_baseline_vis_field():
    text = source()
    assert 'VIS_LABEL_COLUMN = "VIS_CATRCT_COM"' in text
    assert "[VIS_LABEL_COLUMN],\n    eligible_ids=" in text
    assert '"fields_used_for_classifier_label": [VIS_LABEL_COLUMN]' in text
    assert '"ICQ_fields_used_for_classifier_label": []' in text
    assert 'embedding_source["visit"].eq("BL")' in text


def test_split_is_participant_disjoint_70_30_and_validation_is_locked():
    text = source()
    assert "train_fraction = 0.70" in text
    assert "validation_fraction = 0.30" in text
    assert "stratify=classifier_cohort[\"cataract_label\"]" in text
    assert "if development_ids & validation_ids" in text
    assert "validation_probability = final_model.predict_proba(x_validation)" in text
    assert "select_youden_threshold(\n    y_development" in text


def test_requested_metrics_and_questionnaire_definitions_are_present():
    text = source()
    for metric in (
        "roc_auc",
        "pr_auc_average_precision",
        "sensitivity_recall",
        "specificity",
        "ppv_precision",
        "npv",
        "balanced_accuracy",
        "brier_score",
        "calibration_slope",
    ):
        assert f'"{metric}"' in text
    assert '"current_combined_icq_or_vis": "cataract_current_combined"' in text
    assert '"strict_icq_only": "cataract_icq_only"' in text
    assert '"strict_vis_only": "cataract_vis_only"' in text

