import numpy as np
import pandas as pd
import pytest

from fundus_retfound_pipeline import RETFoundConfig, extract_retfound_embeddings
from retfound_fairness import (
    benjamini_hochberg,
    classify_racial_background,
    fairness_metric_table,
    match_group_to_reference,
    matched_outcome_contrasts,
    pool_age_predictions_to_participants,
)


def test_racial_background_preserves_multiple_response_status():
    frame = pd.DataFrame(
        {
            # The baseline ZIP reader uses dtype="string".  Mapping those
            # values produces NumPy boolean scalars rather than Python bools.
            "white": pd.Series(["1", "0", "1", None], dtype="string"),
            "black": pd.Series(["0", "1", "1", None], dtype="string"),
            "south_asian": pd.Series(["0", "0", "0", None], dtype="string"),
        }
    )
    classified = classify_racial_background(
        frame,
        {"white": "White", "black": "Black", "south_asian": "South Asian"},
    )
    assert classified["racial_background"].tolist()[:3] == [
        "White",
        "Black",
        "Multiple groups",
    ]
    assert pd.isna(classified.loc[3, "racial_background"])
    assert classified.loc[2, "racial_background_selection_count"] == 2


def test_pool_predictions_uses_baseline_and_one_participant_row():
    frame = pd.DataFrame(
        {
            "participant_id": ["A", "A", "A", "B"],
            "visit": ["BL", "BL", "F1", "F1"],
            "age": [60, 60, 63, 70],
            "retinal_age_prediction_oof": [61, 63, 64, 68],
            "sex": ["Female"] * 3 + ["Male"],
        }
    )
    pooled = pool_age_predictions_to_participants(frame, carry_columns=["sex"])
    assert pooled["participant_id"].is_unique
    participant_a = pooled.set_index("participant_id").loc["A"]
    assert participant_a["visit"] == "BL"
    assert participant_a["retinal_age_prediction_oof"] == 62
    assert participant_a["retinal_age_gap_oof"] == 2


def _matching_frame():
    return pd.DataFrame(
        {
            "participant_id": ["T1", "T2", "W1", "W2", "W3", "W4"],
            "racial_background": ["Black", "Black", "White", "White", "White", "White"],
            "age": [60.0, 70.0, 60.4, 59.5, 70.2, 71.0],
            "sex_at_birth": ["Female", "Male", "Female", "Female", "Male", "Male"],
            "diabetes": [1, 0, 1, 0, 0, 1],
            "hypertension": [1, 1, 1, 0, 1, 1],
            "retinal_age_gap_oof": [3.0, 4.0, 1.0, 0.0, 1.0, 2.0],
            "absolute_error_oof": [3.0, 4.0, 1.0, 0.0, 1.0, 2.0],
        }
    )


def test_matching_obeys_age_sex_and_no_reuse():
    frame = _matching_frame()
    pairs, audit, membership = match_group_to_reference(
        frame,
        "Black",
        age_caliper_years=1.0,
        ratio=2,
        exact_columns=["sex_at_birth"],
        distance_columns=["diabetes", "hypertension"],
    )
    assert audit["matched"].all()
    assert pairs["reference_participant_id"].is_unique
    assert pairs["absolute_age_difference_years"].le(1).all()
    assert set(membership["match_role"]) == {"target", "reference"}


def test_matched_contrast_is_target_minus_reference():
    frame = _matching_frame()
    _, _, membership = match_group_to_reference(
        frame,
        "Black",
        age_caliper_years=1.0,
        ratio=1,
        exact_columns=["sex_at_birth"],
        distance_columns=["diabetes", "hypertension"],
    )
    result = matched_outcome_contrasts(
        frame,
        membership,
        outcomes=["retinal_age_gap_oof"],
        bootstrap_repetitions=100,
        random_state=1,
    )
    assert result.loc[0, "target_minus_reference"] > 0
    assert result.loc[0, "n_matched_sets"] == 2


def test_fairness_metrics_reject_repeated_participants():
    repeated = pd.DataFrame(
        {
            "participant_id": ["A", "A"],
            "race": ["White", "White"],
            "age": [60, 60],
            "prediction": [61, 62],
        }
    )
    try:
        fairness_metric_table(
            repeated,
            "race",
            prediction_column="prediction",
            bootstrap_repetitions=0,
        )
    except ValueError as error:
        assert "one row per participant" in str(error)
    else:
        raise AssertionError("Repeated participants were accepted")


def test_benjamini_hochberg_is_monotone_in_rank():
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, np.nan])
    assert np.allclose(adjusted[:3], [0.03, 0.04, 0.04])
    assert np.isnan(adjusted[3])


def test_all_failed_embedding_batch_still_writes_failure_ledger(tmp_path):
    quality = pd.DataFrame(
        {
            "image_path": [str(tmp_path / "missing.jpeg")],
            "quality_pass": [True],
        }
    )
    with pytest.raises(RuntimeError, match="Every image failed"):
        extract_retfound_embeddings(
            quality,
            tmp_path / "embedding_batch",
            RETFoundConfig(device="cpu", allow_downloads=False, batch_size=1),
            model=object(),
            device="cpu",
            force=True,
        )
    failures = pd.read_csv(
        tmp_path / "embedding_batch" / "retfound_embedding_failures.csv"
    )
    assert failures["image_path"].tolist() == [str(tmp_path / "missing.jpeg")]
