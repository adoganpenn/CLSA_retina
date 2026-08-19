import sys
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from two_group_retinal_pipeline import (  # noqa: E402
    MatchConfig,
    aggregate_participant_embeddings,
    batch_ranges,
    fit_grouped_oof_classifier,
    matched_set_permutation_inference,
    match_participants,
)


def test_batch_ranges_cover_every_row_once():
    assert batch_ranges(0, 500) == []
    assert batch_ranges(1001, 500) == [(0, 500), (500, 1000), (1000, 1001)]


def test_matching_is_participant_level_and_without_reuse():
    participants = pd.DataFrame(
        {
            "participant_id": ["a", "b", "c", "d", "e", "f"],
            "age": [60, 70, 60.2, 60.4, 69.7, 71],
            "group_label": [1, 1, 0, 0, 0, 0],
            "sex": ["F", "M", "F", "F", "M", "M"],
        }
    )
    pairs, audit, membership = match_participants(
        participants,
        MatchConfig(ratio=2, caliper_years=2, exact_columns=("sex",)),
    )
    assert len(pairs) == 4
    assert pairs["control_id"].nunique() == 4
    assert audit["matched"].all()
    assert membership["participant_id"].nunique() == 6
    assert membership.groupby("match_set_id")["group_label"].sum().eq(1).all()


def test_participant_embedding_mean_and_grouped_cv_do_not_leak_sets():
    rng = np.random.default_rng(7)
    image_rows = []
    for match_index in range(12):
        for label in (0, 1):
            participant = f"p{match_index}_{label}"
            for eye in ("L", "R"):
                vector = rng.normal(size=8) + label * 0.4
                image_rows.append(
                    {
                        "participant_id": participant,
                        "embedding": vector,
                        "group_label": label,
                        "age": 50 + match_index,
                        "match_set_id": f"m{match_index}",
                        "eye": eye,
                    }
                )
    participant = aggregate_participant_embeddings(
        pd.DataFrame(image_rows), expected_dim=8
    )
    assert len(participant) == 24
    assert participant["n_embedded_images"].eq(2).all()
    predictions, heads, final_head = fit_grouped_oof_classifier(
        participant,
        folds=4,
        inner_folds=3,
        expected_dim=8,
        c_grid=(0.01, 0.1),
    )
    assert predictions["group_b_probability_oof"].between(0, 1).all()
    assert len(heads) == 4
    assert final_head["embedding_dim"] == 8
    assert (
        predictions.groupby("match_set_id")["fold"].nunique().eq(1).all()
    )


def test_matched_permutation_returns_adjusted_results():
    frame = pd.DataFrame(
        {
            "match_set_id": np.repeat(["a", "b", "c", "d"], 2),
            "group_label": np.tile([0, 1], 4),
            "metric": [0, 1, 0, 2, 1, 2, 0, 3],
        }
    )
    result = matched_set_permutation_inference(
        frame,
        ["metric"],
        permutations=99,
        bootstrap_repetitions=99,
    )
    assert result.loc[0, "group_b_minus_group_a"] == 1.75
    assert 0 <= result.loc[0, "max_t_adjusted_p_value"] <= 1
