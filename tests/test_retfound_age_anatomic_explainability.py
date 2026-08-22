import importlib.util
from pathlib import Path

import numpy as np


LOCAL_MODULE = Path(__file__).with_name("retfound_age_anatomic_explainability.py")
REPO_MODULE = Path(__file__).resolve().parents[1] / "src" / "retfound_age_anatomic_explainability.py"
MODULE_PATH = LOCAL_MODULE if LOCAL_MODULE.exists() else REPO_MODULE
SPEC = importlib.util.spec_from_file_location("age_xai", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Estimator:
    coef_ = np.array([1.0, -2.0, 3.0])
    intercept_ = 4.0


def test_effective_head_global_intercept_calibration():
    coef, intercept = MODULE.effective_linear_head(
        {
            "estimator": Estimator(),
            "calibration": {
                "mode": "intercept",
                "mean_y": 70.0,
                "mean_prediction": 68.5,
            },
        }
    )
    np.testing.assert_allclose(coef, [1.0, -2.0, 3.0])
    assert intercept == 5.5


def test_effective_head_balanced_subgroup_offset():
    coef, intercept = MODULE.effective_linear_head(
        {"estimator": Estimator(), "calibration_offset": -1.25}
    )
    np.testing.assert_allclose(coef, [1.0, -2.0, 3.0])
    assert intercept == 2.75


def test_coefficient_similarity_identity_and_difference():
    frame = MODULE.coefficient_similarity_table(
        {
            "global": (np.array([1.0, 2.0, 3.0]), 0.0),
            "race::A": (np.array([1.0, -2.0, 3.0]), 0.0),
        },
        top_k=2,
    ).set_index("model")
    assert np.isclose(frame.loc["global", "coefficient_cosine"], 1.0)
    assert frame.loc["race::A", "coefficient_cosine"] < 1.0
    assert frame.loc["global", "top_2_jaccard"] == 1.0


def test_paired_inference_detects_large_paired_difference():
    import pandas as pd

    rows = []
    for index in range(20):
        for model, value in (("global", 1.0 + index * 0.001), ("race::A", 2.0 + index * 0.001)):
            rows.append(
                {
                    "participant_id": str(index),
                    "racial_background": "A",
                    "model_name": model,
                    "roi": value,
                }
            )
    result = MODULE.paired_model_inference(
        pd.DataFrame(rows),
        ["roi"],
        permutations=199,
        bootstrap_repetitions=199,
        random_state=1,
    )
    assert len(result) == 1
    assert np.isclose(result.loc[0, "subgroup_minus_global"], 1.0)
    assert result.loc[0, "signflip_p_max_t"] <= 0.02
