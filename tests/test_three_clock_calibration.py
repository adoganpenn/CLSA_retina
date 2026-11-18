import ast
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from three_clock_calibration import (  # noqa: E402
    CalibrationConfig,
    age_bin_performance,
    cross_fitted_calibrate,
    metric_table,
    paired_clock_comparisons,
    prepare_three_clock_frame,
)


def synthetic_frame(n=250, seed=7):
    rng = np.random.default_rng(seed)
    age = rng.uniform(45, 85, n)
    return pd.DataFrame(
        {
            "participant_id": [f"P{index:04d}" for index in range(n)],
            "chronological_age": age,
            "retinal_age_raw_oof": 10 + 0.80 * age + rng.normal(0, 3.0, n),
            "retinal_age_oof": 8 + 0.82 * age + rng.normal(0, 3.0, n),
            "epigenetic_dnam_age": age - 5 + rng.normal(0, 5.0, n),
            "epigenetic_hannum_age": 1.10 * age - 5 + rng.normal(0, 4.0, n),
            "cv_fold": np.arange(n) % 5,
            "sex_labeled": np.where(np.arange(n) % 2, "Female", "Male"),
        }
    )


def test_prepare_prefers_raw_retinal_and_common_complete_cohort():
    source = synthetic_frame()
    source.loc[0, "epigenetic_dnam_age"] = np.nan
    analysis, flow = prepare_three_clock_frame(source)
    assert len(analysis) == len(source) - 1
    assert flow["retinal_input_column"].iloc[0] == "retinal_age_raw_oof"
    assert bool(flow["retinal_input_is_raw_oof"].iloc[0])
    assert analysis["participant_id"].is_unique
    assert analysis["calibration_fold"].nunique() == 5


def test_crossfit_audit_excludes_held_out_fold_and_is_deterministic():
    analysis, _ = prepare_three_clock_frame(synthetic_frame())
    config = CalibrationConfig(seed=123)
    first, audit = cross_fitted_calibrate(
        analysis, "retinal_age_input", config=config
    )
    second, _ = cross_fitted_calibrate(
        analysis, "retinal_age_input", config=config
    )
    np.testing.assert_allclose(first, second)
    assert np.isfinite(first).all()
    for row in audit.itertuples(index=False):
        assert str(row.held_out_fold) not in row.training_folds.split("|")
        assert row.n_train + row.n_test == len(analysis)


def test_symmetric_calibration_improves_offset_and_slope_on_synthetic_data():
    analysis, _ = prepare_three_clock_frame(synthetic_frame(n=500))
    raw_columns = {
        "Retinal": "retinal_age_input",
        "Horvath": "horvath_age_input",
        "Hannum": "hannum_age_input",
    }
    calibrated_columns = {}
    for clock, column in raw_columns.items():
        output, _ = cross_fitted_calibrate(analysis, column)
        output_column = f"{clock.lower()}_calibrated"
        analysis[output_column] = output
        calibrated_columns[clock] = output_column
    raw = metric_table(analysis, raw_columns, stage="raw").set_index("clock")
    calibrated = metric_table(
        analysis, calibrated_columns, stage="calibrated"
    ).set_index("clock")
    for clock in raw_columns:
        assert abs(calibrated.loc[clock, "mean_error"]) < abs(
            raw.loc[clock, "mean_error"]
        )
        assert calibrated.loc[clock, "mae"] < raw.loc[clock, "mae"]


def test_paired_comparison_uses_identical_participants_and_expected_direction():
    rng = np.random.default_rng(11)
    age = np.linspace(45, 85, 180)
    frame = pd.DataFrame(
        {
            "chronological_age": age,
            "better": age + rng.normal(0, 1, len(age)),
            "worse": age + rng.normal(0, 5, len(age)),
            "middle": age + rng.normal(0, 3, len(age)),
        }
    )
    result = paired_clock_comparisons(
        frame,
        {"Better": "better", "Worse": "worse", "Middle": "middle"},
        stage="test",
        repetitions=200,
        seed=17,
    )
    comparison = result[
        result["first_clock"].eq("Better")
        & result["second_clock"].eq("Worse")
    ].iloc[0]
    assert comparison["n"] == len(frame)
    assert comparison["delta_mae"] < 0
    assert comparison["delta_mae_ci_high"] < 0
    assert comparison["mae_difference_direction"] == "first_clock_lower_mae"


def test_age_bins_have_all_three_clocks_when_large_enough():
    analysis, _ = prepare_three_clock_frame(synthetic_frame(n=300))
    for clock, input_column in {
        "Retinal": "retinal_age_input",
        "Horvath": "horvath_age_input",
        "Hannum": "hannum_age_input",
    }.items():
        analysis[f"{clock}_cf"] = cross_fitted_calibrate(
            analysis, input_column
        )[0]
    result = age_bin_performance(
        analysis,
        {clock: f"{clock}_cf" for clock in ("Retinal", "Horvath", "Hannum")},
        minimum_n=20,
    )
    counts = result.groupby("age_bin_5y")["clock"].nunique()
    assert not result.empty
    assert counts.eq(3).all()


def test_databricks_notebook_source_parses_and_contains_required_outputs():
    notebook = REPO_ROOT / "notebooks" / "08_three_clock_symmetric_calibration.py"
    source = notebook.read_text(encoding="utf-8")
    ast.parse(source)
    for required in (
        "cross_fitted_calibrate",
        "three_clock_calibration_metric_intervals.csv",
        "three_clock_paired_performance_comparisons.csv",
        "three_clock_calibration_fold_audit.csv",
        "three_clock_calibrated_subgroup_performance.csv",
        "three_clock_calibration_manifest.json",
        "predictive_r2_definition",
        "upstream_training_provenance_is_symmetric",
    ):
        assert required in source
