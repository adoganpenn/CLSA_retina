import unittest

import numpy as np
import pandas as pd

from src.three_cohort_glaucoma import (
    adjusted_group_effect,
    apply_source_harmonizer,
    canonical_sex,
    cross_validated_domain_auc,
    crossfit_source_harmonizer,
    embedding_shift_summary,
    fit_additive_source_harmonizer,
    greedy_match,
    nested_harmonization_domain_auc,
    paired_outcome_effect,
    validate_embedding_frame,
)


class ThreeCohortGlaucomaTests(unittest.TestCase):
    def test_canonical_sex_handles_released_and_string_codes(self) -> None:
        self.assertEqual(canonical_sex("Female"), "F")
        self.assertEqual(canonical_sex("1"), "M")
        self.assertEqual(canonical_sex(None), "MISSING")

    def test_embedding_validation_preserves_pre_normalized_sex(self) -> None:
        frame = pd.DataFrame(
            {
                "participant_id": ["p1", "p2"],
                "age": [60, 61],
                "sex_normalized": ["F", "M"],
                "embedding": [np.ones(3), np.ones(3) * 2],
            }
        )
        validated = validate_embedding_frame(frame, "test", expected_dim=3)
        self.assertEqual(validated["sex_normalized"].tolist(), ["F", "M"])

        frame["sex"] = [None, "Male"]
        validated = validate_embedding_frame(frame, "test", expected_dim=3)
        self.assertEqual(validated["sex_normalized"].tolist(), ["F", "M"])

    def test_harmonizer_removes_known_source_shift_and_preserves_clsa(self) -> None:
        rng = np.random.default_rng(11)
        rows = []
        dimension = 8
        disease_effect = np.linspace(0.2, 0.8, dimension)
        source_effect = np.linspace(1.0, 2.0, dimension)
        for source, glaucoma, count in (
            ("CLSA", 0, 120),
            ("CLSA", 1, 80),
            ("Zeiss", 1, 90),
        ):
            for index in range(count):
                age = 55 + rng.normal(0, 8)
                vector = (
                    rng.normal(0, 0.15, dimension)
                    + glaucoma * disease_effect
                    + (source == "Zeiss") * source_effect
                    + (age - 55) * 0.01
                )
                rows.append(
                    {
                        "participant_id": f"{source}_{glaucoma}_{index}",
                        "age": age,
                        "sex": "F" if index % 2 else "M",
                        "sex_normalized": "F" if index % 2 else "M",
                        "source": source,
                        "glaucoma": glaucoma,
                        "embedding": vector,
                    }
                )
        frame = pd.DataFrame(rows)
        bundle = fit_additive_source_harmonizer(frame, expected_dim=dimension)
        corrected = apply_source_harmonizer(frame, bundle, mode="location")
        clsa_original = np.stack(
            frame.loc[frame["source"] == "CLSA", "embedding"]
        )
        clsa_corrected = np.stack(
            corrected.loc[corrected["source"] == "CLSA", "embedding"]
        )
        np.testing.assert_allclose(clsa_original, clsa_corrected, atol=1e-6)

        before = embedding_shift_summary(
            frame[(frame["source"] == "CLSA") & (frame["glaucoma"] == 1)],
            frame[frame["source"] == "Zeiss"],
        )
        after = embedding_shift_summary(
            corrected[
                (corrected["source"] == "CLSA")
                & (corrected["glaucoma"] == 1)
            ],
            corrected[corrected["source"] == "Zeiss"],
        )
        self.assertGreater(before["median_absolute_feature_smd"], 3)
        self.assertLess(after["median_absolute_feature_smd"], 0.2)

    def test_harmonizer_omits_sex_when_target_sex_is_entirely_missing(self) -> None:
        rng = np.random.default_rng(21)
        rows = []
        for source, glaucoma in (("CLSA", 0), ("CLSA", 1), ("Zeiss", 1)):
            for index in range(30):
                rows.append(
                    {
                        "participant_id": f"{source}_{glaucoma}_{index}",
                        "age": 60 + rng.normal(),
                        "sex": None if source == "Zeiss" else ("F" if index % 2 else "M"),
                        "sex_normalized": "MISSING" if source == "Zeiss" else ("F" if index % 2 else "M"),
                        "source": source,
                        "glaucoma": glaucoma,
                        "embedding": rng.normal(size=4) + (source == "Zeiss"),
                    }
                )
        bundle = fit_additive_source_harmonizer(
            pd.DataFrame(rows), expected_dim=4
        )
        self.assertFalse(
            any(name.startswith("sex=") for name in bundle["design_columns"])
        )

    def test_crossfit_harmonizer_never_uses_evaluation_records(self) -> None:
        rng = np.random.default_rng(27)
        rows = []
        dimension = 6
        for source, glaucoma in (("CLSA", 0), ("CLSA", 1), ("Zeiss", 1)):
            for index in range(24):
                rows.append(
                    {
                        "participant_id": f"{source}_{glaucoma}_{index}",
                        "age": 60 + rng.normal(0, 5),
                        "sex_normalized": "F" if index % 2 else "M",
                        "source": source,
                        "glaucoma": glaucoma,
                        "embedding": (
                            rng.normal(0, 0.2, dimension)
                            + glaucoma * 0.3
                            + (source == "Zeiss") * 1.5
                        ),
                    }
                )
        frame = pd.DataFrame(rows)
        corrected, bundles = crossfit_source_harmonizer(
            frame,
            folds=3,
            mode="location_scale",
            expected_dim=dimension,
        )
        self.assertEqual(len(bundles), 3)
        self.assertTrue(corrected["harmonization_cross_fitted"].all())
        self.assertEqual(set(corrected["harmonization_fold"]), {0, 1, 2})
        clsa = frame[frame["source"] == "CLSA"].reset_index(drop=True)
        corrected_clsa = corrected[corrected["source"] == "CLSA"].reset_index(drop=True)
        np.testing.assert_allclose(
            np.stack(clsa["embedding"]),
            np.stack(corrected_clsa["embedding"]),
            atol=1e-6,
        )

    def test_domain_auc_is_direction_invariant(self) -> None:
        rng = np.random.default_rng(44)
        reference = pd.DataFrame(
            {"embedding": [row for row in rng.normal(0, 0.2, (80, 4))]}
        )
        target = pd.DataFrame(
            {"embedding": [row for row in rng.normal(2, 0.2, (80, 4))]}
        )
        result = cross_validated_domain_auc(
            reference,
            target,
            max_per_domain=100,
            folds=4,
        )
        self.assertGreaterEqual(result["domain_auc_mean"], 0.5)
        self.assertLessEqual(result["domain_auc_mean"], 1.0)
        self.assertIn("domain_auc_signed_mean", result)

    def test_nested_domain_auc_uses_outer_evaluation_folds(self) -> None:
        rng = np.random.default_rng(45)
        rows = []
        for source, glaucoma in (("CLSA", 0), ("CLSA", 1), ("Zeiss", 1)):
            for index in range(30):
                rows.append(
                    {
                        "participant_id": f"{source}_{glaucoma}_{index}",
                        "age": 62 + rng.normal(0, 4),
                        "sex_normalized": "F" if index % 2 else "M",
                        "source": source,
                        "glaucoma": glaucoma,
                        "embedding": (
                            rng.normal(0, 0.3, 5)
                            + glaucoma * 0.2
                            + (source == "Zeiss") * 1.0
                        ),
                    }
                )
        result = nested_harmonization_domain_auc(
            pd.DataFrame(rows),
            mode="location_scale",
            folds=3,
            expected_dim=5,
            max_per_source=100,
        )
        self.assertEqual(len(result["fold_results"]), 3)
        self.assertGreaterEqual(result["domain_auc_mean"], 0.5)
        self.assertEqual(
            result["evaluation_design"],
            "outer-fold harmonizer and classifier evaluation",
        )

    def test_adjusted_group_effect_recovers_exposure_difference(self) -> None:
        rng = np.random.default_rng(12)
        n = 600
        age = rng.normal(65, 8, n)
        exposed = rng.integers(0, 2, n)
        outcome = 4.0 * exposed + 0.25 * (age - 65) + rng.normal(0, 1, n)
        frame = pd.DataFrame(
            {
                "age": age,
                "exposed": exposed,
                "sex_normalized": np.where(np.arange(n) % 2, "F", "M"),
                "outcome": outcome,
            }
        )
        result = adjusted_group_effect(frame, outcome_column="outcome")
        self.assertAlmostEqual(result["adjusted_difference"], 4.0, delta=0.2)
        self.assertGreater(result["ci_95_low"], 3.5)

    def test_matching_is_unique_within_caliper_and_paired_effect(self) -> None:
        exposed = pd.DataFrame(
            {
                "participant_id": ["e1", "e2", "e3"],
                "age": [60.0, 62.0, 80.0],
                "sex_normalized": ["F", "M", "F"],
                "gap": [5.0, 6.0, 9.0],
            }
        )
        reference = pd.DataFrame(
            {
                "participant_id": ["r1", "r2", "r3"],
                "age": [60.2, 62.4, 75.0],
                "sex_normalized": ["F", "M", "F"],
                "gap": [1.0, 2.0, 3.0],
            }
        )
        pairs = greedy_match(exposed, reference, caliper_years=1.0)
        self.assertEqual(len(pairs), 2)
        self.assertFalse(pairs["reference_id"].duplicated().any())
        result, analysis = paired_outcome_effect(
            pairs,
            exposed,
            reference,
            outcome_column="gap",
            bootstrap_repetitions=200,
        )
        self.assertEqual(result["n_pairs"], 2)
        self.assertAlmostEqual(result["mean_difference"], 4.0)
        self.assertEqual(len(analysis), 2)


if __name__ == "__main__":
    unittest.main()
