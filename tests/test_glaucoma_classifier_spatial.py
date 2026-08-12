import unittest

import numpy as np
import pandas as pd

from src.glaucoma_classifier_spatial import (
    classifier_metrics,
    fit_oof_linear_classifier,
    match_controls_ratio,
    optic_nerve_masks,
    paired_auc_difference,
    predict_linear_head,
    regional_attribution_metrics,
    sample_equal_area_control_masks,
)


class GlaucomaClassifierSpatialTests(unittest.TestCase):
    def test_ratio_matching_is_unique_and_respects_exact_fields(self) -> None:
        cases = pd.DataFrame(
            {
                "participant_id": ["g1", "g2"],
                "age": [60.0, 70.0],
                "sex_normalized": ["F", "M"],
                "visit": ["BL", "F1"],
            }
        )
        controls = pd.DataFrame(
            {
                "participant_id": ["c1", "c2", "c3", "c4", "c5"],
                "age": [60.1, 59.9, 70.2, 69.7, 60.0],
                "sex_normalized": ["F", "F", "M", "M", "M"],
                "visit": ["BL", "BL", "F1", "F1", "BL"],
            }
        )
        matches = match_controls_ratio(cases, controls, ratio=2, caliper_years=0.5)
        self.assertEqual(len(matches), 4)
        self.assertFalse(matches["control_id"].duplicated().any())
        self.assertTrue((matches["absolute_age_difference"] <= 0.5).all())

    def test_nested_oof_classifier_recovers_embedding_signal(self) -> None:
        rng = np.random.default_rng(8)
        rows = []
        dimension = 8
        for label in (0, 1):
            for index in range(60):
                vector = rng.normal(0, 0.8, dimension)
                vector[:3] += label * 1.4
                rows.append(
                    {
                        "participant_id": f"p{label}_{index}",
                        "glaucoma_label": label,
                        "embedding": vector,
                    }
                )
        frame = pd.DataFrame(rows)
        oof, heads, final_head = fit_oof_linear_classifier(
            frame,
            folds=4,
            inner_folds=3,
            c_grid=(0.01, 0.1),
            expected_dim=dimension,
        )
        metrics = classifier_metrics(
            oof["glaucoma_label"],
            oof["glaucoma_probability_oof"],
            bootstrap_repetitions=200,
        )
        self.assertEqual(len(heads), 4)
        self.assertGreater(metrics["auroc"], 0.9)
        predicted = predict_linear_head(frame.head(5), final_head)
        self.assertTrue(predicted["glaucoma_probability"].between(0, 1).all())

    def test_optic_nerve_masks_and_regional_metrics(self) -> None:
        masks = optic_nerve_masks(
            (100, 100),
            center_x_fraction=0.25,
            center_y_fraction=0.5,
            radius_fraction=0.08,
        )
        self.assertGreater(masks["optic_disc"].sum(), 150)
        self.assertGreater(
            masks["peripapillary_annulus"].sum(), masks["optic_disc"].sum()
        )
        grid = np.zeros((10, 10), dtype=float)
        grid[4:7, 1:4] = 5.0
        metrics = regional_attribution_metrics(
            grid,
            np.ones((100, 100), dtype=bool),
            masks,
        )
        self.assertGreater(metrics["optic_disc_positive_enrichment"], 1.0)
        controls = sample_equal_area_control_masks(
            np.ones((100, 100), dtype=bool),
            masks["optic_disc_plus_peripapillary"],
            target_area=int(masks["optic_disc"].sum()),
            n_masks=5,
        )
        self.assertEqual(len(controls), 5)
        self.assertTrue(
            all(not np.any(mask & masks["optic_disc_plus_peripapillary"]) for mask in controls)
        )

    def test_paired_auc_difference_uses_identical_participants(self) -> None:
        reference = pd.DataFrame(
            {
                "participant_id": [f"p{index}" for index in range(12)],
                "glaucoma_label": [0] * 6 + [1] * 6,
                "reference": np.linspace(0.05, 0.95, 12),
            }
        )
        comparator = reference[["participant_id", "glaucoma_label"]].copy()
        comparator["comparator"] = 0.5
        result = paired_auc_difference(
            reference,
            comparator,
            reference_probability="reference",
            comparator_probability="comparator",
            bootstrap_repetitions=100,
        )
        self.assertGreater(result["auroc_difference"], 0.4)
        self.assertEqual(result["n_participants"], 12)


if __name__ == "__main__":
    unittest.main()
