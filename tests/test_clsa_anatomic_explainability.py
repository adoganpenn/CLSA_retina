import unittest

import numpy as np
import pandas as pd

from src.clsa_anatomic_explainability import (
    attribution_region_metrics,
    build_anatomic_masks,
    disc_fovea_affine_matrix,
    participant_permutation_inference,
    sample_translated_control_masks,
    select_probability_extremes,
)


class CLSAAnatomicExplainabilityTests(unittest.TestCase):
    def test_build_masks_distinguishes_localized_rois_and_vessels(self):
        shape = (100, 100)
        retina = np.ones(shape, dtype=bool)
        vessels = np.zeros(shape, dtype=float)
        vessels[:, 48:51] = 1.0
        masks, metadata = build_anatomic_masks(
            shape,
            (30, 50, 70, 50),
            vessels,
            retina,
            minimum_vessel_fraction=0.001,
        )
        self.assertTrue(metadata["anatomy_valid"])
        self.assertIn("not pixel segmentation", metadata["optic_disc_definition"])
        self.assertGreater(masks["optic_disc_roi"].sum(), 100)
        self.assertGreater(masks["vessels"].sum(), 0)
        self.assertFalse(
            np.any(masks["vessels_elsewhere"] & masks["optic_disc_roi"])
        )

    def test_attribution_metrics_detect_region_enrichment(self):
        grid = np.zeros((10, 10), dtype=float)
        grid[4:7, 6:9] = 3.0
        retina = np.ones((100, 100), dtype=bool)
        region = np.zeros((100, 100), dtype=bool)
        region[40:70, 60:90] = True
        metrics = attribution_region_metrics(
            grid,
            retina,
            {"optic_disc_roi": region},
        )
        self.assertGreater(metrics["optic_disc_roi_positive_enrichment"], 1.0)

    def test_translated_controls_preserve_most_target_area(self):
        retina = np.ones((100, 100), dtype=bool)
        target = np.zeros((100, 100), dtype=bool)
        target[40:60, 45:55] = True
        excluded = np.zeros_like(target)
        excluded[30:70, 35:65] = True
        controls = sample_translated_control_masks(
            target,
            retina,
            excluded,
            n_masks=5,
        )
        self.assertEqual(len(controls), 5)
        self.assertTrue(all(mask.sum() >= 0.8 * target.sum() for mask in controls))
        self.assertTrue(all(not np.any(mask & excluded) for mask in controls))

    def test_permutation_inference_returns_adjusted_p_values(self):
        rng = np.random.default_rng(3)
        rows = []
        for label in (0, 1):
            for index in range(30):
                rows.append(
                    {
                        "participant_id": f"p{label}_{index}",
                        "glaucoma_label": label,
                        "optic": rng.normal(label * 1.5, 0.5),
                        "vessel": rng.normal(0, 1),
                    }
                )
        result = participant_permutation_inference(
            pd.DataFrame(rows),
            ["optic", "vessel"],
            permutations=200,
            bootstrap_repetitions=200,
        )
        optic = result.set_index("metric").loc["optic"]
        self.assertGreater(optic["glaucoma_minus_healthy"], 1.0)
        self.assertLess(optic["permutation_p_max_t"], 0.05)

    def test_disc_fovea_registration_maps_to_canonical_axis(self):
        import cv2

        matrix = disc_fovea_affine_matrix(
            (30, 50, 70, 50), output_size=100
        )
        transformed = cv2.transform(
            np.asarray([[[70, 50], [30, 50]]], dtype=np.float32), matrix
        )[0]
        np.testing.assert_allclose(transformed[0], [28, 50], atol=1e-4)
        np.testing.assert_allclose(transformed[1], [72, 50], atol=1e-4)

    def test_probability_extremes_are_participant_level_and_disjoint(self):
        frame = pd.DataFrame(
            {
                "participant_id": [f"p{index:02d}" for index in range(20)],
                "glaucoma_probability_oof": np.linspace(0.01, 0.99, 20),
            }
        )
        result = select_probability_extremes(frame, fraction=0.10)
        self.assertEqual(len(result), 4)
        self.assertEqual(result["participant_id"].nunique(), 4)
        self.assertEqual(
            set(result["confidence_extreme"]),
            {"bottom_healthy_like", "top_glaucoma_like"},
        )
        self.assertLess(
            result.loc[
                result["confidence_extreme_code"] == 0,
                "glaucoma_probability_oof",
            ].max(),
            result.loc[
                result["confidence_extreme_code"] == 1,
                "glaucoma_probability_oof",
            ].min(),
        )


if __name__ == "__main__":
    unittest.main()
