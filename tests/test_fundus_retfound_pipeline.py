from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.fundus_retfound_pipeline import (
    QualityConfig,
    _apply_calibration,
    _decompose_layer_norm_mean_pool,
    preprocess_fundus,
    read_embedding_failure_paths,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class FundusQualityTests(unittest.TestCase):
    def test_retinal_foreground_is_cropped_and_resized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "synthetic_fundus.png"
            size = 512
            y, x = np.ogrid[:size, :size]
            circle = (x - size / 2) ** 2 + (y - size / 2) ** 2 <= 210**2
            image = np.zeros((size, size, 3), dtype=np.uint8)
            gradient = np.linspace(40, 190, size, dtype=np.uint8)
            image[..., 0][circle] = np.broadcast_to(gradient, (size, size))[circle]
            image[..., 1][circle] = 80
            image[..., 2][circle] = 45
            Image.fromarray(image).save(path)

            result = preprocess_fundus(path, QualityConfig())

            self.assertEqual(result.image.size, (256, 256))
            self.assertGreater(result.retina_fraction, 0.40)
            self.assertLess(result.retina_fraction, 0.70)
            self.assertGreater(result.contrast_std, 5.0)
            self.assertGreaterEqual(result.crop_x0, 0)
            self.assertLess(result.crop_x1, size)

    def test_empty_embedding_failure_logs_mean_no_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            newline_only = root / "newline_only.csv"
            newline_only.write_text("\n", encoding="utf-8")
            self.assertEqual(
                read_embedding_failure_paths(newline_only),
                set(),
            )

            header_only = root / "header_only.csv"
            header_only.write_text(
                "image_path,error\n",
                encoding="utf-8",
            )
            self.assertEqual(
                read_embedding_failure_paths(header_only),
                set(),
            )


class CalibrationAndAttributionTests(unittest.TestCase):
    def test_intercept_calibration(self) -> None:
        prediction = np.array([50.0, 60.0])
        calibrated = _apply_calibration(
            prediction,
            {
                "mode": "intercept",
                "mean_y": 65.0,
                "mean_prediction": 55.0,
            },
        )
        np.testing.assert_allclose(calibrated, [60.0, 70.0])

    def test_exact_patch_contributions_sum_to_prediction(self) -> None:
        rng = np.random.default_rng(20260727)
        patch_tokens = rng.normal(size=(196, 8))
        gamma = rng.normal(size=8)
        beta = rng.normal(size=8)
        coefficients = rng.normal(size=8)
        intercept = 3.5

        result = _decompose_layer_norm_mean_pool(
            patch_tokens,
            gamma,
            beta,
            1e-6,
            coefficients,
            intercept,
        )

        self.assertLess(result["reconstruction_error"], 1e-10)
        self.assertAlmostEqual(
            result["additive_contributions"].sum(),
            result["prediction_from_feature"],
            places=10,
        )


class RETFoundNotebookContractTests(unittest.TestCase):
    def test_full_run_flag_uses_all_visits_without_balancing(self) -> None:
        notebook = (
            REPOSITORY_ROOT / "notebooks" / "02_run_fundus_retfound.py"
        ).read_text(encoding="utf-8")

        self.assertIn('"run_all_images", "false"', notebook)
        self.assertIn('"pipeline_batch_size", "500"', notebook)
        self.assertIn('"resume_batches", "true"', notebook)
        self.assertIn("if run_all_images:", notebook)
        self.assertIn("max_images = 0", notebook)
        self.assertIn('manifest_spark.groupBy("visit")', notebook)
        self.assertIn("counts are intentionally allowed to differ", notebook)
        self.assertIn("quality_batches_root", notebook)
        self.assertIn("embedding_batches_root", notebook)
        self.assertIn("resumed", notebook)
        self.assertIn('cast("array<float>")', notebook)
        self.assertIn('.option("overwriteSchema", "true")', notebook)


if __name__ == "__main__":
    unittest.main()
