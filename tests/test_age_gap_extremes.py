import sys
from pathlib import Path
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from age_gap_extremes import (
    attribution_physiology_metrics,
    benjamini_hochberg,
    build_participant_extremes,
    fundus_physiology_proxies,
    paired_permutation_patch_comparison,
    permutation_patch_comparison,
)


class AgeGapExtremeTests(unittest.TestCase):
    def test_bh_adjustment_is_monotonic_in_rank(self) -> None:
        adjusted = benjamini_hochberg([0.01, 0.04, 0.03, np.nan])
        self.assertTrue(np.isnan(adjusted[-1]))
        self.assertTrue(np.all((adjusted[:3] >= 0) & (adjusted[:3] <= 1)))

    def test_participant_deciles_do_not_duplicate_eyes(self) -> None:
        import pandas as pd

        rows = []
        for cohort in ("CLSA healthy", "Zeiss glaucoma"):
            for participant in range(20):
                for eye in ("R", "L"):
                    rows.append(
                        {
                            "cohort": cohort,
                            "participant_id": f"{cohort}-{participant}",
                            "image_path": f"/{cohort}/{participant}/{eye}.jpg",
                            "eye": eye,
                            "age": 50 + participant,
                            "age_gap": participant - 10,
                            "sex": "F" if participant % 2 else "M",
                        }
                    )
        participant, representative, thresholds = build_participant_extremes(
            pd.DataFrame(rows), quantile=0.10
        )
        self.assertEqual(len(participant), 40)
        self.assertEqual(len(representative), 40)
        self.assertEqual(len(thresholds), 2)
        self.assertFalse(
            representative.duplicated(["cohort", "participant_id"]).any()
        )

    def test_max_t_detects_large_patch_effect(self) -> None:
        rng = np.random.default_rng(7)
        bottom = rng.normal(size=(20, 4, 4))
        top = rng.normal(size=(20, 4, 4))
        top[:, 1, 2] += 4
        result = permutation_patch_comparison(
            top, bottom, n_permutations=200, random_state=7
        )
        self.assertLess(result["p_fwer"][1, 2], 0.05)

    def test_physiology_proxy_metrics_are_finite(self) -> None:
        size = 128
        yy, xx = np.indices((size, size))
        retina = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 < 50**2
        image = np.zeros((size, size, 3), dtype=np.uint8)
        image[retina] = [130, 80, 50]
        image[60:64, 20:108, 1] = 10
        proxies = fundus_physiology_proxies(image)
        metrics = attribution_physiology_metrics(np.ones((8, 8)), proxies)
        self.assertTrue(np.isfinite(metrics["vessel_proxy_attribution_mass"]))
        self.assertGreater(proxies["retina"].sum(), 0)

    def test_paired_sign_flip_detects_consistent_effect(self) -> None:
        rng = np.random.default_rng(9)
        bottom = rng.normal(size=(16, 3, 3))
        top = bottom + rng.normal(scale=0.1, size=(16, 3, 3))
        top[:, 0, 1] += 2
        result = paired_permutation_patch_comparison(
            top, bottom, n_permutations=200, random_state=9
        )
        self.assertLess(result["p_fwer"][0, 1], 0.05)


if __name__ == "__main__":
    unittest.main()
