import unittest

import numpy as np
import pandas as pd

from src.questionnaire_epigenetic_analysis import (
    age_measure_agreement,
    benjamini_hochberg,
    compare_questionnaire_groups,
    correlate_age_accelerations,
    lins_concordance,
    questionnaire_group_descriptives,
)


class QuestionnaireEpigeneticAnalysisTests(unittest.TestCase):
    def test_bh_preserves_missing_values(self):
        adjusted = benjamini_hochberg([0.01, np.nan, 0.04])
        self.assertAlmostEqual(adjusted[0], 0.02)
        self.assertTrue(np.isnan(adjusted[1]))
        self.assertAlmostEqual(adjusted[2], 0.04)

    def test_questionnaire_comparison_detects_binary_difference(self):
        rng = np.random.default_rng(4)
        rows = []
        for index in range(120):
            label = index >= 60
            rows.append(
                {
                    "participant_id": f"p{index}",
                    "match_set_id": f"m{index // 3}",
                    "glaucoma_label": int(label),
                    "age": rng.normal(65, 5),
                    "sex_at_birth": "1" if index % 2 else "2",
                    "visit": "BL",
                    "hypertension": int(rng.random() < (0.80 if label else 0.20)),
                    "score": rng.normal(2 if label else 0, 1),
                }
            )
        results, missingness = compare_questionnaire_groups(
            pd.DataFrame(rows),
            {
                "hypertension": {"type": "binary", "label": "Hypertension"},
                "score": {"type": "continuous", "label": "Score"},
            },
        )
        indexed = results.set_index("variable")
        self.assertLess(indexed.loc["hypertension", "fdr_q_value"], 0.05)
        self.assertGreater(
            indexed.loc["hypertension", "raw_difference_glaucoma_minus_healthy"],
            0.3,
        )
        self.assertEqual(len(missingness), 2)

    def test_age_agreement_returns_difference_and_concordance(self):
        frame = pd.DataFrame(
            {
                "glaucoma_label": [0] * 15 + [1] * 15,
                "retinal_age": np.arange(30, 60, dtype=float),
                "chronological_age": np.arange(30, 60, dtype=float) - 2,
            }
        )
        result = age_measure_agreement(
            frame,
            retinal_age_column="retinal_age",
            comparator_columns={"chronological_age": "Chronological age"},
        )
        overall = result[result["stratum"] == "all"].iloc[0]
        self.assertAlmostEqual(
            overall["mean_difference_retinal_minus_comparator"], 2.0
        )
        self.assertGreater(overall["lins_concordance"], 0.95)

    def test_age_gap_correlations_are_stratified(self):
        values = np.arange(30, dtype=float)
        frame = pd.DataFrame(
            {
                "glaucoma_label": [0] * 15 + [1] * 15,
                "retinal_gap": values,
                "ieaa": values * 0.5,
            }
        )
        result = correlate_age_accelerations(
            frame,
            retinal_gap_column="retinal_gap",
            epigenetic_acceleration_columns={"ieaa": "IEAA"},
        )
        self.assertEqual(set(result["stratum"]), {"all", "healthy", "glaucoma"})
        self.assertTrue((result["pearson_r"] > 0.99).all())

    def test_lins_concordance_is_one_for_identical_values(self):
        values = np.arange(10, dtype=float)
        self.assertAlmostEqual(lins_concordance(values, values), 1.0)

    def test_questionnaire_descriptives_report_category_fractions(self):
        frame = pd.DataFrame(
            {
                "glaucoma_label": [0, 0, 1, 1],
                "smoking": ["never", "current", "never", "never"],
            }
        )
        result = questionnaire_group_descriptives(
            frame,
            {"smoking": {"type": "categorical", "label": "Smoking"}},
        )
        glaucoma_never = result[
            (result["glaucoma_label"] == 1) & (result["level"] == "never")
        ].iloc[0]
        self.assertEqual(glaucoma_never["level_count"], 2)
        self.assertEqual(glaucoma_never["level_fraction_among_observed"], 1.0)


if __name__ == "__main__":
    unittest.main()
