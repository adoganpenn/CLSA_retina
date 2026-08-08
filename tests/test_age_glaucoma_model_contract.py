from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class AgeGlaucomaModelContractTests(unittest.TestCase):
    def test_training_notebook_uses_grouped_oof_and_frozen_model(self) -> None:
        source = (
            REPOSITORY_ROOT
            / "Age_Glaucoma"
            / "03_train_clsa_healthy_age_model.py"
        ).read_text(encoding="utf-8")

        self.assertIn("train_age_head", source)
        self.assertIn('"model_name": "CLSA_healthy"', source)
        self.assertIn('"frozen": True', source)
        self.assertIn("CLSA_healthy_oof_predictions.parquet", source)
        self.assertIn("exact_sex=False", source)
        self.assertIn("common_support_min_age", source)
        self.assertIn("matched_pair_level_analysis.parquet", source)
        self.assertIn("bootstrap_95_ci_low", source)
        self.assertIn("embedding_domain_shift_summary.csv", source)
        self.assertIn("median_absolute_feature_smd", source)
        self.assertNotIn("dbutils.widgets.set", source)

    def test_explainability_is_source_specific_and_reproduction_gated(self) -> None:
        source = (
            REPOSITORY_ROOT
            / "Age_Glaucoma"
            / "04_compare_matched_explainability.py"
        ).read_text(encoding="utf-8")

        self.assertIn("prepare_zeiss_dicom_input", source)
        self.assertIn("prepare_model_input", source)
        self.assertIn("embedding_cosine_threshold", source)
        self.assertIn("valid_for_group_comparison", source)
        self.assertIn("attribution_group_statistics", source)
        self.assertIn("patch_location_comparison.csv", source)
        self.assertIn("match_set_spatial_outliers.parquet", source)
        self.assertNotIn("train_age_head", source)
        self.assertNotIn("dbutils.widgets.set", source)


if __name__ == "__main__":
    unittest.main()
