from pathlib import Path
import json
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from fundus_retfound_pipeline import _json_safe


class AgeGlaucomaModelContractTests(unittest.TestCase):
    def test_runtime_plan_metrics_are_json_safe(self) -> None:
        class PlanMetrics:
            def __str__(self) -> str:
                return "runtime metrics"

        safe = _json_safe({"oof_metrics": PlanMetrics()})
        encoded = json.dumps(safe)
        self.assertIn("PlanMetrics", encoded)
        self.assertIn("runtime metrics", encoded)

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
        self.assertIn("recovered_partial_training", source)
        self.assertIn("metadata-serialization failure", source)
        self.assertIn("importlib.reload", source)
        self.assertIn("write_metadata=False", source)
        self.assertIn("metadata writing is disabled", source)
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
