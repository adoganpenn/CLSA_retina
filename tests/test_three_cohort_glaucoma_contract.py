from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ThreeCohortGlaucomaNotebookContractTests(unittest.TestCase):
    def test_notebook_uses_same_domain_primary_and_oof_healthy_predictions(self) -> None:
        source = (
            REPOSITORY_ROOT
            / "Age_Glaucoma"
            / "06_three_cohort_glaucoma_analysis.py"
        ).read_text(encoding="utf-8")

        self.assertIn("CLSA glaucoma versus CLSA healthy", source)
        self.assertIn("primary disease comparison", source.lower())
        self.assertIn("CLSA_healthy_participant_visit_oof.parquet", source)
        self.assertIn("exclude_age_model_training_overlap", source)
        self.assertIn("strict_never_glaucoma_controls", source)
        self.assertIn("glaucoma_only_ocular", source)
        self.assertIn("require_complete_glaucoma_embeddings", source)
        self.assertNotIn("dbutils.widgets.set", source)

    def test_notebook_reuses_completed_embeddings_without_inference(self) -> None:
        source = (
            REPOSITORY_ROOT
            / "Age_Glaucoma"
            / "06_three_cohort_glaucoma_analysis.py"
        ).read_text(encoding="utf-8")

        self.assertIn("load_existing_clsa_embeddings", source)
        self.assertIn("completed_batches_cached_as_delta", source)
        self.assertIn("RETFound inference performed by this notebook: false", source)
        self.assertNotIn("load_retfound_model", source)
        self.assertNotIn("extract_retfound_embeddings", source)

    def test_notebook_has_triangular_harmonization_guardrails(self) -> None:
        source = (
            REPOSITORY_ROOT
            / "Age_Glaucoma"
            / "06_three_cohort_glaucoma_analysis.py"
        ).read_text(encoding="utf-8")

        self.assertIn("fit_additive_source_harmonizer", source)
        self.assertIn("cross_validated_domain_auc", source)
        self.assertIn("embedding_shift_summary", source)
        self.assertIn("source-by-disease interaction", source)
        self.assertIn("harmonized_sensitivity", source)
        self.assertIn("residual_domain_signal_flag_auc_gt_0_60", source)
        self.assertIn("A cross-sectional retinal-age gap", source)

    def test_notebook_writes_privacy_safe_summary_and_auditable_outputs(self) -> None:
        source = (
            REPOSITORY_ROOT
            / "Age_Glaucoma"
            / "06_three_cohort_glaucoma_analysis.py"
        ).read_text(encoding="utf-8")

        self.assertIn("clsa_glaucoma_only_embeddings_delta", source)
        self.assertIn("THREE_COHORT_GLAUCOMA_RESULTS.csv", source)
        self.assertIn("THREE_COHORT_GLAUCOMA_SUMMARY.json", source)
        self.assertIn("residual_domain_diagnostics.csv", source)
        self.assertNotIn('display(pairs)', source)
        self.assertNotIn('display(paired_rows)', source)


if __name__ == "__main__":
    unittest.main()
