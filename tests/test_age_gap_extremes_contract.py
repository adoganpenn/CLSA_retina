from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class AgeGapExtremesNotebookContractTests(unittest.TestCase):
    def test_notebook_is_participant_level_resumable_and_multiplicity_aware(self) -> None:
        source = (
            REPOSITORY_ROOT
            / "Age_Glaucoma"
            / "05_age_gap_extremes_explainability.py"
        ).read_text(encoding="utf-8")

        self.assertIn('%pip install "timm==1.0.28"', source)
        self.assertIn("Participant duplication remains", source)
        self.assertIn('"one_image_per_participant": True', source)
        self.assertIn("analysis_signature", source)
        self.assertIn("resume_batches", source)
        self.assertIn("permutation_patch_comparison", source)
        self.assertIn("paired_permutation_patch_comparison", source)
        self.assertIn("p_fwer", source)
        self.assertIn("p_fdr", source)
        self.assertIn("age_sex_matched_sensitivity", source)
        self.assertIn("hc3_extreme_effect", source)
        self.assertIn("physiology_proxy_age_sex_adjusted_hc3.csv", source)
        self.assertNotIn("dbutils.widgets.set", source)

    def test_notebook_has_anatomy_and_artifact_guardrails(self) -> None:
        source = (
            REPOSITORY_ROOT
            / "Age_Glaucoma"
            / "05_age_gap_extremes_explainability.py"
        ).read_text(encoding="utf-8")

        self.assertIn("fundus_physiology_proxies", source)
        self.assertIn("vessel_proxy_enrichment", source)
        self.assertIn("optic_disc_proxy_enrichment", source)
        self.assertIn("border_proxy_enrichment", source)
        self.assertIn("background_attribution_fraction", source)
        self.assertIn("not validated clinical segmentations", source)
        self.assertIn("figure_05_04_image_heatmap_physiology", source)
        self.assertIn("source_model", source)


if __name__ == "__main__":
    unittest.main()
