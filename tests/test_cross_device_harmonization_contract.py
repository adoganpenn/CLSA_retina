from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class CrossDeviceHarmonizationNotebookContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (
            REPOSITORY_ROOT
            / "Age_Glaucoma"
            / "07_cross_device_harmonization_validation.py"
        ).read_text(encoding="utf-8")

    def test_notebook_reuses_locked_embeddings_and_model(self) -> None:
        self.assertIn("RETFound inference performed by this notebook: false", self.source)
        self.assertIn("CLSA_healthy", self.source)
        self.assertIn("clsa_healthy_participants.parquet", self.source)
        self.assertIn("zeiss_glaucoma_participants.parquet", self.source)
        self.assertNotIn("load_retfound_model", self.source)
        self.assertNotIn("extract_retfound_embeddings", self.source)
        self.assertNotIn("dbutils.widgets.set", self.source)

    def test_notebook_crossfits_and_nests_domain_validation(self) -> None:
        self.assertIn("crossfit_source_harmonizer", self.source)
        self.assertIn("nested_harmonization_domain_auc", self.source)
        self.assertIn('modes = ("location", "location_scale")', self.source)
        self.assertIn("maximum_acceptable_domain_auc", self.source)
        self.assertIn("maximum_acceptable_p90_feature_smd", self.source)
        self.assertIn("passes_source_removal_gate", self.source)

    def test_notebook_preserves_primary_analysis_hierarchy(self) -> None:
        self.assertIn("within-CLSA glaucoma comparison", self.source)
        self.assertIn("external_result_is_confirmatory", self.source)
        self.assertIn("source-by-glaucoma interaction", self.source)
        self.assertIn("paired-device images are absent", self.source)


if __name__ == "__main__":
    unittest.main()
