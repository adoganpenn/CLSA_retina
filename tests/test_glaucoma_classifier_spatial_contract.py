import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Age_Glaucoma" / "08_glaucoma_classifier_spatial_validation.py"


class GlaucomaClassifierSpatialContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = NOTEBOOK.read_text(encoding="utf-8")
        ast.parse(cls.source)

    def test_reuses_embeddings_and_participant_held_out_classifier(self):
        self.assertIn("fit_oof_linear_classifier", self.source)
        self.assertIn("paired_auc_difference", self.source)
        self.assertIn("participant_predictions", self.source)
        self.assertIn("classifier_logit_oof", self.source)
        self.assertIn("retfound_embeddings_recalculated_for_training", self.source)
        self.assertNotIn("load_age_head", self.source)

    def test_locked_head_is_transported_to_zeiss_without_claiming_specificity(self):
        self.assertIn("Zeiss_glaucoma_external_predictions", self.source)
        self.assertIn("zeiss_participant_scores", self.source)
        self.assertIn("they do not estimate specificity", self.source)

    def test_spatial_analysis_has_exact_replay_and_negative_controls(self):
        for required in (
            "exact_linear_patch_map_from_array",
            "linear_head_score_from_array",
            "optic_nerve_masks",
            "sample_equal_area_control_masks",
            "disc_specific_occlusion_drop",
            "patch_reconstruction_error",
            "anatomic_claim_ready",
        ):
            self.assertIn(required, self.source)

    def test_validated_anatomy_is_required_for_claim(self):
        self.assertIn("require_validated_disc_for_claim", self.source)
        self.assertIn("validated_annotation_coverage", self.source)
        self.assertIn("automatic bright-disc proxies remain exploratory", self.source)

    def test_notebook_never_programmatically_sets_widgets(self):
        self.assertNotIn("dbutils.widgets.set", self.source)

    def test_hugging_face_token_widget_is_temporary(self):
        self.assertIn('dbutils.widgets.text("hf_token"', self.source)
        self.assertIn('os.environ["HF_TOKEN"] = temporary_hf_token', self.source)
        self.assertIn('os.environ.pop("HF_TOKEN", None)', self.source)
        self.assertIn('dbutils.widgets.remove("hf_token")', self.source)


if __name__ == "__main__":
    unittest.main()
