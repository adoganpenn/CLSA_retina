import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Age_Glaucoma" / "09_clsa_anatomic_explainability.py"


class CLSAAnatomicExplainabilityContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = NOTEBOOK.read_text(encoding="utf-8")
        ast.parse(cls.source)

    def test_toolbox_is_pinned_and_models_are_used(self):
        self.assertIn("d7757e28fbf639856b53cfe00019f605af8c1f17", self.source)
        self.assertIn("fit.load_fovea_od_model", self.source)
        self.assertIn("fit.load_segmentation_ensemble", self.source)
        self.assertIn("fit.ensemble_predict_segmentation", self.source)

    def test_databricks_uses_one_pinned_headless_opencv(self):
        self.assertIn("%pip uninstall -y opencv-python", self.source)
        self.assertIn('opencv-python-headless==4.11.0.86', self.source)
        self.assertIn("dbutils.library.restartPython()", self.source)
        self.assertIn("OpenCV headless smoke test", self.source)

    def test_anatomy_batches_report_stage_and_durable_progress(self):
        self.assertIn("Anatomy plan:", self.source)
        self.assertIn("stage=landmarks", self.source)
        self.assertIn("stage=vessels", self.source)
        self.assertIn("stage=artifacts", self.source)
        self.assertIn("progress={completed_images:,}", self.source)
        self.assertIn("write_frame(batch_manifest, local_manifest_path)", self.source)
        self.assertIn(
            "publish_local_artifact(local_manifest_path, manifest_path)",
            self.source,
        )
        self.assertIn("def publish_local_artifact", self.source)
        self.assertIn("local_mask_path", self.source)
        self.assertIn("local_overlay_path", self.source)
        self.assertIn("artifacts_ready", self.source)

    def test_scope_is_clsa_only(self):
        self.assertIn('attribution_manifest["source"].astype(str) == "CLSA"', self.source)
        self.assertIn('"scope": "CLSA only; Zeiss excluded"', self.source)
        self.assertNotIn("prepare_zeiss_dicom_input", self.source)

    def test_anatomic_outputs_are_not_overclaimed(self):
        self.assertIn("not pixel-level disc", self.source)
        self.assertIn("circular ROI around localized center; not segmentation", self.source)
        self.assertIn("vessels_elsewhere_positive_enrichment", self.source)
        self.assertIn("peripapillary_annulus", self.source)
        self.assertIn("fovea_roi", self.source)

    def test_inference_is_participant_level_and_multiplicity_controlled(self):
        self.assertIn("participant_permutation_inference", self.source)
        self.assertIn("permutation_p_max_t", self.source)
        self.assertIn("participant_anatomy", self.source)

    def test_targeted_occlusion_is_optional_and_strictly_replayed(self):
        self.assertIn("run_targeted_occlusion", self.source)
        self.assertIn("occlusion_stored_logit_replay_error", self.source)
        self.assertIn("sample_translated_control_masks", self.source)
        self.assertIn("sample_equal_area_control_masks", self.source)

    def test_credentials_are_temporary_and_widgets_are_not_set(self):
        self.assertNotIn("dbutils.widgets.set", self.source)
        self.assertIn('os.environ.pop("HF_TOKEN", None)', self.source)
        self.assertIn('dbutils.widgets.remove("hf_token")', self.source)

    def test_resume_preserves_notebook08_image_identity(self):
        self.assertIn('"image_key": str(record["image_key"])', self.source)
        self.assertIn('"artifact_key": key', self.source)
        self.assertIn('set(existing["image_key"].astype(str))', self.source)


if __name__ == "__main__":
    unittest.main()
