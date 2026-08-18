import ast
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOKS = [
    ROOT / "Age_Glaucoma" / "02_debug_age_matching.py",
    ROOT / "Age_Glaucoma" / "03_train_clsa_healthy_age_model.py",
    ROOT / "Age_Glaucoma" / "04_compare_matched_explainability.py",
    ROOT / "Age_Glaucoma" / "05_age_gap_extremes_explainability.py",
    ROOT / "Age_Glaucoma" / "06_three_cohort_glaucoma_analysis.py",
    ROOT / "Age_Glaucoma" / "07_cross_device_harmonization_validation.py",
    ROOT / "Age_Glaucoma" / "08_glaucoma_classifier_spatial_validation.py",
    ROOT / "Age_Glaucoma" / "09_clsa_anatomic_explainability_source.py",
    ROOT / "Age_Glaucoma" / "10_questionnaire_epigenetic_aging_source.py",
    ROOT / "notebooks" / "01_build_clsa_dataset.py",
    ROOT / "notebooks" / "02_run_fundus_retfound.py",
    ROOT / "notebooks" / "03_build_sap_analysis_dataset.py",
    ROOT / "notebooks" / "correlation.py",
    ROOT / "notebooks" / "retfound_quality_passed.py",
]


class HardcodedNotebookConfigurationTests(unittest.TestCase):
    def test_python_notebooks_only_read_temporary_secrets_from_widgets(self):
        allowed_widget_reads = {"archive_password", "hf_token"}
        for path in SOURCE_NOTEBOOKS:
            with self.subTest(path=path.name):
                source = path.read_text(encoding="utf-8")
                ast.parse(source)
                widget_reads = set(
                    re.findall(r'dbutils\.widgets\.get\("([^"]+)"\)', source)
                )
                self.assertLessEqual(widget_reads, allowed_widget_reads)
                widget_definitions = set(
                    re.findall(
                        r'dbutils\.widgets\.(?:text|dropdown)\(\s*"([^"]+)"',
                        source,
                    )
                )
                self.assertLessEqual(widget_definitions, allowed_widget_reads)

    def test_canonical_workspace_and_volume_roots_are_fixed(self):
        canonical_repo = (
            "/Workspace/Users/ad0038@pennmedicine.upenn.edu/CLSA/CLSA_retina"
        )
        canonical_volume = (
            "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset"
        )
        for path in SOURCE_NOTEBOOKS:
            with self.subTest(path=path.name):
                source = path.read_text(encoding="utf-8")
                if path.name != "02_debug_age_matching.py":
                    self.assertIn(canonical_repo, source)
                self.assertIn(canonical_volume, source)

    def test_ipynb_only_notebooks_use_widgets_only_for_secrets(self):
        paths = [
            ROOT / "Age_Glaucoma" / "01_build_age_matched_cohort.ipynb",
            ROOT / "notebooks" / "smoketest.ipynb",
        ]
        for path in paths:
            with self.subTest(path=path.name):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                source = "\n".join(
                    "".join(cell.get("source", []))
                    for cell in notebook["cells"]
                )
                widget_reads = set(
                    re.findall(
                        r'dbutils\.widgets\.get\("([^"]+)"\)', source
                    )
                )
                self.assertLessEqual(
                    widget_reads,
                    {"archive_password", "hf_token"},
                )
                widget_definitions = set(
                    re.findall(
                        r'dbutils\.widgets\.(?:text|dropdown)\(\s*"([^"]+)"',
                        source,
                    )
                )
                self.assertLessEqual(
                    widget_definitions,
                    {"archive_password", "hf_token"},
                )
                self.assertIn(
                    "/Volumes/ophthalmology_analytics/dev_optic/",
                    source,
                )


if __name__ == "__main__":
    unittest.main()
