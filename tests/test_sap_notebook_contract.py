import ast
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    REPOSITORY_ROOT / "notebooks" / "03_build_sap_analysis_dataset.py"
)


def literal_assignment(tree: ast.AST, name: str):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Assignment was not found: {name}")


class SapNotebookContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = NOTEBOOK_PATH.read_text()
        cls.tree = ast.parse(cls.source)

    def test_visit_specific_age_sources_are_explicit(self) -> None:
        metadata = literal_assignment(self.tree, "VISIT_METADATA")
        self.assertEqual(metadata["BL"]["age_source"], "AGE_NMBR_COM")
        self.assertEqual(metadata["F1"]["age_source"], "AGE_NMBR_COF1")

    def test_actual_strata_and_epigenetic_fields_are_preserved(self) -> None:
        baseline = literal_assignment(self.tree, "BASELINE_MAP")
        metadata = literal_assignment(self.tree, "VISIT_METADATA")
        self.assertEqual(baseline["sampling_strata"], "GEOSTRATA_COM")
        self.assertEqual(metadata["BL"]["epigenetic_dnam"], "DNAmAge_COM")
        self.assertEqual(
            metadata["BL"]["epigenetic_hannum"],
            "Hannum_Age_COM",
        )
        expected_epigenetic_sources = {
            "epigenetic_dnam_age": "DNAmAge_COM",
            "epigenetic_age_acceleration_difference": (
                "AgeAccelerationDifference_COM"
            ),
            "epigenetic_age_acceleration_residual": (
                "AgeAccelerationResidual_COM"
            ),
            "epigenetic_ieaa": "IEAA_COM",
            "epigenetic_eeaa": "EEAA_COM",
            "epigenetic_hannum_age": "Hannum_Age_COM",
        }
        epigenetic_variables = literal_assignment(
            self.tree,
            "EPIGENETIC_BASELINE_VARIABLES",
        )
        self.assertEqual(
            {
                name: specification["source_column"]
                for name, specification in epigenetic_variables.items()
            },
            expected_epigenetic_sources,
        )

    def test_epigenetic_pipeline_has_provenance_qc_and_fundus_linkage(self) -> None:
        for output_name in (
            "sap_epigenetic_variable_dictionary",
            "sap_epigenetic_baseline",
            "sap_epigenetic_baseline_qc",
            "sap_epigenetic_formula_qc",
            "sap_fundus_epigenetic_linkage_audit",
            "sap_fundus_epigenetic_analysis",
        ):
            self.assertIn(output_name, self.source)
        self.assertIn(
            "epigenetic_age_acceleration_difference_recomputed",
            self.source,
        )
        self.assertIn(
            "epigenetic_difference_release_minus_recomputed",
            self.source,
        )
        self.assertIn("CLSA_NUMERIC_MISSING_CODES", self.source)
        self.assertIn("no raw DNA files are required", self.source)

    def test_image_link_is_participant_and_visit_specific(self) -> None:
        self.assertIn(
            '["participant_id", "visit"]',
            self.source,
        )
        self.assertIn("exact_fundus_capture_timestamp_available", self.source)
        self.assertIn("sap_fundus_image_linkage_audit", self.source)
        self.assertIn("sap_fundus_image_exclusions", self.source)
        self.assertIn(
            'F.col("image_age_link_status") == "visit_age_matched"',
            self.source,
        )
        self.assertNotIn(
            "Some fundus images have an unparsed participant ID",
            self.source,
        )
        self.assertNotIn(".cache()", self.source)

    def test_questionnaire_csv_and_duplicates_are_handled_safely(self) -> None:
        self.assertIn('.option("multiLine", "true")', self.source)
        self.assertIn("exact_duplicate_rows_removed", self.source)
        self.assertIn("sap_questionnaire_duplicate_conflicts", self.source)
        self.assertIn(r'rlike(r"^\d{7}$")', self.source)


if __name__ == "__main__":
    unittest.main()
