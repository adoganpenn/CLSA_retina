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

    def test_image_link_is_participant_and_visit_specific(self) -> None:
        self.assertIn(
            '["participant_id", "visit"]',
            self.source,
        )
        self.assertIn("exact_fundus_capture_timestamp_available", self.source)
        self.assertNotIn(".cache()", self.source)


if __name__ == "__main__":
    unittest.main()
