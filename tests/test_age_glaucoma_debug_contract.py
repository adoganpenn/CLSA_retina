from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class AgeGlaucomaDebugContractTests(unittest.TestCase):
    def test_debug_notebook_emits_only_aggregate_diagnostics(self) -> None:
        source = (
            REPOSITORY_ROOT / "Age_Glaucoma" / "02_debug_age_matching.py"
        ).read_text(encoding="utf-8")

        self.assertIn("condition_missingness_aggregates.csv", source)
        self.assertIn("clsa_control_attrition.csv", source)
        self.assertIn("caliper_feasibility.csv", source)
        self.assertIn("maximum_one_to_one_age_pairs", source)
        self.assertIn('"contains_patient_identifiers": False', source)
        self.assertIn("Do **not** share the original match audit", source)
        self.assertNotIn("display(zeiss_source", source)
        self.assertNotIn("display(clsa_eligible_images", source)
        self.assertNotIn("display(match_pairs", source)
        self.assertNotIn("display(match_audit", source)


if __name__ == "__main__":
    unittest.main()
