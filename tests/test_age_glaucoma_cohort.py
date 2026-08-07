from pathlib import Path
import importlib.util
import json
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from age_glaucoma_cohort import greedy_age_match


class AgeGlaucomaCohortTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas not installed")
    def test_matches_without_reusing_participants(self) -> None:
        import pandas as pd

        cases = pd.DataFrame(
            {"patient_id": ["Z1", "Z2"], "age": [70.0, 70.4]}
        )
        controls = pd.DataFrame(
            {
                "participant_id": ["C1", "C2", "C3"],
                "visit": ["BL", "F1", "BL"],
                "age_at_fundus_years": [70.1, 70.5, 75.0],
            }
        )
        pairs, audit = greedy_age_match(cases, controls, caliper_years=1.0)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs["clsa_participant_id"].nunique(), 2)
        self.assertTrue(audit["matched"].all())

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas not installed")
    def test_ratio_matching_does_not_use_both_visits_of_one_control(self) -> None:
        import pandas as pd

        cases = pd.DataFrame({"patient_id": ["Z1"], "age": [70.0]})
        controls = pd.DataFrame(
            {
                "participant_id": ["C1", "C1", "C2"],
                "visit": ["BL", "F1", "BL"],
                "age_at_fundus_years": [70.0, 70.1, 70.2],
            }
        )
        pairs, _ = greedy_age_match(
            cases, controls, ratio=2, caliper_years=1.0
        )
        self.assertEqual(set(pairs["clsa_participant_id"]), {"C1", "C2"})

    def test_notebook_contract(self) -> None:
        path = REPOSITORY_ROOT / "Age_Glaucoma" / "01_build_age_matched_cohort.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertNotIn("run_zeiss_clsa_quality", source)
        self.assertIn("zeiss_source_pipeline_pass", source)
        self.assertIn("valid_1024_element_embedding_in_completed_chunk", source)
        self.assertIn("greedy_age_match", source)
        self.assertIn("screen_complete", source)
        self.assertIn("quality_pass", source)
        self.assertIn("age_match_pairs", source)
        self.assertIn("load_completed_clsa_embeddings", source)
        self.assertIn("load_completed_clsa_quality", source)
        self.assertIn("completed_batch_parquets_cached_as_delta", source)
        self.assertIn("PARQUET_COLUMN_DATA_TYPE_MISMATCH", source)
        self.assertIn('F.col("embedding").cast("array<float>")', source)
        self.assertNotIn("dbutils.widgets.set", source)


if __name__ == "__main__":
    unittest.main()
