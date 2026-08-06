from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class CorrelationNotebookContractTests(unittest.TestCase):
    def test_notebook_uses_participant_visit_aggregation_and_grouped_cv(self) -> None:
        notebook = (
            REPOSITORY_ROOT / "notebooks" / "correlation.py"
        ).read_text(encoding="utf-8")
        module = (
            REPOSITORY_ROOT / "src" / "correlation_analysis.py"
        ).read_text(encoding="utf-8")

        self.assertIn('groupBy("participant_id", "visit")', notebook)
        self.assertIn("retinal_age_gap", notebook)
        self.assertIn("COMORBIDITY_OUTCOMES", notebook)
        self.assertIn('dbutils.widgets.text("expected_embedding_dim", "1024")', notebook)
        self.assertIn("retfound_embedding", module)
        self.assertIn("GroupKFold", module)
        self.assertIn("train_groups & test_groups", module)
        self.assertIn("p_value_fdr_bh", module)
        self.assertNotIn("dbutils.widgets.set", notebook)


if __name__ == "__main__":
    unittest.main()
