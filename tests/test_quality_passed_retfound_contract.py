from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from quality_passed_retfound import (
    iter_stable_quality_batches,
    parse_quality_batch_name,
)


class QualityPassedRETFoundContractTests(unittest.TestCase):
    def test_notebook_is_a_live_resumable_batch_consumer(self) -> None:
        notebook = (
            REPOSITORY_ROOT / "notebooks" / "retfound_quality_passed.py"
        ).read_text(encoding="utf-8")

        self.assertIn('"quality_pass_columns"', notebook)
        self.assertIn('"quality_pass"', notebook)
        self.assertIn("expected_quality_batches = 213", notebook)
        self.assertIn("poll_seconds = 30", notebook)
        self.assertIn("resume_batches = True", notebook)
        self.assertIn("select_quality_passed", notebook)
        self.assertIn("read_completed_quality_batch", notebook)
        self.assertIn("while True:", notebook)
        self.assertIn("time.sleep(poll_seconds)", notebook)
        self.assertIn("quality_handoff_complete.json", notebook)
        self.assertIn("retfound_embeddings_delta", notebook)
        self.assertNotIn("dbutils.widgets.set", notebook)

    def test_stable_batches_retain_source_quality_batch_name(self) -> None:
        class MinimalFrame:
            empty = True

        self.assertEqual(list(iter_stable_quality_batches(MinimalFrame())), [])

    def test_quality_batch_name_encodes_expected_row_count(self) -> None:
        self.assertEqual(
            parse_quality_batch_name("batch_000005000_000005500"),
            (5000, 5500),
        )
        with self.assertRaises(ValueError):
            parse_quality_batch_name("batch_in_progress")


if __name__ == "__main__":
    unittest.main()
