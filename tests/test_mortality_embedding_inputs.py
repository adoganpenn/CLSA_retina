"""Exercise notebook input resolution without importing Databricks libraries."""

import ast
from pathlib import Path
import tempfile
import unittest


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "04_mortality_prediction_retfound_epigenetic.py"
)


class MortalityEmbeddingInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.embedding_root = self.root / "fundus_retfound/02_embeddings"
        self.rollup = (
            self.root
            / "Age_Glaucoma/16_algorithm_fairness/01_private/participant_visit_embeddings.parquet"
        )
        self.fairness = (
            self.root
            / "Age_Glaucoma/16_algorithm_fairness/00_full_image_pipeline/02_embedding_batches"
        )
        tree = ast.parse(NOTEBOOK.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "resolve_embedding_inputs"
        )
        namespace = {
            "Path": Path,
            "derived_root": str(self.root),
            "embedding_root": self.embedding_root,
            "participant_embedding_path": self.rollup,
            "databricks_path_exists": lambda path: Path(path).exists(),
        }
        exec(
            compile(
                ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), "exec"
            ),
            namespace,
        )
        self.resolve = namespace["resolve_embedding_inputs"]

    def touch(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return str(path)

    def test_completed_rollup_has_priority_over_delta(self):
        delta = self.embedding_root / "retfound_embeddings_delta"
        (delta / "_delta_log").mkdir(parents=True)
        path = self.touch(self.rollup)
        self.assertEqual(
            self.resolve("", "auto"), ([path], "parquet", "participant_visit")
        )

    def test_delta_export_is_used_without_completed_rollup(self):
        delta = self.embedding_root / "retfound_embeddings_delta"
        (delta / "_delta_log").mkdir(parents=True)
        self.assertEqual(self.resolve("", "auto"), ([str(delta)], "delta", "image"))

    def test_completed_rollup_is_used_without_delta(self):
        path = self.touch(self.rollup)
        self.assertEqual(
            self.resolve("", "auto"), ([path], "parquet", "participant_visit")
        )

    def test_batches_precede_consolidated_and_include_fairness_completion(self):
        self.touch(self.embedding_root / "retfound_embeddings.parquet")
        paths = [
            self.touch(
                self.embedding_root / "batches/batch_002/retfound_embeddings.parquet"
            ),
            self.touch(
                self.embedding_root / "batches/batch_001/retfound_embeddings.parquet"
            ),
            self.touch(self.fairness / "batch_003/retfound_embeddings.parquet"),
        ]
        self.assertEqual(self.resolve("", "auto"), (sorted(paths), "parquet", "image"))

    def test_consolidated_parquet_fallback(self):
        path = self.touch(self.embedding_root / "retfound_embeddings.parquet")
        self.assertEqual(self.resolve("", "auto"), ([path], "parquet", "image"))

    def test_explicit_parquet_overrides_discovery(self):
        path = self.touch(self.root / "custom.parquet")
        self.assertEqual(
            self.resolve(path, "auto"), ([path], "parquet", "schema_detected")
        )

    def test_explicit_parquet_directory_requires_format(self):
        path = self.root / "custom_dataset"
        path.mkdir()
        with self.assertRaisesRegex(ValueError, "Cannot infer"):
            self.resolve(str(path), "auto")
        self.assertEqual(
            self.resolve(str(path), "parquet"),
            ([str(path)], "parquet", "schema_detected"),
        )

    def test_missing_explicit_input_does_not_fall_back(self):
        self.touch(self.rollup)
        with self.assertRaisesRegex(FileNotFoundError, "Explicit RETFound"):
            self.resolve(str(self.root / "missing.parquet"), "auto")

    def test_missing_all_inputs_names_expected_producers(self):
        with self.assertRaisesRegex(
            FileNotFoundError, "01_retfound_age_fairness"
        ) as caught:
            self.resolve("", "auto")
        self.assertIn("retfound_embeddings_delta", str(caught.exception))
        self.assertIn("participant_visit_embeddings.parquet", str(caught.exception))

    def test_rollups_keep_original_image_counts(self):
        source = NOTEBOOK.read_text()
        self.assertIn(
            'F.col("n_embedded_images").cast("long").alias("n_retinal_images")', source
        )
        self.assertIn("spark.read.parquet(*embedding_paths)", source)
        self.assertIn('"embedding_input_unit": embedding_unit', source)


if __name__ == "__main__":
    unittest.main()
