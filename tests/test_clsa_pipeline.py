from pathlib import Path
import tempfile
import unittest

from src.clsa_pipeline import (
    _safe_archive_destination,
    load_json,
    load_variable_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PipelineConfigurationTests(unittest.TestCase):
    def test_configuration_and_manifest_load(self) -> None:
        config = load_json(REPOSITORY_ROOT / "config/clsa_pipeline_config.json")
        manifest = load_variable_manifest(
            REPOSITORY_ROOT / "config/variable_manifest.csv"
        )

        self.assertEqual(
            config["volume_root"],
            "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset",
        )
        self.assertEqual(len(manifest), 38)
        names = [spec.standard_name for spec in manifest]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("self_reported_vision", names)
        self.assertIn("epigenetic_age", names)

    def test_archive_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaises(ValueError):
                _safe_archive_destination(root, "../../outside.jpg")

    def test_safe_archive_member_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            destination = _safe_archive_destination(
                root, "participant_001/right.jpg"
            )
            self.assertEqual(
                destination, (root / "participant_001/right.jpg").resolve()
            )


if __name__ == "__main__":
    unittest.main()
