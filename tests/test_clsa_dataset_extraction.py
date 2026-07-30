from pathlib import Path
import sys
import tempfile
import unittest
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from clsa_dataset_extraction import extract_fundus_zip_release


class DatasetExtractionTests(unittest.TestCase):
    def test_supported_image_extracts_and_aes_is_logged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "fundus.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "2209017_BL/7732381/retinal_left.jpeg",
                    b"left-image",
                )
                aes_info = zipfile.ZipInfo(
                    "2209017_BL/7732381/retinal_right.jpeg"
                )
                aes_info.extra = b"\x01\x99\x00\x00"
                archive.writestr(aes_info, b"aes-image")

            output_root = root / "output"
            rows = extract_fundus_zip_release(
                archive_path,
                output_root,
                "unused-for-unencrypted-test",
                release_name="fundus_baseline",
                visit="BL",
            )

            statuses = {
                row["member_path"]: row["status"]
                for row in rows
            }
            self.assertEqual(
                statuses[
                    "2209017_BL/7732381/retinal_left.jpeg"
                ],
                "extracted",
            )
            self.assertEqual(
                statuses[
                    "2209017_BL/7732381/retinal_right.jpeg"
                ],
                "skipped_unsupported_aes",
            )
            self.assertTrue(
                (
                    output_root
                    / "2209017_BL"
                    / "7732381"
                    / "retinal_left.jpeg"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
