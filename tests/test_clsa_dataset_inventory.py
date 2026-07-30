import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from src.clsa_dataset_inventory import (
    build_dataset_readme,
    file_extension,
    fundus_member_identity,
    inspect_zip_archive,
    inspect_zip_csv_headers,
)


class DatasetInventoryTests(unittest.TestCase):
    def test_compound_extensions_and_fundus_identity(self) -> None:
        self.assertEqual(file_extension("clsa_imp_1_v3.bgen.bgi"), ".bgen.bgi")
        self.assertEqual(file_extension("file.csv.gz"), ".csv.gz")
        self.assertEqual(
            fundus_member_identity(
                "2209017_BL/7732381/retinal_left.jpeg"
            ),
            ("7732381", "L"),
        )

    def test_zip_inspection_counts_pairs_and_does_not_extract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "2209017_BL.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "2209017_BL/7732381/retinal_left.jpeg", b"left"
                )
                archive.writestr(
                    "2209017_BL/7732381/retinal_right.jpeg", b"right"
                )
                aes_info = zipfile.ZipInfo(
                    "2209017_BL/3369736/retinal_left.jpeg"
                )
                aes_info.extra = b"\x01\x99\x00\x00"
                archive.writestr(aes_info, b"aes-marker")

            rows, summary = inspect_zip_archive(
                archive_path,
                release_name="fundus_baseline",
                role="fundus_imaging",
                visit="BL",
            )

            self.assertEqual(len(rows), 3)
            self.assertEqual(summary["image_count"], 3)
            self.assertEqual(summary["participant_count"], 2)
            self.assertEqual(summary["complete_eye_pair_count"], 1)
            self.assertEqual(summary["possible_aes_count"], 1)
            self.assertFalse((root / "2209017_BL").exists())

    def test_questionnaire_header_and_readme_are_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "questionnaire.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "delivery/comprehensive.csv",
                    "participant_id,age,sex\n1111111,60,F\n",
                )
            release = {
                "name": "questionnaire_followup2",
                "role": "questionnaire",
                "visit": "FUP2",
                "path": str(archive_path),
            }
            members, summary = inspect_zip_archive(
                archive_path,
                release_name=release["name"],
                role=release["role"],
                visit=release["visit"],
            )
            schemas = inspect_zip_csv_headers([release])
            self.assertEqual(schemas[0]["column_count"], 3)
            self.assertEqual(
                json.loads(schemas[0]["columns_json"]),
                ["participant_id", "age", "sex"],
            )

            readiness = {
                "direct_genotypes": {
                    "bed": False,
                    "bim": True,
                    "fam": True,
                },
                "direct_ready": False,
                "missing_bgen_chromosomes": list(range(1, 24)),
                "missing_bgi_chromosomes": [],
                "imputed_ready": False,
                "missing_metadata": [],
            }
            readme = build_dataset_readme(
                volume_root="/Volumes/example",
                output_root="/Volumes/example/derived",
                file_inventory=[
                    {
                        "name": "clsa_gen_v3.bim",
                        "relative_path": "Genomics3_clsa/clsa_gen_v3.bim",
                        "extension": ".bim",
                        "bytes": 10,
                    },
                    {
                        "name": "clsa_gen_v3.fam",
                        "relative_path": "Genomics3_clsa/clsa_gen_v3.fam",
                        "extension": ".fam",
                        "bytes": 10,
                    },
                    {
                        "name": "clsa_imp_1_v3.bgen.bgi",
                        "relative_path": (
                            "Genomics3_clsa/clsa_imp_1_v3.bgen.bgi"
                        ),
                        "extension": ".bgen.bgi",
                        "bytes": 10,
                    },
                ],
                releases=[release],
                zip_members=members,
                zip_summaries=[summary],
                csv_schemas=schemas,
                dictionary_profile={
                    "path": "/Volumes/example/dictionary.xlsx",
                    "sheet_count": 1,
                    "sheets": [
                        {
                            "sheet_name": "Variables",
                            "data_rows": 2,
                            "column_count": 2,
                            "headers": ["table", "name"],
                            "table_count": 1,
                            "tables": ["Comprehensive"],
                        }
                    ],
                },
                genetics_readiness=readiness,
                generated_utc="2026-07-30T00:00:00+00:00",
            )

            self.assertIn("questionnaire_followup2", readme)
            self.assertIn("participant_id", readme)
            self.assertIn("Direct genotype dataset ready: **False**", readme)
            self.assertNotIn("1111111", readme)


if __name__ == "__main__":
    unittest.main()
