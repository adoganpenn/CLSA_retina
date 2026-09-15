"""Regression tests against the released-column loader in epigenetics.ipynb."""

import ast
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/04_mortality_prediction_retfound_epigenetic.py"
WORKING = ROOT / "Age_Glaucoma/Algorithm Fairness/epigenetics.ipynb"
if not WORKING.exists():
    WORKING = ROOT / "epigenetics_publication_renderer.ipynb"


def assignment(tree, name):
    return ast.literal_eval(
        next(
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            )
        )
    )


class MortalityEpigeneticInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.archive_path = Path(self.temp.name) / "baseline.zip"
        self.suffix = "CoPv7_Qx_CANUE_PA_BS.csv"
        self.member = "baseline/released_" + self.suffix
        self.tree = ast.parse(NOTEBOOK.read_text())
        self.mapping = assignment(self.tree, "EPIGENETIC_SOURCE_VARIABLES")
        self.missing = assignment(self.tree, "numeric_missing_codes")
        function = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "load_baseline_epigenetic_phenotypes"
        )
        namespace = {
            "Path": Path,
            "pd": pd,
            "zipfile": zipfile,
            "EPIGENETIC_SOURCE_VARIABLES": self.mapping,
            "numeric_missing_codes": self.missing,
        }
        exec(
            compile(
                ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), "exec"
            ),
            namespace,
        )
        self.load = namespace["load_baseline_epigenetic_phenotypes"]

    def write_release(self, rows, drop=(), members=None):
        frame = pd.DataFrame(rows)
        for column in self.mapping:
            if column not in frame:
                frame[column] = "-8"
        frame = frame.drop(columns=list(drop))
        with zipfile.ZipFile(self.archive_path, "w") as archive:
            for member in members or [self.member]:
                archive.writestr(member, frame.to_csv(index=False))

    def test_mapping_and_sentinels_exactly_match_working_notebook(self):
        notebook = json.loads(WORKING.read_text())
        source = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            and "AgeAccelerationResidual_COM" in "".join(cell["source"])
        )
        working_tree = ast.parse(source)
        self.assertEqual(
            self.mapping, assignment(working_tree, "EPIGENETIC_SOURCE_VARIABLES")
        )
        self.assertEqual(
            self.missing, assignment(working_tree, "numeric_missing_codes")
        )

    def test_all_six_released_values_loaded_and_negative_acceleration_retained(self):
        values = [62.5, -3.25, -1.875, -0.75, 2.125, 64.75]
        self.write_release(
            [{"entity_id": " 1000001 ", **dict(zip(self.mapping, values))}]
        )
        result, metadata = self.load(self.archive_path, self.suffix, ["1000001"], 1)
        self.assertEqual(result.participant_id.tolist(), ["1000001"])
        for raw, expected in zip(self.mapping, values):
            self.assertEqual(result.loc[0, self.mapping[raw]], expected)
        self.assertEqual(metadata["baseline_member"], self.member)
        self.assertFalse(metadata["imaging_restricted_checkpoint_used"])

    def test_missing_codes_follow_working_loader_without_treating_negative_values_as_missing(
        self,
    ):
        rows = [
            {
                "entity_id": f"{1000001 + i}",
                "AgeAccelerationResidual_COM": code,
                "DNAmAge_COM": "60",
            }
            for i, code in enumerate(sorted(self.missing))
        ]
        rows += [{"entity_id": "2000001", "AgeAccelerationResidual_COM": "-2.5"}]
        self.write_release(rows)
        result, _ = self.load(
            self.archive_path, self.suffix, [row["entity_id"] for row in rows], 2
        )
        missing_rows = result.loc[
            result.participant_id != "2000001", "epigenetic_age_acceleration_residual"
        ]
        self.assertTrue(missing_rows.isna().all())
        self.assertEqual(
            result.loc[
                result.participant_id == "2000001",
                "epigenetic_age_acceleration_residual",
            ].iloc[0],
            -2.5,
        )

    def test_separate_epigenetic_cohort_is_not_restricted_to_retinal_ids(self):
        self.write_release(
            [
                {"entity_id": "1000001", "AgeAccelerationResidual_COM": "1"},
                {"entity_id": "1000002", "AgeAccelerationResidual_COM": "2"},
                {"entity_id": "1000003", "AgeAccelerationResidual_COM": "3"},
            ]
        )
        sap_ids = ["1000001", "1000002"]
        result, _ = self.load(self.archive_path, self.suffix, sap_ids, 1)
        self.assertEqual(result.participant_id.tolist(), sap_ids)
        self.assertEqual(len(result), 2)  # even when only one has retinal vectors

    def test_all_missing_participants_are_excluded(self):
        self.write_release([{"entity_id": "1000001"}])
        result, _ = self.load(self.archive_path, self.suffix, ["1000001"], 1)
        self.assertTrue(result.empty)
        self.assertTrue(set(self.mapping.values()).issubset(result.columns))

    def test_duplicate_participant_is_not_silently_deduplicated(self):
        self.write_release(
            [
                {"entity_id": "1000001", "IEAA_COM": "1"},
                {"entity_id": "1000001", "IEAA_COM": "2"},
            ]
        )
        with self.assertRaisesRegex(ValueError, "not unique"):
            self.load(self.archive_path, self.suffix, ["1000001"], 1)

    def test_missing_released_field_fails_with_actual_column_name(self):
        self.write_release(
            [{"entity_id": "1000001"}], drop=["AgeAccelerationResidual_COM"]
        )
        with self.assertRaisesRegex(ValueError, "AgeAccelerationResidual_COM"):
            self.load(self.archive_path, self.suffix, ["1000001"], 1)

    def test_ambiguous_archive_member_fails(self):
        self.write_release(
            [{"entity_id": "1000001"}], members=[self.member, "other/" + self.suffix]
        )
        with self.assertRaisesRegex(ValueError, "found 2"):
            self.load(self.archive_path, self.suffix, ["1000001"], 1)

    def test_correct_working_oof_prediction_path_is_present(self):
        self.assertIn(
            "CLSA_full_cohort_age_predictions_oof.parquet", NOTEBOOK.read_text()
        )

    def test_models_have_not_switched_primary_measure(self):
        self.assertEqual(
            assignment(self.tree, "primary_epigenetic_measure"),
            "epigenetic_age_acceleration_residual",
        )

    def test_raw_release_to_separate_mortality_cohort_linkage(self):
        self.write_release(
            [
                {"entity_id": "1000001", "AgeAccelerationResidual_COM": "-1.5"},
                {"entity_id": "1000002", "AgeAccelerationResidual_COM": "2.5"},
            ]
        )
        sap = pd.DataFrame(
            {
                "participant_id": ["1000001", "1000002"],
                "age_at_fundus_years": [60.0, 65.0],
                "sex_female": [0.0, 1.0],
            }
        )
        released, _ = self.load(self.archive_path, self.suffix, sap.participant_id, 1)
        predictors = sap.merge(
            released, on="participant_id", how="left", validate="one_to_one"
        )
        predictors["epigenetic_index_date"] = pd.Timestamp("2012-01-01")
        status = pd.DataFrame(
            {
                "participant_id": ["1000001", "1000002"],
                "death_event": pd.Series([1, 0], dtype="Int64"),
                "death_date": [pd.Timestamp("2018-01-01"), pd.NaT],
                "censor_date": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")],
                "baseline_date": [
                    pd.Timestamp("2012-01-01"),
                    pd.Timestamp("2012-01-01"),
                ],
            }
        )
        functions = [
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"parse_date", "attach_survival_outcome"}
        ]
        namespace = {"pd": pd, "status_core": status}
        exec(
            compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), "exec"),
            namespace,
        )
        cohort, flow = namespace["attach_survival_outcome"](
            predictors, "epigenetic_index_date", "Epigenetic clocks"
        )
        self.assertEqual(cohort.participant_id.tolist(), ["1000001", "1000002"])
        self.assertEqual(cohort.event.tolist(), [1, 0])
        self.assertEqual(
            cohort.epigenetic_age_acceleration_residual.tolist(), [-1.5, 2.5]
        )
        self.assertTrue(cohort.followup_years.gt(0).all())
        self.assertEqual(flow.iloc[-1]["n"], 2)
        self.assertNotIn("death_date", cohort.columns)


if __name__ == "__main__":
    unittest.main()
