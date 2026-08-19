import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Age_Glaucoma" / "10_questionnaire_epigenetic_aging_source.py"


class QuestionnaireEpigeneticNotebookContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = NOTEBOOK.read_text(encoding="utf-8")
        ast.parse(cls.source)

    def test_uses_saved_08_and_09_outputs(self):
        self.assertIn("CLSA_glaucoma_participant_oof_predictions.parquet", self.source)
        self.assertIn("matched_control_sets_private.parquet", self.source)
        self.assertIn("clsa_anatomic_explainability_private.parquet", self.source)
        self.assertIn("participant_anatomic_metrics_private.parquet", self.source)
        self.assertIn("require_complete_notebook09", self.source)

    def test_questionnaire_join_is_visit_matched_and_participant_level(self):
        self.assertIn("sap_questionnaire_visit", self.source)
        self.assertIn('on=["participant_id", "visit"]', self.source)
        self.assertIn("match_set_id", self.source)
        self.assertIn("QUESTIONNAIRE_SPECIFICATIONS", self.source)
        self.assertIn("questionnaire_missingness", self.source)
        self.assertIn("questionnaire_group_descriptives", self.source)

    def test_complete_sap_question_set_is_folded_in(self):
        for variable in (
            "ethnicity_spirometry",
            "education_level_sap",
            "marital_status",
            "visual_acuity_left",
            "visual_acuity_right",
            "oa_hand",
            "oa_hip",
            "oa_knee",
            "asthma",
            "copd",
            "social_outside_household",
            "social_religious",
            "social_education_culture",
            "social_club",
            "social_association",
            "social_other",
        ):
            self.assertIn(f'"{variable}"', self.source)
        self.assertIn("PRIMARY_SAP_SPECIFICATIONS", self.source)
        self.assertIn("SECONDARY_SAP_COMPONENT_SPECIFICATIONS", self.source)
        self.assertIn("sap_questionnaire_coverage_audit.csv", self.source)

    def test_design_fields_are_not_questionnaire_outcomes(self):
        self.assertIn('"analytic_weight"', self.source)
        self.assertIn('"sampling_strata"', self.source)
        primary_block = self.source.split(
            "PRIMARY_SAP_SPECIFICATIONS = {", 1
        )[1].split("SECONDARY_SAP_COMPONENT_SPECIFICATIONS", 1)[0]
        self.assertNotIn('"analytic_weight": {', primary_block)
        self.assertNotIn('"sampling_strata": {', primary_block)
        self.assertIn("lacks a documented PSU", self.source)

    def test_all_released_epigenetic_outputs_are_used(self):
        for variable in (
            "epigenetic_dnam_age",
            "epigenetic_age_acceleration_difference",
            "epigenetic_age_acceleration_residual",
            "epigenetic_ieaa",
            "epigenetic_eeaa",
            "epigenetic_hannum_age",
        ):
            self.assertIn(variable, self.source)
        self.assertIn('master.loc[master["visit"] != "BL"', self.source)

    def test_epigenetics_come_from_the_same_baseline_questionnaire_source(self):
        self.assertIn("2209017_UOttawa_EFreeman_BL.zip", self.source)
        self.assertIn(
            "2209017_UOttawa_EFreeman_Baseline_CoPv7_Qx_CANUE_PA_BS.csv",
            self.source,
        )
        for raw_variable in (
            "DNAmAge_COM",
            "AgeAccelerationDifference_COM",
            "AgeAccelerationResidual_COM",
            "IEAA_COM",
            "EEAA_COM",
            "Hannum_Age_COM",
        ):
            self.assertIn(f'"{raw_variable}"', self.source)
        self.assertIn("questionnaire_source_spark", self.source)
        self.assertIn("baseline_archive.open", self.source)
        self.assertIn("usecols=raw_epigenetic_columns", self.source)
        self.assertIn("chunksize=100_000", self.source)
        self.assertIn('RAW_PARTICIPANT_ID_COLUMN = "entity_id"', self.source)
        self.assertIn('"separate_epigenetic_dataset_loaded": False', self.source)
        self.assertNotIn("sap_epigenetic_baseline", self.source)
        for unsupported_column in (
            "epigenetic_measures_available_count",
            "epigenetic_complete_six_measure_panel",
            "epigenetic_difference_qc_status",
            "epigenetic_clock_range_qc_status",
        ):
            self.assertNotIn(unsupported_column, self.source)

    def test_retinal_age_prediction_modes_are_not_conflated(self):
        self.assertIn("CLSA_healthy_grouped_out_of_fold", self.source)
        self.assertIn("CLSA_healthy_frozen_model_application", self.source)
        self.assertIn("retinal_age_prediction_mode", self.source)

    def test_multiplicity_agreement_and_figures_are_present(self):
        self.assertIn("fdr_q_value", self.source)
        self.assertIn("age_measure_agreement", self.source)
        self.assertIn("correlate_age_accelerations", self.source)
        self.assertIn("Bland–Altman", self.source)
        self.assertIn("questionnaire_binary_adjusted_odds_ratios.png", self.source)

    def test_no_widget_values_are_programmatically_set(self):
        self.assertNotIn("dbutils.widgets.set", self.source)


if __name__ == "__main__":
    unittest.main()
