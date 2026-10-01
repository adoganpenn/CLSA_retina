"""Execute the entire appended cell without Spark using simulated survival data."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'notebooks/mortality_combined_final_cell.py').read_text()

class CombinedMortalityCellTests(unittest.TestCase):
    def cohort(self):
        rng = np.random.default_rng(321)
        n = 250
        age = rng.uniform(45, 85, n)
        retinal = age + rng.normal(0, 5, n)
        horvath = age + rng.normal(0, 5, n)
        hannum = age + rng.normal(0, 7, n)
        death_time = rng.exponential(10 * np.exp(-0.025 * (age - 65)), n)
        censor_time = rng.uniform(8, 14, n)
        return pd.DataFrame({
            'participant_id': pd.Series([f'id_{i:04}' for i in range(n)], dtype='string'),
            'age_at_fundus_years': age, 'sex_female': rng.integers(0, 2, n),
            'retinal_age': retinal, 'epigenetic_dnam_age': horvath,
            'epigenetic_hannum_age': hannum,
            'event': (death_time <= censor_time).astype(int),
            'followup_years': np.minimum(death_time, censor_time),
        })

    def namespace(self, directory, frame):
        return {'retinal_cohort': frame.copy(), 'epigenetic_cohort': frame.copy(),
                'output_root': Path(directory), 'outer_folds': 2, 'inner_folds': 2,
                'ridge_penalizer_grid': (0.01, 0.1), 'bootstrap_repetitions': 20,
                'evaluation_horizons_years': (5., 10.), 'random_seed': 456}

    def execute(self, namespace):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(SOURCE, 'combined-final-cell', 'exec'), namespace)

    def test_full_cell_and_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace = self.namespace(directory, self.cohort())
            self.execute(namespace)
            performance = namespace['cm_performance'].copy()
            self.assertEqual(len(performance), 4)
            self.assertEqual(len(namespace['cm_incremental']), 5)
            self.assertEqual(len(namespace['cm_oof']), 250 * 4)
            output = Path(directory) / 'combined_retinal_dnam_ages'
            self.assertEqual(len(list(output.glob('*.csv'))), 8)
            pdf = PdfReader(output / 'combined_mortality_summary.pdf')
            self.assertIn('Common-cohort prediction', pdf.pages[0].extract_text())
            self.assertNotIn('-log2(p)', namespace['cm_ph'].columns)
            first_csv = (output / 'oof_predictions.csv').read_bytes()
            self.execute(namespace)
            pd.testing.assert_frame_equal(performance, namespace['cm_performance'])
            self.assertEqual(first_csv, (output / 'oof_predictions.csv').read_bytes())
            # Copy only the synthetic preview for visual QA.
            import shutil
            preview = ROOT / 'tmp/pdfs/combined_mortality_synthetic_preview.pdf'
            preview.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(output / 'combined_mortality_summary.pdf', preview)

    def test_one_available_clock_and_gap_inverse(self):
        frame = self.cohort().drop(columns='epigenetic_hannum_age')
        frame['retinal_age_gap'] = frame['retinal_age'] - frame['age_at_fundus_years']
        frame = frame.drop(columns='retinal_age')
        with tempfile.TemporaryDirectory() as directory:
            namespace = self.namespace(directory, frame)
            self.execute(namespace)
            self.assertEqual(namespace['cm_clocks'], ['epigenetic_dnam_age'])
            self.assertEqual(len(namespace['cm_performance']), 4)

    def test_missing_ages_fail_loudly(self):
        frame = self.cohort()
        frame['retinal_age'] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            namespace = self.namespace(directory, frame)
            with self.assertRaisesRegex(ValueError, 'No existing retinal ages'):
                self.execute(namespace)

    def test_no_survival_events_fail_before_fit(self):
        frame = self.cohort()
        frame['event'] = 0
        with tempfile.TemporaryDirectory() as directory:
            namespace = self.namespace(directory, frame)
            with self.assertRaisesRegex(ValueError, '0 deaths'):
                self.execute(namespace)

    def test_notebook_has_exactly_one_new_cell(self):
        original_path = Path('/Users/adogan/Downloads/04_mortality_prediction_retfound_epigenetic_updated.ipynb')
        if not original_path.is_file():
            self.skipTest('Original user attachment is local-only; all executable cell tests still run.')
        original = json.loads(original_path.read_text())
        updated = json.loads((ROOT / 'notebooks/04_mortality_prediction_combined_model.ipynb').read_text())
        self.assertEqual(len(updated['cells']), len(original['cells']) + 2)
        for index, cell in enumerate(original['cells']):
            if index != 29:
                self.assertEqual(cell, updated['cells'][index])
        self.assertEqual(''.join(updated['cells'][-2]['source']), SOURCE)
        self.assertIn('CLSA_corrected_within_fold_evaluation', updated['cells'][-1]['metadata']['tags'])

if __name__ == '__main__':
    unittest.main()
