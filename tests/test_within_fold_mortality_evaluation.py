"""Verify paired within-fold bootstrap, ties, invariance, and disk resumption."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'notebooks/mortality_within_fold_evaluation_cell.py').read_text()
NAMES = ['Age + sex', 'Age + sex + retinal age', 'Age + sex + DNAm ages',
         'Age + sex + retinal + DNAm ages']

class WithinFoldEvaluationTests(unittest.TestCase):
    def predictions(self):
        rng = np.random.default_rng(314)
        n = 120
        risk = rng.normal(size=n)
        time = np.round(rng.exponential(8 * np.exp(-risk / 2), n)) + 1
        event = rng.integers(0, 2, n)
        base = pd.DataFrame({'participant_id': [f'p{i:04}' for i in range(n)],
            'fold': np.repeat([1, 2, 3], n // 3), 'followup_years': time, 'event': event})
        parts = []
        for j, name in enumerate(NAMES):
            frame = base.copy()
            frame['model'] = name
            frame['log_risk'] = risk + rng.normal(0, 1.3 - j * 0.2, n)
            parts.append(frame)
        return pd.concat(parts, ignore_index=True)

    def execute(self, frame, directory, disk=False):
        namespace = {'cm_export_root': Path(directory), 'random_seed': 123,
                     'bootstrap_repetitions': 50}
        if disk:
            frame.to_csv(Path(directory) / 'oof_predictions.csv', index=False)
        else:
            namespace['cm_oof'] = frame
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(SOURCE, 'within-fold-cell', 'exec'), namespace)
        return namespace

    def test_bootstrap_matches_direct_lifelines_including_ties(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace = self.execute(self.predictions(), directory)
            rng = np.random.default_rng(namespace['wf_seed'])
            direct_c = []
            groups = namespace['wf_groups']
            for fold in namespace['wf_folds']:
                selected = [g.loc[g['fold'].eq(fold)].reset_index(drop=True) for g in groups]
                indices = rng.integers(0, len(selected[0]), len(selected[0]))
                values = []
                for group in selected:
                    sample = group.iloc[indices]
                    values.append(concordance_index(sample['followup_years'],
                                                    -sample['log_risk'], sample['event']))
                direct_c.append(values)
            np.testing.assert_allclose(namespace['wf_boot_mean'][0], np.mean(direct_c, axis=0), atol=1e-12)
            self.assertEqual(len(namespace['wf_performance']), 8)
            self.assertEqual(len(namespace['wf_incremental']), 10)
            self.assertFalse(namespace['wf_metadata']['model_refitting'])
            self.assertEqual(len(list(Path(directory).iterdir())), 5)

    def test_invariant_to_arbitrary_fold_scaling_and_centering(self):
        frame = self.predictions()
        changed = frame.copy()
        for j, name in enumerate(NAMES):
            for fold in [1, 2, 3]:
                mask = changed['model'].eq(name) & changed['fold'].eq(fold)
                changed.loc[mask, 'log_risk'] = (changed.loc[mask, 'log_risk']
                    * (0.2 + fold + j) + 100 * fold * (j + 1))
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first, second = self.execute(frame, a), self.execute(changed, b)
            pd.testing.assert_frame_equal(first['wf_performance'], second['wf_performance'])
            pd.testing.assert_frame_equal(first['wf_incremental'], second['wf_incremental'])

    def test_disk_resumption_and_idempotent_exports(self):
        frame = self.predictions()
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first, second = self.execute(frame, a), self.execute(frame, b, disk=True)
            pd.testing.assert_frame_equal(first['wf_performance'], second['wf_performance'])
            saved = (Path(a) / 'corrected_paired_within_fold_increment.csv').read_bytes()
            self.execute(frame, a)
            self.assertEqual(saved, (Path(a) / 'corrected_paired_within_fold_increment.csv').read_bytes())

    def test_mismatched_folds_fail_without_silent_dropping(self):
        frame = self.predictions()
        frame.loc[frame.index[-1], 'fold'] = 99
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Models differ'):
                self.execute(frame, directory)

    def test_identical_models_have_exactly_zero_paired_increment(self):
        frame = self.predictions()
        scores = frame.loc[frame['model'].eq(NAMES[0]), 'log_risk'].to_numpy()
        for name in NAMES:
            frame.loc[frame['model'].eq(name), 'log_risk'] = scores
        with tempfile.TemporaryDirectory() as directory:
            namespace = self.execute(frame, directory)
            for column in ['delta_c_index', 'delta_c_lower_95', 'delta_c_upper_95']:
                self.assertTrue(namespace['wf_incremental'][column].eq(0).all())

if __name__ == '__main__':
    unittest.main()
