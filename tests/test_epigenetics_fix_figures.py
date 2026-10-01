"""PNG fixes must be standalone, analysis-free, and use persisted statistics."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT if (ROOT / 'epigenetics_fix_png_renderer.py').is_file() else ROOT / 'Age_Glaucoma/Algorithm Fairness'
spec = importlib.util.spec_from_file_location('fixed_png', SOURCE_ROOT / 'epigenetics_fix_png_renderer.py')
fix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fix)
STATS = Path('/private/tmp/clsa_epigenetics_figures_actual_20260915/03_statistics')

def engine():
    module = types.ModuleType('_clsa_test_saved_png')
    sys.modules[module.__name__] = module
    exec(compile((SOURCE_ROOT / 'clsa_figure_renderer.py').read_text(), 'saved-core', 'exec'), module.__dict__)
    return module

class EpigeneticsPngFixTests(unittest.TestCase):
    def test_standalone_notebook_parses_and_embeds_both_modules(self):
        notebook = json.loads((SOURCE_ROOT / 'epigenetics_fix_figures.ipynb').read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
        source = ''.join(notebook['cells'][2]['source'])
        self.assertIn('_core_source =', source)
        self.assertIn('_fix_source =', source)
        self.assertNotIn('sys.path.insert', source)
        self.assertIn('fixed_pngs', ''.join(notebook['cells'][1]['source']))

    def test_no_model_fit_or_statistical_tests_in_new_renderer(self):
        tree = ast.parse((SOURCE_ROOT / 'epigenetics_fix_png_renderer.py').read_text())
        forbidden = {'fit', 'fit_transform', 'ols', 'pearsonr', 'set_theme', 'tight_layout'}
        for call in ast.walk(tree):
            if isinstance(call, ast.Call):
                name = call.func.attr if isinstance(call.func, ast.Attribute) else (
                    call.func.id if isinstance(call.func, ast.Name) else '')
                self.assertNotIn(name, forbidden)

    def test_missing_statistics_names_file_and_producing_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            renderer = fix.FixedPngRenderer(engine(), Path(directory) / 'missing', Path(directory) / 'out')
            with self.assertRaisesRegex(FileNotFoundError, r'figure_01_locked_validation_points.*cell 41'):
                renderer.build_01B()

    @unittest.skipUnless(STATS.is_dir(), 'Actual downloaded statistics are local-only')
    def test_scatter_statistics_live_outside_data_axes(self):
        with tempfile.TemporaryDirectory() as directory:
            renderer = fix.FixedPngRenderer(engine(), STATS, directory, dpi=100)
            with mpl.rc_context(renderer.e._style(renderer.profile)):
                for builder, count in [(renderer.build_01B, 2), (renderer.build_02A, 6),
                                       (lambda: renderer.build_single_scatter('A'), 1)]:
                    fig, record = builder()
                    try:
                        plots = [ax for ax in fig.axes if ax.get_aspect() == 1.0 and ax.collections]
                        self.assertEqual(len(plots), count)
                        self.assertTrue(all(not ax.texts for ax in plots))
                        renderer.check_layout(fig, record.axes)
                        renderer.e._number_manifest_check([record], Path(directory))
                    finally:
                        plt.close(fig)

    @unittest.skipUnless(STATS.is_dir(), 'Actual downloaded statistics are local-only')
    def test_png_is_deterministic_and_style_context_restores_rc(self):
        with tempfile.TemporaryDirectory() as directory:
            renderer = fix.FixedPngRenderer(engine(), STATS, directory, dpi=100)
            before = mpl.rcParams['font.size']
            results = []
            for _ in range(2):
                with mpl.rc_context(renderer.e._style(renderer.profile)):
                    fig, record = renderer.build_01B()
                    try:
                        results.append(renderer.save(fig, '01', record, whole=True))
                    finally:
                        plt.close(fig)
            self.assertEqual(results[0]['sha256'], results[1]['sha256'])
            self.assertEqual(before, mpl.rcParams['font.size'])
            metrics = pd.read_csv(STATS / 'locked_discovery_validation_metrics.csv')
            expected = metrics.loc[metrics.analysis.eq('Locked validation: discovery-calibrated'), 'mae'].iloc[0]
            self.assertEqual(results[0]['statistics']['comparisons']['Locked validation']['mae'], expected)

if __name__ == '__main__':
    unittest.main()
