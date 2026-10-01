"""Panel-first, PNG-only rendering from the completed CLSA statistics exports.

The established, analysis-free renderer is supplied as an isolated module.
No old PNG is edited, no model is fitted, and no statistical test is rerun.
"""
from pathlib import Path
from dataclasses import replace, asdict
import hashlib
import json
import shutil
import tempfile
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.text import Text

FIGURE_TITLES = {
    '01': 'Cohort construction and locked validation',
    '02': 'Shared chronological age and modality-specific acceleration',
    '03': 'Distinct phenotypic domains',
    '04': 'Retinal stability across eyes, specifications, and time',
    '05': 'Cross-modal longitudinal prediction',
    'S1': 'Complete age-model diagnostics',
    'S2': 'Selection and demographic performance',
    'S3': 'Alternative cross-clock comparisons',
    'S4': 'Exploratory three-clock decomposition',
    'S5': 'Complete questionnaire-wide associations',
    'S6': 'Expanded retinal sensitivity analyses',
    'S7': 'Same-eye acuity and complete longitudinal specifications',
    'S8': 'Exploratory comorbidity analyses',
    'S9': 'Expanded methylation-age prediction beyond age',
    'S10': 'Construct robustness across epigenetic definitions',
    'S11': 'Expanded longitudinal change, attrition, and stability',
}

class FixedPngRenderer:
    def __init__(self, engine, statistics_root, output_root, dpi=220):
        self.e = engine
        self.statistics_root = Path(statistics_root)
        self.output_root = Path(output_root)
        self.dpi = int(dpi)
        self.current_id = None
        # Changes affect ONLY this notebook's isolated renderer module.
        self.e._statistics_root = lambda unused: self.statistics_root
        self.profile = replace(engine.PROFILES['slide'], width_in=14,
            max_height_in=18, base_font=10, tick_font=10, axis_font=11,
            panel_font=12, annotation_font=10, line_width=0.8,
            formats=('png',), transparent=False, min_font=9, dpi=self.dpi)
        self.e._figure = self._large_canvas
        # Panel IDs live in filenames/gallery headings; don't collide with titles.
        self.e._panel_letter = lambda ax, letter, profile: ax.text(0, 1, '')
        label_file = self.statistics_root / 'clsa_variable_labels_used_in_figures.csv'
        if label_file.is_file():
            labels = pd.read_csv(label_file)
            if {'variable', 'publication_label'}.issubset(labels.columns):
                self.e.VARIABLE_LABELS.update(dict(zip(labels.variable, labels.publication_label)))

    def _large_canvas(self, profile, publication_height_mm, slide_height_in):
        height = {'01': 9, '02': 12, '03': 8, '04': 12, '05': 11,
                  'S1': 12, 'S3': 13, 'S5': 12, 'S6': 12,
                  'S7': 12, 'S8': 13, 'S11': 12}.get(self.current_id, 9)
        return plt.figure(figsize=(14, height), layout='constrained')

    def read(self, name, cell, columns=()):
        return self.e._read(self.output_root, self.e.Source(name, cell), columns)

    def row(self, table, analysis, name, cell):
        return self.e._row(table, table.analysis.eq(analysis), self.e.Source(name, cell), analysis)

    def check_n(self, points, row, column='n'):
        if len(points) != int(row[column]):
            raise ValueError(f'Point/statistic mismatch: {len(points)} plotted, stored {column}={row[column]}')

    def scatter(self, ax, points, x, y, limits, xlabel, ylabel):
        hb = ax.hexbin(points[x], points[y], gridsize=40, mincnt=1,
                       extent=(limits[0], limits[1], limits[0], limits[1]),
                       cmap='viridis', linewidths=0)
        ax.plot(limits, limits, '--', color=self.e.REFERENCE, linewidth=0.8)
        ax.set(xlim=limits, ylim=limits, xlabel=xlabel, ylabel=ylabel)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(labelsize=10)
        return hb

    def density_key(self, fig, artists, cax):
        maximum = max(float(h.get_array().max()) for h in artists)
        norm = mpl.colors.Normalize(vmin=0, vmax=max(1, maximum))
        for artist in artists:
            artist.set_norm(norm)
        cb = fig.colorbar(artists[0], cax=cax, orientation='horizontal')
        cb.set_label('Point density (participants per hexbin)', fontsize=10)
        cb.ax.tick_params(labelsize=9)

    def card(self, ax, heading, lines):
        ax.set_axis_off()
        ax.text(0.03, 0.97, heading, transform=ax.transAxes,
                va='top', fontsize=11, fontweight='bold', color=self.e.REFERENCE)
        ax.text(0.03, 0.85, '\n'.join(lines), transform=ax.transAxes,
                va='top', fontsize=10, linespacing=1.7, color=self.e.REFERENCE)

    def metric_lines(self, row):
        f = self.e.fmt
        return [f"n = {f(row.n, 'n')}", f"MAE = {f(row.mae, 'mae')} years",
                f"R² = {f(row.r2, 'r2')}", f"Pearson r = {f(row.pearson_r, 'r')}",
                f"Calibration slope = {f(row.calibration_slope, 'slope')}",
                f"CCC = {f(row.ccc, 'ccc')}"]

    def record(self, panel, names, axes, n=None, **stats):
        return self.e._record(panel, [self.e.Source(name, cell) for name, cell in names],
                              axes, n=n, **stats)

    def build_01B(self):
        names = [('figure_01_locked_validation_points.parquet', 41),
                 ('locked_discovery_validation_metrics.csv', 41)]
        points = self.read(*names[0], ['analysis_phase', 'chronological_age', 'retinal_age_locked'])
        metrics = self.read(*names[1], ['analysis', 'n', 'mae', 'r2', 'pearson_r', 'calibration_slope', 'ccc'])
        limits = self.e._equal_limits(points.chronological_age, points.retinal_age_locked)
        fig = plt.figure(figsize=(12.5, 5.6), layout='constrained')
        outer = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.67], height_ratios=[1, 0.06])
        axes = [fig.add_subplot(outer[0, i]) for i in range(2)]
        rail = outer[0, 2].subgridspec(2, 1)
        cards = [fig.add_subplot(rail[i, 0]) for i in range(2)]
        artists, statistics = [], {}
        for i, phase in enumerate(['Discovery grouped OOF', 'Locked validation']):
            subset = points.loc[points.analysis_phase.eq(phase)].dropna(
                subset=['chronological_age', 'retinal_age_locked']).sort_values(
                ['chronological_age', 'retinal_age_locked'], kind='stable')
            row = self.row(metrics, phase + ': discovery-calibrated', *names[1])
            self.check_n(subset, row)
            artists.append(self.scatter(axes[i], subset, 'chronological_age', 'retinal_age_locked',
                limits, 'Chronological age (years)', 'Calibrated retinal age (years)'))
            axes[i].set_title('Discovery OOF' if i == 0 else 'Locked validation', fontsize=11)
            self.card(cards[i], 'Discovery OOF' if i == 0 else 'Locked validation', self.metric_lines(row))
            statistics[phase] = {k: float(row[k]) for k in
                ['n', 'mae', 'r2', 'pearson_r', 'calibration_slope', 'ccc']}
        self.density_key(fig, artists, fig.add_subplot(outer[1, :2]))
        fig.suptitle('Figure 1B · Retinal-age validation (shared axes)', fontsize=12, fontweight='bold')
        return fig, self.record('B', names, fig.axes, len(points), comparisons=statistics)

    def build_02A(self):
        names = [('figure_02_three_age_points.parquet', 17), ('three_age_agreement_metrics.csv', 17)]
        points = self.read(*names[0], ['chronological_age', 'retinal_age_oof',
                                    'epigenetic_dnam_age', 'epigenetic_hannum_age'])
        metrics = self.read(*names[1], ['analysis', 'n', 'pearson_r', 'mean_error'])
        limits = self.e._equal_limits(*(points[c] for c in ['chronological_age', 'retinal_age_oof',
                                      'epigenetic_dnam_age', 'epigenetic_hannum_age']))
        fig = plt.figure(figsize=(12, 10.2), layout='constrained')
        outer = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.045])
        artists, statistics = [], {}
        for ri, (clock, column, short) in enumerate([
            ('Horvath DNAm age', 'epigenetic_dnam_age', 'Horvath'),
            ('Hannum epigenetic age', 'epigenetic_hannum_age', 'Hannum')]):
            for ci in range(3):
                grid = outer[ri, ci].subgridspec(2, 1, height_ratios=[1, 0.14])
                ax, note = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
                if ci == 0:
                    x, y = 'chronological_age', 'retinal_age_oof'
                    analysis = f'Retinal age versus chronological age: {clock} subset'
                    xlabel, ylabel = 'Chronological age (years)', 'Retinal age (years)'
                    title, difference = f'{short} subset: retinal vs age', 'retinal − chronological'
                elif ci == 1:
                    x, y = 'chronological_age', column
                    analysis = f'{clock} versus chronological age'
                    xlabel, ylabel = 'Chronological age (years)', f'{short} age (years)'
                    title, difference = f'{short} vs chronological age', 'DNAm − chronological'
                else:
                    x, y = 'retinal_age_oof', column
                    analysis = f'Retinal age versus {clock}'
                    xlabel, ylabel = 'Retinal age (years)', f'{short} age (years)'
                    title, difference = f'{short} vs retinal age', 'retinal − DNAm'
                subset = points[[x, y]].dropna().sort_values([x, y], kind='stable')
                row = self.row(metrics, analysis, *names[1])
                self.check_n(subset, row)
                artists.append(self.scatter(ax, subset, x, y, limits, xlabel, ylabel))
                ax.set_title(title, fontsize=10, fontweight='bold')
                note.set_axis_off()
                note.text(0.5, 0.65, f"n = {int(row.n):,}; r = {self.e.fmt(row.pearson_r, 'r')}\n"
                    f"Mean ({difference}) = {self.e.fmt(row.mean_error, 'years')} years",
                    ha='center', va='center', fontsize=9.5, transform=note.transAxes)
                statistics[analysis] = {'n': int(row.n), 'r': float(row.pearson_r),
                                        'mean_difference': float(row.mean_error)}
        self.density_key(fig, artists, fig.add_subplot(outer[2, :]))
        fig.suptitle('Figure 2A · Three-age relationships (all panels share age limits)',
                     fontsize=12, fontweight='bold')
        return fig, self.record('A', names, fig.axes, len(points), comparisons=statistics)

    def build_single_scatter(self, panel):
        if panel == 'A':
            names = [('figure_04_inter_eye_points.parquet', 47), ('inter_eye_reliability.csv', 47)]
            x, y = 'right_retinal_age', 'left_retinal_age'
            points, row = self.read(*names[0], [x, y]), self.read(*names[1]).iloc[0]
            self.check_n(points, row, 'n_paired_participants')
            title, xlabel, ylabel = 'Inter-eye agreement', 'Right-eye retinal age (years)', 'Left-eye retinal age (years)'
            stats = {k: float(row[k]) for k in ['n_paired_participants', 'pearson_r', 'ccc',
                'mean_right_minus_left_years', 'sd_right_minus_left_years', 'mae_between_eyes_years']}
            lines = [f"n = {int(row.n_paired_participants):,}", f"Pearson r = {row.pearson_r:.3f}",
                f"CCC = {row.ccc:.3f}", f"MAE = {row.mae_between_eyes_years:.2f} years",
                f"Mean (right − left) = {self.e.fmt(row.mean_right_minus_left_years, 'years')} years",
                f"SD (right − left) = {row.sd_right_minus_left_years:.2f} years"]
        else:
            names = [('figure_04_longitudinal_points.parquet', 53), ('longitudinal_retinal_change_repeatability.csv', 53)]
            x, y = 'retinal_acceleration_bl', 'retinal_acceleration_f1'
            points = self.read(*names[0], [x, y])
            metrics = self.read(*names[1])
            selected = metrics.loc[metrics.analysis.str.startswith('All participants')]
            if len(selected) != 1:
                raise ValueError('Expected one all-participant longitudinal summary.')
            row = selected.iloc[0]
            self.check_n(points, row)
            title = 'Same-person longitudinal stability'
            xlabel, ylabel = 'Baseline retinal acceleration (years)', 'Follow-up retinal acceleration (years)'
            stats = {k: float(row[k]) for k in ['n', 'acceleration_stability_r',
                                              'direction_correct_proportion', 'mean_followup_years']}
            lines = [f"n = {int(row.n):,}", f"Pearson r = {row.acceleration_stability_r:.3f}",
                f"Retinal-age predictions increasing = {row.direction_correct_proportion * 100:.1f}%",
                f"Mean follow-up = {row.mean_followup_years:.2f} years"]
        fig = plt.figure(figsize=(9, 5.5), layout='constrained')
        grid = fig.add_gridspec(2, 2, width_ratios=[1, 0.72], height_ratios=[1, 0.06])
        ax, side = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
        hb = self.scatter(ax, points, x, y, self.e._equal_limits(points[x], points[y]), xlabel, ylabel)
        self.card(side, 'Saved results', lines)
        self.density_key(fig, [hb], fig.add_subplot(grid[1, 0]))
        fig.suptitle(f'Figure 4{panel} · {title}', fontsize=12, fontweight='bold')
        return fig, self.record(panel, names, fig.axes, len(points), summary=stats)

    def build_S1A(self):
        names = [('figure_S1_age_diagnostic_points.parquet', 12), ('model_standard_metrics.csv', 12)]
        points, metrics = self.read(*names[0]), self.read(*names[1])
        fig = plt.figure(figsize=(13, 10), layout='constrained')
        grid = fig.add_gridspec(2, 3, height_ratios=[1, 0.04])
        artists, statistics = [], {}
        for i, (label, analysis, short) in enumerate([
            ('Retinal age', 'Chronological age head: participant', 'Retinal'),
            ('Horvath DNAm age', 'Horvath DNAm age head: baseline participant', 'Horvath DNAm'),
            ('Hannum epigenetic age', 'Hannum epigenetic age head: baseline participant', 'Hannum DNAm')]):
            sub = grid[0, i].subgridspec(3, 1, height_ratios=[1, 0.20, 0.8])
            ax, note, bland = (fig.add_subplot(sub[j]) for j in range(3))
            work = points.loc[points.analysis.eq(label)].dropna(subset=['target_age', 'predicted_age'])
            row = self.row(metrics, analysis, *names[1])
            self.check_n(work, row)
            artists.append(self.scatter(ax, work, 'target_age', 'predicted_age',
                self.e._equal_limits(work.target_age, work.predicted_age),
                'Observed age (years)', 'Predicted age (years)'))
            ax.set_title(f'{short} age head', fontsize=11)
            note.set_axis_off()
            note.text(0.5, 0.5, f"n = {int(row.n):,}; MAE = {row.mae:.2f} years\n"
                f"R² = {row.r2:.3f}; r = {row.pearson_r:.3f}\n"
                f"Slope = {row.calibration_slope:.3f}; CCC = {row.ccc:.3f}",
                fontsize=9.5, ha='center', va='center', transform=note.transAxes)
            mean_age = (work.target_age + work.predicted_age) / 2
            difference = work.predicted_age - work.target_age
            bland.hexbin(mean_age, difference, gridsize=35, mincnt=1, cmap='viridis', linewidths=0)
            bias, sd = float(row.mean_error), float(row.sd_error)
            for offset in [0, -1.96 * sd, 1.96 * sd]:
                bland.axhline(bias + offset, color=self.e.REFERENCE,
                              linestyle='-' if offset == 0 else '--', linewidth=0.8)
            bland.set(xlabel='Mean observed/predicted age (years)', ylabel='Predicted − observed (years)')
            bland.set_title('Bland–Altman', fontsize=11)
            statistics[label] = {k: float(row[k]) for k in ['n', 'mae', 'r2', 'pearson_r',
                'calibration_slope', 'ccc', 'mean_error', 'sd_error']}
        self.density_key(fig, artists, fig.add_subplot(grid[1, :]))
        fig.suptitle('Figure S1A · Age-head diagnostics (statistics outside scatterplots)',
                     fontsize=12, fontweight='bold')
        return fig, self.record('A', names, fig.axes, len(points), comparisons=statistics)

    def build_05A(self):
        names = [('figure_05a_partial_residual_points.parquet', 39),
                 ('baseline_epigenetic_to_followup_retinal_associations.csv', 39)]
        points = self.read(*names[0], ['partial_residual_epigenetic',
            'partial_residual_followup_retinal', 'fitted', 'ci_low', 'ci_high']).sort_values(
                'partial_residual_epigenetic', kind='stable')
        table = self.read(*names[1])
        primary = table.loc[table.predictor.eq('z_epigenetic_mean_acceleration')]
        if len(primary) != 1:
            raise ValueError('Expected one primary mean-epigenetic longitudinal estimate.')
        row = primary.iloc[0]
        self.check_n(points, row)
        fig = plt.figure(figsize=(10, 5.7), layout='constrained')
        grid = fig.add_gridspec(1, 2, width_ratios=[1, 0.48])
        ax, side = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
        ax.scatter(points.partial_residual_epigenetic, points.partial_residual_followup_retinal,
                   s=10, color=self.e.EPIGENETIC, alpha=0.35, linewidths=0)
        ax.plot(points.partial_residual_epigenetic, points.fitted, color=self.e.EPIGENETIC)
        ax.fill_between(points.partial_residual_epigenetic, points.ci_low, points.ci_high,
                        color=self.e.EPIGENETIC, alpha=0.18, linewidth=0)
        ax.axhline(0, linestyle='--', color=self.e.REFERENCE, linewidth=0.8)
        ax.set(xlabel='Baseline mean epigenetic acceleration (SD)',
               ylabel='Follow-up retinal acceleration partial residual (years)')
        self.card(side, 'Saved primary estimate', [f'n = {int(row.n):,}',
            f"β = {self.e.fmt(row.coefficient, 'beta')} years / SD",
            f"95% CI: {self.e.fmt(row.ci_low, 'beta')} to {self.e.fmt(row.ci_high, 'beta')}",
            f"p = {self.e.fmt(row.p_value, 'p')}", f"Global q = {self.e.fmt(row.fdr_q_value, 'q')}"])
        fig.suptitle('Figure 5A · Baseline DNAm acceleration and follow-up retinal acceleration',
                     fontsize=12, fontweight='bold')
        stats = {k: float(row[k]) for k in ['n', 'coefficient', 'ci_low', 'ci_high', 'p_value', 'fdr_q_value']}
        return fig, self.record('A', names, fig.axes, len(points), primary=stats)

    def check_layout(self, fig, axes):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fig.canvas.draw()
        problems = [str(w.message) for w in caught if 'Glyph' in str(w.message)
                    or 'constrained_layout not applied' in str(w.message)]
        if problems:
            raise AssertionError('; '.join(problems))
        for text in fig.findobj(match=lambda a: isinstance(a, Text)):
            if text.get_visible() and text.get_text().strip() and text.get_fontsize() < 9:
                raise AssertionError(f'Font below 9 points: {text.get_text()!r}')
        self.e.assert_axes_nonoverlap(fig)
        # Require substantial scatter area, rather than merely checking font size.
        w, h = fig.get_size_inches()
        for ax in axes:
            if ax.get_aspect() == 1.0 and ax.collections:
                box = ax.get_position()
                if min(box.width * w, box.height * h) < 2.0:
                    raise AssertionError('Scatter plotting area below 2 inches; enlarge the canvas.')
        fig.set_layout_engine('none')

    def save(self, fig, fid, record, whole=False):
        self.e._number_manifest_check([record], self.output_root)
        axes = list(record.axes)
        # Preserve associated colorbars when exporting an individual panel.
        for ax in fig.axes:
            cb = getattr(ax, '_colorbar', None)
            if cb is not None and getattr(cb.mappable, 'axes', None) in axes and ax not in axes:
                axes.append(ax)
        self.check_layout(fig, axes)
        bbox = None if whole else self.e._group_bbox(fig, axes, pad_in=0.16)
        directory = self.output_root / f'figure_{fid}'
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f'figure_{fid}{record.panel}.png'
        with tempfile.TemporaryDirectory(prefix='clsa_fixed_png_') as stage:
            local = Path(stage) / target.name
            # Crop padding must not leak neighboring panel titles or axis labels.
            hidden = [ax for ax in fig.axes if ax not in axes and ax.get_visible()]
            for ax in hidden:
                ax.set_visible(False)
            try:
                fig.savefig(local, dpi=self.dpi, bbox_inches=bbox, facecolor='white',
                            transparent=False, metadata={'Software': 'CLSA saved-results PNG renderer'})
            finally:
                for ax in hidden:
                    ax.set_visible(True)
            shutil.copyfile(local, target)
        return {'figure_id': fid, 'panel': record.panel, 'path': str(target),
            'title': FIGURE_TITLES[fid], 'n': record.n,
            'sources': [{'path': str(self.statistics_root / s.file), 'cell': s.cell} for s in record.sources],
            'statistics': record.statistics, 'font': self.e.resolve_font(), 'dpi': self.dpi,
            'canvas_inches': fig.get_size_inches().tolist(),
            'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
            'notes': [t.get_text() for t in fig.texts if t.get_text().strip()]}

    def render(self, figure_ids=None):
        ids = list(FIGURE_TITLES) if figure_ids is None else [self.e._normalize_figure_id(i) for i in figure_ids]
        entries = []
        custom = {'01': {'B': self.build_01B}, '02': {'A': self.build_02A},
                  '04': {'A': lambda: self.build_single_scatter('A'),
                         'C': lambda: self.build_single_scatter('C')},
                  '05': {'A': self.build_05A},
                  'S1': {'A': self.build_S1A}}
        for fid in ids:
            if fid not in FIGURE_TITLES:
                raise ValueError(f'Unknown manuscript figure: {fid}')
            self.current_id = fid
            with mpl.rc_context(self.e._style(self.profile)):
                # Existing panel functions preserve all other selectors/statistics.
                fig, records = self.e.BUILDERS[fid](self.profile, self.output_root)
                try:
                    for record in records:
                        if record.panel not in custom.get(fid, {}):
                            entries.append(self.save(fig, fid, record))
                finally:
                    plt.close(fig)
                for panel, builder in custom.get(fid, {}).items():
                    fig, record = builder()
                    try:
                        entries.append(self.save(fig, fid, record, whole=True))
                    finally:
                        plt.close(fig)
            print(f'Figure {fid}: rebuilt PNG panels from saved statistics.')
        entries.sort(key=lambda r: (ids.index(r['figure_id']), r['panel']))
        self.output_root.mkdir(parents=True, exist_ok=True)
        manifest = {'statistics_root': str(self.statistics_root), 'model_fitting': False,
                    'statistical_tests_recomputed': False, 'settings': asdict(self.profile), 'panels': entries}
        with tempfile.TemporaryDirectory(prefix='clsa_png_index_') as stage:
            path = Path(stage) / 'fixed_png_manifest.json'
            path.write_text(json.dumps(self.e._json_clean(manifest), indent=2, ensure_ascii=False), encoding='utf-8')
            shutil.copyfile(path, self.output_root / path.name)
        return entries

def display_png_series(entries):
    from IPython.display import display, Markdown, Image
    current = None
    for entry in entries:
        if current != entry['figure_id']:
            current = entry['figure_id']
            display(Markdown(f"## Figure {current}: {entry['title']}"))
        display(Markdown(f"### Panel {entry['panel']} · `{Path(entry['path']).name}`"))
        display(Image(data=Path(entry['path']).read_bytes(), width=1100))
        source_lines = ', '.join(f"{Path(s['path']).name} (cell {s['cell']})" for s in entry['sources'])
        display(Markdown(f"Source: {source_lines}"))
