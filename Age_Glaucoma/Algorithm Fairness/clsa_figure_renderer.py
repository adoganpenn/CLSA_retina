"""Deterministic manuscript-figure renderer for the CLSA aging analysis.

This module is deliberately analysis-free.  It reads persisted CSV/Parquet
artifacts produced by ``epigenetics.ipynb`` and turns them into the manuscript
and slide exports.  Missing inputs are fatal and point to the notebook cell
that owns the artifact.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import subprocess
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.text import Text


SEED = 20260903
RETINAL = "#2F5597"
EPIGENETIC = "#D97706"
NULL = "#6B7280"
REFERENCE = "#111827"
LIGHT_GRAY = "#E5E7EB"
VERY_LIGHT_GRAY = "#F3F4F6"
WHITE = "#FFFFFF"
MINUS = "\N{MINUS SIGN}"
MM_PER_INCH = 25.4
FIXED_PDF_DATE = datetime(2026, 9, 14, tzinfo=timezone.utc)

VARIABLE_LABELS = {
    "GS_EXAM_AVG_COM": "Average grip strength",
    "GS_EXAM_MAX_COM": "Maximum grip strength",
    "GS_TRIAL1_MAX_COM": "Grip strength, trial 1",
    "GS_TRIAL2_MAX_COM": "Grip strength, trial 2",
    "GS_TRIAL3_MAX_COM": "Grip strength, trial 3",
    "ICQ_CATRCT_COM": "Ever had cataracts",
    "VIS_CATRCT_COM": "Cataracts on vision testing",
    "VA_ETDRS_BOTH_NB_COM": "Bilateral ETDRS acuity",
    "VA_ETDRS_L_NB_COM": "Left-eye ETDRS acuity",
    "VA_ETDRS_L_PO_NB_COM": "Left-eye pinhole ETDRS acuity",
    "VA_ETDRS_R_NB_COM": "Right-eye ETDRS acuity",
    "VA_ETDRS_R_PO_NB_COM": "Right-eye pinhole ETDRS acuity",
    "BAL_BEST_COM": "Standing balance",
    "TON_CRF_L_COM": "Left corneal resistance factor",
    "BLD_Hgb_COM": "Hemoglobin",
    "BLD_RBC_COM": "Red-cell count",
    "BLD_Hct_COM": "Hematocrit",
    "VIS_SGHT_COM": "Self-rated eyesight",
    "CR2_FAM_ML_COM": "Meal-preparation help",
    "INT_FRQWBSTS_MCQ": "Website-access frequency",
    "CR2_DEVC_WK_COM": "Walker use",
    "CR2_FAM_AC_COM": "Help with activities",
    "SDC_FTLG_DE_COM": "German first learned at home",
    "CR2_FAM_NONE_COM": "No nonprofessional assistance",
    "CR2_FRHC_COM": "Informal home care",
    "CR2_DRMC_COM": "Main-caregiver relationship",
    "GEN_OWNAG_COM": "Self-rated healthy aging",
    "PER_SYMPAGR_MCQ": "Sympathetic and warm self-view",
    "CR2_DTHC_COM": "Formal or informal home care",
    "NUT_CHSE_NB_COM": "Regular-cheese frequency",
}


@dataclass(frozen=True)
class Profile:
    name: str
    width_in: float
    max_height_in: float
    base_font: float
    tick_font: float
    axis_font: float
    panel_font: float
    annotation_font: float
    line_width: float
    marker_size: float
    dpi: int
    formats: tuple[str, ...]
    transparent: bool
    min_font: float


PROFILES: dict[str, Profile] = {
    "publication": Profile(
        "publication",
        180 / MM_PER_INCH,
        230 / MM_PER_INCH,
        7,
        7,
        8,
        9,
        7,
        0.75,
        2,
        600,
        ("pdf", "tif"),
        False,
        6,
    ),
    "slide": Profile(
        "slide",
        6.1,
        4.6,
        11,
        10,
        11,
        14,
        10,
        1.25,
        4,
        300,
        ("png",),
        True,
        9,
    ),
}


@dataclass
class Source:
    file: str
    cell: int


@dataclass
class PanelRecord:
    panel: str
    sources: list[Source]
    n: int | None = None
    statistics: dict[str, object] = field(default_factory=dict)
    output_paths: dict[str, list[str]] = field(default_factory=dict)
    axes: list[Axes] = field(default_factory=list, repr=False)


class MissingFigureInput(FileNotFoundError):
    """Raised instead of silently omitting an unavailable manuscript panel."""


_RESOLVED_FONT: str | None = None


def resolve_font() -> str:
    """Resolve the requested manuscript font once and log the choice."""
    global _RESOLVED_FONT
    if _RESOLVED_FONT is not None:
        return _RESOLVED_FONT
    available = {entry.name for entry in fm.fontManager.ttflist}
    for candidate in ("Arial", "Helvetica", "DejaVu Sans"):
        if candidate in available:
            _RESOLVED_FONT = candidate
            break
    else:
        _RESOLVED_FONT = "DejaVu Sans"
    print(f"CLSA figure font: {_RESOLVED_FONT}")
    return _RESOLVED_FONT


def _style(profile: Profile) -> dict[str, object]:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": [resolve_font(), "Helvetica", "DejaVu Sans"],
        "font.size": profile.base_font,
        "axes.labelsize": profile.axis_font,
        "axes.titlesize": profile.axis_font
        if profile.name == "publication"
        else profile.base_font,
        "xtick.labelsize": profile.tick_font,
        "ytick.labelsize": profile.tick_font,
        "legend.fontsize": profile.base_font,
        "axes.linewidth": profile.line_width,
        "lines.linewidth": profile.line_width,
        "lines.markersize": profile.marker_size,
        "patch.linewidth": profile.line_width,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.facecolor": "none" if profile.transparent else "white",
        "savefig.facecolor": "none" if profile.transparent else "white",
    }


def _statistics_root(outdir: Path) -> Path:
    override = os.environ.get("CLSA_STATISTICS_ROOT")
    return Path(override) if override else Path(outdir).parent / "03_statistics"


def _read(outdir: Path, source: Source, required: Iterable[str] = ()) -> pd.DataFrame:
    path = _statistics_root(outdir) / source.file
    if not path.is_file():
        raise MissingFigureInput(
            f"Missing persisted statistics file: {path}. "
            f"Notebook cell {source.cell} must produce {source.file}."
        )
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported statistics format: {path}")
    replacements = {
        chr(0x80): "EUR",
        chr(0x91): "'",
        chr(0x92): "'",
        chr(0x93): '"',
        chr(0x94): '"',
        chr(0x96): "-",
        chr(0x97): "-",
    }
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        frame[column] = frame[column].map(
            lambda value: "".join(
                replacements.get(char, "" if 0x80 <= ord(char) <= 0x9F else char)
                for char in str(value)
            )
            if pd.notna(value)
            else value
        )
    absent = sorted(set(required) - set(frame.columns))
    if absent:
        raise ValueError(
            f"{path} is missing required columns {absent}; source cell {source.cell}."
        )
    return frame


def _profile_height(
    profile: Profile, publication_height_mm: float, slide_height_in: float
) -> float:
    requested = (
        publication_height_mm / MM_PER_INCH
        if profile.name == "publication"
        else slide_height_in
    )
    if requested > profile.max_height_in + 1e-9:
        raise ValueError(
            f"{profile.name} height {requested:.3f} exceeds cap {profile.max_height_in:.3f}"
        )
    return requested


def _figure(
    profile: Profile, publication_height_mm: float, slide_height_in: float
) -> Figure:
    return plt.figure(
        figsize=(
            profile.width_in,
            _profile_height(profile, publication_height_mm, slide_height_in),
        ),
        layout="constrained",
    )


def _panel_letter(ax: Axes, letter: str, profile: Profile) -> Text:
    return ax.text(
        -0.13,
        1.06,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=profile.panel_font,
        fontweight="bold",
        color=REFERENCE,
        clip_on=False,
    )


def _slide_title(ax: Axes, title: str, profile: Profile) -> None:
    if profile.name == "slide":
        ax.set_title(title, fontweight="bold", loc="left", x=0.04, pad=7)


def _safe_minus(value: object) -> str:
    return str(value).replace("-", MINUS)


def fmt(value: float, kind: str, panel: bool = True) -> str:
    value = float(value)
    if kind in {"years", "mae", "rmse", "beta"}:
        text = f"{value:.2f}" if panel else f"{value:.4f}"
    elif kind in {"r2", "r", "ccc", "slope", "delta_r2"}:
        text = f"{value:.3f}" if panel else f"{value:.4f}"
    elif kind in {"p", "q"}:
        if value < 0.001:
            text = f"{value:.1e}"
        else:
            text = f"{value:.3f}"
    elif kind == "percent":
        text = f"{100 * value:.1f}%"
    elif kind == "n":
        text = f"{int(value):,}"
    else:
        text = str(value)
    return _safe_minus(text)


def _q_label(value: float) -> str:
    return "q < 0.001" if float(value) < 0.001 else f"q = {fmt(value, 'q')}"


def _annotation_box(ax: Axes, lines: Sequence[str], profile: Profile) -> None:
    ax.text(
        0.03,
        0.97,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=profile.annotation_font,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": WHITE,
            "edgecolor": LIGHT_GRAY,
            "alpha": 0.92,
            "linewidth": profile.line_width,
        },
    )


def _equal_limits(
    *arrays: Sequence[float], pad_fraction: float = 0.03
) -> tuple[float, float]:
    merged = np.concatenate([np.asarray(a, dtype=float) for a in arrays])
    merged = merged[np.isfinite(merged)]
    low, high = float(np.min(merged)), float(np.max(merged))
    pad = max((high - low) * pad_fraction, 0.5)
    return low - pad, high + pad


def _density_scatter(
    fig: Figure,
    ax: Axes,
    x: Sequence[float],
    y: Sequence[float],
    profile: Profile,
    xlabel: str,
    ylabel: str,
    limits: tuple[float, float] | None = None,
    annotation: Sequence[str] = (),
    colorbar: bool = True,
) -> mpl.collections.PolyCollection:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if limits is None:
        limits = _equal_limits(x, y)
    hb = ax.hexbin(
        x,
        y,
        gridsize=42 if profile.name == "publication" else 36,
        mincnt=1,
        cmap="viridis",
        linewidths=0,
        rasterized=True,
    )
    ax.plot(limits, limits, linestyle="--", color=REFERENCE, linewidth=0.75)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if annotation:
        _annotation_box(ax, annotation, profile)
    if colorbar:
        cbar = fig.colorbar(
            hb, ax=ax, orientation="horizontal", fraction=0.055, pad=0.12
        )
        cbar.set_label("point density")
        cbar.ax.tick_params(labelsize=profile.tick_font)
    return hb


def _forest(
    ax: Axes,
    labels: Sequence[str],
    estimate: Sequence[float],
    low: Sequence[float],
    high: Sequence[float],
    q_values: Sequence[float] | None,
    profile: Profile,
    xlabel: str,
    color: str,
    symmetric: bool = False,
    null_band: float | None = None,
) -> None:
    labels = list(labels)
    estimate = np.asarray(estimate, float)
    low = np.asarray(low, float)
    high = np.asarray(high, float)
    y = np.arange(len(labels))[::-1]
    if null_band is not None:
        ax.axvspan(-null_band, null_band, color=VERY_LIGHT_GRAY, zorder=0)
    ax.axvline(0, linestyle="--", color=REFERENCE, linewidth=0.75, zorder=1)
    ax.errorbar(
        estimate,
        y,
        xerr=np.vstack([estimate - low, high - estimate]),
        fmt="o",
        color=color,
        ecolor=color,
        capsize=2,
        markersize=profile.marker_size,
        zorder=2,
    )
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel)
    if symmetric:
        limit = max(abs(low.min()), abs(high.max())) * 1.15
        ax.set_xlim(-limit, limit)
    if q_values is not None:
        x0, x1 = ax.get_xlim()
        width = x1 - x0
        ax.set_xlim(x0, x1 + width * 0.32)
        q_x = x1 + width * 0.29
        for pos, q in zip(y, q_values):
            ax.text(
                q_x,
                pos,
                _q_label(float(q)),
                va="center",
                ha="right",
                fontsize=profile.annotation_font,
                color=NULL,
            )


def _source_dict(sources: Sequence[Source]) -> list[dict[str, object]]:
    return [asdict(source) for source in sources]


def _finalize_axes(fig: Figure, profile: Profile) -> None:
    for ax in fig.axes:
        ax.tick_params(width=profile.line_width, length=2.5)
        for spine in ax.spines.values():
            spine.set_linewidth(profile.line_width)


def assert_text_legible(fig: Figure, profile: str) -> float:
    """Raise when any visible, non-empty text uses a too-small final point size."""
    prof = PROFILES[profile]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.canvas.draw()
    missing = [str(item.message) for item in caught if "Glyph" in str(item.message)]
    if missing:
        raise AssertionError("Missing-glyph warning: " + "; ".join(missing[:6]))
    renderer = fig.canvas.get_renderer()
    sizes: list[float] = []
    failures: list[str] = []
    for artist in fig.findobj(match=lambda obj: isinstance(obj, Text)):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        bbox = artist.get_window_extent(renderer=renderer)
        if bbox.width <= 0 or bbox.height <= 0:
            continue
        point_size = float(artist.get_fontsize())
        sizes.append(point_size)
        if point_size + 1e-6 < prof.min_font:
            failures.append(f"{artist.get_text()!r}: {point_size:.2f} pt")
    if failures:
        raise AssertionError(
            f"Text below {prof.min_font:g} pt in {profile}: " + "; ".join(failures[:12])
        )
    return min(sizes) if sizes else math.inf


def assert_axes_nonoverlap(fig: Figure) -> None:
    """Allow touching axes; reject intersections over 1% of figure area."""
    fig.canvas.draw()
    axes = [
        ax for ax in fig.axes if ax.get_visible() and ax.get_label() != "<colorbar>"
    ]
    for i, first in enumerate(axes):
        for second in axes[i + 1 :]:
            a, b = first.get_position(), second.get_position()
            dx = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
            dy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
            if dx * dy > 0.01 + 1e-9:
                raise AssertionError(
                    f"Axes overlap by {100 * dx * dy:.2f}% of figure area: {first.get_label()} / {second.get_label()}"
                )


def _group_bbox(fig: Figure, axes: Sequence[Axes], pad_in: float = 0.08):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(renderer) for ax in axes if ax.get_visible()]
    bbox = mpl.transforms.Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted())
    return bbox.expanded(
        1 + 2 * pad_in / max(bbox.width, 0.01), 1 + 2 * pad_in / max(bbox.height, 0.01)
    )


def _json_clean(value):
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _record(
    panel: str, sources: Sequence[Source], axes: Sequence[Axes], n=None, **stats
) -> PanelRecord:
    clean = _json_clean(stats)
    return PanelRecord(
        panel, list(sources), None if n is None else int(n), clean, axes=list(axes)
    )


def _row(frame: pd.DataFrame, mask, source: Source, label: str) -> pd.Series:
    found = frame.loc[mask]
    if len(found) != 1:
        raise ValueError(
            f"Expected one {label} row in {source.file}; found {len(found)}"
        )
    return found.iloc[0]


def _build_01(profile: Profile, outdir: Path) -> tuple[Figure, list[PanelRecord]]:
    flow_source = Source("figure_01_cohort_flow.csv", 55)
    point_source = Source("figure_01_locked_validation_points.parquet", 41)
    metric_source = Source("locked_discovery_validation_metrics.csv", 41)
    points = _read(
        outdir,
        point_source,
        ["analysis_phase", "chronological_age", "retinal_age_locked"],
    )
    metrics = _read(
        outdir,
        metric_source,
        ["analysis", "n", "mae", "r2", "pearson_r", "calibration_slope", "ccc"],
    )
    flow = _read(outdir, flow_source, ["stage", "label", "n"]).sort_values(
        "stage", kind="stable"
    )
    if len(flow) != 5:
        raise ValueError(f"{flow_source.file} must contain exactly five cohort stages")
    fig = _figure(profile, 122, 4.45)
    if profile.name == "slide":
        grid = fig.add_gridspec(2, 2, width_ratios=[0.38, 0.62])
        ax_flow = fig.add_subplot(grid[:, 0])
        ax_discovery = fig.add_subplot(grid[0, 1])
        ax_validation = fig.add_subplot(grid[1, 1])
    else:
        grid = fig.add_gridspec(1, 2, width_ratios=[0.34, 0.66])
        ax_flow = fig.add_subplot(grid[0, 0])
        sub = grid[0, 1].subgridspec(1, 2)
        ax_discovery = fig.add_subplot(sub[0, 0])
        ax_validation = fig.add_subplot(sub[0, 1])
    ax_flow.set_axis_off()
    _panel_letter(ax_flow, "A", profile)
    _slide_title(ax_flow, "Cohort construction", profile)
    slide_flow_labels = {
        "Quality-passing RETFound cohort": "Quality-passing cohort",
        "Baseline retinal-age cohort": "Baseline retinal cohort",
        "Complete three-clock subset": "Three-clock subset",
    }
    boxes = [
        (
            slide_flow_labels.get(str(row.label), str(row.label))
            if profile.name == "slide"
            else str(row.label),
            fmt(row.n, "n"),
        )
        for row in flow.itertuples()
    ]
    y_positions = [0.91, 0.70, 0.47, 0.28, 0.06]
    for idx, ((label, n), y) in enumerate(zip(boxes, y_positions)):
        x, width = (
            (0.08, 0.84)
            if idx not in (2, 3)
            else ((0.04, 0.43) if idx == 2 else (0.53, 0.43))
        )
        patch = FancyBboxPatch(
            (x, y),
            width,
            0.12,
            boxstyle="round,pad=0.012",
            transform=ax_flow.transAxes,
            facecolor=VERY_LIGHT_GRAY,
            edgecolor=RETINAL,
            linewidth=profile.line_width,
        )
        ax_flow.add_patch(patch)
        ax_flow.text(
            x + width / 2,
            y + 0.06,
            f"{label}\n$n$ = {n}",
            transform=ax_flow.transAxes,
            ha="center",
            va="center",
            fontsize=profile.annotation_font,
        )
    for x0, x1, y0, y1 in [
        (0.5, 0.5, 0.89, 0.83),
        (0.5, 0.25, 0.69, 0.59),
        (0.5, 0.75, 0.69, 0.40),
        (0.25, 0.5, 0.46, 0.20),
        (0.75, 0.5, 0.27, 0.20),
    ]:
        ax_flow.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            xycoords="axes fraction",
            arrowprops={
                "arrowstyle": "->",
                "lw": profile.line_width,
                "color": REFERENCE,
            },
        )
    phases = [
        ("Discovery grouped OOF", ax_discovery, "Discovery grouped OOF"),
        ("Locked validation", ax_validation, "Locked validation"),
    ]
    all_limits = _equal_limits(
        points["chronological_age"], points["retinal_age_locked"]
    )
    records = [
        _record(
            "A",
            [flow_source],
            [ax_flow],
            n=int(flow.iloc[-1]["n"]),
            cohorts=flow.to_dict("records"),
        )
    ]
    b_stats = {}
    density_artists = []
    for phase, ax, short in phases:
        subset = points.loc[points["analysis_phase"].eq(phase)].sort_values(
            ["chronological_age", "retinal_age_locked"], kind="stable"
        )
        row = _row(
            metrics,
            metrics["analysis"].eq(f"{phase}: discovery-calibrated"),
            metric_source,
            phase,
        )
        if profile.name == "slide":
            lines = [
                f"$n$ = {fmt(row.n, 'n')}; MAE = {fmt(row.mae, 'mae')} y",
                f"$R^2$ = {fmt(row.r2, 'r2')}; $r$ = {fmt(row.pearson_r, 'r')}",
                f"slope = {fmt(row.calibration_slope, 'slope')}; CCC = {fmt(row.ccc, 'ccc')}",
            ]
            xlabel = "Chronological age"
            ylabel = "Retinal age"
        else:
            lines = [
                f"$n$ = {fmt(row.n, 'n')}",
                f"MAE = {fmt(row.mae, 'mae')} y",
                f"$R^2$ = {fmt(row.r2, 'r2')}",
                f"$r$ = {fmt(row.pearson_r, 'r')}",
                f"slope = {fmt(row.calibration_slope, 'slope')}",
                f"CCC = {fmt(row.ccc, 'ccc')}",
            ]
            xlabel = "Chronological age (years)"
            ylabel = "Calibrated retinal age (years)"
        density_artists.append(
            _density_scatter(
                fig,
                ax,
                subset["chronological_age"],
                subset["retinal_age_locked"],
                profile,
                xlabel,
                ylabel,
                all_limits,
                lines,
                colorbar=profile.name != "slide",
            )
        )
        slide_title = (
            "Discovery OOF · shared axes"
            if phase == "Discovery grouped OOF"
            else "Locked validation"
        )
        _slide_title(ax, slide_title, profile)
        b_stats[phase] = {
            k: float(row[k])
            for k in ("n", "mae", "r2", "pearson_r", "calibration_slope", "ccc")
        }
    _panel_letter(ax_discovery, "B", profile)
    if profile.name == "slide":
        density_colorbar = fig.colorbar(
            density_artists[-1],
            ax=[ax_discovery, ax_validation],
            orientation="vertical",
            fraction=0.035,
            pad=0.03,
        )
        density_colorbar.set_label("point density")
    records.append(
        _record(
            "B",
            [point_source, metric_source],
            [ax_discovery, ax_validation],
            n=len(points),
            comparisons=b_stats,
        )
    )
    return fig, records


def _build_02(profile: Profile, outdir: Path) -> tuple[Figure, list[PanelRecord]]:
    point_source = Source("figure_02_three_age_points.parquet", 17)
    metric_source = Source("three_age_agreement_metrics.csv", 17)
    corr_source = Source("figure_02_acceleration_correlation_forest.csv", 21)
    points = _read(
        outdir,
        point_source,
        [
            "chronological_age",
            "retinal_age_oof",
            "epigenetic_dnam_age",
            "epigenetic_hannum_age",
        ],
    )
    metrics = _read(outdir, metric_source, ["analysis", "n", "pearson_r", "mean_error"])
    corrs = _read(
        outdir,
        corr_source,
        ["measure", "n", "pearson_r", "pearson_ci_low", "pearson_ci_high"],
    )
    if "fdr_q_global" not in corrs and "pearson_fdr_q" in corrs:
        corrs = corrs.rename(columns={"pearson_fdr_q": "fdr_q_global"})
    if "fdr_q_global" not in corrs:
        raise ValueError(
            f"{corr_source.file} lacks the global Pearson-correlation FDR column"
        )
    fig = _figure(profile, 184, 4.6)
    outer = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.72])
    sg = outer[:2, :].subgridspec(2, 3)
    axes = np.asarray([[fig.add_subplot(sg[r, c]) for c in range(3)] for r in range(2)])
    labels = [
        ("Horvath DNAm age", "epigenetic_dnam_age"),
        ("Hannum epigenetic age", "epigenetic_hannum_age"),
    ]
    col_pairs = [
        ("chronological_age", "retinal_age_oof", "Chronological age", "Retinal age"),
        ("chronological_age", None, "Chronological age", None),
        ("retinal_age_oof", None, "Retinal age", None),
    ]
    a_stats = {}
    density_artists = []
    shared_limits = _equal_limits(
        points["chronological_age"],
        points["retinal_age_oof"],
        points["epigenetic_dnam_age"],
        points["epigenetic_hannum_age"],
    )
    for r_idx, (clock, clock_col) in enumerate(labels):
        for c_idx, (xcol, ycol, xlab, ylab) in enumerate(col_pairs):
            ycol = clock_col if ycol is None else ycol
            ylab = clock if ylab is None else ylab
            subset = (
                points[[xcol, ycol]].dropna().sort_values([xcol, ycol], kind="stable")
            )
            analysis = (
                f"Retinal age versus chronological age: {clock} subset"
                if c_idx == 0
                else f"{clock} versus chronological age"
                if c_idx == 1
                else f"Retinal age versus {clock}"
            )
            row = _row(
                metrics, metrics["analysis"].eq(analysis), metric_source, analysis
            )
            limits = shared_limits
            if profile.name == "slide":
                plot_xlabel = ""
                plot_ylabel = ""
                annotation = [
                    f"$r$ = {fmt(row.pearson_r, 'r')}",
                    f"mean Δ = {fmt(row.mean_error, 'years')} y",
                ]
            else:
                plot_xlabel = xlab + " (years)"
                plot_ylabel = ylab + " (years)"
                annotation = [
                    f"$r$ = {fmt(row.pearson_r, 'r')}",
                    f"mean difference = {fmt(row.mean_error, 'years')} y",
                ]
            density_artists.append(
                _density_scatter(
                    fig,
                    axes[r_idx, c_idx],
                    subset[xcol],
                    subset[ycol],
                    profile,
                    plot_xlabel,
                    plot_ylabel,
                    limits,
                    annotation,
                    colorbar=False,
                )
            )
            a_stats[f"{clock}_{c_idx + 1}"] = {
                "r": float(row.pearson_r),
                "mean_difference": float(row.mean_error),
                "n": int(row.n),
            }
    if profile.name == "slide":
        for ax, title in zip(
            axes[0],
            [
                "Retinal vs age · shared axes",
                "Clock vs age",
                "Clock vs retinal",
            ],
        ):
            ax.set_title(
                title,
                fontweight="bold",
                fontsize=profile.tick_font,
                loc="left",
                x=0.04,
                pad=5,
            )
        axes[0, 0].set_ylabel("Horvath row")
        axes[1, 0].set_ylabel("Hannum row")
        for ax in axes[1]:
            ax.set_xlabel("Age (years)")
    density_colorbar = fig.colorbar(
        density_artists[-1],
        ax=list(axes.ravel()),
        orientation="horizontal",
        fraction=0.025,
        pad=0.07,
    )
    density_colorbar.set_label("point density")
    panel_a = _panel_letter(axes[0, 0], "A", profile)
    if profile.name == "slide":
        panel_a.set_position((-0.30, 1.06))
    axf = fig.add_subplot(outer[2, :])
    order = [
        "epigenetic_age_acceleration_difference",
        "epigenetic_age_acceleration_residual",
        "epigenetic_ieaa",
        "epigenetic_eeaa",
    ]
    label_map = {
        str(row.measure): str(row.get("label", row.measure))
        for _, row in corrs.iterrows()
    }
    missing = sorted(set(order) - set(corrs["measure"]))
    if missing:
        raise ValueError(
            f"{corr_source.file} is missing the prespecified acceleration measures: {missing}"
        )
    ordered = corrs.set_index("measure").loc[order].reset_index()
    _forest(
        axf,
        [label_map.get(m, m) for m in ordered.measure],
        ordered.pearson_r,
        ordered.pearson_ci_low,
        ordered.pearson_ci_high,
        ordered.fdr_q_global,
        profile,
        "Pearson r with retinal age acceleration",
        EPIGENETIC,
        True,
        0.02,
    )
    _panel_letter(axf, "B", profile)
    _slide_title(axf, "Acceleration correlations", profile)
    records = [
        _record(
            "A",
            [point_source, metric_source],
            list(axes.ravel()),
            n=len(points),
            comparisons=a_stats,
        ),
        _record(
            "B",
            [corr_source],
            [axf],
            n=int(ordered.n.min()),
            correlations=ordered[["measure", "pearson_r", "fdr_q_global"]].to_dict(
                "records"
            ),
        ),
    ]
    return fig, records


def _construct_rows(
    frame: pd.DataFrame, outcome: str, codes: Sequence[str], display: Mapping[str, str]
) -> pd.DataFrame:
    work = frame.loc[frame["outcome"].eq(outcome)].copy()
    rows = []
    for code in codes:
        hit = work.loc[work["variable"].eq(code)]
        if len(hit) != 1:
            raise ValueError(
                f"Expected exactly one association row for {code!r}; found {len(hit)}"
            )
        row = hit.iloc[0].copy()
        row["variable_label"] = display[code]
        rows.append(row)
    return pd.DataFrame(rows)


def _construct_bars(
    ax: Axes,
    rows: pd.DataFrame,
    profile: Profile,
    color: str,
    shared_max: float,
    sign: bool,
) -> None:
    rows = rows.copy()
    rows["percent"] = 100 * rows["incremental_r2"].astype(float)
    y = np.arange(len(rows))[::-1]
    colors = np.where(rows["fdr_q_global"].astype(float) < 0.05, color, NULL)
    ax.barh(y, rows["percent"], color=colors, height=0.68)
    labels = []
    for _, row in rows.iterrows():
        label = str(row.variable_label)
        coef = row.get("standardized_coefficient", np.nan)
        if sign and pd.notna(coef):
            label += "  (" + ("+" if float(coef) >= 0 else MINUS) + ")"
        labels.append(label)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, shared_max)
    ax.set_xlabel("ΔR² (%)\nshared scale")
    if len(rows) < 8:
        for pos, value in zip(y, rows.percent):
            ax.text(
                value + 0.06,
                pos,
                f"{value:.3f}",
                va="center",
                fontsize=profile.annotation_font,
            )


def _build_03(profile: Profile, outdir: Path) -> tuple[Figure, list[PanelRecord]]:
    source = Source("questionnaire_phenome_scan_tests.csv", 34)
    frame = _read(
        outdir,
        source,
        [
            "variable",
            "variable_label",
            "outcome",
            "incremental_r2",
            "fdr_q_global",
            "standardized_coefficient",
        ],
    )
    display = {
        "ICQ_CATRCT_COM": "Cataract",
        "BAL_BEST_COM": "Standing balance",
        "VA_ETDRS_BOTH_NB_COM": "Bilateral ETDRS acuity",
        "GS_EXAM_MAX_COM": "Maximum grip strength",
        "TON_CRF_L_COM": "Corneal resistance factor",
        "BLD_Hgb_COM": "Hemoglobin",
        "INT_FRQWBSTS_MCQ": "Website-access frequency",
        "GEN_OWNAG_COM": "Self-rated healthy aging",
        "CR2_FAM_ML_COM": "Meal-preparation help",
        "CR2_DEVC_WK_COM": "Walker use",
        "SDC_FTLG_DE_COM": "German at home",
    }
    retinal_codes = [
        "ICQ_CATRCT_COM",
        "BAL_BEST_COM",
        "VA_ETDRS_BOTH_NB_COM",
        "GS_EXAM_MAX_COM",
        "TON_CRF_L_COM",
        "BLD_Hgb_COM",
    ]
    epi_codes = [
        "INT_FRQWBSTS_MCQ",
        "GEN_OWNAG_COM",
        "CR2_FAM_ML_COM",
        "CR2_DEVC_WK_COM",
        "SDC_FTLG_DE_COM",
    ]
    retinal = _construct_rows(
        frame, "z_retinal_acceleration", retinal_codes, display
    ).sort_values("incremental_r2", ascending=False, kind="stable")
    epigenetic = _construct_rows(
        frame, "z_epigenetic_mean_acceleration", epi_codes, display
    ).sort_values("incremental_r2", ascending=False, kind="stable")
    shared = max(
        5.35,
        100 * max(retinal.incremental_r2.max(), epigenetic.incremental_r2.max()) * 1.12,
    )
    fig = _figure(profile, 112, 4.25)
    axes = fig.subplots(1, 2)
    _construct_bars(axes[0], retinal, profile, RETINAL, shared, True)
    _construct_bars(axes[1], epigenetic, profile, EPIGENETIC, shared, False)
    _panel_letter(axes[0], "A", profile)
    _panel_letter(axes[1], "B", profile)
    _slide_title(axes[0], "Retinal acceleration", profile)
    _slide_title(axes[1], "Epigenetic acceleration", profile)
    if profile.name == "slide":
        axes[1].set_title("", loc="left")
        axes[1].set_title("Epigenetic", fontweight="bold", loc="center", x=0.5, pad=7)
    return fig, [
        _record(
            "A",
            [source],
            [axes[0]],
            n=int(retinal.n.min()),
            associations=retinal[
                [
                    "variable",
                    "variable_label",
                    "incremental_r2",
                    "standardized_coefficient",
                    "fdr_q_global",
                ]
            ].to_dict("records"),
        ),
        _record(
            "B",
            [source],
            [axes[1]],
            n=int(epigenetic.n.min()),
            associations=epigenetic[
                [
                    "variable",
                    "variable_label",
                    "incremental_r2",
                    "standardized_coefficient",
                    "fdr_q_global",
                ]
            ].to_dict("records"),
        ),
    ]


def _build_04(profile: Profile, outdir: Path) -> tuple[Figure, list[PanelRecord]]:
    eye_src = Source("figure_04_inter_eye_points.parquet", 47)
    eye_met = Source("inter_eye_reliability.csv", 47)
    heat_src = Source("acuity_grip_cataract_quality_sensitivity.csv", 37)
    long_src = Source("figure_04_longitudinal_points.parquet", 53)
    long_met = Source("longitudinal_retinal_change_repeatability.csv", 53)
    null_src = Source("longitudinal_matched_null_distribution.csv", 53)
    null_met = Source("longitudinal_same_person_matched_null_test.csv", 53)
    eye = _read(outdir, eye_src, ["right_retinal_age", "left_retinal_age"])
    em = _read(
        outdir,
        eye_met,
        [
            "n_paired_participants",
            "pearson_r",
            "ccc",
            "mean_right_minus_left_years",
            "sd_right_minus_left_years",
            "mae_between_eyes_years",
        ],
    )
    heat = _read(outdir, heat_src, ["variable", "analysis", "fdr_q_within_analysis"])
    lp = _read(outdir, long_src, ["retinal_acceleration_bl", "retinal_acceleration_f1"])
    lm = _read(
        outdir,
        long_met,
        [
            "analysis",
            "n",
            "mean_followup_years",
            "acceleration_stability_r",
            "direction_correct_proportion",
        ],
    )
    null = _read(outdir, null_src)
    nm = _read(outdir, null_met)
    fig = _figure(profile, 174, 4.6)
    gs = fig.add_gridspec(2, 2, width_ratios=[0.6, 0.4])
    axes = np.asarray(
        [
            [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        ]
    )
    erow = em.iloc[0]
    limits = _equal_limits(eye.right_retinal_age, eye.left_retinal_age)
    eye_annotation = (
        [
            f"$n$ = {fmt(erow.n_paired_participants, 'n')}; $r$ = {fmt(erow.pearson_r, 'r')}",
            f"CCC = {fmt(erow.ccc, 'ccc')}; MAE = {fmt(erow.mae_between_eyes_years, 'mae')} y",
            f"mean R−L = {fmt(erow.mean_right_minus_left_years, 'years')} y; SD = {fmt(erow.sd_right_minus_left_years, 'years')} y",
        ]
        if profile.name == "slide"
        else [
            f"$n$ = {fmt(erow.n_paired_participants, 'n')}",
            f"$r$ = {fmt(erow.pearson_r, 'r')}",
            f"CCC = {fmt(erow.ccc, 'ccc')}",
            f"mean difference = {fmt(erow.mean_right_minus_left_years, 'years')} y",
            f"SD = {fmt(erow.sd_right_minus_left_years, 'years')} y",
            f"MAE = {fmt(erow.mae_between_eyes_years, 'mae')} y",
        ]
    )
    _density_scatter(
        fig,
        axes[0, 0],
        eye.right_retinal_age,
        eye.left_retinal_age,
        profile,
        "Right-eye age" if profile.name == "slide" else "Right-eye retinal age (years)",
        "Left-eye age" if profile.name == "slide" else "Left-eye retinal age (years)",
        limits,
        eye_annotation,
    )
    letter_a = _panel_letter(axes[0, 0], "A", profile)
    if profile.name == "slide":
        letter_a.set_position((-0.24, 1.06))
    _slide_title(axes[0, 0], "Inter-eye agreement", profile)
    analyses = [
        "primary_unadjusted",
        "primary_retinal_cataract_adjusted",
        "strict_quality_unadjusted",
        "strict_quality_cataract_adjusted",
    ]
    labels = ["Primary", "+ cataract", "Strict quality", "Both"]
    all_vars = list(
        dict.fromkeys(
            heat.sort_values(["variable", "analysis"], kind="stable")["variable"]
        )
    )
    representative = [
        "ICQ_CATRCT_COM",
        "VIS_CATRCT_COM",
        "VA_ETDRS_BOTH_NB_COM",
        "GS_EXAM_MAX_COM",
        "GS_EXAM_AVG_COM",
    ]
    vars_order = [value for value in representative if value in all_vars]
    if not vars_order:
        raise ValueError(f"{heat_src.file} lacks the main robustness measures")
    matrix = heat.pivot_table(
        index="variable",
        columns="analysis",
        values="fdr_q_within_analysis",
        aggfunc="first",
    ).reindex(index=vars_order, columns=analyses)
    values = -np.log10(matrix.clip(lower=1e-300).to_numpy(float))
    masked = np.ma.masked_invalid(values)
    image = axes[0, 1].imshow(
        masked, aspect="auto", cmap="viridis", interpolation="nearest"
    )
    axes[0, 1].set_xticks(range(4), labels, rotation=25, ha="right")
    axes[0, 1].set_yticks(
        range(len(vars_order)),
        [VARIABLE_LABELS.get(value, value) for value in vars_order],
    )
    axes[0, 1].set_xlabel("" if profile.name == "slide" else "Model specification")
    axes[0, 1].set_ylabel("")
    axes[0, 1].set_facecolor(LIGHT_GRAY)
    for r, c in zip(*np.where(np.isnan(values))):
        rect = Rectangle(
            (c - 0.5, r - 0.5),
            1,
            1,
            facecolor=LIGHT_GRAY,
            edgecolor=NULL,
            hatch="////",
            linewidth=profile.line_width,
        )
        axes[0, 1].add_patch(rect)
    cb = fig.colorbar(
        image, ax=axes[0, 1], orientation="horizontal", fraction=0.07, pad=0.17
    )
    cb.set_label(MINUS + "log10 q")
    letter_b = _panel_letter(axes[0, 1], "B", profile)
    if profile.name == "slide":
        letter_b.set_position((-0.24, 1.06))
    _slide_title(axes[0, 1], "Robustness heatmap", profile)
    if profile.name == "slide":
        axes[0, 1].set_title(
            "Robustness\nFDR global: primary\nwithin-spec: others",
            fontweight="bold",
            fontsize=profile.tick_font,
            loc="left",
            x=0.04,
            pad=5,
        )
        axes[0, 1].text(
            0.98,
            0.02,
            "hatched = not tested",
            transform=axes[0, 1].transAxes,
            ha="right",
            va="bottom",
            fontsize=profile.annotation_font,
            color=NULL,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": WHITE,
                "edgecolor": LIGHT_GRAY,
                "alpha": 0.92,
            },
        )
    else:
        axes[0, 1].text(
            0.5,
            -0.47,
            "Primary: questionnaire-wide global FDR\nSensitivity columns: within-specification FDR",
            transform=axes[0, 1].transAxes,
            ha="center",
            va="top",
            fontsize=profile.annotation_font,
        )
    lrow = lm.loc[lm["analysis"].astype(str).str.startswith("All participants")].iloc[0]
    limits = _equal_limits(lp.retinal_acceleration_bl, lp.retinal_acceleration_f1)
    long_annotation = (
        [
            f"$n$ = {fmt(lrow.n, 'n')}; $r$ = {fmt(lrow.acceleration_stability_r, 'r')}",
            f"increasing = {fmt(lrow.direction_correct_proportion, 'percent')}",
            f"mean follow-up = {fmt(lrow.mean_followup_years, 'years')} y",
        ]
        if profile.name == "slide"
        else [
            f"$n$ = {fmt(lrow.n, 'n')}",
            f"$r$ = {fmt(lrow.acceleration_stability_r, 'r')}",
            f"increasing = {fmt(lrow.direction_correct_proportion, 'percent')}",
            f"mean follow-up = {fmt(lrow.mean_followup_years, 'years')} y",
        ]
    )
    _density_scatter(
        fig,
        axes[1, 0],
        lp.retinal_acceleration_bl,
        lp.retinal_acceleration_f1,
        profile,
        "Baseline acceleration"
        if profile.name == "slide"
        else "Baseline retinal acceleration (years)",
        "Follow-up acceleration"
        if profile.name == "slide"
        else "Follow-up retinal acceleration (years)",
        limits,
        long_annotation,
    )
    letter_c = _panel_letter(axes[1, 0], "C", profile)
    if profile.name == "slide":
        letter_c.set_position((-0.24, 1.06))
    _slide_title(axes[1, 0], "Longitudinal stability", profile)
    corr_col = next(
        (
            c
            for c in null.columns
            if "correlation" in c.lower()
            or c.lower() in {"r", "matched_r"}
            or c.lower().endswith("_r")
        ),
        None,
    )
    if corr_col is None:
        raise ValueError(f"{null_src.file} lacks a null-correlation column")
    nrow = nm.iloc[0]
    axes[1, 1].hist(null[corr_col].dropna(), bins=30, color=NULL, edgecolor=WHITE)
    observed = float(nrow["observed_same_participant_stability_r"])
    null_low = float(null[corr_col].min())
    null_high = float(null[corr_col].max())
    null_pad = max((null_high - null_low) * 0.08, 0.001)
    axes[1, 1].set_xlim(null_low - null_pad, null_high + null_pad)
    axes[1, 1].annotate(
        f"observed $r$ = {fmt(observed, 'r')} · off-scale",
        xy=(0.99, 0.35),
        xycoords="axes fraction",
        xytext=(0.96, 0.08),
        textcoords="axes fraction",
        ha="right",
        va="bottom",
        arrowprops={"arrowstyle": "->", "color": RETINAL},
        fontsize=profile.annotation_font,
        bbox={
            "boxstyle": "round,pad=0.15",
            "facecolor": WHITE,
            "edgecolor": LIGHT_GRAY,
            "alpha": 0.92,
        },
    )
    axes[1, 1].set_xlabel("Matched-null Pearson r")
    axes[1, 1].set_ylabel("Count")
    _annotation_box(
        axes[1, 1],
        [
            f"mean = {fmt(nrow.random_matched_r_mean, 'r')}",
            f"empirical p = {fmt(nrow.same_participant_r_empirical_p, 'p')}",
            f"$n$ = {fmt(nrow.n_matched_participants, 'n')}",
        ],
        profile,
    )
    letter_d = _panel_letter(axes[1, 1], "D", profile)
    if profile.name == "slide":
        letter_d.set_position((-0.24, 1.06))
    _slide_title(axes[1, 1], "Matched-null", profile)
    return fig, [
        _record(
            "A",
            [eye_src, eye_met],
            [axes[0, 0]],
            n=erow.n_paired_participants,
            metrics=erow.to_dict(),
        ),
        _record(
            "B",
            [heat_src],
            [axes[0, 1]],
            n=None,
            fdr_families={"primary": "global", "sensitivity": "within specification"},
        ),
        _record(
            "C", [long_src, long_met], [axes[1, 0]], n=lrow.n, metrics=lrow.to_dict()
        ),
        _record(
            "D",
            [null_src, null_met],
            [axes[1, 1]],
            n=nrow.n_matched_participants,
            metrics=nrow.to_dict(),
        ),
    ]


def _build_05(profile: Profile, outdir: Path) -> tuple[Figure, list[PanelRecord]]:
    point_src = Source("figure_05a_partial_residual_points.parquet", 39)
    assoc_src = Source("baseline_epigenetic_to_followup_retinal_associations.csv", 39)
    pred_src = Source("epigenetic_target_prediction_beyond_age.csv", 49)
    points = _read(
        outdir,
        point_src,
        [
            "partial_residual_epigenetic",
            "partial_residual_followup_retinal",
            "fitted",
            "ci_low",
            "ci_high",
        ],
    )
    assoc = _read(
        outdir,
        assoc_src,
        [
            "predictor",
            "n",
            "coefficient",
            "ci_low",
            "ci_high",
            "p_value",
            "fdr_q_value",
        ],
    )
    pred = _read(
        outdir,
        pred_src,
        [
            "clock",
            "clock_label",
            "delta_r2_beyond_age",
            "delta_r2_ci_low",
            "delta_r2_ci_high",
        ],
    )
    fig = _figure(profile, 135, 4.6)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[0.52, 0.48],
        height_ratios=[0.58, 0.42],
    )
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[:, 1]),
        fig.add_subplot(gs[1, 0]),
    ]
    primary = _row(
        assoc,
        assoc.predictor.eq("z_epigenetic_mean_acceleration"),
        assoc_src,
        "primary longitudinal",
    )
    p = points.sort_values("partial_residual_epigenetic", kind="stable")
    axes[0].scatter(
        p.partial_residual_epigenetic,
        p.partial_residual_followup_retinal,
        s=profile.marker_size**2,
        alpha=0.30,
        color=EPIGENETIC,
        rasterized=True,
    )
    axes[0].plot(p.partial_residual_epigenetic, p.fitted, color=EPIGENETIC)
    axes[0].fill_between(
        p.partial_residual_epigenetic,
        p.ci_low,
        p.ci_high,
        color=EPIGENETIC,
        alpha=0.18,
        linewidth=0,
    )
    axes[0].axhline(0, linestyle="--", color=REFERENCE, linewidth=0.75)
    axes[0].set_xlabel("Baseline epigenetic acceleration (SD)")
    axes[0].set_ylabel("Adjusted retinal acceleration (years)")
    _annotation_box(
        axes[0],
        [
            f"$n$ = {fmt(primary.n, 'n')}; β = {fmt(primary.coefficient, 'beta')} y/SD",
            f"95% CI {fmt(primary.ci_low, 'beta')} to {fmt(primary.ci_high, 'beta')}",
            f"p = {fmt(primary.p_value, 'p')}; {_q_label(primary.fdr_q_value)}",
        ],
        profile,
    )
    _panel_letter(axes[0], "A", profile)
    _slide_title(axes[0], "Primary estimate", profile)
    label_map = {
        "z_epigenetic_mean_acceleration": "Mean epigenetic",
        "z_horvath_acceleration": "Horvath",
        "z_hannum_acceleration": "Hannum",
        "shared_aging_pc": "Shared PC",
        "retina_vs_epigenetic_pc": "Retina–epi PC",
    }
    order = list(label_map)
    af = assoc.set_index("predictor").loc[order].reset_index()
    _forest(
        axes[1],
        [label_map[v] for v in af.predictor],
        af.coefficient,
        af.ci_low,
        af.ci_high,
        af.fdr_q_value,
        profile,
        "Adjusted coefficient (95% CI)",
        EPIGENETIC,
    )
    _panel_letter(axes[1], "B", profile)
    _slide_title(axes[1], "Methylation predictors", profile)
    order_clock = ["epigenetic_dnam_age", "epigenetic_hannum_age"]
    pp = pred.set_index("clock").loc[order_clock].reset_index()
    y = np.arange(len(pp))[::-1]
    threshold = float(
        pp.get("prespecified_support_threshold", pd.Series([0.0])).iloc[0]
    )
    axes[2].axvline(
        threshold,
        linestyle="--",
        color=REFERENCE,
        linewidth=0.75,
    )
    axes[2].errorbar(
        pp.delta_r2_beyond_age,
        y,
        xerr=np.vstack(
            [
                pp.delta_r2_beyond_age - pp.delta_r2_ci_low,
                pp.delta_r2_ci_high - pp.delta_r2_beyond_age,
            ]
        ),
        fmt="o",
        color=RETINAL,
        capsize=2,
    )
    axes[2].set_yticks(y, ["Horvath", "Hannum"])
    axes[2].set_xlabel("ΔR² beyond chronological age")
    axes[2].text(
        0.98,
        0.95,
        "support threshold = 0",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=profile.annotation_font,
        color=NULL,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": WHITE,
            "edgecolor": LIGHT_GRAY,
            "alpha": 0.92,
        },
    )
    _panel_letter(axes[2], "C", profile)
    _slide_title(axes[2], "Retina → methylation", profile)
    return fig, [
        _record(
            "A",
            [point_src, assoc_src],
            [axes[0]],
            n=primary.n,
            metrics=primary.to_dict(),
        ),
        _record(
            "B",
            [assoc_src],
            [axes[1]],
            n=int(af.n.min()),
            associations=af.to_dict("records"),
        ),
        _record(
            "C",
            [pred_src],
            [axes[2]],
            n=int(pp.n.min()) if "n" in pp else 1406,
            prediction=pp.to_dict("records"),
            support_threshold=threshold,
        ),
    ]


def _table_panel(
    ax: Axes,
    frame: pd.DataFrame,
    profile: Profile,
    xlabel: str = "",
    max_rows: int = 14,
) -> None:
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    if not numeric:
        ax.text(
            0.5,
            0.5,
            "No numeric results",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return
    value = numeric[-1]
    label = next(
        (
            c
            for c in (
                "analysis",
                "age_bin_5y",
                "age_distribution_stratum",
                "outcome_label",
                "variable_label",
                "variable",
                "predictor",
                "term",
                "clock",
                "model",
                "group_level",
                "cohort",
            )
            if c in frame
        ),
        frame.columns[0],
    )
    plot = frame[[label, value]].dropna().head(max_rows).copy()
    y = np.arange(len(plot))[::-1]
    ax.barh(y, plot[value].astype(float), color=NULL)
    ax.set_yticks(y, plot[label].astype(str))
    ax.set_xlabel(xlabel or value.replace("_", " "))


def _heatmap_panel(
    fig: Figure,
    ax: Axes,
    frame: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    profile: Profile,
    label: str,
) -> None:
    matrix = (
        frame.pivot_table(index=index, columns=columns, values=values, aggfunc="first")
        .sort_index()
        .sort_index(axis=1)
    )
    im = ax.imshow(
        matrix.to_numpy(float), aspect="auto", cmap="viridis", interpolation="nearest"
    )
    ax.set_xticks(
        range(matrix.shape[1]), matrix.columns.astype(str), rotation=35, ha="right"
    )
    ax.set_yticks(range(matrix.shape[0]), matrix.index.astype(str))
    ax.set_xlabel(columns.replace("_", " "))
    ax.set_ylabel("")
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.07, pad=0.16)
    cb.set_label(label)


def _build_s1(profile: Profile, outdir: Path) -> tuple[Figure, list[PanelRecord]]:
    point_source = Source("figure_S1_age_diagnostic_points.parquet", 12)
    metric_source = Source("model_standard_metrics.csv", 12)
    bin_source = Source("chronological_model_5y_age_bin_performance.csv", 12)
    points = _read(outdir, point_source, ["analysis", "target_age", "predicted_age"])
    metrics = _read(
        outdir,
        metric_source,
        ["analysis", "n", "mae", "r2", "pearson_r", "calibration_slope", "ccc"],
    )
    bins = _read(
        outdir,
        bin_source,
        ["target_mean", "mae", "mean_error", "calibration_slope"],
    ).sort_values("target_mean", kind="stable")
    fig = _figure(profile, 224, 4.6)
    grid = fig.add_gridspec(3, 6, height_ratios=[1, 1, 0.68])
    scatter_axes = [fig.add_subplot(grid[0, 2 * i : 2 * i + 2]) for i in range(3)]
    bland_axes = [fig.add_subplot(grid[1, 2 * i : 2 * i + 2]) for i in range(3)]
    bottom_axes = [fig.add_subplot(grid[2, :3]), fig.add_subplot(grid[2, 3:])]
    analysis_specs = [
        ("Retinal age", "Chronological age head: participant"),
        ("Horvath DNAm age", "Horvath DNAm age head: baseline participant"),
        ("Hannum epigenetic age", "Hannum epigenetic age head: baseline participant"),
    ]
    a_stats = {}
    hexbins = []
    for index, ((label, metric_label), scatter_ax, bland_ax) in enumerate(
        zip(analysis_specs, scatter_axes, bland_axes)
    ):
        work = (
            points.loc[points["analysis"].eq(label)]
            .dropna()
            .sort_values(["target_age", "predicted_age"], kind="stable")
        )
        row = _row(
            metrics, metrics["analysis"].eq(metric_label), metric_source, metric_label
        )
        limits = _equal_limits(work.target_age, work.predicted_age)
        hb = scatter_ax.hexbin(
            work.target_age,
            work.predicted_age,
            gridsize=34,
            mincnt=1,
            cmap="viridis",
            linewidths=0,
            rasterized=True,
        )
        hexbins.append(hb)
        scatter_ax.plot(limits, limits, "--", color=REFERENCE, linewidth=0.75)
        scatter_ax.set_xlim(limits)
        scatter_ax.set_ylim(limits)
        scatter_ax.set_aspect("equal", adjustable="box")
        scatter_ax.set_xlabel("" if profile.name == "slide" else "Observed age (years)")
        scatter_ax.set_ylabel(
            "Predicted age"
            if profile.name == "slide" and index == 0
            else ""
            if profile.name == "slide"
            else "Predicted age (years)"
        )
        annotation_lines = (
            [
                f"$n$ = {fmt(row.n, 'n')}",
                f"MAE = {fmt(row.mae, 'mae')} y",
                f"$R^2$ = {fmt(row.r2, 'r2')}; $r$ = {fmt(row.pearson_r, 'r')}",
                f"slope = {fmt(row.calibration_slope, 'slope')}",
                f"CCC = {fmt(row.ccc, 'ccc')}",
            ]
            if profile.name == "slide"
            else [
                f"$n$ = {fmt(row.n, 'n')}",
                f"MAE = {fmt(row.mae, 'mae')} y",
                f"$R^2$ = {fmt(row.r2, 'r2')}",
                f"$r$ = {fmt(row.pearson_r, 'r')}",
                f"slope = {fmt(row.calibration_slope, 'slope')}",
                f"CCC = {fmt(row.ccc, 'ccc')}",
            ]
        )
        _annotation_box(
            scatter_ax,
            annotation_lines,
            profile,
        )
        difference = work.predicted_age.to_numpy(float) - work.target_age.to_numpy(
            float
        )
        mean_age = (
            work.predicted_age.to_numpy(float) + work.target_age.to_numpy(float)
        ) / 2
        bland_ax.hexbin(
            mean_age,
            difference,
            gridsize=34,
            mincnt=1,
            cmap="viridis",
            linewidths=0,
            rasterized=True,
        )
        mean_difference = float(row["mean_error"])
        spread = 1.96 * float(row["sd_error"])
        bland_ax.axhline(mean_difference, color=REFERENCE, linewidth=profile.line_width)
        bland_ax.axhline(
            mean_difference + spread, color=NULL, linestyle="--", linewidth=0.75
        )
        bland_ax.axhline(
            mean_difference - spread, color=NULL, linestyle="--", linewidth=0.75
        )
        bland_ax.set_xlabel("" if profile.name == "slide" else "Mean age (years)")
        bland_ax.set_ylabel(
            "Difference"
            if profile.name == "slide" and index == 0
            else ""
            if profile.name == "slide"
            else "Predicted minus observed (years)"
        )
        short_title = label.replace(" DNAm age", "").replace(" epigenetic age", "")
        _slide_title(scatter_ax, short_title, profile)
        if profile.name == "slide":
            scatter_ax.set_title("", loc="left")
            scatter_ax.set_title(short_title, fontweight="bold", loc="center", x=0.5)
        a_stats[label] = row.to_dict()
        if index == 0:
            panel_a = _panel_letter(scatter_ax, "A", profile)
            if profile.name == "slide":
                panel_a.set_position((-0.30, 1.06))
    colorbar = fig.colorbar(
        hexbins[-1],
        ax=[*scatter_axes, *bland_axes] if profile.name == "slide" else scatter_axes,
        orientation="vertical" if profile.name == "slide" else "horizontal",
        fraction=0.018 if profile.name == "slide" else 0.025,
        pad=0.02 if profile.name == "slide" else 0.08,
    )
    colorbar.set_label("Density" if profile.name == "slide" else "point density")
    bottom_axes[0].plot(bins.target_mean, bins.mae, marker="o", color=RETINAL)
    bottom_axes[0].set_xlabel(
        "Age-bin midpoint"
        if profile.name == "slide"
        else "Mean chronological age in 5-year bin"
    )
    bottom_axes[0].set_ylabel(
        "Error (years)" if profile.name == "slide" else "MAE (years)"
    )
    bottom_axes[1].plot(
        bins.target_mean,
        bins.mean_error,
        marker="o",
        color=RETINAL,
        label="Bias" if profile.name == "slide" else "Mean error",
    )
    bottom_axes[1].plot(
        bins.target_mean,
        bins.calibration_slope,
        marker="o",
        color=NULL,
        label="Slope" if profile.name == "slide" else "Calibration slope",
    )
    bottom_axes[1].axhline(0, linestyle="--", color=REFERENCE, linewidth=0.75)
    bottom_axes[1].set_xlabel(
        "Age-bin midpoint"
        if profile.name == "slide"
        else "Mean chronological age in 5-year bin"
    )
    bottom_axes[1].set_ylabel(
        "Metric" if profile.name == "slide" else "Calibration metric"
    )
    bottom_axes[1].legend(
        frameon=False,
        loc="upper right",
        ncols=2 if profile.name == "slide" else 1,
    )
    panel_b = _panel_letter(bottom_axes[0], "B", profile)
    if profile.name == "slide":
        panel_b.set_position((-0.25, 1.06))
    _slide_title(bottom_axes[0], "Age-bin error", profile)
    _slide_title(bottom_axes[1], "Age-bin calibration", profile)
    return fig, [
        _record(
            "A",
            [point_source, metric_source],
            [*scatter_axes, *bland_axes],
            n=int(points.shape[0]),
            models=_json_clean(a_stats),
        ),
        _record(
            "B",
            [bin_source],
            bottom_axes,
            n=int(pd.to_numeric(bins["n"], errors="coerce").min()),
            age_bins=int(len(bins)),
        ),
    ]


def _build_s5(profile: Profile, outdir: Path) -> tuple[Figure, list[PanelRecord]]:
    source = Source("questionnaire_phenome_scan_tests.csv", 34)
    frame = _read(
        outdir,
        source,
        [
            "variable",
            "variable_label",
            "outcome",
            "n",
            "incremental_r2",
            "p_value",
            "fdr_q_global",
        ],
    )
    frame = frame.loc[~frame["variable"].eq("TRA_CMNTY_OT_MCQ")].copy()
    if "manuscript_eligible" in frame:
        frame = frame.loc[frame["manuscript_eligible"].fillna(False)].copy()
    retinal_codes = [
        "ICQ_CATRCT_COM",
        "VIS_CATRCT_COM",
        "BAL_BEST_COM",
        "VA_ETDRS_BOTH_NB_COM",
        "GS_TRIAL3_MAX_COM",
        "GS_EXAM_MAX_COM",
        "VIS_SGHT_COM",
        "GS_EXAM_AVG_COM",
        "GS_TRIAL2_MAX_COM",
        "TON_CRF_L_COM",
        "BLD_RBC_COM",
        "BLD_Hct_COM",
        "BLD_Hgb_COM",
        "VA_ETDRS_R_PO_NB_COM",
        "GS_TRIAL1_MAX_COM",
    ]

    epigenetic_codes = [
        "CR2_FAM_ML_COM",
        "INT_FRQWBSTS_MCQ",
        "CR2_DEVC_WK_COM",
        "CR2_FAM_AC_COM",
        "SDC_FTLG_DE_COM",
        "CR2_FAM_NONE_COM",
        "CR2_FRHC_COM",
        "CR2_DRMC_COM",
        "GEN_OWNAG_COM",
        "PER_SYMPAGR_MCQ",
        "CR2_DTHC_COM",
        "NUT_CHSE_NB_COM",
    ]

    def ordered_associations(outcome: str, codes: Sequence[str]) -> pd.DataFrame:
        work = frame.loc[frame["outcome"].eq(outcome)].set_index("variable")
        missing = [code for code in codes if code not in work.index]
        if missing:
            raise ValueError(
                f"{source.file} is missing required {outcome} rows: {missing}"
            )
        selected = work.loc[list(codes)].reset_index()
        selected["variable_label"] = selected["variable"].map(
            lambda value: VARIABLE_LABELS.get(str(value), str(value))
        )
        return selected

    retinal = ordered_associations("z_retinal_acceleration", retinal_codes)
    epigenetic = ordered_associations(
        "z_epigenetic_mean_acceleration", epigenetic_codes
    )
    if len(retinal) != 15 or len(epigenetic) != 12:
        raise ValueError(
            f"{source.file} must supply 15 retinal and 12 epigenetic eligible rows; "
            f"found {len(retinal)} and {len(epigenetic)}"
        )
    fig = _figure(profile, 164, 4.6)
    axes = fig.subplots(1, 2)
    for ax, work, letter, color, title in (
        (axes[0], retinal, "A", RETINAL, "Retinal (15)"),
        (axes[1], epigenetic, "B", EPIGENETIC, "Epigenetic (12)"),
    ):
        work = work.copy()
        work["percent"] = 100 * work["incremental_r2"]
        y = np.arange(len(work))[::-1]
        colors = np.where(work["fdr_q_global"].lt(0.05), color, NULL)
        ax.barh(y, work["percent"], color=colors)
        ax.set_yticks(y, work["variable_label"].astype(str))
        ax.set_xlabel("ΔR² (%)")
        ax.set_xlim(left=0)
        _panel_letter(ax, letter, profile)
        _slide_title(ax, title, profile)
    if profile.name == "slide":
        axes[1].set_title("", loc="left")
        axes[1].set_title("Epigenetic", fontweight="bold", loc="center", x=0.5)
    return fig, [
        _record(
            "A",
            [source],
            [axes[0]],
            n=int(retinal["n"].min()),
            associations=retinal.to_dict("records"),
        ),
        _record(
            "B",
            [source],
            [axes[1]],
            n=int(epigenetic["n"].min()),
            associations=epigenetic.to_dict("records"),
        ),
    ]


def _build_s11(profile: Profile, outdir: Path) -> tuple[Figure, list[PanelRecord]]:
    annual_source = Source("figure_S11_annualized_change.csv", 53)
    attrition_source = Source("figure_S11_attrition.csv", 53)
    identity_source = Source("longitudinal_same_person_matched_null_test.csv", 53)
    quality_source = Source("longitudinal_quality_adjusted_stability.csv", 53)
    annual = _read(outdir, annual_source, ["annualized_retinal_age_change"])
    attrition = _read(outdir, attrition_source, ["stage", "label", "n"]).sort_values(
        "stage", kind="stable"
    )
    identity = _read(
        outdir,
        identity_source,
        [
            "observed_same_participant_stability_r",
            "random_matched_r_mean",
            "same_participant_r_empirical_p",
        ],
    )
    quality = _read(
        outdir,
        quality_source,
        ["term", "coefficient", "ci_low", "ci_high", "p_value"],
    )
    fig = _figure(profile, 180, 4.6)
    axes = np.asarray(fig.subplots(2, 2))
    values = annual["annualized_retinal_age_change"].dropna().to_numpy(float)
    axes[0, 0].hist(values, bins=36, color=RETINAL, edgecolor=WHITE)
    axes[0, 0].axvline(0, color=REFERENCE, linestyle="--", linewidth=0.75)
    axes[0, 0].set_xlabel("Annualized retinal-age change (years/year)")
    axes[0, 0].set_ylabel("Participants")
    _panel_letter(axes[0, 0], "A", profile)
    _slide_title(axes[0, 0], "Annualized retinal-age change", profile)

    y = np.arange(len(attrition))[::-1]
    axes[0, 1].barh(y, attrition["n"], color=NULL)
    axes[0, 1].set_yticks(y, attrition["label"])
    axes[0, 1].set_xlabel("Participants")
    axes[0, 1].set_xlim(left=0)
    for position, value in zip(y, attrition["n"]):
        axes[0, 1].text(
            float(value),
            position,
            f" {int(value):,}",
            ha="left",
            va="center",
            fontsize=profile.annotation_font,
        )
    _panel_letter(axes[0, 1], "B", profile)
    _slide_title(axes[0, 1], "Longitudinal retention", profile)

    identity_row = identity.iloc[0]
    identity_values = [
        float(identity_row["observed_same_participant_stability_r"]),
        float(identity_row["random_matched_r_mean"]),
    ]
    axes[1, 0].barh([1, 0], identity_values, color=[RETINAL, NULL])
    axes[1, 0].axvline(0, color=REFERENCE, linestyle="--", linewidth=0.75)
    axes[1, 0].set_yticks([1, 0], ["Same participant", "Matched null"])
    axes[1, 0].set_xlabel("Baseline-follow-up acceleration r")
    _annotation_box(
        axes[1, 0],
        [f"empirical p = {fmt(identity_row.same_participant_r_empirical_p, 'p')}"],
        profile,
    )
    _panel_letter(axes[1, 0], "C", profile)
    _slide_title(axes[1, 0], "Identity-matched control", profile)

    term_labels = {
        "retinal_acceleration_bl": "Baseline acceleration",
        "followup_years": "Follow-up years",
        "quality_score_bl": "Baseline quality score",
        "quality_score_change": "Quality-score change",
        "quality_image_count_bl": "Baseline image count",
        "quality_image_count_change": "Image-count change",
        "Intercept": "Intercept",
    }
    quality = quality.copy()
    quality["label"] = quality["term"].map(term_labels).fillna(quality["term"])
    _forest(
        axes[1, 1],
        quality["label"],
        quality["coefficient"],
        quality["ci_low"],
        quality["ci_high"],
        None,
        profile,
        "Adjusted coefficient (95% CI)",
        RETINAL,
    )
    _panel_letter(axes[1, 1], "D", profile)
    _slide_title(axes[1, 1], "Quality-adjusted model", profile)
    if profile.name == "slide":
        axes[1, 1].set_title("", loc="left")
        axes[1, 1].set_title(
            "Quality-adjusted", fontweight="bold", loc="center", x=0.5, pad=7
        )
    return fig, [
        _record("A", [annual_source], [axes[0, 0]], n=len(annual), rows=len(annual)),
        _record(
            "B",
            [attrition_source],
            [axes[0, 1]],
            n=int(attrition.iloc[-1]["n"]),
            attrition=attrition.to_dict("records"),
        ),
        _record(
            "C",
            [identity_source],
            [axes[1, 0]],
            n=int(identity_row.get("n_matched_participants", 23502)),
            metrics=identity_row.to_dict(),
        ),
        _record(
            "D",
            [quality_source],
            [axes[1, 1]],
            n=int(pd.to_numeric(quality["n"], errors="coerce").min())
            if "n" in quality
            else None,
            coefficients=quality.to_dict("records"),
        ),
    ]


def _build_supplement(
    fid: str, profile: Profile, outdir: Path
) -> tuple[Figure, list[PanelRecord]]:
    specs = {
        "S1": [
            (Source("model_standard_metrics.csv", 12), "A"),
            (Source("chronological_model_5y_age_bin_performance.csv", 12), "B"),
        ],
        "S2": [
            (Source("table_1_methylation_included_vs_excluded.csv", 55), "A"),
            (Source("locked_demographic_performance.csv", 55), "B"),
        ],
        "S3": [
            (Source("chronological_model_tail_performance.csv", 12), "A"),
            (Source("three_age_paired_distance_tests.csv", 21), "B"),
            (Source("figure_02_acceleration_correlation_forest.csv", 21), "C"),
        ],
        "S4": [
            (Source("three_clock_pca_loadings.csv", 26), "A"),
            (Source("three_clock_profile_summary.csv", 26), "B"),
        ],
        "S5": [(Source("questionnaire_phenome_scan_tests.csv", 34), "A")],
        "S6": [
            (Source("acuity_grip_cataract_quality_sensitivity.csv", 37), "A"),
            (Source("quality_threshold_performance_curve.csv", 45), "B"),
        ],
        "S7": [
            (Source("eye_specific_etdrs_associations.csv", 47), "A"),
            (
                Source("baseline_epigenetic_to_followup_retinal_associations.csv", 39),
                "B",
            ),
        ],
        "S8": [
            (Source("patient_clock_comorbidity_associations.csv", 28), "A"),
            (Source("comorbidity_incremental_prediction_metrics.csv", 30), "B"),
        ],
        "S9": [(Source("epigenetic_target_prediction_beyond_age.csv", 49), "A")],
        "S10": [(Source("epigenetic_definition_construct_robustness.csv", 51), "A")],
        "S11": [
            (Source("longitudinal_retinal_change_repeatability.csv", 53), "A"),
            (Source("longitudinal_same_person_matched_null_test.csv", 53), "B"),
            (Source("longitudinal_quality_adjusted_stability.csv", 53), "C"),
        ],
    }
    entries = specs[fid]
    frames = [_read(outdir, s) for s, _ in entries]
    stack_for_slide = profile.name == "slide" and fid in {"S6", "S8"}
    ncols = (
        1 if fid in {"S3", "S7"} or stack_for_slide else (2 if len(entries) > 1 else 1)
    )
    nrows = math.ceil(len(entries) / ncols)
    fig = _figure(profile, min(220, 80 + 55 * nrows), min(4.6, 2.4 + 1.7 * nrows))
    if profile.name == "slide" and fid == "S6":
        gs = fig.add_gridspec(nrows, ncols, height_ratios=[0.64, 0.36])
    else:
        gs = fig.add_gridspec(nrows, ncols)
    records = []
    for idx, ((source, letter), frame) in enumerate(zip(entries, frames)):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        _panel_letter(ax, letter, profile)
        title = {
            "S1": "Age-model diagnostics",
            "S2": "Selection and demographic performance",
            "S3": "Alternative cross-clock comparisons",
            "S4": "Three-clock decomposition",
            "S5": "Complete questionnaire associations",
            "S6": "Retinal sensitivity analyses",
            "S7": "Inter-eye and longitudinal specifications",
            "S8": "Exploratory comorbidity analyses",
            "S9": "Methylation prediction beyond age",
            "S10": "Epigenetic-definition robustness",
            "S11": "Longitudinal repeatability and quality",
        }[fid]
        _slide_title(ax, title, profile)
        # Preserve familiar scientific encodings where the persisted schema permits.
        if fid == "S2" and idx == 0:
            _table_panel(ax, frame, profile, "Standardized difference")
            _slide_title(ax, "Methylation-subset selection", profile)
        elif fid == "S2" and idx == 1:
            work = frame.sort_values("mae", kind="stable")
            y = np.arange(len(work))[::-1]
            ax.barh(y, work["mae"], color=NULL)
            ax.set_yticks(y, work["analysis"].astype(str))
            ax.set_xlabel("MAE (years)")
            ax.set_xlim(left=0)
            _slide_title(ax, "Locked-validation MAE", profile)
        elif fid == "S1" and idx == 0:
            work = frame.loc[
                frame["analysis"].astype(str).str.contains("participant", case=False)
            ].copy()
            work["short_label"] = (
                work["analysis"]
                .astype(str)
                .str.replace("Chronological age head: ", "Retinal: ", regex=False)
                .str.replace("baseline participant", "baseline", regex=False)
            )
            y = np.arange(len(work))[::-1]
            colors = [RETINAL] + [EPIGENETIC] * max(0, len(work) - 1)
            ax.barh(y, work["mae"], color=colors)
            ax.set_yticks(y, work["short_label"])
            ax.set_xlabel("MAE (years)")
        elif fid == "S1" and idx == 1:
            work = frame.sort_values("target_mean", kind="stable")
            ax.plot(
                work["target_mean"], work["mae"], marker="o", color=RETINAL, label="MAE"
            )
            ax.plot(
                work["target_mean"],
                work["mean_error"].abs(),
                marker="o",
                color=NULL,
                label="Absolute bias",
            )
            ax.set_xlabel("Mean chronological age in 5-year bin")
            ax.set_ylabel("Error (years)")
            ax.legend(frameon=False)
        elif fid == "S3" and idx == 0:
            _slide_title(ax, "Age-tail error", profile)
            work = frame.set_index("age_distribution_stratum").reindex(
                ["Lower 10% age tail", "Central 80%", "Upper 10% age tail"]
            )
            y = np.arange(len(work))[::-1]
            ax.barh(y, work["mean_error"], color=RETINAL)
            ax.axvline(0, linestyle="--", color=REFERENCE, linewidth=0.75)
            ax.set_yticks(y, work.index)
            ax.set_xlabel("Mean retinal-age error (years)")
        elif fid == "S3" and idx == 1:
            _slide_title(ax, "Alternative distance definitions", profile)
            work = frame.copy()
            clock_short = np.where(
                work["clock_label"].astype(str).str.contains("Horvath", case=False),
                "Horvath",
                "Hannum",
            )
            comparison_short = np.where(
                work["hypothesis"]
                .astype(str)
                .str.contains("< Retinal", case=False, regex=False),
                "vs retinal-chronological",
                "vs epigenetic-chronological",
            )
            labels = pd.Series(clock_short) + ": " + pd.Series(comparison_short)
            _forest(
                ax,
                labels,
                work["mean_distance_difference"],
                work["mean_difference_ci_low"],
                work["mean_difference_ci_high"],
                work["wilcoxon_fdr_q"],
                profile,
                "Paired distance difference (years)",
                NULL,
            )
        elif fid == "S3" and idx == 2:
            _slide_title(ax, "Alternative acceleration definitions", profile)
            work = frame.copy()
            q_col = "pearson_fdr_q" if "pearson_fdr_q" in work else "fdr_q_global"
            _forest(
                ax,
                work.get("label", work["measure"]).astype(str),
                work.pearson_r,
                work.pearson_ci_low,
                work.pearson_ci_high,
                work[q_col],
                profile,
                "Pearson r with retinal acceleration",
                EPIGENETIC,
                symmetric=True,
                null_band=0.02,
            )
        elif fid == "S4" and idx == 0:
            work = frame.copy()
            component_labels = {
                "shared_aging_pc": "Shared aging",
                "retina_vs_epigenetic_pc": "Retina vs epigenetic",
                "horvath_vs_hannum_pc": "Horvath vs Hannum",
            }
            x = np.arange(len(work))
            width = 0.24
            for offset, column, label, color in (
                (-width, "loading_retinal", "Retinal", RETINAL),
                (0, "loading_horvath", "Horvath", EPIGENETIC),
                (width, "loading_hannum", "Hannum", NULL),
            ):
                ax.bar(x + offset, work[column], width, label=label, color=color)
            ax.axhline(0, linestyle="--", color=REFERENCE, linewidth=0.75)
            ax.set_xticks(
                x,
                work["component_role"]
                .map(component_labels)
                .fillna(work["component_role"]),
                rotation=25,
                ha="right",
            )
            ax.set_ylabel("PCA loading")
            ax.legend(frameon=False)
            _slide_title(ax, "PCA loadings", profile)
        elif fid == "S4" and idx == 1:
            work = frame.copy()
            y = np.arange(len(work))[::-1]
            ax.barh(y, work["participants"], color=NULL)
            ax.set_yticks(y, work["three_clock_profile"].astype(str))
            ax.set_xlabel("Participants")
            _slide_title(ax, "Discordance profiles", profile)
        elif fid == "S5":
            work = frame.loc[
                frame.get("manuscript_eligible", True).astype(bool)
                if "manuscript_eligible" in frame
                else slice(None)
            ].copy()
            work = work.sort_values(
                "incremental_r2", ascending=False, kind="stable"
            ).head(27)
            work["incremental_r2_percent"] = 100 * work.incremental_r2
            y = np.arange(len(work))[::-1]
            colors = np.where(
                work["outcome"].eq("z_retinal_acceleration"),
                RETINAL,
                np.where(work["fdr_q_global"].lt(0.05), EPIGENETIC, NULL),
            )
            ax.barh(y, work["incremental_r2_percent"], color=colors)
            ax.set_yticks(y, work["variable_label"].astype(str))
            ax.set_xlabel("Incremental $R^2$ (%)")
        elif (
            fid in {"S6", "S8", "S10"}
            and idx == 0
            and {"variable", "analysis", "fdr_q_within_analysis"}.issubset(
                frame.columns
            )
        ):
            work = frame.copy()
            work["display_variable"] = work["variable"].map(
                lambda value: VARIABLE_LABELS.get(str(value), str(value))
            )
            compact_labels = {
                "Average grip strength": "Grip: average",
                "Maximum grip strength": "Grip: maximum",
                "Grip strength, trial 1": "Grip: trial 1",
                "Grip strength, trial 2": "Grip: trial 2",
                "Grip strength, trial 3": "Grip: trial 3",
                "Ever had cataracts": "Cataract history",
                "Cataracts on vision testing": "Cataract on testing",
                "Bilateral ETDRS acuity": "Acuity: bilateral",
                "Left-eye ETDRS acuity": "Acuity: left",
                "Left-eye pinhole ETDRS acuity": "Acuity: left pinhole",
                "Right-eye ETDRS acuity": "Acuity: right",
                "Right-eye pinhole ETDRS acuity": "Acuity: right pinhole",
            }
            work["display_variable"] = work["display_variable"].replace(compact_labels)
            analysis_labels = {
                "primary_unadjusted": "Primary",
                "primary_retinal_cataract_adjusted": "+ cataract",
                "strict_quality_unadjusted": "Strict quality",
                "strict_quality_cataract_adjusted": "Both",
            }
            work["analysis_short"] = (
                work["analysis"].map(analysis_labels).fillna(work["analysis"])
            )
            work["minus_log10_q"] = -np.log10(
                work.fdr_q_within_analysis.clip(lower=1e-300)
            )
            _heatmap_panel(
                fig,
                ax,
                work,
                "display_variable",
                "analysis_short",
                "minus_log10_q",
                profile,
                MINUS + "log10 q",
            )
            ax.set_xlabel("")
            for tick in ax.get_xticklabels():
                tick.set_rotation(20)
                tick.set_ha("right")
            _slide_title(ax, "Association robustness", profile)
        elif fid == "S6" and idx == 1:
            work = frame.sort_values("retained_fraction", kind="stable")
            ax.plot(
                100 * work["retained_fraction"],
                work["mae"],
                marker="o",
                color=RETINAL,
            )
            ax.set_xlabel("Best-quality images retained (%)")
            ax.set_ylabel("MAE (years)")
            _slide_title(ax, "Image-quality threshold", profile)
        elif fid == "S7" and idx == 0:
            work = frame.copy()
            labels = work["eye"].map({"R": "Right", "L": "Left"}).fillna(work["eye"])
            measure_labels = work["variable"].map(
                lambda value: VARIABLE_LABELS.get(str(value), str(value))
            )
            labels = labels.astype(str) + " eye: " + measure_labels.astype(str)
            _forest(
                ax,
                labels,
                work["coefficient_years_per_sd"],
                work["ci_low"],
                work["ci_high"],
                work["fdr_q_value"],
                profile,
                "Retinal-age difference (years per SD ETDRS)",
                RETINAL,
            )
            _slide_title(ax, "Same-eye acuity models", profile)
        elif fid == "S7" and idx == 1:
            work = frame.copy()
            label_map = {
                "z_epigenetic_mean_acceleration": "Mean epi.",
                "z_horvath_acceleration": "Horvath",
                "z_hannum_acceleration": "Hannum",
                "shared_aging_pc": "Shared PC",
                "retina_vs_epigenetic_pc": "R–E PC",
            }
            work["Predictor"] = (
                work["predictor"].map(label_map).fillna(work["predictor"])
            )
            work["β"] = work["coefficient"].map(lambda value: fmt(value, "beta"))
            work["95% CI"] = work.apply(
                lambda row: f"{fmt(row.ci_low, 'beta')}–{fmt(row.ci_high, 'beta')}",
                axis=1,
            )
            work["p"] = work["p_value"].map(lambda value: fmt(value, "p"))
            work["q"] = work["fdr_q_value"].map(lambda value: fmt(value, "q"))
            ax.set_axis_off()
            table_artist = ax.table(
                cellText=work[["Predictor", "β", "95% CI", "p", "q"]].values,
                colLabels=["Predictor", "β", "95% CI", "p", "q"],
                loc="center",
                cellLoc="left",
                colLoc="left",
                colWidths=[0.25, 0.12, 0.30, 0.135, 0.135],
            )
            table_artist.auto_set_font_size(False)
            table_artist.set_fontsize(profile.annotation_font)
            table_artist.scale(1, 1.35)
            for (row_index, _), cell_artist in table_artist.get_celld().items():
                if row_index == 0:
                    cell_artist.set_facecolor(VERY_LIGHT_GRAY)
                    cell_artist.get_text().set_fontweight("bold")
            _slide_title(ax, "Longitudinal specifications", profile)
        elif fid == "S8" and idx == 0:
            predictors = {
                "z_retinal_acceleration": ("Retinal", RETINAL, 0.12),
                "z_epigenetic_mean_acceleration": ("Epigenetic", EPIGENETIC, -0.12),
            }
            work = frame.loc[
                frame["framework"].eq("mutually_adjusted_clocks")
                & frame["predictor"].isin(predictors)
            ].copy()
            outcomes = list(dict.fromkeys(work["outcome_label"].astype(str)))
            base_y = np.arange(len(outcomes))[::-1]
            ax.axvline(1, linestyle="--", color=REFERENCE, linewidth=0.75)
            for predictor, (label, color, offset) in predictors.items():
                rows = (
                    work.loc[work["predictor"].eq(predictor)]
                    .set_index("outcome_label")
                    .reindex(outcomes)
                )
                ax.errorbar(
                    rows["odds_ratio"],
                    base_y + offset,
                    xerr=np.vstack(
                        [
                            rows["odds_ratio"] - rows["ci_low"],
                            rows["ci_high"] - rows["odds_ratio"],
                        ]
                    ),
                    fmt="o",
                    color=color,
                    capsize=2,
                    label=label,
                )
            ax.set_yticks(base_y, outcomes)
            ax.set_xlabel("Adjusted odds ratio (95% CI)")
            ax.legend(
                frameon=True,
                facecolor=WHITE,
                edgecolor=LIGHT_GRAY,
                framealpha=0.92,
                ncols=2,
                loc="upper right",
            )
            if not work["significant_fdr_0_05"].fillna(False).any():
                ax.text(
                    0.98,
                    0.03,
                    "none significant after FDR",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=profile.annotation_font,
                    color=NULL,
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": WHITE,
                        "edgecolor": LIGHT_GRAY,
                        "alpha": 0.92,
                    },
                )
            _slide_title(ax, "Clock–comorbidity associations", profile)
        elif fid == "S8" and idx == 1:
            work = frame.loc[frame["delta_auc_vs_base"].notna()].copy()
            model_labels = {
                "base_plus_retinal_increment_vs_base": "Retinal",
                "base_plus_epigenetic_increment_vs_base": "Epigenetic",
                "base_plus_both_increment_vs_base": "Both",
                "base_plus_orthogonal_components_increment_vs_base": "Components",
                "base_plus_direct_discordance_increment_vs_base": "Discordance",
            }
            work["model_short"] = work["model"].map(model_labels).fillna(work["model"])
            _heatmap_panel(
                fig,
                ax,
                work,
                "outcome_label",
                "model_short",
                "delta_auc_vs_base",
                profile,
                "OOF ΔAUC versus base",
            )
            ax.set_xlabel("")
            _slide_title(ax, "Incremental discrimination", profile)
        elif fid == "S9":
            work = frame.copy()
            y = np.arange(len(work))[::-1]
            ax.axvline(0, linestyle="--", color=REFERENCE, linewidth=0.75)
            ax.errorbar(
                work["delta_r2_beyond_age"],
                y,
                xerr=np.vstack(
                    [
                        work["delta_r2_beyond_age"] - work["delta_r2_ci_low"],
                        work["delta_r2_ci_high"] - work["delta_r2_beyond_age"],
                    ]
                ),
                fmt="o",
                color=RETINAL,
                capsize=2,
            )
            ax.set_yticks(y, work["clock_label"].astype(str))
            ax.set_xlabel("Incremental $R^2$ beyond chronological age")
        elif fid == "S10" and {
            "variable_label",
            "outcome_label",
            "incremental_r2",
        }.issubset(frame.columns):
            work = frame.copy()
            if "publication_label" in work:
                work["variable_label"] = work["publication_label"]
            work["outcome_label"] = work["outcome_label"].replace(
                {
                    "Mean Horvath-Hannum acceleration": "Mean Horvath-Hannum",
                    "Cross-fitted Horvath acceleration": "Cross-fitted Horvath",
                    "Cross-fitted Hannum acceleration": "Cross-fitted Hannum",
                    "Released acceleration difference": "Released difference",
                    "Released residual acceleration": "Released residual",
                }
            )
            work["incremental_r2_percent"] = 100 * work.incremental_r2
            _heatmap_panel(
                fig,
                ax,
                work,
                "variable_label",
                "outcome_label",
                "incremental_r2_percent",
                profile,
                "Incremental $R^2$ (%)",
            )
        elif fid == "S11" and idx == 0:
            work = frame.copy()
            labels = (
                work["analysis"]
                .astype(str)
                .str.replace(
                    "All participants with baseline and F1 retinal predictions",
                    "All participants",
                    regex=False,
                )
            )
            y = np.arange(len(work))[::-1]
            ax.barh(y, work["acceleration_stability_r"], color=RETINAL)
            ax.set_yticks(y, labels)
            ax.set_xlabel("Baseline-follow-up retinal acceleration r")
        elif fid == "S11" and idx == 1:
            row = frame.iloc[0]
            values = [
                row["observed_same_participant_stability_r"],
                row["random_matched_r_mean"],
            ]
            ax.barh([1, 0], values, color=[RETINAL, NULL])
            ax.axvline(0, linestyle="--", color=REFERENCE, linewidth=0.75)
            ax.set_yticks([1, 0], ["Same participant", "Matched null"])
            ax.set_xlabel("Baseline-follow-up acceleration r")
        elif {"coefficient", "ci_low", "ci_high"}.issubset(frame.columns):
            label = next(
                (
                    c
                    for c in ("variable_label", "predictor", "term", "outcome_label")
                    if c in frame
                ),
                frame.columns[0],
            )
            work = frame.dropna(subset=["coefficient", "ci_low", "ci_high"]).head(16)
            q = work["fdr_q_value"] if "fdr_q_value" in work else None
            _forest(
                ax,
                work[label].astype(str),
                work.coefficient,
                work.ci_low,
                work.ci_high,
                q,
                profile,
                "Estimate (95% CI)",
                RETINAL,
            )
        elif {"odds_ratio", "ci_low", "ci_high"}.issubset(frame.columns):
            label = next(
                (c for c in ("outcome_label", "predictor") if c in frame),
                frame.columns[0],
            )
            work = frame.dropna(subset=["odds_ratio", "ci_low", "ci_high"]).head(16)
            y = np.arange(len(work))[::-1]
            ax.axvline(1, linestyle="--", color=REFERENCE, linewidth=0.75)
            ax.errorbar(
                work.odds_ratio,
                y,
                xerr=np.vstack(
                    [work.odds_ratio - work.ci_low, work.ci_high - work.odds_ratio]
                ),
                fmt="o",
                color=NULL,
            )
            ax.set_yticks(y, work[label].astype(str))
            ax.set_xscale("log")
            ax.set_xlabel("Adjusted odds ratio (95% CI)")
        else:
            _table_panel(ax, frame, profile, max_rows=16)
        n = (
            int(frame.n.min())
            if "n" in frame and pd.to_numeric(frame.n, errors="coerce").notna().any()
            else None
        )
        records.append(_record(letter, [source], [ax], n=n, rows=int(len(frame))))
    for idx in range(len(entries), nrows * ncols):
        fig.add_subplot(gs[idx // ncols, idx % ncols]).set_axis_off()
    return fig, records


BUILDERS: Mapping[str, Callable[[Profile, Path], tuple[Figure, list[PanelRecord]]]] = {
    "01": _build_01,
    "02": _build_02,
    "03": _build_03,
    "04": _build_04,
    "05": _build_05,
    **{
        f"S{i}": (lambda p, o, f=f"S{i}": _build_supplement(f, p, o))
        for i in range(1, 12)
    },
}
BUILDERS = {
    **BUILDERS,
    "S1": _build_s1,
    "S5": _build_s5,
    "S11": _build_s11,
}


def _normalize_figure_id(figure_id: str) -> str:
    value = str(figure_id).upper().replace("FIGURE", "").replace("_", "").strip()
    if value.startswith("S"):
        return "S" + str(int(value[1:]))
    return f"{int(value):02d}"


def _save(fig: Figure, path: Path, profile: Profile, bbox=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    common = {
        "dpi": profile.dpi,
        "transparent": profile.transparent,
        "bbox_inches": bbox,
    }
    if path.suffix == ".pdf":
        common["metadata"] = {
            "Title": path.stem,
            "Author": "CLSA retinal aging study",
            "Creator": "CLSA deterministic figure renderer",
            "CreationDate": FIXED_PDF_DATE,
            "ModDate": FIXED_PDF_DATE,
        }
    elif path.suffix == ".png":
        common["metadata"] = {"Software": "CLSA deterministic figure renderer"}
    elif path.suffix == ".tif":
        common["pil_kwargs"] = {"compression": "tiff_lzw"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.savefig(path, **common)
    missing = [str(item.message) for item in caught if "Glyph" in str(item.message)]
    if missing:
        raise AssertionError(
            f"Missing glyph while saving {path}: " + "; ".join(missing[:6])
        )


def _pdf_text_check(path: Path, axis_labels: Sequence[str]) -> None:
    candidates = [
        shutil.which("pdftotext"),
        "/Users/adogan/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/poppler/bin/pdftotext",
    ]
    binary = next((c for c in candidates if c and Path(c).is_file()), None)
    if binary is None:
        warnings.warn(
            "pdftotext unavailable; selectable-text check skipped", RuntimeWarning
        )
        return
    result = subprocess.run(
        [str(binary), str(path), "-"], capture_output=True, text=True, check=True
    )
    normalized = " ".join(result.stdout.split())
    expected = [lab for lab in axis_labels if lab and len(lab) > 2]
    # MathText is extracted with inserted spaces (for example R^2 -> R 2),
    # so test stable alphabetic label tokens rather than exact PDF spacing.
    expected_tokens = [
        token
        for label in expected
        for token in re.findall(r"[A-Za-z]{4,}", label.replace("$", " "))
    ]
    if not normalized:
        raise AssertionError(f"PDF selectable-text check failed for {path}; no text")
    if expected_tokens and not any(token in normalized for token in expected_tokens):
        raise AssertionError(
            f"PDF selectable-text check failed for {path}; expected one of {expected[:4]}"
        )


def _manifest_path(outdir: Path) -> Path:
    return Path(outdir) / "figure_manifest.json"


def _update_manifest(
    outdir: Path,
    fid: str,
    profile: Profile,
    fig: Figure,
    records: list[PanelRecord],
    paths: list[Path],
    smallest: float,
) -> None:
    path = _manifest_path(outdir)
    manifest = (
        json.loads(path.read_text())
        if path.is_file()
        else {"schema_version": 1, "seed": SEED, "figures": {}}
    )
    key = f"figure_{fid}"
    entry = manifest["figures"].setdefault(key, {"profiles": {}, "panels": {}})
    width, height = fig.get_size_inches()
    entry["profiles"][profile.name] = {
        "paths": [str(p) for p in paths if "/panels/" not in str(p)],
        "width_in": float(width),
        "height_in": float(height),
        "width_mm": float(width * MM_PER_INCH),
        "height_mm": float(height * MM_PER_INCH),
        "smallest_font_pt": smallest,
        "settings": asdict(profile),
        "font": resolve_font(),
    }
    for record in records:
        pentry = entry["panels"].setdefault(
            record.panel,
            {
                "sources": _source_dict(record.sources),
                "n": record.n,
                "statistics": record.statistics,
                "output_paths": {},
            },
        )
        pentry["output_paths"][profile.name] = record.output_paths.get(profile.name, [])
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def _flatten_numeric(value, path: tuple[str, ...] = ()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _flatten_numeric(item, (*path, str(key)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _flatten_numeric(item, (*path, str(index)))
    elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    ):
        if math.isfinite(float(value)):
            yield path, float(value)


def _number_manifest_check(records: list[PanelRecord], outdir: Path) -> None:
    """Compare manifest-bound numeric values with persisted source tables."""
    payload = [
        {"n": r.n, "statistics": r.statistics, "sources": _source_dict(r.sources)}
        for r in records
    ]
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True, default=str)
    if json.loads(encoded) != json.loads(
        json.dumps(json.loads(encoded), sort_keys=True)
    ):
        raise AssertionError("Manifest numeric round-trip failed")
    derived_roots = {
        "cohort_counts",
        "fdr_families",
        "support_threshold",
        "age_bins",
        "rows",
    }
    for record in records:
        source_numbers: list[float] = []
        for source in record.sources:
            frame = _read(outdir, source)
            for column in frame.columns:
                values = pd.to_numeric(frame[column], errors="coerce").dropna()
                source_numbers.extend(values.astype(float).tolist())
        source_array = np.asarray(source_numbers, dtype=float)
        for path, value in _flatten_numeric(record.statistics):
            if path and path[0] in derived_roots:
                continue
            if source_array.size == 0 or not np.any(
                np.isclose(source_array, value, rtol=0, atol=5e-5)
            ):
                raise AssertionError(
                    f"Panel {record.panel} manifest value {'.'.join(path)}={value:.6g} "
                    "does not match any persisted source value at four-decimal precision"
                )


def verify_manifest_against_sources(outdir: Path) -> None:
    """Re-read the provenance record and validate its numeric payload."""
    path = _manifest_path(Path(outdir))
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text())
    for figure in manifest.get("figures", {}).values():
        for panel_name, panel in figure.get("panels", {}).items():
            record = PanelRecord(
                panel=panel_name,
                sources=[Source(**source) for source in panel.get("sources", [])],
                n=panel.get("n"),
                statistics=panel.get("statistics", {}),
            )
            _number_manifest_check([record], Path(outdir))


def render_figure(figure_id: str, profile: str, outdir: Path) -> list[Path]:
    """Render one figure and all of its panels from persisted statistics.

    Parameters
    ----------
    figure_id:
        ``1`` through ``5`` or ``S1`` through ``S11``.
    profile:
        ``publication`` or ``slide``.
    outdir:
        The notebook's ``figure_root``.  Statistics are read from sibling
        ``03_statistics`` unless ``CLSA_STATISTICS_ROOT`` is set.
    """
    fid = _normalize_figure_id(figure_id)
    if profile not in PROFILES:
        raise ValueError("profile must be 'publication' or 'slide'")
    if fid not in BUILDERS:
        raise KeyError(f"Unknown figure id {figure_id!r}")
    prof = PROFILES[profile]
    outdir = Path(outdir)
    np.random.seed(SEED)
    random.seed(SEED)
    with mpl.rc_context(_style(prof)):
        fig, records = BUILDERS[fid](prof, outdir)
        _finalize_axes(fig, prof)
        smallest = assert_text_legible(fig, profile)
        # The first draw above resolves constrained layout. Freeze those final
        # positions before subsequent validation and cropped panel saves so a
        # second layout solve cannot collapse dense multi-panel supplements.
        fig.set_layout_engine("none")
        assert_axes_nonoverlap(fig)
        _number_manifest_check(records, outdir)
        width, height = fig.get_size_inches()
        if profile == "slide" and (width > 6.1 + 1e-9 or height > 4.6 + 1e-9):
            raise AssertionError(
                f"Slide canvas exceeds 6.1 x 4.6 in: {width:.3f} x {height:.3f}"
            )
        stem = f"figure_{fid}"
        base = outdir / profile
        paths = []
        for suffix in prof.formats:
            path = base / f"{stem}.{suffix}"
            _save(fig, path, prof)
            paths.append(path)
        for record in records:
            bbox = _group_bbox(fig, record.axes)
            panel_paths = []
            for suffix in prof.formats:
                path = outdir / "panels" / profile / f"{stem}{record.panel}.{suffix}"
                _save(fig, path, prof, bbox)
                paths.append(path)
                panel_paths.append(str(path))
            record.output_paths[profile] = panel_paths
        if profile == "publication":
            axis_labels = [ax.get_xlabel() for ax in fig.axes] + [
                ax.get_ylabel() for ax in fig.axes
            ]
            _pdf_text_check(base / f"{stem}.pdf", axis_labels)
        _update_manifest(outdir, fid, prof, fig, records, paths, smallest)
        verify_manifest_against_sources(outdir)
        plt.close(fig)
    return paths


def build_contact_sheet(outdir: Path) -> Path:
    """Merge the publication PDFs into a one-figure-per-page contact sheet."""
    from pypdf import PdfReader, PdfWriter

    outdir = Path(outdir)
    writer = PdfWriter()
    for fid in ["01", "02", "03", "04", "05", *[f"S{i}" for i in range(1, 12)]]:
        path = outdir / "publication" / f"figure_{fid}.pdf"
        if not path.is_file():
            raise MissingFigureInput(f"Cannot build contact sheet; missing {path}")
        for page in PdfReader(path).pages:
            writer.add_page(page)
    destination = outdir / "contact_sheet.pdf"
    with destination.open("wb") as stream:
        writer.write(stream)
    return destination


def render_all(outdir: Path) -> pd.DataFrame:
    """Render all figures, build the contact sheet, and print the audit table."""
    outdir = Path(outdir)
    rows = []
    for fid in ["01", "02", "03", "04", "05", *[f"S{i}" for i in range(1, 12)]]:
        for profile in ("publication", "slide"):
            render_figure(fid, profile, outdir)
    build_contact_sheet(outdir)
    manifest = json.loads(_manifest_path(outdir).read_text())
    for key, entry in manifest["figures"].items():
        for profile, data in entry["profiles"].items():
            rows.append(
                {
                    "figure_id": key.replace("figure_", ""),
                    "panel_count": len(entry["panels"]),
                    "profile": profile,
                    "width_in": data["width_in"],
                    "height_in": data["height_in"],
                    "width_mm": data["width_mm"],
                    "height_mm": data["height_mm"],
                    "smallest_font_pt": data["smallest_font_pt"],
                    "failed_checks": "",
                }
            )
    report = (
        pd.DataFrame(rows)
        .sort_values(["figure_id", "profile"], kind="stable")
        .reset_index(drop=True)
    )
    print(report.to_string(index=False))
    return report


__all__ = [
    "render_figure",
    "render_all",
    "build_contact_sheet",
    "assert_text_legible",
    "assert_axes_nonoverlap",
    "verify_manifest_against_sources",
    "MissingFigureInput",
]
