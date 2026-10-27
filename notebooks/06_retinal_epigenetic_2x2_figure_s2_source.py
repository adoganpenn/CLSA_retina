# Databricks notebook source
# MAGIC %md
# MAGIC # Supplemental Figure S2: pairwise acceleration concordance
# MAGIC
# MAGIC This standalone notebook rebuilds the published three-clock profile check
# MAGIC and then draws granular 2 × 2 tables for retinal versus Horvath and retinal
# MAGIC versus Hannum acceleration. It uses the same empirical 80th-percentile
# MAGIC thresholds as the original three-clock pipeline.

# COMMAND ----------
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.decomposition import PCA


# ------------------------------- CONFIG ---------------------------------
DATASET_ROOT = Path("/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset")
DERIVED_ROOT = DATASET_ROOT / "derived" / "clsa_retinal_aging"
OUTPUT_ROOT = (
    DERIVED_ROOT
    / "Age_Glaucoma"
    / "16_algorithm_fairness"
    / "08_epigenetics"
)

# This is written by cell 29 of the existing epigenetics notebook.
DF_PATH = OUTPUT_ROOT / "01_private" / "patient_three_clock_phenotypes_private.parquet"
STATISTICS_ROOT = OUTPUT_ROOT / "03_statistics"
FIGURE_ROOT = OUTPUT_ROOT / "04_figures"

COL_ID = "participant_id"
COL_RET = "z_retinal_acceleration"
COL_HOR = "z_horvath_acceleration"
COL_HAN = "z_hannum_acceleration"

EXTREME_QUANTILE = 0.80
PROFILE_COLUMN = "three_clock_profile"
PROFILE_ORDER = [
    "Concordantly accelerated",
    "Concordantly decelerated",
    "Epigenetic-only accelerated",
    "Mixed or intermediate",
    "Retina-only accelerated",
]

FIGURE_PNG = FIGURE_ROOT / "figure_S2.png"
FIGURE_PDF = FIGURE_ROOT / "figure_S2.pdf"
CELL_TABLE_PATH = STATISTICS_ROOT / "table_S2_retinal_individual_clock_2x2_cells.csv"
SUMMARY_TABLE_PATH = (
    STATISTICS_ROOT / "table_S2_retinal_individual_clock_association.csv"
)
PROFILE_CHECK_PATH = STATISTICS_ROOT / "table_S2_original_profile_check.csv"
MANIFEST_PATH = STATISTICS_ROOT / "figure_S2_manifest.json"
PCA_TABLE_PATH = STATISTICS_ROOT / "figure_S2_pca_loadings.csv"
PROFILE_TABLE_PATH = STATISTICS_ROOT / "figure_S2_profile_counts.csv"
PCA_PROFILE_PNG = FIGURE_ROOT / "figure_S2_pca_and_profile_counts.png"
PCA_PROFILE_PDF = FIGURE_ROOT / "figure_S2_pca_and_profile_counts.pdf"
# ------------------------------------------------------------------------


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Cross-modal phenotype file is missing columns: {missing}")


def two_sided_fisher(table: list[list[int]]) -> tuple[float, float]:
    result = fisher_exact(table, alternative="two-sided")
    if hasattr(result, "statistic"):
        return float(result.statistic), float(result.pvalue)
    odds_ratio, p_value = result
    return float(odds_ratio), float(p_value)


def odds_ratio_ci(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """Woolf 95% CI, with Haldane–Anscombe correction only for a zero cell."""
    cells = np.asarray([a, b, c, d], dtype=float)
    if np.any(cells == 0):
        cells = cells + 0.5
    aa, bb, cc, dd = cells
    log_or = np.log((aa * dd) / (bb * cc))
    standard_error = np.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    low, high = np.exp(log_or + np.asarray([-1.96, 1.96]) * standard_error)
    return float(low), float(high)


def format_p(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def classify_original_three_clock_profiles(
    frame: pd.DataFrame,
    low_thresholds: pd.Series,
    high_thresholds: pd.Series,
) -> pd.Series:
    """Exact five-group rule used in the original epigenetics notebook."""
    retinal_high = frame[COL_RET].ge(high_thresholds[COL_RET])
    retinal_low = frame[COL_RET].le(low_thresholds[COL_RET])
    epigenetic_high = frame[COL_HOR].ge(high_thresholds[COL_HOR]) & frame[
        COL_HAN
    ].ge(high_thresholds[COL_HAN])
    epigenetic_low = frame[COL_HOR].le(low_thresholds[COL_HOR]) & frame[
        COL_HAN
    ].le(low_thresholds[COL_HAN])

    return pd.Series(
        np.select(
            [
                retinal_high & epigenetic_high,
                retinal_low & epigenetic_low,
                retinal_high & ~epigenetic_high,
                ~retinal_high & epigenetic_high,
            ],
            [
                "Concordantly accelerated",
                "Concordantly decelerated",
                "Retina-only accelerated",
                "Epigenetic-only accelerated",
            ],
            default="Mixed or intermediate",
        ),
        index=frame.index,
        name="three_clock_profile_recreated",
    )


def recompute_oriented_pca(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute and orient the three PCA components as in the source pipeline."""
    acceleration_columns = [COL_RET, COL_HOR, COL_HAN]
    matrix = frame[acceleration_columns].to_numpy(float)
    pca = PCA(n_components=3)
    raw_scores = pca.fit_transform(matrix)
    loadings = pca.components_.copy()
    signs = np.ones(3)

    shared_index = int(np.argmax(np.abs(loadings.sum(axis=1))))
    if loadings[shared_index].mean() < 0:
        signs[shared_index] = -1

    remaining = [index for index in range(3) if index != shared_index]
    retina_index = max(
        remaining,
        key=lambda index: abs(loadings[index, 0] - loadings[index, 1:].mean()),
    )
    if loadings[retina_index, 0] - loadings[retina_index, 1:].mean() < 0:
        signs[retina_index] = -1

    horvath_hannum_index = next(
        index for index in range(3) if index not in {shared_index, retina_index}
    )
    if loadings[horvath_hannum_index, 1] - loadings[horvath_hannum_index, 2] < 0:
        signs[horvath_hannum_index] = -1

    oriented_loadings = loadings * signs[:, None]
    # Retain this computation to mirror the source pipeline and ensure the
    # orientation is valid for both loadings and participant scores.
    oriented_scores = raw_scores * signs[None, :]
    if not np.isfinite(oriented_scores).all():
        raise ValueError("Oriented PCA scores contain nonfinite values")

    component_roles = {
        "Shared aging": shared_index,
        "Retinal vs epigenetic": retina_index,
        "Horvath vs Hannum": horvath_hannum_index,
    }
    rows = []
    for role, component_index in component_roles.items():
        rows.append(
            {
                "component_role": role,
                "original_component_number": component_index + 1,
                "explained_variance_ratio": float(
                    pca.explained_variance_ratio_[component_index]
                ),
                "Retinal": float(oriented_loadings[component_index, 0]),
                "Horvath": float(oriented_loadings[component_index, 1]),
                "Hannum": float(oriented_loadings[component_index, 2]),
            }
        )
    return pd.DataFrame(rows)


def build_pairwise_table(
    frame: pd.DataFrame,
    clock_column: str,
    clock_label: str,
    high_thresholds: pd.Series,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    retinal_accelerated = frame[COL_RET].ge(high_thresholds[COL_RET])
    clock_accelerated = frame[clock_column].ge(high_thresholds[clock_column])

    a = int((retinal_accelerated & clock_accelerated).sum())
    b = int((retinal_accelerated & ~clock_accelerated).sum())
    c = int((~retinal_accelerated & clock_accelerated).sum())
    d = int((~retinal_accelerated & ~clock_accelerated).sum())
    n = a + b + c + d
    if n != len(frame):
        raise RuntimeError(f"{clock_label}: 2x2 table does not contain every participant")

    odds_ratio, fisher_p = two_sided_fisher([[a, b], [c, d]])
    ci_low, ci_high = odds_ratio_ci(a, b, c, d)
    expected_both = (a + b) * (a + c) / n

    summary = {
        "clock": clock_label,
        "n": n,
        "retinal_threshold": float(high_thresholds[COL_RET]),
        "clock_threshold": float(high_thresholds[clock_column]),
        "both_accelerated": a,
        "retinal_only_accelerated": b,
        "clock_only_accelerated": c,
        "neither_accelerated": d,
        "expected_both_under_independence": float(expected_both),
        "odds_ratio": odds_ratio,
        "odds_ratio_ci_low": ci_low,
        "odds_ratio_ci_high": ci_high,
        "fisher_exact_p": fisher_p,
    }

    specifications = [
        ("Retinal accelerated", f"{clock_label} accelerated", a, a + b, a + c),
        ("Retinal accelerated", f"{clock_label} not accelerated", b, a + b, b + d),
        ("Retinal not accelerated", f"{clock_label} accelerated", c, c + d, a + c),
        (
            "Retinal not accelerated",
            f"{clock_label} not accelerated",
            d,
            c + d,
            b + d,
        ),
    ]
    cells = []
    for retinal_status, clock_status, count, row_total, column_total in specifications:
        cells.append(
            {
                "clock": clock_label,
                "retinal_status": retinal_status,
                "clock_status": clock_status,
                "count": count,
                "percent_total": 100 * count / n,
                "percent_within_retinal_status": 100 * count / row_total,
                "percent_within_clock_status": 100 * count / column_total,
                "row_total": row_total,
                "column_total": column_total,
                "n": n,
            }
        )
    return summary, cells


def draw_table_panel(
    axis: plt.Axes,
    summary: pd.Series,
    panel_letter: str,
) -> None:
    clock = str(summary["clock"])
    a = int(summary["both_accelerated"])
    b = int(summary["retinal_only_accelerated"])
    c = int(summary["clock_only_accelerated"])
    d = int(summary["neither_accelerated"])
    n = int(summary["n"])

    row_totals = [a + b, c + d]
    column_totals = [a + c, b + d]
    counts = [[a, b], [c, d]]
    cell_text = []
    for row in range(2):
        rendered_row = []
        for column in range(2):
            count = counts[row][column]
            rendered_row.append(
                f"{count:,}\n{100 * count / n:.1f}% total\n"
                f"{100 * count / row_totals[row]:.1f}% row"
            )
        rendered_row.append(f"{row_totals[row]:,}\n({100 * row_totals[row] / n:.1f}%)")
        cell_text.append(rendered_row)
    cell_text.append(
        [
            f"{column_totals[0]:,}\n({100 * column_totals[0] / n:.1f}%)",
            f"{column_totals[1]:,}\n({100 * column_totals[1] / n:.1f}%)",
            f"{n:,}\n(100.0%)",
        ]
    )

    axis.set_axis_off()
    axis.set_title(
        f"{panel_letter}  Retinal × {clock} acceleration",
        loc="left",
        fontsize=10.5,
        fontweight="bold",
        pad=10,
    )
    axis.text(
        0.64,
        0.91,
        f"{clock} status",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
    )
    table = axis.table(
        cellText=cell_text,
        rowLabels=[
            "Retinal\naccelerated",
            "Retinal\nnot accelerated",
            "Total",
        ],
        colLabels=[
            "Accelerated",
            "Not accelerated",
            "Total",
        ],
        cellLoc="center",
        rowLoc="right",
        bbox=[0.28, 0.27, 0.70, 0.62],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(1.5)
        if column == -1:
            cell.set_width(0.28)
        elif column in (0, 1):
            cell.set_width(0.27)
        elif column == 2:
            cell.set_width(0.16)
        if row == 0 or column == -1:
            cell.set_facecolor("#E9EEF4")
            cell.set_text_props(fontweight="bold")
        elif row == 1 and column == 0:
            cell.set_facecolor("#BFD7EA")
        elif row == 1 and column == 1:
            cell.set_facecolor("#F5D6B8")
        elif row == 2 and column == 0:
            cell.set_facecolor("#F5D6B8")
        elif row == 2 and column == 1:
            cell.set_facecolor("#DDE8D5")
        else:
            cell.set_facecolor("#F3F5F7")

    axis.text(
        0.5,
        0.16,
        (
            f"OR {summary['odds_ratio']:.2f} "
            f"(95% CI {summary['odds_ratio_ci_low']:.2f}–"
            f"{summary['odds_ratio_ci_high']:.2f}); "
            f"Fisher p {format_p(float(summary['fisher_exact_p']))}"
        ),
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=8.6,
        fontweight="bold",
    )
    axis.text(
        0.5,
        0.075,
        (
            "Observed both accelerated: "
            f"{a:,}; expected under independence: "
            f"{summary['expected_both_under_independence']:.1f}"
        ),
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=7.2,
        color="#444444",
    )


def draw_pca_profile_figure(
    pca_loadings: pd.DataFrame,
    profile_counts: pd.DataFrame,
) -> plt.Figure:
    """Recreate the attached PCA-loading and profile-count two-panel figure."""
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 6.2),
        gridspec_kw={"width_ratios": [1.05, 1.15]},
        layout="constrained",
    )

    component_order = ["Shared aging", "Retinal vs epigenetic", "Horvath vs Hannum"]
    work = pca_loadings.set_index("component_role").loc[component_order]
    x = np.arange(len(component_order))
    width = 0.24
    colors = {"Retinal": "#2F5597", "Horvath": "#D97706", "Hannum": "#737B8C"}
    for offset, clock in zip((-width, 0, width), ("Retinal", "Horvath", "Hannum")):
        axes[0].bar(
            x + offset,
            work[clock].to_numpy(float),
            width=width,
            color=colors[clock],
            label=clock,
        )
    axes[0].axhline(0, color="#8A8A8A", linestyle=":", linewidth=0.9)
    axes[0].set_xticks(x, component_order, rotation=24, ha="right")
    axes[0].set_ylabel("PCA loading")
    axes[0].set_title("A   PCA loadings per model", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].spines[["top", "right"]].set_visible(False)

    counts = (
        profile_counts.set_index("three_clock_profile")
        .reindex(PROFILE_ORDER)["participants"]
        .fillna(0)
        .astype(int)
    )
    y = np.arange(len(PROFILE_ORDER))
    axes[1].barh(y, counts.to_numpy(), color="#737B8C")
    axes[1].set_yticks(y, PROFILE_ORDER)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Participants")
    axes[1].set_title(
        "B   Three-clock acceleration profiles",
        loc="left",
        fontweight="bold",
    )
    axes[1].spines[["top", "right"]].set_visible(False)
    maximum = max(1, int(counts.max()))
    axes[1].set_xlim(0, maximum * 1.10)
    for position, value in enumerate(counts.to_numpy()):
        axes[1].text(
            value + maximum * 0.012,
            position,
            f"{value:,}",
            va="center",
            fontsize=9,
        )

    return figure


def main() -> None:
    if not DF_PATH.is_file():
        raise FileNotFoundError(
            f"Cross-modal phenotype file not found: {DF_PATH}. "
            "Run the epigenetics notebook through cell 29 first."
        )
    STATISTICS_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    source_frame = pd.read_parquet(DF_PATH)
    source_frame.attrs = {}
    required = [COL_ID, COL_RET, COL_HOR, COL_HAN]
    require_columns(source_frame, required)
    selected_columns = required + (
        [PROFILE_COLUMN] if PROFILE_COLUMN in source_frame.columns else []
    )
    frame = source_frame[selected_columns].copy()
    for column in (COL_RET, COL_HOR, COL_HAN):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=required)

    if frame[COL_ID].duplicated().any():
        raise ValueError("The cross-modal cohort must contain one row per participant")

    acceleration_columns = [COL_RET, COL_HOR, COL_HAN]
    low_thresholds = frame[acceleration_columns].quantile(1 - EXTREME_QUANTILE)
    high_thresholds = frame[acceleration_columns].quantile(EXTREME_QUANTILE)

    recreated_profile = classify_original_three_clock_profiles(
        frame, low_thresholds, high_thresholds
    )
    if PROFILE_COLUMN in frame.columns:
        stored_profile = frame[PROFILE_COLUMN].astype("string").str.strip()
        unknown_profiles = sorted(
            set(stored_profile.dropna().astype(str)) - set(PROFILE_ORDER)
        )
        if unknown_profiles:
            raise ValueError(
                f"{PROFILE_COLUMN} contains unrecognized profiles: {unknown_profiles}"
            )
        authoritative_profile = stored_profile.fillna(recreated_profile).astype(str)
        profile_source = (
            f"stored {PROFILE_COLUMN}; missing stored values filled from the "
            "current 80th-percentile rule"
        )
    else:
        stored_profile = pd.Series(pd.NA, index=frame.index, dtype="string")
        authoritative_profile = recreated_profile.astype(str)
        profile_source = "recomputed from the current parquet file"

    frame["profile_used_for_figure"] = authoritative_profile
    stored_counts = stored_profile.value_counts().to_dict()
    recreated_counts = recreated_profile.value_counts().to_dict()
    used_counts = authoritative_profile.value_counts().to_dict()
    profile_check = pd.DataFrame(
        [
            {
                "three_clock_profile": profile,
                "stored_in_parquet": int(stored_counts.get(profile, 0)),
                "recomputed_current_rule": int(recreated_counts.get(profile, 0)),
                "used_for_figure": int(used_counts.get(profile, 0)),
                "stored_minus_recomputed": int(stored_counts.get(profile, 0))
                - int(recreated_counts.get(profile, 0)),
            }
            for profile in PROFILE_ORDER
        ]
    )
    profile_check.to_csv(PROFILE_CHECK_PATH, index=False)
    profile_counts = profile_check[
        ["three_clock_profile", "used_for_figure"]
    ].rename(columns={"used_for_figure": "participants"})
    profile_counts["percent"] = 100 * profile_counts["participants"] / len(frame)
    profile_counts.to_csv(PROFILE_TABLE_PATH, index=False)

    pca_loadings = recompute_oriented_pca(frame)
    pca_loadings.to_csv(PCA_TABLE_PATH, index=False)

    summaries = []
    cells = []
    for column, label in ((COL_HOR, "Horvath"), (COL_HAN, "Hannum")):
        summary, clock_cells = build_pairwise_table(
            frame, column, label, high_thresholds
        )
        summaries.append(summary)
        cells.extend(clock_cells)
    summary_table = pd.DataFrame(summaries)
    cell_table = pd.DataFrame(cells)
    summary_table.to_csv(SUMMARY_TABLE_PATH, index=False)
    cell_table.to_csv(CELL_TABLE_PATH, index=False)

    display_function = globals().get("display")
    if callable(display_function):
        display_function(profile_check)
        display_function(pca_loadings.round(4))
        display_function(summary_table.round(4))
        display_function(cell_table.round(2))
    else:
        print("Profile-count audit:\n", profile_check.to_string(index=False))
        print("\nRecomputed PCA loadings:\n", pca_loadings.round(4).to_string(index=False))
        print("\nPairwise association summary:\n", summary_table.round(4).to_string(index=False))
        print("\nGranular 2x2 cells:\n", cell_table.round(2).to_string(index=False))

    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), layout="constrained")
        for axis, (_, row), letter in zip(
            axes, summary_table.iterrows(), ("A", "B")
        ):
            draw_table_panel(axis, row, letter)
        figure.suptitle(
            "Supplemental Figure S2. Pairwise acceleration classifications",
            fontsize=13,
            fontweight="bold",
        )
        figure.text(
            0.5,
            0.005,
            (
                "Accelerated = at or above the clock-specific empirical 80th percentile "
                f"in the same {len(frame):,} participants. ‘Not accelerated’ includes the "
                "remaining 80%. Percentages inside cells are cohort and row percentages."
            ),
            ha="center",
            va="bottom",
            fontsize=7.4,
            wrap=True,
        )
        figure.savefig(FIGURE_PNG, dpi=600, bbox_inches="tight", pad_inches=0.08)
        figure.savefig(FIGURE_PDF, bbox_inches="tight", pad_inches=0.08)
        plt.show()
        plt.close(figure)

        pca_profile_figure = draw_pca_profile_figure(
            pca_loadings,
            profile_counts,
        )
        pca_profile_figure.savefig(
            PCA_PROFILE_PNG,
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.08,
        )
        pca_profile_figure.savefig(
            PCA_PROFILE_PDF,
            bbox_inches="tight",
            pad_inches=0.08,
        )
        plt.show()
        plt.close(pca_profile_figure)

    manifest = {
        "figure": "Supplemental Figure S2",
        "source_path": str(DF_PATH),
        "n": int(len(frame)),
        "extreme_quantile": EXTREME_QUANTILE,
        "threshold_rule": "clock-specific empirical 80th percentile",
        "high_thresholds": {
            column: float(high_thresholds[column]) for column in acceleration_columns
        },
        "low_thresholds": {
            column: float(low_thresholds[column]) for column in acceleration_columns
        },
        "profile_source": profile_source,
        "profile_count_audit": profile_check.to_dict("records"),
        "pca_loadings": pca_loadings.to_dict("records"),
        "pairwise_results": summary_table.to_dict("records"),
        "outputs": {
            "two_by_two_png": str(FIGURE_PNG),
            "two_by_two_pdf": str(FIGURE_PDF),
            "pca_profile_png": str(PCA_PROFILE_PNG),
            "pca_profile_pdf": str(PCA_PROFILE_PDF),
            "cell_table": str(CELL_TABLE_PATH),
            "association_table": str(SUMMARY_TABLE_PATH),
            "profile_check": str(PROFILE_CHECK_PATH),
            "pca_table": str(PCA_TABLE_PATH),
            "profile_counts": str(PROFILE_TABLE_PATH),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Profile counts used from: {profile_source}")
    print(profile_check.to_string(index=False))
    print("\nPairwise retinal × individual-clock results:")
    print(summary_table.round(4).to_string(index=False))
    print("\nSaved Figure S2:", FIGURE_PNG)
    print("Saved vector Figure S2:", FIGURE_PDF)
    print("Saved PCA/profile figure:", PCA_PROFILE_PNG)
    print("Saved vector PCA/profile figure:", PCA_PROFILE_PDF)
    print("Saved granular cell table:", CELL_TABLE_PATH)
    print("Saved association summary:", SUMMARY_TABLE_PATH)


if __name__ == "__main__":
    main()
