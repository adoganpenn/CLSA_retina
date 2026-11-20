# Databricks notebook source
# MAGIC %md
# MAGIC # Complete CLSA retinal-aging manuscript pipeline
# MAGIC
# MAGIC Run this single notebook to reproduce the analyses, persisted statistics, five main
# MAGIC figures, eleven supplemental figures, publication exports, slide-size previews,
# MAGIC provenance manifests, and an internal governed results bundle for the retinal–DNA
# MAGIC methylation aging manuscript.
# MAGIC
# MAGIC The main epigenetics notebook remains the source of statistical estimates. This runner
# MAGIC also launches the symmetric three-clock calibration comparison and the granular
# MAGIC retinal-by-individual-clock concordance analysis, audits every statistics file named by
# MAGIC the canonical renderer, rebuilds all figures, and displays all canonical figures inline.
# MAGIC
# MAGIC Cross-validation, reporting, and visualization implementation was audited with
# MAGIC procedural guidance from Kassis T, Agarwal V, He Y, Patel D, and Brueckner AM.
# MAGIC *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*.
# MAGIC arXiv:2609.00065 (2026). https://doi.org/10.48550/arXiv.2609.00065

# COMMAND ----------
# MAGIC %pip install pypdf Pillow openpyxl

# COMMAND ----------

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import pandas as pd


REPO_ROOT_DEFAULT = Path(
    "/Workspace/Users/ad0038@pennmedicine.upenn.edu/CLSA/CLSA_retina"
)
EPIGENETICS_ROOT_DEFAULT = Path(
    "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/derived/"
    "clsa_retinal_aging/Age_Glaucoma/16_algorithm_fairness/08_epigenetics"
)


def widget(name: str, default: str, label: str) -> str:
    try:
        dbutils.widgets.text(name, default, label)  # noqa: F821
        return dbutils.widgets.get(name).strip()  # noqa: F821
    except NameError:
        return default


def as_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Expected a boolean value, received {value!r}")


repo_root = Path(widget("repo_root", str(REPO_ROOT_DEFAULT), "Repository root"))
epigenetics_root = Path(
    widget("epigenetics_root", str(EPIGENETICS_ROOT_DEFAULT), "Epigenetics output root")
)
run_main_analysis = as_bool(widget("run_main_analysis", "true", "Run main analysis"))
run_symmetric_calibration = as_bool(
    widget("run_symmetric_calibration", "true", "Run symmetric calibration")
)
run_pairwise_concordance = as_bool(
    widget("run_pairwise_concordance", "true", "Run pairwise concordance")
)
render_canonical_figures = as_bool(
    widget("render_canonical_figures", "true", "Render all canonical figures")
)
show_figures_inline = as_bool(widget("show_figures_inline", "true", "Display figures"))
show_statistics_previews = as_bool(
    widget("show_statistics_previews", "true", "Display figure statistics previews")
)
statistics_preview_rows = int(
    widget("statistics_preview_rows", "20", "Rows per statistics preview")
)
build_results_bundle = as_bool(widget("build_results_bundle", "true", "Build results ZIP"))
stop_on_child_failure = as_bool(
    widget(
        "stop_on_child_failure",
        "false",
        "Stop immediately on child-notebook failure",
    )
)
timeout_seconds = int(widget("timeout_seconds", "0", "Child notebook timeout; 0 = unlimited"))
bootstrap_repetitions = int(
    widget("bootstrap_repetitions", "2000", "Calibration bootstrap repetitions")
)
paired_repetitions = int(
    widget("paired_repetitions", "10000", "Calibration paired repetitions")
)
random_seed = int(widget("random_seed", "20261118", "Calibration random seed"))

statistics_root = epigenetics_root / "03_statistics"
figure_root = epigenetics_root / "04_figures"
symmetric_root = epigenetics_root / "06_symmetric_calibration"
run_root = epigenetics_root / "07_complete_manuscript_run"
run_root.mkdir(parents=True, exist_ok=True)

source_root = repo_root / "src"
renderer_root = repo_root / "Age_Glaucoma" / "Algorithm Fairness"
for required_directory in (source_root, renderer_root):
    if not required_directory.is_dir():
        raise FileNotFoundError(f"Required repository directory not found: {required_directory}")
for import_root in (source_root, renderer_root):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import manuscript_pipeline as pipeline  # noqa: E402

pipeline = importlib.reload(pipeline)

configuration = {
    "repo_root": str(repo_root),
    "epigenetics_root": str(epigenetics_root),
    "run_main_analysis": run_main_analysis,
    "run_symmetric_calibration": run_symmetric_calibration,
    "run_pairwise_concordance": run_pairwise_concordance,
    "render_canonical_figures": render_canonical_figures,
    "show_figures_inline": show_figures_inline,
    "show_statistics_previews": show_statistics_previews,
    "statistics_preview_rows": statistics_preview_rows,
    "build_results_bundle": build_results_bundle,
    "stop_on_child_failure": stop_on_child_failure,
    "timeout_seconds": timeout_seconds,
    "bootstrap_repetitions": bootstrap_repetitions,
    "paired_repetitions": paired_repetitions,
    "random_seed": random_seed,
}
print(json.dumps(configuration, indent=2))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Run the analysis notebooks
# MAGIC
# MAGIC The master runner calls the existing analysis notebooks rather than duplicating their
# MAGIC statistical code. Existing checkpoints are reused by those notebooks. A child notebook
# MAGIC is retried under its alternate Databricks path only when the first path is not found;
# MAGIC analysis errors are never silently retried or ignored.

# COMMAND ----------

stage_records: list[pipeline.StageRecord] = []
stage_log_path = run_root / "pipeline_stage_log.csv"


def checkpoint_stage_records() -> None:
    """Persist progress even when a remote child notebook aborts the run."""

    columns = [
        "stage",
        "status",
        "started_utc",
        "finished_utc",
        "elapsed_seconds",
        "detail",
    ]
    pd.DataFrame(
        [record.__dict__ for record in stage_records], columns=columns
    ).to_csv(stage_log_path, index=False)


def notebook_candidates(relative_path: str) -> list[str]:
    path = repo_root / relative_path
    without_suffix = str(path.with_suffix(""))
    with_suffix = str(path)
    if path.suffix.lower() == ".py":
        return [without_suffix, with_suffix]
    return [with_suffix, without_suffix]


def missing_notebook_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        token in message
        for token in (
            "not found",
            "does not exist",
            "resource_does_not_exist",
            "notebook not found",
        )
    )


def run_child_notebook(
    stage: str,
    relative_path: str,
    arguments: dict[str, str] | None = None,
) -> str:
    started_clock = time.perf_counter()
    started_utc = pipeline.utc_now()
    last_missing_error = None
    try:
        runner = dbutils.notebook.run  # noqa: F821
    except NameError as error:
        raise RuntimeError(
            "This master runner must execute in Databricks because it launches child notebooks"
        ) from error

    try:
        for candidate in notebook_candidates(relative_path):
            try:
                result = runner(candidate, timeout_seconds, arguments or {})
                finished_utc = pipeline.utc_now()
                stage_records.append(
                    pipeline.StageRecord(
                        stage=stage,
                        status="completed",
                        started_utc=started_utc,
                        finished_utc=finished_utc,
                        elapsed_seconds=time.perf_counter() - started_clock,
                        detail=f"notebook={candidate}; result={str(result)[:500]}",
                    )
                )
                checkpoint_stage_records()
                print(f"Completed {stage}: {candidate}")
                return str(result)
            except Exception as error:
                if missing_notebook_error(error):
                    last_missing_error = error
                    continue
                raise
        raise FileNotFoundError(
            f"No Databricks notebook path resolved for {relative_path}: {last_missing_error}"
        )
    except Exception as error:
        stage_records.append(
            pipeline.StageRecord(
                stage=stage,
                status="failed",
                started_utc=started_utc,
                finished_utc=pipeline.utc_now(),
                elapsed_seconds=time.perf_counter() - started_clock,
                detail=f"{type(error).__name__}: {error}\n{traceback.format_exc()[-3000:]}",
            )
        )
        checkpoint_stage_records()
        print(f"FAILED {stage}: {type(error).__name__}: {error}")
        if stop_on_child_failure:
            raise
        print(
            "Continuing to the persisted-output audit because "
            "stop_on_child_failure=false. Databricks often exposes only a wrapper "
            "Py4JJavaError to the parent notebook. If all required statistics were "
            "already written, the master runner can safely complete rendering; if "
            "not, the audit below will name every missing file and its producing cell."
        )
        return ""


if run_main_analysis:
    run_child_notebook(
        "main_epigenetics_analysis",
        "Age_Glaucoma/Algorithm Fairness/epigenetics.ipynb",
    )
else:
    stage_records.append(
        pipeline.StageRecord(
            stage="main_epigenetics_analysis",
            status="skipped",
            started_utc=pipeline.utc_now(),
            finished_utc=pipeline.utc_now(),
            elapsed_seconds=0.0,
            detail="run_main_analysis=false; existing persisted statistics will be audited",
        )
    )
    checkpoint_stage_records()

if run_symmetric_calibration:
    run_child_notebook(
        "symmetric_three_clock_calibration",
        "notebooks/08_three_clock_symmetric_calibration.py",
        {
            "repo_root": str(repo_root),
            "input_path": str(
                epigenetics_root / "01_private" / "baseline_three_age_master_private.parquet"
            ),
            "retinal_oof_path": str(
                epigenetics_root
                / "02_models"
                / "retfound_chronological_age"
                / "retfound_chronological_age_oof.parquet"
            ),
            "output_root": str(symmetric_root),
            "bootstrap_repetitions": str(bootstrap_repetitions),
            "paired_repetitions": str(paired_repetitions),
            "random_seed": str(random_seed),
        },
    )

if run_pairwise_concordance:
    run_child_notebook(
        "pairwise_acceleration_concordance",
        "notebooks/06_retinal_epigenetic_2x2_figure_s2.ipynb",
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## Audit all canonical figure inputs

# COMMAND ----------

renderer_path = renderer_root / "clsa_figure_renderer.py"
requirements = pipeline.discover_renderer_statistics(renderer_path)
statistics_audit = pipeline.audit_required_statistics(statistics_root, requirements)
statistics_audit.to_csv(run_root / "canonical_statistics_input_audit.csv", index=False)

missing_statistics = statistics_audit.loc[~statistics_audit["exists"]]
print(
    f"Canonical renderer inputs: {len(statistics_audit)} required; "
    f"{len(missing_statistics)} missing"
)
if not missing_statistics.empty:
    print(missing_statistics[["file", "producing_cell"]].to_string(index=False))
    checkpoint_stage_records()
    failure_report_path = run_root / "pipeline_failure_diagnostic.md"
    failed_stages = [record for record in stage_records if record.status == "failed"]
    failure_lines = [
        "# Pipeline failure diagnostic",
        "",
        "The remote child notebook failed before all canonical statistics inputs were present.",
        "Open the linked child run in Databricks and copy the traceback from its red failing cell; the parent Py4JJavaError does not contain that traceback.",
        "",
        "## Missing persisted outputs",
        "",
    ]
    failure_lines.extend(
        f"- `{row.file}` (producing cell {row.producing_cell})"
        for row in missing_statistics.itertuples(index=False)
    )
    if failed_stages:
        failure_lines.extend(["", "## Failed child stages", ""])
        failure_lines.extend(
            f"- {record.stage}: {record.detail.splitlines()[0]}"
            for record in failed_stages
        )
    failure_report_path.write_text(
        "\n".join(failure_lines) + "\n", encoding="utf-8"
    )
    print("Failure diagnostic:", failure_report_path)
pipeline.require_complete_statistics(statistics_audit)

try:
    display(statistics_audit)  # noqa: F821
except NameError:
    print(statistics_audit.to_string(index=False))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Render and validate every manuscript figure
# MAGIC
# MAGIC The canonical renderer creates publication PDF/TIFF files, screen-preview PNG files,
# MAGIC panel-level exports, a contact sheet, and a provenance manifest. It re-reads the source
# MAGIC tables and checks that manifest-bound numbers match persisted statistics.

# COMMAND ----------

if render_canonical_figures:
    render_started = time.perf_counter()
    render_started_utc = pipeline.utc_now()
    try:
        os.environ["CLSA_STATISTICS_ROOT"] = str(statistics_root)
        import clsa_figure_renderer

        clsa_figure_renderer = importlib.reload(clsa_figure_renderer)
        render_report = clsa_figure_renderer.render_all(figure_root)
        render_report.to_csv(run_root / "canonical_figure_render_report.csv", index=False)
        stage_records.append(
            pipeline.StageRecord(
                stage="render_canonical_figures",
                status="completed",
                started_utc=render_started_utc,
                finished_utc=pipeline.utc_now(),
                elapsed_seconds=time.perf_counter() - render_started,
                detail=f"rendered={len(pipeline.CANONICAL_FIGURE_IDS)} figures; profiles=2",
            )
        )
    except Exception as error:
        stage_records.append(
            pipeline.StageRecord(
                stage="render_canonical_figures",
                status="failed",
                started_utc=render_started_utc,
                finished_utc=pipeline.utc_now(),
                elapsed_seconds=time.perf_counter() - render_started,
                detail=f"{type(error).__name__}: {error}\n{traceback.format_exc()[-3000:]}",
            )
        )
        checkpoint_stage_records()
        raise

figure_manifest_path = figure_root / "figure_manifest.json"
figure_validation = pipeline.validate_figure_manifest(figure_manifest_path)
figure_validation.to_csv(run_root / "canonical_figure_output_audit.csv", index=False)
figure_catalog = pipeline.flatten_figure_manifest(figure_manifest_path)
figure_catalog.to_csv(run_root / "canonical_figure_catalog.csv", index=False)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Catalog statistics, display all figures, and build the governed bundle

# COMMAND ----------

statistics_catalog = pipeline.catalog_files(
    [
        ("main_statistics", statistics_root),
        ("symmetric_calibration_statistics", symmetric_root / "02_statistics"),
    ],
    suffixes=(".csv", ".json", ".md", ".txt", ".parquet"),
)
statistics_catalog.to_csv(run_root / "complete_statistics_catalog.csv", index=False)

figure_file_catalog = pipeline.catalog_files(
    [
        ("canonical_and_companion_figures", figure_root),
        ("symmetric_calibration_figures", symmetric_root / "03_figures"),
    ],
    suffixes=(".png", ".pdf", ".tif", ".tiff", ".json"),
)
figure_file_catalog.to_csv(run_root / "complete_figure_file_catalog.csv", index=False)

checkpoint_stage_records()
run_log_path = stage_log_path

report_lines = [
    "# Complete manuscript pipeline run",
    "",
    f"- Canonical figures: {len(pipeline.CANONICAL_FIGURE_IDS)}",
    f"- Canonical renderer statistics inputs: {len(requirements)}",
    f"- Statistics files cataloged: {len(statistics_catalog)}",
    f"- Figure files cataloged: {len(figure_file_catalog)}",
    "- Publication profiles: PDF and 600-dpi LZW TIFF",
    "- Screen profile: PNG",
    "- Private participant tables: excluded from the governed results bundle",
    "",
    "## Stages",
    "",
]
for record in stage_records:
    report_lines.append(
        f"- {record.stage}: {record.status} ({record.elapsed_seconds:.1f} seconds)"
    )
run_report_path = run_root / "complete_manuscript_run_report.md"
run_report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

bundle_path = (
    run_root / "CLSA_retinal_aging_manuscript_figures_and_statistics_internal.zip"
    if build_results_bundle
    else None
)
manifest_path = run_root / "complete_manuscript_pipeline_manifest.json"
pipeline.write_run_manifest(
    manifest_path,
    configuration=configuration,
    stages=stage_records,
    required_statistics_count=len(requirements),
    figure_count=len(pipeline.CANONICAL_FIGURE_IDS),
    bundle_path=bundle_path,
)

if build_results_bundle:
    bundle_inventory = pipeline.build_governed_bundle(
        bundle_path,
        figure_root=figure_root,
        statistics_roots=[
            ("main", statistics_root),
            ("symmetric_calibration", symmetric_root / "02_statistics"),
        ],
        extra_files=[
            run_log_path,
            run_report_path,
            manifest_path,
            run_root / "canonical_statistics_input_audit.csv",
            run_root / "canonical_figure_output_audit.csv",
            run_root / "canonical_figure_catalog.csv",
            run_root / "complete_statistics_catalog.csv",
            run_root / "complete_figure_file_catalog.csv",
        ],
    )
    bundle_inventory.to_csv(run_root / "governed_bundle_inventory.csv", index=False)
    print("Governed internal results bundle:", bundle_path)

try:
    display(figure_validation)  # noqa: F821
    display(statistics_catalog)  # noqa: F821
except NameError:
    print(figure_validation.to_string(index=False))
    print(statistics_catalog.to_string(index=False))

if show_figures_inline:
    from IPython.display import Image, Markdown, display as ipython_display

    for path in pipeline.canonical_slide_previews(figure_root):
        figure_id = path.stem.replace("figure_", "")
        ipython_display(Markdown(f"## Figure {figure_id}"))
        ipython_display(Image(data=path.read_bytes(), width=1100))
        associated = figure_catalog.loc[
            figure_catalog["figure_id"].astype(str).eq(figure_id),
            ["panel", "n", "source_files", "source_cells"],
        ].drop_duplicates()
        if not associated.empty:
            ipython_display(Markdown("### Figure statistics and provenance"))
            ipython_display(associated)
        if show_statistics_previews and not associated.empty:
            source_names = sorted(
                {
                    source_name
                    for value in associated["source_files"].dropna().astype(str)
                    for source_name in value.split("|")
                    if source_name
                }
            )
            for source_name in source_names:
                source_path = statistics_root / source_name
                if source_path.suffix.lower() == ".csv":
                    preview = pd.read_csv(source_path, nrows=statistics_preview_rows)
                elif source_path.suffix.lower() == ".parquet":
                    preview = pd.read_parquet(source_path).head(statistics_preview_rows)
                else:
                    continue
                ipython_display(
                    Markdown(
                        f"#### `{source_name}` — first {min(len(preview), statistics_preview_rows)} rows"
                    )
                )
                ipython_display(preview)

    companion_previews = [
        figure_root / "figure_S2.png",
        figure_root / "figure_S2_pca_and_profile_counts.png",
        symmetric_root / "03_figures" / "figure_three_clock_symmetric_calibration_scatter.png",
        symmetric_root / "03_figures" / "figure_three_clock_symmetric_calibration_summary.png",
    ]
    existing_companions = [path for path in companion_previews if path.is_file()]
    if existing_companions:
        ipython_display(Markdown("# Companion and reviewer-requested analyses"))
    for path in existing_companions:
        ipython_display(Markdown(f"## {path.stem.replace('_', ' ')}"))
        ipython_display(Image(data=path.read_bytes(), width=1100))

print(run_report_path.read_text(encoding="utf-8"))
print("Canonical contact sheet:", figure_root / "contact_sheet.pdf")
print("Pipeline manifest:", manifest_path)
print("Complete manuscript pipeline finished successfully.")
