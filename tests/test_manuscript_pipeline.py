import ast
import json
from pathlib import Path
import sys
import tempfile
import zipfile

REPO_ROOT = Path(__file__).parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from manuscript_pipeline import (  # noqa: E402
    CANONICAL_FIGURE_IDS,
    CANONICAL_PROFILES,
    StageRecord,
    audit_required_statistics,
    build_governed_bundle,
    discover_renderer_statistics,
    require_complete_statistics,
    validate_figure_manifest,
    write_run_manifest,
)


def test_canonical_renderer_statistics_inventory_is_complete():
    renderer = (
        REPO_ROOT
        / "Age_Glaucoma"
        / "Algorithm Fairness"
        / "clsa_figure_renderer.py"
    )
    requirements = discover_renderer_statistics(renderer)
    files = {requirement.file for requirement in requirements}
    assert len(requirements) == 34
    assert "figure_01_cohort_flow.csv" in files
    assert "figure_02_three_age_points.parquet" in files
    assert "questionnaire_phenome_scan_tests.csv" in files
    assert "longitudinal_quality_adjusted_stability.csv" in files
    assert all(requirement.producing_cell > 0 for requirement in requirements)


def test_statistics_audit_names_missing_file_and_cell():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        renderer = root / "renderer.py"
        renderer.write_text(
            'a = Source("present.csv", 12)\nb = Source("missing.parquet", 55)\n',
            encoding="utf-8",
        )
        (root / "present.csv").write_text("x\n1\n", encoding="utf-8")
        requirements = discover_renderer_statistics(renderer)
        audit = audit_required_statistics(root, requirements)
        assert audit["exists"].sum() == 1
        try:
            require_complete_statistics(audit)
        except FileNotFoundError as error:
            message = str(error)
            assert "missing.parquet" in message
            assert "55" in message
        else:
            raise AssertionError("Missing statistics should fail the audit")


def test_manifest_validation_requires_every_figure_and_profile():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        figures = {}
        for figure_id in CANONICAL_FIGURE_IDS:
            profiles = {}
            for profile in CANONICAL_PROFILES:
                path = root / profile / f"figure_{figure_id}.dat"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"figure")
                profiles[profile] = {"paths": [str(path)]}
            figures[f"figure_{figure_id}"] = {"profiles": profiles, "panels": {}}
        manifest = root / "figure_manifest.json"
        manifest.write_text(json.dumps({"figures": figures}), encoding="utf-8")
        result = validate_figure_manifest(manifest)
        assert len(result) == len(CANONICAL_FIGURE_IDS) * len(CANONICAL_PROFILES)
        assert result["all_outputs_exist"].all()


def test_governed_bundle_excludes_private_paths():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        figures = root / "figures"
        statistics = root / "statistics"
        (figures / "slide").mkdir(parents=True)
        (figures / "01_private").mkdir(parents=True)
        statistics.mkdir()
        (figures / "slide" / "figure_01.png").write_bytes(b"png")
        (figures / "01_private" / "participant.png").write_bytes(b"private")
        (statistics / "metrics.csv").write_text("metric,value\nmae,4.2\n")
        (statistics / "patient_private.csv").write_text("id\n1\n")
        destination = root / "bundle.zip"
        inventory = build_governed_bundle(
            destination,
            figure_root=figures,
            statistics_roots=[("main", statistics)],
        )
        with zipfile.ZipFile(destination) as archive:
            names = archive.namelist()
        assert "figures/slide/figure_01.png" in names
        assert "statistics/main/metrics.csv" in names
        assert not any("private" in name.lower() for name in names)
        assert len(inventory) == 2


def test_run_manifest_records_configuration_and_stages():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "manifest.json"
        stage = StageRecord(
            stage="analysis",
            status="completed",
            started_utc="2026-01-01T00:00:00+00:00",
            finished_utc="2026-01-01T00:01:00+00:00",
            elapsed_seconds=60.0,
        )
        write_run_manifest(
            path,
            configuration={"run": True},
            stages=[stage],
            required_statistics_count=34,
            figure_count=16,
            bundle_path=Path("/governed/bundle.zip"),
        )
        payload = json.loads(path.read_text())
        assert payload["required_statistics_count"] == 34
        assert payload["canonical_figure_count"] == 16
        assert payload["stages"][0]["status"] == "completed"


def test_master_databricks_notebook_is_single_complete_entrypoint():
    notebook = REPO_ROOT / "notebooks" / "09_run_complete_manuscript_pipeline.py"
    source = notebook.read_text(encoding="utf-8")
    ast.parse(source)
    for required in (
        "epigenetics.ipynb",
        "08_three_clock_symmetric_calibration.py",
        "06_retinal_epigenetic_2x2_figure_s2.ipynb",
        "discover_renderer_statistics",
        "render_all",
        "validate_figure_manifest",
        "canonical_slide_previews",
        "build_governed_bundle",
        "show_figures_inline",
        "show_statistics_previews",
        "Figure statistics and provenance",
        '"stop_on_child_failure",\n        "false",',
        "checkpoint_stage_records",
        "pipeline_failure_diagnostic.md",
        "contact_sheet.pdf",
    ):
        assert required in source
