"""Audit and package the complete CLSA retinal-aging manuscript workflow.

This module contains filesystem-only helpers used by the Databricks master
runner. It never fits models and never reads restricted private participant
tables. The analysis notebooks remain responsible for statistical estimation;
the canonical figure renderer remains responsible for figure construction.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence
import zipfile

import pandas as pd


CANONICAL_FIGURE_IDS = (
    "01",
    "02",
    "03",
    "04",
    "05",
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "S7",
    "S8",
    "S9",
    "S10",
    "S11",
)
CANONICAL_PROFILES = ("publication", "slide")


@dataclass(frozen=True)
class StatisticRequirement:
    file: str
    producing_cell: int


@dataclass
class StageRecord:
    stage: str
    status: str
    started_utc: str
    finished_utc: str
    elapsed_seconds: float
    detail: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def discover_renderer_statistics(renderer_path: Path) -> list[StatisticRequirement]:
    """Extract every persisted Source(file, cell) used by the renderer."""

    renderer_path = Path(renderer_path)
    source = renderer_path.read_text(encoding="utf-8")
    matches = re.findall(
        r"Source\(\s*[\"']([^\"']+)[\"']\s*,\s*(\d+)\s*\)", source
    )
    requirements = sorted(
        {
            StatisticRequirement(file=file, producing_cell=int(cell))
            for file, cell in matches
        },
        key=lambda item: (item.file, item.producing_cell),
    )
    if not requirements:
        raise ValueError(f"No figure Source declarations found in {renderer_path}")
    return requirements


def audit_required_statistics(
    statistics_root: Path,
    requirements: Sequence[StatisticRequirement],
) -> pd.DataFrame:
    """Return a complete pre-render audit of canonical figure inputs."""

    statistics_root = Path(statistics_root)
    rows = []
    for requirement in requirements:
        path = statistics_root / requirement.file
        exists = path.is_file()
        rows.append(
            {
                "file": requirement.file,
                "producing_cell": requirement.producing_cell,
                "path": str(path),
                "exists": exists,
                "bytes": int(path.stat().st_size) if exists else 0,
                "sha256": sha256_file(path) if exists else "",
            }
        )
    return pd.DataFrame(rows)


def require_complete_statistics(audit: pd.DataFrame) -> None:
    missing = audit.loc[~audit["exists"].astype(bool)]
    if missing.empty:
        return
    details = "; ".join(
        f"{row.file} (main analysis cell {row.producing_cell})"
        for row in missing.itertuples(index=False)
    )
    raise FileNotFoundError(
        "Canonical manuscript statistics are incomplete. Missing: " + details
    )


def flatten_figure_manifest(manifest_path: Path) -> pd.DataFrame:
    """Convert the renderer provenance manifest to a figure/panel catalog."""

    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for figure_key, figure in payload.get("figures", {}).items():
        figure_id = figure_key.replace("figure_", "")
        for panel, panel_data in figure.get("panels", {}).items():
            sources = panel_data.get("sources", [])
            for profile, paths in panel_data.get("output_paths", {}).items():
                rows.append(
                    {
                        "figure_id": figure_id,
                        "panel": panel,
                        "profile": profile,
                        "output_paths": "|".join(paths),
                        "n": panel_data.get("n"),
                        "source_files": "|".join(
                            source.get("file", "") for source in sources
                        ),
                        "source_cells": "|".join(
                            str(source.get("cell", "")) for source in sources
                        ),
                    }
                )
    return pd.DataFrame(rows)


def validate_figure_manifest(
    manifest_path: Path,
    *,
    expected_figure_ids: Sequence[str] = CANONICAL_FIGURE_IDS,
    expected_profiles: Sequence[str] = CANONICAL_PROFILES,
) -> pd.DataFrame:
    """Require every canonical figure and export profile to be present on disk."""

    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    figures = payload.get("figures", {})
    rows = []
    for figure_id in expected_figure_ids:
        key = f"figure_{figure_id}"
        figure = figures.get(key, {})
        for profile in expected_profiles:
            profile_data = figure.get("profiles", {}).get(profile, {})
            paths = [Path(value) for value in profile_data.get("paths", [])]
            exists = bool(paths) and all(path.is_file() for path in paths)
            rows.append(
                {
                    "figure_id": figure_id,
                    "profile": profile,
                    "manifest_entry": bool(profile_data),
                    "output_count": len(paths),
                    "all_outputs_exist": exists,
                    "paths": "|".join(str(path) for path in paths),
                }
            )
    result = pd.DataFrame(rows)
    failed = result.loc[
        ~result["manifest_entry"].astype(bool)
        | ~result["all_outputs_exist"].astype(bool)
    ]
    if not failed.empty:
        labels = ", ".join(
            f"{row.figure_id}/{row.profile}" for row in failed.itertuples(index=False)
        )
        raise FileNotFoundError(f"Incomplete canonical figure exports: {labels}")
    return result


def catalog_files(
    roots: Sequence[tuple[str, Path]],
    *,
    suffixes: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Catalog output files with size and hash without opening their contents."""

    allowed = {value.lower() for value in suffixes} if suffixes else None
    rows = []
    for category, root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for path in sorted(value for value in root.rglob("*") if value.is_file()):
            if allowed is not None and path.suffix.lower() not in allowed:
                continue
            rows.append(
                {
                    "category": category,
                    "root": str(root),
                    "relative_path": str(path.relative_to(root)),
                    "path": str(path),
                    "suffix": path.suffix.lower(),
                    "bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
    return pd.DataFrame(rows)


def canonical_slide_previews(figure_root: Path) -> list[Path]:
    figure_root = Path(figure_root)
    paths = [
        figure_root / "slide" / f"figure_{figure_id}.png"
        for figure_id in CANONICAL_FIGURE_IDS
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing slide previews: " + ", ".join(str(path) for path in missing)
        )
    return paths


def build_governed_bundle(
    destination: Path,
    *,
    figure_root: Path,
    statistics_roots: Sequence[tuple[str, Path]],
    extra_files: Sequence[Path] = (),
) -> pd.DataFrame:
    """Bundle figures and non-private statistics inside the governed volume.

    Directories or filenames containing ``private`` are excluded. The bundle is
    an internal governed artifact; it is not automatically copied to Git or a
    public location.
    """

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    allowed_statistics = {".csv", ".json", ".md", ".txt", ".parquet"}
    entries: list[tuple[Path, str, str]] = []

    figure_root = Path(figure_root)
    for path in sorted(value for value in figure_root.rglob("*") if value.is_file()):
        if any("private" in part.lower() for part in path.parts):
            continue
        entries.append((path, f"figures/{path.relative_to(figure_root)}", "figure"))

    for label, root in statistics_roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for path in sorted(value for value in root.rglob("*") if value.is_file()):
            if path.suffix.lower() not in allowed_statistics:
                continue
            if any("private" in part.lower() for part in path.parts):
                continue
            entries.append(
                (path, f"statistics/{label}/{path.relative_to(root)}", "statistics")
            )

    for path in extra_files:
        path = Path(path)
        if path.is_file() and "private" not in path.name.lower():
            entries.append((path, f"run/{path.name}", "run"))

    seen = set()
    unique_entries = []
    for entry in entries:
        if entry[1] in seen:
            continue
        seen.add(entry[1])
        unique_entries.append(entry)

    rows = []
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, archive_path, category in unique_entries:
            archive.write(path, archive_path)
            rows.append(
                {
                    "category": category,
                    "archive_path": archive_path,
                    "source_path": str(path),
                    "bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
    return pd.DataFrame(rows)


def write_run_manifest(
    path: Path,
    *,
    configuration: dict[str, object],
    stages: Sequence[StageRecord],
    required_statistics_count: int,
    figure_count: int,
    bundle_path: Path | None,
) -> None:
    payload = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "configuration": configuration,
        "stages": [asdict(record) for record in stages],
        "required_statistics_count": int(required_statistics_count),
        "canonical_figure_count": int(figure_count),
        "bundle_path": str(bundle_path) if bundle_path else None,
        "governance": (
            "Outputs remain in the approved CLSA governed volume. Private participant "
            "tables are excluded from the bundle and must not be committed to Git."
        ),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
