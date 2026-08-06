"""Utilities for handing completed quality-control batches to RETFound."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterator, Sequence


PASS_VALUES = {
    "1",
    "true",
    "t",
    "y",
    "yes",
    "pass",
    "passed",
    "gradable",
    "usable",
}
FAIL_VALUES = {
    "0",
    "false",
    "f",
    "n",
    "no",
    "fail",
    "failed",
    "ungradable",
    "unusable",
}

QUALITY_BATCH_PATTERN = re.compile(r"^batch_(\d{9})_(\d{9})$")


def parse_quality_batch_name(name: str) -> tuple[int, int]:
    """Return the manifest row interval encoded in a quality batch name."""

    match = QUALITY_BATCH_PATTERN.fullmatch(name)
    if not match:
        raise ValueError(
            "Quality batch name must match batch_000000000_000000500; "
            f"received {name!r}."
        )
    start, stop = (int(value) for value in match.groups())
    if stop <= start:
        raise ValueError(f"Quality batch has an invalid interval: {name}")
    return start, stop


def read_completed_quality_batch(path: str | Path):
    """Read a quality Parquet only when its encoded row interval is complete.

    Returns ``None`` while a file is unreadable, has the wrong number of rows,
    or contains incomplete/duplicate image paths. The live consumer can safely
    retry it on its next polling cycle.
    """

    import pandas as pd

    path = Path(path)
    start, stop = parse_quality_batch_name(path.parent.name)
    expected_rows = stop - start
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return None
    if "image_path" not in frame.columns or len(frame) != expected_rows:
        return None
    paths = frame["image_path"]
    if paths.isna().any() or paths.astype(str).nunique() != expected_rows:
        return None
    frame = frame.copy()
    frame["quality_source_batch"] = path.parent.name
    frame["quality_source_path"] = str(path)
    return frame


def _normalized_status(value: object) -> str | None:
    import pandas as pd

    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip().lower()


def select_quality_passed(
    quality_frame,
    pass_columns: Sequence[str] = ("quality_pass",),
):
    """Return rows that explicitly pass every requested quality flag.

    Missing values are failures. Unknown non-missing values raise an error so a
    newly introduced AutoMorph code cannot silently enter the RETFound cohort.
    """

    import pandas as pd

    if "image_path" not in quality_frame.columns:
        raise ValueError("Quality manifest must contain image_path.")
    if not pass_columns:
        raise ValueError("At least one pass column is required.")
    missing = sorted(set(pass_columns) - set(quality_frame.columns))
    if missing:
        raise ValueError(f"Quality manifest is missing pass columns: {missing}")

    working = quality_frame.copy()
    combined = pd.Series(True, index=working.index, dtype=bool)
    for column in pass_columns:
        normalized = working[column].map(_normalized_status)
        unknown = sorted(
            {
                value
                for value in normalized.dropna().unique().tolist()
                if value not in PASS_VALUES and value not in FAIL_VALUES
            }
        )
        if unknown:
            raise ValueError(
                f"Unrecognized values in {column}: {unknown[:10]}. "
                "Map these to an explicit pass/fail flag before RETFound."
            )
        combined &= normalized.isin(PASS_VALUES)

    working["retfound_quality_eligible"] = combined
    passed = working.loc[combined].copy()
    return passed, working


def load_completed_quality_manifests(
    quality_root: str | Path,
):
    """Load completed per-batch manifests, or the consolidated manifest."""

    import pandas as pd

    root = Path(quality_root)
    batch_paths = sorted(
        root.glob("batches/batch_*/fundus_quality_manifest.parquet")
    )
    if batch_paths:
        paths = batch_paths
        source_mode = "completed_quality_batches"
    else:
        consolidated = root / "fundus_quality_manifest.parquet"
        if not consolidated.exists():
            raise FileNotFoundError(
                "No completed quality manifests were found under "
                f"{root}. Expected batches/batch_*/fundus_quality_manifest.parquet "
                "or fundus_quality_manifest.parquet."
            )
        paths = [consolidated]
        source_mode = "consolidated_quality_manifest"

    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        if "image_path" not in frame.columns:
            raise ValueError(f"Quality batch lacks image_path: {path}")
        source_batch = path.parent.name if path.parent.name.startswith("batch_") else ""
        frame = frame.copy()
        frame["quality_source_batch"] = source_batch
        frame["quality_source_path"] = str(path)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["image_path"] = combined["image_path"].astype(str)
    combined = combined.drop_duplicates(subset=["image_path"], keep="last")
    return combined, [str(path) for path in paths], source_mode


def iter_stable_quality_batches(
    passed_frame,
    batch_size: int = 500,
) -> Iterator[tuple[str, object]]:
    """Yield deterministic batches that remain stable as new QC batches arrive."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if passed_frame.empty:
        return

    working = passed_frame.copy()
    if "quality_source_batch" not in working.columns:
        working["quality_source_batch"] = ""
    working["quality_source_batch"] = working["quality_source_batch"].fillna("")
    working = working.sort_values(
        ["quality_source_batch", "image_path"], kind="stable"
    )

    for source_batch, source_frame in working.groupby(
        "quality_source_batch", sort=True, dropna=False
    ):
        source_frame = source_frame.reset_index(drop=True)
        for start in range(0, len(source_frame), batch_size):
            stop = min(start + batch_size, len(source_frame))
            if source_batch:
                safe_source = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_batch)
                key = safe_source
                if len(source_frame) > batch_size:
                    key += f"_part_{start:09d}_{stop:09d}"
            else:
                key = f"batch_{start:09d}_{stop:09d}"
            yield key, source_frame.iloc[start:stop].copy()
