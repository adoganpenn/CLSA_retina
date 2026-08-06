"""Cohort utilities for the CLSA/Zeiss age-matched retinal study.

The module deliberately contains no Databricks or Spark globals.  It decodes
Zeiss DICOMs to auditable lossless RGB files, sends those files through the
same quality function used by the CLSA pipeline, and performs deterministic
participant-level age matching without replacement.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def _require_columns(frame: Any, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _to_rgb_uint8(pixel_array: Any, photometric: str = "") -> Any:
    """Convert a decoded DICOM pixel array to an H x W x 3 uint8 array."""
    import numpy as np

    array = np.asarray(pixel_array)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    elif array.ndim == 3 and array.shape[0] in {1, 3, 4} and array.shape[-1] > 4:
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3:
        raise ValueError(f"Unsupported DICOM pixel shape: {array.shape}")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[-1] >= 3:
        array = array[..., :3]
    else:
        raise ValueError(f"Unsupported DICOM channel count: {array.shape[-1]}")

    array = array.astype(np.float32, copy=False)
    finite = np.isfinite(array)
    if not finite.any():
        raise ValueError("DICOM contains no finite pixel values")
    low, high = np.nanpercentile(array[finite], [0.5, 99.5])
    if high <= low:
        low = float(np.nanmin(array[finite]))
        high = float(np.nanmax(array[finite]))
    if high <= low:
        raise ValueError("DICOM has a constant pixel array")
    array = np.clip((array - low) / (high - low), 0.0, 1.0)
    if str(photometric).upper() == "MONOCHROME1":
        array = 1.0 - array
    return np.rint(array * 255.0).astype(np.uint8)


def materialize_dicom_rgb(dcm_path: str, destination: str | Path) -> Path:
    """Decode one DICOM and save an RGB PNG for the shared CLSA QC code."""
    import pydicom
    from PIL import Image

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset = pydicom.dcmread(str(dcm_path), force=True)
    rgb = _to_rgb_uint8(
        dataset.pixel_array,
        str(getattr(dataset, "PhotometricInterpretation", "")),
    )
    Image.fromarray(rgb, mode="RGB").save(destination, format="PNG")
    return destination


def load_zeiss_embedding_chunks(
    chunks_dir: str | Path,
    expected_embedding_dim: int = 1024,
) -> Any:
    """Read and validate the completed Zeiss RETFound chunk Parquets."""
    import numpy as np
    import pandas as pd

    chunks_dir = Path(chunks_dir)
    paths = sorted(chunks_dir.glob("chunk_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No chunk_*.parquet files found: {chunks_dir}")
    frames = [pd.read_parquet(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    _require_columns(
        frame,
        ["dcm_path", "patient_id", "age", "embedding"],
        "Zeiss embedding chunks",
    )
    frame["dcm_path"] = frame["dcm_path"].astype(str)
    frame["patient_id"] = frame["patient_id"].astype(str)
    duplicate_count = int(frame["dcm_path"].duplicated(keep=False).sum())
    if duplicate_count:
        frame = frame.drop_duplicates("dcm_path", keep="last").reset_index(drop=True)
        print(f"Removed {duplicate_count:,} duplicate Zeiss embedding rows by dcm_path")
    dimensions = frame["embedding"].map(
        lambda value: int(np.asarray(value).reshape(-1).size)
    )
    invalid = dimensions.ne(expected_embedding_dim)
    if invalid.any():
        raise ValueError(
            f"{int(invalid.sum()):,} Zeiss vectors do not have "
            f"{expected_embedding_dim} elements"
        )
    return frame


def run_zeiss_clsa_quality(
    zeiss_embeddings: Any,
    output_root: str | Path,
    quality_config: Any,
    batch_size: int = 500,
    resume: bool = True,
) -> Any:
    """Decode Zeiss DICOMs and run the repository's exact CLSA QC in batches."""
    import pandas as pd

    from fundus_retfound_pipeline import run_quality_pipeline, write_frame

    _require_columns(
        zeiss_embeddings,
        ["dcm_path", "patient_id", "age"],
        "Zeiss embedding table",
    )
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    output_root = Path(output_root)
    decoded_root = output_root / "decoded_rgb"
    batches_root = output_root / "quality_batches"
    decoded_root.mkdir(parents=True, exist_ok=True)
    batches_root.mkdir(parents=True, exist_ok=True)

    source = (
        zeiss_embeddings.drop(columns=["embedding"], errors="ignore")
        .drop_duplicates("dcm_path", keep="last")
        .sort_values("dcm_path", kind="stable")
        .reset_index(drop=True)
    )
    results = []
    total_batches = (len(source) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(range(0, len(source), batch_size), 1):
        stop = min(start + batch_size, len(source))
        batch_name = f"batch_{start:09d}_{stop:09d}"
        batch_dir = batches_root / batch_name
        output_path = batch_dir / "fundus_quality_manifest.parquet"
        expected_paths = set(source.iloc[start:stop]["dcm_path"].astype(str))
        if resume and output_path.exists():
            cached = pd.read_parquet(output_path)
            if (
                "dcm_path" in cached.columns
                and len(cached) == stop - start
                and set(cached["dcm_path"].astype(str)) == expected_paths
            ):
                print(f"[Zeiss QC {batch_number}/{total_batches}] resumed {batch_name}")
                results.append(cached)
                continue

        batch = source.iloc[start:stop].copy()
        image_paths = []
        decode_errors = []
        print(
            f"[Zeiss QC {batch_number}/{total_batches}] decoding and checking "
            f"rows {start:,}:{stop:,}"
        )
        for dcm_path in batch["dcm_path"].astype(str):
            digest = hashlib.sha1(dcm_path.encode("utf-8")).hexdigest()
            destination = decoded_root / f"{digest}.png"
            error = None
            try:
                if not destination.exists():
                    materialize_dicom_rgb(dcm_path, destination)
            except Exception as exc:  # retained as an auditable QC failure
                error = f"{type(exc).__name__}: {str(exc)}"[:500]
            image_paths.append(str(destination))
            decode_errors.append(error)
        batch["image_path"] = image_paths
        batch["dicom_decode_error"] = decode_errors
        quality = run_quality_pipeline(batch, batch_dir, quality_config)
        quality.loc[quality["dicom_decode_error"].notna(), "quality_pass"] = False
        quality.loc[
            quality["dicom_decode_error"].notna(), "quality_reasons"
        ] = "dicom_decode_failed"
        write_frame(quality, output_path)
        results.append(quality)
        print(
            f"[Zeiss QC {batch_number}/{total_batches}] saved {len(quality):,} rows; "
            f"passed {int(quality['quality_pass'].sum()):,}"
        )
    output = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    write_frame(output, output_root / "zeiss_clsa_quality_manifest.parquet")
    metadata = {
        "n_source_images": int(len(source)),
        "n_quality_rows": int(len(output)),
        "n_quality_pass": int(output["quality_pass"].sum()) if len(output) else 0,
        "batch_size": int(batch_size),
        "resume": bool(resume),
        "quality_config": asdict(quality_config),
    }
    (output_root / "zeiss_clsa_quality_run.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return output


def greedy_age_match(
    cases: Any,
    controls: Any,
    ratio: int = 1,
    caliper_years: float = 1.0,
    exact_sex: bool = False,
    case_id: str = "patient_id",
    case_age: str = "age",
    control_id: str = "participant_id",
    control_age: str = "age_at_fundus_years",
) -> tuple[Any, Any]:
    """Greedy nearest-age matching without reuse of a CLSA participant."""
    import pandas as pd

    _require_columns(cases, [case_id, case_age], "Cases")
    _require_columns(controls, [control_id, control_age], "Controls")
    if ratio < 1:
        raise ValueError("ratio must be at least 1")
    if caliper_years < 0:
        raise ValueError("caliper_years cannot be negative")
    if exact_sex:
        _require_columns(cases, ["sex"], "Cases")
        _require_columns(controls, ["sex_at_birth"], "Controls")

    case_frame = cases.copy()
    control_frame = controls.copy()
    case_frame[case_age] = pd.to_numeric(case_frame[case_age], errors="coerce")
    control_frame[control_age] = pd.to_numeric(
        control_frame[control_age], errors="coerce"
    )
    case_frame = case_frame.dropna(subset=[case_id, case_age]).copy()
    control_frame = control_frame.dropna(subset=[control_id, control_age]).copy()
    case_frame[case_id] = case_frame[case_id].astype(str)
    control_frame[control_id] = control_frame[control_id].astype(str)
    case_frame = case_frame.sort_values([case_age, case_id], kind="stable")

    used_controls: set[str] = set()
    pair_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for case_index, (_, case) in enumerate(case_frame.iterrows(), 1):
        candidates = control_frame[
            ~control_frame[control_id].isin(used_controls)
        ].copy()
        candidates["age_difference_years"] = (
            candidates[control_age] - float(case[case_age])
        )
        candidates["absolute_age_difference_years"] = candidates[
            "age_difference_years"
        ].abs()
        candidates = candidates[
            candidates["absolute_age_difference_years"] <= caliper_years
        ]
        if exact_sex:
            case_sex = str(case["sex"]).strip().upper()
            candidates = candidates[
                candidates["sex_at_birth"].astype(str).str.strip().str.upper()
                == case_sex
            ]
        sort_columns = ["absolute_age_difference_years", control_id]
        if "visit" in candidates.columns:
            sort_columns.append("visit")
        candidates = (
            candidates.sort_values(sort_columns, kind="stable")
            # BL and F1 records from the same CLSA participant are alternatives,
            # never two separate controls within one match set.
            .drop_duplicates(control_id, keep="first")
        )
        if len(candidates) < ratio:
            audit_rows.append(
                {
                    "case_id": str(case[case_id]),
                    "case_age": float(case[case_age]),
                    "matched": False,
                    "reason": "insufficient_controls_within_caliper",
                    "eligible_controls_remaining": int(len(candidates)),
                }
            )
            continue
        selected = candidates.head(ratio)
        match_set_id = f"M{case_index:07d}"
        for rank, (_, control) in enumerate(selected.iterrows(), 1):
            used_controls.add(str(control[control_id]))
            pair_rows.append(
                {
                    "match_set_id": match_set_id,
                    "match_rank": rank,
                    "zeiss_patient_id": str(case[case_id]),
                    "zeiss_age_years": float(case[case_age]),
                    "clsa_participant_id": str(control[control_id]),
                    "clsa_visit": control.get("visit"),
                    "clsa_age_years": float(control[control_age]),
                    "age_difference_years": float(
                        control["age_difference_years"]
                    ),
                    "absolute_age_difference_years": float(
                        control["absolute_age_difference_years"]
                    ),
                }
            )
        audit_rows.append(
            {
                "case_id": str(case[case_id]),
                "case_age": float(case[case_age]),
                "matched": True,
                "reason": None,
                "eligible_controls_remaining": int(len(candidates)),
            }
        )
    return pd.DataFrame(pair_rows), pd.DataFrame(audit_rows)
