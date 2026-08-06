"""Cohort utilities for the CLSA/Zeiss age-matched retinal study.

The module deliberately contains no Databricks or Spark globals. It validates
the completed source-specific Zeiss RETFound chunks and performs deterministic
participant-level age matching without replacement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def _require_columns(frame: Any, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


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
