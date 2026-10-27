# Databricks notebook source
# MAGIC %md
# MAGIC # Export ten CLSA fundus photographs by glaucoma history
# MAGIC
# MAGIC This standalone notebook exports exactly one quality-passing baseline
# MAGIC fundus photograph from each of ten unique CLSA participants:
# MAGIC
# MAGIC - 3 participants who answered **No** to baseline `ICQ_GLAUC_COM`; and
# MAGIC - 7 participants who answered **Yes** to baseline `ICQ_GLAUC_COM`.
# MAGIC
# MAGIC Files are named `<participant_id>_0.<extension>` for No and
# MAGIC `<participant_id>_1.<extension>` for Yes. The image folder contains only
# MAGIC the ten fundus files. Because participant identifiers are retained in the
# MAGIC filenames, the output must remain in the governed CLSA environment.

# COMMAND ----------
from pathlib import Path
import json
import re
import shutil
import uuid
import zipfile

import numpy as np
import pandas as pd


# ------------------------------- CONFIG ---------------------------------
DATASET_ROOT = Path("/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset")
DERIVED_ROOT = DATASET_ROOT / "derived" / "clsa_retinal_aging"

QUALITY_MANIFEST_PATH = (
    DERIVED_ROOT
    / "fundus_retfound"
    / "01_quality"
    / "fundus_quality_manifest.parquet"
)
BASELINE_QUESTIONNAIRE_PATH = (
    DERIVED_ROOT
    / "questionnaire_extracted_sap"
    / "BL"
    / "2209017_UOttawa_EFreeman_Baseline_CoPv7_Qx_CANUE_PA_BS.csv"
)
BASELINE_QUESTIONNAIRE_ZIP = DATASET_ROOT / "2209017_UOttawa_EFreeman_BL.zip"
BASELINE_MEMBER_SUFFIX = "CoPv7_Qx_CANUE_PA_BS.csv"

OUTPUT_ROOT = (
    DERIVED_ROOT
    / "Age_Glaucoma"
    / "18_glaucoma_history_10_scan_export"
)
IMAGE_OUTPUT_DIR = OUTPUT_ROOT / "fundus_10"
PRIVATE_SELECTION_MANIFEST = OUTPUT_ROOT / "selection_manifest_private.csv"
RUN_MANIFEST = OUTPUT_ROOT / "run_manifest.json"

PARTICIPANT_COLUMN = "participant_id"
QUESTIONNAIRE_ID_COLUMN = "entity_id"
GLAUCOMA_HISTORY_COLUMN = "ICQ_GLAUC_COM"
VISIT = "BL"

N_NO_GLAUCOMA = 3
N_YES_GLAUCOMA = 7
RANDOM_SEED = 20261027

# Prefer the standardized quality-pipeline image when it was persisted;
# otherwise copy the original fundus image. No image is transformed here.
IMAGE_PATH_PREFERENCE = ("processed_image_path", "image_path")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".dcm"}
# ------------------------------------------------------------------------


YES_CODES = {"1", "11", "YES", "Y", "TRUE", "T"}
NO_CODES = {"0", "2", "NO", "N", "FALSE", "F"}
MISSING_CODES = {
    "",
    "8",
    "08",
    "9",
    "09",
    "-8",
    "-9",
    "-77771",
    "-77772",
    "-77777",
    "-88880",
    "-88888",
    "-88889",
    "-99991",
    "-99993",
    "-99998",
    "-99999",
    "NA",
    "N/A",
    "NAN",
    "NONE",
    "NULL",
    "MISSING",
    "DON'T KNOW",
    "DO NOT KNOW",
    "REFUSED",
}


def normalized_code(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .str.replace(r"\.0$", "", regex=True)
    )


def normalize_identifier(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .mask(lambda values: values.isin(["", "NAN", "NONE", "<NA>"]))
    )


def normalize_visit(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .replace({"BASELINE": "BL", "FUP1": "F1"})
    )


def coerce_glaucoma_history(series: pd.Series) -> pd.Series:
    """Strictly map the released baseline Yes/No history field."""
    codes = normalized_code(series)
    missing = codes.isna() | codes.isin(MISSING_CODES)
    unknown = ~(missing | codes.isin(YES_CODES | NO_CODES))
    if unknown.any():
        counts = codes[unknown].value_counts().head(20).to_dict()
        raise ValueError(
            f"{GLAUCOMA_HISTORY_COLUMN} contains unmapped response codes: {counts}"
        )
    output = pd.Series(np.nan, index=series.index, dtype=float)
    output.loc[codes.isin(YES_CODES)] = 1.0
    output.loc[codes.isin(NO_CODES)] = 0.0
    return output.mask(missing)


def coerce_quality_pass(series: pd.Series) -> pd.Series:
    codes = normalized_code(series)
    positive = codes.isin({"1", "TRUE", "T", "YES", "Y"})
    negative = codes.isin({"0", "FALSE", "F", "NO", "N"})
    unknown = ~(codes.isna() | positive | negative)
    if unknown.any():
        counts = codes[unknown].value_counts().head(20).to_dict()
        raise ValueError(f"quality_pass contains unmapped values: {counts}")
    output = pd.Series(pd.NA, index=series.index, dtype="boolean")
    output.loc[positive] = True
    output.loc[negative] = False
    return output


def normalized_fuse_path(value: object) -> Path | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    if text.startswith("file:"):
        text = text.removeprefix("file:")
    elif text.startswith("dbfs:/Volumes/"):
        text = "/Volumes/" + text.removeprefix("dbfs:/Volumes/")
    elif text.startswith("dbfs:/"):
        text = "/dbfs/" + text.removeprefix("dbfs:/")
    return Path(text)


def safe_participant_identifier(value: object) -> str:
    original = str(value).strip()
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", original)
    if not safe:
        raise ValueError(f"Participant identifier cannot form a filename: {value!r}")
    if safe != original:
        raise ValueError(
            "Participant identifier contains characters that would be changed in "
            f"a filename: {original!r}"
        )
    return safe


def load_baseline_glaucoma_history() -> tuple[pd.DataFrame, str]:
    columns = [QUESTIONNAIRE_ID_COLUMN, GLAUCOMA_HISTORY_COLUMN]
    if BASELINE_QUESTIONNAIRE_PATH.is_file():
        frame = pd.read_csv(
            BASELINE_QUESTIONNAIRE_PATH,
            usecols=columns,
            dtype="string",
            low_memory=False,
        )
        source = str(BASELINE_QUESTIONNAIRE_PATH)
    elif BASELINE_QUESTIONNAIRE_ZIP.is_file():
        with zipfile.ZipFile(BASELINE_QUESTIONNAIRE_ZIP) as archive:
            members = [
                name
                for name in archive.namelist()
                if name.endswith(BASELINE_MEMBER_SUFFIX)
            ]
            if len(members) != 1:
                raise ValueError(
                    "Expected exactly one baseline questionnaire member ending in "
                    f"{BASELINE_MEMBER_SUFFIX!r}; found {members}"
                )
            with archive.open(members[0]) as member:
                frame = pd.read_csv(
                    member,
                    usecols=columns,
                    dtype="string",
                    low_memory=False,
                )
            source = f"{BASELINE_QUESTIONNAIRE_ZIP}!{members[0]}"
    else:
        raise FileNotFoundError(
            "Baseline questionnaire was not found at either:\n- "
            f"{BASELINE_QUESTIONNAIRE_PATH}\n- {BASELINE_QUESTIONNAIRE_ZIP}"
        )

    frame = frame.rename(columns={QUESTIONNAIRE_ID_COLUMN: PARTICIPANT_COLUMN})
    frame[PARTICIPANT_COLUMN] = normalize_identifier(frame[PARTICIPANT_COLUMN])
    frame["glaucoma_history_label"] = coerce_glaucoma_history(
        frame[GLAUCOMA_HISTORY_COLUMN]
    )
    frame = frame.dropna(subset=[PARTICIPANT_COLUMN])

    conflicting = (
        frame.dropna(subset=["glaucoma_history_label"])
        .groupby(PARTICIPANT_COLUMN)["glaucoma_history_label"]
        .nunique()
    )
    if conflicting.gt(1).any():
        raise ValueError(
            "Baseline questionnaire contains conflicting glaucoma responses for "
            f"{int(conflicting.gt(1).sum())} participants"
        )
    frame = frame.drop_duplicates(PARTICIPANT_COLUMN, keep="first")
    return frame, source


def load_quality_manifest() -> tuple[pd.DataFrame, str]:
    if QUALITY_MANIFEST_PATH.is_file() or QUALITY_MANIFEST_PATH.is_dir():
        frame = pd.read_parquet(QUALITY_MANIFEST_PATH)
        source = str(QUALITY_MANIFEST_PATH)
    else:
        batch_paths = sorted(
            QUALITY_MANIFEST_PATH.parent.glob(
                "batches/batch_*/fundus_quality_manifest.parquet"
            )
        )
        if not batch_paths:
            raise FileNotFoundError(
                "No consolidated or batched quality manifest was found under "
                f"{QUALITY_MANIFEST_PATH.parent}"
            )
        frame = pd.concat(
            [pd.read_parquet(path) for path in batch_paths],
            ignore_index=True,
        )
        source = f"{len(batch_paths)} completed batch manifests"

    required = {PARTICIPANT_COLUMN, "visit", "image_path", "quality_pass"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Quality manifest is missing columns: {missing}")
    if not any(column in frame.columns for column in IMAGE_PATH_PREFERENCE):
        raise ValueError(
            "Quality manifest has none of the configured image path columns: "
            f"{IMAGE_PATH_PREFERENCE}"
        )

    frame = frame.copy()
    frame[PARTICIPANT_COLUMN] = normalize_identifier(frame[PARTICIPANT_COLUMN])
    frame["visit"] = normalize_visit(frame["visit"])
    frame["quality_pass_clean"] = coerce_quality_pass(frame["quality_pass"])
    frame = frame[
        frame["visit"].eq(VISIT)
        & frame["quality_pass_clean"].eq(True)
        & frame[PARTICIPANT_COLUMN].notna()
    ].copy()
    frame = frame.drop_duplicates("image_path", keep="last")
    return frame, source


def choose_existing_image(row: pd.Series) -> tuple[Path | None, str | None]:
    for column in IMAGE_PATH_PREFERENCE:
        if column not in row.index:
            continue
        path = normalized_fuse_path(row[column])
        if path is not None and path.is_file():
            return path, column
    return None, None


def select_one_image_per_participant(
    candidates: pd.DataFrame,
    label: int,
    number: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    group = candidates[candidates["glaucoma_history_label"].eq(label)].copy()
    participant_ids = np.asarray(
        sorted(group[PARTICIPANT_COLUMN].dropna().astype(str).unique())
    )
    if len(participant_ids) < number:
        raise ValueError(
            f"Only {len(participant_ids)} eligible participants are available for "
            f"glaucoma label {label}; {number} are required"
        )
    selected_ids = rng.choice(participant_ids, size=number, replace=False)
    selected_rows = []
    for participant_id in selected_ids:
        available = group[group[PARTICIPANT_COLUMN].eq(participant_id)].sort_values(
            [column for column in ("eye", "image_path") if column in group.columns],
            kind="stable",
        )
        chosen_index = int(rng.integers(0, len(available)))
        record = available.iloc[chosen_index].to_dict()
        record["glaucoma_history_label"] = int(label)
        selected_rows.append(record)
    return selected_rows


def main() -> None:
    if N_NO_GLAUCOMA != 3 or N_YES_GLAUCOMA != 7:
        raise ValueError("This export is prespecified as exactly 3 No and 7 Yes")

    labels, questionnaire_source = load_baseline_glaucoma_history()
    quality, quality_source = load_quality_manifest()

    raw_code_audit = (
        labels.assign(raw_code=normalized_code(labels[GLAUCOMA_HISTORY_COLUMN]))
        .groupby(["raw_code", "glaucoma_history_label"], dropna=False)
        .size()
        .reset_index(name="participants")
    )
    print("Baseline glaucoma-history response audit:")
    print(raw_code_audit.to_string(index=False))

    quality[["resolved_source_path", "resolved_source_column"]] = quality.apply(
        lambda row: pd.Series(choose_existing_image(row)), axis=1
    )
    quality = quality.dropna(subset=["resolved_source_path"]).copy()
    quality["resolved_source_path"] = quality["resolved_source_path"].map(Path)

    candidates = quality.merge(
        labels[
            [
                PARTICIPANT_COLUMN,
                GLAUCOMA_HISTORY_COLUMN,
                "glaucoma_history_label",
            ]
        ],
        on=PARTICIPANT_COLUMN,
        how="inner",
        validate="many_to_one",
    ).dropna(subset=["glaucoma_history_label"])
    candidates["glaucoma_history_label"] = candidates[
        "glaucoma_history_label"
    ].astype(int)
    if candidates.empty:
        raise ValueError("No baseline quality-passing fundus images joined to glaucoma history")

    available_summary = (
        candidates.groupby("glaucoma_history_label", as_index=False)
        .agg(
            participants=(PARTICIPANT_COLUMN, "nunique"),
            images=("resolved_source_path", "count"),
        )
        .sort_values("glaucoma_history_label")
    )
    print("\nEligible baseline image pool:")
    print(available_summary.to_string(index=False))

    rng = np.random.default_rng(RANDOM_SEED)
    selected_records = [
        *select_one_image_per_participant(
            candidates, 0, N_NO_GLAUCOMA, rng
        ),
        *select_one_image_per_participant(
            candidates, 1, N_YES_GLAUCOMA, rng
        ),
    ]
    selected = pd.DataFrame(selected_records)
    if len(selected) != 10 or selected[PARTICIPANT_COLUMN].nunique() != 10:
        raise RuntimeError("Selection did not produce ten unique participants")
    observed_counts = selected["glaucoma_history_label"].value_counts().to_dict()
    if observed_counts != {1: 7, 0: 3}:
        raise RuntimeError(f"Selection has incorrect class counts: {observed_counts}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if IMAGE_OUTPUT_DIR.exists():
        existing = list(IMAGE_OUTPUT_DIR.iterdir())
        raise FileExistsError(
            f"Output folder already exists with {len(existing)} entries: "
            f"{IMAGE_OUTPUT_DIR}. Remove or rename it before rerunning."
        )

    staging = OUTPUT_ROOT / f".fundus_10_staging_{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    manifest_rows = []
    try:
        for record in selected.to_dict("records"):
            source_path = Path(record["resolved_source_path"])
            extension = source_path.suffix.lower()
            if extension not in SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"Unsupported image extension {extension!r}: {source_path}"
                )
            participant_id = safe_participant_identifier(
                record[PARTICIPANT_COLUMN]
            )
            label = int(record["glaucoma_history_label"])
            output_name = f"{participant_id}_{label}{extension}"
            destination = staging / output_name
            if destination.exists():
                raise FileExistsError(f"Duplicate output filename: {destination.name}")
            shutil.copy2(source_path, destination)
            if not destination.is_file() or destination.stat().st_size == 0:
                raise IOError(f"Copied image is missing or empty: {destination}")
            manifest_rows.append(
                {
                    "participant_id": participant_id,
                    "glaucoma_history_label": label,
                    "raw_glaucoma_history_response": record[
                        GLAUCOMA_HISTORY_COLUMN
                    ],
                    "source_image_path": str(source_path),
                    "source_path_column": record["resolved_source_column"],
                    "output_filename": output_name,
                    "output_bytes": int(destination.stat().st_size),
                    "random_seed": RANDOM_SEED,
                }
            )

        staged_files = [path for path in staging.iterdir() if path.is_file()]
        if len(staged_files) != 10:
            raise RuntimeError(
                f"Staging folder contains {len(staged_files)} files instead of 10"
            )
        staging.replace(IMAGE_OUTPUT_DIR)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    private_manifest = pd.DataFrame(manifest_rows).sort_values(
        ["glaucoma_history_label", "participant_id"], kind="stable"
    )
    private_manifest.to_csv(PRIVATE_SELECTION_MANIFEST, index=False)
    run_manifest = {
        "completed": True,
        "questionnaire_field": GLAUCOMA_HISTORY_COLUMN,
        "visit": VISIT,
        "questionnaire_source": questionnaire_source,
        "quality_source": quality_source,
        "quality_passing_only": True,
        "one_image_per_participant": True,
        "random_seed": RANDOM_SEED,
        "requested_counts": {"no_0": N_NO_GLAUCOMA, "yes_1": N_YES_GLAUCOMA},
        "selected_counts": {
            "no_0": int((private_manifest["glaucoma_history_label"] == 0).sum()),
            "yes_1": int((private_manifest["glaucoma_history_label"] == 1).sum()),
        },
        "image_output_dir": str(IMAGE_OUTPUT_DIR),
        "private_selection_manifest": str(PRIVATE_SELECTION_MANIFEST),
        "filename_contract": "<participant_id>_<0=no,1=yes>.<source extension>",
    }
    RUN_MANIFEST.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    final_files = sorted(path.name for path in IMAGE_OUTPUT_DIR.iterdir())
    if len(final_files) != 10:
        raise RuntimeError("Final image folder does not contain exactly ten files")
    print("\nExport complete:", IMAGE_OUTPUT_DIR)
    print("Files:")
    for filename in final_files:
        print(" -", filename)
    print("Private selection manifest:", PRIVATE_SELECTION_MANIFEST)


if __name__ == "__main__":
    main()
