# Databricks notebook source
# ruff: noqa: F821 - spark, dbutils, and display are Databricks notebook globals
# MAGIC %md
# MAGIC # 04 - Mortality prediction from RETFound vectors and epigenetic ages
# MAGIC
# MAGIC **Primary question:** do baseline retinal representations or baseline
# MAGIC DNA-methylation age measures predict subsequent all-cause mortality in
# MAGIC CLSA?
# MAGIC
# MAGIC The retinal and epigenetic analyses are deliberately separate. They use
# MAGIC the same outcome definition and reporting standards, but they are not
# MAGIC restricted to the much smaller overlap between imaging and methylation.
# MAGIC
# MAGIC Primary analyses:
# MAGIC
# MAGIC 1. **Retinal cohort:** compare age + sex, retinal-age gap, RETFound-only,
# MAGIC    and age + sex + RETFound models. Both eyes are averaged before any
# MAGIC    modelling. RETFound scaling, PCA, and ridge-Cox tuning occur inside
# MAGIC    participant-level nested cross-validation.
# MAGIC 2. **Epigenetic cohort:** add each released clock measure separately to
# MAGIC    an age + sex Cox model. The primary clock is released residual age
# MAGIC    acceleration; difference acceleration, IEAA, EEAA, Horvath DNAm age,
# MAGIC    and Hannum age are secondary.
# MAGIC
# MAGIC This is prognostic association/internal-validation work, not a clinical
# MAGIC deployment model. Cause-specific mortality is not analysed because the
# MAGIC participant-status table supplies all-cause vital status and death date;
# MAGIC proxy-reported causes in the decedent questionnaire are a different data
# MAGIC source and require a separate validation plan.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Environment
# MAGIC
# MAGIC Run once on a fresh Databricks cluster, then restart Python:
# MAGIC
# MAGIC ```python
# MAGIC %pip install lifelines==0.30.0 scikit-learn==1.7.1
# MAGIC dbutils.library.restartPython()
# MAGIC ```

# COMMAND ----------
from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath
import warnings
import zipfile

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import multivariate_logrank_test
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# COMMAND ----------
# Reproducible configuration. Raw participant data and derived participant-level
# outputs remain in the approved CLSA volume and must not be committed to Git.

repo_root = "/Workspace/Users/ad0038@pennmedicine.upenn.edu/CLSA/CLSA_retina"
volume_root = "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset"
derived_root = f"{volume_root}/derived/clsa_retinal_aging"

sap_path = f"{derived_root}/sap_questionnaire_visit"
embedding_path = (
    f"{derived_root}/fundus_retfound/02_embeddings/retfound_embeddings_delta"
)
retinal_age_prediction_path_requested = ""
mortality_archive_path = (
    f"{volume_root}/2209017_UOttawa_EFreeman_Mortality_DRU_Aug2025.zip"
)
mortality_member_path = (
    "2209017_UOttawa_EFreeman_Mortality_DRU_Aug2025/"
    "2209017_UOttawa_EFreeman_ParticipantStatus_CoP_v4_May2025.csv"
)
output_root = Path(f"{derived_root}/mortality_prediction")

expected_embedding_dim = 1024
outer_folds = 5
inner_folds = 3
pca_component_grid = (8, 16, 32, 64)
ridge_penalizer_grid = (0.01, 0.1, 1.0)
evaluation_horizons_years = (5.0, 10.0)
minimum_events = 30
bootstrap_repetitions = 1000
early_death_landmark_years = 2.0
random_seed = 20260914

primary_epigenetic_measure = "epigenetic_age_acceleration_residual"
epigenetic_measures = {
    "epigenetic_age_acceleration_residual": "Horvath residual acceleration",
    "epigenetic_age_acceleration_difference": "Horvath DNAm age - age",
    "epigenetic_ieaa": "Intrinsic epigenetic age acceleration",
    "epigenetic_eeaa": "Extrinsic epigenetic age acceleration",
    "epigenetic_dnam_age": "Horvath DNAm age",
    "epigenetic_hannum_age": "Hannum DNAm age",
}

if expected_embedding_dim != 1024:
    raise ValueError("This notebook expects 1,024-element RETFound vectors.")
if outer_folds < 3 or inner_folds < 2:
    raise ValueError("Use at least 3 outer folds and 2 inner folds.")
if minimum_events < 20:
    raise ValueError("minimum_events should not be below 20.")

output_root.mkdir(parents=True, exist_ok=True)
figure_root = output_root / "figures"
table_root = output_root / "tables"
extract_root = output_root / "restricted_extract"
figure_root.mkdir(parents=True, exist_ok=True)
table_root.mkdir(parents=True, exist_ok=True)
extract_root.mkdir(parents=True, exist_ok=True)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Literature-informed design
# MAGIC
# MAGIC The design follows the main conventions used in prior retinal- and
# MAGIC methylation-age mortality work:
# MAGIC
# MAGIC - Retinal age-gap studies used prospective all-cause mortality, Cox
# MAGIC   proportional-hazards models, continuous age gap, multivariable
# MAGIC   adjustment, and risk-group displays. Zhu et al. reported HRs per
# MAGIC   one-year retinal age gap (BJO 2022; DOI:
# MAGIC   [10.1136/bjophthalmol-2021-319807](https://doi.org/10.1136/bjophthalmol-2021-319807)).
# MAGIC - eyeAge compared retinal and phenotypic acceleration with age/sex-
# MAGIC   adjusted Cox models and presented a compact HR forest plot (Ahadi et
# MAGIC   al., eLife 2023; DOI:
# MAGIC   [10.7554/eLife.82364](https://doi.org/10.7554/eLife.82364)).
# MAGIC - Horvath/Hannum mortality studies used age acceleration in prospective
# MAGIC   proportional-hazards models, commonly reporting HRs per 5-year or SD
# MAGIC   increase (Marioni et al., Genome Biology 2015; DOI:
# MAGIC   [10.1186/s13059-015-0584-6](https://doi.org/10.1186/s13059-015-0584-6);
# MAGIC   Chen et al., Aging 2016; DOI:
# MAGIC   [10.18632/aging.101020](https://doi.org/10.18632/aging.101020)).
# MAGIC - TRIPOD+AI recommends participant-independent evaluation plus both
# MAGIC   discrimination and calibration. Accordingly, this notebook reports
# MAGIC   out-of-fold C-index, paired incremental C-index, horizon-specific
# MAGIC   calibration, cohort flow, Kaplan-Meier curves, and PH diagnostics
# MAGIC   (Collins et al., BMJ 2024; DOI:
# MAGIC   [10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378)).
# MAGIC
# MAGIC **Prespecified safeguards:** baseline-only index, one record per person,
# MAGIC reverse-causation landmark sensitivity, fold-contained preprocessing and
# MAGIC tuning, no selection based on observed p-values, and no raw death dates
# MAGIC in durable analytic outputs.

# COMMAND ----------
def databricks_path_exists(path: str) -> bool:
    try:
        dbutils.fs.ls(path)
        return True
    except Exception:
        return Path(path).exists()


def require_columns(frame, columns: list[str], label: str) -> None:
    available = set(frame.columns)
    missing = sorted(set(columns) - available)
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def display_pandas(frame: pd.DataFrame, label: str) -> None:
    print(f"{label}: {len(frame):,} rows")
    if frame.empty:
        print("No rows to display.")
        return
    display(spark.createDataFrame(frame.replace({np.nan: None})))


def write_pandas_output(frame: pd.DataFrame, stem: str) -> None:
    """Write nonempty summary outputs; avoids Spark empty-schema failures."""
    if frame.empty:
        print(f"No rows to write for {stem}.")
        return
    clean = frame.replace({np.nan: None})
    csv_path = table_root / f"{stem}.csv"
    clean.to_csv(csv_path, index=False)
    spark_path = str(table_root / stem)
    (
        spark.createDataFrame(clean)
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(spark_path)
    )
    print("Wrote:", csv_path)
    print("Wrote:", spark_path)


def save_figure(figure: plt.Figure, stem: str) -> None:
    png_path = figure_root / f"{stem}.png"
    pdf_path = figure_root / f"{stem}.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print("Wrote:", png_path)
    print("Wrote:", pdf_path)


def extract_exact_member(
    archive_path: str,
    member_path: str,
    destination_root: Path,
) -> Path:
    """Restartably extract one governed unencrypted CSV from a release ZIP."""
    if not Path(archive_path).exists():
        raise FileNotFoundError(f"Mortality release not found: {archive_path}")
    destination = destination_root / PurePosixPath(member_path).name
    partial = destination.with_suffix(destination.suffix + ".partial")
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo(member_path)
        if info.flag_bits & 0x1:
            raise RuntimeError("Participant-status CSV unexpectedly is encrypted.")
        if destination.exists() and destination.stat().st_size == info.file_size:
            return destination
        if partial.exists():
            partial.unlink()
        with archive.open(info) as source, partial.open("wb") as target:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                target.write(block)
        if partial.stat().st_size != info.file_size:
            raise IOError("Mortality CSV extraction size check failed.")
        partial.replace(destination)
    return destination


def parse_date(series: pd.Series) -> pd.Series:
    """Parse mixed CLSA date strings without silently choosing day-first."""
    values = series.astype("string").str.strip().replace("", pd.NA)
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    return parsed


def normalize_death(series: pd.Series) -> pd.Series:
    """Map only explicit vital-status values; stop on unfamiliar encodings."""
    values = series.astype("string").str.strip().str.lower()
    missing = values.isna() | values.isin({"", "<na>", "nan", "-8", "-9"})
    yes = values.isin({"1", "yes", "y", "true", "dead", "deceased"})
    no = values.isin({"0", "2", "no", "n", "false", "alive", "living"})
    unknown = ~(missing | yes | no)
    if unknown.any():
        counts = values.loc[unknown].value_counts().head(20).to_dict()
        raise ValueError(
            "Unrecognized death encoding. Review the mortality dictionary: "
            f"{counts}"
        )
    result = pd.Series(pd.NA, index=series.index, dtype="Int64")
    result.loc[yes] = 1
    result.loc[no] = 0
    return result


def normalize_sex(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.lower()
    female = values.isin({"2", "f", "female", "woman", "women"})
    male = values.isin({"1", "m", "male", "man", "men"})
    result = pd.Series(np.nan, index=series.index, dtype=float)
    result.loc[male] = 0.0
    result.loc[female] = 1.0
    return result


def first_existing(columns: list[str], candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"None of the candidate columns exists: {candidates}")


def resolve_prediction_path(requested: str) -> tuple[str, str]:
    if requested:
        if not databricks_path_exists(requested):
            raise FileNotFoundError(requested)
        return requested, "explicit locked prediction file"
    age_root = str(Path(embedding_path).parent.parent / "03_age_model")
    candidates = (
        (
            f"{age_root}/retfound_age_predictions_oof.parquet",
            "CLSA out-of-fold retinal-age prediction",
        ),
        (
            f"{age_root}/retinal_age_predictions.parquet",
            "externally trained locked retinal-age prediction",
        ),
    )
    for path, label in candidates:
        if databricks_path_exists(path):
            return path, label
    return "", "not available"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
RETINAL_COLOR = "#087E8B"
EPIGENETIC_COLOR = "#6C4AB6"
CLINICAL_COLOR = "#555555"

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Mortality outcome and censoring
# MAGIC
# MAGIC The event is all-cause death. Time zero is the baseline CLSA date for
# MAGIC epigenetic models and the baseline fundus timing proxy for retinal
# MAGIC models. Participants alive at last contact are censored on their
# MAGIC last-known-status date. Records with death before/on the index date,
# MAGIC missing end date, nonpositive follow-up, or indeterminate vital status
# MAGIC are excluded and counted.

# COMMAND ----------
mortality_csv_path = extract_exact_member(
    mortality_archive_path,
    mortality_member_path,
    extract_root,
)
status = pd.read_csv(mortality_csv_path, dtype="string", low_memory=False)
require_columns(status, ["entity_id", "death", "date_death"], "Mortality release")

censor_date_column = first_existing(
    status.columns.tolist(),
    (
        "clsa_status_last_known_date",
        "clsa_status_date",
        "clsa_last_known",
    ),
)
baseline_date_column = first_existing(
    status.columns.tolist(),
    ("clsa_baseline_date",),
)

status_core = pd.DataFrame(
    {
        "participant_id": status["entity_id"].astype("string").str.strip(),
        "death_event": normalize_death(status["death"]),
        "death_date": parse_date(status["date_death"]),
        "censor_date": parse_date(status[censor_date_column]),
        "baseline_date": parse_date(status[baseline_date_column]),
    }
)
status_core = status_core.loc[
    status_core["participant_id"].str.fullmatch(r"\d{7}", na=False)
].drop_duplicates("participant_id")

status_qc = pd.DataFrame(
    {
        "metric": [
            "participant-status rows",
            "unique valid participant IDs",
            "deaths flagged",
            "deaths with date",
            "alive with censor date",
            "unknown vital status",
        ],
        "n": [
            len(status),
            status_core["participant_id"].nunique(),
            int(status_core["death_event"].eq(1).sum()),
            int(
                (
                    status_core["death_event"].eq(1)
                    & status_core["death_date"].notna()
                ).sum()
            ),
            int(
                (
                    status_core["death_event"].eq(0)
                    & status_core["censor_date"].notna()
                ).sum()
            ),
            int(status_core["death_event"].isna().sum()),
        ],
    }
)
display_pandas(status_qc, "Mortality release QC")

# COMMAND ----------
def attach_survival_outcome(
    predictors: pd.DataFrame,
    index_date_column: str,
    cohort_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = predictors.merge(status_core, on="participant_id", how="inner")
    frame["index_date"] = parse_date(frame[index_date_column])
    frame["event"] = frame["death_event"].astype("Int64")
    frame["end_date"] = frame["censor_date"]
    frame.loc[frame["event"].eq(1), "end_date"] = frame.loc[
        frame["event"].eq(1), "death_date"
    ]
    frame["followup_years"] = (
        (frame["end_date"] - frame["index_date"]).dt.total_seconds()
        / (365.25 * 24 * 60 * 60)
    )

    valid_id = frame["participant_id"].notna()
    known_event = frame["event"].notna()
    known_index = frame["index_date"].notna()
    known_end = frame["end_date"].notna()
    positive_time = frame["followup_years"].gt(0)
    valid = valid_id & known_event & known_index & known_end & positive_time

    flow = pd.DataFrame(
        {
            "cohort": cohort_label,
            "stage": [
                "predictor records before mortality linkage",
                "matched to participant status",
                "known vital status",
                "known index date",
                "known death/censor date",
                "positive prospective follow-up",
                "final analytic cohort",
            ],
            "n": [
                len(predictors),
                len(frame),
                int(known_event.sum()),
                int(known_index.sum()),
                int(known_end.sum()),
                int(positive_time.sum()),
                int(valid.sum()),
            ],
        }
    )
    analytic = frame.loc[valid].copy()
    analytic["event"] = analytic["event"].astype(int)
    analytic["cohort"] = cohort_label

    # Raw dates are not retained in durable analysis outputs.
    analytic = analytic.drop(
        columns=[
            index_date_column,
            "death_event",
            "death_date",
            "censor_date",
            "baseline_date",
            "baseline_date_x",
            "baseline_date_y",
            "end_date",
            "index_date",
        ],
        errors="ignore",
    )
    return analytic, flow


# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Baseline retinal and epigenetic cohorts
# MAGIC
# MAGIC Baseline is the primary landmark because using a later image for people
# MAGIC who survived long enough to attend follow-up would introduce immortal-
# MAGIC time/selection bias. A separate F1 landmark can be added later as a
# MAGIC sensitivity analysis, conditional on being alive at F1.

# COMMAND ----------
if not databricks_path_exists(sap_path):
    raise FileNotFoundError(f"SAP participant-visit table not found: {sap_path}")
if not databricks_path_exists(embedding_path):
    raise FileNotFoundError(f"RETFound embeddings not found: {embedding_path}")

sap_spark = spark.read.format("delta").load(sap_path)
require_columns(
    sap_spark,
    ["participant_id", "visit", "age_at_fundus_years", "sex_at_birth"],
    "SAP participant-visit table",
)

available_epigenetic = [
    column for column in epigenetic_measures if column in sap_spark.columns
]
missing_epigenetic = sorted(set(epigenetic_measures) - set(available_epigenetic))
if missing_epigenetic:
    warnings.warn(f"Unavailable epigenetic measures will be skipped: {missing_epigenetic}")
if primary_epigenetic_measure not in available_epigenetic:
    raise ValueError(
        f"Primary epigenetic measure missing: {primary_epigenetic_measure}"
    )

sap_columns = [
    "participant_id",
    "visit",
    "age_at_fundus_years",
    "sex_at_birth",
    *available_epigenetic,
]
for optional in (
    "fundus_visit_timestamp_proxy",
    "visual_acuity_better_eye",
    "self_reported_vision",
    "smoking_status",
    "education_level_sap_harmonized",
    "diabetes",
    "hypertension",
    "heart_disease",
    "stroke",
    "cancer",
):
    if optional in sap_spark.columns:
        sap_columns.append(optional)

sap_bl = (
    sap_spark.filter(F.upper(F.col("visit")) == "BL")
    .select(*sap_columns)
    .dropDuplicates(["participant_id"])
    .toPandas()
)
sap_bl["participant_id"] = sap_bl["participant_id"].astype("string")
sap_bl["age_at_fundus_years"] = pd.to_numeric(
    sap_bl["age_at_fundus_years"], errors="coerce"
)
sap_bl["sex_female"] = normalize_sex(sap_bl["sex_at_birth"])
for column in available_epigenetic:
    sap_bl[column] = pd.to_numeric(sap_bl[column], errors="coerce")

# Prefer the visit-timing proxy retained by notebook 03, then baseline date.
if "fundus_visit_timestamp_proxy" in sap_bl.columns:
    sap_bl["retinal_index_date"] = parse_date(
        sap_bl["fundus_visit_timestamp_proxy"]
    )
else:
    sap_bl["retinal_index_date"] = pd.NaT
sap_bl = sap_bl.merge(
    status_core[["participant_id", "baseline_date"]],
    on="participant_id",
    how="left",
)
sap_bl["retinal_index_date"] = sap_bl["retinal_index_date"].fillna(
    sap_bl["baseline_date"]
)
sap_bl["epigenetic_index_date"] = sap_bl["baseline_date"]

# Aggregate both eyes/images to one vector per participant before modelling.
embeddings_spark = spark.read.format("delta").load(embedding_path)
require_columns(
    embeddings_spark,
    ["participant_id", "visit", "embedding", "embedding_dim"],
    "RETFound embedding table",
)
embedding_sum_expression = (
    "aggregate(collect_list(embedding), "
    f"array_repeat(CAST(0.0 AS FLOAT), {expected_embedding_dim}), "
    "(acc, x) -> zip_with(acc, x, (a, b) -> CAST(a + b AS FLOAT)))"
)
retinal_vectors = (
    embeddings_spark.filter(
        (F.upper(F.col("visit")) == "BL")
        & F.col("embedding").isNotNull()
        & (F.col("embedding_dim") == expected_embedding_dim)
        & (F.size("embedding") == expected_embedding_dim)
    )
    .groupBy(F.col("participant_id").cast("string").alias("participant_id"))
    .agg(
        F.count("*").alias("n_retinal_images"),
        F.expr(embedding_sum_expression).alias("embedding_sum"),
    )
    .withColumn(
        "embedding",
        F.expr(
            "transform(embedding_sum, x -> CAST(x / n_retinal_images AS FLOAT))"
        ),
    )
    .drop("embedding_sum")
    .toPandas()
)
retinal_vectors["embedding"] = retinal_vectors["embedding"].map(
    lambda value: np.asarray(value, dtype=np.float32)
)

retinal_predictors = retinal_vectors.merge(
    sap_bl.drop(columns=["baseline_date"], errors="ignore"),
    on="participant_id",
    how="inner",
)

# Optional retinal-age-gap comparator. Use only OOF CLSA predictions or an
# externally trained locked age head; never use CLSA in-sample fitted values.
prediction_path, prediction_provenance = resolve_prediction_path(
    retinal_age_prediction_path_requested
)
if prediction_path:
    predictions_spark = spark.read.parquet(prediction_path)
    prediction_column = next(
        (
            column
            for column in (
                "retinal_age_prediction_oof",
                "retinal_age_prediction",
                "retinal_age",
            )
            if column in predictions_spark.columns
        ),
        None,
    )
    if prediction_column is None:
        raise ValueError("Retinal-age prediction column was not recognized.")
    if "participant_id" not in predictions_spark.columns:
        require_columns(predictions_spark, ["image_path"], "Age predictions")
        embedding_keys = embeddings_spark.select(
            "image_path", "participant_id", "visit"
        ).dropDuplicates(["image_path"])
        predictions_spark = predictions_spark.join(
            embedding_keys, "image_path", "left"
        )
    retinal_age = (
        predictions_spark.filter(F.upper(F.col("visit")) == "BL")
        .groupBy(F.col("participant_id").cast("string").alias("participant_id"))
        .agg(
            F.avg(F.col(prediction_column).cast("double")).alias("retinal_age")
        )
        .toPandas()
    )
    retinal_predictors = retinal_predictors.merge(
        retinal_age, on="participant_id", how="left"
    )
    retinal_predictors["retinal_age_gap"] = (
        retinal_predictors["retinal_age"]
        - retinal_predictors["age_at_fundus_years"]
    )
else:
    retinal_predictors["retinal_age_gap"] = np.nan

retinal_cohort, retinal_flow = attach_survival_outcome(
    retinal_predictors,
    "retinal_index_date",
    "Retinal RETFound",
)

epigenetic_predictors = sap_bl.loc[
    sap_bl[available_epigenetic].notna().any(axis=1)
].drop(columns=["baseline_date"], errors="ignore").copy()
epigenetic_cohort, epigenetic_flow = attach_survival_outcome(
    epigenetic_predictors,
    "epigenetic_index_date",
    "Epigenetic clocks",
)

cohort_flow = pd.concat([retinal_flow, epigenetic_flow], ignore_index=True)
display_pandas(cohort_flow, "Cohort flow")

cohort_summary = pd.DataFrame(
    [
        {
            "cohort": "Retinal RETFound",
            "participants": len(retinal_cohort),
            "deaths": int(retinal_cohort["event"].sum()),
            "median_followup_years": retinal_cohort["followup_years"].median(),
            "max_followup_years": retinal_cohort["followup_years"].max(),
        },
        {
            "cohort": "Epigenetic clocks",
            "participants": len(epigenetic_cohort),
            "deaths": int(epigenetic_cohort["event"].sum()),
            "median_followup_years": epigenetic_cohort[
                "followup_years"
            ].median(),
            "max_followup_years": epigenetic_cohort["followup_years"].max(),
        },
    ]
)
display_pandas(cohort_summary.round(3), "Analytic cohort summary")

if retinal_cohort["event"].sum() < minimum_events:
    raise ValueError(
        f"Retinal cohort has fewer than {minimum_events} deaths; do not fit models."
    )
if epigenetic_cohort["event"].sum() < minimum_events:
    raise ValueError(
        f"Epigenetic cohort has fewer than {minimum_events} deaths; "
        "report feasibility only and do not fit multivariable models."
    )

# Store only de-identified modelling quantities; raw dates were dropped above.
retinal_durable = retinal_cohort.drop(
    columns=["embedding", "retinal_index_date", "epigenetic_index_date"],
    errors="ignore",
)
epigenetic_durable = epigenetic_cohort.drop(
    columns=["retinal_index_date", "epigenetic_index_date"], errors="ignore"
)
(
    spark.createDataFrame(retinal_durable.replace({np.nan: None}))
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(str(output_root / "retinal_survival_analysis_dataset"))
)
(
    spark.createDataFrame(epigenetic_durable.replace({np.nan: None}))
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(str(output_root / "epigenetic_survival_analysis_dataset"))
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Survival modelling helpers

# COMMAND ----------
def clinical_design(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    age_train = pd.to_numeric(train["age_at_fundus_years"], errors="coerce")
    age_test = pd.to_numeric(test["age_at_fundus_years"], errors="coerce")
    age_median = float(age_train.median())
    age_train = age_train.fillna(age_median)
    age_test = age_test.fillna(age_median)
    age_mean = float(age_train.mean())
    age_sd = float(age_train.std(ddof=0)) or 1.0
    age_train_z = (age_train.to_numpy() - age_mean) / age_sd
    age_test_z = (age_test.to_numpy() - age_mean) / age_sd

    sex_train = pd.to_numeric(train["sex_female"], errors="coerce")
    sex_test = pd.to_numeric(test["sex_female"], errors="coerce")
    sex_median = float(sex_train.median()) if sex_train.notna().any() else 0.0
    sex_train = sex_train.fillna(sex_median).to_numpy()
    sex_test = sex_test.fillna(sex_median).to_numpy()

    train_matrix = np.column_stack(
        [age_train_z, age_train_z**2, sex_train]
    ).astype(float)
    test_matrix = np.column_stack(
        [age_test_z, age_test_z**2, sex_test]
    ).astype(float)
    return train_matrix, test_matrix, ["age_z", "age_z_squared", "sex_female"]


def scalar_design(
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictor: str,
    include_clinical: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, float]]:
    x_train = pd.to_numeric(train[predictor], errors="coerce")
    x_test = pd.to_numeric(test[predictor], errors="coerce")
    median = float(x_train.median())
    x_train = x_train.fillna(median)
    x_test = x_test.fillna(median)
    mean = float(x_train.mean())
    sd = float(x_train.std(ddof=0)) or 1.0
    train_z = ((x_train.to_numpy() - mean) / sd)[:, None]
    test_z = ((x_test.to_numpy() - mean) / sd)[:, None]
    if include_clinical:
        c_train, c_test, names = clinical_design(train, test)
        train_z = np.column_stack([c_train, train_z])
        test_z = np.column_stack([c_test, test_z])
        names = [*names, f"{predictor}_z"]
    else:
        names = [f"{predictor}_z"]
    return train_z, test_z, names, {"mean": mean, "sd": sd}


def vector_design(
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_components: int,
    include_clinical: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x_train = np.stack(train["embedding"].to_numpy()).astype(np.float64)
    x_test = np.stack(test["embedding"].to_numpy()).astype(np.float64)
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    usable_components = min(
        int(n_components), x_train.shape[0] - 1, x_train.shape[1]
    )
    pca = PCA(
        n_components=usable_components,
        svd_solver="randomized",
        random_state=random_seed,
    ).fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    names = [f"retfound_pc_{index + 1:02d}" for index in range(usable_components)]
    if include_clinical:
        c_train, c_test, clinical_names = clinical_design(train, test)
        x_train = np.column_stack([c_train, x_train])
        x_test = np.column_stack([c_test, x_test])
        names = [*clinical_names, *names]
    return x_train, x_test, names


def fit_cox_and_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    x_train: np.ndarray,
    x_test: np.ndarray,
    names: list[str],
    penalizer: float,
) -> tuple[np.ndarray, dict[float, np.ndarray], CoxPHFitter]:
    train_model = pd.DataFrame(x_train, columns=names, index=train.index)
    test_model = pd.DataFrame(x_test, columns=names, index=test.index)
    train_model["followup_years"] = train["followup_years"].astype(float)
    train_model["event"] = train["event"].astype(int)
    cph = CoxPHFitter(penalizer=float(penalizer))
    cph.fit(
        train_model,
        duration_col="followup_years",
        event_col="event",
        show_progress=False,
    )
    log_risk = cph.predict_log_partial_hazard(test_model).to_numpy(dtype=float)
    survival = {}
    for horizon in evaluation_horizons_years:
        values = cph.predict_survival_function(
            test_model, times=[float(horizon)]
        ).iloc[0]
        survival[float(horizon)] = values.to_numpy(dtype=float)
    return log_risk, survival, cph


def harrell_c(frame: pd.DataFrame, risk_column: str = "log_risk") -> float:
    if frame["event"].sum() == 0:
        return np.nan
    return float(
        concordance_index(
            frame["followup_years"],
            -frame[risk_column],
            frame["event"],
        )
    )


def choose_vector_hyperparameters(
    development: pd.DataFrame,
    include_clinical: bool,
) -> tuple[int, float, pd.DataFrame]:
    splitter = StratifiedKFold(
        n_splits=inner_folds,
        shuffle=True,
        random_state=random_seed + (1 if include_clinical else 0),
    )
    rows = []
    y = development["event"].to_numpy(dtype=int)
    for components in pca_component_grid:
        for penalizer in ridge_penalizer_grid:
            scores = []
            for inner_train, inner_valid in splitter.split(development, y):
                train = development.iloc[inner_train]
                valid = development.iloc[inner_valid]
                x_train, x_valid, names = vector_design(
                    train, valid, components, include_clinical
                )
                try:
                    risk, _, _ = fit_cox_and_predict(
                        train,
                        valid,
                        x_train,
                        x_valid,
                        names,
                        penalizer,
                    )
                    score_frame = valid[["followup_years", "event"]].copy()
                    score_frame["log_risk"] = risk
                    scores.append(harrell_c(score_frame))
                except Exception as error:
                    warnings.warn(
                        f"Inner Cox fit failed for PCs={components}, "
                        f"penalizer={penalizer}: {error}"
                    )
                    scores.append(np.nan)
            rows.append(
                {
                    "n_components": components,
                    "penalizer": penalizer,
                    "include_clinical": include_clinical,
                    "mean_inner_c_index": np.nanmean(scores),
                }
            )
    tuning = pd.DataFrame(rows).sort_values(
        ["mean_inner_c_index", "n_components", "penalizer"],
        ascending=[False, True, False],
    )
    if tuning["mean_inner_c_index"].notna().sum() == 0:
        raise RuntimeError("Every nested RETFound Cox fit failed.")
    best = tuning.iloc[0]
    return int(best["n_components"]), float(best["penalizer"]), tuning


def prediction_rows(
    test: pd.DataFrame,
    risk: np.ndarray,
    survival: dict[float, np.ndarray],
    model: str,
    modality: str,
    fold: int,
    predictor: str = "",
) -> pd.DataFrame:
    output = test[
        ["participant_id", "followup_years", "event"]
    ].reset_index(drop=True)
    output["log_risk"] = risk
    output["model"] = model
    output["modality"] = modality
    output["predictor"] = predictor
    output["fold"] = fold
    for horizon, values in survival.items():
        output[f"predicted_survival_{horizon:g}y"] = values
    return output


# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Nested-CV RETFound models

# COMMAND ----------
retinal_model_frame = retinal_cohort.loc[
    retinal_cohort["age_at_fundus_years"].notna()
    & retinal_cohort["sex_female"].notna()
    & retinal_cohort["embedding"].notna()
].copy()
retinal_model_frame = retinal_model_frame.reset_index(drop=True)

outer_splitter = StratifiedKFold(
    n_splits=outer_folds,
    shuffle=True,
    random_state=random_seed,
)
retinal_oof_parts = []
retinal_tuning_parts = []

for fold, (train_index, test_index) in enumerate(
    outer_splitter.split(retinal_model_frame, retinal_model_frame["event"]),
    start=1,
):
    train = retinal_model_frame.iloc[train_index].copy()
    test = retinal_model_frame.iloc[test_index].copy()

    # Clinical comparator.
    x_train, x_test, names = clinical_design(train, test)
    risk, survival, _ = fit_cox_and_predict(
        train, test, x_train, x_test, names, penalizer=0.01
    )
    retinal_oof_parts.append(
        prediction_rows(test, risk, survival, "Age + sex", "Retinal", fold)
    )

    # Retinal-age gap is included only when a valid OOF/locked prediction exists.
    rag_train = train["retinal_age_gap"].notna()
    rag_test = test["retinal_age_gap"].notna()
    if rag_train.sum() >= 100 and rag_test.sum() >= 20:
        rag_development = train.loc[rag_train]
        rag_evaluation = test.loc[rag_test]
        x_train, x_test, names, _ = scalar_design(
            rag_development,
            rag_evaluation,
            "retinal_age_gap",
            include_clinical=True,
        )
        risk, survival, _ = fit_cox_and_predict(
            rag_development,
            rag_evaluation,
            x_train,
            x_test,
            names,
            penalizer=0.01,
        )
        retinal_oof_parts.append(
            prediction_rows(
                rag_evaluation,
                risk,
                survival,
                "Age + sex + retinal-age gap",
                "Retinal",
                fold,
                "retinal_age_gap",
            )
        )

    for include_clinical, model_name in (
        (False, "RETFound vectors only"),
        (True, "Age + sex + RETFound vectors"),
    ):
        components, penalizer, tuning = choose_vector_hyperparameters(
            train, include_clinical
        )
        tuning["outer_fold"] = fold
        tuning["selected"] = (
            (tuning["n_components"] == components)
            & (tuning["penalizer"] == penalizer)
        )
        retinal_tuning_parts.append(tuning)
        x_train, x_test, names = vector_design(
            train, test, components, include_clinical
        )
        risk, survival, _ = fit_cox_and_predict(
            train,
            test,
            x_train,
            x_test,
            names,
            penalizer,
        )
        rows = prediction_rows(
            test, risk, survival, model_name, "Retinal", fold
        )
        rows["selected_pca_components"] = components
        rows["selected_penalizer"] = penalizer
        retinal_oof_parts.append(rows)

retinal_oof = pd.concat(retinal_oof_parts, ignore_index=True)
retinal_tuning = pd.concat(retinal_tuning_parts, ignore_index=True)
display_pandas(
    retinal_tuning.loc[retinal_tuning["selected"]],
    "Selected RETFound hyperparameters",
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Epigenetic models, one clock at a time
# MAGIC
# MAGIC Each augmented model is evaluated against an age + sex comparator in
# MAGIC exactly the same complete-case subset for that clock. Raw DNAm ages are
# MAGIC secondary; acceleration variables are preferred because they more
# MAGIC directly represent deviation from chronological age.

# COMMAND ----------
epigenetic_oof_parts = []
epigenetic_availability_rows = []

for measure in available_epigenetic:
    measure_frame = epigenetic_cohort.loc[
        epigenetic_cohort[measure].notna()
        & epigenetic_cohort["age_at_fundus_years"].notna()
        & epigenetic_cohort["sex_female"].notna()
    ].copy()
    deaths = int(measure_frame["event"].sum())
    epigenetic_availability_rows.append(
        {
            "measure": measure,
            "label": epigenetic_measures[measure],
            "participants": len(measure_frame),
            "deaths": deaths,
            "eligible_for_model": deaths >= minimum_events,
        }
    )
    if deaths < minimum_events:
        warnings.warn(
            f"Skipping {measure}: only {deaths} deaths (< {minimum_events})."
        )
        continue
    measure_frame = measure_frame.reset_index(drop=True)
    splitter = StratifiedKFold(
        n_splits=outer_folds,
        shuffle=True,
        random_state=random_seed,
    )
    for fold, (train_index, test_index) in enumerate(
        splitter.split(measure_frame, measure_frame["event"]), start=1
    ):
        train = measure_frame.iloc[train_index]
        test = measure_frame.iloc[test_index]

        x_train, x_test, names = clinical_design(train, test)
        risk, survival, _ = fit_cox_and_predict(
            train, test, x_train, x_test, names, penalizer=0.01
        )
        epigenetic_oof_parts.append(
            prediction_rows(
                test,
                risk,
                survival,
                "Age + sex",
                "Epigenetic",
                fold,
                measure,
            )
        )

        x_train, x_test, names, _ = scalar_design(
            train, test, measure, include_clinical=True
        )
        risk, survival, _ = fit_cox_and_predict(
            train, test, x_train, x_test, names, penalizer=0.01
        )
        epigenetic_oof_parts.append(
            prediction_rows(
                test,
                risk,
                survival,
                f"Age + sex + {epigenetic_measures[measure]}",
                "Epigenetic",
                fold,
                measure,
            )
        )

epigenetic_availability = pd.DataFrame(epigenetic_availability_rows)
epigenetic_oof = (
    pd.concat(epigenetic_oof_parts, ignore_index=True)
    if epigenetic_oof_parts
    else pd.DataFrame()
)
if epigenetic_oof.empty:
    raise ValueError(
        "No epigenetic measure has enough deaths for internally validated "
        "mortality modelling. Report the feasibility counts and stop here."
    )
display_pandas(epigenetic_availability, "Epigenetic outcome availability")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Discrimination, incremental value, and calibration
# MAGIC
# MAGIC Harrell's C-index is calculated from held-out predictions only.
# MAGIC Confidence intervals use participant bootstrap of the completed OOF
# MAGIC predictions. Incremental C-index uses paired bootstrap within the same
# MAGIC participants and measure-specific cohort.

# COMMAND ----------
def bootstrap_c_index(
    frame: pd.DataFrame,
    repetitions: int = bootstrap_repetitions,
) -> tuple[float, float, float]:
    point = harrell_c(frame)
    rng = np.random.default_rng(random_seed)
    values = []
    n = len(frame)
    for _ in range(repetitions):
        sample = frame.iloc[rng.integers(0, n, n)]
        if sample["event"].sum() == 0:
            continue
        try:
            values.append(harrell_c(sample))
        except ZeroDivisionError:
            continue
    if not values:
        return point, np.nan, np.nan
    return point, *np.quantile(values, [0.025, 0.975])


def summarize_discrimination(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["modality", "predictor", "model"]
    for key, group in oof.groupby(keys, dropna=False):
        point, lower, upper = bootstrap_c_index(group)
        rows.append(
            {
                "modality": key[0],
                "predictor": key[1],
                "model": key[2],
                "participants": len(group),
                "deaths": int(group["event"].sum()),
                "c_index": point,
                "c_index_lower_95": lower,
                "c_index_upper_95": upper,
            }
        )
    return pd.DataFrame(rows)


def paired_delta_c_index(
    augmented: pd.DataFrame,
    baseline: pd.DataFrame,
) -> tuple[float, float, float]:
    merge_columns = ["participant_id", "followup_years", "event"]
    paired = augmented[merge_columns + ["log_risk"]].merge(
        baseline[merge_columns + ["log_risk"]],
        on=merge_columns,
        suffixes=("_augmented", "_baseline"),
    )
    if paired.empty:
        return np.nan, np.nan, np.nan
    point = harrell_c(
        paired.rename(columns={"log_risk_augmented": "log_risk"})
    ) - harrell_c(
        paired.rename(columns={"log_risk_baseline": "log_risk"})
    )
    rng = np.random.default_rng(random_seed + 17)
    values = []
    for _ in range(bootstrap_repetitions):
        sample = paired.iloc[rng.integers(0, len(paired), len(paired))]
        if sample["event"].sum() == 0:
            continue
        augmented_c = harrell_c(
            sample.rename(columns={"log_risk_augmented": "log_risk"})
        )
        baseline_c = harrell_c(
            sample.rename(columns={"log_risk_baseline": "log_risk"})
        )
        values.append(augmented_c - baseline_c)
    return point, *np.quantile(values, [0.025, 0.975])


def summarize_incremental_value(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (modality, predictor), group in oof.groupby(
        ["modality", "predictor"], dropna=False
    ):
        baseline = group.loc[group["model"] == "Age + sex"]
        if baseline.empty and modality == "Retinal":
            baseline = oof.loc[
                (oof["modality"] == "Retinal")
                & (oof["model"] == "Age + sex")
            ]
        for model, augmented in group.groupby("model"):
            if model == "Age + sex":
                continue
            point, lower, upper = paired_delta_c_index(augmented, baseline)
            rows.append(
                {
                    "modality": modality,
                    "predictor": predictor,
                    "model": model,
                    "delta_c_index": point,
                    "delta_c_lower_95": lower,
                    "delta_c_upper_95": upper,
                }
            )
    return pd.DataFrame(rows)


all_oof = pd.concat([retinal_oof, epigenetic_oof], ignore_index=True)
discrimination = summarize_discrimination(all_oof)
incremental_value = summarize_incremental_value(all_oof)
display_pandas(discrimination.round(4), "OOF discrimination")
display_pandas(incremental_value.round(4), "Paired incremental discrimination")

# COMMAND ----------
def observed_risk_km(
    frame: pd.DataFrame,
    horizon: float,
) -> tuple[float, float, float]:
    km = KaplanMeierFitter().fit(
        frame["followup_years"], event_observed=frame["event"]
    )
    survival = float(km.predict(horizon))
    timeline = km.confidence_interval_survival_function_.index.to_numpy()
    position = int(np.searchsorted(timeline, horizon, side="right") - 1)
    position = max(0, min(position, len(timeline) - 1))
    ci = km.confidence_interval_survival_function_.iloc[position]
    return 1.0 - survival, 1.0 - float(ci.iloc[1]), 1.0 - float(ci.iloc[0])


def calibration_table(oof: pd.DataFrame, groups: int = 5) -> pd.DataFrame:
    rows = []
    for (modality, predictor, model), frame in oof.groupby(
        ["modality", "predictor", "model"], dropna=False
    ):
        for horizon in evaluation_horizons_years:
            survival_column = f"predicted_survival_{horizon:g}y"
            if survival_column not in frame.columns:
                continue
            usable = frame.loc[frame[survival_column].notna()].copy()
            if usable.empty or usable["followup_years"].max() < horizon:
                continue
            usable["predicted_risk"] = 1.0 - usable[survival_column]
            try:
                usable["risk_group"] = pd.qcut(
                    usable["predicted_risk"],
                    q=min(groups, usable["predicted_risk"].nunique()),
                    labels=False,
                    duplicates="drop",
                )
            except ValueError:
                continue
            for risk_group, subset in usable.groupby("risk_group"):
                observed, lower, upper = observed_risk_km(subset, horizon)
                rows.append(
                    {
                        "modality": modality,
                        "predictor": predictor,
                        "model": model,
                        "horizon_years": horizon,
                        "risk_group": int(risk_group) + 1,
                        "participants": len(subset),
                        "deaths": int(subset["event"].sum()),
                        "mean_predicted_risk": subset["predicted_risk"].mean(),
                        "observed_km_risk": observed,
                        "observed_lower_95": lower,
                        "observed_upper_95": upper,
                    }
                )
    return pd.DataFrame(rows)


calibration = calibration_table(all_oof)
display_pandas(calibration.round(4), "OOF calibration by risk quintile")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Adjusted hazard ratios and proportional-hazards checks

# COMMAND ----------
def adjusted_scalar_cox(
    frame: pd.DataFrame,
    predictor: str,
    label: str,
    modality: str,
    landmark_years: float = 0.0,
) -> tuple[dict[str, object], pd.DataFrame]:
    model_frame = frame.loc[
        frame[predictor].notna()
        & frame["age_at_fundus_years"].notna()
        & frame["sex_female"].notna()
    ].copy()
    if landmark_years > 0:
        model_frame = model_frame.loc[
            model_frame["followup_years"] > landmark_years
        ].copy()
        model_frame["followup_years"] -= landmark_years
    x_train, _, names, scaling = scalar_design(
        model_frame, model_frame, predictor, include_clinical=True
    )
    design = pd.DataFrame(x_train, columns=names, index=model_frame.index)
    design["followup_years"] = model_frame["followup_years"].astype(float)
    design["event"] = model_frame["event"].astype(int)
    cph = CoxPHFitter(penalizer=0.001).fit(
        design,
        duration_col="followup_years",
        event_col="event",
        robust=True,
    )
    term = f"{predictor}_z"
    coefficient = float(cph.params_[term])
    se = float(cph.standard_errors_[term])
    z = coefficient / se
    p_value = float(math.erfc(abs(z) / math.sqrt(2.0)))
    native_5y_factor = 5.0 / scaling["sd"]
    result = {
        "modality": modality,
        "predictor": predictor,
        "label": label,
        "landmark_years": landmark_years,
        "participants": len(model_frame),
        "deaths": int(model_frame["event"].sum()),
        "predictor_sd_years": scaling["sd"],
        "hr_per_sd": math.exp(coefficient),
        "hr_per_sd_lower_95": math.exp(coefficient - 1.96 * se),
        "hr_per_sd_upper_95": math.exp(coefficient + 1.96 * se),
        "hr_per_5_years": math.exp(coefficient * native_5y_factor),
        "hr_per_5_years_lower_95": math.exp(
            (coefficient - 1.96 * se) * native_5y_factor
        ),
        "hr_per_5_years_upper_95": math.exp(
            (coefficient + 1.96 * se) * native_5y_factor
        ),
        "p_value": p_value,
    }
    ph = proportional_hazard_test(
        cph, design, time_transform="rank"
    ).summary.reset_index(names="term")
    ph["modality"] = modality
    ph["predictor"] = predictor
    ph["landmark_years"] = landmark_years
    return result, ph


association_rows = []
ph_parts = []
if retinal_cohort["retinal_age_gap"].notna().sum() >= 100:
    for landmark in (0.0, early_death_landmark_years):
        row, ph = adjusted_scalar_cox(
            retinal_cohort,
            "retinal_age_gap",
            "Retinal-age gap",
            "Retinal",
            landmark,
        )
        association_rows.append(row)
        ph_parts.append(ph)

for measure in available_epigenetic:
    eligible_deaths = epigenetic_cohort.loc[
        epigenetic_cohort[measure].notna(), "event"
    ].sum()
    if eligible_deaths < minimum_events:
        continue
    for landmark in (0.0, early_death_landmark_years):
        row, ph = adjusted_scalar_cox(
            epigenetic_cohort,
            measure,
            epigenetic_measures[measure],
            "Epigenetic",
            landmark,
        )
        association_rows.append(row)
        ph_parts.append(ph)

hazard_ratios = pd.DataFrame(association_rows)
ph_diagnostics = pd.concat(ph_parts, ignore_index=True) if ph_parts else pd.DataFrame()
display_pandas(hazard_ratios.round(4), "Adjusted all-cause mortality HRs")
display_pandas(ph_diagnostics.round(4), "Schoenfeld PH diagnostics")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Publication figures
# MAGIC
# MAGIC Main-paper candidates:
# MAGIC
# MAGIC - Figure 1: parallel cohort flow and mortality follow-up summary.
# MAGIC - Figure 2: adjusted HR forest plot (per SD; retinal and epigenetic kept
# MAGIC   visually distinct).
# MAGIC - Figure 3: OOF C-index and paired incremental C-index.
# MAGIC - Figure 4: Kaplan-Meier curves for RETFound OOF risk and primary
# MAGIC   epigenetic acceleration quartiles.
# MAGIC - Figure 5: 5- and 10-year OOF calibration.
# MAGIC
# MAGIC PH diagnostics, early-death landmark results, every alternative clock,
# MAGIC tuning stability, and complete numeric tables belong in the supplement.

# COMMAND ----------
def plot_cohort_flow(flow: pd.DataFrame, summary: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(13, 6.2), layout="constrained")
    for axis, cohort, color in zip(
        axes,
        ["Retinal RETFound", "Epigenetic clocks"],
        [RETINAL_COLOR, EPIGENETIC_COLOR],
    ):
        subset = flow.loc[flow["cohort"] == cohort].reset_index(drop=True)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, len(subset) + 1)
        axis.axis("off")
        axis.set_title(cohort, color=color, fontweight="bold", pad=12)
        for index, row in subset.iterrows():
            y = len(subset) - index
            box = FancyBboxPatch(
                (0.08, y - 0.34),
                0.84,
                0.55,
                boxstyle="round,pad=0.02,rounding_size=0.03",
                linewidth=1.2,
                edgecolor=color,
                facecolor="white",
            )
            axis.add_patch(box)
            axis.text(
                0.5,
                y - 0.065,
                f"{row['stage']}\n$n$ = {int(row['n']):,}",
                ha="center",
                va="center",
                fontsize=9,
            )
            if index < len(subset) - 1:
                axis.annotate(
                    "",
                    xy=(0.5, y - 0.8),
                    xytext=(0.5, y - 0.36),
                    arrowprops={"arrowstyle": "->", "color": "#777777"},
                )
        summary_row = summary.loc[summary["cohort"] == cohort].iloc[0]
        axis.text(
            0.5,
            0.25,
            f"Deaths: {int(summary_row['deaths']):,} | "
            f"median follow-up: {summary_row['median_followup_years']:.1f} y",
            ha="center",
            va="center",
            color=color,
            fontweight="bold",
        )
    figure.suptitle(
        "Baseline mortality cohorts and outcome completeness",
        fontsize=15,
        fontweight="bold",
    )
    return figure


figure_1 = plot_cohort_flow(cohort_flow, cohort_summary)
save_figure(figure_1, "figure_1_mortality_cohort_flow")
plt.show()

# COMMAND ----------
def plot_hazard_ratio_forest(results: pd.DataFrame) -> plt.Figure:
    data = results.loc[results["landmark_years"] == 0].copy()
    data = data.sort_values(["modality", "hr_per_sd"])
    positions = np.arange(len(data))
    figure, axis = plt.subplots(figsize=(10.5, max(4.8, 0.55 * len(data))))
    colors = data["modality"].map(
        {"Retinal": RETINAL_COLOR, "Epigenetic": EPIGENETIC_COLOR}
    )
    for position, (_, row), color in zip(positions, data.iterrows(), colors):
        axis.errorbar(
            row["hr_per_sd"],
            position,
            xerr=[
                [row["hr_per_sd"] - row["hr_per_sd_lower_95"]],
                [row["hr_per_sd_upper_95"] - row["hr_per_sd"]],
            ],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=6,
        )
        axis.text(
            max(data["hr_per_sd_upper_95"].max() * 1.03, 1.15),
            position,
            f"{row['hr_per_sd']:.2f} "
            f"({row['hr_per_sd_lower_95']:.2f}-{row['hr_per_sd_upper_95']:.2f})",
            va="center",
            fontsize=9,
        )
    axis.axvline(1.0, color="#333333", linewidth=1, linestyle="--")
    axis.set_yticks(positions, data["label"])
    axis.set_xlabel("Hazard ratio per 1-SD higher age measure (95% CI)")
    axis.set_title(
        "All-cause mortality associations adjusted for age and sex",
        fontweight="bold",
    )
    axis.grid(axis="x", color="#E5E5E5", linewidth=0.8)
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=RETINAL_COLOR,
               markeredgecolor=RETINAL_COLOR, label="Retinal"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=EPIGENETIC_COLOR,
               markeredgecolor=EPIGENETIC_COLOR, label="Epigenetic"),
    ]
    axis.legend(handles=legend, loc="lower right", frameon=False)
    figure.subplots_adjust(left=0.35, right=0.78)
    return figure


figure_2 = plot_hazard_ratio_forest(hazard_ratios)
save_figure(figure_2, "figure_2_adjusted_hazard_ratios")
plt.show()

# COMMAND ----------
def performance_label(row: pd.Series) -> str:
    if row["modality"] == "Epigenetic" and row["model"] != "Age + sex":
        return epigenetic_measures.get(row["predictor"], row["predictor"])
    return row["model"]


def plot_discrimination(metrics: pd.DataFrame) -> plt.Figure:
    retinal = metrics.loc[metrics["modality"] == "Retinal"].copy()
    # De-duplicate identical retinal clinical rows created by predictor grouping.
    retinal = retinal.drop_duplicates(["model", "participants", "deaths"])
    epigenetic = metrics.loc[
        (metrics["modality"] == "Epigenetic")
        & (metrics["model"] != "Age + sex")
    ].copy()
    panels = [
        (retinal, "Retinal models", RETINAL_COLOR),
        (epigenetic, "Epigenetic models", EPIGENETIC_COLOR),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.8), layout="constrained")
    for axis, data, title, color in zip(axes, *zip(*panels)):
        data = data.sort_values("c_index").reset_index(drop=True)
        y = np.arange(len(data))
        axis.errorbar(
            data["c_index"],
            y,
            xerr=np.vstack(
                [
                    data["c_index"] - data["c_index_lower_95"],
                    data["c_index_upper_95"] - data["c_index"],
                ]
            ),
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
        )
        axis.axvline(0.5, color="#777777", linestyle="--", linewidth=1)
        axis.set_yticks(y, [performance_label(row) for _, row in data.iterrows()])
        axis.set_xlabel("Out-of-fold Harrell C-index (95% CI)")
        axis.set_title(title, color=color, fontweight="bold")
        axis.grid(axis="x", color="#E5E5E5")
        lower = min(0.48, float(data["c_index_lower_95"].min()) - 0.02)
        upper = min(1.0, float(data["c_index_upper_95"].max()) + 0.03)
        axis.set_xlim(lower, upper)
    figure.suptitle(
        "Internal validation of mortality discrimination",
        fontsize=15,
        fontweight="bold",
    )
    return figure


figure_3 = plot_discrimination(discrimination)
save_figure(figure_3, "figure_3_oof_discrimination")
plt.show()

# COMMAND ----------
def add_km_quartile_panel(
    axis: plt.Axes,
    frame: pd.DataFrame,
    score: pd.Series,
    title: str,
    palette: list[str],
) -> None:
    data = frame[["followup_years", "event"]].copy()
    data["score"] = score.to_numpy()
    data = data.dropna()
    data["quartile"] = pd.qcut(
        data["score"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
    )
    fitters = []
    for color, (quartile, subset) in zip(palette, data.groupby("quartile", observed=True)):
        km = KaplanMeierFitter(label=str(quartile)).fit(
            subset["followup_years"], subset["event"]
        )
        fitters.append(km)
        km.plot_survival_function(
            ax=axis, ci_show=False, color=color, linewidth=2
        )
    test = multivariate_logrank_test(
        data["followup_years"], data["quartile"], data["event"]
    )
    axis.set_title(f"{title}\nlog-rank p = {test.p_value:.2g}", fontweight="bold")
    axis.set_xlabel("Years from baseline")
    axis.set_ylabel("Overall survival probability")
    axis.set_ylim(0.5, 1.01)
    axis.grid(color="#EEEEEE", linewidth=0.8)
    axis.legend(title="Risk quartile", frameon=False)
    add_at_risk_counts(*fitters, ax=axis, rows_to_show=["At risk"])


retinal_km_model = "Age + sex + RETFound vectors"
retinal_km = retinal_oof.loc[retinal_oof["model"] == retinal_km_model].copy()
epi_km = epigenetic_cohort.loc[
    epigenetic_cohort[primary_epigenetic_measure].notna()
].copy()

figure_4, axes = plt.subplots(1, 2, figsize=(14, 7.2), layout="constrained")
add_km_quartile_panel(
    axes[0],
    retinal_km,
    retinal_km["log_risk"],
    "RETFound OOF mortality risk",
    ["#BFE8E2", "#75C7BD", "#2FA39A", RETINAL_COLOR],
)
add_km_quartile_panel(
    axes[1],
    epi_km,
    epi_km[primary_epigenetic_measure],
    "Horvath residual acceleration",
    ["#DDD2F4", "#B99EE4", "#9270CF", EPIGENETIC_COLOR],
)
figure_4.suptitle(
    "Prospective all-cause mortality by baseline risk quartile",
    fontsize=15,
    fontweight="bold",
)
save_figure(figure_4, "figure_4_kaplan_meier_quartiles")
plt.show()

# COMMAND ----------
def select_calibration_models(table: pd.DataFrame) -> pd.DataFrame:
    retinal_selected = table.loc[
        (table["modality"] == "Retinal")
        & (table["model"] == "Age + sex + RETFound vectors")
    ]
    epi_model = f"Age + sex + {epigenetic_measures[primary_epigenetic_measure]}"
    epigenetic_selected = table.loc[
        (table["modality"] == "Epigenetic")
        & (table["predictor"] == primary_epigenetic_measure)
        & (table["model"] == epi_model)
    ]
    return pd.concat([retinal_selected, epigenetic_selected], ignore_index=True)


def plot_calibration(table: pd.DataFrame) -> plt.Figure:
    selected = select_calibration_models(table)
    figure, axes = plt.subplots(
        2,
        len(evaluation_horizons_years),
        figsize=(11.5, 9),
        layout="constrained",
        squeeze=False,
    )
    modalities = ["Retinal", "Epigenetic"]
    for row_index, modality in enumerate(modalities):
        color = RETINAL_COLOR if modality == "Retinal" else EPIGENETIC_COLOR
        for column_index, horizon in enumerate(evaluation_horizons_years):
            axis = axes[row_index, column_index]
            data = selected.loc[
                (selected["modality"] == modality)
                & (selected["horizon_years"] == horizon)
            ]
            axis.plot([0, 1], [0, 1], "--", color="#777777", linewidth=1)
            if not data.empty:
                axis.errorbar(
                    data["mean_predicted_risk"],
                    data["observed_km_risk"],
                    yerr=np.vstack(
                        [
                            data["observed_km_risk"] - data["observed_lower_95"],
                            data["observed_upper_95"] - data["observed_km_risk"],
                        ]
                    ),
                    fmt="o-",
                    color=color,
                    ecolor=color,
                    capsize=3,
                )
                maximum = max(
                    data["mean_predicted_risk"].max(),
                    data["observed_upper_95"].max(),
                )
                axis.set_xlim(0, min(1.0, maximum * 1.25 + 0.01))
                axis.set_ylim(0, min(1.0, maximum * 1.25 + 0.01))
            axis.set_title(f"{modality}: {horizon:g}-year risk", color=color)
            axis.set_xlabel("Mean OOF predicted risk")
            axis.set_ylabel("Kaplan-Meier observed risk")
            axis.grid(color="#EEEEEE")
            axis.set_aspect("equal", adjustable="box")
    figure.suptitle(
        "Mortality calibration by predicted-risk quintile",
        fontsize=15,
        fontweight="bold",
    )
    return figure


figure_5 = plot_calibration(calibration)
save_figure(figure_5, "figure_5_oof_calibration")
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. Durable outputs and interpretation guardrails
# MAGIC
# MAGIC - Interpret HRs as prospective associations, not causal effects.
# MAGIC - The vector model is internally validated within CLSA; external
# MAGIC   validation is still required before clinical claims.
# MAGIC - Prefer paired delta C-index over comparing models evaluated in
# MAGIC   different complete-case samples.
# MAGIC - If PH tests are significant, report time-varying effects or stratified
# MAGIC   estimates rather than a single constant HR.
# MAGIC - The 2-year landmark results address, but do not eliminate, reverse
# MAGIC   causation from terminal illness.
# MAGIC - Retinal-age-gap results are valid only when prediction provenance is
# MAGIC   OOF within CLSA or locked external. The notebook records provenance.
# MAGIC - Report epigenetic subset selection/transportability because methylation
# MAGIC   is available in only a subset of CLSA.

# COMMAND ----------
write_pandas_output(status_qc, "status_qc")
write_pandas_output(cohort_flow, "cohort_flow")
write_pandas_output(cohort_summary, "cohort_summary")
write_pandas_output(epigenetic_availability, "epigenetic_availability")
write_pandas_output(retinal_tuning, "retfound_nested_cv_tuning")
write_pandas_output(retinal_oof, "retinal_oof_predictions")
write_pandas_output(epigenetic_oof, "epigenetic_oof_predictions")
write_pandas_output(discrimination, "oof_discrimination")
write_pandas_output(incremental_value, "paired_incremental_c_index")
write_pandas_output(calibration, "oof_calibration")
write_pandas_output(hazard_ratios, "adjusted_hazard_ratios")
write_pandas_output(ph_diagnostics, "proportional_hazards_diagnostics")

run_metadata = {
    "analysis": "all-cause mortality prediction",
    "status_release": mortality_archive_path,
    "status_member": mortality_member_path,
    "censor_date_column": censor_date_column,
    "sap_path": sap_path,
    "embedding_path": embedding_path,
    "retinal_age_prediction_path": prediction_path,
    "retinal_age_prediction_provenance": prediction_provenance,
    "primary_landmark": "baseline",
    "retinal_unit": "participant; mean of all quality-passing baseline images",
    "epigenetic_primary_measure": primary_epigenetic_measure,
    "epigenetic_measures_tested": available_epigenetic,
    "outer_folds": outer_folds,
    "inner_folds": inner_folds,
    "pca_component_grid": list(pca_component_grid),
    "ridge_penalizer_grid": list(ridge_penalizer_grid),
    "evaluation_horizons_years": list(evaluation_horizons_years),
    "minimum_events": minimum_events,
    "bootstrap_repetitions": bootstrap_repetitions,
    "early_death_landmark_years": early_death_landmark_years,
    "random_seed": random_seed,
    "raw_dates_persisted": False,
    "external_validation": False,
}
metadata_path = output_root / "mortality_analysis_metadata.json"
metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
print("Run metadata:", metadata_path)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Recommended manuscript placement
# MAGIC
# MAGIC **Main text:** cohort flow; compact standardized-HR forest plot;
# MAGIC retinal and epigenetic OOF discrimination with delta C-index; primary
# MAGIC KM/calibration panels.
# MAGIC
# MAGIC **Supplement:** full measure-specific tables, 5-year native-unit HRs,
# MAGIC 2-year landmark results, Schoenfeld tests, nested-CV tuning selections,
# MAGIC alternative clocks, and missingness/eligibility details.
