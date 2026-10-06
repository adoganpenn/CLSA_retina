# Databricks notebook source
# ruff: noqa: E402,F821
# MAGIC %md
# MAGIC # Baseline CLSA cataract classification from RETFound embeddings
# MAGIC ## and cataract-definition sensitivity for retinal-age acceleration
# MAGIC
# MAGIC This standalone Databricks script performs two analyses:
# MAGIC
# MAGIC 1. It trains an embedding-only binary classifier for cataract status using
# MAGIC    **only the baseline `VIS_CATRCT_COM` field as the outcome**. The unit of
# MAGIC    analysis and splitting is the participant. A stratified 70% development
# MAGIC    / 30% locked-validation split is created before model selection.
# MAGIC 2. It repeats the questionnaire/retinal-age-acceleration scan under three
# MAGIC    cataract adjustment definitions: the current combined ICQ-or-VIS
# MAGIC    construct, `ICQ_CATRCT_COM` alone, and `VIS_CATRCT_COM` alone.
# MAGIC
# MAGIC The classifier is a research model for a CLSA vision-testing indicator.
# MAGIC `VIS_CATRCT_COM` is participant-level and is not an eye-specific,
# MAGIC ophthalmologist-adjudicated cataract grade. This script must not be used
# MAGIC as a clinical diagnostic device.

# COMMAND ----------
# MAGIC %pip install -q "scikit-learn>=1.4,<2" "statsmodels>=0.14,<1" "pyarrow>=14,<22" "joblib>=1.3,<2" "seaborn>=0.13,<1"

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
from pathlib import Path
import json
import math
import re
import zipfile

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 120)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC Defaults point to the completed CLSA retinal-aging outputs. Paths and
# MAGIC analysis constants can be changed with Databricks widgets without editing
# MAGIC the source. Keep `train_fraction=0.70` for the prespecified analysis.

# COMMAND ----------
DEFAULT_DATASET_ROOT = (
    "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset"
)
DEFAULT_DERIVED_ROOT = f"{DEFAULT_DATASET_ROOT}/derived/clsa_retinal_aging"
DEFAULT_EPIGENETICS_ROOT = (
    f"{DEFAULT_DERIVED_ROOT}/Age_Glaucoma/16_algorithm_fairness/08_epigenetics"
)

dbutils.widgets.text(
    "participant_embedding_path",
    (
        f"{DEFAULT_DERIVED_ROOT}/Age_Glaucoma/16_algorithm_fairness/"
        "01_private/participant_visit_embeddings.parquet"
    ),
)
dbutils.widgets.text(
    "baseline_questionnaire_archive_path",
    f"{DEFAULT_DATASET_ROOT}/2209017_UOttawa_EFreeman_BL.zip",
)
dbutils.widgets.text(
    "baseline_questionnaire_member_suffix",
    "CoPv7_Qx_CANUE_PA_BS.csv",
)
dbutils.widgets.text(
    "three_clock_phenotype_path",
    f"{DEFAULT_EPIGENETICS_ROOT}/01_private/patient_three_clock_phenotypes_private.parquet",
)
dbutils.widgets.text(
    "questionnaire_answers_path",
    f"{DEFAULT_EPIGENETICS_ROOT}/01_private/all_baseline_questionnaire_answers_private.parquet",
)
dbutils.widgets.text(
    "questionnaire_audit_path",
    f"{DEFAULT_EPIGENETICS_ROOT}/03_statistics/questionnaire_variable_audit.csv",
)
dbutils.widgets.text(
    "output_root",
    f"{DEFAULT_DERIVED_ROOT}/Age_Glaucoma/17_retfound_cataract_classifier",
)
dbutils.widgets.text("random_state", "20261006")
dbutils.widgets.text("bootstrap_repetitions", "2000")
dbutils.widgets.text("inner_cv_folds", "5")
dbutils.widgets.text("expected_embedding_dim", "1024")
dbutils.widgets.text("minimum_questionnaire_observed", "100")
dbutils.widgets.text("minimum_questionnaire_level_n", "30")
dbutils.widgets.text("minimum_questionnaire_manuscript_n", "300")

participant_embedding_path = Path(
    dbutils.widgets.get("participant_embedding_path").strip()
)
baseline_questionnaire_archive_path = Path(
    dbutils.widgets.get("baseline_questionnaire_archive_path").strip()
)
baseline_questionnaire_member_suffix = dbutils.widgets.get(
    "baseline_questionnaire_member_suffix"
).strip()
three_clock_phenotype_path = Path(
    dbutils.widgets.get("three_clock_phenotype_path").strip()
)
questionnaire_answers_path = Path(
    dbutils.widgets.get("questionnaire_answers_path").strip()
)
questionnaire_audit_path = Path(
    dbutils.widgets.get("questionnaire_audit_path").strip()
)
output_root = Path(dbutils.widgets.get("output_root").strip())

random_state = int(dbutils.widgets.get("random_state"))
bootstrap_repetitions = int(dbutils.widgets.get("bootstrap_repetitions"))
inner_cv_folds = int(dbutils.widgets.get("inner_cv_folds"))
expected_embedding_dim = int(dbutils.widgets.get("expected_embedding_dim"))
minimum_questionnaire_observed = int(
    dbutils.widgets.get("minimum_questionnaire_observed")
)
minimum_questionnaire_level_n = int(
    dbutils.widgets.get("minimum_questionnaire_level_n")
)
minimum_questionnaire_manuscript_n = int(
    dbutils.widgets.get("minimum_questionnaire_manuscript_n")
)

train_fraction = 0.70
validation_fraction = 0.30
VIS_LABEL_COLUMN = "VIS_CATRCT_COM"
ICQ_HISTORY_COLUMN = "ICQ_CATRCT_COM"
C_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)

if not math.isclose(train_fraction + validation_fraction, 1.0):
    raise ValueError("Training and validation fractions must sum to one")
if not math.isclose(train_fraction, 0.70):
    raise ValueError("The prespecified development fraction must remain 0.70")
if bootstrap_repetitions < 500:
    raise ValueError("Use at least 500 bootstrap repetitions")
if inner_cv_folds < 3:
    raise ValueError("Use at least three inner cross-validation folds")
if expected_embedding_dim < 1:
    raise ValueError("expected_embedding_dim must be positive")

classifier_root = output_root / "01_classifier"
questionnaire_root = output_root / "02_questionnaire_sensitivity"
figure_root = output_root / "03_figures"
private_root = output_root / "04_private"
for directory in (
    classifier_root,
    questionnaire_root,
    figure_root,
    private_root,
):
    directory.mkdir(parents=True, exist_ok=True)

required_paths = {
    "participant RETFound embeddings": participant_embedding_path,
    "baseline questionnaire archive": baseline_questionnaire_archive_path,
    "three-clock phenotype table": three_clock_phenotype_path,
    "questionnaire answer table": questionnaire_answers_path,
    "questionnaire variable audit": questionnaire_audit_path,
}
missing_paths = [
    f"{label}: {path}"
    for label, path in required_paths.items()
    if not path.exists()
]
if missing_paths:
    raise FileNotFoundError(
        "Required inputs are missing:\n- " + "\n- ".join(missing_paths)
    )

print("Output root:", output_root)
print("Classifier label (exact field):", VIS_LABEL_COLUMN)
print("Development/locked-validation split: 70%/30%")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Shared helpers and strict CLSA response coding

# COMMAND ----------
CLSA_MISSING_CODES = {
    "",
    "-8",
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
# CLSA single-response yes/no questionnaire items use 8 (or 08) for
# "Don't know/No answer" and 9 (or 09) for "Refused". Keep these codes
# specific to binary coercion: 8 and 9 can be valid response levels for
# other, nonbinary questionnaire variables analyzed later in this script.
CLSA_BINARY_MISSING_CODES = CLSA_MISSING_CODES | {"8", "08", "9", "09"}
BINARY_CODE_MAP = {
    "1": 1.0,
    "Y": 1.0,
    "YES": 1.0,
    "TRUE": 1.0,
    "T": 1.0,
    "0": 0.0,
    "2": 0.0,
    "N": 0.0,
    "NO": 0.0,
    "FALSE": 0.0,
    "F": 0.0,
}


def normalize_identifier(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def normalize_visit(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .replace({"FUP1": "F1", "BASELINE": "BL"})
    )


def normalized_code(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .str.replace(r"\.0$", "", regex=True)
    )


def coerce_binary_strict(
    series: pd.Series,
    *,
    variable: str,
    fail_on_unknown: bool = True,
) -> pd.Series:
    """Map documented yes/no encodings while preserving missing responses."""
    codes = normalized_code(series)
    missing = codes.isna() | codes.isin(CLSA_BINARY_MISSING_CODES)
    unknown = ~(missing | codes.isin(BINARY_CODE_MAP))
    if fail_on_unknown and unknown.any():
        counts = codes[unknown].value_counts().head(20).to_dict()
        raise ValueError(
            f"{variable} contains unmapped response codes: {counts}. "
            "Confirm the CLSA data dictionary before proceeding."
        )
    output = codes.map(BINARY_CODE_MAP).astype(float)
    return output.mask(missing)


def write_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported table extension: {path}")
    return path


def write_json(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, default=str)
    return path


def benjamini_hochberg(p_values) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(values[valid])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def safe_zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    standard_deviation = float(values.std(ddof=1))
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (values - float(values.mean())) / standard_deviation


def finite_embedding(value, expected_dim: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != expected_dim:
        raise ValueError(
            f"Expected {expected_dim} RETFound features; found {vector.size}"
        )
    if not np.isfinite(vector).all():
        raise ValueError("RETFound embedding contains nonfinite values")
    return vector


def stable_value(series: pd.Series):
    values = series.dropna()
    if values.empty:
        return None
    modes = values.mode(dropna=True)
    return modes.iloc[0] if not modes.empty else values.iloc[0]


def locate_baseline_member(archive_path: Path, suffix: str) -> tuple[str, list[str]]:
    with zipfile.ZipFile(archive_path) as archive:
        matches = [name for name in archive.namelist() if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one archive member ending {suffix!r}; "
                f"found {len(matches)}"
            )
        member = matches[0]
        with archive.open(member) as stream:
            header = pd.read_csv(stream, nrows=0).columns.tolist()
    return member, header


def find_identifier_column(columns: list[str]) -> str:
    candidates = (
        "entity_id",
        "participant_id",
        "ID",
        "id",
        "Entity_ID",
        "ENTITY_ID",
    )
    identifier = next((column for column in candidates if column in columns), None)
    if identifier is None:
        raise ValueError("Unable to identify the baseline participant ID column")
    return identifier


def load_baseline_fields(
    archive_path: Path,
    member: str,
    identifier_column: str,
    fields: list[str],
    *,
    eligible_ids: set[str] | None = None,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    usecols = [identifier_column, *fields]
    retained = []
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member) as stream:
            for chunk in pd.read_csv(
                stream,
                usecols=usecols,
                dtype="string",
                chunksize=chunksize,
                low_memory=False,
            ):
                chunk[identifier_column] = normalize_identifier(
                    chunk[identifier_column]
                )
                if eligible_ids is not None:
                    chunk = chunk[chunk[identifier_column].isin(eligible_ids)]
                if not chunk.empty:
                    retained.append(chunk)
    frame = (
        pd.concat(retained, ignore_index=True)
        if retained
        else pd.DataFrame(columns=usecols)
    ).rename(columns={identifier_column: "participant_id"})
    if frame["participant_id"].duplicated().any():
        duplicates = int(frame["participant_id"].duplicated(keep=False).sum())
        raise ValueError(
            f"Baseline questionnaire source contains {duplicates} duplicate "
            "participant rows"
        )
    return frame


baseline_member, baseline_header = locate_baseline_member(
    baseline_questionnaire_archive_path,
    baseline_questionnaire_member_suffix,
)
baseline_id_column = find_identifier_column(baseline_header)
for required_field in (VIS_LABEL_COLUMN, ICQ_HISTORY_COLUMN):
    if required_field not in baseline_header:
        raise ValueError(
            f"Required baseline field {required_field} is absent from "
            f"{baseline_member}"
        )

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Assemble one baseline embedding and one VIS cataract label per participant
# MAGIC
# MAGIC No ICQ cataract field is loaded into the classifier cohort. If the input
# MAGIC has multiple baseline embedding rows for a participant, the rows are
# MAGIC averaged before splitting so that no participant can cross the split.

# COMMAND ----------
embedding_source = pd.read_parquet(participant_embedding_path)
embedding_source.attrs = {}
required_embedding_columns = {"participant_id", "visit", "embedding"}
missing_embedding_columns = required_embedding_columns - set(
    embedding_source.columns
)
if missing_embedding_columns:
    raise ValueError(
        "Participant embedding table is missing columns: "
        f"{sorted(missing_embedding_columns)}"
    )

embedding_source["participant_id"] = normalize_identifier(
    embedding_source["participant_id"]
)
embedding_source["visit"] = normalize_visit(embedding_source["visit"])
embedding_source = embedding_source[
    embedding_source["visit"].eq("BL")
    & embedding_source["participant_id"].notna()
].copy()
if embedding_source.empty:
    raise ValueError("No baseline RETFound embedding records were found")
embedding_source["embedding"] = embedding_source["embedding"].map(
    lambda value: finite_embedding(value, expected_embedding_dim)
)

metadata_columns = [
    column
    for column in (
        "age",
        "age_years",
        "age_at_fundus_years",
        "sex",
        "sex_at_birth",
        "n_embedded_images",
    )
    if column in embedding_source.columns
]


def aggregate_embedding_group(group: pd.DataFrame) -> pd.Series:
    matrix = np.stack(group["embedding"].to_numpy()).astype(np.float64)
    row = {
        "embedding": matrix.mean(axis=0).astype(np.float32),
        "source_embedding_rows": int(len(group)),
    }
    for column in metadata_columns:
        row[column] = stable_value(group[column])
    return pd.Series(row)


participant_embeddings = (
    embedding_source.groupby("participant_id", sort=True, observed=True)[
        ["embedding", *metadata_columns]
    ]
    .apply(aggregate_embedding_group)
    .reset_index()
)
if participant_embeddings["participant_id"].duplicated().any():
    raise RuntimeError("Participant embedding aggregation did not create unique rows")

# Load exactly the classifier outcome and participant ID from the baseline file.
classifier_labels = load_baseline_fields(
    baseline_questionnaire_archive_path,
    baseline_member,
    baseline_id_column,
    [VIS_LABEL_COLUMN],
    eligible_ids=set(participant_embeddings["participant_id"].astype(str)),
)
classifier_labels["cataract_label"] = coerce_binary_strict(
    classifier_labels[VIS_LABEL_COLUMN],
    variable=VIS_LABEL_COLUMN,
)

label_code_audit = (
    classifier_labels.assign(
        raw_code=normalized_code(classifier_labels[VIS_LABEL_COLUMN]),
        analysis_label=classifier_labels["cataract_label"],
    )
    .groupby(["raw_code", "analysis_label"], dropna=False)
    .size()
    .reset_index(name="participants")
)
write_frame(label_code_audit, classifier_root / "vis_catrct_com_code_audit.csv")

classifier_cohort = participant_embeddings.merge(
    classifier_labels[["participant_id", "cataract_label"]],
    on="participant_id",
    how="inner",
    validate="one_to_one",
).dropna(subset=["cataract_label"])
classifier_cohort["cataract_label"] = classifier_cohort[
    "cataract_label"
].astype(int)

class_counts = classifier_cohort["cataract_label"].value_counts().reindex(
    [0, 1], fill_value=0
)
if (class_counts < max(50, inner_cv_folds)).any():
    raise ValueError(
        "Both VIS_CATRCT_COM classes require at least "
        f"{max(50, inner_cv_folds)} participants; counts={class_counts.to_dict()}"
    )
if classifier_cohort["participant_id"].duplicated().any():
    raise ValueError("Classifier cohort must contain one row per participant")

classifier_cohort_summary = pd.DataFrame(
    [
        {
            "stage": "baseline_embeddings",
            "participants": int(len(participant_embeddings)),
            "events": np.nan,
            "event_prevalence": np.nan,
        },
        {
            "stage": "nonmissing_VIS_CATRCT_COM",
            "participants": int(len(classifier_cohort)),
            "events": int(classifier_cohort["cataract_label"].sum()),
            "event_prevalence": float(classifier_cohort["cataract_label"].mean()),
        },
    ]
)

display(classifier_cohort_summary.round(4))
display(label_code_audit)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Prespecified 70/30 participant split and development-only tuning
# MAGIC
# MAGIC The 30% validation partition remains untouched during hyperparameter and
# MAGIC threshold selection. L2 regularization is chosen by stratified five-fold
# MAGIC cross-validation within the 70% development partition. The operating
# MAGIC threshold maximizes Youden's J on development-set out-of-fold predictions;
# MAGIC it is then applied unchanged to validation.

# COMMAND ----------
development_index, validation_index = train_test_split(
    np.arange(len(classifier_cohort)),
    test_size=validation_fraction,
    stratify=classifier_cohort["cataract_label"].to_numpy(),
    random_state=random_state,
    shuffle=True,
)
development = classifier_cohort.iloc[development_index].copy().reset_index(drop=True)
validation = classifier_cohort.iloc[validation_index].copy().reset_index(drop=True)

development_ids = set(development["participant_id"].astype(str))
validation_ids = set(validation["participant_id"].astype(str))
if development_ids & validation_ids:
    raise RuntimeError("Participant leakage detected across the 70/30 split")
if len(development) + len(validation) != len(classifier_cohort):
    raise RuntimeError("The 70/30 split did not retain every eligible participant")

split_assignment = pd.concat(
    [
        development[["participant_id", "cataract_label"]].assign(
            split="development_70"
        ),
        validation[["participant_id", "cataract_label"]].assign(
            split="locked_validation_30"
        ),
    ],
    ignore_index=True,
)
write_frame(split_assignment, private_root / "classifier_split_private.parquet")

split_summary = (
    split_assignment.groupby("split", as_index=False)
    .agg(
        participants=("participant_id", "nunique"),
        cataract_events=("cataract_label", "sum"),
        cataract_prevalence=("cataract_label", "mean"),
    )
)
classifier_cohort_summary = pd.concat(
    [
        classifier_cohort_summary,
        split_summary.rename(
            columns={
                "split": "stage",
                "cataract_events": "events",
                "cataract_prevalence": "event_prevalence",
            }
        ),
    ],
    ignore_index=True,
)
write_frame(classifier_cohort_summary, classifier_root / "cohort_flow.csv")

x_development = np.stack(development["embedding"].to_numpy()).astype(np.float64)
y_development = development["cataract_label"].to_numpy(int)
x_validation = np.stack(validation["embedding"].to_numpy()).astype(np.float64)
y_validation = validation["cataract_label"].to_numpy(int)

minimum_development_class = int(np.bincount(y_development).min())
usable_inner_folds = min(inner_cv_folds, minimum_development_class)
if usable_inner_folds < 3:
    raise ValueError("Development data cannot support at least three CV folds")

inner_cv = StratifiedKFold(
    n_splits=usable_inner_folds,
    shuffle=True,
    random_state=random_state + 1,
)
base_pipeline = Pipeline(
    [
        ("scale", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                penalty="l2",
                solver="liblinear",
                max_iter=5000,
                random_state=random_state,
            ),
        ),
    ]
)
grid = GridSearchCV(
    estimator=base_pipeline,
    param_grid={"classifier__C": list(C_GRID)},
    scoring="roc_auc",
    cv=inner_cv,
    n_jobs=-1,
    refit=True,
    return_train_score=False,
    error_score="raise",
)
grid.fit(x_development, y_development)
selected_pipeline = clone(grid.best_estimator_)

development_oof_probability = cross_val_predict(
    selected_pipeline,
    x_development,
    y_development,
    cv=inner_cv,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]


def select_youden_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y_true, probability
    )
    finite = np.isfinite(thresholds)
    candidates = pd.DataFrame(
        {
            "threshold": thresholds[finite],
            "youden_j": (
                true_positive_rate[finite] - false_positive_rate[finite]
            ),
        }
    )
    best = candidates[candidates["youden_j"].eq(candidates["youden_j"].max())]
    return float(best.iloc[(best["threshold"] - 0.5).abs().argmin()]["threshold"])


operating_threshold = select_youden_threshold(
    y_development, development_oof_probability
)
final_model = clone(grid.best_estimator_).fit(x_development, y_development)
validation_probability = final_model.predict_proba(x_validation)[:, 1]

if not np.isfinite(validation_probability).all():
    raise RuntimeError("Validation probabilities contain nonfinite values")

cv_results = pd.DataFrame(grid.cv_results_)[
    [
        "param_classifier__C",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
    ]
].rename(
    columns={
        "param_classifier__C": "C",
        "mean_test_score": "mean_inner_cv_auc",
        "std_test_score": "sd_inner_cv_auc",
        "rank_test_score": "rank",
    }
)
write_frame(cv_results, classifier_root / "development_hyperparameter_search.csv")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Locked-validation performance and uncertainty

# COMMAND ----------
def metric_dictionary(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc_average_precision": float(
            average_precision_score(y_true, probability)
        ),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, prediction)
        ),
        "sensitivity_recall": float(recall_score(y_true, prediction)),
        "specificity": float(specificity),
        "ppv_precision": float(
            precision_score(y_true, prediction, zero_division=np.nan)
        ),
        "npv": float(npv),
        "f1": float(f1_score(y_true, prediction)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "events": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
    }


def calibration_statistics(
    y_true: np.ndarray, probability: np.ndarray
) -> dict[str, float]:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped))
    design = pd.DataFrame({"outcome": y_true, "logit": logit})
    try:
        import statsmodels.api as sm

        fit = sm.GLM.from_formula(
            "outcome ~ logit", data=design, family=sm.families.Binomial()
        ).fit()
        confidence = fit.conf_int()
        return {
            "calibration_intercept": float(fit.params["Intercept"]),
            "calibration_intercept_ci_low": float(
                confidence.loc["Intercept", 0]
            ),
            "calibration_intercept_ci_high": float(
                confidence.loc["Intercept", 1]
            ),
            "calibration_slope": float(fit.params["logit"]),
            "calibration_slope_ci_low": float(confidence.loc["logit", 0]),
            "calibration_slope_ci_high": float(confidence.loc["logit", 1]),
        }
    except Exception:
        return {
            "calibration_intercept": np.nan,
            "calibration_intercept_ci_low": np.nan,
            "calibration_intercept_ci_high": np.nan,
            "calibration_slope": np.nan,
            "calibration_slope_ci_low": np.nan,
            "calibration_slope_ci_high": np.nan,
        }


development_metrics = metric_dictionary(
    y_development,
    development_oof_probability,
    operating_threshold,
)
validation_metrics = metric_dictionary(
    y_validation,
    validation_probability,
    operating_threshold,
)
calibration = calibration_statistics(
    y_validation, validation_probability
)
validation_metrics.update(calibration)


def stratified_bootstrap_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    negative = np.flatnonzero(y_true == 0)
    positive = np.flatnonzero(y_true == 1)
    rng = np.random.default_rng(seed)
    rows = []
    for repetition in range(repetitions):
        index = np.concatenate(
            [
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            ]
        )
        sampled = metric_dictionary(
            y_true[index], probability[index], threshold
        )
        sampled["bootstrap_repetition"] = repetition + 1
        rows.append(sampled)
    return pd.DataFrame(rows)


validation_bootstrap = stratified_bootstrap_metrics(
    y_validation,
    validation_probability,
    operating_threshold,
    bootstrap_repetitions,
    random_state + 200,
)
interval_metric_names = [
    "roc_auc",
    "pr_auc_average_precision",
    "accuracy",
    "balanced_accuracy",
    "sensitivity_recall",
    "specificity",
    "ppv_precision",
    "npv",
    "f1",
    "brier_score",
    "log_loss",
]
validation_performance = pd.DataFrame(
    [
        {
            "metric": metric,
            "estimate": validation_metrics[metric],
            "ci_low": float(validation_bootstrap[metric].quantile(0.025)),
            "ci_high": float(validation_bootstrap[metric].quantile(0.975)),
            "confidence_interval": (
                "participant-stratified nonparametric bootstrap"
            ),
            "bootstrap_repetitions": bootstrap_repetitions,
        }
        for metric in interval_metric_names
    ]
)
validation_performance = pd.concat(
    [
        validation_performance,
        pd.DataFrame(
            [
                {
                    "metric": "calibration_intercept",
                    "estimate": calibration["calibration_intercept"],
                    "ci_low": calibration["calibration_intercept_ci_low"],
                    "ci_high": calibration["calibration_intercept_ci_high"],
                    "confidence_interval": "model-based Wald interval",
                    "bootstrap_repetitions": 0,
                },
                {
                    "metric": "calibration_slope",
                    "estimate": calibration["calibration_slope"],
                    "ci_low": calibration["calibration_slope_ci_low"],
                    "ci_high": calibration["calibration_slope_ci_high"],
                    "confidence_interval": "model-based Wald interval",
                    "bootstrap_repetitions": 0,
                },
            ]
        ),
    ],
    ignore_index=True,
)

confusion_counts = pd.DataFrame(
    {
        "cell": ["TN", "FP", "FN", "TP"],
        "count": [
            validation_metrics["true_negative"],
            validation_metrics["false_positive"],
            validation_metrics["false_negative"],
            validation_metrics["true_positive"],
        ],
        "threshold": operating_threshold,
    }
)

validation_predictions = validation.drop(columns=["embedding"]).copy()
validation_predictions["cataract_probability"] = validation_probability
validation_predictions["cataract_prediction"] = (
    validation_probability >= operating_threshold
).astype(int)
development_predictions = development.drop(columns=["embedding"]).copy()
development_predictions["cataract_probability_oof"] = (
    development_oof_probability
)
development_predictions["cataract_prediction_oof"] = (
    development_oof_probability >= operating_threshold
).astype(int)

write_frame(
    validation_performance,
    classifier_root / "locked_validation_performance.csv",
)
write_frame(confusion_counts, classifier_root / "locked_validation_confusion.csv")
write_frame(
    development_predictions,
    private_root / "development_oof_predictions_private.parquet",
)
write_frame(
    validation_predictions,
    private_root / "locked_validation_predictions_private.parquet",
)
joblib.dump(
    {
        "model": final_model,
        "operating_threshold": operating_threshold,
        "embedding_dim": expected_embedding_dim,
        "label_column": VIS_LABEL_COLUMN,
        "visit": "BL",
        "development_fraction": train_fraction,
        "random_state": random_state,
        "selected_C": float(grid.best_params_["classifier__C"]),
    },
    classifier_root / "retfound_vis_cataract_classifier.joblib",
)

display(split_summary.round(4))
display(validation_performance.round(4))
display(confusion_counts)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. ROC, precision–recall, and calibration figures

# COMMAND ----------
false_positive_rate, true_positive_rate, _ = roc_curve(
    y_validation, validation_probability
)
precision, recall, _ = precision_recall_curve(
    y_validation, validation_probability
)
calibration_true, calibration_predicted = calibration_curve(
    y_validation,
    validation_probability,
    n_bins=10,
    strategy="quantile",
)

figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
axes[0].plot(false_positive_rate, true_positive_rate, linewidth=2)
axes[0].plot([0, 1], [0, 1], "--", color="black", linewidth=1)
axes[0].set(
    title=f"ROC curve (AUC={validation_metrics['roc_auc']:.3f})",
    xlabel="1 − specificity",
    ylabel="Sensitivity",
    xlim=(0, 1),
    ylim=(0, 1),
)
axes[1].plot(recall, precision, linewidth=2)
axes[1].axhline(
    float(y_validation.mean()), linestyle="--", color="black", linewidth=1
)
axes[1].set(
    title=(
        "Precision–recall curve "
        f"(AP={validation_metrics['pr_auc_average_precision']:.3f})"
    ),
    xlabel="Sensitivity (recall)",
    ylabel="PPV (precision)",
    xlim=(0, 1),
    ylim=(0, 1),
)
axes[2].plot(calibration_predicted, calibration_true, "o-", linewidth=2)
axes[2].plot([0, 1], [0, 1], "--", color="black", linewidth=1)
axes[2].set(
    title="Validation calibration",
    xlabel="Mean predicted probability",
    ylabel="Observed fraction",
    xlim=(0, 1),
    ylim=(0, 1),
)
figure.suptitle(
    "Locked 30% validation: baseline VIS_CATRCT_COM from RETFound embeddings"
)
figure.savefig(
    figure_root / "cataract_classifier_locked_validation.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
plt.close(figure)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Questionnaire sensitivity: combined, ICQ-only, and VIS-only
# MAGIC
# MAGIC All three specifications use retinal age acceleration as the outcome.
# MAGIC Cataract variables are excluded from the questionnaire predictor family
# MAGIC to avoid testing a cataract predictor while conditioning on the same
# MAGIC construct. HC3 robust inference is used for coefficients, nested-model F
# MAGIC tests quantify the questionnaire variable's overall contribution, and
# MAGIC Benjamini–Hochberg correction is applied separately within each
# MAGIC prespecified cataract-definition scan.

# COMMAND ----------
phenotype = pd.read_parquet(three_clock_phenotype_path)
questionnaire = pd.read_parquet(questionnaire_answers_path)
questionnaire_audit = pd.read_csv(questionnaire_audit_path)
for frame in (phenotype, questionnaire):
    frame.attrs = {}
    if "participant_id" not in frame.columns:
        raise ValueError("Questionnaire sensitivity input lacks participant_id")
    frame["participant_id"] = normalize_identifier(frame["participant_id"])
    if frame["participant_id"].duplicated().any():
        raise ValueError("Questionnaire sensitivity inputs must be participant-unique")

required_phenotype_columns = {
    "participant_id",
    "z_retinal_acceleration",
    "chronological_age",
    "n_embedded_images",
}
missing_phenotype_columns = required_phenotype_columns - set(phenotype.columns)
if missing_phenotype_columns:
    raise ValueError(
        "Three-clock phenotype table is missing: "
        f"{sorted(missing_phenotype_columns)}"
    )
for required_questionnaire_column in (ICQ_HISTORY_COLUMN, VIS_LABEL_COLUMN):
    if required_questionnaire_column not in questionnaire.columns:
        raise ValueError(
            "Questionnaire table is missing required cataract definition: "
            f"{required_questionnaire_column}"
        )

cataract_pattern = re.compile(r"(?i)^(ICQ|VIS)_CATRCT(?:_|$)")
combined_source_columns = [
    column for column in questionnaire.columns if cataract_pattern.search(column)
]
if not combined_source_columns:
    raise ValueError("No ICQ/VIS cataract columns were found for current composite")

cataract_components = pd.DataFrame(
    {
        column: coerce_binary_strict(
            questionnaire[column], variable=column, fail_on_unknown=False
        )
        for column in combined_source_columns
    },
    index=questionnaire.index,
)
combined_any = cataract_components.max(axis=1, skipna=True).mask(
    cataract_components.notna().sum(axis=1).eq(0)
)
questionnaire_status = questionnaire[["participant_id"]].copy()
questionnaire_status["cataract_current_combined"] = combined_any
questionnaire_status["cataract_icq_only"] = coerce_binary_strict(
    questionnaire[ICQ_HISTORY_COLUMN], variable=ICQ_HISTORY_COLUMN
)
questionnaire_status["cataract_vis_only"] = coerce_binary_strict(
    questionnaire[VIS_LABEL_COLUMN], variable=VIS_LABEL_COLUMN
)

cataract_definition_audit = []
for definition, source_description in (
    (
        "cataract_current_combined",
        "Any positive baseline ICQ_CATRCT* or VIS_CATRCT* field",
    ),
    ("cataract_icq_only", ICQ_HISTORY_COLUMN),
    ("cataract_vis_only", VIS_LABEL_COLUMN),
):
    values = questionnaire_status[definition]
    cataract_definition_audit.append(
        {
            "definition": definition,
            "source": source_description,
            "participants": int(len(values)),
            "nonmissing": int(values.notna().sum()),
            "events": int(values.eq(1).sum()),
            "prevalence": float(values.mean()),
        }
    )
cataract_definition_audit = pd.DataFrame(cataract_definition_audit)
write_frame(
    cataract_definition_audit,
    questionnaire_root / "cataract_definition_audit.csv",
)

analysis_frame = phenotype.merge(
    questionnaire,
    on="participant_id",
    how="left",
    validate="one_to_one",
).merge(
    questionnaire_status,
    on="participant_id",
    how="left",
    validate="one_to_one",
)
analysis_frame["chronological_age"] = pd.to_numeric(
    analysis_frame["chronological_age"], errors="coerce"
)
analysis_frame["chronological_age_sq"] = analysis_frame[
    "chronological_age"
] ** 2
analysis_frame["log_n_embedded_images"] = np.log1p(
    pd.to_numeric(analysis_frame["n_embedded_images"], errors="coerce").fillna(0)
)

base_terms = [
    "chronological_age",
    "chronological_age_sq",
    "log_n_embedded_images",
]
base_complete_columns = [
    "z_retinal_acceleration",
    "chronological_age",
    "chronological_age_sq",
    "log_n_embedded_images",
]
for categorical in ("sex_labeled", "racial_background", "smoking_status"):
    if categorical not in analysis_frame.columns:
        continue
    values = analysis_frame[categorical].astype("string").str.strip()
    counts = values.value_counts(dropna=True)
    frequent = counts[counts >= 10].index
    if len(frequent) < 2:
        continue
    model_column = f"{categorical}_association"
    analysis_frame[model_column] = (
        values.where(values.isna() | values.isin(frequent), "Other/low-frequency")
        .fillna("Missing")
        .astype(str)
    )
    base_terms.append(f"C({model_column})")
    base_complete_columns.append(model_column)

for continuous in ("bmi", "body_mass_index"):
    if continuous in analysis_frame.columns:
        analysis_frame[continuous] = pd.to_numeric(
            analysis_frame[continuous], errors="coerce"
        )
        if analysis_frame[continuous].notna().sum() >= 100:
            base_terms.append(continuous)
            base_complete_columns.append(continuous)
            break

if "analyzed" not in questionnaire_audit.columns:
    raise ValueError("Questionnaire audit lacks the analyzed flag")
analyzed_flag = questionnaire_audit["analyzed"].astype(str).str.lower().isin(
    {"true", "1", "yes"}
)
candidate_audit = questionnaire_audit[analyzed_flag].copy()
candidate_audit = candidate_audit[
    ~candidate_audit["variable"].astype(str).map(
        lambda value: bool(cataract_pattern.search(value))
    )
]
candidate_audit = candidate_audit[
    candidate_audit["variable"].isin(analysis_frame.columns)
]
if candidate_audit.empty:
    raise ValueError("No non-cataract questionnaire variables are available")

questionnaire_specifications = {
    "current_combined_icq_or_vis": "cataract_current_combined",
    "strict_icq_only": "cataract_icq_only",
    "strict_vis_only": "cataract_vis_only",
}
questionnaire_rows = []
questionnaire_failure_rows = []

for specification, cataract_covariate in questionnaire_specifications.items():
    for record in candidate_audit.to_dict("records"):
        variable = str(record["variable"])
        variable_type = str(record.get("analysis_type", "categorical")).lower()
        variable_label = str(record.get("variable_label", variable))
        columns = list(
            dict.fromkeys(
                [
                    "z_retinal_acceleration",
                    *base_complete_columns,
                    cataract_covariate,
                    variable,
                ]
            )
        )
        work = analysis_frame[columns].copy()
        work["z_retinal_acceleration"] = pd.to_numeric(
            work["z_retinal_acceleration"], errors="coerce"
        )
        work[cataract_covariate] = pd.to_numeric(
            work[cataract_covariate], errors="coerce"
        )
        if variable_type == "numeric":
            work["question_value_numeric"] = safe_zscore(
                pd.to_numeric(work[variable], errors="coerce")
            )
            question_term = "question_value_numeric"
            modeled_question_column = "question_value_numeric"
            minimum_modeled_level_n = np.nan
        else:
            cleaned = work[variable].astype("string").str.strip()
            cleaned = cleaned.mask(
                cleaned.str.upper().isin(CLSA_MISSING_CODES) | cleaned.isna()
            )
            counts = cleaned.value_counts(dropna=True)
            frequent = counts[counts >= minimum_questionnaire_level_n].index
            cleaned = cleaned.where(
                cleaned.isna() | cleaned.isin(frequent),
                "Other/low-frequency",
            )
            collapsed_counts = cleaned.value_counts(dropna=True)
            if (
                "Other/low-frequency" in collapsed_counts
                and collapsed_counts["Other/low-frequency"]
                < minimum_questionnaire_level_n
            ):
                cleaned = cleaned.mask(cleaned.eq("Other/low-frequency"))
            work["question_value_categorical"] = pd.Categorical(cleaned)
            question_term = "C(question_value_categorical)"
            modeled_question_column = "question_value_categorical"
            minimum_modeled_level_n = (
                int(cleaned.value_counts(dropna=True).min())
                if cleaned.notna().any()
                else 0
            )

        complete_columns = list(
            dict.fromkeys(
                [
                    "z_retinal_acceleration",
                    *base_complete_columns,
                    cataract_covariate,
                    modeled_question_column,
                ]
            )
        )
        work = work.dropna(subset=complete_columns)
        if len(work) < minimum_questionnaire_observed:
            questionnaire_failure_rows.append(
                {
                    "specification": specification,
                    "variable": variable,
                    "status": "insufficient_complete_cases",
                    "n": int(len(work)),
                }
            )
            continue
        if work[cataract_covariate].nunique() < 2:
            questionnaire_failure_rows.append(
                {
                    "specification": specification,
                    "variable": variable,
                    "status": "cataract_covariate_lacks_both_levels",
                    "n": int(len(work)),
                }
            )
            continue
        if variable_type != "numeric":
            work["question_value_categorical"] = work[
                "question_value_categorical"
            ].cat.remove_unused_categories()
            level_counts = work["question_value_categorical"].value_counts()
            if len(level_counts) < 2:
                questionnaire_failure_rows.append(
                    {
                        "specification": specification,
                        "variable": variable,
                        "status": "fewer_than_two_complete_case_levels",
                        "n": int(len(work)),
                    }
                )
                continue
            minimum_modeled_level_n = int(level_counts.min())

        specification_terms = [*base_terms, cataract_covariate]
        base_formula = "z_retinal_acceleration ~ " + " + ".join(
            specification_terms
        )
        full_formula = base_formula + " + " + question_term
        try:
            base_model = smf.ols(base_formula, data=work)
            full_model = smf.ols(full_formula, data=work)
            base_rank = int(np.linalg.matrix_rank(base_model.exog))
            full_rank = int(np.linalg.matrix_rank(full_model.exog))
            if (
                base_rank < base_model.exog.shape[1]
                or full_rank < full_model.exog.shape[1]
            ):
                questionnaire_failure_rows.append(
                    {
                        "specification": specification,
                        "variable": variable,
                        "status": "rank_deficient_design",
                        "n": int(len(work)),
                    }
                )
                continue
            base_fit = base_model.fit()
            full_fit = full_model.fit()
            robust_fit = full_model.fit(cov_type="HC3")
            nested_f, nested_p, nested_df = full_fit.compare_f_test(base_fit)
            result = {
                "specification": specification,
                "cataract_covariate": cataract_covariate,
                "variable": variable,
                "variable_label": variable_label,
                "analysis_type": variable_type,
                "n": int(full_fit.nobs),
                "observed_levels": int(work[modeled_question_column].nunique()),
                "minimum_modeled_level_n": minimum_modeled_level_n,
                "base_r2": float(base_fit.rsquared),
                "full_r2": float(full_fit.rsquared),
                "incremental_r2": float(full_fit.rsquared - base_fit.rsquared),
                "omnibus_f": float(nested_f),
                "p_value": float(nested_p),
                "df_difference": float(nested_df),
            }
            if variable_type == "numeric":
                interval = robust_fit.conf_int().loc["question_value_numeric"]
                result.update(
                    {
                        "standardized_coefficient": float(
                            robust_fit.params["question_value_numeric"]
                        ),
                        "coefficient_ci_low": float(interval.iloc[0]),
                        "coefficient_ci_high": float(interval.iloc[1]),
                        "robust_coefficient_p": float(
                            robust_fit.pvalues["question_value_numeric"]
                        ),
                    }
                )
            questionnaire_rows.append(result)
        except Exception as error:
            questionnaire_failure_rows.append(
                {
                    "specification": specification,
                    "variable": variable,
                    "status": f"failed:{type(error).__name__}",
                    "error_message": str(error)[:500],
                    "n": int(len(work)),
                }
            )

questionnaire_results = pd.DataFrame(questionnaire_rows)
questionnaire_failures = pd.DataFrame(questionnaire_failure_rows)
if questionnaire_results.empty:
    raise RuntimeError("Every questionnaire sensitivity model failed")

questionnaire_results["fdr_q_within_specification"] = np.nan
for specification, index in questionnaire_results.groupby(
    "specification"
).groups.items():
    questionnaire_results.loc[index, "fdr_q_within_specification"] = (
        benjamini_hochberg(
            questionnaire_results.loc[index, "p_value"].to_numpy(float)
        )
    )
questionnaire_results["manuscript_eligible"] = (
    questionnaire_results["n"].ge(minimum_questionnaire_manuscript_n)
    & (
        questionnaire_results["analysis_type"].eq("numeric")
        | questionnaire_results["minimum_modeled_level_n"].ge(
            minimum_questionnaire_level_n
        )
    )
)
questionnaire_results["significant_fdr_0_05"] = questionnaire_results[
    "fdr_q_within_specification"
].lt(0.05)

questionnaire_robustness = (
    questionnaire_results.pivot_table(
        index=["variable", "variable_label"],
        columns="specification",
        values="fdr_q_within_specification",
        aggfunc="first",
    )
    .reset_index()
)
for specification in questionnaire_specifications:
    if specification not in questionnaire_robustness.columns:
        questionnaire_robustness[specification] = np.nan
    questionnaire_robustness[f"significant_{specification}"] = (
        questionnaire_robustness[specification].lt(0.05)
    )
questionnaire_robustness["significant_in_all_three"] = questionnaire_robustness[
    [
        f"significant_{specification}"
        for specification in questionnaire_specifications
    ]
].all(axis=1)

write_frame(
    questionnaire_results,
    questionnaire_root
    / "retinal_acceleration_questionnaire_three_cataract_definitions.csv",
)
write_frame(
    questionnaire_failures,
    questionnaire_root
    / "retinal_acceleration_questionnaire_failures.csv",
)
write_frame(
    questionnaire_robustness,
    questionnaire_root
    / "retinal_acceleration_questionnaire_robustness.csv",
)

display(cataract_definition_audit.round(4))
display(
    questionnaire_results.sort_values(
        ["specification", "fdr_q_within_specification", "p_value"]
    ).groupby("specification", as_index=False).head(20).round(6)
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Questionnaire comparison figure and manuscript-ready results text

# COMMAND ----------
eligible_results = questionnaire_results[
    questionnaire_results["manuscript_eligible"]
].copy()
if not eligible_results.empty:
    eligible_results["minus_log10_q"] = -np.log10(
        eligible_results["fdr_q_within_specification"].clip(lower=1e-300)
    )
    strongest_variables = (
        eligible_results.groupby("variable")["minus_log10_q"]
        .max()
        .nlargest(25)
        .index
    )
    heatmap = (
        eligible_results[eligible_results["variable"].isin(strongest_variables)]
        .pivot_table(
            index="variable_label",
            columns="specification",
            values="minus_log10_q",
            aggfunc="first",
        )
        .reindex(columns=list(questionnaire_specifications))
        .fillna(0)
    )
    if not heatmap.empty:
        figure, axis = plt.subplots(
            figsize=(12, max(7, 0.42 * len(heatmap))), layout="constrained"
        )
        sns.heatmap(
            heatmap,
            cmap="mako",
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "−log10(FDR q-value)"},
            ax=axis,
        )
        axis.set(
            title=(
                "Questionnaire associations with retinal age acceleration "
                "across cataract definitions"
            ),
            xlabel="Cataract adjustment specification",
            ylabel="",
        )
        axis.set_xticklabels(
            ["Combined ICQ/VIS", "ICQ only", "VIS only"],
            rotation=15,
            ha="right",
        )
        figure.savefig(
            figure_root / "questionnaire_cataract_definition_sensitivity.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
        plt.close(figure)


def format_interval(metric_name: str) -> str:
    row = validation_performance[
        validation_performance["metric"].eq(metric_name)
    ].iloc[0]
    return (
        f"{row['estimate']:.3f} "
        f"(95% CI {row['ci_low']:.3f}–{row['ci_high']:.3f})"
    )


result_lines = [
    "RETFound cataract classifier",
    (
        f"Among {len(classifier_cohort):,} baseline participants with a valid "
        f"RETFound embedding and nonmissing {VIS_LABEL_COLUMN}, "
        f"{int(classifier_cohort['cataract_label'].sum()):,} "
        f"({100 * classifier_cohort['cataract_label'].mean():.1f}%) had cataract "
        "noted during vision testing."
    ),
    (
        f"Participants were split into development (n={len(development):,}; 70%) "
        f"and locked validation (n={len(validation):,}; 30%) sets. "
        f"The development-selected threshold was {operating_threshold:.3f}."
    ),
    (
        "In locked validation, ROC AUC was "
        f"{format_interval('roc_auc')}, average precision was "
        f"{format_interval('pr_auc_average_precision')}, sensitivity was "
        f"{format_interval('sensitivity_recall')}, specificity was "
        f"{format_interval('specificity')}, PPV was "
        f"{format_interval('ppv_precision')}, and NPV was "
        f"{format_interval('npv')}."
    ),
    "",
    "Questionnaire sensitivity to cataract definition",
]
for specification, cataract_covariate in questionnaire_specifications.items():
    subset = questionnaire_results[
        questionnaire_results["specification"].eq(specification)
        & questionnaire_results["manuscript_eligible"]
    ].sort_values(["fdr_q_within_specification", "p_value"])
    significant = subset[subset["significant_fdr_0_05"]]
    if significant.empty:
        summary = "no manuscript-eligible variables passed FDR q<0.05"
    else:
        names = ", ".join(significant["variable_label"].head(5).astype(str))
        suffix = "" if len(significant) <= 5 else ", among others"
        summary = (
            f"{len(significant)} manuscript-eligible variables passed FDR q<0.05; "
            f"the leading findings were {names}{suffix}"
        )
    result_lines.append(
        f"Under {specification} ({cataract_covariate}), {summary}."
    )
robust_count = int(questionnaire_robustness["significant_in_all_three"].sum())
result_lines.append(
    f"Overall, {robust_count} questionnaire associations were significant under "
    "all three cataract definitions."
)

results_text_path = questionnaire_root / "manuscript_ready_results.txt"
results_text_path.write_text("\n".join(result_lines) + "\n", encoding="utf-8")
print("\n".join(result_lines))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Reproducibility and model card

# COMMAND ----------
model_card = {
    "analysis": "baseline_RETFound_VIS_cataract_classifier",
    "outcome": {
        "column": VIS_LABEL_COLUMN,
        "visit": "BL",
        "fields_used_for_classifier_label": [VIS_LABEL_COLUMN],
        "ICQ_fields_used_for_classifier_label": [],
        "interpretation": "cataract noted during CLSA vision testing",
        "limitations": [
            "participant-level rather than eye-specific",
            "not an ophthalmologist-adjudicated cataract grade",
            "not intended for clinical diagnosis",
        ],
    },
    "features": {
        "type": "participant-level mean frozen RETFound embedding",
        "dimension": expected_embedding_dim,
        "clinical_covariates_in_classifier": [],
    },
    "split": {
        "development_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "participant_disjoint": True,
        "stratified_by_outcome": True,
        "random_state": random_state,
    },
    "model": {
        "estimator": "standardized L2 logistic regression",
        "C_grid": list(C_GRID),
        "selected_C": float(grid.best_params_["classifier__C"]),
        "selection_metric": "inner cross-validated ROC AUC",
        "inner_cv_folds": usable_inner_folds,
        "threshold_rule": "maximum Youden J on development OOF predictions",
        "operating_threshold": operating_threshold,
    },
    "locked_validation": validation_metrics,
    "questionnaire_cataract_definitions": questionnaire_specifications,
    "software": {
        "python": "Databricks runtime Python",
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "statsmodels": __import__("statsmodels").__version__,
        "matplotlib": __import__("matplotlib").__version__,
        "seaborn": sns.__version__,
        "joblib": joblib.__version__,
    },
    "outputs": {
        "validation_performance": str(
            classifier_root / "locked_validation_performance.csv"
        ),
        "model": str(classifier_root / "retfound_vis_cataract_classifier.joblib"),
        "questionnaire_results": str(
            questionnaire_root
            / "retinal_acceleration_questionnaire_three_cataract_definitions.csv"
        ),
        "results_text": str(results_text_path),
    },
}
write_json(model_card, output_root / "ANALYSIS_MODEL_CARD.json")

print("Completed classifier and questionnaire sensitivity analysis.")
print("Model card:", output_root / "ANALYSIS_MODEL_CARD.json")
print("Locked validation results:", classifier_root / "locked_validation_performance.csv")
print("Questionnaire results:", questionnaire_root / "retinal_acceleration_questionnaire_three_cataract_definitions.csv")
