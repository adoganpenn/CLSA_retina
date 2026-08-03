# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Build the CLSA retinal-aging SAP analysis dataset
# MAGIC
# MAGIC This notebook extracts only the required baseline and Follow-up 1
# MAGIC questionnaire CSV members, standardizes the SAP variables, derives the
# MAGIC prespecified measures, and links them to fundus images by participant
# MAGIC **and visit**.
# MAGIC
# MAGIC Age linkage is deliberately visit-specific:
# MAGIC
# MAGIC - baseline (`BL`) fundus images → `AGE_NMBR_COM`
# MAGIC - Follow-up 1 (`F1`) fundus images → `AGE_NMBR_COF1`
# MAGIC
# MAGIC The fundus ZIPs do not contain a released capture timestamp. Assessment,
# MAGIC questionnaire, and participant-status dates are retained as timing
# MAGIC proxies and must not be described as exact image-acquisition dates.

# COMMAND ----------
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile

from pyspark.sql import functions as F

# COMMAND ----------
dbutils.widgets.text(
    "repo_root",
    "/Workspace/Users/ad0038@pennmedicine.upenn.edu/CLSA/CLSA_retina",
)
dbutils.widgets.text(
    "volume_root",
    "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset",
)
dbutils.widgets.text(
    "output_root",
    "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/derived/"
    "clsa_retinal_aging",
)
dbutils.widgets.text(
    "image_manifest_path",
    "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/derived/"
    "clsa_retinal_aging/fundus_image_manifest",
)
dbutils.widgets.dropdown(
    "extract_questionnaire_csv",
    "true",
    ["true", "false"],
)
dbutils.widgets.dropdown(
    "overwrite_extracted_csv",
    "false",
    ["false", "true"],
)
dbutils.widgets.dropdown(
    "confirm_visual_acuity_logmar",
    "false",
    ["false", "true"],
)

# COMMAND ----------
repo_root = dbutils.widgets.get("repo_root").rstrip("/")
volume_root = dbutils.widgets.get("volume_root").rstrip("/")
output_root = dbutils.widgets.get("output_root").rstrip("/")
image_manifest_path = dbutils.widgets.get("image_manifest_path").strip()
extract_questionnaire = (
    dbutils.widgets.get("extract_questionnaire_csv") == "true"
)
overwrite_extracted = (
    dbutils.widgets.get("overwrite_extracted_csv") == "true"
)
visual_acuity_logmar_confirmed = (
    dbutils.widgets.get("confirm_visual_acuity_logmar") == "true"
)

module_path = Path(repo_root) / "src" / "clsa_pipeline.py"
if not module_path.exists():
    raise FileNotFoundError(f"Pipeline module was not found: {module_path}")
if str(module_path.parent) not in sys.path:
    sys.path.insert(0, str(module_path.parent))

from clsa_pipeline import derive_retinal_metrics, write_delta  # noqa: E402

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Exact source releases
# MAGIC
# MAGIC These member names were observed in `DATASET_README.md`. The large
# MAGIC questionnaire files are unencrypted. Only the three required CSVs are
# MAGIC extracted; dictionary workbooks and decedent files are left untouched.

# COMMAND ----------
QUESTIONNAIRE_SOURCES = {
    "BL": {
        "archive_path": f"{volume_root}/2209017_UOttawa_EFreeman_BL.zip",
        "member_path": (
            "2209017_UOttawa_EFreeman_BL/"
            "2209017_UOttawa_EFreeman_Baseline_CoPv7_Qx_CANUE_PA_BS.csv"
        ),
    },
    "F1": {
        "archive_path": f"{volume_root}/2209017_UOttawa_EFreeman_FUP1.zip",
        "member_path": (
            "2209017_UOttawa_EFreeman_FUP1/"
            "2209017_UOttawa_EFreeman_FUP1_CoPv4_Qx_PA_BS.csv"
        ),
    },
    "STATUS": {
        "archive_path": f"{volume_root}/2209017_UOttawa_EFreeman_FUP1.zip",
        "member_path": (
            "2209017_UOttawa_EFreeman_FUP1/"
            "2209017_UOttawa_EFreeman_ParticipantStatus_CoP_v3_Sep2022.csv"
        ),
    },
}

for label, source in QUESTIONNAIRE_SOURCES.items():
    if not Path(source["archive_path"]).exists():
        raise FileNotFoundError(
            f"{label} questionnaire archive was not found: "
            f"{source['archive_path']}"
        )

# COMMAND ----------
def extract_exact_member(
    archive_path: str,
    member_path: str,
    destination_root: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Restartably extract one governed, unencrypted ZIP member."""
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / PurePosixPath(member_path).name
    partial = destination.with_suffix(destination.suffix + ".partial")

    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo(member_path)
        if info.flag_bits & 0x1:
            raise RuntimeError(
                f"Questionnaire member unexpectedly requires a password: "
                f"{member_path}"
            )
        if (
            destination.exists()
            and destination.stat().st_size == info.file_size
            and not overwrite
        ):
            status = "already_present"
        else:
            if partial.exists():
                partial.unlink()
            with archive.open(info) as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            if partial.stat().st_size != info.file_size:
                raise IOError(
                    f"Extracted byte count does not match ZIP metadata: "
                    f"{member_path}"
                )
            partial.replace(destination)
            status = "extracted"

    return {
        "archive_path": archive_path,
        "member_path": member_path,
        "output_path": str(destination),
        "bytes": int(destination.stat().st_size),
        "status": status,
    }


questionnaire_extract_root = (
    Path(output_root) / "questionnaire_extracted_sap"
)
extraction_rows = []
questionnaire_paths = {}
for label, source in QUESTIONNAIRE_SOURCES.items():
    destination = (
        questionnaire_extract_root
        / label
        / PurePosixPath(source["member_path"]).name
    )
    if extract_questionnaire:
        result = extract_exact_member(
            source["archive_path"],
            source["member_path"],
            questionnaire_extract_root / label,
            overwrite=overwrite_extracted,
        )
        extraction_rows.append({"release": label, **result})
    elif not destination.exists():
        raise FileNotFoundError(
            f"Extraction is disabled and the expected CSV is absent: {destination}"
        )
    questionnaire_paths[label] = str(destination)

extraction_schema = (
    "release string, archive_path string, member_path string, output_path string, "
    "bytes long, status string"
)
if extraction_rows:
    extraction_log = spark.createDataFrame(
        extraction_rows,
        schema=extraction_schema,
    )
    write_delta(
        extraction_log,
        f"{output_root}/sap_questionnaire_extraction_log",
    )
    display(extraction_log.groupBy("release", "status").count())

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Read only the required participant releases

# COMMAND ----------
def read_release_csv(path: str):
    return (
        spark.read.option("header", "true")
        .option("inferSchema", "false")
        .option("mode", "FAILFAST")
        .option("quote", '"')
        .option("escape", '"')
        # CLSA questionnaire exports can contain quoted line breaks. Without
        # multiline parsing, continuation lines become malformed extra rows and
        # can appear as duplicate/null participant records.
        .option("multiLine", "true")
        .option("maxColumns", "10000")
        .csv(path)
    )


baseline_raw = read_release_csv(questionnaire_paths["BL"])
followup1_raw = read_release_csv(questionnaire_paths["F1"])
status_raw = read_release_csv(questionnaire_paths["STATUS"])

print("Baseline columns:", len(baseline_raw.columns))
print("Follow-up 1 columns:", len(followup1_raw.columns))
print("Participant-status columns:", len(status_raw.columns))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. SAP variable map
# MAGIC
# MAGIC The map preserves source-variable provenance. Missing fields are created
# MAGIC as typed nulls rather than silently substituted from another construct.

# COMMAND ----------
STANDARD_COLUMNS = [
    "self_reported_vision",
    "visual_acuity_left",
    "visual_acuity_right",
    "visual_acuity_both",
    "cesd10_score",
    "age_years",
    "sex_at_birth",
    "ethnicity_spirometry",
    "education_level_sap",
    "marital_status",
    "household_income_band",
    "smoking_status",
    "diabetes",
    "hypertension",
    "heart_disease",
    "stroke",
    "oa_hand",
    "oa_hip",
    "oa_knee",
    "osteoporosis",
    "asthma",
    "copd",
    "cancer",
    "low_back_pain",
    "hearing_noise",
    "hearing_aid",
    "social_outside_household",
    "social_religious",
    "social_education_culture",
    "social_club",
    "social_association",
    "social_other",
    "adl_class",
    "self_rated_healthy_aging",
    "analytic_weight",
    "sampling_strata",
    "frailty",
    "epigenetic_age",
]

BASELINE_MAP = {
    "self_reported_vision": "VIS_SGHT_COM",
    "visual_acuity_left": "VA_ETDRS_L_RSLT_COM",
    "visual_acuity_right": "VA_ETDRS_R_RSLT_COM",
    "cesd10_score": "DEP_CESD10_COM",
    "age_years": "AGE_NMBR_COM",
    "sex_at_birth": "SEX_ASK_COM",
    "ethnicity_spirometry": "SPR_OUTPUT_ETHN_COM",
    "education_level_sap": "ED_UDR04_COM",
    "marital_status": "SDC_MRTL_COM",
    "household_income_band": "INC_TOT_COM",
    "smoking_status": "ICQ_SMOKE_COM",
    "diabetes": "DIA_DIAB_COM",
    "hypertension": "CCC_HBP_COM",
    "heart_disease": "CCC_HEART_COM",
    "stroke": "CCC_CVA_COM",
    "oa_hand": "CCC_OAHAND_COM",
    "oa_hip": "CCC_OAHIP_COM",
    "oa_knee": "CCC_OAKNEE_COM",
    "osteoporosis": "CCC_OSTPO_COM",
    "asthma": "CCC_ASTHM_COM",
    "copd": "CCC_COPD_COM",
    "cancer": "CCC_CANC_COM",
    "low_back_pain": "OST_BP_COM",
    "hearing_noise": "HRG_NOIS_COM",
    "hearing_aid": "HRG_AID_COM",
    "social_outside_household": "SPA_OUTS_COM",
    "social_religious": "SPA_CHRCH_COM",
    "social_education_culture": "SPA_EDUC_COM",
    "social_club": "SPA_CLUB_COM",
    "social_association": "SPA_NEIBR_COM",
    "social_other": "SPA_OTACT_COM",
    "adl_class": "ADL_DCLS_COM",
    "self_rated_healthy_aging": "GEN_OWNAG_COM",
    "analytic_weight": "WGHTS_ANALYTIC_COM",
    # The released field is GEOSTRATA_COM, not WGHTS_GEOSTRAT_COM.
    "sampling_strata": "GEOSTRATA_COM",
}

FOLLOWUP1_MAP = {
    "self_reported_vision": "VIS_SGHT_COF1",
    "visual_acuity_left": "VA_ETDRS_L_RSLT_COF1",
    "visual_acuity_right": "VA_ETDRS_R_RSLT_COF1",
    "cesd10_score": "DEP_CESD10_COF1",
    "age_years": "AGE_NMBR_COF1",
    "sex_at_birth": "SDC_BTHSEX_COF1",
    "ethnicity_spirometry": "SPR_OUTPUT_ETHN_COF1",
    "marital_status": "SDC_MRTL_COF1",
    "household_income_band": "INC_TOT_COF1",
    "smoking_status": "ICQ_SMOKE_COF1",
    "diabetes": "DIA_DIAB_COF1",
    "hypertension": "CCC_HBP_COF1",
    "heart_disease": "CCC_HEART_COF1",
    "stroke": "CCC_CVA_COF1",
    "oa_hand": "CCC_OAHAND_COF1",
    "oa_hip": "CCC_OAHIP_COF1",
    "oa_knee": "CCC_OAKNEE_COF1",
    "osteoporosis": "CCC_OSTPO_COF1",
    "asthma": "CCC_ASTHM_COF1",
    "copd": "CCC_COPD_COF1",
    "cancer": "CCC_CANC_COF1",
    "low_back_pain": "OST_BP_COF1",
    "hearing_noise": "HRG_NOIS_COF1",
    "hearing_aid": "HRG_AID_COF1",
    "social_outside_household": "SPA_OUTS_COF1",
    "social_religious": "SPA_CHRCH_COF1",
    "social_education_culture": "SPA_EDUC_COF1",
    "social_club": "SPA_CLUB_COF1",
    "social_association": "SPA_NEIBR_COF1",
    "social_other": "SPA_OTACT_COF1",
    "adl_class": "ADL_DCLS_COF1",
    "self_rated_healthy_aging": "GEN_OWNAG_COF1",
}

VISIT_METADATA = {
    "BL": {
        "questionnaire_start": "startdate_COM",
        "assessment_start": "ADM_startDate_COM",
        "age_source": "AGE_NMBR_COM",
        "provincial_weight": "WGHTS_PROV_COM",
        "epigenetic_indicator": "ADM_EPIGEN2_COM",
        "epigenetic_dnam": "DNAmAge_COM",
        "epigenetic_hannum": "Hannum_Age_COM",
    },
    "F1": {
        "questionnaire_start": "startdate_COF1",
        "assessment_start": "ADM_startDate_COF1",
        "age_source": "AGE_NMBR_COF1",
        "provincial_weight": "WGHTS_PROV_COF1",
    },
}

# COMMAND ----------
def require_columns(df, columns, label: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def assert_unique(df, columns, label: str) -> None:
    duplicates = (
        df.groupBy(*columns)
        .count()
        .filter(F.col("count") != 1)
    )
    if duplicates.limit(1).count():
        raise ValueError(f"{label} is not unique on {columns}.")


def project_visit(df, visit: str, source_map: dict[str, str]):
    metadata = VISIT_METADATA[visit]
    required = [
        "entity_id",
        metadata["questionnaire_start"],
        metadata["assessment_start"],
        metadata["age_source"],
        metadata["provincial_weight"],
        *source_map.values(),
    ]
    for optional_name in (
        "epigenetic_indicator",
        "epigenetic_dnam",
        "epigenetic_hannum",
    ):
        if optional_name in metadata:
            required.append(metadata[optional_name])
    require_columns(df, required, f"{visit} questionnaire")

    selections = [
        F.trim(F.col("entity_id").cast("string")).alias("participant_id"),
        F.lit(visit).alias("visit"),
    ]
    for standard_name in STANDARD_COLUMNS:
        source_name = source_map.get(standard_name)
        expression = (
            F.col(source_name).cast("string")
            if source_name
            else F.lit(None).cast("string")
        )
        selections.append(expression.alias(standard_name))

    def optional_source(metadata_name: str):
        source_name = metadata.get(metadata_name)
        return (
            F.col(source_name).cast("string")
            if source_name
            else F.lit(None).cast("string")
        )

    selections.extend(
        [
            F.col(metadata["questionnaire_start"])
            .cast("string")
            .alias("questionnaire_start_raw"),
            F.col(metadata["assessment_start"])
            .cast("string")
            .alias("assessment_start_raw"),
            F.lit(metadata["age_source"]).alias("age_source_variable"),
            F.col(metadata["provincial_weight"])
            .cast("string")
            .alias("provincial_weight_raw"),
            optional_source("epigenetic_indicator").alias(
                "epigenetic_data_indicator_raw"
            ),
            optional_source("epigenetic_dnam").alias(
                "epigenetic_dnam_age_raw"
            ),
            optional_source("epigenetic_hannum").alias(
                "epigenetic_hannum_age_raw"
            ),
        ]
    )
    return df.select(*selections)


baseline = project_visit(baseline_raw, "BL", BASELINE_MAP)
followup1 = project_visit(followup1_raw, "F1", FOLLOWUP1_MAP)
questionnaire_visit_raw = baseline.unionByName(followup1)
for acuity_column in ("visual_acuity_left", "visual_acuity_right"):
    questionnaire_visit_raw = questionnaire_visit_raw.withColumn(
        acuity_column,
        F.when(
            F.trim(F.col(acuity_column)) == "-88.8",
            F.lit(None).cast("string"),
        ).otherwise(F.col(acuity_column)),
    )
questionnaire_visit_raw = questionnaire_visit_raw.withColumn(
    "smoking_status",
    F.when(
        F.trim(F.col("smoking_status")).isin("-8", "8", "9"),
        F.lit(None).cast("string"),
    ).otherwise(F.col("smoking_status")),
)

# The fundus release layout uses seven-digit participant IDs. Exclude malformed
# CSV continuation rows and remove only exact duplicate projected records.
questionnaire_rows_read = questionnaire_visit_raw.count()
invalid_questionnaire_ids = questionnaire_visit_raw.filter(
    ~F.coalesce(F.col("participant_id"), F.lit("")).rlike(r"^\d{7}$")
).count()
questionnaire_visit_valid = questionnaire_visit_raw.filter(
    F.coalesce(F.col("participant_id"), F.lit("")).rlike(r"^\d{7}$")
)
questionnaire_valid_rows = questionnaire_visit_valid.count()
questionnaire_visit_raw = questionnaire_visit_valid.dropDuplicates()
questionnaire_distinct_rows = questionnaire_visit_raw.count()
exact_duplicate_rows_removed = (
    questionnaire_valid_rows - questionnaire_distinct_rows
)

conflicting_duplicate_keys = (
    questionnaire_visit_raw.groupBy("participant_id", "visit")
    .count()
    .filter(F.col("count") > 1)
)
conflicting_duplicate_key_count = conflicting_duplicate_keys.count()

questionnaire_qc_rows = [
    {"metric": "rows_read", "value": questionnaire_rows_read},
    {"metric": "invalid_or_malformed_ids", "value": invalid_questionnaire_ids},
    {"metric": "valid_id_rows", "value": questionnaire_valid_rows},
    {
        "metric": "exact_duplicate_rows_removed",
        "value": exact_duplicate_rows_removed,
    },
    {
        "metric": "conflicting_duplicate_participant_visits",
        "value": conflicting_duplicate_key_count,
    },
]
questionnaire_input_qc = spark.createDataFrame(
    questionnaire_qc_rows,
    schema="metric string, value long",
)
write_delta(
    questionnaire_input_qc,
    f"{output_root}/sap_questionnaire_input_qc",
)
display(questionnaire_input_qc)

if conflicting_duplicate_key_count:
    conflicting_duplicate_records = questionnaire_visit_raw.join(
        conflicting_duplicate_keys.select("participant_id", "visit"),
        ["participant_id", "visit"],
        "inner",
    )
    conflict_path = f"{output_root}/sap_questionnaire_duplicate_conflicts"
    write_delta(
        conflicting_duplicate_records,
        conflict_path,
        partition_by=("visit",),
    )
    raise ValueError(
        f"Found {conflicting_duplicate_key_count} participant-visits with "
        "genuinely different records after multiline-safe parsing and exact "
        f"deduplication. Review the governed audit: {conflict_path}"
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Visit dates and age provenance

# COMMAND ----------
require_columns(
    status_raw,
    ["entity_id", "clsa_baseline_date", "clsa_fup1_date"],
    "Participant-status release",
)
status_visit = status_raw.select(
    F.trim(F.col("entity_id").cast("string")).alias("participant_id"),
    F.col("clsa_baseline_date").cast("string").alias("status_bl_date_raw"),
    F.col("clsa_fup1_date").cast("string").alias("status_f1_date_raw"),
).filter(
    F.coalesce(F.col("participant_id"), F.lit("")).rlike(r"^\d{7}$")
).dropDuplicates()
assert_unique(status_visit, ["participant_id"], "Participant-status table")

questionnaire_visit_raw = questionnaire_visit_raw.join(
    status_visit,
    "participant_id",
    "left",
).withColumn(
    "status_visit_date_raw",
    F.when(F.col("visit") == "BL", F.col("status_bl_date_raw"))
    .when(F.col("visit") == "F1", F.col("status_f1_date_raw"))
    .otherwise(F.lit(None).cast("string")),
)


def try_parse_timestamp(column_name: str):
    escaped = column_name.replace("`", "``")
    value = f"`{escaped}`"
    formats = [
        None,
        "yyyy-MM-dd HH:mm:ss",
        "yyyy-MM-dd",
        "yyyy/MM/dd HH:mm:ss",
        "yyyy/MM/dd",
        "MM/dd/yyyy HH:mm:ss",
        "MM/dd/yyyy",
        "dd/MM/yyyy HH:mm:ss",
        "dd/MM/yyyy",
    ]
    expressions = []
    for date_format in formats:
        if date_format is None:
            expressions.append(F.expr(f"try_to_timestamp({value})"))
        else:
            expressions.append(
                F.expr(
                    f"try_to_timestamp({value}, '{date_format}')"
                )
            )
    return F.coalesce(*expressions)


questionnaire_visit_raw = (
    questionnaire_visit_raw
    .withColumn(
        "questionnaire_start_timestamp",
        try_parse_timestamp("questionnaire_start_raw"),
    )
    .withColumn(
        "assessment_start_timestamp",
        try_parse_timestamp("assessment_start_raw"),
    )
    .withColumn(
        "status_visit_timestamp",
        try_parse_timestamp("status_visit_date_raw"),
    )
    .withColumn(
        "fundus_visit_timestamp_proxy",
        F.coalesce(
            "assessment_start_timestamp",
            "questionnaire_start_timestamp",
            "status_visit_timestamp",
        ),
    )
    .withColumn(
        "fundus_visit_timestamp_proxy_source",
        F.when(
            F.col("assessment_start_timestamp").isNotNull(),
            F.lit("assessment_start"),
        )
        .when(
            F.col("questionnaire_start_timestamp").isNotNull(),
            F.lit("questionnaire_start"),
        )
        .when(
            F.col("status_visit_timestamp").isNotNull(),
            F.lit("participant_status_visit_date"),
        )
        .otherwise(F.lit("unavailable")),
    )
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. SAP derivations
# MAGIC
# MAGIC The visual-acuity threshold remains disabled unless the released scale
# MAGIC is explicitly confirmed as logMAR. Household income is retained as its
# MAGIC released band; it is not mislabeled as a quartile. The SAP's 32-condition
# MAGIC multimorbidity definition cannot be reconstructed from its abbreviated
# MAGIC variable list, so the derived count is explicitly labeled `selected`.

# COMMAND ----------
questionnaire_visit = derive_retinal_metrics(
    questionnaire_visit_raw,
    visual_acuity_scale_confirmed_logmar=visual_acuity_logmar_confirmed,
    require_both_eyes_for_better_eye=True,
)

questionnaire_visit = (
    questionnaire_visit
    .withColumn(
        "education_level_sap_harmonized",
        F.when(F.col("education_level_sap") == "1", F.lit(1))
        .when(F.col("education_level_sap").isin("2", "3"), F.lit(2))
        .when(F.col("education_level_sap") == "4", F.lit(3))
        .otherwise(F.lit(None).cast("int")),
    )
    .withColumn("income_quartile", F.lit(None).cast("int"))
    .withColumn(
        "income_quartile_status",
        F.lit(
            "not_derived: INC_TOT is a released category, not a continuous amount"
        ),
    )
    .withColumn(
        "epigenetic_dnam_age",
        F.when(
            F.trim(F.col("epigenetic_dnam_age_raw")).isin(
                "-8",
                "-77771",
                "-77772",
                "-88880",
                "-88888",
                "-99991",
                "-99993",
                "-99999",
            ),
            F.lit(None).cast("double"),
        ).otherwise(F.expr("try_cast(epigenetic_dnam_age_raw as double)")),
    )
    .withColumn(
        "epigenetic_hannum_age",
        F.when(
            F.trim(F.col("epigenetic_hannum_age_raw")).isin(
                "-8",
                "-77771",
                "-77772",
                "-88880",
                "-88888",
                "-99991",
                "-99993",
                "-99999",
            ),
            F.lit(None).cast("double"),
        ).otherwise(F.expr("try_cast(epigenetic_hannum_age_raw as double)")),
    )
    .withColumn(
        "epigenetic_age_status",
        F.when(
            F.col("epigenetic_dnam_age").isNotNull()
            | F.col("epigenetic_hannum_age").isNotNull(),
            F.lit("available_baseline_clock_outputs; clock must be specified"),
        ).otherwise(F.lit("not_available_at_this_visit")),
    )
    .withColumn(
        "frailty_status",
        F.lit("not_derived: SAP does not specify a frailty definition"),
    )
    .withColumn(
        "survey_design_status",
        F.when(
            F.col("visit") == "BL",
            F.lit("baseline analytic weight and GEOSTRATA_COM available"),
        ).otherwise(
            F.lit(
                "F1 has WGHTS_PROV_COF1 only; select an approved longitudinal weight"
            )
        ),
    )
    .withColumn("survey_psu", F.lit(None).cast("string"))
    .withColumn(
        "survey_psu_status",
        F.lit(
            "not identified in the released headers; participant_id is not "
            "assumed to be a sampling unit"
        ),
    )
    .withColumn(
        "age_at_fundus_years",
        F.col("age_years").cast("double"),
    )
    .withColumn(
        "age_at_fundus_source_variable",
        F.col("age_source_variable"),
    )
    .withColumn(
        "age_at_fundus_timing_basis",
        F.concat(
            F.lit("visit-matched released age; image visit="),
            F.col("visit"),
        ),
    )
    .withColumn(
        "exact_fundus_capture_timestamp_available",
        F.lit(0).cast("int"),
    )
)

# Preserve selected baseline-only context for F1 without pretending it was
# re-measured at F1.
baseline_context = questionnaire_visit.filter(F.col("visit") == "BL").select(
    "participant_id",
    F.col("sex_at_birth").alias("baseline_sex_at_birth"),
    F.col("education_level_sap_harmonized").alias(
        "baseline_education_level_sap_harmonized"
    ),
    F.col("analytic_weight").alias("baseline_analytic_weight"),
    F.col("sampling_strata").alias("baseline_sampling_strata"),
    F.col("epigenetic_dnam_age").alias("baseline_epigenetic_dnam_age"),
    F.col("epigenetic_hannum_age").alias(
        "baseline_epigenetic_hannum_age"
    ),
)
questionnaire_visit = questionnaire_visit.join(
    baseline_context,
    "participant_id",
    "left",
)

write_delta(
    questionnaire_visit,
    f"{output_root}/sap_questionnaire_visit",
    partition_by=("visit",),
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Link images to the matching visit age

# COMMAND ----------
if not Path(image_manifest_path).exists():
    raise FileNotFoundError(
        "The fundus image manifest is absent. Finish the extraction/image-manifest "
        f"phase of notebook 01 first: {image_manifest_path}"
    )

images_raw = spark.read.format("delta").load(image_manifest_path)
require_columns(
    images_raw,
    ["visit", "image_path"],
    "Fundus image manifest",
)
if "participant_id_parsed" in images_raw.columns:
    image_participant_column = "participant_id_parsed"
elif "participant_id" in images_raw.columns:
    image_participant_column = "participant_id"
else:
    raise ValueError(
        "Fundus image manifest must contain participant_id_parsed or "
        "participant_id."
    )

image_selections = [
    F.col(image_participant_column).cast("string").alias("participant_id"),
    F.when(F.upper(F.col("visit")).isin("F1", "FUP1"), F.lit("F1"))
    .when(F.upper(F.col("visit")) == "BL", F.lit("BL"))
    .otherwise(F.lit(None).cast("string"))
    .alias("visit"),
    F.col("image_path"),
]
for column_name in (
    "relative_path",
    "filename",
    "eye_parsed",
    "bytes",
    "width_px",
    "height_px",
    "sha256",
    "parse_ok",
):
    if column_name in images_raw.columns:
        image_selections.append(F.col(column_name))

images = images_raw.select(*image_selections)
invalid_images = images.filter(
    F.col("participant_id").isNull() | F.col("visit").isNull()
)
valid_images = images.filter(
    F.col("participant_id").isNotNull() & F.col("visit").isNotNull()
)

sap_fundus_image_linkage = (
    valid_images.join(
        questionnaire_visit,
        ["participant_id", "visit"],
        "left",
    )
    .withColumn(
        "image_age_link_status",
        F.when(
            F.col("age_at_fundus_years").isNotNull(),
            F.lit("visit_age_matched"),
        )
        .when(F.col("age_source_variable").isNotNull(), F.lit("visit_age_missing"))
        .otherwise(F.lit("questionnaire_visit_not_matched")),
    )
)

invalid_image_exclusions = invalid_images.withColumn(
    "image_age_link_status",
    F.when(
        F.col("participant_id").isNull(),
        F.lit("participant_id_unparsed"),
    ).otherwise(F.lit("unsupported_image_visit")),
)
age_link_exclusions = sap_fundus_image_linkage.filter(
    F.col("image_age_link_status") != "visit_age_matched"
)
sap_fundus_image_exclusions = invalid_image_exclusions.unionByName(
    age_link_exclusions,
    allowMissingColumns=True,
)
sap_fundus_images = sap_fundus_image_linkage.filter(
    F.col("image_age_link_status") == "visit_age_matched"
)

if not sap_fundus_images.limit(1).count():
    raise ValueError(
        "No fundus images have a visit-matched age after exclusions. Review "
        "sap_fundus_image_exclusions and questionnaire linkage."
    )

write_delta(
    sap_fundus_image_linkage,
    f"{output_root}/sap_fundus_image_linkage_audit",
    partition_by=("visit",),
)
write_delta(
    sap_fundus_image_exclusions,
    f"{output_root}/sap_fundus_image_exclusions",
)

write_delta(
    sap_fundus_images,
    f"{output_root}/sap_fundus_image_analysis",
    partition_by=("visit",),
)

display(
    sap_fundus_image_linkage.groupBy("visit", "image_age_link_status")
    .agg(
        F.count("*").alias("images"),
        F.countDistinct("participant_id").alias("participants"),
    )
    .orderBy("visit", "image_age_link_status")
)
display(
    sap_fundus_image_exclusions.groupBy("image_age_link_status")
    .agg(
        F.count("*").alias("excluded_images"),
        F.countDistinct("participant_id").alias("participants"),
    )
    .orderBy("image_age_link_status")
)
print(
    "Age-linked images retained:",
    sap_fundus_images.count(),
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Longitudinal age consistency checks
# MAGIC
# MAGIC This check compares the released age increment with elapsed time between
# MAGIC visit-date proxies. It is quality control only and does not manufacture
# MAGIC a more precise age than the released `AGE_NMBR` fields.

# COMMAND ----------
age_linkage_qc = questionnaire_visit.groupBy("participant_id").agg(
    F.first(
        F.when(F.col("visit") == "BL", F.col("age_at_fundus_years")),
        ignorenulls=True,
    ).alias("age_bl"),
    F.first(
        F.when(F.col("visit") == "F1", F.col("age_at_fundus_years")),
        ignorenulls=True,
    ).alias("age_f1"),
    F.first(
        F.when(
            F.col("visit") == "BL",
            F.col("fundus_visit_timestamp_proxy"),
        ),
        ignorenulls=True,
    ).alias("date_proxy_bl"),
    F.first(
        F.when(
            F.col("visit") == "F1",
            F.col("fundus_visit_timestamp_proxy"),
        ),
        ignorenulls=True,
    ).alias("date_proxy_f1"),
)

age_linkage_qc = (
    age_linkage_qc
    .withColumn("released_age_increment", F.col("age_f1") - F.col("age_bl"))
    .withColumn(
        "proxy_elapsed_years",
        F.months_between("date_proxy_f1", "date_proxy_bl") / F.lit(12.0),
    )
    .withColumn(
        "age_elapsed_difference_years",
        F.col("released_age_increment") - F.col("proxy_elapsed_years"),
    )
    .withColumn(
        "age_timing_qc_status",
        F.when(
            F.col("age_bl").isNull() | F.col("age_f1").isNull(),
            F.lit("insufficient_age_data"),
        )
        .when(F.col("released_age_increment") < 0, F.lit("review_age_decrease"))
        .when(
            F.col("proxy_elapsed_years").isNull(),
            F.lit("age_available_date_proxy_missing"),
        )
        .when(
            F.abs(F.col("age_elapsed_difference_years")) <= 1.0,
            F.lit("consistent_with_date_proxy"),
        )
        .otherwise(F.lit("review_age_date_difference")),
    )
)

write_delta(
    age_linkage_qc,
    f"{output_root}/sap_age_linkage_qc",
)
display(age_linkage_qc.groupBy("age_timing_qc_status").count())

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Variable provenance and availability audit

# COMMAND ----------
availability_rows = []
for visit, source_map in (("BL", BASELINE_MAP), ("F1", FOLLOWUP1_MAP)):
    for standard_name in STANDARD_COLUMNS:
        source_column = source_map.get(standard_name, "")
        status = "found" if source_column else "not_released_for_visit"
        note = ""
        if standard_name == "frailty":
            status = "definition_required"
            note = "SAP row is blank; no frailty variable was derived."
        elif standard_name == "epigenetic_age":
            status = (
                "clock_outputs_preserved_separately"
                if visit == "BL"
                else "not_released_for_visit"
            )
            note = "BL contains DNAmAge_COM and Hannum_Age_COM."
        elif standard_name == "analytic_weight" and visit == "F1":
            status = "approved_longitudinal_weight_required"
            note = "F1 contains WGHTS_PROV_COF1 but no analytic weight field."
        elif standard_name == "sampling_strata" and visit == "BL":
            note = "Actual released name is GEOSTRATA_COM."
        availability_rows.append(
            {
                "visit": visit,
                "standard_name": standard_name,
                "source_column": source_column,
                "status": status,
                "note": note,
            }
        )

availability_schema = (
    "visit string, standard_name string, source_column string, "
    "status string, note string"
)
availability = spark.createDataFrame(
    availability_rows,
    schema=availability_schema,
)
write_delta(
    availability,
    f"{output_root}/sap_variable_availability",
)
display(availability.orderBy("standard_name", "visit"))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Durable outputs
# MAGIC
# MAGIC - `sap_questionnaire_extraction_log`: exact extracted CSV provenance
# MAGIC - `sap_questionnaire_input_qc`: malformed-ID and duplicate-row counts
# MAGIC - `sap_questionnaire_visit`: one row per participant and visit
# MAGIC - `sap_fundus_image_linkage_audit`: every parseable BL/F1 image and its
# MAGIC   age-link status
# MAGIC - `sap_fundus_image_exclusions`: images excluded for an unparsed ID,
# MAGIC   unsupported visit, unmatched questionnaire visit, or missing age
# MAGIC - `sap_fundus_image_analysis`: only images with a visit-matched age
# MAGIC - `sap_age_linkage_qc`: BL/F1 released-age versus date-proxy checks
# MAGIC - `sap_variable_availability`: source-column and unresolved-item audit
# MAGIC - `sap_questionnaire_duplicate_conflicts`: written only when genuinely
# MAGIC   different records share a participant and visit
# MAGIC
# MAGIC Participant-level outputs remain in the governed Unity Catalog Volume
# MAGIC and must not be copied into Git.
