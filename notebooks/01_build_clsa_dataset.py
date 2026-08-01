# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Build and document the CLSA retinal-aging dataset
# MAGIC
# MAGIC The first half of this notebook is metadata-only. It inventories the
# MAGIC Unity Catalog Volume, reads ZIP central directories without passwords or
# MAGIC extraction, inspects questionnaire CSV headers, profiles the Follow-up 2
# MAGIC dictionary, audits genetics readiness, and writes a comprehensive
# MAGIC `DATASET_README.md`.
# MAGIC
# MAGIC Fundus extraction and participant-table construction are separate,
# MAGIC explicitly gated phases. Keep both extraction and questionnaire building
# MAGIC disabled for the first documentation run.

# COMMAND ----------

import json
from pathlib import Path
import re
import sys

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
    "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/derived/clsa_retinal_aging",
)
dbutils.widgets.text(
    "dictionary_path",
    "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/"
    "follow-up2_data_dictionaries_tracking_and_comprehensive_v2.xlsx",
)
dbutils.widgets.text(
    "genomics_root",
    "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/Genomics3_clsa",
)
dbutils.widgets.text(
    "genomics_readme_path",
    "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/"
    "Genomics3_clsa/README.md",
)
dbutils.widgets.dropdown(
    "inspect_zip_archives", "true", ["true", "false"]
)
dbutils.widgets.dropdown(
    "generate_dataset_readme", "true", ["true", "false"]
)
dbutils.widgets.dropdown(
    "extract_fundus_archives", "false", ["false", "true"]
)
dbutils.widgets.text(
    "archive_password", "", "Archive password (temporary; extraction only)"
)
dbutils.widgets.text("extraction_batch_size", "500")
dbutils.widgets.text("max_images_per_release_per_run", "5000")
dbutils.widgets.text("extraction_progress_every", "100")
dbutils.widgets.dropdown(
    "restart_fundus_extraction", "false", ["false", "true"]
)
dbutils.widgets.text("image_root", "")
dbutils.widgets.dropdown(
    "build_questionnaire", "false", ["false", "true"]
)
dbutils.widgets.text("tracking_data_path", "")
dbutils.widgets.text("comprehensive_data_path", "")
dbutils.widgets.text("participant_genetics_crosswalk_path", "")
dbutils.widgets.text("crosswalk_participant_id_column", "participant_id")
dbutils.widgets.text("crosswalk_gwas_id_column", "ADM_GWAS_COM")

# COMMAND ----------

repo_root = dbutils.widgets.get("repo_root").rstrip("/")
volume_root = dbutils.widgets.get("volume_root").rstrip("/")
output_root = dbutils.widgets.get("output_root").rstrip("/")
dictionary_path = dbutils.widgets.get("dictionary_path").strip()
genomics_root = dbutils.widgets.get("genomics_root").rstrip("/")
genomics_readme_path = dbutils.widgets.get("genomics_readme_path").strip()
inspect_archives = dbutils.widgets.get("inspect_zip_archives") == "true"
generate_readme = dbutils.widgets.get("generate_dataset_readme") == "true"
extract_archives = dbutils.widgets.get("extract_fundus_archives") == "true"
extraction_batch_size = int(dbutils.widgets.get("extraction_batch_size"))
max_images_per_release_per_run = int(
    dbutils.widgets.get("max_images_per_release_per_run")
)
extraction_progress_every = int(
    dbutils.widgets.get("extraction_progress_every")
)
restart_fundus_extraction = (
    dbutils.widgets.get("restart_fundus_extraction") == "true"
)
image_root = dbutils.widgets.get("image_root").strip()
build_questionnaire = dbutils.widgets.get("build_questionnaire") == "true"
tracking_data_path = dbutils.widgets.get("tracking_data_path").strip()
comprehensive_data_path = dbutils.widgets.get("comprehensive_data_path").strip()
crosswalk_path = dbutils.widgets.get(
    "participant_genetics_crosswalk_path"
).strip()
crosswalk_participant_id = dbutils.widgets.get(
    "crosswalk_participant_id_column"
)
crosswalk_gwas_id = dbutils.widgets.get("crosswalk_gwas_id_column")

src_path = f"{repo_root}/src"
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from clsa_dataset_inventory import (  # noqa: E402
    build_dataset_readme,
    inspect_zip_csv_headers,
    inspect_zip_releases,
    profile_dictionary_workbook,
)
from clsa_dataset_extraction import (  # noqa: E402
    iter_fundus_zip_release_batches,
)
from clsa_pipeline import (  # noqa: E402
    build_file_inventory,
    build_image_manifest,
    derive_retinal_metrics,
    dictionary_missing_code_map,
    harmonize_cohort,
    load_dictionary_sheets,
    load_json,
    load_variable_manifest,
    normalize_column_names,
    read_tabular_auto,
    read_whitespace_table,
    validate_genetics_inventory,
    write_delta,
)

config = load_json(f"{repo_root}/config/clsa_pipeline_config.json")
variable_manifest = load_variable_manifest(
    f"{repo_root}/config/variable_manifest.csv"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Release catalog
# MAGIC
# MAGIC These are the exact governed source paths for the current delivery.
# MAGIC Imaging archives contain encrypted JPEGs. Questionnaire archives are
# MAGIC inspected for member names and CSV headers only.

# COMMAND ----------

release_catalog = [
    {
        "name": "fundus_baseline",
        "role": "fundus_imaging",
        "visit": "BL",
        "path": f"{volume_root}/2209017_BL.zip",
    },
    {
        "name": "fundus_followup1",
        "role": "fundus_imaging",
        "visit": "F1",
        "path": f"{volume_root}/2209017_F1.zip",
    },
    {
        "name": "questionnaire_baseline",
        "role": "questionnaire",
        "visit": "BL",
        "path": f"{volume_root}/2209017_UOttawa_EFreeman_BL.zip",
    },
    {
        "name": "questionnaire_followup1",
        "role": "questionnaire",
        "visit": "FUP1",
        "path": f"{volume_root}/2209017_UOttawa_EFreeman_FUP1.zip",
    },
    {
        "name": "questionnaire_followup2",
        "role": "questionnaire",
        "visit": "FUP2",
        "path": f"{volume_root}/2209017_UOttawa_EFreeman_FUP2.zip",
    },
    {
        "name": "questionnaire_followup2_jun2025",
        "role": "questionnaire",
        "visit": "FUP2",
        "path": f"{volume_root}/2209017_UOttawa_EFreeman_FUP2_Jun2025.zip",
    },
    {
        "name": "mortality_aug2025",
        "role": "questionnaire",
        "visit": "MORTALITY",
        "path": (
            f"{volume_root}/"
            "2209017_UOttawa_EFreeman_Mortality_DRU_Aug2025.zip"
        ),
    },
]

missing_release_paths = [
    release["path"]
    for release in release_catalog
    if not Path(release["path"]).exists()
]
if missing_release_paths:
    raise FileNotFoundError(
        "Configured release paths are missing:\n"
        + "\n".join(missing_release_paths)
    )
if not Path(dictionary_path).exists():
    raise FileNotFoundError(f"Dictionary workbook not found: {dictionary_path}")
if not Path(genomics_root).exists():
    raise FileNotFoundError(f"Genomics directory not found: {genomics_root}")

release_catalog_df = spark.createDataFrame(release_catalog)
write_delta(release_catalog_df, f"{output_root}/release_catalog")
display(release_catalog_df.orderBy("role", "visit", "name"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Volume file inventory and genetics readiness

# COMMAND ----------

inventory = build_file_inventory(spark, volume_root).filter(
    ~F.col("path").startswith(output_root)
    & ~F.col("path").startswith(f"dbfs:{output_root}")
)
write_delta(inventory, f"{output_root}/file_inventory")

display(
    inventory.groupBy("extension")
    .agg(
        F.count("*").alias("files"),
        F.sum("bytes").alias("bytes"),
    )
    .orderBy(F.desc("bytes"))
)

genetics_readiness = validate_genetics_inventory(
    inventory,
    config["genetics"]["expected_chromosomes"],
)
print(json.dumps(genetics_readiness, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Metadata-only ZIP inspection
# MAGIC
# MAGIC This reads central directories only. It does not need the archive
# MAGIC password and does not extract images or participant datasets.

# COMMAND ----------

zip_member_rows = []
zip_summary_rows = []
csv_schema_rows = []

if inspect_archives:
    zip_member_rows, zip_summary_rows = inspect_zip_releases(
        release_catalog
    )
    csv_schema_rows = inspect_zip_csv_headers(release_catalog)

    zip_members_df = spark.createDataFrame(zip_member_rows)
    zip_summary_df = spark.createDataFrame(zip_summary_rows)
    csv_schemas_df = spark.createDataFrame(csv_schema_rows)

    write_delta(
        zip_members_df,
        f"{output_root}/zip_member_inventory",
        partition_by=("release_name",),
    )
    write_delta(
        zip_summary_df,
        f"{output_root}/zip_archive_summary",
    )
    write_delta(
        csv_schemas_df,
        f"{output_root}/questionnaire_csv_schemas",
        partition_by=("release_name",),
    )

    display(
        zip_summary_df.select(
            "release_name",
            "role",
            "visit",
            "file_count",
            "uncompressed_bytes",
            "image_count",
            "tabular_count",
            "dictionary_count",
            "encrypted_count",
            "possible_aes_count",
            "participant_count",
            "complete_eye_pair_count",
        ).orderBy("role", "visit")
    )
    display(
        csv_schemas_df.select(
            "release_name",
            "visit",
            "member_path",
            "column_count",
            "inspection_error",
        ).orderBy("release_name", "member_path")
    )
else:
    print(
        "ZIP inspection disabled. Enable it before generating the dataset README."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Dictionary metadata

# COMMAND ----------

dictionary_profile = profile_dictionary_workbook(dictionary_path)
print(json.dumps(dictionary_profile, indent=2))

variables_dictionary, categories_dictionary = load_dictionary_sheets(
    spark, dictionary_path
)
write_delta(
    variables_dictionary,
    f"{output_root}/dictionary_variables",
)
write_delta(
    categories_dictionary,
    f"{output_root}/dictionary_categories",
)
write_delta(
    dictionary_missing_code_map(categories_dictionary),
    f"{output_root}/dictionary_missing_codes",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Genetics metadata tables
# MAGIC
# MAGIC Indexes are not treated as genotype payloads. `direct_ready` requires
# MAGIC BED/BIM/FAM, and `imputed_ready` requires every BGEN/BGI chromosome pair.

# COMMAND ----------

sample_qc_candidates = [
    row["path"]
    for row in inventory.filter(
        F.lower("name") == F.lit("clsa_sqc_v3.txt")
    ).select("path").collect()
]
hla_candidates = [
    row["path"]
    for row in inventory.filter(
        F.lower("name") == F.lit("clsa_hla_v3.csv")
    ).select("path").collect()
]

genetics_sample_qc = None
if len(sample_qc_candidates) == 1:
    genetics_sample_qc = normalize_column_names(
        read_whitespace_table(spark, sample_qc_candidates[0])
    )
    sqc_id_column = next(
        (
            column
            for column in genetics_sample_qc.columns
            if column.casefold() == "adm_gwas_com"
        ),
        genetics_sample_qc.columns[0],
    )
    genetics_sample_qc = genetics_sample_qc.withColumnRenamed(
        sqc_id_column,
        "ADM_GWAS_COM",
    )
    duplicates = (
        genetics_sample_qc.groupBy("ADM_GWAS_COM")
        .count()
        .filter(F.col("count") != 1)
    )
    if duplicates.limit(1).count():
        raise ValueError(
            "Sample QC is not one row per ADM_GWAS_COM."
        )
    write_delta(
        genetics_sample_qc,
        f"{output_root}/genetics_sample_qc",
    )
elif len(sample_qc_candidates) > 1:
    raise ValueError("Multiple clsa_sqc_v3.txt files were found.")

if len(hla_candidates) == 1:
    with open(
        hla_candidates[0],
        encoding="utf-8",
        errors="replace",
    ) as stream:
        hla_header = stream.readline()
    hla_raw = (
        spark.read.option("header", "true").csv(hla_candidates[0])
        if "," in hla_header
        else read_whitespace_table(spark, hla_candidates[0])
    )
    hla = normalize_column_names(hla_raw)
    hla_id_column = next(
        (
            column
            for column in hla.columns
            if column.casefold() == "adm_gwas_com"
        ),
        hla.columns[0],
    )
    hla = hla.withColumnRenamed(
        hla_id_column,
        "ADM_GWAS_COM",
    )
    write_delta(hla, f"{output_root}/genetics_hla")
elif len(hla_candidates) > 1:
    raise ValueError("Multiple clsa_hla_v3.csv files were found.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Generate the comprehensive dataset README

# COMMAND ----------

inventory_rows = [
    row.asDict(recursive=True)
    for row in inventory.orderBy("relative_path").collect()
]
genomics_readme_text = ""
if genomics_readme_path and Path(genomics_readme_path).exists():
    genomics_readme_text = Path(genomics_readme_path).read_text(
        encoding="utf-8",
        errors="replace",
    )

if generate_readme:
    if not zip_summary_rows:
        raise ValueError(
            "README generation requires inspect_zip_archives=true."
        )
    dataset_readme = build_dataset_readme(
        volume_root=volume_root,
        output_root=output_root,
        file_inventory=inventory_rows,
        releases=release_catalog,
        zip_members=zip_member_rows,
        zip_summaries=zip_summary_rows,
        csv_schemas=csv_schema_rows,
        dictionary_profile=dictionary_profile,
        genetics_readiness=genetics_readiness,
        genomics_readme_text=genomics_readme_text,
    )
    dataset_readme_path = (
        Path(output_root) / "DATASET_README.md"
    )
    dataset_readme_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    dataset_readme_path.write_text(
        dataset_readme,
        encoding="utf-8",
    )
    print("Generated:", dataset_readme_path)
    print("README bytes:", dataset_readme_path.stat().st_size)
    print("\n".join(dataset_readme.splitlines()[:160]))
else:
    print("Dataset README generation disabled.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Optional full fundus extraction
# MAGIC
# MAGIC Keep this disabled for documentation-only runs. The two imaging archives
# MAGIC expand to approximately 55.6 GiB. Extraction is restartable and preserves
# MAGIC release and visit metadata. Work is committed to Delta after every
# MAGIC bounded batch, and a per-release checkpoint resumes at the next ZIP
# MAGIC member after a cluster restart. With the default limit, each run extracts
# MAGIC at most 5,000 images from each release; rerun this section until both
# MAGIC releases report `complete`. Set the limit to `0` only for an unbounded
# MAGIC run. The temporary password is cleared from Python memory when extraction
# MAGIC finishes.

# COMMAND ----------

image_manifest = None
extraction_manifest = None
if extract_archives:
    from delta.tables import DeltaTable
    from pyspark.sql.types import (
        LongType,
        StringType,
        StructField,
        StructType,
    )

    archive_password = dbutils.widgets.get("archive_password")
    if not archive_password:
        raise ValueError(
            "Enter the archive password in the temporary widget."
        )
    if extraction_batch_size < 1:
        raise ValueError("extraction_batch_size must be at least 1.")
    if max_images_per_release_per_run < 0:
        raise ValueError(
            "max_images_per_release_per_run cannot be negative."
        )
    if extraction_progress_every < 1:
        raise ValueError("extraction_progress_every must be at least 1.")

    extraction_schema = StructType(
        [
            StructField("release_name", StringType(), False),
            StructField("visit", StringType(), False),
            StructField("archive_path", StringType(), False),
            StructField("member_path", StringType(), False),
            StructField("output_path", StringType(), True),
            StructField("participant_id", StringType(), True),
            StructField("eye", StringType(), True),
            StructField("uncompressed_bytes", LongType(), False),
            StructField("crc32", StringType(), False),
            StructField("status", StringType(), False),
            StructField("error", StringType(), False),
            StructField("member_index", LongType(), False),
        ]
    )

    extracted_root = Path(output_root) / "fundus_extracted"
    extraction_manifest_path = f"{output_root}/fundus_extraction_manifest"
    checkpoint_root = Path(output_root) / "fundus_extraction_checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    def delta_manifest_exists():
        return DeltaTable.isDeltaTable(spark, extraction_manifest_path)

    def upsert_extraction_batch(rows):
        batch_df = spark.createDataFrame(rows, schema=extraction_schema)
        if not delta_manifest_exists():
            write_delta(
                batch_df,
                extraction_manifest_path,
                partition_by=("visit",),
            )
            return

        target = DeltaTable.forPath(spark, extraction_manifest_path)
        if "member_index" not in target.toDF().columns:
            spark.sql(
                f"ALTER TABLE delta.`{extraction_manifest_path}` "
                "ADD COLUMNS (member_index BIGINT)"
            )
            target = DeltaTable.forPath(spark, extraction_manifest_path)
        (
            target.alias("target")
            .merge(
                batch_df.alias("source"),
                "target.archive_path = source.archive_path AND "
                "target.member_path = source.member_path",
            )
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )

    def load_release_checkpoint(release, checkpoint_path):
        if restart_fundus_extraction or not delta_manifest_exists():
            return 0
        if not checkpoint_path.exists():
            return 0
        try:
            checkpoint = json.loads(
                checkpoint_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            print(
                f"[{release['name']}] checkpoint is unreadable; "
                "restarting at image 0.",
                flush=True,
            )
            return 0
        archive_stat = Path(release["path"]).stat()
        archive_matches = (
            checkpoint.get("archive_path") == release["path"]
            and checkpoint.get("archive_bytes") == archive_stat.st_size
            and checkpoint.get("archive_mtime_ns") == archive_stat.st_mtime_ns
        )
        if not archive_matches:
            print(
                f"[{release['name']}] archive changed; restarting at image 0.",
                flush=True,
            )
            return 0
        return int(checkpoint.get("next_member_index", 0))

    def save_release_checkpoint(
        release,
        checkpoint_path,
        next_member_index,
    ):
        archive_stat = Path(release["path"]).stat()
        payload = {
            "release_name": release["name"],
            "visit": release["visit"],
            "archive_path": release["path"],
            "archive_bytes": archive_stat.st_size,
            "archive_mtime_ns": archive_stat.st_mtime_ns,
            "next_member_index": int(next_member_index),
        }
        temporary_path = checkpoint_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(checkpoint_path)

    try:
        for release in release_catalog:
            if release["role"] != "fundus_imaging":
                continue
            release_output = extracted_root / release["visit"]
            checkpoint_path = checkpoint_root / (
                f"{release['name']}.json"
            )
            start_member_index = load_release_checkpoint(
                release,
                checkpoint_path,
            )
            run_limit = (
                max_images_per_release_per_run
                if max_images_per_release_per_run > 0
                else None
            )
            print(
                f"\n[{release['name']}] checkpoint starts at image "
                f"{start_member_index:,}.",
                flush=True,
            )
            for rows in iter_fundus_zip_release_batches(
                release["path"],
                str(release_output),
                archive_password,
                release_name=release["name"],
                visit=release["visit"],
                overwrite=False,
                batch_size=extraction_batch_size,
                start_member_index=start_member_index,
                max_members=run_limit,
                progress_every=extraction_progress_every,
                progress_callback=lambda message: print(
                    message,
                    flush=True,
                ),
            ):
                upsert_extraction_batch(rows)
                failed_rows = [
                    row for row in rows if row["status"] == "failed"
                ]
                if failed_rows:
                    first_failure = failed_rows[0]
                    raise RuntimeError(
                        f"[{release['name']}] extraction batch had "
                        f"{len(failed_rows)} failure(s); the checkpoint was "
                        "not advanced. First failure: "
                        f"{first_failure['member_path']}: "
                        f"{first_failure['error']}"
                    )
                next_member_index = rows[-1]["member_index"] + 1
                save_release_checkpoint(
                    release,
                    checkpoint_path,
                    next_member_index,
                )
                print(
                    f"[{release['name']}] committed checkpoint through "
                    f"image {next_member_index:,}.",
                    flush=True,
                )
    finally:
        archive_password = ""

    if delta_manifest_exists():
        extraction_manifest = spark.read.format("delta").load(
            extraction_manifest_path
        )
    image_root = str(extracted_root)

if image_root:
    if extraction_manifest is not None:
        image_root_pattern = rf"^{re.escape(image_root.rstrip('/') + '/')}"
        image_manifest = (
            extraction_manifest.filter(
                F.col("status").isin("extracted", "already_present")
                & F.col("output_path").isNotNull()
            )
            .select(
                F.col("output_path").alias("image_path"),
                F.regexp_replace(
                    "output_path",
                    image_root_pattern,
                    "",
                ).alias("relative_path"),
                F.element_at(
                    F.split("output_path", "/"),
                    -1,
                ).alias("filename"),
                F.regexp_extract(
                    F.lower("output_path"),
                    r"(\.[^.]+)$",
                    1,
                ).alias("extension"),
                F.col("uncompressed_bytes").alias("bytes"),
                F.lit(None).cast("timestamp").alias("modified_utc"),
                F.col("participant_id").alias(
                    "participant_id_parsed"
                ),
                F.col("eye").alias("eye_parsed"),
                F.lit(None).cast("int").alias("width_px"),
                F.lit(None).cast("int").alias("height_px"),
                F.lit(None).cast("string").alias("sha256"),
                F.col("participant_id").isNotNull().alias("parse_ok"),
                "visit",
            )
            .dropDuplicates(["image_path"])
        )
        print(
            "Building image manifest directly from checkpointed extraction "
            "rows (no driver-side directory crawl).",
            flush=True,
        )
    else:
        image_manifest = build_image_manifest(
            spark,
            image_root,
            participant_id_regex=(
                r"(?i)(?:^|/)(?P<participant_id>\d{7})"
                r"(?=/retinal_(?:left|right)\.jpeg$)"
            ),
            eye_regex=config["eye_regex"],
            probe_dimensions=config["probe_image_dimensions"],
            compute_sha256=config["compute_image_sha256"],
        ).withColumn(
            "visit",
            F.when(
                F.col("relative_path").contains("2209017_BL"),
                F.lit("BL"),
            )
            .when(
                F.col("relative_path").contains("2209017_F1"),
                F.lit("F1"),
            )
            .otherwise(F.lit(None).cast("string")),
        )
    write_delta(
        image_manifest,
        f"{output_root}/fundus_image_manifest",
        partition_by=("visit", "extension"),
    )
    display(
        image_manifest.groupBy("visit").agg(
            F.count("*").alias("images"),
            F.sum(
                (~F.col("parse_ok")).cast("int")
            ).alias("unparsed_ids"),
            F.countDistinct(
                "participant_id_parsed"
            ).alias("participants"),
        )
    )
else:
    print(
        "Fundus extraction disabled; metadata inventory and README are complete."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Optional questionnaire harmonization
# MAGIC
# MAGIC The archive inventory identifies the CSV member names. Extract the chosen
# MAGIC Tracking and Comprehensive participant files into governed,
# MAGIC release-specific directories, then provide their explicit paths in the
# MAGIC widgets before enabling this phase.

# COMMAND ----------

questionnaire = None
mapping_audits = []
if build_questionnaire:
    cohort_frames = []
    if tracking_data_path:
        tracking_raw = read_tabular_auto(spark, tracking_data_path)
        tracking_harmonized, tracking_audit = harmonize_cohort(
            tracking_raw,
            cohort="tracking",
            manifest=variable_manifest,
            id_candidates=config["participant_id_candidates"],
        )
        cohort_frames.append(tracking_harmonized)
        mapping_audits.extend(tracking_audit)
    if comprehensive_data_path:
        comprehensive_raw = read_tabular_auto(
            spark,
            comprehensive_data_path,
        )
        comprehensive_harmonized, comprehensive_audit = harmonize_cohort(
            comprehensive_raw,
            cohort="comprehensive",
            manifest=variable_manifest,
            id_candidates=config["participant_id_candidates"],
        )
        cohort_frames.append(comprehensive_harmonized)
        mapping_audits.extend(comprehensive_audit)
    if not cohort_frames:
        raise ValueError(
            "Provide tracking_data_path and/or comprehensive_data_path."
        )

    questionnaire = cohort_frames[0]
    for frame in cohort_frames[1:]:
        questionnaire = questionnaire.unionByName(
            frame,
            allowMissingColumns=True,
        )
    questionnaire = derive_retinal_metrics(
        questionnaire,
        visual_acuity_scale_confirmed_logmar=config[
            "visual_acuity_scale_confirmed_logmar"
        ],
        require_both_eyes_for_better_eye=config[
            "require_both_eyes_for_better_eye"
        ],
    )
    if questionnaire.filter(
        F.col("participant_id").isNull()
    ).limit(1).count():
        raise ValueError(
            "Participant IDs contain nulls after harmonization."
        )
    duplicates = (
        questionnaire.groupBy("cohort", "participant_id")
        .count()
        .filter(F.col("count") != 1)
    )
    if duplicates.limit(1).count():
        raise ValueError(
            "Participant data are not unique within cohort."
        )
    write_delta(
        questionnaire,
        f"{output_root}/questionnaire_retinal_metrics",
        partition_by=("cohort",),
    )
    mapping_audit_df = spark.createDataFrame(mapping_audits)
    write_delta(
        mapping_audit_df,
        f"{output_root}/variable_mapping_audit",
    )
    display(
        mapping_audit_df.groupBy("cohort", "status").count()
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Optional authorized linkage
# MAGIC
# MAGIC A project-authorized crosswalk is required to link participant IDs to
# MAGIC `ADM_GWAS_COM`. Never infer linkage from row order or approximate IDs.

# COMMAND ----------

if questionnaire is not None:
    master = questionnaire

    if image_manifest is not None:
        image_by_participant = image_manifest.groupBy(
            F.col("participant_id_parsed").alias("participant_id")
        ).agg(
            F.count("*").alias("fundus_image_count"),
            F.collect_set("visit").alias("fundus_visits"),
            F.collect_set("eye_parsed").alias("fundus_eyes"),
            F.collect_list(
                F.struct(
                    "image_path",
                    "visit",
                    "eye_parsed",
                    "filename",
                )
            ).alias("fundus_images"),
        )
        master = master.join(
            image_by_participant,
            "participant_id",
            "left",
        )

    if crosswalk_path:
        crosswalk = read_tabular_auto(
            spark,
            crosswalk_path,
        ).select(
            F.col(crosswalk_participant_id)
            .cast("string")
            .alias("participant_id"),
            F.col(crosswalk_gwas_id)
            .cast("string")
            .alias("ADM_GWAS_COM"),
        )
        duplicate_participant = (
            crosswalk.groupBy("participant_id")
            .count()
            .filter(F.col("count") != 1)
        )
        duplicate_gwas = (
            crosswalk.groupBy("ADM_GWAS_COM")
            .count()
            .filter(F.col("count") != 1)
        )
        if (
            duplicate_participant.limit(1).count()
            or duplicate_gwas.limit(1).count()
        ):
            raise ValueError(
                "The participant/genetics crosswalk is not one-to-one."
            )
        master = master.join(
            crosswalk,
            "participant_id",
            "left",
        )
        if genetics_sample_qc is not None:
            master = master.join(
                genetics_sample_qc,
                "ADM_GWAS_COM",
                "left",
            )
    else:
        master = master.withColumn(
            "ADM_GWAS_COM",
            F.lit(None).cast("string"),
        )

    master = (
        master.withColumn(
            "has_fundus_image",
            (
                F.coalesce(
                    F.col("fundus_image_count"),
                    F.lit(0),
                )
                > 0
            ).cast("int")
            if "fundus_image_count" in master.columns
            else F.lit(0),
        )
        .withColumn(
            "has_genetics_link",
            F.col("ADM_GWAS_COM").isNotNull().cast("int"),
        )
        .withColumn(
            "analysis_complete_case",
            (
                (F.col("has_fundus_image") == 1)
                & (F.col("has_genetics_link") == 1)
                & F.col("age_years").isNotNull()
                & F.col("sex_at_birth").isNotNull()
            ).cast("int"),
        )
    )
    write_delta(
        master,
        f"{output_root}/participant_analysis_master",
        partition_by=("cohort",),
    )
    display(
        master.groupBy("cohort").agg(
            F.count("*").alias("participants"),
            F.sum("has_fundus_image").alias("with_fundus"),
            F.sum("has_genetics_link").alias("with_genetics_link"),
            F.sum("analysis_complete_case").alias("complete_case"),
        )
    )
