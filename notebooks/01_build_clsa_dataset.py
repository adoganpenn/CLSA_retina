# Databricks notebook source
# MAGIC %md
# MAGIC # CLSA retinal-aging extraction pipeline
# MAGIC
# MAGIC Run this notebook in phases. The first run should inventory the Volume
# MAGIC with extraction disabled. Supply explicit paths only after reviewing the
# MAGIC inventory candidates. Archive passwords are read from a Databricks secret
# MAGIC scope and never written to widgets, logs, or Delta tables.

# COMMAND ----------
import json
from pathlib import Path
import sys

from pyspark.sql import functions as F

# COMMAND ----------
dbutils.widgets.text("repo_root", "/Workspace/Repos/<user>/<repo>")
dbutils.widgets.text(
    "volume_root", "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset"
)
dbutils.widgets.text(
    "output_root",
    "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/derived/clsa_retinal_aging",
)
dbutils.widgets.text("dictionary_path", "")
dbutils.widgets.text("tracking_data_path", "")
dbutils.widgets.text("comprehensive_data_path", "")
dbutils.widgets.text("image_root", "")
dbutils.widgets.dropdown("extract_fundus_archives", "false", ["false", "true"])
dbutils.widgets.dropdown("build_questionnaire", "false", ["false", "true"])
dbutils.widgets.text("participant_genetics_crosswalk_path", "")
dbutils.widgets.text("crosswalk_participant_id_column", "participant_id")
dbutils.widgets.text("crosswalk_gwas_id_column", "ADM_GWAS_COM")

# COMMAND ----------
repo_root = dbutils.widgets.get("repo_root").rstrip("/")
volume_root = dbutils.widgets.get("volume_root").rstrip("/")
output_root = dbutils.widgets.get("output_root").rstrip("/")
dictionary_path = dbutils.widgets.get("dictionary_path").strip()
tracking_data_path = dbutils.widgets.get("tracking_data_path").strip()
comprehensive_data_path = dbutils.widgets.get("comprehensive_data_path").strip()
image_root = dbutils.widgets.get("image_root").strip()
extract_archives = dbutils.widgets.get("extract_fundus_archives") == "true"
build_questionnaire = dbutils.widgets.get("build_questionnaire") == "true"
crosswalk_path = dbutils.widgets.get("participant_genetics_crosswalk_path").strip()
crosswalk_participant_id = dbutils.widgets.get("crosswalk_participant_id_column")
crosswalk_gwas_id = dbutils.widgets.get("crosswalk_gwas_id_column")

sys.path.insert(0, f"{repo_root}/src")

from clsa_pipeline import (  # noqa: E402
    build_file_inventory,
    build_image_manifest,
    derive_retinal_metrics,
    dictionary_missing_code_map,
    discover_paths,
    extract_fundus_archives_from_secret,
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
manifest = load_variable_manifest(f"{repo_root}/config/variable_manifest.csv")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Inventory and genetics readiness

# COMMAND ----------
inventory = (
    build_file_inventory(spark, volume_root)
    .filter(
        ~F.col("path").startswith(output_root)
        & ~F.col("path").startswith(f"dbfs:{output_root}")
    )
    .cache()
)
write_delta(inventory, f"{output_root}/file_inventory")

display(
    inventory.groupBy("extension")
    .agg(F.count("*").alias("files"), F.sum("bytes").alias("bytes"))
    .orderBy(F.desc("bytes"))
)

dictionary_candidates = discover_paths(
    inventory, config["dictionary_filename_regex"]
)
tracking_candidates = discover_paths(inventory, config["tracking_data_regex"])
comprehensive_candidates = discover_paths(
    inventory, config["comprehensive_data_regex"]
)
archive_candidates = discover_paths(inventory, config["fundus_archive_regex"])
sample_qc_candidates = discover_paths(
    inventory, r"(?i)(^|/)clsa_sqc_v3\.txt$"
)
hla_candidates = discover_paths(
    inventory, r"(?i)(^|/)clsa_hla_v3\.csv$"
)

print(
    json.dumps(
        {
            "dictionary_candidates": dictionary_candidates,
            "tracking_candidates": tracking_candidates,
            "comprehensive_candidates": comprehensive_candidates,
            "fundus_archive_candidates": archive_candidates,
            "sample_qc_candidates": sample_qc_candidates,
            "hla_candidates": hla_candidates,
            "genetics_readiness": validate_genetics_inventory(
                inventory, config["genetics"]["expected_chromosomes"]
            ),
        },
        indent=2,
    )
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1b. Genetics metadata tables

# COMMAND ----------
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
        sqc_id_column, "ADM_GWAS_COM"
    )
    duplicate_sqc = (
        genetics_sample_qc.groupBy("ADM_GWAS_COM")
        .count()
        .filter(F.col("count") != 1)
    )
    if duplicate_sqc.limit(1).count():
        raise ValueError("Sample QC is not one row per ADM_GWAS_COM.")
    write_delta(genetics_sample_qc, f"{output_root}/genetics_sample_qc")
elif len(sample_qc_candidates) > 1:
    raise ValueError("Multiple clsa_sqc_v3.txt files found; resolve explicitly.")

if len(hla_candidates) == 1:
    with open(hla_candidates[0], encoding="utf-8", errors="replace") as stream:
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
    hla = hla.withColumnRenamed(hla_id_column, "ADM_GWAS_COM")
    write_delta(hla, f"{output_root}/genetics_hla")
elif len(hla_candidates) > 1:
    raise ValueError("Multiple clsa_hla_v3.csv files found; resolve explicitly.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Optional encrypted fundus extraction

# COMMAND ----------
if extract_archives:
    if not archive_candidates:
        raise ValueError("No fundus archive candidate was found in the Volume.")
    extracted_root = f"{output_root}/fundus_extracted"
    extraction_rows = extract_fundus_archives_from_secret(
        dbutils,
        archive_candidates,
        extracted_root,
        secret_scope=config["fundus_secret"]["scope"],
        secret_key=config["fundus_secret"]["key"],
        overwrite=False,
    )
    extraction_manifest = spark.createDataFrame(extraction_rows)
    write_delta(extraction_manifest, f"{output_root}/fundus_extraction_manifest")
    image_root = extracted_root
elif not image_root:
    print(
        "Fundus extraction is disabled. Set image_root to an existing image "
        "directory or enable extraction after reviewing archive candidates."
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Fundus image manifest

# COMMAND ----------
image_manifest = None
if image_root:
    image_manifest = build_image_manifest(
        spark,
        image_root,
        participant_id_regex=config["participant_id_regex"],
        eye_regex=config["eye_regex"],
        probe_dimensions=config["probe_image_dimensions"],
        compute_sha256=config["compute_image_sha256"],
    )
    write_delta(
        image_manifest,
        f"{output_root}/fundus_image_manifest",
        partition_by=("extension",),
    )
    display(
        image_manifest.agg(
            F.count("*").alias("images"),
            F.sum((~F.col("parse_ok")).cast("int")).alias("unparsed_ids"),
            F.countDistinct("participant_id_parsed").alias("parsed_participants"),
        )
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Dictionary metadata and cohort harmonization

# COMMAND ----------
if not dictionary_path:
    if len(dictionary_candidates) == 1:
        dictionary_path = dictionary_candidates[0]
    elif dictionary_candidates:
        raise ValueError(
            "Multiple dictionary candidates found. Set dictionary_path explicitly."
        )
    else:
        raise ValueError(
            "The Follow-up 2 dictionary was not found. Upload it to the Volume "
            "and set dictionary_path."
        )

variables_dictionary, categories_dictionary = load_dictionary_sheets(
    spark, dictionary_path
)
write_delta(variables_dictionary, f"{output_root}/dictionary_variables")
write_delta(categories_dictionary, f"{output_root}/dictionary_categories")
write_delta(
    dictionary_missing_code_map(categories_dictionary),
    f"{output_root}/dictionary_missing_codes",
)

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
            manifest=manifest,
            id_candidates=config["participant_id_candidates"],
        )
        cohort_frames.append(tracking_harmonized)
        mapping_audits.extend(tracking_audit)
    if comprehensive_data_path:
        comprehensive_raw = read_tabular_auto(spark, comprehensive_data_path)
        comprehensive_harmonized, comprehensive_audit = harmonize_cohort(
            comprehensive_raw,
            cohort="comprehensive",
            manifest=manifest,
            id_candidates=config["participant_id_candidates"],
        )
        cohort_frames.append(comprehensive_harmonized)
        mapping_audits.extend(comprehensive_audit)
    if not cohort_frames:
        raise ValueError(
            "Set at least one of tracking_data_path or comprehensive_data_path. "
            "The dictionary workbook describes variables but contains no participant rows."
        )

    questionnaire = cohort_frames[0]
    for frame in cohort_frames[1:]:
        questionnaire = questionnaire.unionByName(frame, allowMissingColumns=True)
    questionnaire = derive_retinal_metrics(
        questionnaire,
        visual_acuity_scale_confirmed_logmar=config[
            "visual_acuity_scale_confirmed_logmar"
        ],
        require_both_eyes_for_better_eye=config[
            "require_both_eyes_for_better_eye"
        ],
    )
    if questionnaire.filter(F.col("participant_id").isNull()).limit(1).count():
        raise ValueError("Participant IDs contain null values after harmonization.")
    duplicate_questionnaire = (
        questionnaire.groupBy("cohort", "participant_id")
        .count()
        .filter(F.col("count") != 1)
    )
    if duplicate_questionnaire.limit(1).count():
        raise ValueError(
            "Participant data are not one row per participant within cohort."
        )
    write_delta(
        questionnaire,
        f"{output_root}/questionnaire_retinal_metrics",
        partition_by=("cohort",),
    )
    mapping_audit_df = spark.createDataFrame(mapping_audits)
    write_delta(mapping_audit_df, f"{output_root}/variable_mapping_audit")
    display(mapping_audit_df.groupBy("cohort", "status").count())

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Authorized linkage and participant-level analysis table
# MAGIC
# MAGIC The crosswalk must map the project-specific participant ID to
# MAGIC `ADM_GWAS_COM`. Do not infer this linkage from row order or approximate
# MAGIC identifiers.

# COMMAND ----------
if questionnaire is not None:
    master = questionnaire

    if image_manifest is not None:
        image_by_participant = image_manifest.groupBy(
            F.col("participant_id_parsed").alias("participant_id")
        ).agg(
            F.count("*").alias("fundus_image_count"),
            F.collect_set("eye_parsed").alias("fundus_eyes"),
            F.collect_list(
                F.struct("image_path", "eye_parsed", "filename")
            ).alias("fundus_images"),
        )
        master = master.join(image_by_participant, "participant_id", "left")

    if crosswalk_path:
        crosswalk = read_tabular_auto(spark, crosswalk_path).select(
            F.col(crosswalk_participant_id).cast("string").alias("participant_id"),
            F.col(crosswalk_gwas_id).cast("string").alias("ADM_GWAS_COM"),
        )
        duplicate_participant = (
            crosswalk.groupBy("participant_id").count().filter(F.col("count") != 1)
        )
        duplicate_gwas = (
            crosswalk.groupBy("ADM_GWAS_COM").count().filter(F.col("count") != 1)
        )
        if (
            duplicate_participant.limit(1).count()
            or duplicate_gwas.limit(1).count()
        ):
            raise ValueError(
                "The participant/genetics crosswalk is not one-to-one."
            )
        master = master.join(crosswalk, "participant_id", "left")
        if genetics_sample_qc is not None:
            master = master.join(
                genetics_sample_qc, "ADM_GWAS_COM", "left"
            )
    else:
        master = master.withColumn("ADM_GWAS_COM", F.lit(None).cast("string"))

    master = (
        master.withColumn(
            "has_fundus_image",
            (F.coalesce(F.col("fundus_image_count"), F.lit(0)) > 0).cast("int")
            if "fundus_image_count" in master.columns
            else F.lit(0),
        )
        .withColumn(
            "has_genetics_link", F.col("ADM_GWAS_COM").isNotNull().cast("int")
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
