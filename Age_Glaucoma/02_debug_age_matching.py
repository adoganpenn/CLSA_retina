# Databricks notebook source
# MAGIC %md
# MAGIC # Age_Glaucoma: privacy-safe matching diagnostics
# MAGIC
# MAGIC This notebook diagnoses zero-match runs without displaying or writing any
# MAGIC Zeiss patient ID, CLSA participant ID, image path, or row-level record.
# MAGIC All outputs are aggregate counts, distributions, or age-range summaries.
# MAGIC
# MAGIC It answers four questions:
# MAGIC
# MAGIC 1. Did the ocular screen leave any eligible CLSA controls?
# MAGIC 2. Which condition or missing field caused control attrition?
# MAGIC 3. Do the Zeiss and CLSA age ranges overlap?
# MAGIC 4. How many age matches are theoretically possible at several calipers?

# COMMAND ----------
from pathlib import Path
import json

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

# COMMAND ----------
dbutils.widgets.text(
    "age_glaucoma_output_root",
    "/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/derived/"
    "clsa_retinal_aging/Age_Glaucoma",
)
dbutils.widgets.text("current_age_caliper_years", "5.0")

# COMMAND ----------
output_root = Path(dbutils.widgets.get("age_glaucoma_output_root").strip())
current_caliper = float(dbutils.widgets.get("current_age_caliper_years"))
if current_caliper < 0:
    raise ValueError("current_age_caliper_years cannot be negative")

zeiss_patient_path = (
    output_root / "01_zeiss_source_cohort" / "zeiss_patient_cohort.parquet"
)
clsa_screen_path = output_root / "03_clsa_controls" / "ocular_screen_delta"
clsa_eligible_images_path = (
    output_root / "03_clsa_controls" / "eligible_images_delta"
)
debug_root = output_root / "05_matching_debug"
debug_root.mkdir(parents=True, exist_ok=True)

for required_path in (
    zeiss_patient_path,
    clsa_screen_path,
    clsa_eligible_images_path,
):
    if not required_path.exists():
        raise FileNotFoundError(
            f"Required cohort output is missing: {required_path}. "
            "Run notebook 01 through CLSA control construction first."
        )

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Load only the fields required for aggregate diagnostics
# MAGIC
# MAGIC Patient identifiers are used internally only to count unique people and
# MAGIC collapse multiple visits. They are dropped before conversion to pandas
# MAGIC and are never passed to `display`, `print`, CSV, JSON, or exceptions.

# COMMAND ----------
zeiss_source = pd.read_parquet(zeiss_patient_path)
if "age" not in zeiss_source.columns:
    raise ValueError("Zeiss patient cohort has no age column")
zeiss_ages = pd.to_numeric(zeiss_source["age"], errors="coerce").dropna()
zeiss_total = int(len(zeiss_source))
zeiss_age_missing = int(len(zeiss_source) - len(zeiss_ages))
del zeiss_source

clsa_screen = spark.read.format("delta").load(str(clsa_screen_path))
clsa_eligible_images = spark.read.format("delta").load(
    str(clsa_eligible_images_path)
)

required_screen_columns = {
    "visit",
    "screen_complete",
    "ocular_screen_positive",
    "visual_impairment_self_report",
    "age_at_fundus_years",
    "control_eligible",
    "control_exclusion_reason",
}
missing_screen_columns = required_screen_columns - set(clsa_screen.columns)
if missing_screen_columns:
    raise ValueError(
        "CLSA screen table lacks diagnostic columns: "
        f"{sorted(missing_screen_columns)}"
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. CLSA control-screen attrition

# COMMAND ----------
condition_columns = [
    "retinal_detachment",
    "cataract",
    "glaucoma",
    "macular_degeneration",
    "diabetic_retinopathy",
]
condition_rows = []
for visit in ("BL", "F1"):
    visit_frame = clsa_screen.filter(F.col("visit") == visit)
    for condition in condition_columns:
        if condition not in clsa_screen.columns:
            condition_rows.append(
                {
                    "visit": visit,
                    "condition": condition,
                    "observed_negative": 0,
                    "observed_positive": 0,
                    "missing": int(visit_frame.count()),
                    "column_absent": True,
                }
            )
            continue
        counts = visit_frame.agg(
            F.sum(F.when(F.col(condition) == 0, 1).otherwise(0)).alias("negative"),
            F.sum(F.when(F.col(condition) == 1, 1).otherwise(0)).alias("positive"),
            F.sum(F.when(F.col(condition).isNull(), 1).otherwise(0)).alias("missing"),
        ).first()
        condition_rows.append(
            {
                "visit": visit,
                "condition": condition,
                "observed_negative": int(counts["negative"] or 0),
                "observed_positive": int(counts["positive"] or 0),
                "missing": int(counts["missing"] or 0),
                "column_absent": False,
            }
        )
condition_diagnostics = pd.DataFrame(condition_rows)
display(condition_diagnostics)

# COMMAND ----------
attrition_rows = []
for visit in ("BL", "F1"):
    frame = clsa_screen.filter(F.col("visit") == visit)
    counts = frame.agg(
        F.count("*").alias("participant_visits"),
        F.sum(F.when(F.col("screen_complete"), 1).otherwise(0)).alias(
            "complete_screen"
        ),
        F.sum(
            F.when(
                F.col("screen_complete")
                & ~F.col("ocular_screen_positive"),
                1,
            ).otherwise(0)
        ).alias("complete_and_ocular_negative"),
        F.sum(
            F.when(F.col("visual_impairment_self_report") == 0, 1).otherwise(0)
        ).alias("no_visual_impairment"),
        F.sum(
            F.when(F.col("age_at_fundus_years").isNotNull(), 1).otherwise(0)
        ).alias("age_observed"),
        F.sum(F.when(F.col("control_eligible"), 1).otherwise(0)).alias(
            "control_eligible"
        ),
    ).first()
    attrition_rows.append(
        {name: int(counts[name] or 0) for name in counts.asDict()}
        | {"visit": visit}
    )
clsa_attrition = pd.DataFrame(attrition_rows)[
    [
        "visit",
        "participant_visits",
        "complete_screen",
        "complete_and_ocular_negative",
        "no_visual_impairment",
        "age_observed",
        "control_eligible",
    ]
]
display(clsa_attrition)

exclusion_reasons = (
    clsa_screen.groupBy("visit", "control_exclusion_reason")
    .agg(F.count("*").alias("participant_visits"))
    .orderBy("visit", F.desc("participant_visits"))
)
display(exclusion_reasons)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Image/embedding linkage and age distributions

# COMMAND ----------
eligible_image_counts_row = clsa_eligible_images.agg(
    F.count("*").alias("eligible_images"),
    F.countDistinct("participant_id").alias("eligible_participants"),
    F.countDistinct(F.struct("participant_id", "visit")).alias(
        "eligible_participant_visits"
    ),
).first()
eligible_image_counts = pd.DataFrame(
    [
        {
            name: int(eligible_image_counts_row[name] or 0)
            for name in eligible_image_counts_row.asDict()
        }
    ]
)
display(eligible_image_counts)

# Collapse CLSA to one aggregate age per person before collecting. Identifiers
# remain on the Spark side and are removed from the collected table.
clsa_person_ages_spark = (
    clsa_eligible_images.groupBy("participant_id")
    .agg(
        F.expr("percentile_approx(age_at_fundus_years, 0.5)").cast("double").alias(
            "age"
        )
    )
    .select("age")
)
clsa_ages = pd.to_numeric(
    clsa_person_ages_spark.toPandas()["age"], errors="coerce"
).dropna()


def age_summary(label: str, ages: pd.Series) -> dict:
    values = pd.to_numeric(ages, errors="coerce").dropna().to_numpy(float)
    if not len(values):
        return {
            "cohort": label,
            "n_with_age": 0,
            "mean": np.nan,
            "sd": np.nan,
            "minimum": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "maximum": np.nan,
        }
    quantiles = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "cohort": label,
        "n_with_age": int(len(values)),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
        "minimum": float(np.min(values)),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "maximum": float(np.max(values)),
    }


age_distributions = pd.DataFrame(
    [
        age_summary("Zeiss", zeiss_ages),
        age_summary("CLSA eligible controls", clsa_ages),
    ]
)
display(age_distributions.round(2))

# COMMAND ----------
age_edges = [-np.inf, 18, 30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, np.inf]
age_labels = [
    "<18",
    "18-29",
    "30-39",
    "40-44",
    "45-49",
    "50-54",
    "55-59",
    "60-64",
    "65-69",
    "70-74",
    "75-79",
    "80-84",
    "85-89",
    "90+",
]
age_bin_rows = []
for label, ages in (("Zeiss", zeiss_ages), ("CLSA eligible controls", clsa_ages)):
    bins = pd.cut(ages, bins=age_edges, labels=age_labels, right=False)
    counts = bins.value_counts(sort=False)
    age_bin_rows.extend(
        {
            "cohort": label,
            "age_bin": str(age_bin),
            "people": int(count),
        }
        for age_bin, count in counts.items()
    )
age_bins = pd.DataFrame(age_bin_rows)
display(age_bins)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Caliper feasibility without exposing match pairs
# MAGIC
# MAGIC `cases_with_any_age_candidate` allows reuse and diagnoses age-range
# MAGIC coverage. `maximum_1_to_1_age_pairs` is the maximum number of pairs from a
# MAGIC deterministic two-pointer match on the two sorted age arrays. Neither
# MAGIC calculation retains or displays an identity or match pair.

# COMMAND ----------
def nearest_age_distances(case_ages, control_ages):
    cases = np.sort(np.asarray(case_ages, dtype=float))
    controls = np.sort(np.asarray(control_ages, dtype=float))
    if not len(cases):
        return np.asarray([], dtype=float)
    if not len(controls):
        return np.full(len(cases), np.inf)
    insertion = np.searchsorted(controls, cases)
    right_index = np.clip(insertion, 0, len(controls) - 1)
    left_index = np.clip(insertion - 1, 0, len(controls) - 1)
    return np.minimum(
        np.abs(cases - controls[left_index]),
        np.abs(cases - controls[right_index]),
    )


def maximum_one_to_one_age_pairs(case_ages, control_ages, caliper):
    cases = np.sort(np.asarray(case_ages, dtype=float))
    controls = np.sort(np.asarray(control_ages, dtype=float))
    case_index = 0
    control_index = 0
    pairs = 0
    while case_index < len(cases) and control_index < len(controls):
        difference = controls[control_index] - cases[case_index]
        if difference < -caliper:
            control_index += 1
        elif difference > caliper:
            case_index += 1
        else:
            pairs += 1
            case_index += 1
            control_index += 1
    return pairs


nearest_distances = nearest_age_distances(zeiss_ages, clsa_ages)
caliper_grid = sorted(
    {0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, float(current_caliper)}
)
caliper_rows = []
for caliper in caliper_grid:
    any_candidate = int(np.sum(nearest_distances <= caliper))
    maximum_pairs = int(
        maximum_one_to_one_age_pairs(zeiss_ages, clsa_ages, caliper)
    )
    caliper_rows.append(
        {
            "caliper_years": float(caliper),
            "zeiss_patients": int(len(zeiss_ages)),
            "cases_with_any_age_candidate": any_candidate,
            "cases_without_any_age_candidate": int(len(zeiss_ages) - any_candidate),
            "maximum_1_to_1_age_pairs": maximum_pairs,
            "maximum_match_percentage": (
                100.0 * maximum_pairs / len(zeiss_ages)
                if len(zeiss_ages)
                else np.nan
            ),
        }
    )
caliper_diagnostics = pd.DataFrame(caliper_rows)
display(caliper_diagnostics.round(2))

finite_nearest = nearest_distances[np.isfinite(nearest_distances)]
nearest_summary = age_summary(
    "Nearest CLSA-control age difference",
    pd.Series(finite_nearest),
)
nearest_summary["n_without_any_control"] = int(
    np.sum(~np.isfinite(nearest_distances))
)
nearest_diagnostics = pd.DataFrame([nearest_summary])
display(nearest_diagnostics.round(2))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Automated interpretation and privacy-safe output

# COMMAND ----------
screen_eligible_total = int(clsa_attrition["control_eligible"].sum())
eligible_clsa_people = int(len(clsa_ages))
current_row = caliper_diagnostics.loc[
    np.isclose(caliper_diagnostics["caliper_years"], current_caliper)
].iloc[0]
diagnostic_flags = []

if zeiss_age_missing:
    diagnostic_flags.append(
        f"Zeiss has {zeiss_age_missing:,}/{zeiss_total:,} patients without age."
    )
if screen_eligible_total == 0:
    diagnostic_flags.append(
        "The ocular questionnaire screen produced zero eligible CLSA "
        "participant-visits. Inspect the condition missingness table."
    )
elif eligible_clsa_people == 0:
    diagnostic_flags.append(
        "CLSA controls passed the questionnaire screen, but none survived the "
        "quality/embedding/image-path linkage. Inspect the linkage counts."
    )
elif int(current_row["cases_with_any_age_candidate"]) == 0:
    diagnostic_flags.append(
        f"There is no Zeiss-to-CLSA age candidate within ±{current_caliper:g} "
        "years. The problem is age-range overlap, not the matching algorithm."
    )
elif int(current_row["maximum_1_to_1_age_pairs"]) > 0:
    diagnostic_flags.append(
        f"Age-only matching should produce up to "
        f"{int(current_row['maximum_1_to_1_age_pairs']):,} pairs within "
        f"±{current_caliper:g} years. A zero-pair result therefore indicates "
        "stale notebook state, a different caliper value at execution, or an "
        "additional matching restriction rather than lack of age overlap."
    )
else:
    diagnostic_flags.append(
        "Controls and cases were loaded, but one-to-one age matching has zero "
        "capacity at the configured caliper."
    )

print("PRIVACY-SAFE DIAGNOSTIC INTERPRETATION")
for flag in diagnostic_flags:
    print("-", flag)

condition_diagnostics.to_csv(
    debug_root / "condition_missingness_aggregates.csv", index=False
)
clsa_attrition.to_csv(debug_root / "clsa_control_attrition.csv", index=False)
age_distributions.to_csv(debug_root / "age_distributions.csv", index=False)
age_bins.to_csv(debug_root / "age_bins.csv", index=False)
caliper_diagnostics.to_csv(
    debug_root / "caliper_feasibility.csv", index=False
)
nearest_diagnostics.to_csv(
    debug_root / "nearest_age_difference_summary.csv", index=False
)

privacy_safe_summary = {
    "contains_patient_identifiers": False,
    "zeiss_patients": zeiss_total,
    "zeiss_patients_with_age": int(len(zeiss_ages)),
    "zeiss_patients_missing_age": zeiss_age_missing,
    "clsa_screen_eligible_participant_visits": screen_eligible_total,
    "clsa_eligible_people_with_age": eligible_clsa_people,
    "configured_caliper_years": current_caliper,
    "cases_with_any_age_candidate": int(
        current_row["cases_with_any_age_candidate"]
    ),
    "maximum_1_to_1_age_pairs": int(
        current_row["maximum_1_to_1_age_pairs"]
    ),
    "diagnostic_flags": diagnostic_flags,
}
(debug_root / "privacy_safe_debug_summary.json").write_text(
    json.dumps(privacy_safe_summary, indent=2), encoding="utf-8"
)
print("Aggregate diagnostics saved under:", debug_root)
print(json.dumps(privacy_safe_summary, indent=2))

# COMMAND ----------
# MAGIC %md
# MAGIC ## What to share for debugging
# MAGIC
# MAGIC Copy the following aggregate outputs back into the Codex conversation:
# MAGIC
# MAGIC - `PRIVACY-SAFE DIAGNOSTIC INTERPRETATION`
# MAGIC - The two-row age-distribution table
# MAGIC - The CLSA attrition table
# MAGIC - The condition-missingness table
# MAGIC - The caliper-feasibility table
# MAGIC
# MAGIC Do **not** share the original match audit because it contains identifiers.
