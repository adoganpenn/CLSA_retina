# CLSA Retina

Reproducible pipelines for assembling CLSA retinal-aging analysis data and
processing fundus photographs with quality control, RETFound embeddings,
retinal-age models, and image-level explainability.

Start with:

- [`README_MASTER.md`](README_MASTER.md) for the complete CLSA data,
  questionnaire, imaging, phenotype, and genetics workflow.
- [`notebooks/01_build_clsa_dataset.py`](notebooks/01_build_clsa_dataset.py)
  to inventory the exact release ZIPs, audit genetics readiness, generate the
  governed `DATASET_README.md`, and optionally assemble analysis tables.
- [`src/clsa_dataset_inventory.py`](src/clsa_dataset_inventory.py) for the
  metadata-only ZIP, CSV-schema, dictionary, and README-generation functions.
- [`notebooks/smoketest.ipynb`](notebooks/smoketest.ipynb) for the validated
  eight-image crop, quality-control, RETFound, and vector smoke test.
- [`RUN_RETFOUND.md`](RUN_RETFOUND.md) for the full fundus quality-control,
  RETFound, retinal-age, and explainability run sequence.
- [`notebooks/02_run_fundus_retfound.py`](notebooks/02_run_fundus_retfound.py)
  to run the complete fundus pipeline in Databricks.
- [`notebooks/retfound_quality_passed.py`](notebooks/retfound_quality_passed.py)
  as a live producer/consumer handoff: while notebook 02 continues writing its
  213 technical-quality batches, it detects each completed 500-row Parquet and
  immediately vectorizes only images that explicitly pass every QC flag.
- [`notebooks/03_build_sap_analysis_dataset.py`](notebooks/03_build_sap_analysis_dataset.py)
  to extract the exact BL/F1 questionnaire releases, derive the SAP variables,
  and link every fundus image to the age released for its matching visit.
- [`notebooks/correlation.py`](notebooks/correlation.py) to stratify retinal age
  across the SAP measures and evaluate participant-grouped comorbidity models
  from the 1,024-dimensional RETFound vectors.
- [`Age_Glaucoma/01_build_age_matched_cohort.ipynb`](Age_Glaucoma/01_build_age_matched_cohort.ipynb)
  to reuse the completed source-specific Zeiss RETFound vectors and construct
  conservative, visit-specific age-matched CLSA ocular controls without
  reusing participants. Zeiss images are not subjected to CLSA QC thresholds.

Run `correlation.py` after notebooks 02 and 03 have produced the full RETFound
embedding table, retinal-age predictions, and `sap_questionnaire_visit`. It
averages both eyes to one participant-visit record, uses out-of-fold retinal-age
predictions when available, and prevents BL/F1 records from the same participant
from crossing model folds. Derived analysis tables are written under:

```text
/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/derived/
clsa_retinal_aging/correlation
```

The first phase of `01_build_clsa_dataset.py` is metadata-only: it reads ZIP
central directories and questionnaire CSV headers without extracting images or
participant rows. It writes a comprehensive generated README to:

```text
/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/derived/
clsa_retinal_aging/DATASET_README.md
```

## Data governance

CLSA participant data, fundus images, questionnaire extracts, genetic data,
model checkpoints, credentials, and derived participant-level outputs must not
be committed to this repository. Keep restricted inputs and outputs in the
approved Unity Catalog volume:

```text
/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset
```

Only code, configuration templates, variable manifests, tests, and
non-sensitive documentation belong in Git.
