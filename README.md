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
  and link every fundus image to the age released for its matching visit. It
  also parses the six baseline-derived methylation phenotypes (`DNAmAge`, age-
  acceleration difference and residual, IEAA, EEAA, and Hannum age) directly
  from the baseline phenotype CSV, writes provenance/missingness/formula QC,
  and creates a baseline fundus–epigenetic linkage table without reading raw
  DNA, CpG, BGEN, or BGI files.
- [`notebooks/correlation.py`](notebooks/correlation.py) to stratify retinal age
  across the SAP measures and evaluate participant-grouped comorbidity models
  from the 1,024-dimensional RETFound vectors.
- [`Age_Glaucoma/01_build_age_matched_cohort.ipynb`](Age_Glaucoma/01_build_age_matched_cohort.ipynb)
  to reuse the completed source-specific Zeiss RETFound vectors and construct
  conservative, visit-specific age-matched CLSA ocular controls without
  reusing participants. Zeiss images are not subjected to CLSA QC thresholds.
- [`Age_Glaucoma/02_debug_age_matching.py`](Age_Glaucoma/02_debug_age_matching.py)
  for identifier-free aggregate diagnostics of CLSA control attrition, Zeiss
  and CLSA age overlap, nearest-age distances, and feasible match counts across
  several calipers.
- [`Age_Glaucoma/03_train_clsa_healthy_age_model.py`](Age_Glaucoma/03_train_clsa_healthy_age_model.py)
  to train and freeze the participant-grouped `CLSA_healthy` RETFound age head,
  apply it to Zeiss, graph retinal-age gap, and create a common-support,
  participant-level age-only matched comparison.
- [`Age_Glaucoma/04_compare_matched_explainability.py`](Age_Glaucoma/04_compare_matched_explainability.py)
  to reproduce source-specific RETFound inputs, verify stored embeddings, and
  compare matched subgroup attribution and spatial-outlier maps on a GPU.
- [`Age_Glaucoma/05_age_gap_extremes_explainability.py`](Age_Glaucoma/05_age_gap_extremes_explainability.py)
  compares participant-level top and bottom retinal-age-gap deciles with
  resumable exact attribution, max-|T| permutation inference, age/sex-matched
  sensitivity analyses, and exploratory physiology- and artifact-proxy
  overlays. Its `allow_repo_clone=true` default restores the small public
  RETFound source on a fresh cluster while `allow_downloads=false` continues
  to prohibit checkpoint downloads when the persistent checkpoint is present.
- [`Age_Glaucoma/06_three_cohort_glaucoma_analysis.py`](Age_Glaucoma/06_three_cohort_glaucoma_analysis.py)
  materializes the quality-passing, glaucoma-only CLSA vectors from the
  completed full RETFound run; applies the frozen healthy-CLSA age head; and
  compares CLSA healthy, CLSA glaucoma-only, and Zeiss glaucoma participants.
  It never reruns RETFound: the reusable Delta cache is preferred, with an
  automatic fallback to the completed 500-image Parquet batches when needed.
  The same-camera CLSA contrast is primary. Balanced additive location/scale
  harmonization, held-out domain classification, and raw-versus-harmonized
  Zeiss estimates are explicitly labeled sensitivity analyses because the
  healthy-Zeiss design cell remains absent.
- [`Age_Glaucoma/07_cross_device_harmonization_validation.py`](Age_Glaucoma/07_cross_device_harmonization_validation.py)
  reuses notebook 06 participant embeddings to compare raw, cross-fitted
  location, and cross-fitted location/scale source correction. It adds locked
  age/sex-matched domain diagnostics, direction-invariant source AUC, nested
  outer-fold device classification, and prespecified publication acceptance
  gates. It never reruns RETFound and does not claim to identify nonlinear
  device translation without paired-device images or healthy Zeiss controls.
- [`Age_Glaucoma/08_glaucoma_classifier_spatial_validation.py`](Age_Glaucoma/08_glaucoma_classifier_spatial_validation.py)
  trains a nested participant-held-out linear glaucoma classifier from the
  completed CLSA RETFound vectors, benchmarks age/sex/visit and image-quality
  confounding, and exactly decomposes held-out glaucoma logits into retinal
  patch contributions. Its primary spatial analysis compares independently
  validated optic-disc/peripapillary regions with equal-area retinal controls
  and targeted occlusion. The locked CLSA head is also applied to the existing
  Zeiss image vectors as a positive-cohort transport/localization analysis;
  without healthy Zeiss controls it does not report cross-device specificity.
  Bright-disc proxy localization is explicitly exploratory and cannot pass the
  publication claim-readiness gate.
- [`Age_Glaucoma/09_clsa_anatomic_explainability.py`](Age_Glaucoma/09_clsa_anatomic_explainability.py)
  restricts the next-stage anatomy analysis to CLSA glaucoma-only versus CLSA
  healthy participants and reuses notebook 08's participant-held-out exact
  contribution maps. It pins Berens Lab's MIT-licensed Fundus Image Toolbox,
  segments retinal vessels with the FIVES-trained FR-U-Net ensemble, localizes
  optic-disc and foveal centers, and quantifies attribution in optic-disc,
  peripapillary, foveal, and vessel regions. Optic-disc and foveal outputs are
  explicitly treated as localized circular ROIs rather than segmentations.
  Participant-level max-|T| permutation inference controls multiplicity, and
  optional targeted occlusion provides a slower confirmatory analysis.

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
