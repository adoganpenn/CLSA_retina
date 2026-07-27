# CLSA Retina

Reproducible pipelines for assembling CLSA retinal-aging analysis data and
processing fundus photographs with quality control, RETFound embeddings,
retinal-age models, and image-level explainability.

Start with:

- [`README_MASTER.md`](README_MASTER.md) for the complete CLSA data,
  questionnaire, imaging, phenotype, and genetics workflow.
- [`RUN_RETFOUND.md`](RUN_RETFOUND.md) for the fundus quality-control,
  RETFound, retinal-age, and explainability run sequence.
- [`notebooks/01_build_clsa_dataset.py`](notebooks/01_build_clsa_dataset.py)
  to assemble the analysis manifest in Databricks.
- [`notebooks/02_run_fundus_retfound.py`](notebooks/02_run_fundus_retfound.py)
  to run the fundus pipeline in Databricks.

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
