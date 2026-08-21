# RETFound algorithm fairness

Run `01_retfound_age_fairness.ipynb` after notebooks 01–10 and the full
notebook 02 quality/RETFound workflow have completed at least once.

The notebook uses notebook 01's complete `fundus_image_manifest` as the image
denominator. It reuses notebook 02 batches, then runs any previously unprocessed
manifest image through the same quality pipeline and vectorizes every passing
image that has not already been attempted. Use GPU compute if missing vectors
remain; the Hugging Face token widget is only needed when the checkpoint is not
already cached.

The notebook:

- audits participant-level technical-quality pass rates before conditioning on
  image quality;
- checkpoints missing quality/vector batches, verifies complete image-level
  accounting, and incrementally pools all successful vectors;
- trains/resumes one race-blind CLSA-wide Ridge age head with participant-level
  grouped cross-validation;
- evaluates out-of-fold retinal-age calibration, signed error, MAE, and RMSE by
  released racial/cultural background, sex, education, income, and visit;
- separately matches each sufficiently sized non-White released group to White
  participants on age, sex, and available comorbidities using ±1-year primary
  and ±2-year sensitivity calipers; and
- saves match coverage, balance diagnostics, FDR-adjusted matched contrasts,
  publication figures, and a run-specific README.

Default durable outputs are written to:

`/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/derived/clsa_retinal_aging/Age_Glaucoma/16_algorithm_fairness`

Participant identifiers are restricted to `01_private`. The released `Black`
category is self-reported and must not be relabeled as African ancestry.
Observed disparities characterize the complete CLSA processing/model pipeline;
they do not, by themselves, prove bias intrinsic to the pretrained encoder.
