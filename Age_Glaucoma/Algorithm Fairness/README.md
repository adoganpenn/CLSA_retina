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

## Notebook 02: anatomic explainability across age heads

Notebook 01 section 10 retains every participant in each specific released
racial-background group meeting the prespecified inference threshold. It uses
the existing primary ±1-year, sex/comorbidity matching to create a separate 1:2
White comparator for each group. Five-fold participant cross-fitting supplies
OOF performance for all source-population participants; final target and
matched-White heads are saved for spatial analysis. `Other` and `Multiple
groups` remain descriptive because they are not single specific categories.

Run `02_age_model_anatomic_explainability.ipynb` after that section completes.
It consumes the expanded ledger and writes to
`07_all_available_matched_anatomic_explainability`. For every comparison record
it explains only the global, corresponding target, and corresponding
matched-White heads. It reports paired head-map differences on identical target
images and separate target-population versus matched-White-population anatomy
tests. Exact RETFound patch contributions are intersected with FR-U-Net vessel
masks and localized optic-disc, peripapillary, and foveal ROIs. Four-record
batch Parquets make the run restart-safe.

Because all-available target heads have unequal training sample sizes, this
power-enhanced analysis should be reported alongside the equal-N sensitivity
analysis when available; the two analyses answer different questions.
