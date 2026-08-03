# What to run: CLSA fundus quality, RETFound, retinal age, and explainability

The implementation is in:

- `src/fundus_retfound_pipeline.py`
- `notebooks/02_run_fundus_retfound.py`
- `requirements-retfound.txt`

The notebook is only a caller. All important operations are normal Python
functions, so they can be imported, stepped through, tested, or run from a job.

## Immediate credential action

The supplied ODIR notebook contains hard-coded Kaggle and Hugging Face
credentials. Revoke/rotate both credentials before running it again. Do not
copy those values into the new module or notebook.

The new pipeline reads a Hugging Face token only from `HF_TOKEN`. In Databricks,
store it as a secret such as:

- scope: `clsa`
- key: `huggingface-token`

The RETFound CFP checkpoint is gated. The account attached to the token must
first accept the model's access conditions.

## Recommended workflow

### 1. Build the CLSA participant/image table

Run:

`notebooks/01_build_clsa_dataset.py`

The desired source for RETFound is:

`/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/derived/clsa_retinal_aging/participant_analysis_master`

It should contain one participant row, `age_years`, and the `fundus_images`
array. Verify image ID linkage before running any model.

### 2. Start a Databricks GPU cluster

Use a GPU runtime with a working PyTorch/CUDA installation. Do not replace its
PyTorch wheel until this check passes:

```python
import torch

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

RETFound MAE is a large ViT model. CPU execution is suitable only for a very
small debugging sample.

### 3. Install non-PyTorch dependencies

In a Databricks notebook:

```python
%pip install -r /Workspace/Repos/<user>/<repo>/requirements-retfound.txt
dbutils.library.restartPython()
```

The dependency file deliberately does not install PyTorch. Its versions follow
the current official RETFound MAE requirements for NumPy, Pillow,
scikit-learn, matplotlib, and timm.

### 4. Run the 32-image smoke test

Open:

`notebooks/02_run_fundus_retfound.py`

Set these widgets:

```text
repo_root=/Workspace/Repos/<user>/<repo>
source_format=delta
source_path=/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/derived/clsa_retinal_aging/participant_analysis_master
output_root=/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/derived/clsa_retinal_aging/fundus_retfound_smoke
max_images=32
run_all_images=false
pipeline_batch_size=500
resume_batches=true
allow_downloads=true
retfound_repo=
checkpoint_path=
device=cuda
batch_size=4
force_embeddings=true
train_age_head=false
existing_age_model=
run_explainability=false
```

With `allow_downloads=true`, the module clones the configured official
RETFound repository into local cluster storage and downloads the gated CFP
checkpoint through `HF_TOKEN`. If workspace egress is disabled, upload an
approved repository snapshot and checkpoint to governed storage, then populate
`retfound_repo` and `checkpoint_path` explicitly.

Run the notebook through the embedding stage. Check:

- 32 source rows were selected.
- Technical quality pass/fail reasons are plausible.
- CUDA is the resolved device.
- One 1,024-dimensional embedding was produced for every passing image.
- `retfound_embedding_failures.csv` is empty or every failure is understood.

### 5. Review and calibrate quality thresholds

The quality stage reports:

- original dimensions;
- retinal foreground fraction;
- mean retinal brightness;
- contrast;
- gradient-energy sharpness;
- dark and bright clipping;
- crop bounding box;
- pass/fail reasons.

The defaults are conservative technical checks, not a validated clinical
gradability classifier. Before excluding the full cohort:

1. Review distributions by acquisition site, eye, cohort, age, and sex.
2. Manually review a stratified sample of passes and failures.
3. Version the selected `QualityConfig`.
4. Keep failures in the quality table rather than deleting source images.

The AutoMorph-style normalization follows the supplied notebook: dark
background detection, retinal-field crop, median-retina background fill,
square padding, bicubic resize to 256, then resize to 224 for RETFound.

### 6. Run all CLSA images through quality and embeddings

Use a new durable output path and set:

```text
source_path=/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/derived/clsa_retinal_aging/fundus_image_manifest
output_root=/Volumes/ophthalmology_analytics/dev_optic/clsa_dataset/derived/clsa_retinal_aging/fundus_retfound
run_all_images=true
pipeline_batch_size=500
resume_batches=true
batch_size=8
save_preprocessed=false
force_embeddings=false
run_explainability=false
```

`pipeline_batch_size=500` is the durable checkpoint interval. `batch_size=8`
is the GPU minibatch inside each checkpoint and can be tuned independently.
BL and F1 are both processed in full; their image counts are not balanced.
After every 500-image quality or embedding batch, a Parquet result is saved
under the corresponding `batches/` directory. If the cluster terminates,
rerun with the same output path and `resume_batches=true`; validated completed
batches are skipped and only the interrupted batch is repeated.

The first complete run writes:

```text
01_quality/fundus_quality_manifest.parquet
01_quality/fundus_quality_summary.json
01_quality/batches/batch_*/fundus_quality_manifest.parquet
02_embeddings/retfound_embeddings.parquet
02_embeddings/retfound_embedding_failures.csv
02_embeddings/retfound_embedding_metadata.json
02_embeddings/batches/batch_*/retfound_embeddings.parquet
```

Subsequent runs validate and load each saved batch when
`resume_batches=true`. Use a new output path for a clean rerun.
Checkpoint SHA-256 and model configuration are stored with the embedding
metadata.

## Training and applying the retinal-age head

### Preferred confirmatory workflow

Train and lock the age head in a development dataset, then apply that artifact
to CLSA. Do not train on all CLSA images and call the fitted values an external
validation.

The attached ODIR notebook already creates `analysis_manifest.csv`. After
rotating its credentials, that manifest can be passed to the new module.

Example command-line stages:

```bash
python src/fundus_retfound_pipeline.py quality \
  --manifest /path/to/odir/analysis_manifest.csv \
  --output-dir /path/to/odir_run/01_quality

python src/fundus_retfound_pipeline.py embed \
  --quality-manifest /path/to/odir_run/01_quality/fundus_quality_manifest.parquet \
  --output-dir /path/to/odir_run/02_embeddings \
  --retfound-repo /path/to/RETFound \
  --checkpoint /path/to/RETFound_mae_natureCFP.pth \
  --device cuda

python src/fundus_retfound_pipeline.py train-age \
  --embeddings /path/to/odir_run/02_embeddings/retfound_embeddings.parquet \
  --output-dir /path/to/odir_run/03_age_model \
  --calibration intercept
```

The locked model is:

`03_age_model/retfound_age_head.joblib`

Copy the locked artifact and its metadata into a governed, versioned CLSA model
directory. In the CLSA caller notebook set:

```text
existing_age_model=/Volumes/.../models/retfound_age_head.joblib
train_age_head=false
run_explainability=true
n_explain=8
explainability_method=exact
```

The prediction output contains:

- `retinal_age_raw`
- `retinal_age_prediction`
- `retinal_age_gap` when chronological age is present
- `absolute_error` when chronological age is present

### Exploratory CLSA training

For a clearly labeled exploratory run, set `train_age_head=true` and leave
`existing_age_model` blank. The module uses participant-grouped folds so images
from the same participant cannot occur in both training and test folds.
Calibration is fit using training-fold data only.

The resulting OOF metrics are useful for development but do not replace an
external validation.

## Explainability

The default `exact` method is valid for the provided pipeline because:

- RETFound uses global mean pooling across patch tokens;
- the age head is linear and embedding-only;
- the model applies `fc_norm` after mean pooling.

The implementation folds each image's LayerNorm statistics into the age-head
weights. The 196 additive patch contributions sum to the calibrated retinal-age
prediction. Every explanation records the reconstruction error and fails if it
exceeds the tolerance.

Outputs:

```text
04_explainability/explainability_manifest.csv
04_explainability/*_exact_attribution.png
04_explainability/*_exact_patch_scores.csv
```

Red patches contribute toward a higher predicted retinal age and blue patches
toward a lower age. These maps are diagnostic explanations, not causal effects
or statistical significance tests.

Use `explainability_method=occlusion` when testing a non-linear or
non-mean-pooled model. It is much slower because it performs one inference per
patch.

## Calling functions directly from a notebook

```python
import sys
sys.path.insert(0, "/Workspace/Repos/<user>/<repo>/src")

from fundus_retfound_pipeline import (
    QualityConfig,
    RETFoundConfig,
    run_quality_pipeline,
    load_retfound_model,
    extract_retfound_embeddings,
)

quality = run_quality_pipeline(
    manifest,
    "/Volumes/.../fundus_retfound/01_quality",
    QualityConfig(),
)

cfg = RETFoundConfig(
    repo_path="/local_disk0/RETFound",
    checkpoint_path="/Volumes/.../RETFound_mae_natureCFP.pth",
    device="cuda",
    batch_size=16,
)
model, device, repo, checkpoint = load_retfound_model(cfg)

embeddings = extract_retfound_embeddings(
    quality,
    "/Volumes/.../fundus_retfound/02_embeddings",
    cfg,
    model=model,
    device=device,
    checkpoint_path=checkpoint,
)
```

Because each stage returns a pandas DataFrame and accepts explicit
configuration objects, a debugger can stop between image QC, preprocessing,
model loading, embedding extraction, age-head prediction, and explanation.

## Scientific cautions

- ODIR-to-CLSA performance is subject to acquisition and population domain
  shift.
- ODIR has age and sex but no race/ethnicity field in the supplied workflow.
- Quality thresholds should be checked for differential failure rates.
- Retinal age gap is strongly coupled to chronological age if calibration is
  poor; report calibration slope/intercept and stratified performance.
- Explain only the locked model used for prediction.
- Retain model/checkpoint hashes, code commit, preprocessing configuration, and
  source manifest with every run.
