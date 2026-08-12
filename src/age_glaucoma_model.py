"""Robust analysis helpers for the CLSA healthy retinal-age model.

Functions are deliberately free of Databricks globals.  The training notebook
uses the repository's existing grouped-CV Ridge head, while this module handles
input validation, participant-level summaries, paired inference, and
source-specific RETFound attribution.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


def require_columns(frame: Any, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def validate_embedding_frame(
    frame: Any,
    label: str,
    expected_dim: int = 1024,
) -> Any:
    """Validate identifiers, ages, unique image keys, and every vector."""
    import numpy as np
    import pandas as pd

    require_columns(frame, ["participant_id", "image_path", "age", "embedding"], label)
    work = frame.copy()
    if work["participant_id"].isna().any():
        raise ValueError(f"{label} contains missing participant IDs")
    if work["image_path"].isna().any():
        raise ValueError(f"{label} contains missing image paths")
    work["participant_id"] = work["participant_id"].astype(str)
    work["image_path"] = work["image_path"].astype(str)
    work["age"] = pd.to_numeric(work["age"], errors="coerce")
    if work["participant_id"].str.strip().isin(["", "nan", "None"]).any():
        raise ValueError(f"{label} contains missing participant IDs")
    if work["image_path"].str.strip().isin(["", "nan", "None"]).any():
        raise ValueError(f"{label} contains missing image paths")
    if work["image_path"].duplicated().any():
        raise ValueError(f"{label} contains duplicate image paths")
    if work["age"].isna().any():
        raise ValueError(f"{label} contains missing/non-numeric ages")
    if ((work["age"] < 0) | (work["age"] > 120)).any():
        raise ValueError(f"{label} contains ages outside 0-120 years")

    invalid_dimension = 0
    invalid_values = 0
    zero_norm = 0
    for vector in work["embedding"]:
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
        if array.size != expected_dim:
            invalid_dimension += 1
            continue
        if not np.isfinite(array).all():
            invalid_values += 1
        elif float(np.linalg.norm(array)) <= 0:
            zero_norm += 1
    if invalid_dimension or invalid_values or zero_norm:
        raise ValueError(
            f"{label} has invalid vectors: wrong_dim={invalid_dimension}, "
            f"nonfinite={invalid_values}, zero_norm={zero_norm}"
        )
    return work.reset_index(drop=True)


def embedding_dataset_signature(frame: Any) -> str:
    """Hash the non-vector training contract for safe model resumption."""
    require_columns(frame, ["participant_id", "image_path", "age"], "Embedding frame")
    columns = ["participant_id", "image_path", "age"]
    if "visit" in frame.columns:
        columns.append("visit")
    canonical = frame[columns].copy()
    for column in canonical.columns:
        canonical[column] = canonical[column].astype(str)
    canonical = canonical.sort_values(columns, kind="stable")
    digest = hashlib.sha256()
    for row in canonical.itertuples(index=False, name=None):
        digest.update("\x1f".join(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def aggregate_clsa_oof_predictions(predictions: Any) -> Any:
    """Aggregate image OOF predictions to participant-visit records."""
    import pandas as pd

    require_columns(
        predictions,
        [
            "participant_id",
            "visit",
            "age",
            "retinal_age_prediction_oof",
            "retinal_age_gap_oof",
            "absolute_error_oof",
        ],
        "CLSA OOF predictions",
    )
    work = predictions.copy()
    work["participant_id"] = work["participant_id"].astype(str)
    work["visit"] = work["visit"].astype(str)
    work["age"] = pd.to_numeric(work["age"], errors="coerce")
    optional = {}
    if "sex" in work.columns:
        optional["sex"] = ("sex", "first")
    output = (
        work.groupby(["participant_id", "visit"], as_index=False)
        .agg(
            age=("age", "mean"),
            retinal_age_prediction=("retinal_age_prediction_oof", "mean"),
            retinal_age_gap=("retinal_age_gap_oof", "mean"),
            absolute_error=("absolute_error_oof", "mean"),
            n_images=("image_path", "count"),
            **optional,
        )
    )
    return output


def aggregate_zeiss_predictions(predictions: Any) -> Any:
    """Aggregate frozen-model image predictions to one record per patient."""
    import pandas as pd

    require_columns(
        predictions,
        ["participant_id", "age", "retinal_age_prediction", "retinal_age_gap"],
        "Zeiss predictions",
    )
    work = predictions.copy()
    optional = {}
    if "sex" in work.columns:
        optional["sex"] = ("sex", "first")
    output = (
        work.groupby("participant_id", as_index=False)
        .agg(
            age=("age", "median"),
            age_min=("age", "min"),
            age_max=("age", "max"),
            retinal_age_prediction=("retinal_age_prediction", "mean"),
            retinal_age_gap=("retinal_age_gap", "mean"),
            absolute_error=("absolute_error", "mean"),
            n_images=("image_path", "count"),
            **optional,
        )
    )
    output["age_range"] = output["age_max"] - output["age_min"]
    return output


def prediction_summary(frame: Any, cohort: str) -> dict[str, Any]:
    """Return participant-level age and gap summary statistics."""
    import numpy as np
    import pandas as pd

    require_columns(
        frame,
        ["participant_id", "age", "retinal_age_prediction", "retinal_age_gap"],
        cohort,
    )
    work = frame.copy()
    age = pd.to_numeric(work["age"], errors="coerce")
    predicted = pd.to_numeric(work["retinal_age_prediction"], errors="coerce")
    gap = pd.to_numeric(work["retinal_age_gap"], errors="coerce")
    valid = age.notna() & predicted.notna() & gap.notna()
    age, predicted, gap = age[valid], predicted[valid], gap[valid]
    return {
        "cohort": cohort,
        "n_participants": int(work.loc[valid, "participant_id"].nunique()),
        "chronological_age_mean": float(age.mean()),
        "chronological_age_sd": float(age.std(ddof=1)),
        "predicted_age_mean": float(predicted.mean()),
        "predicted_age_sd": float(predicted.std(ddof=1)),
        "age_gap_mean": float(gap.mean()),
        "age_gap_sd": float(gap.std(ddof=1)),
        "age_gap_median": float(gap.median()),
        "age_gap_q1": float(gap.quantile(0.25)),
        "age_gap_q3": float(gap.quantile(0.75)),
        "mae": float(np.mean(np.abs(gap))),
    }


def standardized_mean_difference(values_a: Any, values_b: Any) -> float:
    import numpy as np

    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return math.nan
    pooled = math.sqrt((float(np.var(a, ddof=1)) + float(np.var(b, ddof=1))) / 2)
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0


def bootstrap_mean_ci(
    values: Any,
    n_bootstrap: int = 5000,
    random_state: int = 20260807,
) -> tuple[float, float, float]:
    import numpy as np

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(random_state)
    means = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        means[index] = float(np.mean(rng.choice(array, size=len(array), replace=True)))
    return (
        float(np.mean(array)),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _zeiss_rgb_uint8(pixel_array: Any) -> Any:
    """Reproduce the attached Zeiss notebook's DICOM conversion."""
    import numpy as np

    array = np.asarray(pixel_array)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    elif array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.moveaxis(array, 0, -1)
    elif array.ndim == 3 and array.shape[-1] > 3:
        array = array[..., :3]
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Unsupported Zeiss DICOM pixel shape: {array.shape}")
    if array.dtype != np.uint8:
        low, high = np.nanpercentile(array, [0.5, 99.5])
        array = np.clip(
            (array.astype(np.float32) - low) / max(float(high - low), 1.0),
            0,
            1,
        )
        array = (array * 255).astype(np.uint8)
    return array


def _zeiss_retina_mask(rgb: Any) -> Any:
    import numpy as np

    rgb_float = rgb.astype(np.float32)
    gray = (
        0.299 * rgb_float[..., 0]
        + 0.587 * rgb_float[..., 1]
        + 0.114 * rgb_float[..., 2]
    )
    positive = gray[gray > 0]
    if not positive.size:
        return np.zeros(gray.shape, dtype=bool)
    threshold = max(8.0, float(np.percentile(positive, 2)))
    mask = gray > threshold
    height, width = mask.shape
    rows = mask.sum(axis=1) >= max(8, int(0.02 * width))
    columns = mask.sum(axis=0) >= max(8, int(0.02 * height))
    clean = np.zeros_like(mask)
    if rows.any() and columns.any():
        clean[np.ix_(rows, columns)] = mask[np.ix_(rows, columns)]
    return clean if clean.any() else mask


def prepare_zeiss_dicom_input(
    dcm_path: str | Path,
    output_size: int = 256,
    input_size: int = 224,
) -> tuple[Any, Any]:
    """Reproduce Zeiss AutoMorph crop and ImageNet normalization."""
    import numpy as np
    import pydicom
    from PIL import Image

    dataset = pydicom.dcmread(str(dcm_path), force=True)
    rgb = _zeiss_rgb_uint8(dataset.pixel_array)
    mask = _zeiss_retina_mask(rgb)
    if float(mask.mean()) < 0.01:
        raise ValueError("Zeiss retinal foreground fraction is below 0.01")
    ys, xs = np.where(mask)
    crop = rgb[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    height, width = crop.shape[:2]
    side = max(height, width)
    square = np.zeros((side, side, 3), dtype=np.uint8)
    y_pad, x_pad = (side - height) // 2, (side - width) // 2
    square[y_pad : y_pad + height, x_pad : x_pad + width] = crop
    bicubic = getattr(Image.Resampling, "BICUBIC", Image.BICUBIC)
    display_image = Image.fromarray(square, mode="RGB").resize(
        (output_size, output_size), resample=bicubic
    )
    model_image = display_image.resize((input_size, input_size), resample=bicubic)
    array = np.asarray(model_image, dtype=np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return (array - mean) / std, display_image


def load_zeiss_retfound_model(
    checkpoint_path: str | Path,
    device: str,
) -> Any:
    """Recreate the exact timm encoder used for the stored Zeiss vectors."""
    import timm
    import torch

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Zeiss RETFound checkpoint not found: {checkpoint_path}")
    model = timm.create_model(
        "vit_large_patch16_224",
        pretrained=False,
        num_classes=0,
        global_pool="avg",
    )
    checkpoint_object = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint_object.get("model", checkpoint_object)
    load_message = model.load_state_dict(state, strict=False)
    allowed_missing = {
        "fc_norm.weight",
        "fc_norm.bias",
        "head.weight",
        "head.bias",
    }
    allowed_unexpected = {
        "mask_token",
        "decoder_pos_embed",
        "norm.weight",
        "norm.bias",
    }
    critical_missing = set(load_message.missing_keys) - allowed_missing
    critical_unexpected = {
        key
        for key in load_message.unexpected_keys
        if key not in allowed_unexpected
        and not key.startswith("decoder_")
        and not key.startswith("decoder_blocks.")
    }
    if critical_missing or critical_unexpected:
        raise RuntimeError(
            "Zeiss timm RETFound checkpoint has encoder incompatibilities: "
            f"missing={sorted(critical_missing)}, "
            f"unexpected={sorted(critical_unexpected)}"
        )
    model.to(device)
    model.eval()
    print(
        "Zeiss timm encoder loaded; expected non-encoder keys were ignored: "
        f"missing={len(load_message.missing_keys)}, "
        f"unexpected={len(load_message.unexpected_keys)}."
    )
    return model


def exact_linear_patch_map_from_array(
    model: Any,
    coefficients: Any,
    intercept: float,
    input_array: Any,
    device: str,
    stored_embedding: Any | None = None,
) -> dict[str, Any]:
    """Compute an exact additive map for any source-specific linear head."""
    import numpy as np
    import torch

    from fundus_retfound_pipeline import (
        _decompose_layer_norm_mean_pool,
    )

    array = np.asarray(input_array, dtype=np.float32)
    tensor = torch.from_numpy(array[None, ...]).to(device)
    tensor = torch.einsum("nhwc->nchw", tensor).float()
    model.eval()
    with torch.inference_mode():
        tokens = model.patch_embed(tensor)
        cls_tokens = model.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = model.pos_drop(tokens + model.pos_embed)
        for block in model.blocks:
            tokens = block(tokens)
        patch_tokens = tokens[:, 1:, :]
        pooled = patch_tokens.mean(dim=1)
        feature = model.fc_norm(pooled)
    patch_numpy = patch_tokens.squeeze(0).cpu().numpy().astype(np.float64)
    feature_numpy = feature.squeeze(0).cpu().numpy().astype(np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    intercept = float(intercept)
    if coefficients.size != feature_numpy.size:
        raise ValueError(
            f"Linear head dimension {coefficients.size} != source feature "
            f"dimension {feature_numpy.size}"
        )
    layer_norm = model.fc_norm
    decomposition = _decompose_layer_norm_mean_pool(
        patch_numpy,
        layer_norm.weight.detach().cpu().numpy().astype(np.float64),
        layer_norm.bias.detach().cpu().numpy().astype(np.float64),
        float(layer_norm.eps),
        coefficients,
        intercept,
    )
    grid_size = int(round(math.sqrt(len(patch_numpy))))
    if grid_size * grid_size != len(patch_numpy):
        raise ValueError(f"Patch count is not square: {len(patch_numpy)}")
    cosine = math.nan
    maximum_absolute_difference = math.nan
    if stored_embedding is not None:
        stored = np.asarray(stored_embedding, dtype=np.float64).reshape(-1)
        if stored.shape != feature_numpy.shape:
            raise ValueError(
                f"Stored embedding shape {stored.shape} != reproduced {feature_numpy.shape}"
            )
        denominator = float(np.linalg.norm(stored) * np.linalg.norm(feature_numpy))
        cosine = float(stored @ feature_numpy / denominator) if denominator else math.nan
        maximum_absolute_difference = float(np.max(np.abs(stored - feature_numpy)))
    return {
        "grid": decomposition["additive_contributions"].reshape(grid_size, grid_size),
        "variable_grid": decomposition["variable_contributions"].reshape(
            grid_size, grid_size
        ),
        "prediction": decomposition["prediction_from_feature"],
        "reconstruction_error": decomposition["reconstruction_error"],
        "embedding_cosine": cosine,
        "embedding_max_absolute_difference": maximum_absolute_difference,
    }


def exact_patch_map_from_array(
    model: Any,
    age_model: Mapping[str, Any],
    input_array: Any,
    device: str,
    stored_embedding: Any | None = None,
) -> dict[str, Any]:
    """Backward-compatible source-specific exact map for a linear age head."""
    from fundus_retfound_pipeline import _effective_linear_head

    coefficients, intercept = _effective_linear_head(age_model)
    return exact_linear_patch_map_from_array(
        model=model,
        coefficients=coefficients,
        intercept=intercept,
        input_array=input_array,
        device=device,
        stored_embedding=stored_embedding,
    )


def normalize_attribution_map(grid: Any) -> Any:
    import numpy as np

    array = np.asarray(grid, dtype=float)
    denominator = float(np.sum(np.abs(array)))
    return array / denominator if denominator > 0 else np.zeros_like(array)


def attribution_group_statistics(maps: Any, outlier_z: float = 3.5) -> dict[str, Any]:
    """Aggregate normalized maps and robust patch-location outlier rates."""
    import numpy as np

    array = np.asarray(maps, dtype=float)
    if array.ndim != 3 or not len(array):
        raise ValueError(f"Expected N x H x W attribution maps, got {array.shape}")
    normalized = np.stack([normalize_attribution_map(grid) for grid in array])
    median = np.median(normalized, axis=0)
    mad = np.median(np.abs(normalized - median), axis=0)
    scale = np.maximum(1.4826 * mad, 1e-8)
    robust_z = (normalized - median) / scale
    outliers = np.abs(robust_z) >= outlier_z
    return {
        "normalized_maps": normalized,
        "mean_map": np.mean(normalized, axis=0),
        "median_map": median,
        "sd_map": np.std(normalized, axis=0, ddof=1) if len(array) > 1 else np.zeros_like(median),
        "outlier_frequency": np.mean(outliers, axis=0),
        "robust_z": robust_z,
        "outlier_mask": outliers,
    }
