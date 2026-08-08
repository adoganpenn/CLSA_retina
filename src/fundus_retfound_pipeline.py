"""Fundus quality control, RETFound embeddings, retinal-age modeling, and XAI.

This module is deliberately notebook-friendly: each stage is a normal Python
function with explicit inputs and outputs, and the same stages are exposed
through a small command-line interface for job execution.

Expected manifest columns:
    image_path       required
    participant_id   required for grouped age-model evaluation
    eye              optional
    age              required only for age-head training/evaluation

The implementation follows the RETFound MAE CFP setup used in the supplied ODIR
notebook: AutoMorph-style retinal foreground cropping, 256 px normalization,
224 px model input, per-image/channel standardization, global mean pooling, and
an embedding-only Ridge age head.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Literal, Mapping, Sequence


@dataclass(frozen=True)
class QualityConfig:
    output_size: int = 256
    model_input_size: int = 224
    min_dimension_px: int = 224
    min_retina_fraction: float = 0.01
    min_brightness_mean: float = 10.0
    max_brightness_mean: float = 245.0
    min_contrast_std: float = 5.0
    min_gradient_energy: float = 2.0
    max_dark_fraction: float = 0.90
    max_bright_fraction: float = 0.90
    background_fill: Literal["median_retina", "black"] = "median_retina"
    save_preprocessed: bool = False


@dataclass(frozen=True)
class RETFoundConfig:
    repo_path: str | None = None
    checkpoint_path: str | None = None
    repo_cache_dir: str = "/local_disk0/tmp/retfound_repo_cache"
    repo_url: str = "https://github.com/rmaphoh/RETFound.git"
    hf_repo: str = "YukunZhou/RETFound_mae_natureCFP"
    hf_filename: str = "RETFound_mae_natureCFP.pth"
    allow_downloads: bool = False
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    batch_size: int = 16
    model_name: str = "RETFound_mae"
    input_size: int = 224
    num_classes: int = 5


@dataclass(frozen=True)
class AgeModelConfig:
    alpha: float = 10.0
    max_splits: int = 5
    calibration: Literal["none", "intercept", "shrunk_slope"] = "intercept"
    calibration_shrink_to_one: float = 0.50
    calibration_slope_floor: float = 0.25
    random_state: int = 20260727


@dataclass(frozen=True)
class ExplainabilityConfig:
    n_images: int = 8
    selection: Literal["largest_abs_error", "random", "first"] = (
        "largest_abs_error"
    )
    random_state: int = 20260727
    method: Literal["exact", "occlusion"] = "exact"
    overlay_alpha: float = 0.45


@dataclass
class PreprocessedFundus:
    image: Any
    retina_fraction: float
    original_width: int
    original_height: int
    crop_x0: int
    crop_y0: int
    crop_x1: int
    crop_y1: int
    brightness_mean: float
    contrast_std: float
    gradient_energy: float
    dark_fraction: float
    bright_fraction: float


def _require_columns(frame: Any, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def read_frame(path: str | os.PathLike[str]) -> Any:
    import pandas as pd

    path = Path(path)
    lower = path.name.lower()
    if lower.endswith(".parquet"):
        return pd.read_parquet(path)
    if lower.endswith(".csv") or lower.endswith(".csv.gz"):
        return pd.read_csv(path)
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported tabular file: {path}")


def write_frame(frame: Any, path: str | os.PathLike[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lower = path.name.lower()
    if lower.endswith(".parquet"):
        frame.to_parquet(path, index=False)
    elif lower.endswith(".csv") or lower.endswith(".csv.gz"):
        frame.to_csv(path, index=False)
    else:
        raise ValueError(f"Output must be .parquet, .csv, or .csv.gz: {path}")
    return path


def read_embedding_failure_paths(
    path: str | os.PathLike[str],
) -> set[str]:
    """Read an embedding failure log, treating empty logs as no failures."""
    import pandas as pd

    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        failures = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return set()
    if "image_path" not in failures.columns:
        return set()
    return set(failures["image_path"].dropna().astype(str))


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return _json_safe(value.item())
        if isinstance(value, np.ndarray):
            return [_json_safe(item) for item in value.tolist()]
    except ModuleNotFoundError:
        pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    for method_name in ("to_dict", "as_dict", "_asdict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                converted = method()
            except Exception:
                continue
            if converted is not value:
                return _json_safe(converted)
    # Databricks may attach non-JSON execution metadata such as PlanMetrics to
    # otherwise ordinary results. Preserve a readable diagnostic without
    # allowing that runtime-only object to invalidate the durable JSON artifact.
    return {
        "python_type": f"{type(value).__module__}.{type(value).__name__}",
        "string_value": str(value),
    }


def write_json(value: Mapping[str, Any], path: str | os.PathLike[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(dict(value)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _retina_mask_from_rgb(rgb: Any) -> Any:
    import numpy as np

    rgb_f = rgb.astype(np.float32)
    gray = 0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]
    positive = gray[gray > 0]
    if positive.size == 0:
        return np.zeros(gray.shape, dtype=bool)
    threshold = max(8.0, float(np.percentile(positive, 2)))
    mask = gray > threshold

    height, width = mask.shape
    min_row_pixels = max(8, int(0.02 * width))
    min_col_pixels = max(8, int(0.02 * height))
    rows = mask.sum(axis=1) >= min_row_pixels
    columns = mask.sum(axis=0) >= min_col_pixels
    cleaned = np.zeros_like(mask)
    if rows.any() and columns.any():
        cleaned[np.ix_(rows, columns)] = mask[np.ix_(rows, columns)]
    return cleaned if cleaned.any() else mask


def _image_quality_metrics(rgb: Any, retina_mask: Any) -> dict[str, float]:
    import numpy as np

    rgb_f = rgb.astype(np.float32)
    gray = 0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]
    values = gray[retina_mask] if retina_mask.any() else gray.reshape(-1)
    gradient_x = np.diff(gray, axis=1)
    gradient_y = np.diff(gray, axis=0)
    return {
        "brightness_mean": float(np.mean(values)),
        "contrast_std": float(np.std(values)),
        "gradient_energy": float(
            np.mean(gradient_x**2) + np.mean(gradient_y**2)
        ),
        "dark_fraction": float(np.mean(values <= 5)),
        "bright_fraction": float(np.mean(values >= 250)),
    }


def preprocess_fundus(
    image_path: str | os.PathLike[str],
    config: QualityConfig = QualityConfig(),
) -> PreprocessedFundus:
    import numpy as np
    from PIL import Image

    bicubic = getattr(Image.Resampling, "BICUBIC", Image.BICUBIC)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        original_width, original_height = image.size
        rgb = np.asarray(image, dtype=np.uint8)

    mask = _retina_mask_from_rgb(rgb)
    retina_fraction = float(mask.mean())
    if not mask.any():
        raise ValueError("no retinal foreground detected")

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    crop = rgb[y0 : y1 + 1, x0 : x1 + 1].copy()
    crop_mask = mask[y0 : y1 + 1, x0 : x1 + 1]
    retinal_pixels = crop[crop_mask]
    fill_color = (
        np.median(retinal_pixels, axis=0).astype(np.uint8)
        if retinal_pixels.size
        else np.array([0, 0, 0], dtype=np.uint8)
    )
    if config.background_fill == "median_retina":
        crop[~crop_mask] = fill_color
    elif config.background_fill == "black":
        crop[~crop_mask] = 0
        fill_color = np.array([0, 0, 0], dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported background fill: {config.background_fill}")

    height, width = crop.shape[:2]
    side = max(height, width)
    square = np.empty((side, side, 3), dtype=np.uint8)
    square[:, :] = fill_color
    ypad, xpad = (side - height) // 2, (side - width) // 2
    square[ypad : ypad + height, xpad : xpad + width] = crop
    processed = Image.fromarray(square, mode="RGB").resize(
        (config.output_size, config.output_size), resample=bicubic
    )
    metrics = _image_quality_metrics(rgb, mask)
    return PreprocessedFundus(
        image=processed,
        retina_fraction=retina_fraction,
        original_width=original_width,
        original_height=original_height,
        crop_x0=x0,
        crop_y0=y0,
        crop_x1=x1,
        crop_y1=y1,
        **metrics,
    )


def _quality_reasons(result: PreprocessedFundus, config: QualityConfig) -> list[str]:
    reasons: list[str] = []
    if min(result.original_width, result.original_height) < config.min_dimension_px:
        reasons.append("dimension_below_minimum")
    if result.retina_fraction < config.min_retina_fraction:
        reasons.append("retina_fraction_below_minimum")
    if result.brightness_mean < config.min_brightness_mean:
        reasons.append("too_dark")
    if result.brightness_mean > config.max_brightness_mean:
        reasons.append("too_bright")
    if result.contrast_std < config.min_contrast_std:
        reasons.append("low_contrast")
    if result.gradient_energy < config.min_gradient_energy:
        reasons.append("low_sharpness")
    if result.dark_fraction > config.max_dark_fraction:
        reasons.append("dark_clipping")
    if result.bright_fraction > config.max_bright_fraction:
        reasons.append("bright_clipping")
    return reasons


def _safe_image_stem(record: Mapping[str, Any], image_path: Path) -> str:
    participant = str(record.get("participant_id", "unknown"))
    eye = str(record.get("eye", "NA"))
    suffix = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:10]
    stem = f"{participant}_{eye}_{image_path.stem}_{suffix}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def run_quality_pipeline(
    manifest: Any,
    output_dir: str | os.PathLike[str],
    config: QualityConfig = QualityConfig(),
) -> Any:
    import pandas as pd

    _require_columns(manifest, ["image_path"], "Fundus manifest")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = output_dir / "preprocessed_256"
    if config.save_preprocessed:
        processed_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, record in manifest.reset_index(drop=True).iterrows():
        base = record.to_dict()
        image_path = Path(str(record["image_path"]))
        quality_row = dict(base)
        quality_row.update(
            {
                "quality_pass": False,
                "quality_reasons": "",
                "quality_error": None,
                "processed_image_path": None,
            }
        )
        try:
            result = preprocess_fundus(image_path, config)
            reasons = _quality_reasons(result, config)
            quality_row.update(
                {
                    "original_width": result.original_width,
                    "original_height": result.original_height,
                    "retina_fraction": result.retina_fraction,
                    "brightness_mean": result.brightness_mean,
                    "contrast_std": result.contrast_std,
                    "gradient_energy": result.gradient_energy,
                    "dark_fraction": result.dark_fraction,
                    "bright_fraction": result.bright_fraction,
                    "crop_x0": result.crop_x0,
                    "crop_y0": result.crop_y0,
                    "crop_x1": result.crop_x1,
                    "crop_y1": result.crop_y1,
                    "quality_pass": not reasons,
                    "quality_reasons": "|".join(reasons),
                }
            )
            if config.save_preprocessed and not reasons:
                destination = (
                    processed_dir / f"{_safe_image_stem(base, image_path)}.jpg"
                )
                result.image.save(destination, quality=95, subsampling=0)
                quality_row["processed_image_path"] = str(destination)
        except Exception as exc:
            quality_row["quality_reasons"] = "unreadable_or_preprocessing_failed"
            quality_row["quality_error"] = (
                f"{type(exc).__name__}: {str(exc)}"[:500]
            )
        rows.append(quality_row)
        if (index + 1) % 250 == 0:
            print(f"quality checked {index + 1:,}/{len(manifest):,}")

    output = pd.DataFrame(rows)
    write_frame(output, output_dir / "fundus_quality_manifest.parquet")
    output.drop(columns=["embedding"], errors="ignore").to_csv(
        output_dir / "fundus_quality_manifest.csv", index=False
    )
    summary = {
        "n_images": int(len(output)),
        "n_pass": int(output["quality_pass"].sum()),
        "n_fail": int((~output["quality_pass"]).sum()),
        "pass_rate": float(output["quality_pass"].mean()) if len(output) else None,
        "config": asdict(config),
    }
    write_json(summary, output_dir / "fundus_quality_summary.json")
    return output


def prepare_model_input(
    image_path: str | os.PathLike[str],
    quality_config: QualityConfig = QualityConfig(),
) -> Any:
    import numpy as np
    from PIL import Image

    bicubic = getattr(Image.Resampling, "BICUBIC", Image.BICUBIC)
    result = preprocess_fundus(image_path, quality_config)
    image = result.image.resize(
        (quality_config.model_input_size, quality_config.model_input_size),
        resample=bicubic,
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    for channel in range(3):
        mean = float(array[..., channel].mean())
        standard_deviation = float(array[..., channel].std())
        array[..., channel] = (
            array[..., channel] - mean
        ) / max(standard_deviation, 1e-6)
    return array


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    if (
        requested == "mps"
        and (
            not hasattr(torch.backends, "mps")
            or not torch.backends.mps.is_available()
        )
    ):
        raise RuntimeError("MPS was requested but is unavailable.")
    if requested not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported device: {requested}")
    return requested


def _ensure_retfound_repo(config: RETFoundConfig) -> Path:
    if config.repo_path:
        repository = Path(config.repo_path)
        if not (repository / "models_vit.py").exists():
            raise FileNotFoundError(
                f"models_vit.py was not found in RETFound repo: {repository}"
            )
        return repository

    cache_root = Path(config.repo_cache_dir)
    repository = cache_root / "RETFound"
    if (repository / "models_vit.py").exists():
        return repository
    if not config.allow_downloads:
        raise FileNotFoundError(
            "RETFound repository is unavailable. Set repo_path, or set "
            "allow_downloads=True to clone the configured official repository."
        )
    cache_root.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        ["git", "clone", "--depth", "1", config.repo_url, str(repository)]
    )
    if not (repository / "models_vit.py").exists():
        raise FileNotFoundError(
            f"RETFound clone completed without models_vit.py: {repository}"
        )
    return repository


def _resolve_checkpoint(config: RETFoundConfig) -> Path:
    if config.checkpoint_path:
        checkpoint = Path(config.checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"RETFound checkpoint not found: {checkpoint}")
        return checkpoint
    if not config.allow_downloads:
        raise FileNotFoundError(
            "RETFound checkpoint is unavailable. Set checkpoint_path, or set "
            "allow_downloads=True after accepting the gated Hugging Face model terms."
        )
    from huggingface_hub import hf_hub_download

    checkpoint = hf_hub_download(
        repo_id=config.hf_repo,
        filename=config.hf_filename,
        token=os.environ.get("HF_TOKEN"),
    )
    return Path(checkpoint)


def load_retfound_model(
    config: RETFoundConfig,
) -> tuple[Any, str, Path, Path]:
    import importlib
    import torch

    repository = _ensure_retfound_repo(config)
    checkpoint = _resolve_checkpoint(config)
    device = resolve_device(config.device)
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    models = importlib.import_module("models_vit")
    if config.model_name not in models.__dict__:
        raise KeyError(
            f"Model {config.model_name!r} is unavailable in {repository}/models_vit.py"
        )
    model = models.__dict__[config.model_name](
        img_size=config.input_size,
        num_classes=config.num_classes,
        drop_path_rate=0,
        global_pool=True,
    )

    # The official checkpoint is a gated PyTorch pickle. Load only a checkpoint
    # obtained from the approved RETFound source and record its SHA-256.
    checkpoint_object = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    state = checkpoint_object.get("model", checkpoint_object)
    load_message = model.load_state_dict(state, strict=False)
    expected_missing = {
        "fc_norm.weight",
        "fc_norm.bias",
        "head.weight",
        "head.bias",
    }
    allowed_unexpected_names = {
        "mask_token",
        "decoder_pos_embed",
        "norm.weight",
        "norm.bias",
    }
    missing = set(load_message.missing_keys)
    unexpected = set(load_message.unexpected_keys)
    critical_missing = missing - expected_missing
    critical_unexpected = {
        key
        for key in unexpected
        if key not in allowed_unexpected_names
        and not key.startswith("decoder_")
        and not key.startswith("decoder_blocks.")
    }
    if critical_missing or critical_unexpected:
        raise RuntimeError(
            "RETFound checkpoint has unexpected encoder incompatibilities: "
            f"missing={sorted(critical_missing)}, "
            f"unexpected={sorted(critical_unexpected)}"
        )
    print(
        "RETFound encoder checkpoint loaded; expected untrained output/global-"
        f"pool keys={sorted(missing)}, ignored MAE-only keys={len(unexpected)}."
    )
    model.to(device)
    model.eval()
    return model, device, repository, checkpoint


def extract_retfound_embeddings(
    quality_manifest: Any,
    output_dir: str | os.PathLike[str],
    retfound_config: RETFoundConfig,
    quality_config: QualityConfig = QualityConfig(),
    *,
    model: Any | None = None,
    device: str | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    force: bool = False,
) -> Any:
    import numpy as np
    import pandas as pd
    import torch

    _require_columns(quality_manifest, ["image_path"], "Quality manifest")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "retfound_embeddings.parquet"
    failures_path = output_dir / "retfound_embedding_failures.csv"
    metadata_path = output_dir / "retfound_embedding_metadata.json"
    if cache_path.exists() and not force:
        print(f"Loading cached RETFound embeddings: {cache_path}")
        return pd.read_parquet(cache_path)

    work = quality_manifest.copy()
    if "quality_pass" in work.columns:
        work = work[work["quality_pass"].fillna(False)].copy()
    if work.empty:
        raise ValueError("No quality-passing images are available for RETFound.")

    repository = None
    if model is None:
        model, resolved_device, repository, checkpoint = load_retfound_model(
            retfound_config
        )
        device = resolved_device
        checkpoint_path = checkpoint
    else:
        device = device or resolve_device(retfound_config.device)
        checkpoint = Path(checkpoint_path) if checkpoint_path else None

    checkpoint_hash = sha256_file(checkpoint) if checkpoint else None
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(work), retfound_config.batch_size):
            batch = work.iloc[start : start + retfound_config.batch_size]
            arrays: list[Any] = []
            records: list[Any] = []
            for _, record in batch.iterrows():
                try:
                    arrays.append(
                        prepare_model_input(record["image_path"], quality_config)
                    )
                    records.append(record)
                except Exception as exc:
                    failures.append(
                        {
                            "image_path": record.get("image_path"),
                            "error": f"{type(exc).__name__}: {str(exc)}"[:500],
                        }
                    )
            if arrays:
                tensor = torch.from_numpy(np.stack(arrays)).to(device)
                tensor = torch.einsum("nhwc->nchw", tensor).float()
                features = model.forward_features(tensor)
                if isinstance(features, (tuple, list)):
                    features = features[0]
                features = features.squeeze()
                if features.ndim == 1:
                    features = features.unsqueeze(0)
                vectors = features.detach().cpu().numpy().astype(np.float32)
                for record, vector in zip(records, vectors):
                    row = record.to_dict()
                    row["embedding"] = vector
                    row["embedding_dim"] = int(vector.shape[0])
                    row["retfound_model"] = retfound_config.model_name
                    row["retfound_checkpoint_sha256"] = checkpoint_hash
                    rows.append(row)
            processed = min(start + retfound_config.batch_size, len(work))
            progress_interval = max(retfound_config.batch_size, 100)
            crossed_interval = (
                processed // progress_interval
                != start // progress_interval
            )
            if len(work) <= 32 or processed == len(work) or crossed_interval:
                print(f"embedded {processed:,}/{len(work):,}")

    output = pd.DataFrame(rows)
    if output.empty:
        raise RuntimeError("Every image failed before RETFound embedding extraction.")
    write_frame(output, cache_path)
    # Keep a stable schema even when every image succeeds. Pandas otherwise
    # writes a newline-only file, which raises EmptyDataError when a resume run
    # reads it.
    pd.DataFrame(
        failures,
        columns=["image_path", "error"],
    ).to_csv(failures_path, index=False)
    dimensions = sorted(output["embedding_dim"].dropna().astype(int).unique())
    if len(dimensions) != 1:
        raise ValueError(f"Inconsistent embedding dimensions: {dimensions}")
    write_json(
        {
            "n_input_quality_passing": int(len(work)),
            "n_embedded": int(len(output)),
            "n_failed": int(len(failures)),
            "embedding_dim": dimensions[0],
            "retfound_config": asdict(retfound_config),
            "quality_config": asdict(quality_config),
            "device": device,
            "repository": str(repository) if repository else None,
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint_sha256": checkpoint_hash,
        },
        metadata_path,
    )
    return output


def _feature_matrix(embedding_frame: Any) -> Any:
    import numpy as np

    _require_columns(embedding_frame, ["embedding"], "Embedding frame")
    matrix = np.stack(embedding_frame["embedding"].to_numpy()).astype(np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2-D embedding matrix, got {matrix.shape}")
    return matrix


def calibration_stats(y_true: Any, prediction: Any) -> dict[str, float]:
    import numpy as np

    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    valid = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[valid], pred[valid]
    if len(y) < 3 or np.nanstd(y) == 0:
        return {"slope": math.nan, "intercept": math.nan, "r2": math.nan}
    slope, intercept = np.polyfit(y, pred, 1)
    fitted = slope * y + intercept
    residual_sum = float(np.sum((pred - fitted) ** 2))
    total_sum = float(np.sum((pred - np.mean(pred)) ** 2))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(1 - residual_sum / total_sum)
        if total_sum > 0
        else math.nan,
    }


def _fit_calibration(
    y_train: Any, prediction_train: Any, config: AgeModelConfig
) -> dict[str, float | str]:
    import numpy as np

    y = np.asarray(y_train, dtype=float)
    pred = np.asarray(prediction_train, dtype=float)
    curve = calibration_stats(y, pred)
    mean_y = float(np.mean(y))
    mean_prediction = float(np.mean(pred))
    slope = float(curve["slope"])
    shrunk = (
        (1.0 - config.calibration_shrink_to_one) * slope
        + config.calibration_shrink_to_one
    )
    if not np.isfinite(shrunk) or abs(shrunk) < config.calibration_slope_floor:
        shrunk = (
            1.0
            if not np.isfinite(shrunk)
            else math.copysign(config.calibration_slope_floor, shrunk or 1.0)
        )
    return {
        "mode": config.calibration,
        "mean_y": mean_y,
        "mean_prediction": mean_prediction,
        "slope": slope,
        "shrunk_slope": float(shrunk),
        "intercept": float(curve["intercept"]),
    }


def _apply_calibration(prediction: Any, calibration: Mapping[str, Any]) -> Any:
    import numpy as np

    pred = np.asarray(prediction, dtype=float)
    mode = calibration["mode"]
    if mode == "none":
        return pred
    if mode == "intercept":
        return pred + (
            float(calibration["mean_y"])
            - float(calibration["mean_prediction"])
        )
    if mode == "shrunk_slope":
        return float(calibration["mean_y"]) + (
            pred - float(calibration["mean_prediction"])
        ) / float(calibration["shrunk_slope"])
    raise ValueError(f"Unsupported calibration mode: {mode}")


def _prediction_metrics(frame: Any, prediction_column: str) -> Any:
    import numpy as np
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for level in ("image", "participant"):
        if level == "image":
            evaluation = frame[["participant_id", "age", prediction_column]].copy()
        else:
            evaluation = (
                frame.groupby("participant_id", as_index=False)
                .agg(age=("age", "mean"), prediction=(prediction_column, "mean"))
                .rename(columns={"prediction": prediction_column})
            )
        y = evaluation["age"].to_numpy(float)
        pred = evaluation[prediction_column].to_numpy(float)
        gap = pred - y
        curve = calibration_stats(y, pred)
        rows.append(
            {
                "level": level,
                "n": int(len(evaluation)),
                "n_patients": int(evaluation["participant_id"].nunique()),
                "mae": float(np.mean(np.abs(gap))),
                "median_absolute_error": float(np.median(np.abs(gap))),
                "mean_gap": float(np.mean(gap)),
                "sd_gap": float(np.std(gap, ddof=1))
                if len(gap) > 1
                else math.nan,
                "calibration_slope": curve["slope"],
                "calibration_intercept": curve["intercept"],
                "calibration_r2": curve["r2"],
            }
        )
    return pd.DataFrame(rows)


def train_age_head(
    embedding_frame: Any,
    output_dir: str | os.PathLike[str],
    config: AgeModelConfig = AgeModelConfig(),
) -> tuple[Any, dict[str, Any]]:
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold

    _require_columns(
        embedding_frame,
        ["embedding", "age", "participant_id"],
        "Age training embeddings",
    )
    work = embedding_frame.copy().reset_index(drop=True)
    work["age"] = pd.to_numeric(work["age"], errors="coerce")
    work = work[
        work["age"].notna() & work["participant_id"].notna()
    ].reset_index(drop=True)
    matrix = _feature_matrix(work)
    age = work["age"].to_numpy(float)
    groups = work["participant_id"].astype(str).to_numpy()
    patient_count = len(np.unique(groups))
    if patient_count < 2:
        raise ValueError("At least two participants are required for grouped CV.")
    n_splits = min(config.max_splits, patient_count)
    splitter = GroupKFold(n_splits=n_splits)

    raw_prediction = np.full(len(work), np.nan, dtype=float)
    calibrated_prediction = np.full(len(work), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(matrix, age, groups), start=1
    ):
        estimator = Ridge(alpha=config.alpha)
        estimator.fit(matrix[train_indices], age[train_indices])
        train_prediction = estimator.predict(matrix[train_indices])
        test_prediction = estimator.predict(matrix[test_indices])

        # Calibration is fit on the training fold only. No test-fold ages enter
        # the calibration used for that fold.
        calibration = _fit_calibration(
            age[train_indices], train_prediction, config
        )
        raw_prediction[test_indices] = test_prediction
        calibrated_prediction[test_indices] = _apply_calibration(
            test_prediction, calibration
        )
        fold_rows.append(
            {
                "fold": fold,
                "n_train_images": int(len(train_indices)),
                "n_test_images": int(len(test_indices)),
                "n_train_patients": int(len(np.unique(groups[train_indices]))),
                "n_test_patients": int(len(np.unique(groups[test_indices]))),
                **calibration,
            }
        )

    predictions = work.drop(columns=["embedding"]).copy()
    predictions["retinal_age_raw_oof"] = raw_prediction
    predictions["retinal_age_prediction_oof"] = calibrated_prediction
    predictions["retinal_age_gap_oof"] = calibrated_prediction - age
    predictions["absolute_error_oof"] = np.abs(calibrated_prediction - age)
    predictions["cv_splits"] = n_splits

    final_estimator = Ridge(alpha=config.alpha)
    final_estimator.fit(matrix, age)
    # Calibrate the deployable final model using grouped OOF predictions, not
    # its optimistic in-sample predictions.
    final_calibration = _fit_calibration(age, raw_prediction, config)
    bundle = {
        "estimator": final_estimator,
        "calibration": final_calibration,
        "embedding_dim": int(matrix.shape[1]),
        "feature_set": "retfound_embedding_only",
        "age_model_config": asdict(config),
        "n_training_images": int(len(work)),
        "n_training_patients": int(patient_count),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "retfound_age_head.joblib"
    joblib.dump(bundle, model_path)
    write_frame(predictions, output_dir / "retfound_age_predictions_oof.parquet")
    predictions.to_csv(
        output_dir / "retfound_age_predictions_oof.csv", index=False
    )
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(output_dir / "retfound_age_fold_diagnostics.csv", index=False)
    metrics = _prediction_metrics(
        predictions, "retinal_age_prediction_oof"
    )
    metrics.to_csv(output_dir / "retfound_age_metrics.csv", index=False)
    metadata = {
        key: value for key, value in bundle.items() if key != "estimator"
    }
    metadata["model_path"] = str(model_path)
    metadata["oof_metrics"] = metrics.to_dict(orient="records")
    write_json(metadata, output_dir / "retfound_age_model_metadata.json")
    return predictions, bundle


def load_age_head(path: str | os.PathLike[str]) -> dict[str, Any]:
    import joblib

    bundle = joblib.load(path)
    required = {"estimator", "calibration", "embedding_dim"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Age model bundle is missing keys: {sorted(missing)}")
    return bundle


def predict_retinal_age(
    embedding_frame: Any,
    age_model: Mapping[str, Any] | str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
) -> Any:
    import numpy as np

    bundle = (
        load_age_head(age_model)
        if isinstance(age_model, (str, os.PathLike))
        else dict(age_model)
    )
    matrix = _feature_matrix(embedding_frame)
    if matrix.shape[1] != int(bundle["embedding_dim"]):
        raise ValueError(
            f"Embedding dimension {matrix.shape[1]} does not match age head "
            f"dimension {bundle['embedding_dim']}."
        )
    raw = bundle["estimator"].predict(matrix)
    prediction = _apply_calibration(raw, bundle["calibration"])
    output = embedding_frame.drop(columns=["embedding"]).copy()
    output["retinal_age_raw"] = raw
    output["retinal_age_prediction"] = prediction
    if "age" in output.columns:
        age = output["age"].to_numpy(float)
        output["retinal_age_gap"] = prediction - age
        output["absolute_error"] = np.abs(prediction - age)
    if output_path:
        write_frame(output, output_path)
    return output


def _effective_linear_head(
    age_model: Mapping[str, Any],
) -> tuple[Any, float]:
    import numpy as np

    estimator = age_model["estimator"]
    coefficients = np.asarray(estimator.coef_, dtype=np.float64).reshape(-1)
    intercept = float(estimator.intercept_)
    calibration = age_model["calibration"]
    mode = calibration["mode"]
    if mode == "none":
        return coefficients, intercept
    if mode == "intercept":
        intercept += float(calibration["mean_y"]) - float(
            calibration["mean_prediction"]
        )
        return coefficients, intercept
    if mode == "shrunk_slope":
        slope = float(calibration["shrunk_slope"])
        coefficients = coefficients / slope
        intercept = float(calibration["mean_y"]) + (
            intercept - float(calibration["mean_prediction"])
        ) / slope
        return coefficients, intercept
    raise ValueError(f"Unsupported calibration mode: {mode}")


def _retfound_patch_tokens_and_feature(
    model: Any,
    image_path: str | os.PathLike[str],
    device: str,
    quality_config: QualityConfig,
) -> tuple[Any, Any, Any]:
    import numpy as np
    import torch

    model.eval()
    array = prepare_model_input(image_path, quality_config)
    tensor = torch.from_numpy(array[None, ...]).to(device)
    tensor = torch.einsum("nhwc->nchw", tensor).float()
    with torch.inference_mode():
        tokens = model.patch_embed(tensor)
        cls_tokens = model.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = tokens + model.pos_embed
        tokens = model.pos_drop(tokens)
        for block in model.blocks:
            tokens = block(tokens)
        patch_tokens = tokens[:, 1:, :]
        pooled_raw = patch_tokens.mean(dim=1)
        if not getattr(model, "global_pool", False):
            raise NotImplementedError(
                "Exact patch decomposition requires RETFound global_pool=True."
            )
        pooled_normalized = model.fc_norm(pooled_raw)
    return (
        patch_tokens.squeeze(0).detach().cpu().numpy().astype(np.float64),
        pooled_raw.squeeze(0).detach().cpu().numpy().astype(np.float64),
        pooled_normalized.squeeze(0).detach().cpu().numpy().astype(np.float64),
    )


def _decompose_layer_norm_mean_pool(
    patch_tokens: Any,
    gamma: Any,
    beta: Any,
    epsilon: float,
    coefficients: Any,
    intercept: float,
) -> dict[str, Any]:
    import numpy as np

    patch_tokens = np.asarray(patch_tokens, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    pooled_raw = patch_tokens.mean(axis=0)
    mean = float(np.mean(pooled_raw))
    sigma = float(np.sqrt(np.mean((pooled_raw - mean) ** 2) + epsilon))
    pooled_normalized = gamma * (pooled_raw - mean) / sigma + beta
    token_weights = coefficients * gamma / sigma
    patch_variable = patch_tokens @ token_weights
    constant = float(
        np.sum(coefficients * (beta - gamma * mean / sigma)) + intercept
    )
    patch_count = patch_tokens.shape[0]
    additive_contributions = patch_variable / patch_count + constant / patch_count
    prediction_from_contributions = float(additive_contributions.sum())
    prediction_from_feature = float(
        pooled_normalized @ coefficients + intercept
    )
    return {
        "additive_contributions": additive_contributions,
        "variable_contributions": patch_variable / patch_count,
        "constant": constant,
        "pooled_normalized": pooled_normalized,
        "prediction_from_contributions": prediction_from_contributions,
        "prediction_from_feature": prediction_from_feature,
        "reconstruction_error": abs(
            prediction_from_contributions - prediction_from_feature
        ),
    }


def exact_patch_contributions(
    model: Any,
    age_model: Mapping[str, Any],
    image_path: str | os.PathLike[str],
    device: str,
    quality_config: QualityConfig = QualityConfig(),
) -> dict[str, Any]:
    import numpy as np

    patch_tokens, _, _ = (
        _retfound_patch_tokens_and_feature(
            model, image_path, device, quality_config
        )
    )
    coefficients, intercept = _effective_linear_head(age_model)
    if coefficients.shape[0] != patch_tokens.shape[1]:
        raise ValueError(
            f"Age head has {coefficients.shape[0]} coefficients but RETFound "
            f"patch tokens have dimension {patch_tokens.shape[1]}."
        )

    layer_norm = model.fc_norm
    gamma = layer_norm.weight.detach().cpu().numpy().astype(np.float64)
    beta = layer_norm.bias.detach().cpu().numpy().astype(np.float64)
    decomposition = _decompose_layer_norm_mean_pool(
        patch_tokens,
        gamma,
        beta,
        float(layer_norm.eps),
        coefficients,
        intercept,
    )
    patch_count = patch_tokens.shape[0]
    grid_size = int(round(math.sqrt(patch_count)))
    if grid_size * grid_size != patch_count:
        raise ValueError(f"Patch count is not square: {patch_count}")
    grid = decomposition["additive_contributions"].reshape(
        grid_size, grid_size
    )
    variable_grid = decomposition["variable_contributions"].reshape(
        grid_size, grid_size
    )
    return {
        "grid": grid,
        "variable_grid": variable_grid,
        "constant": decomposition["constant"],
        "prediction_from_grid": decomposition[
            "prediction_from_contributions"
        ],
        "prediction_from_feature": decomposition["prediction_from_feature"],
        "reconstruction_error": decomposition["reconstruction_error"],
    }


def occlusion_sensitivity(
    model: Any,
    age_model: Mapping[str, Any],
    image_path: str | os.PathLike[str],
    device: str,
    quality_config: QualityConfig = QualityConfig(),
) -> dict[str, Any]:
    import numpy as np
    import torch

    coefficients, intercept = _effective_linear_head(age_model)
    array = prepare_model_input(image_path, quality_config)
    patch_size = 16
    grid_size = quality_config.model_input_size // patch_size

    def predict(image_array: Any) -> float:
        tensor = torch.from_numpy(image_array[None, ...]).to(device)
        tensor = torch.einsum("nhwc->nchw", tensor).float()
        with torch.inference_mode():
            feature = model.forward_features(tensor)
            if isinstance(feature, (tuple, list)):
                feature = feature[0]
            feature = feature.squeeze().detach().cpu().numpy().reshape(-1)
        return float(feature @ coefficients + intercept)

    baseline = predict(array)
    drops = np.zeros((grid_size, grid_size), dtype=np.float64)
    for row in range(grid_size):
        for column in range(grid_size):
            masked = array.copy()
            masked[
                row * patch_size : (row + 1) * patch_size,
                column * patch_size : (column + 1) * patch_size,
                :,
            ] = 0.0
            drops[row, column] = baseline - predict(masked)
    return {
        "grid": drops,
        "variable_grid": drops,
        "constant": math.nan,
        "prediction_from_grid": math.nan,
        "prediction_from_feature": baseline,
        "reconstruction_error": math.nan,
    }


def _select_explainability_rows(
    prediction_frame: Any, config: ExplainabilityConfig
) -> Any:
    if config.selection == "largest_abs_error":
        if "absolute_error" in prediction_frame.columns:
            return prediction_frame.sort_values(
                "absolute_error", ascending=False
            ).head(config.n_images)
        if "absolute_error_oof" in prediction_frame.columns:
            return prediction_frame.sort_values(
                "absolute_error_oof", ascending=False
            ).head(config.n_images)
    if config.selection == "random":
        return prediction_frame.sample(
            n=min(config.n_images, len(prediction_frame)),
            random_state=config.random_state,
        )
    return prediction_frame.head(config.n_images)


def _save_explanation(
    record: Mapping[str, Any],
    attribution: Mapping[str, Any],
    output_dir: Path,
    config: ExplainabilityConfig,
    quality_config: QualityConfig,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = Path(str(record["image_path"]))
    stem = _safe_image_stem(record, image_path)
    processed = preprocess_fundus(image_path, quality_config).image
    processed = processed.resize(
        (quality_config.model_input_size, quality_config.model_input_size),
        resample=getattr(Image.Resampling, "BICUBIC", Image.BICUBIC),
    )
    grid = np.asarray(attribution["grid"], dtype=float)
    heat = np.asarray(
        Image.fromarray(grid.astype(np.float32), mode="F").resize(
            (quality_config.model_input_size, quality_config.model_input_size),
            resample=getattr(Image.Resampling, "BILINEAR", Image.BILINEAR),
        ),
        dtype=float,
    )
    limit = float(np.nanpercentile(np.abs(heat), 98))
    if not np.isfinite(limit) or limit == 0:
        limit = 1.0

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(processed)
    axes[0].set_title("RETFound input")
    axes[0].axis("off")
    image = axes[1].imshow(
        heat, cmap="coolwarm", vmin=-limit, vmax=limit
    )
    axes[1].set_title(f"{config.method} attribution")
    axes[1].axis("off")
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].imshow(processed)
    axes[2].imshow(
        heat,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        alpha=config.overlay_alpha,
    )
    axes[2].set_title("overlay")
    axes[2].axis("off")
    figure.suptitle(
        f"participant={record.get('participant_id', 'NA')}; "
        f"eye={record.get('eye', 'NA')}; "
        f"pred={attribution['prediction_from_feature']:.2f}",
        y=1.02,
    )
    figure.tight_layout()
    png_path = output_dir / f"{stem}_{config.method}_attribution.png"
    figure.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    patch_rows: list[dict[str, Any]] = []
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            patch_rows.append(
                {
                    "participant_id": record.get("participant_id"),
                    "image_path": str(image_path),
                    "eye": record.get("eye"),
                    "patch_row": row,
                    "patch_column": column,
                    "patch_contribution": float(grid[row, column]),
                    "variable_contribution": float(
                        attribution["variable_grid"][row, column]
                    ),
                    "prediction_from_grid": attribution["prediction_from_grid"],
                    "prediction_from_feature": attribution[
                        "prediction_from_feature"
                    ],
                    "reconstruction_error": attribution[
                        "reconstruction_error"
                    ],
                    "method": config.method,
                }
            )
    csv_path = output_dir / f"{stem}_{config.method}_patch_scores.csv"
    pd.DataFrame(patch_rows).to_csv(csv_path, index=False)
    return {
        "participant_id": record.get("participant_id"),
        "image_path": str(image_path),
        "eye": record.get("eye"),
        "method": config.method,
        "prediction": attribution["prediction_from_feature"],
        "prediction_from_grid": attribution["prediction_from_grid"],
        "reconstruction_error": attribution["reconstruction_error"],
        "png_path": str(png_path),
        "patch_csv_path": str(csv_path),
    }


def run_explainability(
    embedding_frame: Any,
    age_model: Mapping[str, Any] | str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    retfound_config: RETFoundConfig,
    quality_config: QualityConfig = QualityConfig(),
    config: ExplainabilityConfig = ExplainabilityConfig(),
    *,
    model: Any | None = None,
    device: str | None = None,
) -> Any:
    import pandas as pd

    bundle = (
        load_age_head(age_model)
        if isinstance(age_model, (str, os.PathLike))
        else dict(age_model)
    )
    predictions = predict_retinal_age(embedding_frame, bundle)
    selected = _select_explainability_rows(predictions, config)
    if model is None:
        model, device, _, _ = load_retfound_model(retfound_config)
    else:
        device = device or resolve_device(retfound_config.device)

    rows: list[dict[str, Any]] = []
    output_dir = Path(output_dir)
    for _, record in selected.iterrows():
        if config.method == "exact":
            attribution = exact_patch_contributions(
                model,
                bundle,
                record["image_path"],
                device,
                quality_config,
            )
            if attribution["reconstruction_error"] > 1e-4:
                raise RuntimeError(
                    "Exact attribution failed reconstruction for "
                    f"{record['image_path']}: "
                    f"{attribution['reconstruction_error']}"
                )
        elif config.method == "occlusion":
            attribution = occlusion_sensitivity(
                model,
                bundle,
                record["image_path"],
                device,
                quality_config,
            )
        else:
            raise ValueError(f"Unsupported explainability method: {config.method}")
        rows.append(
            _save_explanation(
                record.to_dict(),
                attribution,
                output_dir,
                config,
                quality_config,
            )
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "explainability_manifest.csv", index=False)
    return manifest


def run_all(
    manifest: Any,
    output_dir: str | os.PathLike[str],
    retfound_config: RETFoundConfig,
    quality_config: QualityConfig = QualityConfig(),
    age_model_config: AgeModelConfig = AgeModelConfig(),
    explainability_config: ExplainabilityConfig = ExplainabilityConfig(),
    *,
    existing_age_model: str | os.PathLike[str] | None = None,
    run_explanations: bool = True,
    force_embeddings: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    quality = run_quality_pipeline(
        manifest, output_dir / "01_quality", quality_config
    )
    model, device, _, checkpoint = load_retfound_model(retfound_config)
    embeddings = extract_retfound_embeddings(
        quality,
        output_dir / "02_embeddings",
        retfound_config,
        quality_config,
        model=model,
        device=device,
        checkpoint_path=checkpoint,
        force=force_embeddings,
    )

    if existing_age_model:
        age_bundle = load_age_head(existing_age_model)
        predictions = predict_retinal_age(
            embeddings,
            age_bundle,
            output_dir / "03_age_model" / "retinal_age_predictions.parquet",
        )
    else:
        if "age" not in embeddings.columns:
            raise ValueError(
                "No age column is available. Supply an existing age model for "
                "prediction, or add age to train a new age head."
            )
        predictions, age_bundle = train_age_head(
            embeddings, output_dir / "03_age_model", age_model_config
        )

    explanations = None
    if run_explanations:
        explanations = run_explainability(
            embeddings,
            age_bundle,
            output_dir / "04_explainability",
            retfound_config,
            quality_config,
            explainability_config,
            model=model,
            device=device,
        )
    return {
        "quality": quality,
        "embeddings": embeddings,
        "predictions": predictions,
        "age_model": age_bundle,
        "explanations": explanations,
    }


def _retfound_config_from_args(arguments: argparse.Namespace) -> RETFoundConfig:
    return RETFoundConfig(
        repo_path=arguments.retfound_repo,
        checkpoint_path=arguments.checkpoint,
        repo_cache_dir=arguments.repo_cache,
        allow_downloads=arguments.allow_downloads,
        device=arguments.device,
        batch_size=arguments.batch_size,
    )


def _add_retfound_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--retfound-repo")
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--repo-cache", default="/local_disk0/tmp/retfound_repo_cache"
    )
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument(
        "--device", choices=["auto", "cuda", "mps", "cpu"], default="auto"
    )
    parser.add_argument("--batch-size", type=int, default=16)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fundus QC, RETFound, retinal-age, and explainability pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    quality = subparsers.add_parser("quality")
    quality.add_argument("--manifest", required=True)
    quality.add_argument("--output-dir", required=True)
    quality.add_argument("--save-preprocessed", action="store_true")

    embed = subparsers.add_parser("embed")
    embed.add_argument("--quality-manifest", required=True)
    embed.add_argument("--output-dir", required=True)
    embed.add_argument("--force", action="store_true")
    _add_retfound_arguments(embed)

    train = subparsers.add_parser("train-age")
    train.add_argument("--embeddings", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--alpha", type=float, default=10.0)
    train.add_argument("--max-splits", type=int, default=5)
    train.add_argument(
        "--calibration",
        choices=["none", "intercept", "shrunk_slope"],
        default="intercept",
    )

    predict = subparsers.add_parser("predict-age")
    predict.add_argument("--embeddings", required=True)
    predict.add_argument("--age-model", required=True)
    predict.add_argument("--output", required=True)

    explain = subparsers.add_parser("explain")
    explain.add_argument("--embeddings", required=True)
    explain.add_argument("--age-model", required=True)
    explain.add_argument("--output-dir", required=True)
    explain.add_argument("--n-images", type=int, default=8)
    explain.add_argument(
        "--selection",
        choices=["largest_abs_error", "random", "first"],
        default="largest_abs_error",
    )
    explain.add_argument(
        "--method", choices=["exact", "occlusion"], default="exact"
    )
    _add_retfound_arguments(explain)

    all_stages = subparsers.add_parser("all")
    all_stages.add_argument("--manifest", required=True)
    all_stages.add_argument("--output-dir", required=True)
    all_stages.add_argument("--age-model")
    all_stages.add_argument("--skip-explainability", action="store_true")
    all_stages.add_argument("--force-embeddings", action="store_true")
    all_stages.add_argument("--n-explain", type=int, default=8)
    _add_retfound_arguments(all_stages)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    if arguments.command == "quality":
        run_quality_pipeline(
            read_frame(arguments.manifest),
            arguments.output_dir,
            QualityConfig(save_preprocessed=arguments.save_preprocessed),
        )
        return 0
    if arguments.command == "embed":
        extract_retfound_embeddings(
            read_frame(arguments.quality_manifest),
            arguments.output_dir,
            _retfound_config_from_args(arguments),
            force=arguments.force,
        )
        return 0
    if arguments.command == "train-age":
        train_age_head(
            read_frame(arguments.embeddings),
            arguments.output_dir,
            AgeModelConfig(
                alpha=arguments.alpha,
                max_splits=arguments.max_splits,
                calibration=arguments.calibration,
            ),
        )
        return 0
    if arguments.command == "predict-age":
        predict_retinal_age(
            read_frame(arguments.embeddings),
            arguments.age_model,
            arguments.output,
        )
        return 0
    if arguments.command == "explain":
        run_explainability(
            read_frame(arguments.embeddings),
            arguments.age_model,
            arguments.output_dir,
            _retfound_config_from_args(arguments),
            config=ExplainabilityConfig(
                n_images=arguments.n_images,
                selection=arguments.selection,
                method=arguments.method,
            ),
        )
        return 0
    if arguments.command == "all":
        run_all(
            read_frame(arguments.manifest),
            arguments.output_dir,
            _retfound_config_from_args(arguments),
            existing_age_model=arguments.age_model,
            run_explanations=not arguments.skip_explainability,
            force_embeddings=arguments.force_embeddings,
            explainability_config=ExplainabilityConfig(
                n_images=arguments.n_explain
            ),
        )
        return 0
    raise AssertionError(f"Unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
