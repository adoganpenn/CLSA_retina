"""Reusable helpers for comparing linear RETFound age heads spatially."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def effective_linear_head(bundle: Mapping[str, Any]) -> tuple[Any, float]:
    """Return coefficients/intercept after applying either supported calibration.

    CLSA's global head stores the standard ``calibration`` mapping.  The
    race-balanced heads created by the fairness workflow store an intercept-only
    ``calibration_offset``.  Supporting both here prevents attribution from
    silently explaining the uncalibrated prediction.
    """
    import numpy as np

    if "estimator" not in bundle:
        raise ValueError("Age-head bundle is missing estimator")
    estimator = bundle["estimator"]
    coefficients = np.asarray(estimator.coef_, dtype=np.float64).reshape(-1)
    intercept = float(np.asarray(estimator.intercept_).reshape(-1)[0])
    if "calibration_offset" in bundle:
        return coefficients, intercept + float(bundle["calibration_offset"])
    calibration = bundle.get("calibration", {"mode": "none"})
    mode = calibration.get("mode", "none")
    if mode == "none":
        return coefficients, intercept
    if mode == "intercept":
        return coefficients, intercept + float(calibration["mean_y"]) - float(
            calibration["mean_prediction"]
        )
    if mode == "shrunk_slope":
        slope = float(calibration["shrunk_slope"])
        if not math.isfinite(slope) or abs(slope) < 1e-12:
            raise ValueError("Invalid shrunk calibration slope")
        return (
            coefficients / slope,
            float(calibration["mean_y"])
            + (intercept - float(calibration["mean_prediction"])) / slope,
        )
    raise ValueError(f"Unsupported calibration mode: {mode}")


def exact_multihead_patch_contributions(
    model: Any,
    heads: Mapping[str, tuple[Any, float]],
    image_path: str,
    device: str,
    quality_config: Any,
) -> dict[str, dict[str, Any]]:
    """Decompose several linear heads after one RETFound encoder pass."""
    import numpy as np

    from fundus_retfound_pipeline import (
        _decompose_layer_norm_mean_pool,
        _retfound_patch_tokens_and_feature,
    )

    if not heads:
        raise ValueError("At least one age head is required")
    patch_tokens, _, _ = _retfound_patch_tokens_and_feature(
        model, image_path, device, quality_config
    )
    layer_norm = model.fc_norm
    gamma = layer_norm.weight.detach().cpu().numpy().astype(np.float64)
    beta = layer_norm.bias.detach().cpu().numpy().astype(np.float64)
    patch_count = int(patch_tokens.shape[0])
    grid_size = int(round(math.sqrt(patch_count)))
    if grid_size * grid_size != patch_count:
        raise ValueError(f"Patch count is not square: {patch_count}")
    output: dict[str, dict[str, Any]] = {}
    for name, (raw_coefficients, raw_intercept) in heads.items():
        coefficients = np.asarray(raw_coefficients, dtype=np.float64).reshape(-1)
        if coefficients.size != patch_tokens.shape[1]:
            raise ValueError(
                f"Head {name!r} has {coefficients.size} coefficients; expected "
                f"{patch_tokens.shape[1]}"
            )
        decomposition = _decompose_layer_norm_mean_pool(
            patch_tokens,
            gamma,
            beta,
            float(layer_norm.eps),
            coefficients,
            float(raw_intercept),
        )
        output[str(name)] = {
            "grid": decomposition["additive_contributions"].reshape(
                grid_size, grid_size
            ),
            "variable_grid": decomposition["variable_contributions"].reshape(
                grid_size, grid_size
            ),
            "constant": float(decomposition["constant"]),
            "prediction_from_grid": float(
                decomposition["prediction_from_contributions"]
            ),
            "prediction_from_feature": float(
                decomposition["prediction_from_feature"]
            ),
            "reconstruction_error": float(decomposition["reconstruction_error"]),
        }
    return output


def coefficient_similarity_table(
    heads: Mapping[str, tuple[Any, float]],
    *,
    reference: str = "global",
    top_k: int = 50,
) -> Any:
    """Compare latent-head directions without calling dimensions anatomy."""
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    if reference not in heads:
        raise ValueError(f"Reference head {reference!r} is absent")
    reference_coef = np.asarray(heads[reference][0], dtype=float).reshape(-1)
    top_k = min(max(int(top_k), 1), reference_coef.size)
    reference_top = set(np.argsort(np.abs(reference_coef))[-top_k:].tolist())
    rows = []
    for name, (raw_coef, _) in heads.items():
        coef = np.asarray(raw_coef, dtype=float).reshape(-1)
        if coef.shape != reference_coef.shape:
            raise ValueError("All heads must have the same coefficient dimension")
        denominator = float(np.linalg.norm(reference_coef) * np.linalg.norm(coef))
        cosine = float(reference_coef @ coef / denominator) if denominator else math.nan
        target_top = set(np.argsort(np.abs(coef))[-top_k:].tolist())
        union = reference_top | target_top
        rows.append(
            {
                "model": str(name),
                "reference_model": reference,
                "embedding_dim": int(coef.size),
                "coefficient_cosine": cosine,
                "coefficient_spearman": float(spearmanr(reference_coef, coef).statistic),
                "coefficient_sign_agreement": float(
                    np.mean(np.sign(reference_coef) == np.sign(coef))
                ),
                f"top_{top_k}_jaccard": float(
                    len(reference_top & target_top) / len(union)
                ),
                "coefficient_l2_distance": float(np.linalg.norm(reference_coef - coef)),
            }
        )
    return pd.DataFrame(rows)


def paired_model_inference(
    frame: Any,
    metric_columns: Sequence[str],
    *,
    model_column: str = "model_name",
    reference_model: str = "global",
    participant_column: str = "participant_id",
    target_group_column: str = "racial_background",
    permutations: int = 5000,
    bootstrap_repetitions: int = 2000,
    random_state: int = 20260821,
) -> Any:
    """Paired subgroup-minus-global tests with max-|T| region correction."""
    import numpy as np
    import pandas as pd

    required = {
        model_column,
        participant_column,
        target_group_column,
        *metric_columns,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Paired inference frame is missing {sorted(missing)}")
    rng = np.random.default_rng(random_state)
    rows = []
    models = sorted(set(frame[model_column].astype(str)) - {reference_model})
    groups = sorted(frame[target_group_column].dropna().astype(str).unique())
    for group in groups:
        group_frame = frame[frame[target_group_column].astype(str) == group]
        reference = group_frame[group_frame[model_column] == reference_model].set_index(
            participant_column
        )
        for model in models:
            comparison = group_frame[group_frame[model_column] == model].set_index(
                participant_column
            )
            shared = reference.index.intersection(comparison.index)
            if len(shared) < 5:
                continue
            differences = (
                comparison.loc[shared, list(metric_columns)].apply(
                    pd.to_numeric, errors="coerce"
                ).to_numpy(float)
                - reference.loc[shared, list(metric_columns)].apply(
                    pd.to_numeric, errors="coerce"
                ).to_numpy(float)
            )
            finite = np.isfinite(differences).all(axis=1)
            differences = differences[finite]
            if len(differences) < 5:
                continue
            mean = differences.mean(axis=0)
            se = differences.std(axis=0, ddof=1) / np.sqrt(len(differences))
            observed_t = mean / np.maximum(se, 1e-12)
            null_t = np.empty((permutations, len(metric_columns)))
            for index in range(permutations):
                signs = rng.choice((-1.0, 1.0), size=(len(differences), 1))
                permuted = differences * signs
                null_t[index] = permuted.mean(axis=0) / np.maximum(
                    permuted.std(axis=0, ddof=1) / np.sqrt(len(permuted)), 1e-12
                )
            maximum_null = np.max(np.abs(null_t), axis=1)
            boot = np.empty((bootstrap_repetitions, len(metric_columns)))
            for index in range(bootstrap_repetitions):
                sample = rng.integers(0, len(differences), len(differences))
                boot[index] = differences[sample].mean(axis=0)
            for column_index, metric in enumerate(metric_columns):
                rows.append(
                    {
                        "target_racial_background": group,
                        "subgroup_model": model,
                        "reference_model": reference_model,
                        "metric": metric,
                        "n_paired_participants": int(len(differences)),
                        "subgroup_minus_global": float(mean[column_index]),
                        "bootstrap_95_ci_low": float(np.quantile(boot[:, column_index], 0.025)),
                        "bootstrap_95_ci_high": float(np.quantile(boot[:, column_index], 0.975)),
                        "paired_t": float(observed_t[column_index]),
                        "signflip_p_raw": float(
                            (1 + np.sum(np.abs(null_t[:, column_index]) >= abs(observed_t[column_index])))
                            / (permutations + 1)
                        ),
                        "signflip_p_max_t": float(
                            (1 + np.sum(maximum_null >= abs(observed_t[column_index])))
                            / (permutations + 1)
                        ),
                    }
                )
    return pd.DataFrame(rows)
