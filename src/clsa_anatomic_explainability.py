"""CLSA retinal-anatomy helpers for RETFound spatial explainability.

Fundus Image Toolbox supplies a vessel segmentation mask and fovea/optic-disc
center coordinates. The latter are converted into prespecified circular ROIs;
they are intentionally described as localized regions, not segmentations.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence


def require_columns(frame: Any, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def build_anatomic_masks(
    shape: tuple[int, int],
    coordinates: Sequence[float],
    vessel_mask: Any,
    retina_mask: Any,
    *,
    optic_disc_radius_scale: float = 0.20,
    fovea_radius_scale: float = 0.20,
    peripapillary_multiplier: float = 2.0,
    minimum_disc_fovea_fraction: float = 0.15,
    maximum_disc_fovea_fraction: float = 0.65,
    minimum_vessel_fraction: float = 0.005,
    maximum_vessel_fraction: float = 0.30,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create anatomically named masks and auditable localization QC.

    ``coordinates`` must be ``(fovea_x, fovea_y, optic_disc_x, optic_disc_y)``
    in pixels of the supplied image. Optic-disc and foveal masks are circular
    ROIs whose radii are prespecified fractions of the disc--fovea distance.
    """
    import numpy as np

    height, width = shape
    if height < 2 or width < 2:
        raise ValueError("Anatomic mask shape must be at least 2 x 2")
    values = np.asarray(coordinates, dtype=float).reshape(-1)
    if values.size != 4 or not np.isfinite(values).all():
        raise ValueError("Coordinates must contain four finite values")
    fovea_x, fovea_y, disc_x, disc_y = values.tolist()
    retina = np.asarray(retina_mask, dtype=bool)
    vessels = np.asarray(vessel_mask, dtype=float)
    if retina.shape != shape or vessels.shape != shape:
        raise ValueError("Retina and vessel masks must match the image shape")
    if not retina.any():
        raise ValueError("Retinal foreground mask is empty")
    if optic_disc_radius_scale <= 0 or fovea_radius_scale <= 0:
        raise ValueError("ROI radius scales must be positive")
    if peripapillary_multiplier <= 1:
        raise ValueError("peripapillary_multiplier must exceed 1")

    coordinate_in_bounds = bool(
        0 <= fovea_x < width
        and 0 <= fovea_y < height
        and 0 <= disc_x < width
        and 0 <= disc_y < height
    )
    distance = float(math.hypot(disc_x - fovea_x, disc_y - fovea_y))
    distance_fraction = distance / float(min(height, width))
    disc_radius = distance * optic_disc_radius_scale
    fovea_radius = distance * fovea_radius_scale
    yy, xx = np.indices(shape)
    disc_distance = np.hypot(xx - disc_x, yy - disc_y)
    fovea_distance = np.hypot(xx - fovea_x, yy - fovea_y)
    optic_disc = (disc_distance <= disc_radius) & retina
    peripapillary_total = (
        disc_distance <= peripapillary_multiplier * disc_radius
    ) & retina
    peripapillary_annulus = peripapillary_total & ~optic_disc
    fovea = (fovea_distance <= fovea_radius) & retina
    vessel_binary = (vessels >= 0.5) & retina
    vessel_fraction = float(vessel_binary.sum() / retina.sum())
    localized_regions = optic_disc | peripapillary_annulus | fovea
    vessels_elsewhere = vessel_binary & ~localized_regions
    other_retina = retina & ~(localized_regions | vessel_binary)
    anatomy_valid = bool(
        coordinate_in_bounds
        and minimum_disc_fovea_fraction
        <= distance_fraction
        <= maximum_disc_fovea_fraction
        and minimum_vessel_fraction
        <= vessel_fraction
        <= maximum_vessel_fraction
        and optic_disc.any()
        and fovea.any()
    )
    masks = {
        "optic_disc_roi": optic_disc,
        "peripapillary_annulus": peripapillary_annulus,
        "optic_disc_plus_peripapillary": peripapillary_total,
        "fovea_roi": fovea,
        "vessels": vessel_binary,
        "vessels_elsewhere": vessels_elsewhere,
        "other_retina": other_retina,
    }
    metadata = {
        "fovea_x_px": float(fovea_x),
        "fovea_y_px": float(fovea_y),
        "optic_disc_x_px": float(disc_x),
        "optic_disc_y_px": float(disc_y),
        "disc_fovea_distance_px": distance,
        "disc_fovea_distance_fraction": distance_fraction,
        "optic_disc_roi_radius_px": float(disc_radius),
        "fovea_roi_radius_px": float(fovea_radius),
        "vessel_fraction_of_retina": vessel_fraction,
        "coordinate_in_bounds": coordinate_in_bounds,
        "anatomy_valid": anatomy_valid,
        "optic_disc_definition": "localized circular ROI; not pixel segmentation",
        "fovea_definition": "localized circular ROI; not pixel segmentation",
        "vessel_definition": "Fundus Image Toolbox FR-U-Net ensemble segmentation",
    }
    return masks, metadata


def attribution_region_metrics(
    attribution_grid: Any,
    retina_mask: Any,
    region_masks: Mapping[str, Any],
) -> dict[str, float]:
    """Quantify signed, positive, and absolute attribution within each region."""
    import numpy as np
    from scipy import ndimage

    grid = np.asarray(attribution_grid, dtype=float)
    retina = np.asarray(retina_mask, dtype=bool)
    if grid.ndim != 2 or retina.ndim != 2 or not retina.any():
        raise ValueError("Expected a 2-D attribution grid and nonempty retina mask")
    resized = ndimage.zoom(
        grid,
        (retina.shape[0] / grid.shape[0], retina.shape[1] / grid.shape[1]),
        order=1,
    )[: retina.shape[0], : retina.shape[1]]
    positive = np.maximum(resized, 0.0) * retina
    absolute = np.abs(resized) * retina
    positive_total = float(positive.sum())
    absolute_total = float(absolute.sum())
    retina_area = int(retina.sum())
    output: dict[str, float] = {}
    for name, raw_mask in region_masks.items():
        mask = np.asarray(raw_mask, dtype=bool) & retina
        area_fraction = float(mask.sum() / retina_area)
        positive_fraction = (
            float(positive[mask].sum() / positive_total)
            if positive_total > 0
            else 0.0
        )
        absolute_fraction = (
            float(absolute[mask].sum() / absolute_total)
            if absolute_total > 0
            else 0.0
        )
        output[f"{name}_area_fraction"] = area_fraction
        output[f"{name}_positive_mass_fraction"] = positive_fraction
        output[f"{name}_absolute_mass_fraction"] = absolute_fraction
        output[f"{name}_positive_enrichment"] = (
            positive_fraction / area_fraction if area_fraction else math.nan
        )
        output[f"{name}_absolute_enrichment"] = (
            absolute_fraction / area_fraction if area_fraction else math.nan
        )
        output[f"{name}_signed_mean"] = (
            float(resized[mask].mean()) if mask.any() else math.nan
        )
    return output


def disc_fovea_affine_matrix(
    coordinates: Sequence[float],
    *,
    output_size: int = 256,
    target_disc_fraction: tuple[float, float] = (0.28, 0.50),
    target_fovea_fraction: tuple[float, float] = (0.72, 0.50),
) -> Any:
    """Return an affine transform that registers the disc--fovea axis.

    Images from both eyes are mapped into one canonical orientation with the
    optic disc on the left and fovea on the right.  The third landmark is a
    synthetic point perpendicular to the disc--fovea axis, which fixes scale
    and rotation without requiring an unvalidated nonlinear registration.
    """
    import cv2
    import numpy as np

    values = np.asarray(coordinates, dtype=float).reshape(-1)
    if values.size != 4 or not np.isfinite(values).all():
        raise ValueError("Coordinates must contain four finite values")
    if output_size < 32:
        raise ValueError("output_size must be at least 32 pixels")
    fovea_x, fovea_y, disc_x, disc_y = values.tolist()
    direction = np.asarray([fovea_x - disc_x, fovea_y - disc_y], dtype=float)
    distance = float(np.linalg.norm(direction))
    if distance < 1.0:
        raise ValueError("Disc and fovea centers are too close to register")
    perpendicular = np.asarray([-direction[1], direction[0]], dtype=float)
    source = np.float32(
        [
            [disc_x, disc_y],
            [fovea_x, fovea_y],
            [disc_x + perpendicular[0], disc_y + perpendicular[1]],
        ]
    )
    target_disc = np.asarray(target_disc_fraction, dtype=float) * output_size
    target_fovea = np.asarray(target_fovea_fraction, dtype=float) * output_size
    target_direction = target_fovea - target_disc
    target_perpendicular = np.asarray(
        [-target_direction[1], target_direction[0]], dtype=float
    )
    target = np.float32(
        [
            target_disc,
            target_fovea,
            target_disc + target_perpendicular,
        ]
    )
    return cv2.getAffineTransform(source, target)


def select_probability_extremes(
    frame: Any,
    *,
    probability_column: str = "glaucoma_probability_oof",
    participant_column: str = "participant_id",
    fraction: float = 0.10,
) -> Any:
    """Select disjoint participant-level bottom and top score fractions.

    Ties are resolved stably by participant ID.  At least one participant is
    retained in each tail, and the two tails are prevented from overlapping.
    """
    import numpy as np
    import pandas as pd

    require_columns(
        frame,
        [participant_column, probability_column],
        "Confidence-analysis frame",
    )
    if not 0 < fraction <= 0.25:
        raise ValueError("fraction must lie in (0, 0.25]")
    work = frame.copy()
    work[participant_column] = work[participant_column].astype(str)
    work[probability_column] = pd.to_numeric(
        work[probability_column], errors="coerce"
    )
    work = work[np.isfinite(work[probability_column])].copy()
    if work[participant_column].duplicated().any():
        raise ValueError("Confidence analysis requires one row per participant")
    if len(work) < 4:
        raise ValueError("At least four participants are required")
    work = work.sort_values(
        [probability_column, participant_column], kind="stable"
    ).reset_index(drop=True)
    tail_size = max(1, int(math.ceil(len(work) * fraction)))
    tail_size = min(tail_size, len(work) // 2)
    bottom = work.head(tail_size).copy()
    bottom["confidence_extreme"] = "bottom_healthy_like"
    bottom["confidence_extreme_code"] = 0
    top = work.tail(tail_size).copy()
    top["confidence_extreme"] = "top_glaucoma_like"
    top["confidence_extreme_code"] = 1
    return pd.concat([bottom, top], ignore_index=True)


def sample_translated_control_masks(
    target_mask: Any,
    retina_mask: Any,
    excluded_mask: Any,
    *,
    n_masks: int = 20,
    minimum_retained_fraction: float = 0.80,
    random_state: int = 20260815,
) -> list[Any]:
    """Translate a mask to retinal controls while preserving its topology."""
    import numpy as np

    target = np.asarray(target_mask, dtype=bool)
    retina = np.asarray(retina_mask, dtype=bool)
    excluded = np.asarray(excluded_mask, dtype=bool)
    if target.shape != retina.shape or target.shape != excluded.shape:
        raise ValueError("Control-mask inputs must have identical shapes")
    target_area = int(target.sum())
    if target_area < 1 or n_masks < 1:
        return []
    rng = np.random.default_rng(random_state)
    height, width = target.shape
    controls = []
    seen: set[tuple[int, int]] = set()
    for _ in range(max(500, n_masks * 50)):
        shift_y = int(rng.integers(-height + 1, height))
        shift_x = int(rng.integers(-width + 1, width))
        if (shift_y, shift_x) in seen or (shift_y == 0 and shift_x == 0):
            continue
        seen.add((shift_y, shift_x))
        shifted = np.roll(target, (shift_y, shift_x), axis=(0, 1))
        if shift_y > 0:
            shifted[:shift_y, :] = False
        elif shift_y < 0:
            shifted[shift_y:, :] = False
        if shift_x > 0:
            shifted[:, :shift_x] = False
        elif shift_x < 0:
            shifted[:, shift_x:] = False
        shifted &= retina & ~excluded
        if shifted.sum() >= minimum_retained_fraction * target_area:
            controls.append(shifted)
        if len(controls) >= n_masks:
            break
    return controls


def participant_permutation_inference(
    frame: Any,
    metric_columns: Sequence[str],
    *,
    label_column: str = "glaucoma_label",
    participant_column: str = "participant_id",
    permutations: int = 5000,
    bootstrap_repetitions: int = 2000,
    random_state: int = 20260815,
) -> Any:
    """Participant-level mean differences with max-|T| permutation control."""
    import numpy as np
    import pandas as pd

    require_columns(
        frame,
        [participant_column, label_column, *metric_columns],
        "Participant anatomy frame",
    )
    work = frame.copy()
    if work[participant_column].astype(str).duplicated().any():
        raise ValueError("Permutation inference requires one row per participant")
    labels = pd.to_numeric(work[label_column], errors="coerce").to_numpy()
    if set(labels) != {0, 1}:
        raise ValueError("Permutation inference requires both binary groups")
    matrix = work[list(metric_columns)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(matrix).all():
        raise ValueError("Permutation metrics must be finite")

    def statistics(group_labels: Any) -> Any:
        case = matrix[group_labels == 1]
        control = matrix[group_labels == 0]
        difference = case.mean(axis=0) - control.mean(axis=0)
        standard_error = np.sqrt(
            case.var(axis=0, ddof=1) / len(case)
            + control.var(axis=0, ddof=1) / len(control)
        )
        return difference / np.maximum(standard_error, 1e-12)

    observed_t = statistics(labels)
    rng = np.random.default_rng(random_state)
    null_t = np.empty((permutations, len(metric_columns)), dtype=float)
    for index in range(permutations):
        null_t[index] = statistics(rng.permutation(labels))
    maximum_null = np.max(np.abs(null_t), axis=1)
    case_values = matrix[labels == 1]
    control_values = matrix[labels == 0]
    observed_difference = case_values.mean(axis=0) - control_values.mean(axis=0)
    bootstrap_difference = np.empty(
        (bootstrap_repetitions, len(metric_columns)), dtype=float
    )
    for index in range(bootstrap_repetitions):
        case_sample = rng.choice(len(case_values), len(case_values), replace=True)
        control_sample = rng.choice(
            len(control_values), len(control_values), replace=True
        )
        bootstrap_difference[index] = (
            case_values[case_sample].mean(axis=0)
            - control_values[control_sample].mean(axis=0)
        )
    rows = []
    for column_index, metric in enumerate(metric_columns):
        raw_p = (1 + np.sum(np.abs(null_t[:, column_index]) >= abs(observed_t[column_index]))) / (
            permutations + 1
        )
        adjusted_p = (1 + np.sum(maximum_null >= abs(observed_t[column_index]))) / (
            permutations + 1
        )
        rows.append(
            {
                "metric": metric,
                "n_glaucoma": int((labels == 1).sum()),
                "n_healthy": int((labels == 0).sum()),
                "glaucoma_minus_healthy": float(observed_difference[column_index]),
                "bootstrap_95_ci_low": float(
                    np.quantile(bootstrap_difference[:, column_index], 0.025)
                ),
                "bootstrap_95_ci_high": float(
                    np.quantile(bootstrap_difference[:, column_index], 0.975)
                ),
                "welch_t": float(observed_t[column_index]),
                "permutation_p_raw": float(raw_p),
                "permutation_p_max_t": float(adjusted_p),
            }
        )
    return pd.DataFrame(rows)
