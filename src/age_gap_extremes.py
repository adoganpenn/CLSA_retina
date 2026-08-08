"""Participant-level age-gap extremes and fundus-attribution physiology proxies."""

from __future__ import annotations

import math
from typing import Any


def benjamini_hochberg(p_values: Any) -> Any:
    """Return Benjamini-Hochberg adjusted p-values with NaNs preserved."""
    import numpy as np

    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return adjusted
    observed = values[finite]
    order = np.argsort(observed)
    ranked = observed[order]
    count = len(ranked)
    corrected = ranked * count / np.arange(1, count + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    restored = np.empty(count, dtype=float)
    restored[order] = np.clip(corrected, 0, 1)
    adjusted[finite] = restored
    return adjusted


def _sex_design(series: Any) -> tuple[Any, list[str]]:
    import numpy as np
    import pandas as pd

    normalized = series.fillna("missing").astype(str).str.upper().str.strip()
    categories = sorted(normalized.unique())
    columns = []
    names = []
    for category in categories[1:]:
        columns.append((normalized == category).to_numpy(dtype=float))
        names.append(f"sex_{category}")
    matrix = np.column_stack(columns) if columns else np.empty((len(series), 0))
    return matrix, names


def residualize_age_gap(frame: Any) -> Any:
    """Residualize gap on centered age, age squared, and available sex."""
    import numpy as np
    import pandas as pd

    age = pd.to_numeric(frame["age"], errors="coerce").to_numpy(float)
    gap = pd.to_numeric(frame["age_gap"], errors="coerce").to_numpy(float)
    centered = age - float(np.nanmean(age))
    design_parts = [
        np.ones(len(frame), dtype=float),
        centered,
        centered**2,
    ]
    if "sex" in frame.columns:
        sex_matrix, _ = _sex_design(frame["sex"])
        if sex_matrix.shape[1]:
            design_parts.extend(sex_matrix.T)
    design = np.column_stack(design_parts)
    valid = np.isfinite(age) & np.isfinite(gap) & np.isfinite(design).all(axis=1)
    residual = np.full(len(frame), np.nan, dtype=float)
    if int(valid.sum()) < design.shape[1] + 2:
        return residual
    coefficients, *_ = np.linalg.lstsq(design[valid], gap[valid], rcond=None)
    residual[valid] = gap[valid] - design[valid] @ coefficients
    return residual


def build_participant_extremes(
    image_frame: Any,
    *,
    quantile: float = 0.10,
    selection_metric: str = "raw_gap",
) -> tuple[Any, Any, Any]:
    """Create participant summaries and one deterministic image per participant.

    Expected normalized columns are cohort, participant_id, image_path, age,
    age_gap, and optionally sex and eye. Decile cut points are calculated within
    each cohort. Group membership is participant-level, never eye-level.
    """
    import numpy as np
    import pandas as pd

    required = {"cohort", "participant_id", "image_path", "age", "age_gap"}
    missing = required - set(image_frame.columns)
    if missing:
        raise ValueError(f"Extreme image table is missing columns: {sorted(missing)}")
    if not 0 < quantile < 0.5:
        raise ValueError("quantile must be between 0 and 0.5")
    if selection_metric not in {"raw_gap", "age_adjusted_gap"}:
        raise ValueError("selection_metric must be raw_gap or age_adjusted_gap")

    work = image_frame.copy()
    if hasattr(work, "attrs"):
        work.attrs = {}
    work["participant_id"] = work["participant_id"].astype(str)
    work["age"] = pd.to_numeric(work["age"], errors="coerce")
    work["age_gap"] = pd.to_numeric(work["age_gap"], errors="coerce")
    work = work.dropna(subset=["participant_id", "image_path", "age", "age_gap"])
    work = work.drop_duplicates(["cohort", "participant_id", "image_path"])
    if work.empty:
        raise ValueError("No complete image-age-gap rows are available")

    aggregations: dict[str, tuple[str, str]] = {
        "age": ("age", "median"),
        "age_gap": ("age_gap", "mean"),
        "age_gap_sd_between_images": ("age_gap", "std"),
        "n_images": ("image_path", "nunique"),
    }
    if "sex" in work.columns:
        aggregations["sex"] = ("sex", "first")
    participant = (
        work.groupby(["cohort", "participant_id"], as_index=False)
        .agg(**aggregations)
    )
    participant["age_adjusted_gap"] = np.nan
    for _, indices in participant.groupby("cohort").groups.items():
        participant.loc[indices, "age_adjusted_gap"] = residualize_age_gap(
            participant.loc[indices]
        )

    metric_column = "age_gap" if selection_metric == "raw_gap" else "age_adjusted_gap"
    participant["selection_metric"] = selection_metric
    participant["extreme_group"] = "middle_80_percent"
    threshold_rows = []
    for cohort, indices in participant.groupby("cohort").groups.items():
        values = participant.loc[indices, metric_column]
        low = float(values.quantile(quantile))
        high = float(values.quantile(1 - quantile))
        participant.loc[indices, "extreme_group"] = np.select(
            [values <= low, values >= high],
            ["bottom_10_percent", "top_10_percent"],
            default="middle_80_percent",
        )
        threshold_rows.append(
            {
                "cohort": cohort,
                "selection_metric": selection_metric,
                "quantile": quantile,
                "bottom_threshold": low,
                "top_threshold": high,
                "n_participants": int(len(indices)),
                "n_bottom": int((values <= low).sum()),
                "n_top": int((values >= high).sum()),
            }
        )
    thresholds = pd.DataFrame(threshold_rows)

    enriched = work.merge(
        participant[
            [
                "cohort",
                "participant_id",
                "age_gap",
                "age_adjusted_gap",
                "extreme_group",
                "selection_metric",
            ]
        ].rename(columns={"age_gap": "participant_age_gap"}),
        on=["cohort", "participant_id"],
        how="inner",
        validate="many_to_one",
    )
    enriched["distance_from_participant_gap"] = (
        enriched["age_gap"] - enriched["participant_age_gap"]
    ).abs()
    if "eye" in enriched.columns:
        eye = enriched["eye"].astype(str).str.upper().str.strip()
        enriched["_eye_priority"] = np.where(
            eye.isin(["R", "RIGHT", "OD"]), 0, 1
        )
    else:
        enriched["_eye_priority"] = 1
    representative = (
        enriched.sort_values(
            [
                "cohort",
                "participant_id",
                "distance_from_participant_gap",
                "_eye_priority",
                "image_path",
            ],
            kind="stable",
        )
        .drop_duplicates(["cohort", "participant_id"], keep="first")
        .drop(columns=["_eye_priority"])
    )
    return participant, representative, thresholds


def permutation_patch_comparison(
    top_maps: Any,
    bottom_maps: Any,
    *,
    n_permutations: int = 2000,
    random_state: int = 20260808,
) -> dict[str, Any]:
    """Patchwise top-minus-bottom effects with max-|T| FWER correction."""
    import numpy as np

    top = np.asarray(top_maps, dtype=float)
    bottom = np.asarray(bottom_maps, dtype=float)
    if top.ndim != 3 or bottom.ndim != 3 or top.shape[1:] != bottom.shape[1:]:
        raise ValueError("top_maps and bottom_maps must be N x H x W with equal H/W")
    if len(top) < 5 or len(bottom) < 5:
        raise ValueError("At least five participant maps per extreme are required")
    if n_permutations < 100:
        raise ValueError("n_permutations must be at least 100")

    def statistic(a: Any, b: Any) -> tuple[Any, Any]:
        difference = np.mean(a, axis=0) - np.mean(b, axis=0)
        variance = np.var(a, axis=0, ddof=1) / len(a) + np.var(b, axis=0, ddof=1) / len(b)
        standard_error = np.sqrt(np.maximum(variance, 1e-16))
        return difference, difference / standard_error

    observed_difference, observed_t = statistic(top, bottom)
    pooled = np.concatenate([top, bottom], axis=0)
    rng = np.random.default_rng(random_state)
    exceed_unadjusted = np.zeros(observed_t.shape, dtype=np.int64)
    maximum_statistics = np.empty(n_permutations, dtype=float)
    n_top = len(top)
    for index in range(n_permutations):
        permutation = rng.permutation(len(pooled))
        _, permuted_t = statistic(
            pooled[permutation[:n_top]], pooled[permutation[n_top:]]
        )
        absolute = np.abs(permuted_t)
        exceed_unadjusted += absolute >= np.abs(observed_t)
        maximum_statistics[index] = float(np.nanmax(absolute))
    p_unadjusted = (exceed_unadjusted + 1) / (n_permutations + 1)
    p_fwer = (
        1
        + np.sum(
            maximum_statistics[:, None, None] >= np.abs(observed_t)[None, :, :],
            axis=0,
        )
    ) / (n_permutations + 1)
    p_fdr = benjamini_hochberg(p_unadjusted.reshape(-1)).reshape(observed_t.shape)
    return {
        "mean_difference": observed_difference,
        "welch_t": observed_t,
        "p_unadjusted": p_unadjusted,
        "p_fdr": p_fdr,
        "p_fwer": p_fwer,
        "n_top": int(len(top)),
        "n_bottom": int(len(bottom)),
        "n_permutations": int(n_permutations),
    }


def paired_permutation_patch_comparison(
    top_maps: Any,
    bottom_maps: Any,
    *,
    n_permutations: int = 2000,
    random_state: int = 20260809,
) -> dict[str, Any]:
    """Paired patch inference using within-pair random sign flips and max-|T|."""
    import numpy as np

    top = np.asarray(top_maps, dtype=float)
    bottom = np.asarray(bottom_maps, dtype=float)
    if top.ndim != 3 or top.shape != bottom.shape:
        raise ValueError("Paired top_maps and bottom_maps must have equal N x H x W")
    if len(top) < 5:
        raise ValueError("At least five matched participant pairs are required")
    if n_permutations < 100:
        raise ValueError("n_permutations must be at least 100")
    difference = top - bottom

    def paired_t(values: Any) -> Any:
        mean = np.mean(values, axis=0)
        standard_error = np.std(values, axis=0, ddof=1) / np.sqrt(len(values))
        return mean, mean / np.maximum(standard_error, 1e-8)

    observed_difference, observed_t = paired_t(difference)
    rng = np.random.default_rng(random_state)
    exceed_unadjusted = np.zeros(observed_t.shape, dtype=np.int64)
    maximum_statistics = np.empty(n_permutations, dtype=float)
    for index in range(n_permutations):
        signs = rng.choice((-1.0, 1.0), size=(len(difference), 1, 1))
        _, permuted_t = paired_t(difference * signs)
        absolute = np.abs(permuted_t)
        exceed_unadjusted += absolute >= np.abs(observed_t)
        maximum_statistics[index] = float(np.nanmax(absolute))
    p_unadjusted = (exceed_unadjusted + 1) / (n_permutations + 1)
    p_fwer = (
        1
        + np.sum(
            maximum_statistics[:, None, None] >= np.abs(observed_t)[None, :, :],
            axis=0,
        )
    ) / (n_permutations + 1)
    p_fdr = benjamini_hochberg(p_unadjusted.reshape(-1)).reshape(observed_t.shape)
    return {
        "mean_difference": observed_difference,
        "paired_t": observed_t,
        "p_unadjusted": p_unadjusted,
        "p_fdr": p_fdr,
        "p_fwer": p_fwer,
        "n_pairs": int(len(difference)),
        "n_permutations": int(n_permutations),
    }


def fundus_physiology_proxies(rgb: Any) -> dict[str, Any]:
    """Generate exploratory image-derived anatomy and artifact proxy maps.

    These are deliberately labeled proxies: they are not validated vessel,
    macula, or optic-disc segmentations and must not be treated as diagnoses.
    """
    import numpy as np
    from scipy import ndimage

    image = np.asarray(rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected H x W x 3 fundus RGB, got {image.shape}")
    if image.max() > 1.5:
        image = image / 255.0
    image = np.clip(image, 0, 1)
    luminance = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    positive = luminance[luminance > 0.01]
    threshold = max(0.03, float(np.percentile(positive, 2))) if positive.size else 0.03
    retina = luminance > threshold
    retina = ndimage.binary_fill_holes(retina)
    retina = ndimage.binary_opening(retina, iterations=2)

    height, width = luminance.shape
    yy, xx = np.indices((height, width))
    if retina.any():
        center_y = float(np.mean(yy[retina]))
        center_x = float(np.mean(xx[retina]))
    else:
        center_y, center_x = (height - 1) / 2, (width - 1) / 2
    radius = max(float(np.sqrt(np.sum(retina) / math.pi)), 1.0)
    radial = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2) / radius

    green_inverted = 1.0 - image[..., 1]
    vesselness = np.zeros_like(luminance)
    for sigma in (1.0, 2.0, 3.0):
        hxx = ndimage.gaussian_filter(green_inverted, sigma, order=(0, 2))
        hyy = ndimage.gaussian_filter(green_inverted, sigma, order=(2, 0))
        hxy = ndimage.gaussian_filter(green_inverted, sigma, order=(1, 1))
        trace = hxx + hyy
        discriminant = np.sqrt(np.maximum((hxx - hyy) ** 2 + 4 * hxy**2, 0))
        eigen_1 = (trace - discriminant) / 2
        eigen_2 = (trace + discriminant) / 2
        swap = np.abs(eigen_1) > np.abs(eigen_2)
        small = np.where(swap, eigen_2, eigen_1)
        large = np.where(swap, eigen_1, eigen_2)
        ratio = np.abs(small) / np.maximum(np.abs(large), 1e-8)
        structure = np.sqrt(small**2 + large**2)
        scale = float(np.percentile(structure[retina], 90)) if retina.any() else 1.0
        response = np.exp(-(ratio**2) / (2 * 0.5**2)) * (
            1 - np.exp(-(structure**2) / (2 * max(scale, 1e-8) ** 2))
        )
        response[large >= 0] = 0
        vesselness = np.maximum(vesselness, response)
    vesselness *= retina
    vessel_cut = float(np.percentile(vesselness[retina], 85)) if retina.any() else 1.0
    vessel_proxy = (vesselness >= vessel_cut) & retina

    smoothed_brightness = ndimage.gaussian_filter(luminance, sigma=max(height, width) / 50)
    disc_search = retina & (radial < 0.85)
    if disc_search.any():
        search_values = np.where(disc_search, smoothed_brightness, -np.inf)
        disc_y, disc_x = np.unravel_index(np.argmax(search_values), search_values.shape)
    else:
        disc_y, disc_x = int(center_y), int(center_x)
    disc_radius = 0.085 * min(height, width)
    optic_disc_proxy = ((yy - disc_y) ** 2 + (xx - disc_x) ** 2 <= disc_radius**2) & retina
    central_proxy = (
        (yy - center_y) ** 2 + (xx - center_x) ** 2
        <= (0.15 * min(height, width)) ** 2
    ) & retina
    peripheral_proxy = retina & (radial >= 0.65)
    border_proxy = retina & ~ndimage.binary_erosion(retina, iterations=max(2, min(height, width) // 30))
    gradient = np.hypot(ndimage.sobel(luminance, axis=0), ndimage.sobel(luminance, axis=1))
    gradient *= retina
    gradient_cut = float(np.percentile(gradient[retina], 85)) if retina.any() else 1.0
    high_gradient_proxy = (gradient >= gradient_cut) & retina

    return {
        "retina": retina,
        "luminance": luminance,
        "vesselness": vesselness,
        "gradient": gradient,
        "vessel_proxy": vessel_proxy,
        "optic_disc_proxy": optic_disc_proxy,
        "central_proxy": central_proxy,
        "peripheral_proxy": peripheral_proxy,
        "border_proxy": border_proxy,
        "high_gradient_proxy": high_gradient_proxy,
    }


def attribution_physiology_metrics(
    attribution_grid: Any,
    proxies: dict[str, Any],
) -> dict[str, float]:
    """Quantify attribution mass and enrichment in image-derived proxy regions."""
    import numpy as np
    from scipy import ndimage, stats

    grid = np.asarray(attribution_grid, dtype=float)
    shape = np.asarray(proxies["retina"]).shape
    resized = ndimage.zoom(
        grid,
        (shape[0] / grid.shape[0], shape[1] / grid.shape[1]),
        order=1,
    )
    resized = resized[: shape[0], : shape[1]]
    if resized.shape != shape:
        padded = np.zeros(shape, dtype=float)
        padded[: resized.shape[0], : resized.shape[1]] = resized
        resized = padded
    retina = np.asarray(proxies["retina"], dtype=bool)
    magnitude = np.abs(resized) * retina
    total = float(magnitude.sum())
    if total <= 0 or not retina.any():
        raise ValueError("Attribution or retinal mask has zero usable mass")
    result: dict[str, float] = {}
    for name in (
        "vessel_proxy",
        "optic_disc_proxy",
        "central_proxy",
        "peripheral_proxy",
        "border_proxy",
        "high_gradient_proxy",
    ):
        mask = np.asarray(proxies[name], dtype=bool) & retina
        area_fraction = float(mask.sum() / retina.sum())
        mass_fraction = float(magnitude[mask].sum() / total)
        result[f"{name}_area_fraction"] = area_fraction
        result[f"{name}_attribution_mass"] = mass_fraction
        result[f"{name}_enrichment"] = (
            mass_fraction / area_fraction if area_fraction > 0 else math.nan
        )
        result[f"{name}_signed_mean"] = (
            float(np.mean(resized[mask])) if mask.any() else math.nan
        )
    for name in ("vesselness", "gradient", "luminance"):
        values = np.asarray(proxies[name], dtype=float)[retina]
        correlation = stats.spearmanr(magnitude[retina], values, nan_policy="omit")
        result[f"absolute_attribution_{name}_spearman"] = float(correlation.statistic)
    result["background_attribution_fraction"] = float(
        np.abs(resized[~retina]).sum() / np.abs(resized).sum()
    ) if (~retina).any() and np.abs(resized).sum() > 0 else 0.0
    return result
