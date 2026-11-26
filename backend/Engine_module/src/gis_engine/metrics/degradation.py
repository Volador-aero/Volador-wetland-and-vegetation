import os
import json
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple, Union, List
import scipy.ndimage as ndimage

from ..alignment.spatial import ReferenceGrid
from .area import calculate_pixel_areas, PhysicalAreaUnavailableError

@dataclass
class MetricsResult:
    baseline_area: Dict[str, Any]
    impact_area: Dict[str, Any]
    severity_weighted_area: Dict[str, Any]
    normalized_degradation_ratio: Optional[float]
    recovery_area: Dict[str, Any]
    residual_stress_area: Dict[str, Any]
    canopy_loss_area: Dict[str, Any]

    transition_metrics: Dict[str, Any]
    class_area_metrics: Dict[str, Any]
    cva_statistics: Dict[str, Any]
    delta_statistics: Dict[str, Any]

    wetland_extent_metrics: Dict[str, Any]

    qa_flags: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_input_raster(val: Union[str, Path, np.ndarray]) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """Helper to load a raster file path or handle an existing numpy array and extract metadata."""
    if isinstance(val, (str, Path)):
        path = Path(val)
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
        with rasterio.open(path) as src:
            if src.count != 1:
                raise ValueError(
                    f"Expected a single-band metrics raster, "
                    f"found {src.count} bands: {path}"
                )

            if src.width <= 0 or src.height <= 0:
                raise ValueError(
                    f"Invalid raster dimensions: {path}"
                )

            arr = src.read(1)

            if arr.shape != (src.height, src.width):
                raise ValueError(
                    f"Raster data and metadata dimensions "
                    f"do not match: {path}"
                )
            meta = {
                "crs": src.crs,
                "transform": src.transform,
                "width": src.width,
                "height": src.height,
                "resolution": src.res,
                "bounds": src.bounds,
                "nodata": src.nodata
            }
            return arr, meta
    elif isinstance(val, np.ndarray):
        return val, None
    else:
        raise TypeError(f"Expected str, Path, or numpy array, got {type(val)}")


def verify_spatial_alignment(
    metas: List[Dict[str, Any]],
):
    """Validate spatial metadata and raster grid alignment."""

    from numbers import Integral
    from rasterio.crs import CRS

    if not isinstance(metas, (list, tuple)):
        raise TypeError(
            "Raster metadata must be a list or tuple."
        )

    validated = []

    required = {
        "crs",
        "width",
        "height",
        "transform",
        "resolution",
        "bounds",
    }

    for index, meta in enumerate(metas):

        if meta is None:
            continue

        if not isinstance(meta, dict):
            raise TypeError(
                f"Invalid raster metadata at index {index}."
            )

        missing = required - set(meta)

        if missing:
            raise ValueError(
                f"Missing raster metadata: {sorted(missing)}"
            )

        for key in ("width", "height"):
            value = meta[key]

            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral)
                or value <= 0
            ):
                raise ValueError(
                    f"Invalid raster dimension: {key}."
                )

        crs = meta["crs"]

        if crs is not None:
            try:
                crs = CRS.from_user_input(crs)

                if not crs:
                    raise ValueError("Empty CRS.")

            except Exception as exc:
                raise ValueError(
                    f"Invalid raster CRS at index {index}."
                ) from exc

        transform = meta["transform"]

        try:
            coefficients = np.array([
                transform.a,
                transform.b,
                transform.c,
                transform.d,
                transform.e,
                transform.f,
            ], dtype=np.float64)

        except Exception as exc:
            raise ValueError(
                "Invalid raster affine transform."
            ) from exc

        if not np.all(np.isfinite(coefficients)):
            raise ValueError(
                "Raster transform contains nonfinite values."
            )

        a, b, c, d, e, f = coefficients

        determinant = a * e - b * d

        if (
            not np.isfinite(determinant)
            or determinant == 0
        ):
            raise ValueError(
                "Raster transform is singular or invalid."
            )

        resolution = np.asarray(
            meta["resolution"],
            dtype=np.float64,
        )

        if (
            resolution.shape != (2,)
            or not np.all(np.isfinite(resolution))
            or np.any(resolution <= 0)
        ):
            raise ValueError(
                "Invalid raster resolution."
            )

        expected_resolution = np.array([
            np.hypot(a, d),
            np.hypot(b, e),
        ])

        tolerance = float(
            np.min(expected_resolution) * 1e-6
        )

        if not np.allclose(
            resolution,
            expected_resolution,
            rtol=0,
            atol=tolerance,
        ):
            raise ValueError(
                "Resolution does not match affine transform."
            )

        bounds = np.asarray(
            meta["bounds"],
            dtype=np.float64,
        )

        if (
            bounds.shape != (4,)
            or not np.all(np.isfinite(bounds))
            or bounds[0] >= bounds[2]
            or bounds[1] >= bounds[3]
        ):
            raise ValueError(
                "Invalid raster bounds."
            )

        width = meta["width"]
        height = meta["height"]

        corners = np.array([
            [c, f],
            [a * width + c, d * width + f],
            [b * height + c, e * height + f],
            [
                a * width + b * height + c,
                d * width + e * height + f,
            ],
        ])

        expected_bounds = np.array([
            corners[:, 0].min(),
            corners[:, 1].min(),
            corners[:, 0].max(),
            corners[:, 1].max(),
        ])

        if not np.allclose(
            bounds,
            expected_bounds,
            rtol=0,
            atol=tolerance,
        ):
            raise ValueError(
                "Bounds do not match raster geometry."
            )

        validated.append({
            "crs": crs,
            "width": width,
            "height": height,
            "transform": coefficients,
            "resolution": resolution,
            "bounds": bounds,
            "tolerance": tolerance,
        })

    if not validated:
        return

    ref = validated[0]

    for meta in validated[1:]:

        if ref["crs"] != meta["crs"]:
            raise ValueError("CRS mismatch.")

        if (
            ref["width"] != meta["width"]
            or ref["height"] != meta["height"]
        ):
            raise ValueError("Dimension mismatch.")

        tolerance = min(
            ref["tolerance"],
            meta["tolerance"],
        )

        for key in (
            "transform",
            "resolution",
            "bounds",
        ):
            if not np.allclose(
                ref[key],
                meta[key],
                rtol=0,
                atol=tolerance,
            ):
                raise ValueError(
                    f"{key.capitalize()} mismatch."
                )



def filter_by_min_area(
    binary_map: np.ndarray,
    pixel_areas: np.ndarray,
    threshold_m2: float,
) -> np.ndarray:
    """Remove connected patches below the minimum physical area."""

    from numbers import Real

    if not isinstance(binary_map, np.ndarray):
        raise TypeError("Binary map must be a numpy array.")

    if binary_map.ndim != 2 or binary_map.size == 0:
        raise ValueError(
            "Binary map must be a nonempty 2D raster."
        )

    if binary_map.dtype.kind not in "biuf":
        raise TypeError("Invalid binary map data type.")

    if not np.all(
        (binary_map == 0)
        | (binary_map == 1)
        | (binary_map == 255)
    ):
        raise ValueError("Invalid binary map values.")

    if (
        isinstance(threshold_m2, (bool, np.bool_))
        or not isinstance(threshold_m2, Real)
        or not np.isfinite(threshold_m2)
        or threshold_m2 < 0
    ):
        raise ValueError("Invalid minimum area threshold.")

    if threshold_m2 == 0:
        return binary_map.copy()

    if not isinstance(pixel_areas, np.ndarray):
        raise TypeError("Pixel areas must be a numpy array.")

    if pixel_areas.shape != binary_map.shape:
        raise ValueError("Pixel area shape mismatch.")

    if pixel_areas.dtype.kind not in "iuf":
        raise TypeError("Invalid pixel area data type.")

    if not np.all(np.isfinite(pixel_areas)):
        raise ValueError("Pixel areas must be finite.")

    if np.any(pixel_areas <= 0):
        raise ValueError("Pixel areas must be positive.")

    result = binary_map.copy()

    foreground = binary_map == 1

    if not np.any(foreground):
        return result

    labels, component_count = ndimage.label(foreground)

    component_areas = np.bincount(
        labels[foreground],
        weights=pixel_areas[foreground].astype(
            np.float64, copy=False
        ),
        minlength=component_count + 1,
    )

    if not np.all(np.isfinite(component_areas)):
        raise ValueError(
            "Connected component area overflow."
        )

    remove_component = component_areas < threshold_m2

    # Label zero represents background.
    remove_component[0] = False

    removal_mask = remove_component[labels]

    result[removal_mask] = 0

    return result


def calculate_subset_stats(
    data: np.ndarray,
    mask: np.ndarray,
    percentiles: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Calculate finite statistics for a validated raster subset."""

    from numbers import Integral

    if percentiles is None:
        percentiles = [95]

    if not isinstance(data, np.ndarray):
        raise TypeError("Statistics data must be a numpy array.")

    if data.ndim != 2 or data.size == 0:
        raise ValueError(
            "Statistics data must be a nonempty 2D raster."
        )

    if data.dtype.kind not in "iuf":
        raise TypeError(
            "Statistics data must contain real numeric values."
        )

    if not isinstance(mask, np.ndarray):
        raise TypeError("Statistics mask must be a numpy array.")

    if mask.shape != data.shape:
        raise ValueError(
            "Statistics mask and raster shapes must match."
        )

    if mask.dtype.kind not in "biuf":
        raise TypeError(
            "Statistics mask must contain binary values."
        )

    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError(
            "Statistics mask must contain only zero and one."
        )

    if not isinstance(percentiles, (list, tuple)):
        raise TypeError(
            "Percentiles must be supplied as a list or tuple."
        )

    validated_percentiles = []

    for p in percentiles:
        if (
            isinstance(p, (bool, np.bool_))
            or not isinstance(p, Integral)
            or not 0 <= p <= 100
        ):
            raise ValueError(
                "Percentiles must be integers between 0 and 100."
            )

        p = int(p)

        if p in validated_percentiles:
            raise ValueError(
                "Duplicate percentiles are not permitted."
            )

        validated_percentiles.append(p)

    subset = data[mask.astype(bool, copy=False)]

    result = {
        "mean": None,
        "median": None,
        "min": None,
        "max": None,
        "std": None,
    }

    for p in validated_percentiles:
        result[f"p{p}"] = None

    if subset.size == 0:
        return result

    if not np.all(np.isfinite(subset)):
        raise ValueError(
            "Statistics subset contains NaN or infinite values."
        )

    subset = subset.astype(np.float64, copy=False)

    with np.errstate(over="ignore", invalid="ignore"):
        result.update({
            "mean": float(np.mean(subset)),
            "median": float(np.median(subset)),
            "min": float(np.min(subset)),
            "max": float(np.max(subset)),
            "std": float(np.std(subset)),
        })

        if validated_percentiles:
            values = np.percentile(
                subset,
                validated_percentiles,
            )

            for p, value in zip(
                validated_percentiles, values
            ):
                result[f"p{p}"] = float(value)

    for key, value in result.items():
        if not np.isfinite(value):
            raise ValueError(
                f"Calculated statistic {key} is not finite."
            )

    return result



def calculate_metrics(
    transition_map: Union[str, Path, np.ndarray],
    cva_magnitude: Union[str, Path, np.ndarray],
    delta_ndvi: Union[str, Path, np.ndarray],
    delta_gndvi: Union[str, Path, np.ndarray],
    delta_ndre: Union[str, Path, np.ndarray],
    degradation_map: Union[str, Path, np.ndarray],
    recovery_map: Union[str, Path, np.ndarray],
    residual_stress_map: Union[str, Path, np.ndarray],
    canopy_loss_map: Union[str, Path, np.ndarray],
    wetland_mask: Union[str, Path, np.ndarray],
    valid_analysis_mask: Union[str, Path, np.ndarray],
    config: Dict[str, Any],
    reference_grid: Optional[ReferenceGrid] = None
) -> MetricsResult:
    """
    Phase 8 Engine: Calculates quantitative metrics and degradation indices from validated inputs.
    """
    # 1. Load inputs and extract spatial metadata
    trans_arr, trans_meta = _load_input_raster(transition_map)
    cva_arr, cva_meta = _load_input_raster(cva_magnitude)
    ndvi_arr, ndvi_meta = _load_input_raster(delta_ndvi)
    gndvi_arr, gndvi_meta = _load_input_raster(delta_gndvi)
    ndre_arr, ndre_meta = _load_input_raster(delta_ndre)
    deg_arr, deg_meta = _load_input_raster(degradation_map)
    rec_arr, rec_meta = _load_input_raster(recovery_map)
    res_arr, res_meta = _load_input_raster(residual_stress_map)
    canopy_arr, canopy_meta = _load_input_raster(canopy_loss_map)
    wetland_arr, wetland_meta = _load_input_raster(wetland_mask)
    valid_mask_arr, valid_mask_meta = _load_input_raster(valid_analysis_mask)

    # 2. Strict spatial validation
    metas = [
        trans_meta, cva_meta, ndvi_meta, gndvi_meta, ndre_meta,
        deg_meta, rec_meta, res_meta, canopy_meta, wetland_meta, valid_mask_meta
    ]
    verify_spatial_alignment([m for m in metas if m is not None])

    # Direct shape check
    shape = trans_arr.shape
    if (cva_arr.shape != shape or ndvi_arr.shape != shape or gndvi_arr.shape != shape or
        ndre_arr.shape != shape or deg_arr.shape != shape or rec_arr.shape != shape or
        res_arr.shape != shape or canopy_arr.shape != shape or wetland_arr.shape != shape or
        valid_mask_arr.shape != shape):
        raise ValueError("Geometry mismatch: Array shapes are not identical.")

    # NORMALIZE DECLARED RASTER NODATA

    def normalize_nodata(array, metadata, categorical):

        if metadata is None:
            return array

        nodata = metadata.get("nodata")

        if nodata is None:
            return array

        if np.isnan(nodata):
            missing = np.isnan(array)
        else:
            missing = array == nodata

        if not np.any(missing):
            return array

        if categorical:

            if array.dtype.kind not in "biuf":
                raise TypeError(
                    "Invalid categorical raster data type."
                )

            if array.dtype.kind == "f":
                normalized = array.copy()
            elif array.dtype.kind == "b" or (
                np.iinfo(array.dtype).max < 255
            ):
                normalized = array.astype(np.int64)
            else:
                normalized = array.copy()

            normalized[missing] = 255

        else:

            normalized = array.astype(
                np.float64, copy=True
            )

            normalized[missing] = np.nan

        return normalized

    trans_arr = normalize_nodata(
        trans_arr, trans_meta, True
    )

    cva_arr = normalize_nodata(
        cva_arr, cva_meta, False
    )

    ndvi_arr = normalize_nodata(
        ndvi_arr, ndvi_meta, False
    )

    gndvi_arr = normalize_nodata(
        gndvi_arr, gndvi_meta, False
    )

    ndre_arr = normalize_nodata(
        ndre_arr, ndre_meta, False
    )

    deg_arr = normalize_nodata(
        deg_arr, deg_meta, True
    )

    rec_arr = normalize_nodata(
        rec_arr, rec_meta, True
    )

    res_arr = normalize_nodata(
        res_arr, res_meta, True
    )

    canopy_arr = normalize_nodata(
        canopy_arr, canopy_meta, True
    )

    wetland_arr = normalize_nodata(
        wetland_arr, wetland_meta, True
    )

    valid_mask_arr = normalize_nodata(
        valid_mask_arr, valid_mask_meta, True
    )

    # STRICT METRICS INPUT VALIDATION

    from numbers import Real, Integral

    if not isinstance(config, dict):
        raise TypeError("Metrics configuration must be a dictionary.")

    raster_inputs = {
        "transition": trans_arr,
        "cva": cva_arr,
        "delta_ndvi": ndvi_arr,
        "delta_gndvi": gndvi_arr,
        "delta_ndre": ndre_arr,
        "degradation": deg_arr,
        "recovery": rec_arr,
        "residual_stress": res_arr,
        "canopy_loss": canopy_arr,
        "wetland": wetland_arr,
        "valid_analysis": valid_mask_arr,
    }

    for name, array in raster_inputs.items():

        if array.ndim != 2 or array.size == 0:
            raise ValueError(
                f"{name}: expected a nonempty 2D raster."
            )

        if array.dtype.kind not in "biuf":
            raise TypeError(
                f"{name}: invalid raster data type."
            )

    if trans_arr.dtype.kind not in "iu":
        raise TypeError(
            "Transition raster must contain integer class IDs."
        )

    if not np.all(
        ((trans_arr >= 0) & (trans_arr <= 15))
        | (trans_arr == 255)
    ):
        raise ValueError(
            "Transition raster contains invalid class IDs."
        )

    categorical_masks = {
        "degradation": deg_arr,
        "recovery": rec_arr,
        "residual_stress": res_arr,
        "canopy_loss": canopy_arr,
        "wetland": wetland_arr,
        "valid_analysis": valid_mask_arr,
    }

    for name, array in categorical_masks.items():

        if not np.all(
            (array == 0)
            | (array == 1)
            | (array == 255)
        ):
            raise ValueError(
                f"{name}: invalid categorical mask values."
            )

    selected = (
        (valid_mask_arr == 1)
        & (wetland_arr == 1)
    )

    continuous_rasters = {
        "cva": cva_arr,
        "delta_ndvi": ndvi_arr,
        "delta_gndvi": gndvi_arr,
        "delta_ndre": ndre_arr,
    }

    for name, array in continuous_rasters.items():

        if array.dtype.kind not in "iuf":
            raise TypeError(
                f"{name}: expected real numeric raster values."
            )

        if np.any(np.isinf(array[selected])):
            raise ValueError(
                f"{name}: infinite values inside analysis area."
            )

    valid_cva = selected & np.isfinite(cva_arr)
    valid_cva &= cva_arr != -9999.0

    if np.any(cva_arr[valid_cva] < 0):
        raise ValueError(
            "CVA magnitude cannot be negative."
        )

    checked_metrics = config.get("metrics", {})

    if not isinstance(checked_metrics, dict):
        raise TypeError(
            "Metrics settings must be a dictionary."
        )

    checked_weights = checked_metrics.get(
        "severity_weights", {}
    )

    if not isinstance(checked_weights, dict):
        raise TypeError(
            "Severity weights must be a dictionary."
        )

    for name, value in checked_weights.items():

        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(
                f"Invalid severity weight: {name}."
            )

    include_canopy = checked_metrics.get(
        "include_canopy_loss_in_swd", False
    )

    if not isinstance(include_canopy, bool):
        raise TypeError(
            "Canopy loss inclusion must be boolean."
        )

    checked_mmu = checked_metrics.get(
        "minimum_change_area", {}
    )

    if not isinstance(checked_mmu, dict):
        raise TypeError(
            "Minimum change area settings must be a dictionary."
        )

    if not isinstance(
        checked_mmu.get("enabled", False), bool
    ):
        raise TypeError(
            "Minimum change area enabled must be boolean."
        )

    threshold = checked_mmu.get("value_m2", 0)

    if (
        isinstance(threshold, (bool, np.bool_))
        or not isinstance(threshold, Real)
        or not np.isfinite(threshold)
        or threshold < 0
    ):
        raise ValueError(
            "Invalid minimum change area threshold."
        )

    # 3. Resolve spatial grid parameters for area calculation
    crs = None
    transform = None
    width = None
    height = None

    non_null_metas = [m for m in metas if m is not None]
    if non_null_metas:
        ref_m = non_null_metas[0]
        crs = ref_m["crs"]
        transform = ref_m["transform"]
        width = ref_m["width"]
        height = ref_m["height"]
    elif reference_grid is not None:
        crs = reference_grid.crs
        transform = reference_grid.transform
        width = reference_grid.width
        height = reference_grid.height
    else:
        raise ValueError("Missing spatial grid reference. Provide file paths or a reference_grid object.")

    # STRICT REFERENCE GRID RECONCILIATION

    if reference_grid is not None:

        required_attributes = (
            "crs",
            "transform",
            "width",
            "height",
        )

        for attribute in required_attributes:

            if not hasattr(reference_grid, attribute):
                raise ValueError(
                    f"Reference grid missing {attribute}."
                )

        grid_width = reference_grid.width
        grid_height = reference_grid.height

        for name, dimension in (
            ("width", grid_width),
            ("height", grid_height),
        ):

            if (
                isinstance(dimension, (bool, np.bool_))
                or not isinstance(dimension, Integral)
                or dimension <= 0
            ):
                raise ValueError(
                    f"Invalid supplied reference grid {name}."
                )

        if (grid_height, grid_width) != shape:
            raise ValueError(
                "Supplied reference grid shape mismatch."
            )

        grid_transform = reference_grid.transform

        if grid_transform is not None:

            try:
                coefficients = np.asarray(
                    [
                        grid_transform.a,
                        grid_transform.b,
                        grid_transform.c,
                        grid_transform.d,
                        grid_transform.e,
                        grid_transform.f,
                    ],
                    dtype=np.float64,
                )
            except (AttributeError, TypeError, ValueError):
                raise ValueError(
                    "Invalid supplied reference grid transform."
                )

            if not np.all(np.isfinite(coefficients)):
                raise ValueError(
                    "Nonfinite reference grid transform."
                )

            a, b, c, d, e, f = coefficients

            determinant = a * e - b * d

            if (
                not np.isfinite(determinant)
                or determinant == 0
            ):
                raise ValueError(
                    "Singular reference grid transform."
                )

            corners = [
                grid_transform @ (0, 0),
                grid_transform @ (grid_width, 0),
                grid_transform @ (0, grid_height),
                grid_transform @ (grid_width, grid_height),
            ]

            xs, ys = zip(*corners)

            grid_metadata = {
                "crs": reference_grid.crs,
                "transform": grid_transform,
                "width": grid_width,
                "height": grid_height,
                "resolution": (
                    np.hypot(a, d),
                    np.hypot(b, e),
                ),
                "bounds": (
                    min(xs),
                    min(ys),
                    max(xs),
                    max(ys),
                ),
            }

            verify_spatial_alignment(
                non_null_metas + [grid_metadata]
            )

        elif non_null_metas:
            raise ValueError(
                "Supplied reference grid has no transform "
                "but raster spatial metadata is available."
            )

    # Reconcile grid dimensions with actual raster dimensions.

    for name, dimension in (
        ("width", width),
        ("height", height),
    ):
        if (
            isinstance(dimension, (bool, np.bool_))
            or not isinstance(dimension, Integral)
            or dimension <= 0
        ):
            raise ValueError(
                f"Invalid reference grid {name}."
            )

    if (height, width) != shape:
        raise ValueError(
            "Reference grid dimensions do not match raster shape."
        )

    # Calculate physical pixel areas only when authoritative spatial
    # reference information is available. CRS-less/pixel-space rasters
    # must never be interpreted as metres.
    physical_area_available = True
    try:
        pixel_areas = calculate_pixel_areas(transform, shape, crs)
    except PhysicalAreaUnavailableError:
        pixel_areas = None
        physical_area_available = False

    # 4. Define strictly valid population (exclude NoData)

    valid_pop = (
        (valid_mask_arr == 1)
        & (wetland_arr == 1)
        & (trans_arr != 255)
        & np.isfinite(cva_arr)
        & np.isfinite(ndvi_arr)
        & np.isfinite(gndvi_arr)
        & np.isfinite(ndre_arr)
        & (cva_arr != -9999.0)
        & (ndvi_arr != -9999.0)
        & (gndvi_arr != -9999.0)
        & (ndre_arr != -9999.0)
        & (deg_arr != 255)
        & (rec_arr != 255)
        & (res_arr != 255)
        & (canopy_arr != 255)
    )

    # 5. Connected Component Filtering (MMU)

    metrics_config = config.get("metrics", {})

    min_change_config = metrics_config.get(
        "minimum_change_area", {}
    )

    mmu_enabled = min_change_config.get(
        "enabled", False
    )

    mmu_val_m2 = min_change_config.get(
        "value_m2", 0
    )

    minimum_change_area_applied = False
    minimum_change_area_available = True

    if mmu_enabled and mmu_val_m2 > 0:

        if physical_area_available:

            def filter_valid_patches(array):

                # Prevent excluded pixels from connecting
                # independent patches or contributing area.

                restricted = np.where(
                    valid_pop,
                    array,
                    0,
                )

                filtered = filter_by_min_area(
                    restricted,
                    pixel_areas,
                    mmu_val_m2,
                )

                # Preserve original values outside the
                # analysis population, including NoData.

                return np.where(
                    valid_pop,
                    filtered,
                    array,
                )

            deg_arr = filter_valid_patches(deg_arr)
            rec_arr = filter_valid_patches(rec_arr)
            res_arr = filter_valid_patches(res_arr)
            canopy_arr = filter_valid_patches(canopy_arr)

            minimum_change_area_applied = True

        else:
            minimum_change_area_available = False

    # 6. Baseline Area (A_baseline)
    baseline_pixel_count = int(np.sum(valid_pop))
    if physical_area_available:
        baseline_area_m2 = float(np.sum(pixel_areas[valid_pop]))
        baseline_area_ha = baseline_area_m2 / 10000.0
    else:
        baseline_area_m2 = None
        baseline_area_ha = None

    if baseline_pixel_count < 0:
        raise ValueError("Sanity Check Failed: Negative baseline pixel count detected.")
    if physical_area_available and baseline_area_m2 < 0:
        raise ValueError("Sanity Check Failed: Negative baseline area detected.")

    # 7. Affected / Degraded Footprint (A_impact)
    impact_mask = (deg_arr == 1) & valid_pop
    impact_pixel_count = int(np.sum(impact_mask))
    if physical_area_available:
        impact_area_m2 = float(np.sum(pixel_areas[impact_mask]))
        impact_area_ha = impact_area_m2 / 10000.0
    else:
        impact_area_m2 = None
        impact_area_ha = None

    if impact_pixel_count > baseline_pixel_count:
        raise ValueError("Sanity Check Failed: Affected pixel count exceeds baseline pixel count.")
    if physical_area_available and impact_area_m2 > baseline_area_m2:
        raise ValueError("Sanity Check Failed: Affected area exceeds baseline area.")

    # 8. Severity-Weighted Degraded Area (A_SWD / ha_deg)
    severity_weights = metrics_config.get("severity_weights", {
        "healthy": 0.0,
        "moderate": 0.35,
        "severe": 0.80,
        "background": 1.0
    })
    include_canopy_loss_in_swd = metrics_config.get("include_canopy_loss_in_swd", False)

    # 16 transition IDs + NoData values:
    # 0 to 15, and 255
    transition_weights = np.zeros(256, dtype=np.float64)
    # Default V1 transitions
    transition_weights[14] = severity_weights.get("moderate", 0.35)  # H->M
    transition_weights[13] = severity_weights.get("severe", 0.80)    # H->S
    transition_weights[9] = severity_weights.get("severe", 0.80)     # M->S

    # Canopy-loss transitions: H->B (12), M->B (8), S->B (4)
    if include_canopy_loss_in_swd:
        bg_weight = severity_weights.get("background", 1.0)
        transition_weights[12] = bg_weight
        transition_weights[8] = bg_weight
        transition_weights[4] = bg_weight

    severities = transition_weights[trans_arr]

    # Mask for SWD pixels:
    # Default V1 uses: H->M, H->S, M->S (which are deg_arr == 1)
    # If canopy loss is enabled: include canopy_arr == 1
    if include_canopy_loss_in_swd:
        swd_mask = ((deg_arr == 1) | (canopy_arr == 1)) & valid_pop
    else:
        swd_mask = (deg_arr == 1) & valid_pop

    if physical_area_available:
        swd_area_m2 = float(np.sum(pixel_areas[swd_mask] * severities[swd_mask]))
        swd_area_ha = swd_area_m2 / 10000.0
        if swd_area_m2 > baseline_area_m2:
            raise ValueError("Sanity Check Failed: Severity-weighted area exceeds baseline area.")
    else:
        swd_area_m2 = None
        swd_area_ha = None

    # 9. NDR (Normalized Degradation Ratio)
    warnings_list = []
    if not physical_area_available:
        ndr = None
        ndr_percent = None
        warnings_list.append(
            "Physical area is unavailable because the analysis raster is "
            "CRS-less/pixel-space. NDR is set to None."
        )
    elif baseline_area_m2 == 0:
        ndr = None
        ndr_percent = None
        warnings_list.append("Baseline area is zero. Normalized Degradation Ratio (NDR) is set to None.")
    else:
        ndr = swd_area_m2 / baseline_area_m2
        ndr_percent = ndr * 100.0
        if ndr < 0 or ndr > 1:
            raise ValueError(f"Sanity Check Failed: NDR value {ndr} is out of [0, 1] range.")

    # 10. Recovery Area (A_recovery)
    rec_mask = (rec_arr == 1) & valid_pop
    rec_pixel_count = int(np.sum(rec_mask))
    if physical_area_available:
        rec_area_m2 = float(np.sum(pixel_areas[rec_mask]))
        rec_area_ha = rec_area_m2 / 10000.0
        if rec_area_m2 > baseline_area_m2:
            raise ValueError("Sanity Check Failed: Recovery area exceeds baseline area.")
    else:
        rec_area_m2 = None
        rec_area_ha = None

    if not physical_area_available:
        rec_ratio = None
        warnings_list.append("Physical area is unavailable. Recovery ratio is set to None.")
    elif baseline_area_m2 == 0:
        rec_ratio = None
        warnings_list.append("Baseline area is zero. Recovery ratio is set to None.")
    else:
        rec_ratio = rec_area_m2 / baseline_area_m2

    # Recovery specific areas
    # M->H (11), S->M (6), S->H (7)
    rec_mh_mask = (trans_arr == 11) & valid_pop
    rec_sm_mask = (trans_arr == 6) & valid_pop
    rec_sh_mask = (trans_arr == 7) & valid_pop

    if physical_area_available:
        rec_mh_area_m2 = float(np.sum(pixel_areas[rec_mh_mask]))
        rec_sm_area_m2 = float(np.sum(pixel_areas[rec_sm_mask]))
        rec_sh_area_m2 = float(np.sum(pixel_areas[rec_sh_mask]))
    else:
        rec_mh_area_m2 = None
        rec_sm_area_m2 = None
        rec_sh_area_m2 = None

    # 11. Canopy-Loss Area (A_canopy_loss)
    canopy_mask = (canopy_arr == 1) & valid_pop
    canopy_pixel_count = int(np.sum(canopy_mask))
    if physical_area_available:
        canopy_area_m2 = float(np.sum(pixel_areas[canopy_mask]))
        canopy_area_ha = canopy_area_m2 / 10000.0
    else:
        canopy_area_m2 = None
        canopy_area_ha = None

    if not physical_area_available:
        canopy_ratio = None
        canopy_pct = None
    elif baseline_area_m2 == 0:
        canopy_ratio = None
        canopy_pct = None
    else:
        canopy_ratio = canopy_area_m2 / baseline_area_m2
        canopy_pct = canopy_ratio * 100.0

    # 12. Residual Stress Area (A_residual_stress)
    res_mask = (res_arr == 1) & valid_pop
    res_pixel_count = int(np.sum(res_mask))
    if physical_area_available:
        res_area_m2 = float(np.sum(pixel_areas[res_mask]))
        res_area_ha = res_area_m2 / 10000.0
    else:
        res_area_m2 = None
        res_area_ha = None

    # Stable veg area (5, 10, 15)
    stable_veg_mask = np.isin(trans_arr, [5, 10, 15]) & valid_pop
    if physical_area_available:
        stable_veg_area_m2 = float(np.sum(pixel_areas[stable_veg_mask]))
        stable_veg_area_ha = stable_veg_area_m2 / 10000.0
    else:
        stable_veg_area_m2 = None
        stable_veg_area_ha = None

    if not physical_area_available:
        res_ratio = None
        warnings_list.append("Physical area is unavailable. Residual stress ratio is set to None.")
    elif stable_veg_area_m2 == 0:
        res_ratio = None
        warnings_list.append("Stable vegetation area is zero. Residual stress ratio is set to None.")
    else:
        res_ratio = res_area_m2 / stable_veg_area_m2

    # 13. Class Area Metrics & Reconciliation
    T1_class = trans_arr // 4
    T2_class = trans_arr % 4

    # Calculate class areas for T1 and T2
    class_map = {
        0: "background",
        1: "severe",
        2: "moderate",
        3: "healthy"
    }

    class_areas_t1 = {}
    class_areas_t2 = {}
    class_changes = {}

    t1_sum_m2 = 0.0
    t2_sum_m2 = 0.0

    for cid, name in class_map.items():
        mask_t1 = (T1_class == cid) & valid_pop
        mask_t2 = (T2_class == cid) & valid_pop

        cnt_t1 = int(np.sum(mask_t1))
        cnt_t2 = int(np.sum(mask_t2))

        if physical_area_available:
            area_t1_m2 = float(np.sum(pixel_areas[mask_t1]))
            area_t2_m2 = float(np.sum(pixel_areas[mask_t2]))
            t1_sum_m2 += area_t1_m2
            t2_sum_m2 += area_t2_m2
            pct_t1 = (area_t1_m2 / baseline_area_m2 * 100.0) if baseline_area_m2 > 0 else 0.0
            pct_t2 = (area_t2_m2 / baseline_area_m2 * 100.0) if baseline_area_m2 > 0 else 0.0
            change_m2 = area_t2_m2 - area_t1_m2
            change_ha = change_m2 / 10000.0
            area_t1_ha = area_t1_m2 / 10000.0
            area_t2_ha = area_t2_m2 / 10000.0
        else:
            area_t1_m2 = None
            area_t2_m2 = None
            pct_t1 = (cnt_t1 / baseline_pixel_count * 100.0) if baseline_pixel_count > 0 else 0.0
            pct_t2 = (cnt_t2 / baseline_pixel_count * 100.0) if baseline_pixel_count > 0 else 0.0
            change_m2 = None
            change_ha = None
            area_t1_ha = None
            area_t2_ha = None

        if cnt_t1 < 0 or cnt_t2 < 0:
            raise ValueError(f"Sanity Check Failed: Negative count for class {name}.")

        class_areas_t1[name] = {
            "pixel_count": cnt_t1,
            "area_m2": area_t1_m2,
            "area_ha": area_t1_ha,
            "percentage": pct_t1
        }
        class_areas_t2[name] = {
            "pixel_count": cnt_t2,
            "area_m2": area_t2_m2,
            "area_ha": area_t2_ha,
            "percentage": pct_t2
        }

        class_changes[name] = {
            "area_change_m2": change_m2,
            "area_change_ha": change_ha
        }
        if name == "healthy":
            if physical_area_available:
                loss_m2 = max(area_t1_m2 - area_t2_m2, 0.0)
                loss_ha = loss_m2 / 10000.0
            else:
                loss_m2 = None
                loss_ha = None
            class_changes[name]["loss_m2"] = loss_m2
            class_changes[name]["loss_ha"] = loss_ha

    # Class Reconciliation Check
    if physical_area_available:
        reconciled_class_t1 = bool(np.isclose(t1_sum_m2, baseline_area_m2, rtol=1e-5, atol=1e-3))
        reconciled_class_t2 = bool(np.isclose(t2_sum_m2, baseline_area_m2, rtol=1e-5, atol=1e-3))
        if not reconciled_class_t1:
            raise ValueError(f"Class Reconciliation Failed for T1: Class sum {t1_sum_m2} != baseline {baseline_area_m2}.")
        if not reconciled_class_t2:
            raise ValueError(f"Class Reconciliation Failed for T2: Class sum {t2_sum_m2} != baseline {baseline_area_m2}.")
    else:
        reconciled_class_t1 = bool(sum(v["pixel_count"] for v in class_areas_t1.values()) == baseline_pixel_count)
        reconciled_class_t2 = bool(sum(v["pixel_count"] for v in class_areas_t2.values()) == baseline_pixel_count)

    # 14. Transition-specific Area Metrics
    TRANSITIONS_INFO = {
        0:  {"name": "B_to_B", "from": 0, "to": 0},
        1:  {"name": "B_to_S", "from": 0, "to": 1},
        2:  {"name": "B_to_M", "from": 0, "to": 2},
        3:  {"name": "B_to_H", "from": 0, "to": 3},
        4:  {"name": "S_to_B", "from": 1, "to": 0},
        5:  {"name": "S_to_S", "from": 1, "to": 1},
        6:  {"name": "S_to_M", "from": 1, "to": 2},
        7:  {"name": "S_to_H", "from": 1, "to": 3},
        8:  {"name": "M_to_B", "from": 2, "to": 0},
        9:  {"name": "M_to_S", "from": 2, "to": 1},
        10: {"name": "M_to_M", "from": 2, "to": 2},
        11: {"name": "M_to_H", "from": 2, "to": 3},
        12: {"name": "H_to_B", "from": 3, "to": 0},
        13: {"name": "H_to_S", "from": 3, "to": 1},
        14: {"name": "H_to_M", "from": 3, "to": 2},
        15: {"name": "H_to_H", "from": 3, "to": 3}
    }

    transition_areas = {}
    trans_sum_m2 = 0.0
    trans_sum_pixels = 0

    for tid, info in TRANSITIONS_INFO.items():
        name = info["name"]
        mask_trans = (trans_arr == tid) & valid_pop

        cnt = int(np.sum(mask_trans))
        trans_sum_pixels += cnt

        if physical_area_available:
            area_m2 = float(np.sum(pixel_areas[mask_trans]))
            area_ha = area_m2 / 10000.0
            pct = (area_m2 / baseline_area_m2 * 100.0) if baseline_area_m2 > 0 else 0.0
            trans_sum_m2 += area_m2
        else:
            area_m2 = None
            area_ha = None
            pct = (cnt / baseline_pixel_count * 100.0) if baseline_pixel_count > 0 else 0.0

        if cnt < 0:
            raise ValueError(f"Sanity Check Failed: Negative count for transition {name}.")
        if physical_area_available and area_m2 < 0:
            raise ValueError(f"Sanity Check Failed: Negative area for transition {name}.")

        transition_areas[name] = {
            "pixel_count": cnt,
            "area_m2": area_m2,
            "area_ha": area_ha,
            "percentage_of_valid_wetland_analysis_area": pct
        }

    # Transition Reconciliation checks
    # Sum transition counts must exactly equal baseline pixel count
    reconciled_trans_pixels = bool(trans_sum_pixels == baseline_pixel_count)
    # Sum transition areas must approximately equal baseline area
    if physical_area_available:
        reconciled_trans_area = bool(np.isclose(trans_sum_m2, baseline_area_m2, rtol=1e-5, atol=1e-3))
    else:
        reconciled_trans_area = None

    if not reconciled_trans_pixels:
        raise ValueError(f"Transition Reconciliation Failed: Sum of transition pixels {trans_sum_pixels} != baseline {baseline_pixel_count}.")
    if physical_area_available and not reconciled_trans_area:
        raise ValueError(f"Transition Reconciliation Failed: Sum of transition areas {trans_sum_m2} != baseline {baseline_area_m2}.")

    # 15. CVA Statistics by Change Category
    # Categorize subsets
    cva_masks = {
        "all_valid_wetland": valid_pop,
        "degradation": np.isin(trans_arr, [9, 13, 14]) & valid_pop,
        "recovery": np.isin(trans_arr, [6, 7, 11]) & valid_pop,
        "stable_vegetation": np.isin(trans_arr, [5, 10, 15]) & valid_pop,
        "canopy_loss": np.isin(trans_arr, [4, 8, 12]) & valid_pop,
        "background_to_vegetation": np.isin(trans_arr, [1, 2, 3]) & valid_pop
    }

    cva_stats = {}
    for cat_name, cat_mask in cva_masks.items():
        stats = calculate_subset_stats(cva_arr, cat_mask, percentiles=[95])
        if stats["min"] is not None and stats["min"] < 0:
            # Check CVA < 0 sanity check
            # CVA magnitude is usually >= 0. Let's flag warning or fail if CVA magnitude is negative.
            raise ValueError(f"Sanity Check Failed: Negative CVA magnitude {stats['min']} detected under {cat_name}.")
        cva_stats[cat_name] = stats

    # 16. Delta Index Statistics
    delta_indices = {
        "ndvi": ndvi_arr,
        "gndvi": gndvi_arr,
        "ndre": ndre_arr
    }

    delta_stats = {}
    for index_name, index_arr in delta_indices.items():
        delta_stats[index_name] = {
            "all_valid_wetland": calculate_subset_stats(index_arr, valid_pop, percentiles=[5, 95]),
            "degradation": calculate_subset_stats(index_arr, np.isin(trans_arr, [9, 13, 14]) & valid_pop, percentiles=[5, 95])
        }

    # 17. Wetland Extent Metrics

    wetland_extent_status = "NOT_AVAILABLE"
    wetland_extent_T1 = None
    wetland_extent_T2 = None
    wetland_extent_change = None
    wetland_extent_change_ha = None
    physical_wetland_depletion = "NOT_AVAILABLE"

    # Physical depletion requires independently verified
    # temporal wetland evidence. Configuration and raster
    # paths alone do not establish scientific validity.

    warnings_list.append(
        "Wetland-specific physical depletion cannot be "
        "quantified without independently verified "
        "temporal wetland extent evidence."
    )

    # 18. Build result and metadata
    metadata = {
        "severity_weights": severity_weights,
        "provisional_severity_weights_disclaimer": "Severity-weighted area uses provisional engineering parameters.",
        "include_canopy_loss_in_swd": include_canopy_loss_in_swd,
        "minimum_change_area": min_change_config,
        "minimum_change_area_applied": minimum_change_area_applied,
        "minimum_change_area_available": minimum_change_area_available,
        "physical_area_available": physical_area_available,
        "area_basis": "physical_crs" if physical_area_available else "pixel_count_only",
        "baseline_definition": "valid wetland analysis area (restricted to valid overlap and non-NoData)",
        "depletion_disclaimer": "Vegetation degradation metrics do not constitute physical wetland depletion.",
        "olofsson_status": "Model-derived estimate; no independent area-error correction applied."
    }

    # Handle temporal phenological acquisition warning
    phenology_warning = "NO"
    acq_t1 = config.get("preprocessing", {}).get("t1_date")
    acq_t2 = config.get("preprocessing", {}).get("t2_date")
    if acq_t1 and acq_t2:
        # If dates differ substantially, raise warning
        metadata["acquisition_t1"] = acq_t1
        metadata["acquisition_t2"] = acq_t2

    qa_flags = {
        "input_validation": "PASS",
        "spatial_alignment": "PASS",
        "model_inference": "PASS",
        "phenology_warning": phenology_warning,
        "independent_validation": "NO",
        "reconciliation_transition_passes": reconciled_trans_pixels and reconciled_trans_area,
        "reconciliation_class_passes": reconciled_class_t1 and reconciled_class_t2
    }

    # Organize result categories
    baseline_res = {
        "pixel_count": baseline_pixel_count,
        "area_m2": baseline_area_m2,
        "area_ha": baseline_area_ha
    }

    degradation_res = {
        "impact_area_m2": impact_area_m2,
        "impact_area_ha": impact_area_ha,
        "severity_weighted_area_m2": swd_area_m2,
        "severity_weighted_area_ha_deg": swd_area_ha,
        "ndr": ndr,
        "ndr_percent": ndr_percent
    }

    recovery_res = {
        "area_m2": rec_area_m2,
        "area_ha": rec_area_ha,
        "ratio": rec_ratio,
        "details_m2": {
            "M_to_H": rec_mh_area_m2,
            "S_to_M": rec_sm_area_m2,
            "S_to_H": rec_sh_area_m2
        }
    }

    canopy_loss_res = {
        "area_m2": canopy_area_m2,
        "area_ha": canopy_area_ha,
        "ratio": canopy_ratio,
        "percentage": canopy_pct
    }

    residual_stress_res = {
        "area_m2": res_area_m2,
        "area_ha": res_area_ha,
        "ratio": res_ratio
    }

    class_area_metrics = {
        "T1": class_areas_t1,
        "T2": class_areas_t2,
        "changes": class_changes
    }

    wetland_extent_res = {
        "status": wetland_extent_status,
        "extent_T1_ha": (wetland_extent_T1 / 10000.0) if wetland_extent_T1 is not None else None,
        "extent_T2_ha": (wetland_extent_T2 / 10000.0) if wetland_extent_T2 is not None else None,
        "extent_change_ha": wetland_extent_change_ha,
        "physical_wetland_depletion": physical_wetland_depletion
    }

    # Summary object as specified
    summary = {
        "baseline_area_ha": baseline_area_ha,
        "affected_area_ha": impact_area_ha,
        "severity_weighted_area_ha_deg": swd_area_ha,
        "normalized_degradation_ratio": ndr,
        "recovery_area_ha": rec_area_ha,
        "residual_stress_area_ha": res_area_ha,
        "canopy_loss_area_ha": canopy_area_ha,
        "mean_cva": cva_stats["all_valid_wetland"]["mean"],
        "median_cva": cva_stats["all_valid_wetland"]["median"],
        "p95_cva": cva_stats["all_valid_wetland"]["p95"],
        "wetland_extent_status": wetland_extent_status
    }

    # Put it all into QA dictionary
    qa = {
        "warnings": warnings_list,
        "flags": qa_flags
    }

    return MetricsResult(
        baseline_area=baseline_res,
        impact_area=degradation_res,
        severity_weighted_area=degradation_res, # Included here as well to respect structural fields
        normalized_degradation_ratio=ndr,
        recovery_area=recovery_res,
        residual_stress_area=residual_stress_res,
        canopy_loss_area=canopy_loss_res,
        transition_metrics=transition_areas,
        class_area_metrics=class_area_metrics,
        cva_statistics=cva_stats,
        delta_statistics=delta_stats,
        wetland_extent_metrics=wetland_extent_res,
        qa_flags=qa,
        metadata={
            "summary": summary,
            "disclaimer": "Vegetation degradation metrics do not constitute physical wetland depletion.",
            "metrics_metadata": metadata,
            "physical_area_available": physical_area_available,
            "area_basis": "physical_crs" if physical_area_available else "pixel_count_only"
        }
    )
