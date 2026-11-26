
import json
import math
from numbers import Real
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import rasterio

from gis_engine.metrics.area import (
    PhysicalAreaUnavailableError,
    calculate_pixel_areas,
)


NODATA_FLOAT = -9999.0
NODATA_UINT8 = 255


def _validate_number(
    value,
    name,
    minimum=None,
    maximum=None,
):
    """Validate finite numeric configuration values."""

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError(
            f"{name} must be a finite number."
        )

    value = float(value)

    if minimum is not None and value < minimum:
        raise ValueError(
            f"{name} must be at least {minimum}."
        )

    if maximum is not None and value > maximum:
        raise ValueError(
            f"{name} must not exceed {maximum}."
        )

    return value


def _validate_array(array, name):
    """Validate numeric input arrays."""

    if not isinstance(array, np.ndarray):
        raise ValueError(
            f"{name} must be a NumPy array."
        )

    if array.ndim not in (1, 2) or array.size == 0:
        raise ValueError(
            f"{name} must be a nonempty 1D or 2D array."
        )

    if array.dtype.kind not in "iuf":
        raise ValueError(
            f"{name} must contain real numeric values."
        )

    return array.astype(np.float64, copy=False)


def _read_raster(path, reference=None):
    """Read one raster band and preserve its validity mask."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Required raster not found: {path}"
        )

    with rasterio.open(path) as src:

        if src.count != 1:
            raise ValueError(
                f"Raster must contain one band: {path.name}"
            )

        if src.width <= 0 or src.height <= 0:
            raise ValueError(
                f"Invalid raster dimensions: {path.name}"
            )

        affine = src.transform

        coefficients = np.asarray(
            tuple(affine),
            dtype=np.float64,
        )

        determinant = (
            affine.a * affine.e
            - affine.b * affine.d
        )

        if (
            not np.all(np.isfinite(coefficients))
            or not math.isfinite(determinant)
            or determinant == 0
        ):
            raise ValueError(
                f"Invalid raster transform: {path.name}"
            )

        metadata = {
            "width": src.width,
            "height": src.height,
            "crs": src.crs,
            "transform": affine,
        }

        if reference is not None:

            for key in (
                "width",
                "height",
                "crs",
                "transform",
            ):

                if metadata[key] != reference[key]:
                    raise ValueError(
                        f"Raster alignment mismatch "
                        f"({key}): {path.name}"
                    )

        raw = src.read(1)

        if raw.dtype.kind not in "iuf":
            raise ValueError(
                f"Unsupported raster dtype: {path.name}"
            )

        valid = src.read_masks(1) > 0

        if src.nodata is not None:

            if np.isnan(src.nodata):
                valid &= ~np.isnan(raw)

            else:
                valid &= raw != src.nodata

        valid &= np.isfinite(raw)

        return raw, valid, metadata


def _read_binary_mask(path, reference):
    """Read an aligned binary mask without treating NoData as valid."""

    raw, valid, _ = _read_raster(
        path,
        reference,
    )

    if not np.all(
        np.isin(raw[valid], [0, 1, 255])
    ):
        raise ValueError(
            f"Invalid binary mask values: {path}"
        )

    return valid & (raw == 1)


def calculate_ndwi(
    green: np.ndarray,
    nir: np.ndarray,
    nodata_mask: Optional[np.ndarray] = None,
    epsilon: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate NDWI with explicit validity protection.

    Invalid pixels are represented by -9999.0.
    """

    epsilon = _validate_number(
        epsilon,
        "epsilon",
        minimum=0.0,
    )

    green = _validate_array(
        green,
        "Green band",
    )

    nir = _validate_array(
        nir,
        "NIR band",
    )

    if green.shape != nir.shape:
        raise ValueError(
            "Green and NIR shapes must match."
        )

    valid_mask = (
        np.isfinite(green)
        & np.isfinite(nir)
    )

    if nodata_mask is not None:

        if (
            not isinstance(nodata_mask, np.ndarray)
            or nodata_mask.shape != green.shape
            or nodata_mask.dtype.kind != "b"
        ):
            raise ValueError(
                "NoData mask must be a Boolean array "
                "matching the input shape."
            )

        valid_mask &= ~nodata_mask

    with np.errstate(
        over="ignore",
        invalid="ignore",
        divide="ignore",
    ):

        raw_denom = green + nir
        denom = raw_denom + epsilon

        valid_mask &= (
            np.isfinite(raw_denom)
            & np.isfinite(denom)
            & (raw_denom > 1e-8)
            & (denom > 1e-8)
        )

        ndwi = np.full(
            green.shape,
            NODATA_FLOAT,
            dtype=np.float32,
        )

        values = (
            green[valid_mask] - nir[valid_mask]
        ) / denom[valid_mask]

        finite = np.isfinite(values)

        positions = np.flatnonzero(valid_mask)

        valid_mask.flat[
            positions[~finite]
        ] = False

        ndwi.flat[positions[finite]] = np.clip(
            values[finite],
            -1.0,
            1.0,
        ).astype(np.float32)

    return ndwi, valid_mask


def classify_optical_water(
    green_path: Path,
    nir_path: Path,
    threshold: float = 0.1,
    nodata: Optional[float] = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Classify water from aligned Green and NIR rasters."""

    threshold = _validate_number(
        threshold,
        "NDWI threshold",
        minimum=-1.0,
        maximum=1.0,
    )

    green, green_valid, reference = _read_raster(
        green_path
    )

    nir, nir_valid, _ = _read_raster(
        nir_path,
        reference,
    )

    if reference["crs"] is None:
        raise ValueError(
            "Optical water classification requires a CRS."
        )

    valid = green_valid & nir_valid

    if nodata is not None:

        if (
            isinstance(nodata, (bool, np.bool_))
            or not isinstance(nodata, Real)
            or math.isinf(nodata)
        ):
            raise ValueError(
                "Invalid explicit NoData value."
            )

        if np.isnan(nodata):

            valid &= (
                ~np.isnan(green)
                & ~np.isnan(nir)
            )

        else:

            valid &= (
                (green != nodata)
                & (nir != nodata)
            )

    ndwi, valid_mask = calculate_ndwi(
        green,
        nir,
        nodata_mask=~valid,
    )

    water_mask = (
        (ndwi >= threshold)
        & valid_mask
    )

    return water_mask, valid_mask


def analyze_hydrology(
    t1_aligned_dir: Path,
    t2_aligned_dir: Path,
    output_dir: Path,
    config: Dict[str, Any],
    wetland_mask_path: Optional[Path] = None,
    overlap_mask_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Analyze temporal surface-water changes.

    Transition classes:
        0   = stable non-water
        1   = stable water
        2   = water gain
        3   = water loss
        255 = invalid / NoData
    """

    t1_aligned_dir = Path(t1_aligned_dir)
    t2_aligned_dir = Path(t2_aligned_dir)

    if output_dir is not None:
        output_dir = Path(output_dir)

    if not isinstance(config, dict):
        raise ValueError(
            "Hydrology configuration must be a dictionary."
        )

    water_config = config.get("water", {})

    if not isinstance(water_config, dict):
        raise ValueError(
            "Water configuration must be a dictionary."
        )

    threshold = _validate_number(
        water_config.get("ndwi_threshold", 0.0),
        "NDWI threshold",
        minimum=-1.0,
        maximum=1.0,
    )

    epsilon = _validate_number(
        water_config.get("epsilon", 1e-6),
        "epsilon",
        minimum=0.0,
    )

    # Read all four rasters against one reference grid.

    t1_green, t1_green_valid, reference = _read_raster(
        t1_aligned_dir / "green.tif"
    )

    t1_nir, t1_nir_valid, _ = _read_raster(
        t1_aligned_dir / "nir.tif",
        reference,
    )

    t2_green, t2_green_valid, _ = _read_raster(
        t2_aligned_dir / "green.tif",
        reference,
    )

    t2_nir, t2_nir_valid, _ = _read_raster(
        t2_aligned_dir / "nir.tif",
        reference,
    )

    crs = reference["crs"]
    transform = reference["transform"]
    width = reference["width"]
    height = reference["height"]

    shape = (height, width)

    # Preserve each band's independent NoData mask.

    ndwi_t1, valid_t1 = calculate_ndwi(
        t1_green,
        t1_nir,
        nodata_mask=~(
            t1_green_valid & t1_nir_valid
        ),
        epsilon=epsilon,
    )

    ndwi_t2, valid_t2 = calculate_ndwi(
        t2_green,
        t2_nir,
        nodata_mask=~(
            t2_green_valid & t2_nir_valid
        ),
        epsilon=epsilon,
    )

    final_valid = valid_t1 & valid_t2

    if overlap_mask_path is not None:

        final_valid &= _read_binary_mask(
            overlap_mask_path,
            reference,
        )

    # Validate optional wetland membership before writing outputs.

    wetland_sub = None

    if wetland_mask_path is not None:

        wetland_sub = _read_binary_mask(
            wetland_mask_path,
            reference,
        )

    # Classify water using the common temporal population.

    water_mask_t1 = np.full(
        shape,
        NODATA_UINT8,
        dtype=np.uint8,
    )

    water_mask_t2 = np.full(
        shape,
        NODATA_UINT8,
        dtype=np.uint8,
    )

    water_mask_t1[final_valid] = (
        ndwi_t1[final_valid] >= threshold
    ).astype(np.uint8)

    water_mask_t2[final_valid] = (
        ndwi_t2[final_valid] >= threshold
    ).astype(np.uint8)

    delta_ndwi = np.full(
        shape,
        NODATA_FLOAT,
        dtype=np.float32,
    )

    delta_ndwi[final_valid] = (
        ndwi_t2[final_valid]
        - ndwi_t1[final_valid]
    )

    water_change = np.full(
        shape,
        NODATA_UINT8,
        dtype=np.uint8,
    )

    compact_ids = (
        water_mask_t1[final_valid] * 2
        + water_mask_t2[final_valid]
    )

    public_mapping = np.array(
        [0, 2, 3, 1],
        dtype=np.uint8,
    )

    water_change[final_valid] = (
        public_mapping[compact_ids]
    )

    # Calculate physical area only when available.

    physical_area_available = True
    area_basis = "physical_area"
    pixel_areas = None

    try:

        pixel_areas = calculate_pixel_areas(
            transform,
            shape,
            crs,
        )

        if (
            pixel_areas.shape != shape
            or not np.all(np.isfinite(pixel_areas))
            or np.any(pixel_areas <= 0)
        ):
            raise ValueError(
                "Invalid physical pixel areas."
            )

    except PhysicalAreaUnavailableError:

        physical_area_available = False
        area_basis = "pixel_count"

    valid_pixel_count = int(
        np.count_nonzero(final_valid)
    )

    nodata_pixel_count = int(
        np.count_nonzero(~final_valid)
    )

    def get_metrics_dict(subset_mask):

        sub_valid = final_valid & subset_mask

        subset_count = int(
            np.count_nonzero(sub_valid)
        )

        w_t1 = sub_valid & (water_mask_t1 == 1)
        w_t2 = sub_valid & (water_mask_t2 == 1)

        w_gain = sub_valid & (water_change == 2)
        w_loss = sub_valid & (water_change == 3)

        w_t1_pixels = int(np.count_nonzero(w_t1))
        w_t2_pixels = int(np.count_nonzero(w_t2))

        w_gain_pixels = int(np.count_nonzero(w_gain))
        w_loss_pixels = int(np.count_nonzero(w_loss))

        if (
            w_t2_pixels - w_t1_pixels
            != w_gain_pixels - w_loss_pixels
        ):
            raise ValueError(
                "Water transition counts do not reconcile."
            )

        result = {
            "status": (
                "AVAILABLE"
                if subset_count > 0
                else "NOT_AVAILABLE"
            ),
            "valid_analysis_pixel_count": subset_count,
            "water_pixel_count_t1": w_t1_pixels,
            "water_pixel_count_t2": w_t2_pixels,
            "water_gain_pixel_count": w_gain_pixels,
            "water_loss_pixel_count": w_loss_pixels,
            "area_basis": area_basis,
            "physical_area_available": (
                physical_area_available
            ),
            "water_area_t1_m2": None,
            "water_area_t2_m2": None,
            "water_area_t1_ha": None,
            "water_area_t2_ha": None,
            "water_gain_m2": None,
            "water_loss_m2": None,
            "water_gain_ha": None,
            "water_loss_ha": None,
            "net_water_change_m2": None,
            "net_water_change_ha": None,
        }

        if (
            physical_area_available
            and pixel_areas is not None
            and subset_count > 0
        ):

            w_t1_m2 = float(
                np.sum(pixel_areas[w_t1], dtype=np.float64)
            )

            w_t2_m2 = float(
                np.sum(pixel_areas[w_t2], dtype=np.float64)
            )

            w_gain_m2 = float(
                np.sum(pixel_areas[w_gain], dtype=np.float64)
            )

            w_loss_m2 = float(
                np.sum(pixel_areas[w_loss], dtype=np.float64)
            )

            net_change_m2 = w_t2_m2 - w_t1_m2

            if not math.isclose(
                net_change_m2,
                w_gain_m2 - w_loss_m2,
                rel_tol=1e-8,
                abs_tol=max(
                    1e-3,
                    1e-12 * max(w_t1_m2, w_t2_m2),
                ),
            ):
                raise ValueError(
                    "Water transition areas do not reconcile."
                )

            result.update({
                "water_area_t1_m2": w_t1_m2,
                "water_area_t2_m2": w_t2_m2,
                "water_area_t1_ha": w_t1_m2 / 10000.0,
                "water_area_t2_ha": w_t2_m2 / 10000.0,
                "water_gain_m2": w_gain_m2,
                "water_loss_m2": w_loss_m2,
                "water_gain_ha": w_gain_m2 / 10000.0,
                "water_loss_ha": w_loss_m2 / 10000.0,
                "net_water_change_m2": net_change_m2,
                "net_water_change_ha": (
                    net_change_m2 / 10000.0
                ),
            })

        return result

    analysis_metrics = get_metrics_dict(
        np.ones(shape, dtype=bool)
    )

    wetland_boundary_used = (
        wetland_sub is not None
        and bool(np.any(wetland_sub))
    )

    t1_parts = t1_aligned_dir.parent.name.split("_")
    t2_parts = t2_aligned_dir.parent.name.split("_")

    metrics_content = {
        "index": "NDWI",
        "t1_date": (
            t1_parts[0]
            if len(t1_parts) > 1
            else "T1"
        ),
        "t2_date": (
            t2_parts[1]
            if len(t2_parts) > 1
            else "T2"
        ),
        "threshold": threshold,
        "epsilon": epsilon,
        "valid_pixel_count": valid_pixel_count,
        "nodata_pixel_count": nodata_pixel_count,
        "physical_area_available": physical_area_available,
        "area_basis": area_basis,
        "wetland_boundary_used": wetland_boundary_used,
        **analysis_metrics,
    }

    if wetland_boundary_used:

        metrics_content["wetland_restricted_metrics"] = (
            get_metrics_dict(wetland_sub)
        )

    # Export existing public output filenames and formats.

    if output_dir is not None:

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        profile_float = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": rasterio.float32,
            "crs": crs,
            "transform": transform,
            "nodata": NODATA_FLOAT,
            "compress": "lzw",
        }

        profile_uint8 = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": rasterio.uint8,
            "crs": crs,
            "transform": transform,
            "nodata": NODATA_UINT8,
            "compress": "lzw",
        }

        float_outputs = {
            "ndwi_t1.tif": ndwi_t1,
            "ndwi_t2.tif": ndwi_t2,
            "delta_ndwi.tif": delta_ndwi,
        }

        categorical_outputs = {
            "water_mask_t1.tif": water_mask_t1,
            "water_mask_t2.tif": water_mask_t2,
            "water_change.tif": water_change,
        }

        for filename, array in float_outputs.items():

            with rasterio.open(
                output_dir / filename,
                "w",
                **profile_float,
            ) as dst:

                dst.write(array, 1)

        for filename, array in categorical_outputs.items():

            with rasterio.open(
                output_dir / filename,
                "w",
                **profile_uint8,
            ) as dst:

                dst.write(array, 1)

        with open(
            output_dir / "water_metrics.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metrics_content,
                file,
                indent=4,
                allow_nan=False,
            )

    return metrics_content
