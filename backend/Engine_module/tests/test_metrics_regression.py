import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio

from rasterio.crs import CRS
from rasterio.transform import from_origin, Affine

from src.gis_engine.metrics.degradation import (
    calculate_metrics,
    calculate_subset_stats,
    filter_by_min_area,
    _load_input_raster,
)
from src.gis_engine.validation.quality import (
    QualityAssuranceEngine,
)


@pytest.fixture
def grid():
    return SimpleNamespace(
        crs=CRS.from_epsg(32644),
        transform=from_origin(
            500000, 1800000, 10, 10
        ),
        width=4,
        height=4,
    )


@pytest.fixture
def config():
    return {
        "metrics": {
            "severity_weights": {
                "healthy": 0.0,
                "moderate": 0.35,
                "severe": 0.8,
                "background": 1.0,
            },
            "minimum_change_area": {
                "enabled": False,
                "value_m2": 0,
            },
            "include_canopy_loss_in_swd": False,
        },
        "wetland": {
            "depletion_claims": {
                "enabled": False,
            },
        },
    }


@pytest.fixture
def inputs():
    shape = (4, 4)

    ones = np.ones(shape, dtype=np.uint8)
    zeros = np.zeros(shape, dtype=np.uint8)
    delta = np.full(shape, -0.1)

    return {
        "transition_map": np.full(
            shape, 14, dtype=np.uint8
        ),
        "cva_magnitude": np.sqrt(3 * delta ** 2),
        "delta_ndvi": delta.copy(),
        "delta_gndvi": delta.copy(),
        "delta_ndre": delta.copy(),
        "degradation_map": ones.copy(),
        "recovery_map": zeros.copy(),
        "residual_stress_map": zeros.copy(),
        "canopy_loss_map": zeros.copy(),
        "wetland_mask": ones.copy(),
        "valid_analysis_mask": ones.copy(),
    }


def write_raster(
    path, array, grid, nodata=None, count=1
):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=count,
        dtype=array.dtype,
        crs=grid.crs,
        transform=grid.transform,
        nodata=nodata,
    ) as dst:

        for band in range(1, count + 1):
            dst.write(array, band)

    return path


# ==========================================
# MMU ACCURACY AND INPUT VALIDATION
# ==========================================

@pytest.mark.parametrize(
    "threshold,expected",
    [
        (0, 3),
        (100, 3),
        (200, 2),
        (201, 0),
    ],
)
def test_mmu_threshold_boundary(threshold, expected):

    raster = np.array(
        [[1, 1, 0, 1, 255]],
        dtype=np.uint8,
    )

    saved = raster.copy()

    result = filter_by_min_area(
        raster,
        np.full(raster.shape, 100.0),
        threshold,
    )

    assert np.count_nonzero(result == 1) == expected
    assert result[0, 4] == 255
    assert np.array_equal(raster, saved)


def test_mmu_four_neighbour_connectivity():

    raster = np.eye(5, dtype=np.uint8)

    result = filter_by_min_area(
        raster,
        np.full(raster.shape, 100.0),
        200,
    )

    assert not np.any(result)


@pytest.mark.parametrize(
    "threshold",
    [-1, np.nan, np.inf, True],
)
def test_mmu_invalid_threshold(threshold):

    with pytest.raises((ValueError, TypeError)):

        filter_by_min_area(
            np.ones((2, 2), dtype=np.uint8),
            np.full((2, 2), 100.0),
            threshold,
        )


@pytest.mark.parametrize(
    "area",
    [0, -1, np.nan, np.inf],
)
def test_mmu_invalid_pixel_area(area):

    with pytest.raises((ValueError, TypeError)):

        filter_by_min_area(
            np.ones((2, 2), dtype=np.uint8),
            np.full((2, 2), area),
            100,
        )


@pytest.mark.parametrize(
    "excluded_key,excluded_value",
    [
        ("valid_analysis_mask", 0),
        ("wetland_mask", 0),
        ("transition_map", 255),
        ("cva_magnitude", -9999),
        ("delta_ndvi", np.nan),
    ],
)
def test_mmu_excluded_pixel_bridge(
    inputs, config, grid,
    excluded_key, excluded_value,
):

    inputs["transition_map"][:] = 15
    inputs["transition_map"][1, :3] = 14

    inputs["degradation_map"][:] = 0
    inputs["degradation_map"][1, :3] = 1

    inputs[excluded_key][1, 1] = excluded_value

    config["metrics"]["minimum_change_area"] = {
        "enabled": True,
        "value_m2": 200,
    }

    saved = deepcopy(inputs)

    result = calculate_metrics(
        **inputs,
        config=config,
        reference_grid=grid,
    )

    assert result.baseline_area["pixel_count"] == 15
    assert result.impact_area["impact_area_m2"] == 0

    for key in inputs:
        assert np.array_equal(
            inputs[key],
            saved[key],
            equal_nan=True,
        )


# ==========================================
# RASTER LOADER VALIDATION
# ==========================================

@pytest.mark.parametrize("count", [2, 3, 5])
def test_loader_rejects_multiband(
    tmp_path, grid, count,
):

    path = write_raster(
        tmp_path / "multi.tif",
        np.ones((4, 4), dtype=np.uint8),
        grid,
        count=count,
    )

    with pytest.raises(ValueError):
        _load_input_raster(path)


def test_loader_preserves_metadata(tmp_path, grid):

    array = np.ones((4, 4), dtype=np.float32)
    array[0, 0] = -32768

    path = write_raster(
        tmp_path / "single.tif",
        array,
        grid,
        nodata=-32768,
    )

    loaded, metadata = _load_input_raster(path)

    assert np.array_equal(loaded, array)
    assert metadata["nodata"] == -32768
    assert metadata["crs"] == grid.crs
    assert metadata["transform"] == grid.transform
    assert metadata["width"] == 4
    assert metadata["height"] == 4


def test_loader_rejects_missing_file(tmp_path):

    with pytest.raises(FileNotFoundError):
        _load_input_raster(tmp_path / "missing.tif")


def test_loader_rejects_corrupted_file(tmp_path):

    path = tmp_path / "invalid.tif"
    path.write_text("Invalid raster")

    with pytest.raises(rasterio.errors.RasterioIOError):
        _load_input_raster(path)


# ==========================================
# CONTINUOUS NODATA
# ==========================================

@pytest.mark.parametrize(
    "key",
    [
        "cva_magnitude",
        "delta_ndvi",
        "delta_gndvi",
        "delta_ndre",
    ],
)
@pytest.mark.parametrize(
    "nodata",
    [-9999.0, -32768.0, 9999.0, np.nan],
)
def test_continuous_raster_nodata(
    inputs, config, grid, tmp_path, key, nodata,
):

    array = inputs[key].astype(np.float32)
    array[0, 0] = nodata

    inputs[key] = write_raster(
        tmp_path / "continuous.tif",
        array,
        grid,
        nodata=nodata,
    )

    result = calculate_metrics(
        **inputs,
        config=config,
        reference_grid=grid,
    )

    assert result.baseline_area["pixel_count"] == 15
    assert result.baseline_area["area_m2"] == 1500
    assert result.impact_area["impact_area_m2"] == 1500


# ==========================================
# CATEGORICAL NODATA
# ==========================================

@pytest.mark.parametrize(
    "key",
    [
        "degradation_map",
        "recovery_map",
        "residual_stress_map",
        "canopy_loss_map",
        "wetland_mask",
        "valid_analysis_mask",
    ],
)
@pytest.mark.parametrize(
    "nodata,dtype",
    [
        (255, np.uint8),
        (-1, np.int16),
        (-32768, np.int16),
        (np.nan, np.float32),
    ],
)
def test_categorical_raster_nodata(
    inputs, config, grid, tmp_path,
    key, nodata, dtype,
):

    array = inputs[key].astype(dtype)
    array[0, 0] = nodata

    inputs[key] = write_raster(
        tmp_path / "categorical.tif",
        array,
        grid,
        nodata=nodata,
    )

    result = calculate_metrics(
        **inputs,
        config=config,
        reference_grid=grid,
    )

    assert result.baseline_area["pixel_count"] == 15
    assert result.baseline_area["area_m2"] == 1500
    assert result.impact_area["impact_area_m2"] == 1500


@pytest.mark.parametrize(
    "nodata,dtype",
    [
        (255, np.uint8),
        (127, np.int8),
        (-1, np.int16),
        (-32768, np.int16),
        (65535, np.uint16),
    ],
)
def test_transition_raster_nodata(
    inputs, config, grid, tmp_path,
    nodata, dtype,
):

    array = inputs["transition_map"].astype(dtype)
    array[0, 0] = nodata

    inputs["transition_map"] = write_raster(
        tmp_path / "transition.tif",
        array,
        grid,
        nodata=nodata,
    )

    result = calculate_metrics(
        **inputs,
        config=config,
        reference_grid=grid,
    )

    assert result.baseline_area["pixel_count"] == 15
    assert result.impact_area["impact_area_m2"] == 1500


# ==========================================
# REFERENCE GRID RECONCILIATION
# ==========================================

@pytest.mark.parametrize(
    "attribute,value",
    [
        ("width", 0),
        ("width", -1),
        ("width", 5),
        ("width", True),
        ("width", 4.5),
        ("width", "4"),
        ("height", 0),
        ("height", -1),
        ("height", 5),
        ("height", True),
        ("height", 4.5),
        ("height", "4"),
        ("crs", CRS.from_epsg(32643)),
        ("crs", None),
        ("transform", None),
        (
            "transform",
            from_origin(500001, 1800000, 10, 10),
        ),
        (
            "transform",
            from_origin(500000, 1800001, 10, 10),
        ),
        (
            "transform",
            from_origin(500000, 1800000, 20, 20),
        ),
        (
            "transform",
            Affine(0, 0, 500000, 0, 0, 1800000),
        ),
        (
            "transform",
            Affine(np.nan, 0, 500000, 0, -10, 1800000),
        ),
    ],
)
def test_reference_grid_conflicts(
    inputs, config, grid, tmp_path,
    attribute, value,
):

    inputs["transition_map"] = write_raster(
        tmp_path / "transition.tif",
        inputs["transition_map"],
        grid,
        nodata=255,
    )

    invalid_grid = deepcopy(grid)
    setattr(invalid_grid, attribute, value)

    with pytest.raises((ValueError, TypeError)):

        calculate_metrics(
            **inputs,
            config=config,
            reference_grid=invalid_grid,
        )


def test_pixel_only_metrics(inputs, config, grid):

    grid.crs = None

    result = calculate_metrics(
        **inputs,
        config=config,
        reference_grid=grid,
    )

    assert result.baseline_area["pixel_count"] == 16
    assert result.baseline_area["area_m2"] is None


# ==========================================
# STATISTICAL VALIDATION
# ==========================================

def test_statistics_exclude_invalid_pixels():

    data = np.array(
        [[1.0, 2.0], [3.0, np.nan]]
    )

    mask = np.array(
        [[True, True], [True, False]]
    )

    result = calculate_subset_stats(data, mask)

    assert result["mean"] == 2.0
    assert result["median"] == 2.0
    assert result["min"] == 1.0
    assert result["max"] == 3.0


@pytest.mark.parametrize(
    "value",
    [np.nan, np.inf, -np.inf],
)
def test_statistics_reject_nonfinite_values(value):

    data = np.full((2, 2), value)
    mask = np.ones((2, 2), dtype=bool)

    with pytest.raises(ValueError):
        calculate_subset_stats(data, mask)


def test_statistics_empty_population():

    result = calculate_subset_stats(
        np.ones((2, 2)),
        np.zeros((2, 2), dtype=bool),
    )

    assert all(
        value is None
        for value in result.values()
    )


# ==========================================
# METRICS QA AND CORRUPTION DETECTION
# ==========================================

def test_metrics_qa_valid_producer(
    inputs, config, grid,
):

    result = calculate_metrics(
        **inputs,
        config=config,
        reference_grid=grid,
    ).to_dict()

    qa = QualityAssuranceEngine(config)

    report = qa.run_metrics_qa(result, config)

    assert report["status"] == "PASS"

    json.dumps(result, allow_nan=False)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("baseline_area", "pixel_count", -1),
        ("baseline_area", "area_m2", np.nan),
        ("baseline_area", "area_ha", 999),
        ("impact_area", "impact_area_m2", 99999),
        ("impact_area", "ndr", 2.0),
        ("impact_area", "ndr_percent", -1),
        ("recovery_area", "area_m2", -100),
        ("canopy_loss_area", "ratio", 2.0),
        ("metadata", "area_basis", "invalid"),
        ("wetland_extent_metrics", "status", "AVAILABLE"),
        ("wetland_extent_metrics", "extent_T1_ha", 1.0),
        (
            "wetland_extent_metrics",
            "physical_wetland_depletion",
            0.5,
        ),
    ],
)
def test_metrics_qa_detects_corruption(
    inputs, config, grid,
    section, key, value,
):

    result = calculate_metrics(
        **inputs,
        config=config,
        reference_grid=grid,
    ).to_dict()

    result[section][key] = value

    qa = QualityAssuranceEngine(config)

    report = qa.run_metrics_qa(result, config)

    assert report["status"] == "FAIL"
