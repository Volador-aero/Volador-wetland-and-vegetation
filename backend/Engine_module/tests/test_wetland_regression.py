
"""Regression tests for wetland boundary, water and fallback safeguards."""

import copy
import json

import numpy as np
import pytest
import rasterio
import shapefile

from rasterio.crs import CRS
from rasterio.transform import from_origin

from gis_engine.wetland.boundary import (
    validate_boundary_geometry,
    reproject_geometry,
    load_vector_boundary,
)

from gis_engine.wetland.water import (
    calculate_ndwi,
    classify_optical_water,
    analyze_hydrology,
)

from gis_engine.wetland.mask import build_wetland_mask

from Engine_module.tests.test_wetland import get_test_grid


WGS84 = CRS.from_epsg(4326)
UTM = CRS.from_epsg(32632)

TRANSFORM = from_origin(
    500000, 5000000, 10, 10
)


# ==========================================================
# SHARED FIXTURES
# ==========================================================

@pytest.fixture
def polygon():
    return {
        "type": "Polygon",
        "coordinates": [[
            [9.0, 45.0],
            [9.0, 45.1],
            [9.1, 45.1],
            [9.1, 45.0],
            [9.0, 45.0],
        ]],
    }


@pytest.fixture
def grid():
    return get_test_grid(shape=(10, 10))


@pytest.fixture
def overlap():
    mask = np.ones((10, 10), dtype=np.uint8)
    mask[0, :] = 0
    mask[1, :] = 255
    return mask


def write_raster(
    path,
    data,
    crs="EPSG:32632",
    affine=TRANSFORM,
    nodata=-9999,
    count=1,
    internal_mask=None,
):
    data = np.asarray(data, dtype=np.float32)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=count,
        dtype="float32",
        crs=crs,
        transform=affine,
        nodata=nodata,
    ) as dst:

        for band in range(1, count + 1):
            dst.write(data, band)

        if internal_mask is not None:
            dst.write_mask(internal_mask)

    return path


@pytest.fixture
def water_inputs(tmp_path):

    t1 = tmp_path / "t1"
    t2 = tmp_path / "t2"

    t1.mkdir()
    t2.mkdir()

    write_raster(
        t1 / "green.tif",
        [[0.6, 0.2], [0.6, 0.2]],
    )

    write_raster(
        t1 / "nir.tif",
        [[0.2, 0.6], [0.2, 0.6]],
    )

    write_raster(
        t2 / "green.tif",
        [[0.2, 0.6], [0.6, 0.2]],
    )

    write_raster(
        t2 / "nir.tif",
        [[0.6, 0.2], [0.2, 0.6]],
    )

    return t1, t2


# ==========================================================
# 1. BOUNDARY GEOMETRY VALIDATION
# ==========================================================

def test_valid_polygon(polygon):
    assert validate_boundary_geometry(polygon) == polygon


def test_valid_multipolygon(polygon):

    geometry = {
        "type": "MultiPolygon",
        "coordinates": [polygon["coordinates"]],
    }

    assert validate_boundary_geometry(geometry) == geometry


@pytest.mark.parametrize(
    "geometry",
    [
        None,
        {},
        {"type": "Point", "coordinates": [9, 45]},
        {"type": "Polygon", "coordinates": []},
        {"type": "Polygon", "coordinates": [[]]},
        {"type": "MultiPolygon", "coordinates": []},
    ],
)
def test_invalid_geometry_rejected(geometry):

    with pytest.raises(ValueError):
        validate_boundary_geometry(geometry)


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        "invalid",
        True,
        None,
    ],
)
def test_invalid_coordinate_rejected(polygon, value):

    geometry = copy.deepcopy(polygon)
    geometry["coordinates"][0][1][0] = value

    with pytest.raises(ValueError):
        validate_boundary_geometry(geometry)


@pytest.mark.parametrize(
    "ring",
    [
        [[0, 0], [1, 1], [0, 0]],
        [[0, 0], [1, 1], [1, 1], [0, 0]],
        [[0, 0], [1, 1], [2, 2], [0, 0]],
        [[0, 0], [0, 1], [1, 1], [2, 2]],
    ],
)
def test_invalid_polygon_ring_rejected(ring):

    with pytest.raises(ValueError):
        validate_boundary_geometry({
            "type": "Polygon",
            "coordinates": [ring],
        })


def test_same_crs_preserves_geometry(polygon):

    assert reproject_geometry(
        polygon, WGS84, WGS84
    ) == polygon


def test_polygon_reprojection(polygon):

    result = reproject_geometry(
        polygon, WGS84, UTM
    )

    assert result["type"] == "Polygon"

    assert (
        result["coordinates"]
        != polygon["coordinates"]
    )


def test_multipolygon_reprojection(polygon):

    geometry = {
        "type": "MultiPolygon",
        "coordinates": [polygon["coordinates"]],
    }

    result = reproject_geometry(
        geometry, WGS84, UTM
    )

    assert result["type"] == "MultiPolygon"


@pytest.mark.parametrize(
    "source,destination",
    [
        (None, UTM),
        (WGS84, None),
        ("INVALID_CRS", UTM),
    ],
)
def test_invalid_reprojection_crs(
    polygon, source, destination
):

    with pytest.raises(ValueError):
        reproject_geometry(
            polygon, source, destination
        )


def test_same_crs_still_validates_geometry():

    with pytest.raises(ValueError):
        reproject_geometry(
            {"type": "Polygon", "coordinates": []},
            WGS84,
            WGS84,
        )


# ==========================================================
# 2. GEOJSON AND SHAPEFILE VALIDATION
# ==========================================================

def test_geojson_default_crs(tmp_path, polygon):

    path = tmp_path / "boundary.geojson"

    path.write_text(
        json.dumps(polygon),
        encoding="utf-8",
    )

    features, crs = load_vector_boundary(path)

    assert len(features) == 1
    assert crs == WGS84


def test_geojson_explicit_crs(tmp_path, polygon):

    polygon["crs"] = {
        "type": "name",
        "properties": {"name": "EPSG:4326"},
    }

    path = tmp_path / "boundary.geojson"

    path.write_text(
        json.dumps(polygon),
        encoding="utf-8",
    )

    features, crs = load_vector_boundary(path)

    assert len(features) == 1
    assert crs == WGS84


@pytest.mark.parametrize(
    "crs_value",
    [
        None,
        {},
        {"properties": {"name": ""}},
        {"properties": {"name": "INVALID_CRS"}},
        {"properties": {"name": 4326}},
    ],
)
def test_invalid_geojson_crs(
    tmp_path, polygon, crs_value
):

    polygon["crs"] = crs_value

    path = tmp_path / "boundary.geojson"

    path.write_text(
        json.dumps(polygon),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_vector_boundary(path)


@pytest.mark.parametrize(
    "data",
    [
        [],
        {"type": "FeatureCollection", "features": "bad"},
        {"type": "FeatureCollection", "features": None},
    ],
)
def test_invalid_geojson_structure(tmp_path, data):

    path = tmp_path / "boundary.geojson"

    path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_vector_boundary(path)


@pytest.mark.parametrize(
    "selected,expected",
    [
        (["marsh"], 1),
        (["forest"], 1),
        (["marsh", "forest"], 2),
        (["desert"], 0),
    ],
)
def test_geojson_class_filtering(
    tmp_path, polygon, selected, expected
):

    data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": polygon,
                "properties": {"class": name},
            }
            for name in ["marsh", "forest"]
        ],
    }

    path = tmp_path / "boundary.geojson"

    path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    features, _ = load_vector_boundary(
        path,
        class_field="class",
        select_classes=selected,
    )

    assert len(features) == expected


@pytest.fixture
def shapefile_path(tmp_path, polygon):

    path = tmp_path / "boundary.shp"

    with shapefile.Writer(
        str(path),
        shapeType=shapefile.POLYGON,
    ) as writer:

        writer.field("class", "C")
        writer.poly(polygon["coordinates"])
        writer.record("marsh")

    return path


@pytest.mark.parametrize(
    "prj_content",
    [None, "", "INVALID_CRS"],
)
def test_invalid_shapefile_crs(
    shapefile_path, prj_content
):

    if prj_content is not None:

        shapefile_path.with_suffix(".prj").write_text(
            prj_content,
            encoding="utf-8",
        )

    with pytest.raises(ValueError):
        load_vector_boundary(shapefile_path)


def test_valid_shapefile(shapefile_path):

    shapefile_path.with_suffix(".prj").write_text(
        WGS84.to_wkt(),
        encoding="utf-8",
    )

    features, crs = load_vector_boundary(
        shapefile_path
    )

    assert len(features) == 1
    assert crs == WGS84


@pytest.mark.parametrize(
    "selected,expected",
    [
        (["marsh"], 1),
        (["forest"], 0),
    ],
)
def test_shapefile_class_filtering(
    shapefile_path, selected, expected
):

    shapefile_path.with_suffix(".prj").write_text(
        WGS84.to_wkt(),
        encoding="utf-8",
    )

    features, _ = load_vector_boundary(
        shapefile_path,
        class_field="class",
        select_classes=selected,
    )

    assert len(features) == expected


def test_missing_boundary_file(tmp_path):

    with pytest.raises(FileNotFoundError):
        load_vector_boundary(
            tmp_path / "missing.geojson"
        )


def test_unsupported_boundary_format(tmp_path):

    path = tmp_path / "boundary.txt"
    path.write_text("test", encoding="utf-8")

    with pytest.raises(ValueError):
        load_vector_boundary(path)


# ==========================================================
# 3. NDWI MATHEMATICAL VALIDATION
# ==========================================================

def test_ndwi_formula():

    ndwi, valid = calculate_ndwi(
        np.array([0.6, 0.2]),
        np.array([0.2, 0.6]),
        epsilon=0,
    )

    assert np.all(valid)
    assert np.allclose(ndwi, [0.5, -0.5])
    assert ndwi.dtype == np.float32
    assert valid.dtype == np.bool_


@pytest.mark.parametrize(
    "epsilon",
    [-1, np.nan, np.inf, True, "invalid"],
)
def test_invalid_ndwi_epsilon(epsilon):

    with pytest.raises(ValueError):
        calculate_ndwi(
            np.array([0.6]),
            np.array([0.2]),
            epsilon=epsilon,
        )


@pytest.mark.parametrize(
    "green,nir",
    [
        (np.array([0.6, 0.2]), np.array([0.2])),
        (np.array([]), np.array([])),
        (np.array([1j]), np.array([1.0])),
        (np.ones((1, 1, 1)), np.ones((1, 1, 1))),
    ],
)
def test_invalid_ndwi_arrays(green, nir):

    with pytest.raises(ValueError):
        calculate_ndwi(green, nir)


@pytest.mark.parametrize(
    "mask",
    [
        np.array([True]),
        np.array([0, 255]),
        np.array([[True, False]]),
    ],
)
def test_invalid_ndwi_mask(mask):

    with pytest.raises(ValueError):
        calculate_ndwi(
            np.array([0.6, 0.2]),
            np.array([0.2, 0.6]),
            nodata_mask=mask,
        )


def test_ndwi_zero_signal():

    ndwi, valid = calculate_ndwi(
        np.array([0.0]),
        np.array([0.0]),
    )

    assert not valid[0]
    assert ndwi[0] == -9999


def test_ndwi_nonfinite_inputs():

    ndwi, valid = calculate_ndwi(
        np.array([np.nan, np.inf]),
        np.array([0.2, 0.3]),
    )

    assert not np.any(valid)
    assert np.all(ndwi == -9999)


def test_ndwi_explicit_nodata_mask():

    ndwi, valid = calculate_ndwi(
        np.array([0.6, 0.2]),
        np.array([0.2, 0.6]),
        nodata_mask=np.array([True, False]),
    )

    assert not valid[0]
    assert valid[1]
    assert ndwi[0] == -9999


# ==========================================================
# 4. OPTICAL WATER RASTER SAFEGUARDS
# ==========================================================

def test_valid_optical_classification(water_inputs):

    t1, _ = water_inputs

    water, valid = classify_optical_water(
        t1 / "green.tif",
        t1 / "nir.tif",
    )

    assert np.count_nonzero(water) == 2
    assert np.count_nonzero(valid) == 4


@pytest.mark.parametrize(
    "threshold",
    [-1.1, 1.1, np.nan, np.inf, True, "invalid"],
)
def test_invalid_optical_threshold(
    water_inputs, threshold
):

    t1, _ = water_inputs

    with pytest.raises(ValueError):
        classify_optical_water(
            t1 / "green.tif",
            t1 / "nir.tif",
            threshold=threshold,
        )


@pytest.mark.parametrize(
    "settings",
    [
        {"crs": "EPSG:4326"},
        {"affine": from_origin(500010, 5000000, 10, 10)},
        {"count": 2},
        {"crs": None},
    ],
)
def test_optical_alignment_rejection(
    water_inputs, settings
):

    t1, _ = water_inputs

    write_raster(
        t1 / "nir.tif",
        [[0.2, 0.6], [0.2, 0.6]],
        **settings,
    )

    with pytest.raises(ValueError):
        classify_optical_water(
            t1 / "green.tif",
            t1 / "nir.tif",
        )


def test_optical_dimension_mismatch(water_inputs):

    t1, _ = water_inputs

    write_raster(
        t1 / "nir.tif",
        [[0.2, 0.6]],
    )

    with pytest.raises(ValueError):
        classify_optical_water(
            t1 / "green.tif",
            t1 / "nir.tif",
        )


def test_independent_band_nodata(water_inputs):

    t1, _ = water_inputs

    write_raster(
        t1 / "nir.tif",
        [[-8888, 0.6], [0.2, 0.6]],
        nodata=-8888,
    )

    water, valid = classify_optical_water(
        t1 / "green.tif",
        t1 / "nir.tif",
    )

    assert not valid[0, 0]
    assert not water[0, 0]


def test_internal_raster_mask(water_inputs):

    t1, _ = water_inputs

    write_raster(
        t1 / "nir.tif",
        [[0.2, 0.6], [0.2, 0.6]],
        internal_mask=np.array(
            [[0, 255], [255, 255]],
            dtype=np.uint8,
        ),
    )

    water, valid = classify_optical_water(
        t1 / "green.tif",
        t1 / "nir.tif",
    )

    assert not valid[0, 0]
    assert not water[0, 0]


def test_optical_missing_crs(water_inputs):

    t1, _ = water_inputs

    write_raster(
        t1 / "green.tif",
        [[0.6, 0.2], [0.6, 0.2]],
        crs=None,
    )

    write_raster(
        t1 / "nir.tif",
        [[0.2, 0.6], [0.2, 0.6]],
        crs=None,
    )

    with pytest.raises(ValueError):
        classify_optical_water(
            t1 / "green.tif",
            t1 / "nir.tif",
        )


# ==========================================================
# 5. TEMPORAL HYDROLOGY SAFEGUARDS
# ==========================================================

def test_hydrology_transition_reconciliation(water_inputs):

    t1, t2 = water_inputs

    result = analyze_hydrology(
        t1,
        t2,
        None,
        {"water": {"ndwi_threshold": 0.0}},
    )

    assert result["water_gain_pixel_count"] == 1
    assert result["water_loss_pixel_count"] == 1

    assert result["water_gain_m2"] == pytest.approx(100)
    assert result["water_loss_m2"] == pytest.approx(100)

    assert result["net_water_change_m2"] == pytest.approx(0)


def test_hydrology_overlap_nodata(
    water_inputs, tmp_path
):

    t1, t2 = water_inputs

    mask = write_raster(
        tmp_path / "overlap.tif",
        [[1, 255], [0, 1]],
        nodata=255,
    )

    result = analyze_hydrology(
        t1,
        t2,
        None,
        {},
        overlap_mask_path=mask,
    )

    assert result["valid_pixel_count"] == 2


def test_hydrology_empty_population(
    water_inputs, tmp_path
):

    t1, t2 = water_inputs

    mask = write_raster(
        tmp_path / "overlap.tif",
        [[0, 0], [0, 0]],
        nodata=255,
    )

    result = analyze_hydrology(
        t1,
        t2,
        None,
        {},
        overlap_mask_path=mask,
    )

    assert result["status"] == "NOT_AVAILABLE"
    assert result["valid_pixel_count"] == 0
    assert result["water_area_t1_m2"] is None


@pytest.mark.parametrize(
    "mask_argument",
    ["overlap_mask_path", "wetland_mask_path"],
)
def test_hydrology_missing_mask(
    water_inputs, tmp_path, mask_argument
):

    t1, t2 = water_inputs

    with pytest.raises(FileNotFoundError):
        analyze_hydrology(
            t1,
            t2,
            None,
            {},
            **{
                mask_argument: tmp_path / "missing.tif"
            },
        )


@pytest.mark.parametrize(
    "mask_argument",
    ["overlap_mask_path", "wetland_mask_path"],
)
def test_hydrology_invalid_mask(
    water_inputs, tmp_path, mask_argument
):

    t1, t2 = water_inputs

    mask = write_raster(
        tmp_path / "invalid.tif",
        [[1, 2], [0, 1]],
        nodata=255,
    )

    with pytest.raises(ValueError):
        analyze_hydrology(
            t1,
            t2,
            None,
            {},
            **{mask_argument: mask},
        )


def test_hydrology_temporal_alignment(water_inputs):

    t1, t2 = water_inputs

    write_raster(
        t2 / "nir.tif",
        [[0.6, 0.2], [0.2, 0.6]],
        affine=from_origin(
            500010, 5000000, 10, 10
        ),
    )

    with pytest.raises(ValueError):
        analyze_hydrology(t1, t2, None, {})


def test_hydrology_output_files(
    water_inputs, tmp_path
):

    t1, t2 = water_inputs

    output = tmp_path / "outputs"

    analyze_hydrology(t1, t2, output, {})

    expected = {
        "ndwi_t1.tif",
        "ndwi_t2.tif",
        "delta_ndwi.tif",
        "water_mask_t1.tif",
        "water_mask_t2.tif",
        "water_change.tif",
        "water_metrics.json",
    }

    assert expected.issubset(
        {path.name for path in output.iterdir()}
    )


# ==========================================================
# 6. UNAVAILABLE WETLAND FALLBACK CONTRACT
# ==========================================================

@pytest.fixture(
    params=["auto", "boundary", "official", "jrc", "sar"]
)
def unavailable_result(
    request, tmp_path, grid, overlap
):

    source = request.param

    settings = {
        "auto": {},
        "boundary": {
            "boundary": {
                "path": str(tmp_path / "missing.geojson")
            }
        },
        "official": {
            "official_dataset": {
                "path": str(tmp_path / "missing.shp")
            }
        },
        "jrc": {
            "jrc": {
                "path": str(tmp_path / "missing.tif")
            }
        },
        "sar": {},
    }

    return build_wetland_mask(
        source=source,
        reference_grid=grid,
        config={"wetland": settings[source]},
        overlap_mask=overlap,
    )


def test_fallback_source_unavailable(unavailable_result):

    assert unavailable_result.source_type == "UNAVAILABLE"


def test_fallback_status(unavailable_result):

    assert (
        unavailable_result.status
        == "WETLAND_BOUNDARY_UNAVAILABLE"
    )


def test_fallback_membership_empty(unavailable_result):

    assert not np.any(
        unavailable_result.wetland_mask
    )


def test_fallback_analysis_empty(unavailable_result):

    assert not np.any(
        unavailable_result.valid_analysis_mask
    )


def test_fallback_wetland_area_unknown(unavailable_result):

    assert unavailable_result.wetland_area_m2 is None
    assert unavailable_result.wetland_area_ha is None


def test_fallback_analysis_area_unknown(unavailable_result):

    assert unavailable_result.valid_analysis_area_m2 is None
    assert unavailable_result.valid_analysis_area_ha is None


def test_fallback_pixel_counts_unknown(unavailable_result):

    metadata = unavailable_result.metadata

    assert metadata["wetland_pixel_count"] is None

    assert (
        metadata["valid_wetland_analysis_pixel_count"]
        is None
    )


def test_fallback_depletion_disabled(unavailable_result):

    assert (
        unavailable_result.physical_wetland_depletion_enabled
        is False
    )


def test_fallback_context_unavailable(unavailable_result):

    metadata = unavailable_result.metadata

    assert metadata["boundary_available"] is False
    assert metadata["wetland_context_available"] is False


def test_fallback_exported_masks(
    tmp_path, grid, overlap
):

    output = tmp_path / "outputs"

    build_wetland_mask(
        source="auto",
        reference_grid=grid,
        config={"wetland": {}},
        overlap_mask=overlap,
        output_dir=output,
    )

    for filename in [
        "wetland_mask.tif",
        "valid_wetland_analysis_mask.tif",
    ]:

        with rasterio.open(output / filename) as src:

            assert not np.any(src.read(1))


def test_fallback_exported_metadata(
    tmp_path, grid, overlap
):

    output = tmp_path / "outputs"

    build_wetland_mask(
        source="auto",
        reference_grid=grid,
        config={"wetland": {}},
        overlap_mask=overlap,
        output_dir=output,
    )

    metadata = json.loads(
        (output / "wetland_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    assert metadata["wetland_area_m2"] is None
    assert metadata["wetland_area_ha"] is None


def test_valid_boundary_zero_coverage(tmp_path, grid):

    path = tmp_path / "outside.geojson"

    path.write_text(
        json.dumps({
            "type": "Polygon",
            "coordinates": [[
                [600000, 6000000],
                [600000, 6000100],
                [600100, 6000100],
                [600100, 6000000],
                [600000, 6000000],
            ]],
            "crs": {
                "type": "name",
                "properties": {
                    "name": str(grid.crs)
                },
            },
        }),
        encoding="utf-8",
    )

    result = build_wetland_mask(
        source=str(path),
        reference_grid=grid,
        config={"wetland": {}},
    )

    assert result.status == "WETLAND_BOUNDARY_AVAILABLE"

    assert result.wetland_area_m2 == 0.0
    assert result.metadata["wetland_pixel_count"] == 0
    assert result.metadata["boundary_available"] is True




# ==========================================================
# 7. SOURCE OBSERVATION COVERAGE REGRESSION
# ==========================================================

@pytest.fixture
def source_rasters(tmp_path, grid):

    def create(name, data, nodata=-9999,
               affine=None, internal_mask=None):

        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)

        return write_raster(
            path,
            data,
            crs=grid.crs,
            affine=(
                grid.transform
                if affine is None
                else affine
            ),
            nodata=nodata,
            internal_mask=internal_mask,
        )

    return create


def build_jrc_result(
    grid,
    path,
    overlap=None,
    output=None,
    aoi=None,
):

    return build_wetland_mask(
        source="jrc",
        reference_grid=grid,
        config={
            "wetland": {
                "jrc": {
                    "enabled": True,
                    "path": str(path),
                }
            }
        },
        overlap_mask=overlap,
        output_dir=output,
        aoi=aoi,
    )


def build_optical_result(
    grid,
    t1,
    t2,
    overlap=None,
    output=None,
    aoi=None,
):

    return build_wetland_mask(
        source="optical",
        reference_grid=grid,
        config={
            "wetland": {
                "optical": {
                    "enabled": True,
                    "threshold": 0.1,
                }
            }
        },
        t1_aligned_dir=t1,
        t2_aligned_dir=t2,
        overlap_mask=overlap,
        output_dir=output,
        aoi=aoi,
    )


# ----------------------------------------------------------
# JRC OBSERVATION VALIDITY
# ----------------------------------------------------------

def test_jrc_zero_is_valid_observation(
    grid, source_rasters
):

    path = source_rasters(
        "jrc.tif",
        np.zeros((10, 10)),
    )

    result = build_jrc_result(grid, path)

    coverage = result.metadata["observation_coverage"]

    assert result.source_type == "JRC_WATER_CONTEXT"
    assert coverage["status"] == "COMPLETE"
    assert coverage["observed_pixel_count"] == 100
    assert coverage["unknown_pixel_count"] == 0
    assert result.wetland_area_m2 == 0


def test_jrc_occurrence_threshold(
    grid, source_rasters
):

    data = np.zeros((10, 10))

    data[0, :4] = [0, 10, 11, 100]

    path = source_rasters("jrc.tif", data)

    result = build_jrc_result(grid, path)

    assert result.wetland_mask[0, :4].tolist() == [
        0, 0, 1, 1
    ]

    assert result.metadata["wetland_pixel_count"] == 2


@pytest.mark.parametrize(
    "invalid_value",
    [-9999, -1, 101, 255, np.nan, np.inf],
)
def test_jrc_invalid_observation_excluded(
    grid, source_rasters, invalid_value
):

    data = np.full((10, 10), 50.0)
    data[0, 0] = invalid_value

    path = source_rasters("jrc.tif", data)

    result = build_jrc_result(grid, path)

    coverage = result.metadata["observation_coverage"]

    assert result.wetland_mask[0, 0] == 0
    assert result.valid_analysis_mask[0, 0] == 0

    assert coverage["status"] == "PARTIAL"
    assert coverage["observed_pixel_count"] == 99
    assert coverage["unknown_pixel_count"] == 1
    assert coverage["coverage_fraction"] == pytest.approx(
        0.99
    )


def test_jrc_internal_mask_excluded(
    grid, source_rasters
):

    internal_mask = np.full(
        (10, 10), 255, dtype=np.uint8
    )

    internal_mask[0, :] = 0

    path = source_rasters(
        "jrc.tif",
        np.full((10, 10), 50),
        internal_mask=internal_mask,
    )

    result = build_jrc_result(grid, path)

    coverage = result.metadata["observation_coverage"]

    assert not np.any(result.wetland_mask[0, :])
    assert coverage["observed_pixel_count"] == 90
    assert coverage["unknown_pixel_count"] == 10


def test_jrc_all_nodata_unavailable(
    grid, source_rasters
):

    path = source_rasters(
        "jrc.tif",
        np.full((10, 10), -9999),
    )

    result = build_jrc_result(grid, path)

    assert result.source_type == "UNAVAILABLE"
    assert result.wetland_area_m2 is None

    assert (
        result.metadata["observation_coverage"]["status"]
        == "UNAVAILABLE"
    )


def test_jrc_outside_reference_grid(
    grid, source_rasters
):

    from affine import Affine

    shifted = grid.transform @ Affine.translation(
        1000, 1000
    )

    path = source_rasters(
        "jrc.tif",
        np.full((10, 10), 50),
        affine=shifted,
    )

    result = build_jrc_result(grid, path)

    assert result.source_type == "UNAVAILABLE"
    assert result.wetland_area_m2 is None
    assert not np.any(result.valid_analysis_mask)


def test_jrc_partial_geographic_coverage(
    grid, source_rasters
):

    path = source_rasters(
        "jrc.tif",
        np.full((5, 10), 50),
    )

    result = build_jrc_result(grid, path)

    coverage = result.metadata["observation_coverage"]

    assert coverage["status"] == "PARTIAL"
    assert coverage["observed_pixel_count"] == 50
    assert coverage["unknown_pixel_count"] == 50

    assert np.count_nonzero(
        result.valid_analysis_mask
    ) == 50


def test_jrc_no_observation_inside_aoi(
    grid, source_rasters
):

    data = np.full((10, 10), -9999.0)
    data[:5, :] = 50

    path = source_rasters("jrc.tif", data)

    aoi = np.zeros((10, 10), dtype=np.uint8)
    aoi[5:, :] = 1

    result = build_jrc_result(
        grid, path, aoi=aoi
    )

    assert result.source_type == "UNAVAILABLE"
    assert result.valid_analysis_area_m2 is None


def test_jrc_validity_raster_export(
    grid, source_rasters, tmp_path
):

    data = np.full((10, 10), 50.0)
    data[0, 0] = -9999

    path = source_rasters("jrc.tif", data)
    output = tmp_path / "jrc_outputs"

    result = build_jrc_result(
        grid, path, output=output
    )

    with rasterio.open(
        output / "wetland_source_validity.tif"
    ) as src:

        validity = src.read(1)

        assert src.transform == grid.transform
        assert src.crs == grid.crs
        assert validity[0, 0] == 0
        assert np.count_nonzero(validity) == 99

    assert result.metadata[
        "area_is_observed_membership_only"
    ] is True


# ----------------------------------------------------------
# OPTICAL TEMPORAL OBSERVATION VALIDITY
# ----------------------------------------------------------

@pytest.fixture
def optical_source_inputs(
    tmp_path, source_rasters
):

    for date in ("t1", "t2"):

        source_rasters(
            f"{date}/green.tif",
            np.full((10, 10), 0.6),
        )

        source_rasters(
            f"{date}/nir.tif",
            np.full((10, 10), 0.2),
        )

    return tmp_path / "t1", tmp_path / "t2"


def test_optical_complete_observations(
    grid, optical_source_inputs
):

    t1, t2 = optical_source_inputs

    result = build_optical_result(
        grid, t1, t2
    )

    coverage = result.metadata["observation_coverage"]

    assert result.source_type == "OPTICAL_WATER_INFERENCE"
    assert coverage["status"] == "COMPLETE"
    assert coverage["observed_pixel_count"] == 100
    assert np.count_nonzero(result.wetland_mask) == 100


@pytest.mark.parametrize(
    "date,band",
    [
        ("t1", "green"),
        ("t1", "nir"),
        ("t2", "green"),
        ("t2", "nir"),
    ],
)
def test_optical_common_validity(
    grid,
    optical_source_inputs,
    source_rasters,
    tmp_path,
    date,
    band,
):

    t1, t2 = optical_source_inputs

    data = np.full(
        (10, 10),
        0.6 if band == "green" else 0.2,
    )

    data[0, 0] = -9999

    source_rasters(
        f"{date}/{band}.tif",
        data,
    )

    output = tmp_path / "optical_outputs"

    result = build_optical_result(
        grid, t1, t2, output=output
    )

    coverage = result.metadata["observation_coverage"]

    assert result.wetland_mask[0, 0] == 0
    assert result.valid_analysis_mask[0, 0] == 0

    assert coverage["status"] == "PARTIAL"
    assert coverage["observed_pixel_count"] == 99

    with rasterio.open(
        output / "wetland_extent_change.tif"
    ) as src:

        change = src.read(1)

        assert change[0, 0] == 255
        assert src.nodata == 255


def test_optical_no_common_observations(
    grid,
    optical_source_inputs,
    source_rasters,
):

    t1, t2 = optical_source_inputs

    first = np.full((10, 10), 0.6)
    second = np.full((10, 10), 0.6)

    first[:5, :] = -9999
    second[5:, :] = -9999

    source_rasters("t1/green.tif", first)
    source_rasters("t2/green.tif", second)

    result = build_optical_result(
        grid, t1, t2
    )

    assert result.source_type == "UNAVAILABLE"
    assert result.wetland_area_m2 is None
    assert not np.any(result.valid_analysis_mask)


def test_optical_valid_dry_observations(
    grid,
    optical_source_inputs,
    source_rasters,
):

    t1, t2 = optical_source_inputs

    for date in ("t1", "t2"):

        source_rasters(
            f"{date}/green.tif",
            np.full((10, 10), 0.2),
        )

        source_rasters(
            f"{date}/nir.tif",
            np.full((10, 10), 0.6),
        )

    result = build_optical_result(
        grid, t1, t2
    )

    assert result.source_type == "OPTICAL_WATER_INFERENCE"
    assert result.wetland_area_m2 == 0

    assert (
        result.metadata["observation_coverage"]["status"]
        == "COMPLETE"
    )


def test_optical_reference_grid_mismatch(
    grid,
    optical_source_inputs,
    source_rasters,
):

    from affine import Affine

    t1, t2 = optical_source_inputs

    shifted = grid.transform @ Affine.translation(
        1, 0
    )

    for date in ("t1", "t2"):

        for band, value in (
            ("green", 0.6),
            ("nir", 0.2),
        ):

            source_rasters(
                f"{date}/{band}.tif",
                np.full((10, 10), value),
                affine=shifted,
            )

    result = build_optical_result(
        grid, t1, t2
    )

    assert result.source_type == "UNAVAILABLE"
    assert result.wetland_area_m2 is None


def test_optical_missing_required_band(
    grid, optical_source_inputs
):

    t1, t2 = optical_source_inputs

    (t2 / "nir.tif").unlink()

    result = build_optical_result(
        grid, t1, t2
    )

    assert result.source_type == "UNAVAILABLE"
    assert result.wetland_area_m2 is None


def test_optical_change_classes(
    grid,
    optical_source_inputs,
    source_rasters,
    tmp_path,
):

    t1, t2 = optical_source_inputs

    t1_green = np.full((10, 10), 0.2)
    t2_green = np.full((10, 10), 0.2)

    t1_nir = np.full((10, 10), 0.6)
    t2_nir = np.full((10, 10), 0.6)

    t1_green[0, 1:3] = 0.6
    t1_nir[0, 1:3] = 0.2

    t2_green[0, 1] = 0.6
    t2_nir[0, 1] = 0.2

    t2_green[0, 3] = 0.6
    t2_nir[0, 3] = 0.2

    t2_green[0, 4] = -9999

    source_rasters("t1/green.tif", t1_green)
    source_rasters("t1/nir.tif", t1_nir)
    source_rasters("t2/green.tif", t2_green)
    source_rasters("t2/nir.tif", t2_nir)

    output = tmp_path / "change_outputs"

    build_optical_result(
        grid, t1, t2, output=output
    )

    with rasterio.open(
        output / "wetland_extent_change.tif"
    ) as src:

        change = src.read(1)

        assert change[0, :5].tolist() == [
            0, 1, 3, 2, 255
        ]


def test_optical_overlap_and_aoi_exclusion(
    grid, optical_source_inputs, tmp_path
):

    t1, t2 = optical_source_inputs

    overlap = np.ones(
        (10, 10), dtype=np.uint8
    )

    overlap[0, :] = 255

    aoi = np.ones(
        (10, 10), dtype=np.uint8
    )

    aoi[1, :] = 0

    output = tmp_path / "restricted_outputs"

    result = build_optical_result(
        grid,
        t1,
        t2,
        overlap=overlap,
        aoi=aoi,
        output=output,
    )

    coverage = result.metadata["observation_coverage"]

    assert coverage["requested_pixel_count"] == 80
    assert coverage["observed_pixel_count"] == 80

    assert np.count_nonzero(
        result.valid_analysis_mask
    ) == 80

    with rasterio.open(
        output / "wetland_extent_change.tif"
    ) as src:

        change = src.read(1)

        assert np.all(change[:2, :] == 255)
        assert np.all(change[2:, :] == 1)


def test_optical_no_valid_observation_inside_aoi(
    grid,
    optical_source_inputs,
    source_rasters,
):

    t1, t2 = optical_source_inputs

    data = np.full((10, 10), 0.6)
    data[0, :] = -9999

    source_rasters("t2/green.tif", data)

    aoi = np.zeros(
        (10, 10), dtype=np.uint8
    )

    aoi[0, :] = 1

    result = build_optical_result(
        grid, t1, t2, aoi=aoi
    )

    assert result.source_type == "UNAVAILABLE"
    assert result.valid_analysis_area_m2 is None
