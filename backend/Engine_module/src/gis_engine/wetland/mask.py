import json
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from rasterio.transform import Affine
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union, Tuple

from ..alignment.spatial import ReferenceGrid
from ..metrics.area import calculate_pixel_areas, PhysicalAreaUnavailableError
from .boundary import load_vector_boundary, reproject_geometry
from .water import classify_optical_water

@dataclass
class WetlandMaskResult:
    wetland_mask: np.ndarray
    valid_analysis_mask: np.ndarray
    source_type: str
    source_metadata: Dict[str, Any]
    wetland_area_m2: Optional[float]
    wetland_area_ha: Optional[float]
    valid_analysis_area_m2: Optional[float]
    valid_analysis_area_ha: Optional[float]
    status: str
    physical_wetland_depletion_enabled: bool
    metadata: Dict[str, Any]

def build_wetland_mask(
    source: Optional[Union[str, Path]] = None,
    reference_grid: Optional[ReferenceGrid] = None,
    aoi: Optional[Union[np.ndarray, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    t1_aligned_dir: Optional[Union[str, Path]] = None,
    t2_aligned_dir: Optional[Union[str, Path]] = None,
    overlap_mask: Optional[Union[np.ndarray, Path]] = None
) -> WetlandMaskResult:
    """
    Builds the wetland mask and valid analysis mask for Phase 7.
    
    Priority order:
    1. Project boundary
    2. Official regional dataset
    3. JRC Global Surface Water context
    4. Local optical water classification
    5. SAR-derived water info (if applicable)
    """
    if reference_grid is None:
        raise ValueError("Reference grid is required for generating the wetland mask.")
    
    if config is None:
        config = {}
    
    wetland_config = config.get("wetland", {})
    
    # 1. Resolve source type to use
    resolved_source = source if source is not None else wetland_config.get("source", "auto")
    
    selected_source_type = "UNAVAILABLE"
    selected_status = "WETLAND_BOUNDARY_UNAVAILABLE"
    depletion_enabled = False
    source_metadata = {}
    
    boundary_path = wetland_config.get("boundary", {}).get("path")
    official_path = wetland_config.get("official_dataset", {}).get("path")
    jrc_enabled = wetland_config.get("jrc", {}).get("enabled", False)
    jrc_path = wetland_config.get("jrc", {}).get("path")
    optical_enabled = wetland_config.get("optical", {}).get("enabled", False)
    sar_enabled = wetland_config.get("sar", {}).get("enabled", False)
    
    # Flag to determine if we should execute a specific type
    source_decision = "auto"
    
    # If source is a direct file path, use it as PROJECT_BOUNDARY
    if isinstance(resolved_source, (str, Path)) and Path(resolved_source).suffix.lower() in (".shp", ".geojson", ".json"):
        boundary_path = str(resolved_source)
        source_decision = "boundary"
    elif isinstance(resolved_source, str):
        lower_src = resolved_source.lower()
        if lower_src in ("boundary", "project_boundary"):
            source_decision = "boundary"
        elif lower_src in ("official", "official_dataset"):
            source_decision = "official"
        elif lower_src in ("jrc", "jrc_water_context"):
            source_decision = "jrc"
        elif lower_src in ("optical", "optical_water_inference"):
            source_decision = "optical"
        elif lower_src in ("sar", "sar_inference"):
            source_decision = "sar"
        elif lower_src == "auto":
            source_decision = "auto"
        else:
            # Check if it is a path string
            if Path(resolved_source).exists() or any(char in resolved_source for char in ("/", "\\", ".")):
                boundary_path = resolved_source
                source_decision = "boundary"
            else:
                source_decision = "auto"
                
    # Decision hierarchy
    final_source_type = "UNAVAILABLE"
    
    if source_decision == "boundary":
        if boundary_path and Path(boundary_path).is_file():
            final_source_type = "PROJECT_BOUNDARY"
    elif source_decision == "official":
        if official_path and Path(official_path).is_file():
            final_source_type = "OFFICIAL_DATASET"
    elif source_decision == "jrc":
        final_source_type = "JRC_WATER_CONTEXT"
    elif source_decision == "optical":
        final_source_type = "OPTICAL_WATER_INFERENCE"
    elif source_decision == "sar":
        final_source_type = "SAR_INFERENCE"
    elif source_decision == "auto":
        # Resolve by priority order
        if boundary_path and Path(boundary_path).is_file():
            final_source_type = "PROJECT_BOUNDARY"
        elif official_path and Path(official_path).is_file():
            final_source_type = "OFFICIAL_DATASET"
        elif jrc_enabled and jrc_path and Path(jrc_path).is_file():
            final_source_type = "JRC_WATER_CONTEXT"
        elif optical_enabled:
            final_source_type = "OPTICAL_WATER_INFERENCE"
        elif sar_enabled:
            final_source_type = "SAR_INFERENCE"
            
    # 2. Build raw wetland mask
    raw_mask = np.zeros((reference_grid.height, reference_grid.width), dtype=np.uint8)
    
    # Track observation coverage separately from membership.
    source_valid_mask = np.ones(raw_mask.shape, dtype=bool)
    optical_change_map = None

    # Rasterization configurations
    rasterization_config = wetland_config.get("rasterization", {})
    all_touched = rasterization_config.get("all_touched", False)
    
    if final_source_type == "PROJECT_BOUNDARY":
        selected_source_type = "PROJECT_BOUNDARY"
        selected_status = "WETLAND_BOUNDARY_AVAILABLE"
        depletion_enabled = False
        
        geoms, src_crs = load_vector_boundary(Path(boundary_path))

        if not geoms:
            raise ValueError(
                "Project boundary contains no usable geometry."
            )
        reprojected_geoms = [reproject_geometry(g, src_crs, reference_grid.crs) for g in geoms]
        
        if reprojected_geoms:
            raw_mask = rasterize(
                shapes=[(g, 1) for g in reprojected_geoms],
                out_shape=(reference_grid.height, reference_grid.width),
                transform=reference_grid.transform,
                fill=0,
                all_touched=all_touched,
                dtype=np.uint8
            )
        source_metadata = {
            "source_name": Path(boundary_path).name,
            "source_path": str(boundary_path),
            "original_crs": str(src_crs),
            "feature_count": len(geoms)
        }
        
    elif final_source_type == "OFFICIAL_DATASET":
        selected_source_type = "OFFICIAL_DATASET"
        selected_status = "WETLAND_BOUNDARY_AVAILABLE"
        depletion_enabled = False
        
        dataset_config = wetland_config.get("official_dataset", {})
        classes = dataset_config.get("classes", [])
        class_field = dataset_config.get("class_field")
        
        geoms, src_crs = load_vector_boundary(
            Path(official_path),
            class_field=class_field,
            select_classes=classes if classes else None
        )
        
        if not geoms:

            if classes and class_field:
                raise ValueError(
                    "Dataset class semantics require configuration."
                )

            raise ValueError(
                "Official dataset contains no selected geometry."
            )

        reprojected_geoms = [reproject_geometry(g, src_crs, reference_grid.crs) for g in geoms]
        
        if reprojected_geoms:
            raw_mask = rasterize(
                shapes=[(g, 1) for g in reprojected_geoms],
                out_shape=(reference_grid.height, reference_grid.width),
                transform=reference_grid.transform,
                fill=0,
                all_touched=all_touched,
                dtype=np.uint8
            )
        else:
            if classes and class_field:
                # If classes were specified but none were parsed, it might require semantics review
                raise ValueError("Dataset class semantics require configuration.")
                
        source_metadata = {
            "source_name": Path(official_path).name,
            "source_path": str(official_path),
            "original_crs": str(src_crs),
            "feature_count": len(geoms),
            "filtered_classes": classes,
            "class_field": class_field
        }
        
    elif final_source_type == "JRC_WATER_CONTEXT":

        selected_source_type = "JRC_WATER_CONTEXT"
        selected_status = "WETLAND_EXTENT_INFERRED"
        depletion_enabled = False

        jrc_path_resolved = Path(jrc_path) if jrc_path else None

        source_metadata = {
            "dataset_name": "JRC Global Surface Water",
            "layer_used": "occurrence",
            "threshold_criterion": ">10%",
            "processing_method": "hydrological/water-history context",
        }

        try:

            if (
                jrc_path_resolved is None
                or not jrc_path_resolved.is_file()
            ):
                raise ValueError("JRC occurrence raster unavailable.")

            with rasterio.open(jrc_path_resolved) as src:

                if src.count != 1:
                    raise ValueError(
                        "JRC occurrence raster must contain one band."
                    )

                if src.crs is None or reference_grid.crs is None:
                    raise ValueError(
                        "JRC and reference grid require valid CRS."
                    )

                if src.width <= 0 or src.height <= 0:
                    raise ValueError("Invalid JRC raster dimensions.")

                coefficients = np.asarray(
                    tuple(src.transform)[:6],
                    dtype=np.float64,
                )

                if (
                    not np.all(np.isfinite(coefficients))
                    or src.transform.a * src.transform.e
                    - src.transform.b * src.transform.d == 0
                ):
                    raise ValueError("Invalid JRC raster transform.")

                jrc_data = src.read(1)

                if jrc_data.dtype.kind not in "iuf":
                    raise ValueError("Unsupported JRC raster data type.")

                valid = (
                    (src.read_masks(1) > 0)
                    & np.isfinite(jrc_data)
                    & (jrc_data >= 0)
                    & (jrc_data <= 100)
                )

                if src.nodata is not None:
                    valid &= jrc_data != src.nodata

                # Preserve zero as a valid occurrence observation.
                # Use a separate sentinel for unknown observations.
                jrc_values = np.full(
                    jrc_data.shape,
                    -9999.0,
                    dtype=np.float32,
                )

                jrc_values[valid] = jrc_data[valid]

                jrc_aligned = np.full(
                    raw_mask.shape,
                    -9999.0,
                    dtype=np.float32,
                )

                reproject(
                    source=jrc_values,
                    destination=jrc_aligned,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=reference_grid.transform,
                    dst_crs=reference_grid.crs,
                    src_nodata=-9999.0,
                    dst_nodata=-9999.0,
                    resampling=Resampling.nearest,
                )

                source_valid_mask = (
                    np.isfinite(jrc_aligned)
                    & (jrc_aligned >= 0)
                    & (jrc_aligned <= 100)
                )

                source_metadata["original_crs"] = str(src.crs)

            if not np.any(source_valid_mask):
                raise ValueError(
                    "No valid JRC observations intersect the reference grid."
                )

            raw_mask = (
                source_valid_mask & (jrc_aligned > 10)
            ).astype(np.uint8)

            source_metadata.update({
                "source_name": jrc_path_resolved.name,
                "source_path": str(jrc_path_resolved),
                "status": "JRC water-history context available.",
                "authoritative_wetland_boundary": False,
            })

        except Exception as exc:

            selected_source_type = "UNAVAILABLE"
            selected_status = "WETLAND_BOUNDARY_UNAVAILABLE"

            raw_mask = np.zeros(raw_mask.shape, dtype=np.uint8)
            source_valid_mask = np.zeros(raw_mask.shape, dtype=bool)

            source_metadata.update({
                "status": "JRC water-history context unavailable.",
                "failure_type": type(exc).__name__,
            })

    elif final_source_type == "OPTICAL_WATER_INFERENCE":
        selected_source_type = "OPTICAL_WATER_INFERENCE"
        selected_status = "WETLAND_EXTENT_INFERRED"
        depletion_enabled = False
        
        threshold = wetland_config.get("optical", {}).get("threshold", 0.1)
        optical_failure_type = None
        
        # Load green/nir bands for T1 and T2
        t1_water = np.zeros((reference_grid.height, reference_grid.width), dtype=bool)
        t2_water = np.zeros((reference_grid.height, reference_grid.width), dtype=bool)
        
        # Explicit aligned directory paths are required — no hardcoded fallback
        if not t1_aligned_dir:
            raise ValueError(
                "t1_aligned_dir must be explicitly provided for OPTICAL_WATER_INFERENCE. "
                "The production pipeline must pass the dynamically resolved aligned path."
            )
        if not t2_aligned_dir:
            raise ValueError(
                "t2_aligned_dir must be explicitly provided for OPTICAL_WATER_INFERENCE. "
                "The production pipeline must pass the dynamically resolved aligned path."
            )
        
        t1_dir = Path(t1_aligned_dir)
        t2_dir = Path(t2_aligned_dir)
        
        t1_green = t1_dir / "green.tif"
        t1_nir = t1_dir / "nir.tif"
        t2_green = t2_dir / "green.tif"
        t2_nir = t2_dir / "nir.tif"
        
        try:
            required_bands = (
                t1_green,
                t1_nir,
                t2_green,
                t2_nir,
            )

            if not all(p.is_file() for p in required_bands):
                raise ValueError(
                    "Required optical inference bands are missing."
                )

            for band_path in required_bands:

                with rasterio.open(band_path) as src:

                    if (
                        src.count != 1
                        or src.width != reference_grid.width
                        or src.height != reference_grid.height
                        or src.crs != reference_grid.crs
                        or src.transform != reference_grid.transform
                    ):
                        raise ValueError(
                            "Optical raster does not match the reference grid."
                        )

            t1_water, t1_valid = classify_optical_water(
                t1_green,
                t1_nir,
                threshold=threshold,
            )

            t2_water, t2_valid = classify_optical_water(
                t2_green,
                t2_nir,
                threshold=threshold,
            )

            source_valid_mask = t1_valid & t2_valid

            if not np.any(source_valid_mask):
                raise ValueError("No common valid optical observations.")

            raw_mask = (
                (t1_water | t2_water) & source_valid_mask
            ).astype(np.uint8)

            # Retain the existing transition class mapping.
            optical_change_map = np.full(
                raw_mask.shape,
                255,
                dtype=np.uint8,
            )

            optical_change_map[
                source_valid_mask & ~t1_water & ~t2_water
            ] = 0

            optical_change_map[
                source_valid_mask & t1_water & t2_water
            ] = 1

            optical_change_map[
                source_valid_mask & ~t1_water & t2_water
            ] = 2

            optical_change_map[
                source_valid_mask & t1_water & ~t2_water
            ] = 3

        except Exception as e:

            source_valid_mask = np.zeros(
                raw_mask.shape,
                dtype=bool,
            )

            optical_change_map = None
            optical_failure_type = type(e).__name__

            selected_source_type = "UNAVAILABLE"
            selected_status = "WETLAND_BOUNDARY_UNAVAILABLE"

            raw_mask = np.zeros(
                (reference_grid.height, reference_grid.width),
                dtype=np.uint8,
            )
            
        source_metadata = {
            "source_name": "Local Optical Water Classifier",
            "threshold": threshold,
            "index": "NDWI",
            "failure_type": optical_failure_type,
            "common_temporal_observations_required": True,
            "authoritative_wetland_boundary": False,
            "status": (
                "Optical inference unavailable."
                if selected_source_type == "UNAVAILABLE"
                else "Wetland extent inferred from water evidence; not an authoritative wetland boundary."
            )
        }
        
    elif final_source_type == "SAR_INFERENCE":
        selected_source_type = "UNAVAILABLE"
        selected_status = "WETLAND_BOUNDARY_UNAVAILABLE"
        depletion_enabled = False
        source_metadata = {
            "source_name": "SAR water/inundation information",
            "status": "Not implemented/unavailable fallback"
        }
        raw_mask = np.zeros((reference_grid.height, reference_grid.width), dtype=np.uint8)
        
    else:
        selected_source_type = "UNAVAILABLE"
        selected_status = "WETLAND_BOUNDARY_UNAVAILABLE"
        depletion_enabled = False
        source_metadata = {
            "status": "No reliable wetland boundary or water context available."
        }
        raw_mask = np.ones((reference_grid.height, reference_grid.width), dtype=np.uint8)

    def validate_binary_mask(mask, name):

        expected_shape = (
            reference_grid.height,
            reference_grid.width,
        )

        if not isinstance(mask, np.ndarray):
            raise ValueError(
                f"{name} must be a NumPy array."
            )

        if mask.ndim != 2 or mask.shape != expected_shape:
            raise ValueError(
                f"{name} shape must match the reference grid."
            )

        if mask.dtype.kind not in "biuf":
            raise ValueError(
                f"{name} has an unsupported data type."
            )

        if not np.all(np.isin(mask, [0, 1, 255])):
            raise ValueError(
                f"{name} contains invalid mask values."
            )

        # Treat explicit NoData as excluded, never valid.
        return (mask == 1).astype(np.uint8)

    # 3. Resolve overlap mask and AOI
    # Resolving overlap mask
    o_mask = None
    if overlap_mask is not None:
        if isinstance(overlap_mask, np.ndarray):
            o_mask = overlap_mask
        else:
            with rasterio.open(overlap_mask) as src:

                if (
                    src.count != 1
                    or src.width != reference_grid.width
                    or src.height != reference_grid.height
                    or src.crs != reference_grid.crs
                    or src.transform != reference_grid.transform
                ):
                    raise ValueError(
                        "Overlap raster must match the reference grid."
                    )

                o_mask = src.read(1, masked=True).filled(0)
    else:
        # No overlap mask provided — default to all-valid.
        # The production pipeline always passes an explicit overlap_mask;
        # this fallback exists only for stand-alone/test use.
        o_mask = np.ones((reference_grid.height, reference_grid.width), dtype=np.uint8)
            
    o_mask = validate_binary_mask(
        o_mask,
        "Overlap mask",
    )

    # Resolve AOI mask
    aoi_mask = None
    if aoi is not None:
        if isinstance(aoi, np.ndarray):
            aoi_mask = aoi
        elif isinstance(aoi, (str, Path)):
            aoi_path = Path(aoi)
            if aoi_path.suffix.lower() in (".shp", ".geojson", ".json"):
                # Vector AOI boundary
                aoi_geoms, aoi_crs = load_vector_boundary(aoi_path)

                if not aoi_geoms:
                    raise ValueError(
                        "AOI contains no usable geometry."
                    )
                aoi_rep_geoms = [reproject_geometry(g, aoi_crs, reference_grid.crs) for g in aoi_geoms]
                if aoi_rep_geoms:
                    aoi_mask = rasterize(
                        shapes=[(g, 1) for g in aoi_rep_geoms],
                        out_shape=(reference_grid.height, reference_grid.width),
                        transform=reference_grid.transform,
                        fill=0,
                        all_touched=all_touched,
                        dtype=np.uint8
                    )
            else:
                # Raster AOI
                with rasterio.open(aoi_path) as src:
                    if src.count != 1:
                        raise ValueError(
                            "AOI raster must contain one band."
                        )

                    aoi_data = src.read(
                        1,
                        masked=True,
                    ).filled(0)

                    if (
                        aoi_data.dtype.kind not in "biuf"
                        or not np.all(
                            np.isin(aoi_data, [0, 1, 255])
                        )
                    ):
                        raise ValueError(
                            "AOI raster contains invalid mask values."
                        )

                    aoi_data = (
                        aoi_data == 1
                    ).astype(np.uint8)

                    if (
                        src.crs != reference_grid.crs
                        or src.transform != reference_grid.transform
                        or src.width != reference_grid.width
                        or src.height != reference_grid.height
                    ):
                        aoi_mask = np.zeros((reference_grid.height, reference_grid.width), dtype=np.uint8)
                        reproject(
                            source=aoi_data,
                            destination=aoi_mask,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=reference_grid.transform,
                            dst_crs=reference_grid.crs,
                            src_nodata=0,
                            dst_nodata=0,
                            resampling=Resampling.nearest
                        )
                    else:
                        aoi_mask = aoi_data
                        
    if aoi_mask is None:

        if aoi is not None:
            raise ValueError(
                "Provided AOI could not be resolved."
            )

        aoi_mask = np.ones(
            (reference_grid.height, reference_grid.width),
            dtype=np.uint8,
        )

    aoi_mask = validate_binary_mask(
        aoi_mask,
        "AOI mask",
    )

    raw_mask = validate_binary_mask(
        raw_mask,
        "Wetland membership mask",
    )
        
    # Unavailable membership must never represent the
    # entire analysis region as confirmed wetland.
    if selected_source_type == "UNAVAILABLE":

        raw_mask = np.zeros(
            (reference_grid.height, reference_grid.width),
            dtype=np.uint8,
        )

        depletion_enabled = False

    # 4. Integrate validity and NoData
    # Raw wetland mask represents spatial wetland membership (0 = outside, 1 = inside)
    # Valid analysis mask restricts raw mask to valid overlap and AOI
    requested_analysis_mask = (o_mask == 1) & (aoi_mask == 1)

    if selected_source_type == "UNAVAILABLE":
        source_valid_mask = np.zeros(raw_mask.shape, dtype=bool)

    requested_pixel_count = int(
        np.count_nonzero(requested_analysis_mask)
    )

    observed_pixel_count = int(
        np.count_nonzero(
            requested_analysis_mask & source_valid_mask
        )
    )

    inferred_source = selected_source_type in (
        "JRC_WATER_CONTEXT",
        "OPTICAL_WATER_INFERENCE",
    )

    if inferred_source and observed_pixel_count == 0:

        source_metadata["status"] = (
            "No valid source observations inside the analysis region."
        )

        selected_source_type = "UNAVAILABLE"
        selected_status = "WETLAND_BOUNDARY_UNAVAILABLE"

        raw_mask = np.zeros(raw_mask.shape, dtype=np.uint8)
        source_valid_mask = np.zeros(raw_mask.shape, dtype=bool)

        optical_change_map = None
        depletion_enabled = False

    valid_data_mask = (
        requested_analysis_mask & source_valid_mask
    )

    if optical_change_map is not None:
        optical_change_map[~valid_data_mask] = 255

    if selected_source_type == "UNAVAILABLE":
        coverage_status = "UNAVAILABLE"
    elif requested_pixel_count == 0:
        coverage_status = "NOT_AVAILABLE"
    elif observed_pixel_count == requested_pixel_count:
        coverage_status = "COMPLETE"
    else:
        coverage_status = "PARTIAL"

    observation_coverage = {
        "status": coverage_status,
        "requested_pixel_count": requested_pixel_count,
        "observed_pixel_count": observed_pixel_count,
        "unknown_pixel_count": (
            requested_pixel_count - observed_pixel_count
        ),
        "coverage_fraction": (
            observed_pixel_count / requested_pixel_count
            if requested_pixel_count > 0
            else None
        ),
        "grid_observed_pixel_count": int(
            np.count_nonzero(source_valid_mask)
        ),
        "grid_pixel_count": int(source_valid_mask.size),
        "basis": (
            "SOURCE_OBSERVATIONS"
            if inferred_source
            else "BOUNDARY_MEMBERSHIP"
            if selected_source_type != "UNAVAILABLE"
            else "UNAVAILABLE"
        ),
    }

    source_metadata["observation_coverage"] = observation_coverage
    
    # Raw wetland mask
    final_wetland_mask = raw_mask.copy().astype(np.uint8)
    
    # Valid analysis mask:
    # 1 inside valid wetland analysis area (inside wetland polygon AND valid overlap AND inside AOI)
    # 0 otherwise (outside wetland OR invalid overlap OR outside AOI)
    final_analysis_mask = np.zeros(raw_mask.shape, dtype=np.uint8)
    final_analysis_mask[valid_data_mask & (raw_mask == 1)] = 1
    
    # 5. Geodetic Area Calculation
    # Physical m²/ha is only reported when the reference grid has
    # authoritative georeferencing. CRS-less identity/pixel-space imagery
    # must not be interpreted as metres.
    physical_area_available = True
    area_basis = "physical"
    pixel_areas = None

    try:
        pixel_areas = calculate_pixel_areas(
            reference_grid.transform,
            (reference_grid.height, reference_grid.width),
            reference_grid.crs,
        )
    except PhysicalAreaUnavailableError:
        physical_area_available = False
        area_basis = "pixel_count"

    # Pixel counts remain valid regardless of georeferencing.
    wetland_pixel_count = int(np.sum(raw_mask == 1))
    valid_pixel_count = int(np.sum(final_analysis_mask == 1))

    if physical_area_available and pixel_areas is not None:
        wetland_area_m2 = float(np.sum(pixel_areas[raw_mask == 1]))
        wetland_area_ha = wetland_area_m2 / 10000.0
        valid_area_m2 = float(np.sum(pixel_areas[final_analysis_mask == 1]))
        valid_area_ha = valid_area_m2 / 10000.0
    else:
        wetland_area_m2 = None
        wetland_area_ha = None
        valid_area_m2 = None
        valid_area_ha = None

    if selected_source_type == "UNAVAILABLE":

        wetland_area_m2 = None
        wetland_area_ha = None
        valid_area_m2 = None
        valid_area_ha = None

        wetland_pixel_count = None
        valid_pixel_count = None

        physical_area_available = False
        area_basis = "unavailable"

    # 6. Save outputs if output_dir is provided
    # Build complete metadata dictionary
    metadata = {
        "source_type": selected_source_type,
        "source_name": source_metadata.get("source_name", "None"),
        "source_version": source_metadata.get("source_version"),
        "source_date": source_metadata.get("source_date"),
        "source_authority": source_metadata.get("source_authority"),
        "original_crs": source_metadata.get("original_crs", str(reference_grid.crs)),
        "target_crs": str(reference_grid.crs),
        "rasterization_method": "rasterio.features.rasterize",
        "all_touched_setting": all_touched,
        "feature_selection_rule": source_metadata.get("filtered_classes", []),
        "aoi_information": {
            "has_aoi": aoi is not None,
            "aoi_source": str(aoi) if aoi is not None else None
        },
        "reference_grid": reference_grid.to_dict(),
        "wetland_pixel_count": wetland_pixel_count,
        "wetland_area_m2": wetland_area_m2,
        "wetland_area_ha": wetland_area_ha,
        "valid_wetland_analysis_pixel_count": valid_pixel_count,
        "valid_wetland_analysis_area_m2": valid_area_m2,
        "valid_wetland_analysis_area_ha": valid_area_ha,
        "physical_area_available": physical_area_available,
        "area_basis": area_basis,
        "confidence_status": selected_status,
        "observation_coverage": observation_coverage,
        "area_is_observed_membership_only": (
            selected_source_type in (
                "JRC_WATER_CONTEXT",
                "OPTICAL_WATER_INFERENCE",
            )
        ),
        "wetland_membership_verified": False,
        "authoritative_boundary_source": (
            selected_source_type in (
                "PROJECT_BOUNDARY",
                "OFFICIAL_DATASET",
            )
        ),
        "boundary_available": (
            selected_source_type in (
                "PROJECT_BOUNDARY",
                "OFFICIAL_DATASET",
            )
        ),
        "wetland_context_available": (
            selected_source_type != "UNAVAILABLE"
        ),
        "wetland_metrics_available": (
            selected_source_type != "UNAVAILABLE"
        ),
        "analysis_scope": (
            "UNAVAILABLE"
            if selected_source_type == "UNAVAILABLE"
            else "AUTHORITATIVE_BOUNDARY"
            if selected_source_type in (
                "PROJECT_BOUNDARY",
                "OFFICIAL_DATASET",
            )
            else "INFERRED_WATER_CONTEXT"
        ),
        "physical_wetland_depletion_enabled": depletion_enabled,
        "source_metadata": source_metadata
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save wetland mask geotiff
        profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "nodata": 255,
            "width": reference_grid.width,
            "height": reference_grid.height,
            "count": 1,
            "crs": reference_grid.crs,
            "transform": reference_grid.transform,
            "compress": "lzw"
        }
        
        with rasterio.open(output_dir / "wetland_mask.tif", "w", **profile) as dst:
            dst.write(final_wetland_mask, 1)
            
        with rasterio.open(output_dir / "valid_wetland_analysis_mask.tif", "w", **profile) as dst:
            dst.write(final_analysis_mask, 1)
            
        with rasterio.open(
            output_dir / "wetland_source_validity.tif",
            "w",
            **profile,
        ) as dst:
            dst.write(source_valid_mask.astype(np.uint8), 1)

        if optical_change_map is not None:

            with rasterio.open(
                output_dir / "wetland_extent_change.tif",
                "w",
                **profile,
            ) as dst:
                dst.write(optical_change_map, 1)

        with open(output_dir / "wetland_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    return WetlandMaskResult(
        wetland_mask=final_wetland_mask,
        valid_analysis_mask=final_analysis_mask,
        source_type=selected_source_type,
        source_metadata=source_metadata,
        wetland_area_m2=wetland_area_m2,
        wetland_area_ha=wetland_area_ha,
        valid_analysis_area_m2=valid_area_m2,
        valid_analysis_area_ha=valid_area_ha,
        status=selected_status,
        physical_wetland_depletion_enabled=depletion_enabled,
        metadata=metadata
    )
