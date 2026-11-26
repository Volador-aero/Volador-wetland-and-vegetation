
import json
import math
import logging
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import shapefile
from rasterio.crs import CRS
from rasterio.warp import transform


logger = logging.getLogger(__name__)


def validate_boundary_geometry(
    geom: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate polygon structure and finite coordinates."""

    if not isinstance(geom, dict):
        raise ValueError(
            "Boundary geometry must be a dictionary."
        )

    geometry_type = geom.get("type")
    coordinates = geom.get("coordinates")

    if geometry_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(
            "Boundary must be Polygon or MultiPolygon."
        )

    if (
        not isinstance(coordinates, (list, tuple))
        or not coordinates
    ):
        raise ValueError(
            "Boundary geometry has no coordinates."
        )

    polygons = (
        [coordinates]
        if geometry_type == "Polygon"
        else coordinates
    )

    for polygon in polygons:

        if (
            not isinstance(polygon, (list, tuple))
            or not polygon
        ):
            raise ValueError(
                "Boundary polygon has no rings."
            )

        for ring in polygon:

            if (
                not isinstance(ring, (list, tuple))
                or len(ring) < 4
            ):
                raise ValueError(
                    "Polygon ring requires four positions."
                )

            points = []

            for point in ring:

                if (
                    not isinstance(point, (list, tuple))
                    or len(point) < 2
                ):
                    raise ValueError(
                        "Invalid polygon coordinate."
                    )

                for value in point:

                    if (
                        isinstance(value, bool)
                        or not isinstance(value, Real)
                        or not math.isfinite(value)
                    ):
                        raise ValueError(
                            "Boundary coordinates must be finite numbers."
                        )

                points.append(
                    (float(point[0]), float(point[1]))
                )

            if points[0] != points[-1]:
                raise ValueError(
                    "Polygon ring must be closed."
                )

            if len(set(points[:-1])) < 3:
                raise ValueError(
                    "Polygon ring has insufficient unique vertices."
                )

            # Shift coordinates to reduce numerical cancellation
            # when projected coordinates have large values.
            origin_x, origin_y = points[0]

            shifted = [
                (x - origin_x, y - origin_y)
                for x, y in points
            ]

            try:
                signed_area = math.fsum(
                    x1 * y2 - x2 * y1
                    for (x1, y1), (x2, y2)
                    in zip(shifted, shifted[1:])
                )

            except (OverflowError, ValueError) as exc:
                raise ValueError(
                    "Polygon ring has invalid area."
                ) from exc

            if (
                not math.isfinite(signed_area)
                or signed_area == 0
            ):
                raise ValueError(
                    "Polygon ring has invalid or zero area."
                )

    return geom


def reproject_geometry(
    geom: Dict[str, Any],
    src_crs: CRS,
    dst_crs: CRS,
) -> Dict[str, Any]:
    """Reproject validated Polygon or MultiPolygon geometry."""

    validate_boundary_geometry(geom)

    if src_crs is None or dst_crs is None:
        raise ValueError(
            "Source and destination CRS are required."
        )

    try:
        src_crs = CRS.from_user_input(src_crs)
        dst_crs = CRS.from_user_input(dst_crs)

    except Exception as exc:
        raise ValueError(
            "Invalid boundary CRS."
        ) from exc

    if not src_crs or not dst_crs:
        raise ValueError(
            "Boundary CRS cannot be empty."
        )

    # Geometry is validated even when no reprojection is needed.
    if src_crs == dst_crs:
        return geom

    def project_ring(ring):

        xs = [point[0] for point in ring]
        ys = [point[1] for point in ring]

        try:
            new_xs, new_ys = transform(
                src_crs,
                dst_crs,
                xs,
                ys,
            )

        except Exception as exc:
            raise ValueError(
                "Boundary coordinate reprojection failed."
            ) from exc

        return list(zip(new_xs, new_ys))

    geometry_type = geom["type"]
    coordinates = geom["coordinates"]

    if geometry_type == "Polygon":

        projected = [
            project_ring(ring)
            for ring in coordinates
        ]

    else:

        projected = [
            [
                project_ring(ring)
                for ring in polygon
            ]
            for polygon in coordinates
        ]

    result = {
        "type": geometry_type,
        "coordinates": projected,
    }

    # Reject nonfinite or malformed transformed geometry.
    return validate_boundary_geometry(result)


def load_vector_boundary(
    file_path: Path,
    class_field: Optional[str] = None,
    select_classes: Optional[List[Any]] = None,
) -> Tuple[List[Dict[str, Any]], CRS]:
    """
    Load polygon boundaries from GeoJSON or Shapefile.

    Supports optional class-based feature filtering.

    Returns:
        Selected polygon geometries and source CRS.
    """

    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Boundary file not found: {file_path}"
        )

    suffix = file_path.suffix.lower()

    # ==========================================================
    # GEOJSON BOUNDARY
    # ==========================================================

    if suffix in (".geojson", ".json"):

        with open(
            file_path,
            "r",
            encoding="utf-8-sig",
        ) as file:

            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError(
                "GeoJSON root must be an object."
            )

        # Standard GeoJSON defaults to WGS84 only when
        # no explicit CRS member is supplied.
        if "crs" not in data:

            crs = CRS.from_epsg(4326)

        else:

            try:
                crs_name = data["crs"]["properties"]["name"]

                if (
                    not isinstance(crs_name, str)
                    or not crs_name.strip()
                ):
                    raise ValueError("Empty CRS name.")

                crs = CRS.from_string(crs_name)

                if not crs:
                    raise ValueError("Empty CRS.")

            except Exception as exc:
                raise ValueError(
                    "Invalid explicit GeoJSON CRS."
                ) from exc

        features = []

        def parse_element(element):

            if not isinstance(element, dict):
                raise ValueError(
                    "Invalid GeoJSON element."
                )

            element_type = element.get("type")

            if element_type == "FeatureCollection":

                collection = element.get("features")

                if not isinstance(collection, list):
                    raise ValueError(
                        "GeoJSON features must be a list."
                    )

                for feature in collection:
                    parse_element(feature)

            elif element_type == "Feature":

                geom = element.get("geometry")
                props = element.get("properties") or {}

                if not isinstance(props, dict):
                    raise ValueError(
                        "GeoJSON properties must be an object."
                    )

                if class_field and select_classes:

                    value = props.get(class_field)

                    if value not in select_classes:
                        return

                if geom is None:
                    return

                if not isinstance(geom, dict):
                    raise ValueError(
                        "Invalid GeoJSON geometry."
                    )

                if geom.get("type") in (
                    "Polygon",
                    "MultiPolygon",
                ):
                    features.append(
                        validate_boundary_geometry(geom)
                    )

            elif element_type in (
                "Polygon",
                "MultiPolygon",
            ):

                features.append(
                    validate_boundary_geometry(element)
                )

        parse_element(data)

        return features, crs

    # ==========================================================
    # SHAPEFILE BOUNDARY
    # ==========================================================

    elif suffix in (".shp", ""):

        prj_path = file_path.with_suffix(".prj")

        if not prj_path.is_file():
            raise ValueError(
                "Shapefile CRS is missing: .prj required."
            )

        try:
            wkt = prj_path.read_text(
                encoding="utf-8-sig"
            ).strip()

            if not wkt:
                raise ValueError(
                    "Empty shapefile CRS."
                )

            crs = CRS.from_string(wkt)

            if not crs:
                raise ValueError(
                    "Empty shapefile CRS."
                )

        except Exception as exc:
            raise ValueError(
                "Invalid shapefile CRS."
            ) from exc

        features = []

        # Context manager ensures the reader is closed,
        # including when validation fails.
        with shapefile.Reader(str(file_path)) as sf:

            fields = [
                field[0]
                for field in sf.fields[1:]
            ]

            for shape_rec in sf.iterShapeRecords():

                properties = dict(
                    zip(fields, shape_rec.record)
                )

                if class_field and select_classes:

                    value = properties.get(class_field)

                    if isinstance(value, bytes):
                        value = value.decode(
                            "utf-8",
                            errors="ignore",
                        )

                    selected_values = [
                        str(item)
                        for item in select_classes
                    ]

                    if str(value) not in selected_values:
                        continue

                geom = shape_rec.shape.__geo_interface__

                if geom.get("type") in (
                    "Polygon",
                    "MultiPolygon",
                ):

                    features.append(
                        validate_boundary_geometry(geom)
                    )

        return features, crs

    else:

        raise ValueError(
            f"Unsupported boundary format: {suffix}"
        )
