import math
import numpy as np

from numbers import Integral, Real
from typing import Dict, Any, Optional, List


def estimate_area_with_reference_samples(
    classification_map: Any,
    reference_samples: Optional[List[Dict[str, Any]]] = None,
    strata_definition: Optional[Dict[int, Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Calculate conditional stratified area estimates safely."""

    def unavailable(message, status="NOT_AVAILABLE"):
        return {
            "status": status,
            "model_derived_estimate": True,
            "independent_area_error_correction": False,
            "independent_validation_verified": False,
            "calculation_completed": False,
            "message": message,
        }

    def valid_class(value):
        return (
            isinstance(value, Integral)
            and not isinstance(value, (bool, np.bool_))
            and 0 <= int(value) <= 3
        )

    if config is not None and not isinstance(config, dict):
        return unavailable(
            "Invalid configuration dictionary.",
            "FAIL",
        )

    if reference_samples is None:
        return unavailable(
            "Model-derived estimate; no independent "
            "area-error correction applied."
        )

    if not isinstance(reference_samples, (list, tuple)):
        return unavailable(
            "Reference samples must be a list or tuple.",
            "FAIL",
        )

    if len(reference_samples) == 0:
        return unavailable(
            "Model-derived estimate; no independent "
            "area-error correction applied."
        )

    samples = []

    for sample in reference_samples:

        if not isinstance(sample, dict):
            return unavailable(
                "Invalid reference sample structure.",
                "FAIL",
            )

        mapped = sample.get("mapped_class")
        reference = sample.get("reference_class")

        if not valid_class(mapped) or not valid_class(reference):
            return unavailable(
                "Invalid mapped or reference class identifier.",
                "FAIL",
            )

        samples.append(
            (int(mapped), int(reference))
        )

    map_counts = {}

    if classification_map is not None:

        if (
            not isinstance(classification_map, np.ndarray)
            or classification_map.ndim != 2
            or classification_map.size == 0
            or classification_map.dtype.kind not in "iu"
        ):
            return unavailable(
                "Classification map must be a nonempty "
                "two-dimensional integer array.",
                "FAIL",
            )

        if not np.all(
            np.isin(classification_map, [0, 1, 2, 3, 255])
        ):
            return unavailable(
                "Classification map contains invalid classes.",
                "FAIL",
            )

        map_counts = {
            c: int(np.count_nonzero(classification_map == c))
            for c in range(4)
        }

    strata_areas = {}

    if strata_definition is not None:

        if not isinstance(strata_definition, dict):
            return unavailable(
                "Invalid strata definition.",
                "FAIL",
            )

        for key, definition in strata_definition.items():

            if not valid_class(key):
                return unavailable(
                    "Invalid stratum identifier.",
                    "FAIL",
                )

            if not isinstance(definition, dict):
                return unavailable(
                    "Invalid stratum definition.",
                    "FAIL",
                )

            area = definition.get("area")

            if (
                not isinstance(area, Real)
                or isinstance(area, (bool, np.bool_))
            ):
                return unavailable(
                    "Stratum area must be numeric.",
                    "FAIL",
                )

            try:
                area = float(area)
            except (TypeError, ValueError, OverflowError):
                return unavailable(
                    "Invalid stratum area.",
                    "FAIL",
                )

            if not math.isfinite(area) or area < 0:
                return unavailable(
                    "Stratum area must be finite and nonnegative.",
                    "FAIL",
                )

            strata_areas[int(key)] = area

        area_basis = "provided_stratum_area"

        if classification_map is not None:

            for c, count in map_counts.items():

                area = strata_areas.get(c, 0.0)

                if count > 0 and area <= 0:
                    return unavailable(
                        "A mapped class has missing or zero stratum area.",
                        "FAIL",
                    )

                if count == 0 and area > 0:
                    return unavailable(
                        "Positive stratum area has no mapped pixels.",
                        "FAIL",
                    )

    elif classification_map is not None:

        strata_areas = {
            c: float(count)
            for c, count in map_counts.items()
            if count > 0
        }

        area_basis = "pixel_count"

    else:

        return unavailable(
            "Mapped stratum areas or classification map required. "
            "Equal-area assumptions are not permitted."
        )

    try:
        total_area = math.fsum(strata_areas.values())
    except OverflowError:
        return unavailable(
            "Total stratum area overflow.",
            "FAIL",
        )

    if not math.isfinite(total_area):
        return unavailable(
            "Total stratum area is not finite.",
            "FAIL",
        )

    if total_area <= 0:
        return unavailable(
            "Total analysis area is zero."
        )

    active_strata = sorted(
        c for c, area in strata_areas.items()
        if area > 0
    )

    classes = sorted(
        set(strata_areas)
        | {reference for _, reference in samples}
    )

    sample_counts = {
        c: 0 for c in classes
    }

    confusion = {
        h: {j: 0 for j in classes}
        for h in active_strata
    }

    for mapped, reference in samples:

        if mapped not in active_strata:
            return unavailable(
                "Reference sample belongs to an unknown "
                "or zero-area mapped stratum.",
                "FAIL",
            )

        sample_counts[mapped] += 1
        confusion[mapped][reference] += 1

    insufficient = [
        h for h in active_strata
        if sample_counts[h] < 2
    ]

    if insufficient:
        result = unavailable(
            "Every positive-area stratum requires at least "
            "two reference samples for variance estimation."
        )

        result["insufficient_strata"] = insufficient
        result["sample_counts"] = sample_counts

        return result

    weights = {
        h: strata_areas[h] / total_area
        for h in active_strata
    }

    adjusted_areas = {}

    for j in classes:

        proportion = math.fsum(
            weights[h]
            * confusion[h][j]
            / sample_counts[h]
            for h in active_strata
        )

        variance = math.fsum(
            weights[h] ** 2
            * (confusion[h][j] / sample_counts[h])
            * (1.0 - confusion[h][j] / sample_counts[h])
            / (sample_counts[h] - 1)
            for h in active_strata
        )

        standard_error = math.sqrt(variance)

        estimate = proportion * total_area
        area_error = standard_error * total_area

        lower = max(
            0.0,
            proportion - 1.96 * standard_error,
        ) * total_area

        upper = min(
            1.0,
            proportion + 1.96 * standard_error,
        ) * total_area

        adjusted_areas[j] = {
            "stratum": j,
            "mapped_area": strata_areas.get(j, 0.0),
            "mapped_proportion": weights.get(j, 0.0),
            "adjusted_area_estimate": estimate,
            "adjusted_proportion_estimate": proportion,
            "standard_error_proportion": standard_error,
            "standard_error_area": area_error,
            "confidence_interval_95_lower": lower,
            "confidence_interval_95_upper": upper,
        }

    try:

        adjusted_total = math.fsum(
            item["adjusted_area_estimate"]
            for item in adjusted_areas.values()
        )

    except OverflowError:

        return unavailable(
            "Adjusted area calculation overflow.",
            "FAIL",
        )

    if not math.isclose(
        adjusted_total,
        total_area,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        return unavailable(
            "Adjusted areas do not reconcile with total area.",
            "FAIL",
        )

    return {
        "status": "WARN",
        "model_derived_estimate": False,
        "independent_area_error_correction": False,
        "independent_validation_verified": False,
        "calculation_completed": True,
        "total_area": total_area,
        "area_basis": area_basis,
        "sample_counts": sample_counts,
        "strata_results": adjusted_areas,
        "message": (
            "Conditional stratified estimates calculated. "
            "Sample independence, sampling design and "
            "reference provenance have not been verified. "
            "Confidence intervals are approximate and conditional "
            "on the assumed sampling design."
        ),
    }
