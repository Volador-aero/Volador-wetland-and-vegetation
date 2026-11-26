import os
import numpy as np
from pathlib import Path
from scipy.ndimage import label
from typing import Dict, Any, List, Optional, Tuple
import onnxruntime as ort

class QualityAssuranceEngine:
    VALID_STATUSES = {
        "NOT_RUN",
        "PASS",
        "WARN",
        "FAIL",
        "NOT_AVAILABLE",
    }

    VALID_CATEGORIES = {
        "input",
        "model",
        "spatial",
        "temporal",
        "change",
        "wetland",
        "metrics",
        "uncertainty",
        "reproducibility",
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.flags = []

        # A category cannot be considered PASS until its checks execute.
        self.categories_status = {
            category: "NOT_RUN"
            for category in self.VALID_CATEGORIES
        }

    def mark_category(self, category: str, status: str):
        """Explicitly update the lifecycle status of a QA category."""

        if category not in self.VALID_CATEGORIES:
            raise ValueError(f"Unknown QA category: {category}")

        if status not in self.VALID_STATUSES:
            raise ValueError(f"Invalid QA status: {status}")

        current = self.categories_status[category]

        # FAIL is final unless the engine is recreated.
        if current == "FAIL":
            return

        if status == "FAIL":
            self.categories_status[category] = "FAIL"
            return

        # WARN must not be downgraded to PASS.
        if current == "WARN":
            return

        if status == "WARN":
            self.categories_status[category] = "WARN"
            return

        # A completed check can replace NOT_RUN or NOT_AVAILABLE.
        if status == "PASS":
            if current in ("NOT_RUN", "NOT_AVAILABLE", "PASS"):
                self.categories_status[category] = "PASS"
            return

        # Mark unavailable only if no completed result exists.
        if status == "NOT_AVAILABLE":
            if current == "NOT_RUN":
                self.categories_status[category] = "NOT_AVAILABLE"

    def add_flag(
        self,
        code: str,
        severity: str,
        message: str,
        phase: str,
        status: str,
        recommended_action: str,
    ):
        """Record a finding and update the correct QA category."""

        severity = severity.strip().upper()
        status = status.strip().upper()

        # Normalize legacy severity values.
        if severity == "WARN":
            severity = "MEDIUM"

        valid_severities = {
            "CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"
        }

        if severity not in valid_severities:
            raise ValueError(f"Invalid QA severity: {severity}")

        if status not in self.VALID_STATUSES:
            raise ValueError(f"Invalid QA flag status: {status}")

        phase_map = {
            "input ingestion": "input",
            "preprocessing": "input",
            "model qa": "model",
            "model inference": "model",
            "spatial alignment": "spatial",
            "nodata qa": "spatial",
            "temporal alignment": "temporal",
            "transition mapping": "change",
            "cva math invariant": "change",
            "cva engine": "change",
            "wetland analysis": "wetland",
            "metrics qa": "metrics",
            "uncertainty qa": "uncertainty",
            "reproducibility qa": "reproducibility",
        }

        category_map = {
            "QA-001": "input",
            "QA-002": "input",
            "QA-003": "spatial",
            "QA-004": "spatial",
            "QA-005": "spatial",
            "QA-006": "temporal",
            "QA-007": "temporal",
            "QA-008": "model",
            "QA-009": "change",
            "QA-010": "change",
            "QA-011": "wetland",
            "QA-012": "uncertainty",
            "QA-013": "model",
            "QA-014": "uncertainty",
            "QA-015": "metrics",
            "QA-016": "metrics",
            "QA-017": "reproducibility",
            "QA-018": "spatial",
            "QA-019": "spatial",
            "QA-020": "metrics",
        }

        category = phase_map.get(
            phase.strip().lower(),
            category_map.get(code)
        )

        if category is None:
            raise ValueError(
                f"Cannot determine QA category for {code}: {phase}"
            )

        flag = {
            "code": code,
            "severity": severity,
            "message": message,
            "phase": phase,
            "status": status,
            "recommended_action": recommended_action,
            "category": category,
        }

        self.flags.append(flag)

        # Severity describes importance.
        # Status describes the actual validation outcome.
        self.mark_category(category, status)


    # 1. INPUT QA
    def run_input_qa(
        self,
        t1_bands: Dict[str, Any],
        t2_bands: Dict[str, Any],
        meta_t1: Any,
        meta_t2: Any,
    ) -> Dict[str, Any]:
        """Validate spectral input bands and array geometry."""

        results = {"status": "PASS", "bands": {}}
        required_bands = ("green", "red", "rededge", "nir")

        def finding(code, severity, message, status, action):
            self.add_flag(
                code,
                severity,
                message,
                "Input Ingestion",
                status,
                action,
            )

            if status == "FAIL":
                results["status"] = "FAIL"
            elif status == "WARN" and results["status"] != "FAIL":
                results["status"] = "WARN"

        observations = (('T1', t1_bands),)
        if getattr(self, "qa_analysis_mode", None) != "single_date":
            observations += (('T2', t2_bands),)

        for (date_name, bands) in observations:
            if not isinstance(bands, dict):
                finding(
                    "QA-001", "CRITICAL",
                    f"{date_name} spectral dataset is missing or invalid.",
                    "FAIL",
                    "Provide a valid spectral band dictionary.",
                )
                continue

            reference_shape = None

            for band in required_bands:
                key = f"{date_name}_{band}"
                arr = bands.get(band)

                if arr is None:
                    finding(
                        "QA-001", "CRITICAL",
                        f"Required band {key} is missing.",
                        "FAIL",
                        f"Provide the missing {band} band.",
                    )
                    continue

                if not isinstance(arr, np.ndarray):
                    finding(
                        "QA-020", "CRITICAL",
                        f"{key} is not a NumPy array.",
                        "FAIL",
                        "Load the band as a numeric raster array.",
                    )
                    continue

                if arr.ndim != 2 or arr.size == 0:
                    finding(
                        "QA-002", "CRITICAL",
                        f"{key} has invalid dimensions or is empty.",
                        "FAIL",
                        "Provide a non-empty two-dimensional raster.",
                    )
                    continue

                if arr.dtype.kind not in "iuf":
                    finding(
                        "QA-020", "CRITICAL",
                        f"{key} has unsupported dtype {arr.dtype}.",
                        "FAIL",
                        "Convert the band to real numeric values.",
                    )
                    continue

                if reference_shape is None:
                    reference_shape = arr.shape
                elif arr.shape != reference_shape:
                    finding(
                        "QA-002", "CRITICAL",
                        f"{key} shape {arr.shape} differs from {reference_shape}.",
                        "FAIL",
                        "Align all spectral bands within the acquisition.",
                    )

                finite_mask = np.isfinite(arr)
                valid_count = int(np.count_nonzero(finite_mask))
                invalid_count = int(arr.size - valid_count)

                stats = {
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "valid_pixels": valid_count,
                    "invalid_pixels": invalid_count,
                    "min": None,
                    "max": None,
                    "mean": None,
                    "median": None,
                    "p1": None,
                    "p99": None,
                }

                results["bands"][key] = stats

                if valid_count == 0:
                    finding(
                        "QA-003", "CRITICAL",
                        f"{key} contains no finite pixels.",
                        "FAIL",
                        "Provide valid raster data.",
                    )
                    continue

                values = arr[finite_mask].astype(np.float64)

                if invalid_count:
                    finding(
                        "QA-003", "HIGH",
                        f"{key} contains {invalid_count} non-finite pixels.",
                        "WARN",
                        "Verify NoData masking and preprocessing.",
                    )

                minimum = float(np.min(values))
                maximum = float(np.max(values))
                p1, median, p99 = np.percentile(values, [1, 50, 99])

                stats.update({
                    "min": minimum,
                    "max": maximum,
                    "mean": float(np.mean(values)),
                    "median": float(median),
                    "p1": float(p1),
                    "p99": float(p99),
                })

                if not all(
                    np.isfinite(stats[name])
                    for name in ("min", "max", "mean", "median", "p1", "p99")
                ):
                    for name in ("min", "max", "mean", "median", "p1", "p99"):
                        if not np.isfinite(stats[name]):
                            stats[name] = None

                    finding(
                        "QA-020", "CRITICAL",
                        f"{key} produced non-finite statistics.",
                        "FAIL",
                        "Check input scaling and numeric range.",
                    )

                if minimum < 0:
                    finding(
                        "QA-010", "MEDIUM",
                        f"Negative values detected in {key}.",
                        "WARN",
                        "Check reflectance scaling.",
                    )

                if maximum - minimum < 1e-6:
                    finding(
                        "QA-010", "HIGH",
                        f"Constant values detected in {key}.",
                        "WARN",
                        "Confirm sensor data quality.",
                    )

        return results

    def run_vegetation_index_qa(
        self,
        t1_indices: Dict[str, Any],
        t2_indices: Dict[str, Any],
        valid_mask: np.ndarray,
    ) -> Dict[str, Any]:
        """Validate vegetation indices within the valid analysis mask."""

        res = {
            "status": "PASS",
            "T1": {},
            "T2": {},
        }

        def finding(code, severity, message, status, action):
            self.add_flag(
                code,
                severity,
                message,
                "Preprocessing",
                status,
                action,
            )

            if status == "FAIL":
                res["status"] = "FAIL"
            elif status == "WARN" and res["status"] != "FAIL":
                res["status"] = "WARN"
            elif status == "NOT_AVAILABLE" and res["status"] == "PASS":
                res["status"] = "NOT_AVAILABLE"

        if (
            not isinstance(valid_mask, np.ndarray)
            or valid_mask.ndim != 2
            or valid_mask.size == 0
        ):
            finding(
                "QA-002", "CRITICAL",
                "Invalid vegetation index analysis mask.",
                "FAIL",
                "Provide a non-empty two-dimensional mask.",
            )
            return res

        if valid_mask.dtype.kind != "b":
            if (
                valid_mask.dtype.kind not in "iuf"
                or not np.all(np.isin(valid_mask, [0, 1]))
            ):
                finding(
                    "QA-020", "CRITICAL",
                    "Analysis mask contains unsupported values.",
                    "FAIL",
                    "Use a boolean or binary 0/1 mask.",
                )
                return res

        mask = valid_mask.astype(bool)
        mask_count = int(np.count_nonzero(mask))

        observations = (
            ("T1", t1_indices),
        )
        if getattr(self, "qa_analysis_mode", None) != "single_date":
            observations += (("T2", t2_indices),)
        else:
            res.pop("T2", None)

        for (name, indices) in observations:
            if not isinstance(indices, dict):
                finding(
                    "QA-001", "CRITICAL",
                    f"{name} vegetation indices are missing.",
                    "FAIL",
                    "Compute the required vegetation indices.",
                )
                continue

            for idx in ("ndvi", "gndvi", "ndre"):
                arr = indices.get(idx)
                key = f"{name} {idx.upper()}"

                if arr is None:
                    finding(
                        "QA-001", "CRITICAL",
                        f"{key} is missing.",
                        "FAIL",
                        "Compute the required vegetation index.",
                    )
                    continue

                if (
                    not isinstance(arr, np.ndarray)
                    or arr.ndim != 2
                    or arr.shape != mask.shape
                    or arr.dtype.kind not in "iuf"
                ):
                    finding(
                        "QA-002", "CRITICAL",
                        f"{key} has invalid shape or data type.",
                        "FAIL",
                        "Align the index raster with the analysis mask.",
                    )
                    continue

                if mask_count == 0:
                    continue

                selected = arr[mask]
                values = selected[np.isfinite(selected)].astype(np.float64)

                invalid_count = int(selected.size - values.size)

                if values.size == 0:
                    finding(
                        "QA-003", "CRITICAL",
                        f"{key} contains no finite valid pixels.",
                        "FAIL",
                        "Check index calculation and NoData masking.",
                    )
                    continue

                if invalid_count:
                    finding(
                        "QA-003", "HIGH",
                        f"{key} contains {invalid_count} non-finite valid pixels.",
                        "WARN",
                        "Review vegetation index preprocessing.",
                    )

                obs_min = float(np.min(values))
                obs_max = float(np.max(values))
                p5, median, p95 = np.percentile(values, [5, 50, 95])

                stats = {
                    "min": obs_min,
                    "max": obs_max,
                    "mean": float(np.mean(values)),
                    "median": float(median),
                    "p5": float(p5),
                    "p95": float(p95),
                    "valid_pixels": int(values.size),
                    "invalid_pixels": invalid_count,
                }

                for stat in ("min", "max", "mean", "median", "p5", "p95"):
                    if not np.isfinite(stats[stat]):
                        stats[stat] = None
                        finding(
                            "QA-020", "CRITICAL",
                            f"{key} produced invalid statistics.",
                            "FAIL",
                            "Check index scaling and numeric range.",
                        )

                res[name][idx] = stats

                if obs_min < -1.05 or obs_max > 1.05:
                    finding(
                        "QA-020", "MEDIUM",
                        f"{key} range [{obs_min:.2f}, {obs_max:.2f}] exceeds expected bounds.",
                        "WARN",
                        "Verify the vegetation index formula.",
                    )

        if mask_count == 0:
            finding(
                "QA-003", "INFO",
                "No valid pixels available for vegetation index QA.",
                "NOT_AVAILABLE",
                "Provide a non-empty valid analysis region.",
            )

        return res

    # 2. MODEL QA
    def run_model_qa(
        self,
        model_path: str,
        logits: Optional[np.ndarray],
        probabilities: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Validate ONNX model metadata and prediction outputs."""

        res = {
            "status": "PASS",
            "calibration_status": "NOT_AVAILABLE",
            "confidence": "NOT_AVAILABLE",
        }

        def finding(code, severity, message, phase, status, action):
            self.add_flag(
                code, severity, message, phase, status, action
            )

            priority = {
                "PASS": 0,
                "NOT_AVAILABLE": 1,
                "WARN": 2,
                "FAIL": 3,
            }

            if priority[status] > priority[res["status"]]:
                res["status"] = status

        try:
            valid_path = (
                isinstance(model_path, (str, os.PathLike))
                and os.path.isfile(model_path)
            )
        except (OSError, ValueError, TypeError):
            valid_path = False

        if not valid_path:
            finding(
                "QA-001", "CRITICAL",
                f"ONNX model file is missing or invalid: {model_path}",
                "Model QA", "FAIL",
                "Deploy a valid ONNX model.",
            )
            return res

        try:
            session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )

            inputs = session.get_inputs()
            outputs = session.get_outputs()

            if not inputs or not outputs:
                raise ValueError("Model inputs or outputs are missing.")

            input_details = inputs[0]
            output_details = outputs[0]

            input_shape = list(input_details.shape)
            output_shape = list(output_details.shape)

            res["onnx"] = {
                "input_name": input_details.name,
                "input_shape": input_shape,
                "output_name": output_details.name,
                "output_shape": output_shape,
                "providers": session.get_providers(),
            }

            if len(input_shape) != 4:
                finding(
                    "QA-002", "CRITICAL",
                    f"Invalid model input shape: {input_shape}",
                    "Model QA", "FAIL",
                    "Use a four-dimensional NCHW input tensor.",
                )

            elif input_shape[1] != 5:
                finding(
                    "QA-002", "CRITICAL",
                    f"Expected five input channels, got {input_shape[1]}.",
                    "Model QA", "FAIL",
                    "Verify NDVI, GNDVI, NDRE, NIR and RedEdge channels.",
                )

        except Exception as exc:
            res["error"] = str(exc)

            finding(
                "QA-001", "CRITICAL",
                f"Failed to load or inspect ONNX model: {exc}",
                "Model QA", "FAIL",
                "Verify model format, integrity and runtime compatibility.",
            )
            return res

        def valid_prediction_array(array, name):
            if (
                not isinstance(array, np.ndarray)
                or array.ndim != 3
                or array.size == 0
                or array.shape[0] != 4
                or array.dtype.kind not in "iuf"
            ):
                finding(
                    "QA-020", "CRITICAL",
                    f"{name} must be a numeric array with shape (4, H, W).",
                    "Model Inference", "FAIL",
                    "Verify prediction dimensions and class count.",
                )
                return False

            if not np.all(np.isfinite(array)):
                finding(
                    "QA-020", "CRITICAL",
                    f"{name} contains NaN or infinity.",
                    "Model Inference", "FAIL",
                    "Review model inference and output preprocessing.",
                )
                return False

            return True

        logits_valid = False

        if logits is not None:
            logits_valid = valid_prediction_array(logits, "Logits")

            if logits_valid:
                res["logits"] = {
                    "shape": list(logits.shape),
                    "min": float(np.min(logits)),
                    "max": float(np.max(logits)),
                }

        if probabilities is None:
            finding(
                "QA-013", "INFO",
                "Probability outputs are unavailable.",
                "Model Inference", "NOT_AVAILABLE",
                "Provide probability outputs for confidence validation.",
            )
            return res

        if not valid_prediction_array(probabilities, "Probabilities"):
            return res

        if logits_valid and logits.shape != probabilities.shape:
            finding(
                "QA-002", "CRITICAL",
                "Logits and probability shapes do not match.",
                "Model Inference", "FAIL",
                "Verify softmax output dimensions.",
            )
            return res

        if np.any(probabilities < 0) or np.any(probabilities > 1):
            finding(
                "QA-020", "CRITICAL",
                "Probabilities contain values outside [0, 1].",
                "Model Inference", "FAIL",
                "Verify softmax calculation.",
            )
            return res

        sums = np.sum(
            probabilities,
            axis=0,
            dtype=np.float64,
        )

        if not np.allclose(sums, 1.0, atol=1e-4, rtol=0):
            finding(
                "QA-020", "HIGH",
                "Class probabilities do not sum to one.",
                "Model Inference", "FAIL",
                "Verify probability normalization.",
            )
            return res

        max_probs = np.max(probabilities, axis=0)

        p5, median, p95 = np.percentile(
            max_probs, [5, 50, 95]
        )

        res["confidence"] = {
            "mean_max_prob": float(np.mean(max_probs)),
            "median_max_prob": float(median),
            "p5_max_prob": float(p5),
            "p95_max_prob": float(p95),
        }

        res["probability_status"] = "PASS"

        return res

    def run_class_distribution_qa(
        self,
        t1_health: np.ndarray,
        t2_health: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Dict[str, Any]:
        """Validate vegetation classes and class distributions."""

        res = {
            "status": "PASS",
            "T1": {},
            "T2": {},
        }

        def finding(code, severity, message, status, action):
            self.add_flag(
                code,
                severity,
                message,
                "Model Inference",
                status,
                action,
            )

            priority = {
                "PASS": 0,
                "NOT_AVAILABLE": 1,
                "WARN": 2,
                "FAIL": 3,
            }

            if priority[status] > priority[res["status"]]:
                res["status"] = status

        if (
            not isinstance(valid_mask, np.ndarray)
            or valid_mask.ndim != 2
            or valid_mask.size == 0
        ):
            finding(
                "QA-002", "CRITICAL",
                "Invalid class distribution analysis mask.",
                "FAIL",
                "Provide a non-empty two-dimensional mask.",
            )
            return res

        if valid_mask.dtype.kind != "b":
            if (
                valid_mask.dtype.kind not in "iuf"
                or not np.all(np.isin(valid_mask, [0, 1]))
            ):
                finding(
                    "QA-020", "CRITICAL",
                    "Analysis mask contains invalid values.",
                    "FAIL",
                    "Use a boolean or binary 0/1 mask.",
                )
                return res

        mask = valid_mask.astype(bool)
        total_pixels = int(np.count_nonzero(mask))

        observations = (
            ("T1", t1_health),
        )
        if getattr(self, "qa_analysis_mode", None) != "single_date":
            observations += (("T2", t2_health),)
        else:
            res.pop("T2", None)

        for (name, arr) in observations:
            if (
                not isinstance(arr, np.ndarray)
                or arr.ndim != 2
                or arr.shape != mask.shape
                or arr.dtype.kind not in "iuf"
            ):
                finding(
                    "QA-002", "CRITICAL",
                    f"{name} health map has invalid shape or dtype.",
                    "FAIL",
                    "Align the health map with the analysis mask.",
                )
                continue

            if total_pixels == 0:
                continue

            valid_arr = arr[mask]

            if not np.all(np.isfinite(valid_arr)):
                finding(
                    "QA-020", "CRITICAL",
                    f"{name} health map contains non-finite valid pixels.",
                    "FAIL",
                    "Review model classification outputs.",
                )
                continue

            invalid_mask = ~np.isin(valid_arr, [0, 1, 2, 3])

            if np.any(invalid_mask):
                invalid_count = int(np.count_nonzero(invalid_mask))

                finding(
                    "QA-019", "CRITICAL",
                    f"{name} contains {invalid_count} invalid class pixels inside the valid region.",
                    "FAIL",
                    "Use classes 0-3 and exclude NoData 255 from valid pixels.",
                )
                continue

            classes, counts = np.unique(
                valid_arr,
                return_counts=True,
            )

            dist = {
                int(c): float(cnt / total_pixels * 100.0)
                for c, cnt in zip(classes, counts)
            }

            res[name] = dist

            for class_id, percentage in dist.items():
                if percentage > 99.0:
                    finding(
                        "QA-008", "MEDIUM",
                        f"Class {class_id} dominates {name} predictions with {percentage:.2f}% of pixels.",
                        "WARN",
                        "Confirm that class dominance reflects actual scene conditions.",
                    )

        if total_pixels == 0:
            finding(
                "QA-008", "INFO",
                "No valid pixels available for class distribution QA.",
                "NOT_AVAILABLE",
                "Provide a non-empty valid analysis region.",
            )

        return res

    # 3. SPATIAL QA
    def run_spatial_qa(
        self,
        raster_metadata_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate raster metadata and common-grid alignment."""

        from rasterio.crs import CRS

        res = {
            "status": "PASS",
            "reference": None,
            "rasters": {},
        }

        def finding(code, message, status, action):
            severity = "CRITICAL" if status == "FAIL" else "INFO"

            self.add_flag(
                code, severity, message,
                "Spatial Alignment", status, action,
            )

            if status == "FAIL":
                res["status"] = "FAIL"
            elif res["status"] == "PASS":
                res["status"] = status

        if not isinstance(raster_metadata_dict, dict):
            finding(
                "QA-002",
                "Spatial metadata must be a dictionary.",
                "FAIL",
                "Provide valid raster metadata.",
            )
            return res

        if not raster_metadata_dict:
            finding(
                "QA-005",
                "Spatial metadata is unavailable.",
                "NOT_AVAILABLE",
                "Provide raster metadata for alignment QA.",
            )
            return res

        validated = {}

        for name, meta in raster_metadata_dict.items():

            if meta is None:
                finding(
                    "QA-005",
                    f"Metadata unavailable for raster '{name}'.",
                    "NOT_AVAILABLE",
                    "Provide metadata for every required raster.",
                )
                continue

            try:
                crs_value = getattr(meta, "crs", None)

                if crs_value is None:
                    raise ValueError("Missing CRS")

                crs = CRS.from_user_input(crs_value)

                if not crs:
                    raise ValueError("Invalid CRS")

            except Exception as exc:
                finding(
                    "QA-004",
                    f"Invalid CRS on raster '{name}': {exc}",
                    "FAIL",
                    "Assign a valid coordinate reference system.",
                )
                continue

            try:
                width = meta.width
                height = meta.height

                if (
                    isinstance(width, (bool, np.bool_))
                    or isinstance(height, (bool, np.bool_))
                    or not isinstance(width, (int, np.integer))
                    or not isinstance(height, (int, np.integer))
                    or width <= 0
                    or height <= 0
                ):
                    raise ValueError("Invalid raster dimensions")

                transform = np.asarray(
                    tuple(meta.transform),
                    dtype=np.float64,
                )

                if transform.shape not in ((6,), (9,)):
                    raise ValueError("Invalid affine transform length")

                if not np.all(np.isfinite(transform)):
                    raise ValueError("Non-finite affine transform")

                if transform.size == 9 and not np.allclose(
                    transform[6:], [0, 0, 1],
                    atol=1e-12, rtol=0,
                ):
                    raise ValueError("Invalid affine transform bottom row")

                transform = transform[:6]

                a, b, c, d, e, f = transform

                determinant = a * e - b * d

                if (
                    not np.isfinite(determinant)
                    or determinant == 0
                ):
                    raise ValueError("Singular affine transform")

                resolution = np.asarray(
                    meta.resolution,
                    dtype=np.float64,
                )

                if (
                    resolution.shape != (2,)
                    or not np.all(np.isfinite(resolution))
                    or np.any(resolution <= 0)
                ):
                    raise ValueError("Invalid raster resolution")

                expected_resolution = np.array([
                    np.hypot(a, d),
                    np.hypot(b, e),
                ])

                if not np.allclose(
                    resolution,
                    expected_resolution,
                    atol=1e-12,
                    rtol=1e-7,
                ):
                    raise ValueError(
                        "Resolution inconsistent with affine transform"
                    )

                bounds_value = getattr(meta, "bounds", None)

                if bounds_value is None:
                    bounds_value = getattr(meta, "extent", None)

                if bounds_value is None:
                    raise ValueError("Missing raster bounds")

                bounds = np.asarray(
                    bounds_value,
                    dtype=np.float64,
                )

                if (
                    bounds.shape != (4,)
                    or not np.all(np.isfinite(bounds))
                    or bounds[0] >= bounds[2]
                    or bounds[1] >= bounds[3]
                ):
                    raise ValueError("Invalid raster bounds")

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
                    np.min(corners[:, 0]),
                    np.min(corners[:, 1]),
                    np.max(corners[:, 0]),
                    np.max(corners[:, 1]),
                ])

                tolerance = float(
                    np.min(resolution) * 1e-6
                )

                if not np.allclose(
                    bounds,
                    expected_bounds,
                    atol=tolerance,
                    rtol=0,
                ):
                    raise ValueError(
                        "Bounds inconsistent with raster transform"
                    )

                validated[name] = {
                    "crs": crs,
                    "transform": transform,
                    "resolution": resolution,
                    "bounds": bounds,
                    "width": int(width),
                    "height": int(height),
                }

                res["rasters"][name] = {
                    "crs": str(crs),
                    "transform": transform.tolist(),
                    "resolution": resolution.tolist(),
                    "bounds": bounds.tolist(),
                    "width": int(width),
                    "height": int(height),
                }

            except Exception as exc:
                finding(
                    "QA-005",
                    f"Invalid spatial metadata for '{name}': {exc}",
                    "FAIL",
                    "Repair raster dimensions, transform, resolution or bounds.",
                )

        if not validated:
            if res["status"] != "FAIL":
                res["status"] = "NOT_AVAILABLE"

            return res

        ref_key = next(iter(validated))
        ref = validated[ref_key]

        res["reference"] = ref_key

        for name, meta in validated.items():

            if name == ref_key:
                continue

            if meta["crs"] != ref["crs"]:
                finding(
                    "QA-004",
                    f"CRS mismatch on '{name}' vs '{ref_key}'.",
                    "FAIL",
                    "Reproject to the common reference grid.",
                )
                continue

            if (
                meta["width"] != ref["width"]
                or meta["height"] != ref["height"]
            ):
                finding(
                    "QA-002",
                    f"Raster dimensions mismatch on '{name}'.",
                    "FAIL",
                    "Resample to the reference grid dimensions.",
                )

            tolerance = float(
                np.min(ref["resolution"]) * 1e-6
            )

            if not np.allclose(
                meta["resolution"],
                ref["resolution"],
                atol=1e-12,
                rtol=1e-7,
            ):
                finding(
                    "QA-005",
                    f"Resolution mismatch on '{name}'.",
                    "FAIL",
                    "Align raster pixel resolutions.",
                )

            linear_indices = [0, 1, 3, 4]

            if not np.allclose(
                meta["transform"][linear_indices],
                ref["transform"][linear_indices],
                atol=1e-12,
                rtol=1e-7,
            ):
                finding(
                    "QA-005",
                    f"Scale or rotation mismatch on '{name}'.",
                    "FAIL",
                    "Align raster affine transformations.",
                )

            if not np.allclose(
                meta["transform"][[2, 5]],
                ref["transform"][[2, 5]],
                atol=tolerance,
                rtol=0,
            ):
                finding(
                    "QA-018",
                    f"Spatial pixel offset detected on '{name}'.",
                    "FAIL",
                    "Align raster origins to the reference grid.",
                )

            if not np.allclose(
                meta["bounds"],
                ref["bounds"],
                atol=tolerance,
                rtol=0,
            ):
                finding(
                    "QA-005",
                    f"Spatial bounds mismatch on '{name}'.",
                    "FAIL",
                    "Align raster geographical extents.",
                )

        if len(validated) < 2 and res["status"] == "PASS":
            finding(
                "QA-005",
                "Only one valid raster is available for alignment comparison.",
                "NOT_AVAILABLE",
                "Provide at least two rasters for cross-raster alignment QA.",
            )

        return res

    def run_nodata_propagation_qa(
        self,
        t1_arr: np.ndarray,
        t2_arr: np.ndarray,
        outputs: Dict[str, np.ndarray],
        valid_overlap: np.ndarray,
    ) -> Dict[str, Any]:
        """Validate NoData propagation and valid analysis pixels."""

        res = {
            "status": "PASS",
            "valid_pixels": 0,
            "nodata_pixels": 0,
            "outputs": {},
        }

        def finding(code, message, status, action):
            severity = "CRITICAL" if status == "FAIL" else "INFO"

            self.add_flag(
                code, severity, message,
                "NoData QA", status, action,
            )

            if status == "FAIL":
                res["status"] = "FAIL"
            elif res["status"] == "PASS":
                res["status"] = status

        if (
            not isinstance(valid_overlap, np.ndarray)
            or valid_overlap.ndim != 2
            or valid_overlap.size == 0
        ):
            finding(
                "QA-002",
                "Invalid overlap mask dimensions.",
                "FAIL",
                "Provide a non-empty two-dimensional overlap mask.",
            )
            return res

        if valid_overlap.dtype.kind != "b":
            if (
                valid_overlap.dtype.kind not in "iuf"
                or not np.all(np.isin(valid_overlap, [0, 1]))
            ):
                finding(
                    "QA-003",
                    "Overlap mask contains invalid values.",
                    "FAIL",
                    "Use a boolean or binary 0/1 mask.",
                )
                return res

        valid_mask = valid_overlap.astype(bool)
        nodata_mask = ~valid_mask

        res["valid_pixels"] = int(np.count_nonzero(valid_mask))
        res["nodata_pixels"] = int(np.count_nonzero(nodata_mask))

        def valid_array(arr, name):
            if (
                not isinstance(arr, np.ndarray)
                or arr.ndim != 2
                or arr.shape != valid_mask.shape
                or arr.dtype.kind not in "iuf"
            ):
                finding(
                    "QA-002",
                    f"Invalid array dimensions or dtype for '{name}'.",
                    "FAIL",
                    "Align raster dimensions and numeric data types.",
                )
                return False

            return True

        for date_name, arr in (
            ("T1", t1_arr),
            ("T2", t2_arr),
        ):
            if not valid_array(arr, date_name):
                continue

            valid_values = arr[valid_mask]

            invalid_count = int(np.count_nonzero(
                ~np.isin(valid_values, [0, 1, 2, 3])
            ))

            if invalid_count:
                finding(
                    "QA-019",
                    f"{date_name} contains {invalid_count} invalid class or NoData pixels inside valid overlap.",
                    "FAIL",
                    "Keep valid classes 0-3 separate from NoData 255.",
                )

        if not isinstance(outputs, dict):
            finding(
                "QA-002",
                "Output rasters must be supplied as a dictionary.",
                "FAIL",
                "Provide named output raster arrays.",
            )
            return res

        if not outputs:
            finding(
                "QA-003",
                "No output rasters available for NoData validation.",
                "NOT_AVAILABLE",
                "Provide output rasters for validation.",
            )

        for name, arr in outputs.items():

            if arr is None:
                finding(
                    "QA-003",
                    f"Output raster '{name}' is unavailable.",
                    "NOT_AVAILABLE",
                    "Provide the required output raster.",
                )
                continue

            if not valid_array(arr, name):
                continue

            outside = arr[nodata_mask]
            inside = arr[valid_mask]

            if arr.dtype.kind in "iu":
                invalid_outside = outside != 255
                invalid_inside = inside == 255

                nodata_value = 255

            else:
                invalid_outside = ~np.isnan(outside)
                invalid_inside = ~np.isfinite(inside)

                nodata_value = "NaN"

            outside_count = int(
                np.count_nonzero(invalid_outside)
            )

            inside_count = int(
                np.count_nonzero(invalid_inside)
            )

            res["outputs"][name] = {
                "nodata_value": nodata_value,
                "invalid_outside_pixels": outside_count,
                "invalid_inside_pixels": inside_count,
                "status": (
                    "FAIL"
                    if outside_count or inside_count
                    else "PASS"
                ),
            }

            if outside_count:
                finding(
                    "QA-003",
                    f"Output '{name}' contains {outside_count} non-NoData pixels outside valid overlap.",
                    "FAIL",
                    "Propagate the correct NoData sentinel outside the analysis region.",
                )

            if inside_count:
                finding(
                    "QA-019",
                    f"Output '{name}' contains {inside_count} NoData or non-finite pixels inside valid overlap.",
                    "FAIL",
                    "Review output generation and valid analysis masks.",
                )

        if res["valid_pixels"] == 0:
            finding(
                "QA-003",
                "No valid overlap pixels are available.",
                "NOT_AVAILABLE",
                "Provide a valid overlapping analysis region.",
            )

        return res

    # 4. TEMPORAL QA
    def run_temporal_qa(
        self,
        t1_date: str,
        t2_date: str,
        meta_t1: Any,
        meta_t2: Any,
        valid_overlap_mask: np.ndarray,
    ) -> Dict[str, Any]:
        """Validate acquisition chronology, seasonality and overlap."""

        from datetime import datetime
        import re

        res = {
            "status": "PASS",
            "date_difference_days": None,
            "seasonal_month_difference": None,
            "phenology_warning": "NOT_AVAILABLE",
            "valid_overlap_pixels": None,
            "total_grid_pixels": None,
            "grid_overlap_percentage": None,
        }

        priority = {
            "PASS": 0,
            "NOT_AVAILABLE": 1,
            "WARN": 2,
            "FAIL": 3,
        }

        def finding(
            code, severity, message,
            phase, status, action,
        ):
            self.add_flag(
                code, severity, message,
                phase, status, action,
            )

            if priority[status] > priority[res["status"]]:
                res["status"] = status

        parsed_dates = {}

        for name, value in (
            ("T1", t1_date),
            ("T2", t2_date),
        ):
            if value is None or (
                isinstance(value, str)
                and not value.strip()
            ):
                finding(
                    "QA-006", "INFO",
                    f"{name} acquisition date is unavailable.",
                    "Temporal Alignment", "NOT_AVAILABLE",
                    "Provide the acquisition date in YYYY-MM-DD format.",
                )
                continue

            if (
                not isinstance(value, str)
                or re.fullmatch(
                    r"[0-9]{4}-[0-9]{2}-[0-9]{2}",
                    value,
                ) is None
            ):
                finding(
                    "QA-006", "CRITICAL",
                    f"{name} acquisition date has an invalid format.",
                    "Temporal Alignment", "FAIL",
                    "Use the YYYY-MM-DD date format.",
                )
                continue

            try:
                parsed_dates[name] = datetime.strptime(
                    value,
                    "%Y-%m-%d",
                ).date()

            except (ValueError, TypeError) as exc:
                finding(
                    "QA-006", "CRITICAL",
                    f"Invalid {name} acquisition date: {exc}",
                    "Temporal Alignment", "FAIL",
                    "Provide a valid calendar date.",
                )

        if len(parsed_dates) == 2:
            d1 = parsed_dates["T1"]
            d2 = parsed_dates["T2"]

            diff_days = (d2 - d1).days

            res["date_difference_days"] = diff_days

            if diff_days <= 0:
                finding(
                    "QA-006", "CRITICAL",
                    f"T2 ({t2_date}) must be after T1 ({t1_date}).",
                    "Temporal Alignment", "FAIL",
                    "Correct acquisition chronology.",
                )

            else:
                raw_month_diff = abs(
                    d1.month - d2.month
                )

                month_diff = min(
                    raw_month_diff,
                    12 - raw_month_diff,
                )

                res["seasonal_month_difference"] = month_diff

                if month_diff >= 3:
                    res["phenology_warning"] = "YES"

                    finding(
                        "QA-007", "MEDIUM",
                        f"Acquisition months differ by {month_diff} months on the annual calendar.",
                        "Temporal Alignment", "WARN",
                        "Review seasonal and vegetation phenology differences.",
                    )

                else:
                    res["phenology_warning"] = "NO"

        if valid_overlap_mask is None:
            finding(
                "QA-002", "INFO",
                "Temporal overlap mask is unavailable.",
                "Spatial Alignment", "NOT_AVAILABLE",
                "Provide the aligned overlap mask.",
            )
            return res

        if (
            not isinstance(valid_overlap_mask, np.ndarray)
            or valid_overlap_mask.ndim != 2
            or valid_overlap_mask.size == 0
        ):
            finding(
                "QA-002", "CRITICAL",
                "Invalid temporal overlap mask dimensions.",
                "Spatial Alignment", "FAIL",
                "Provide a non-empty two-dimensional overlap mask.",
            )
            return res

        if valid_overlap_mask.dtype.kind != "b":
            if (
                valid_overlap_mask.dtype.kind not in "iuf"
                or not np.all(
                    np.isin(valid_overlap_mask, [0, 1])
                )
            ):
                finding(
                    "QA-002", "CRITICAL",
                    "Temporal overlap mask contains invalid values.",
                    "Spatial Alignment", "FAIL",
                    "Use a boolean or binary 0/1 mask.",
                )
                return res

        total_pixels = int(valid_overlap_mask.size)

        overlap_pixels = int(
            np.count_nonzero(valid_overlap_mask)
        )

        res["total_grid_pixels"] = total_pixels
        res["valid_overlap_pixels"] = overlap_pixels

        res["grid_overlap_percentage"] = (
            overlap_pixels / total_pixels * 100.0
        )

        if overlap_pixels == 0:
            finding(
                "QA-002", "CRITICAL",
                "Valid overlap area is zero pixels.",
                "Spatial Alignment", "FAIL",
                "Verify raster alignment and common valid coverage.",
            )

        return res

    # 5. TRANSITION QA
    def run_transition_qa(
        self,
        transition_map: np.ndarray,
        t1_health: np.ndarray,
        t2_health: np.ndarray,
        valid_overlap: np.ndarray,
    ) -> Dict[str, Any]:
        """Validate transition encoding and pixel reconciliation."""

        res = {
            "status": "PASS",
            "valid_overlap_pixels": 0,
            "reported_transition_pixels": 0,
            "formula_mismatch_pixels": 0,
            "invalid_outside_pixels": 0,
            "transition_distribution": {},
        }

        priority = {
            "PASS": 0,
            "NOT_AVAILABLE": 1,
            "WARN": 2,
            "FAIL": 3,
        }

        def finding(code, severity, message, status, action):
            self.add_flag(
                code, severity, message,
                "Transition Mapping", status, action,
            )

            if priority[status] > priority[res["status"]]:
                res["status"] = status

        if (
            not isinstance(valid_overlap, np.ndarray)
            or valid_overlap.ndim != 2
            or valid_overlap.size == 0
        ):
            finding(
                "QA-002", "CRITICAL",
                "Invalid transition overlap mask dimensions.",
                "FAIL",
                "Provide a non-empty two-dimensional mask.",
            )
            return res

        if valid_overlap.dtype.kind != "b":
            if (
                valid_overlap.dtype.kind not in "iuf"
                or not np.all(np.isin(valid_overlap, [0, 1]))
            ):
                finding(
                    "QA-020", "CRITICAL",
                    "Transition overlap mask contains invalid values.",
                    "FAIL",
                    "Use a boolean or binary 0/1 mask.",
                )
                return res

        mask = valid_overlap.astype(bool)

        for name, arr in (
            ("transition_map", transition_map),
            ("T1", t1_health),
            ("T2", t2_health),
        ):
            if (
                not isinstance(arr, np.ndarray)
                or arr.ndim != 2
                or arr.shape != mask.shape
                or arr.dtype.kind not in "iuf"
            ):
                finding(
                    "QA-002", "CRITICAL",
                    f"Invalid raster dimensions or dtype for '{name}'.",
                    "FAIL",
                    "Align transition and health raster dimensions.",
                )

        if res["status"] == "FAIL":
            return res

        total_pixels = int(np.count_nonzero(mask))
        res["valid_overlap_pixels"] = total_pixels

        outside_count = int(np.count_nonzero(
            transition_map[~mask] != 255
        ))

        res["invalid_outside_pixels"] = outside_count

        if outside_count:
            finding(
                "QA-003", "CRITICAL",
                f"{outside_count} transition pixels outside overlap are not NoData.",
                "FAIL",
                "Set transition pixels outside valid overlap to 255.",
            )

        if total_pixels == 0:
            finding(
                "QA-015", "INFO",
                "No valid overlap pixels available for transition QA.",
                "NOT_AVAILABLE",
                "Provide a valid overlapping analysis region.",
            )
            return res

        t1_values = t1_health[mask]
        t2_values = t2_health[mask]
        actual = transition_map[mask]

        health_valid = True

        for name, values in (
            ("T1", t1_values),
            ("T2", t2_values),
        ):
            invalid_count = int(np.count_nonzero(
                ~np.isin(values, [0, 1, 2, 3])
            ))

            if invalid_count:
                health_valid = False

                finding(
                    "QA-020", "CRITICAL",
                    f"{name} contains {invalid_count} invalid health classes inside overlap.",
                    "FAIL",
                    "Restrict valid health classes to integers 0-3.",
                )

        valid_ids = np.isin(actual, np.arange(16))

        reported_pixels = int(
            np.count_nonzero(valid_ids)
        )

        res["reported_transition_pixels"] = reported_pixels

        invalid_ids = total_pixels - reported_pixels

        if invalid_ids:
            finding(
                "QA-020", "CRITICAL",
                f"{invalid_ids} invalid transition IDs detected.",
                "FAIL",
                "Restrict valid transition IDs to integers 0-15.",
            )

        if reported_pixels != total_pixels:
            finding(
                "QA-015", "CRITICAL",
                "Transition pixel reconciliation failed.",
                "FAIL",
                "Ensure transition counts equal valid overlap pixels.",
            )

        if health_valid and invalid_ids == 0:
            expected = (
                t1_values.astype(np.int32) * 4
                + t2_values.astype(np.int32)
            )

            mismatch_count = int(np.count_nonzero(
                expected != actual
            ))

            res["formula_mismatch_pixels"] = mismatch_count

            if mismatch_count:
                finding(
                    "QA-020", "CRITICAL",
                    f"Transition formula mismatch at {mismatch_count} pixels.",
                    "FAIL",
                    "Use transition_id = T1 * 4 + T2.",
                )

        if res["status"] == "FAIL":
            return res

        classes, counts = np.unique(
            actual,
            return_counts=True,
        )

        for class_id, count in zip(classes, counts):
            percentage = float(
                count / total_pixels * 100.0
            )

            res["transition_distribution"][int(class_id)] = {
                "pixel_count": int(count),
                "percentage": percentage,
            }

            if percentage > 99.0:
                finding(
                    "QA-009", "MEDIUM",
                    f"Transition ID {int(class_id)} dominates {percentage:.2f}% of valid pixels.",
                    "WARN",
                    "Review classification and change detection sensitivity.",
                )

        return res

    # 6. CHANGE PATCH QA
    def run_change_patch_qa(
        self,
        degradation: np.ndarray,
        recovery: np.ndarray,
        canopy_loss: np.ndarray,
        residual_stress: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Dict[str, Any]:
        """Validate binary change maps and connected patch statistics."""

        res = {
            "status": "PASS",
            "valid_pixels": 0,
            "connectivity": 4,
        }

        priority = {
            "PASS": 0,
            "NOT_AVAILABLE": 1,
            "WARN": 2,
            "FAIL": 3,
        }

        def finding(code, severity, message, status, action):
            self.add_flag(
                code, severity, message,
                "Transition Mapping", status, action,
            )

            if priority[status] > priority[res["status"]]:
                res["status"] = status

        if (
            not isinstance(valid_mask, np.ndarray)
            or valid_mask.ndim != 2
            or valid_mask.size == 0
        ):
            finding(
                "QA-002", "CRITICAL",
                "Invalid change patch mask dimensions.",
                "FAIL",
                "Provide a non-empty two-dimensional mask.",
            )
            return res

        if valid_mask.dtype.kind != "b":
            if (
                valid_mask.dtype.kind not in "iuf"
                or not np.all(np.isin(valid_mask, [0, 1]))
            ):
                finding(
                    "QA-020", "CRITICAL",
                    "Change patch mask contains invalid values.",
                    "FAIL",
                    "Use a boolean or binary 0/1 mask.",
                )
                return res

        mask = valid_mask.astype(bool)

        valid_pixels = int(np.count_nonzero(mask))

        res["valid_pixels"] = valid_pixels

        structure = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ], dtype=np.uint8)

        for name, arr in (
            ("degradation", degradation),
            ("recovery", recovery),
            ("canopy_loss", canopy_loss),
            ("residual_stress", residual_stress),
        ):
            if arr is None:
                res[name] = {
                    "status": "NOT_AVAILABLE",
                }

                finding(
                    "QA-015", "INFO",
                    f"Change map '{name}' is unavailable.",
                    "NOT_AVAILABLE",
                    "Provide the required change map.",
                )
                continue

            if (
                not isinstance(arr, np.ndarray)
                or arr.ndim != 2
                or arr.shape != mask.shape
                or arr.dtype.kind not in "biuf"
            ):
                res[name] = {"status": "FAIL"}

                finding(
                    "QA-002", "CRITICAL",
                    f"Invalid dimensions or dtype for '{name}'.",
                    "FAIL",
                    "Align change maps with the valid analysis mask.",
                )
                continue

            inside = arr[mask]
            outside = arr[~mask]

            invalid_inside = int(np.count_nonzero(
                ~np.isin(inside, [0, 1])
            ))

            invalid_outside = int(np.count_nonzero(
                ~np.isin(outside, [0, 255])
            ))

            if invalid_inside or invalid_outside:
                res[name] = {
                    "status": "FAIL",
                    "invalid_inside_pixels": invalid_inside,
                    "invalid_outside_pixels": invalid_outside,
                }

                finding(
                    "QA-020", "CRITICAL",
                    f"Invalid binary values or outside-mask changes in '{name}'.",
                    "FAIL",
                    "Use binary values inside the mask and no changes outside.",
                )
                continue

            if valid_pixels == 0:
                res[name] = {
                    "status": "NOT_AVAILABLE",
                }
                continue

            change_pixels = (arr == 1) & mask

            total_change = int(
                np.count_nonzero(change_pixels)
            )

            labeled_arr, num_features = label(
                change_pixels,
                structure=structure,
            )

            num_features = int(num_features)

            if num_features == 0:
                res[name] = {
                    "status": "PASS",
                    "patch_count": 0,
                    "change_pixels": 0,
                    "min_size_px": 0,
                    "median_size_px": 0.0,
                    "max_size_px": 0,
                    "mean_size_px": 0.0,
                    "change_percentage": 0.0,
                }
                continue

            unique_labels, counts = np.unique(
                labeled_arr,
                return_counts=True,
            )

            sizes = counts[unique_labels != 0]

            reconciled = (
                sizes.size == num_features
                and int(np.sum(sizes)) == total_change
            )

            if not reconciled:
                res[name] = {
                    "status": "FAIL",
                }

                finding(
                    "QA-015", "CRITICAL",
                    f"Patch pixel reconciliation failed for '{name}'.",
                    "FAIL",
                    "Verify connected component labeling.",
                )
                continue

            res[name] = {
                "status": "PASS",
                "patch_count": num_features,
                "change_pixels": total_change,
                "min_size_px": int(np.min(sizes)),
                "median_size_px": float(np.median(sizes)),
                "max_size_px": int(np.max(sizes)),
                "mean_size_px": float(np.mean(sizes)),
                "change_percentage": float(
                    total_change / valid_pixels * 100.0
                ),
            }

        if valid_pixels == 0:
            finding(
                "QA-015", "INFO",
                "No valid pixels available for change patch QA.",
                "NOT_AVAILABLE",
                "Provide a valid analysis region.",
            )

        return res

    # 7. CVA QA
    def run_cva_qa(
        self,
        cva_mag: np.ndarray,
        d_ndvi: np.ndarray,
        d_gndvi: np.ndarray,
        d_ndre: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Dict[str, Any]:
        """Validate CVA magnitude, numerical consistency and outliers."""

        res = {
            "status": "PASS",
            "valid_pixels": 0,
            "invalid_pixels": 0,
            "negative_magnitude_pixels": 0,
            "formula_mismatch_pixels": None,
            "max_reconciliation_error": None,
            "max_magnitude_error": None,
            "outliers": None,
        }

        priority = {
            "PASS": 0,
            "NOT_AVAILABLE": 1,
            "WARN": 2,
            "FAIL": 3,
        }

        def finding(
            code, severity, message,
            phase, status, action,
        ):
            self.add_flag(
                code, severity, message,
                phase, status, action,
            )

            if priority[status] > priority[res["status"]]:
                res["status"] = status

        if (
            not isinstance(valid_mask, np.ndarray)
            or valid_mask.ndim not in (1, 2)
            or valid_mask.size == 0
        ):
            finding(
                "QA-002", "CRITICAL",
                "Invalid CVA analysis mask dimensions.",
                "CVA Math Invariant", "FAIL",
                "Provide a non-empty one-dimensional or two-dimensional mask.",
            )
            return res

        if valid_mask.dtype.kind != "b":
            if (
                valid_mask.dtype.kind not in "iuf"
                or not np.all(np.isin(valid_mask, [0, 1]))
            ):
                finding(
                    "QA-020", "CRITICAL",
                    "CVA analysis mask contains invalid values.",
                    "CVA Math Invariant", "FAIL",
                    "Use a boolean or binary 0/1 mask.",
                )
                return res

        mask = valid_mask.astype(bool, copy=False)

        arrays = {
            "CVA magnitude": cva_mag,
            "delta NDVI": d_ndvi,
            "delta GNDVI": d_gndvi,
            "delta NDRE": d_ndre,
        }

        for name, arr in arrays.items():
            if (
                not isinstance(arr, np.ndarray)
                or arr.ndim not in (1, 2)
                or arr.shape != mask.shape
                or arr.dtype.kind not in "iuf"
            ):
                finding(
                    "QA-002", "CRITICAL",
                    f"Invalid dimensions or dtype for {name}.",
                    "CVA Math Invariant", "FAIL",
                    "Provide aligned numeric one-dimensional or two-dimensional arrays.",
                )

        if res["status"] == "FAIL":
            return res

        total_pixels = int(np.count_nonzero(mask))
        res["valid_pixels"] = total_pixels

        if total_pixels == 0:
            finding(
                "QA-020", "INFO",
                "No valid pixels available for CVA validation.",
                "CVA Math Invariant", "NOT_AVAILABLE",
                "Provide a valid overlapping analysis region.",
            )
            return res

        with np.errstate(over="ignore", invalid="ignore"):
            magnitude = cva_mag[mask].astype(np.float64)
            ndvi = d_ndvi[mask].astype(np.float64)
            gndvi = d_gndvi[mask].astype(np.float64)
            ndre = d_ndre[mask].astype(np.float64)

        finite = (
            np.isfinite(magnitude)
            & np.isfinite(ndvi)
            & np.isfinite(gndvi)
            & np.isfinite(ndre)
        )

        invalid_count = int(np.count_nonzero(~finite))
        res["invalid_pixels"] = invalid_count

        if invalid_count:
            finding(
                "QA-020", "CRITICAL",
                f"{invalid_count} CVA pixels contain non-finite values.",
                "CVA Math Invariant", "FAIL",
                "Correct invalid magnitude or delta values inside the mask.",
            )

        negative_count = int(np.count_nonzero(
            np.isfinite(magnitude) & (magnitude < 0)
        ))

        res["negative_magnitude_pixels"] = negative_count

        if negative_count:
            finding(
                "QA-020", "CRITICAL",
                f"{negative_count} negative CVA magnitudes detected.",
                "CVA Math Invariant", "FAIL",
                "Ensure CVA magnitude is non-negative.",
            )

        if res["status"] == "FAIL":
            return res

        # Stable Euclidean magnitude calculation.
        with np.errstate(over="ignore", invalid="ignore"):
            expected = np.hypot(
                np.hypot(ndvi, gndvi),
                ndre,
            )

        if not np.all(np.isfinite(expected)):
            finding(
                "QA-020", "CRITICAL",
                "CVA expected magnitude exceeds numeric limits.",
                "CVA Math Invariant", "FAIL",
                "Review delta scaling and numeric precision.",
            )
            return res

        magnitude_error = np.abs(magnitude - expected)

        res["max_magnitude_error"] = float(
            np.max(magnitude_error)
        )

        # Preserve the original quadratic tolerance of 1e-4.
        # Factorization avoids directly squaring both magnitudes.
        with np.errstate(over="ignore", invalid="ignore"):
            quadratic_error = (
                magnitude_error * (magnitude + expected)
            )

        # Exactly matching magnitudes have zero reconciliation error.
        quadratic_error[magnitude_error == 0] = 0.0

        finite_error = np.isfinite(quadratic_error)

        mismatch = (
            ~finite_error
            | (quadratic_error > 1e-4)
        )

        mismatch_count = int(np.count_nonzero(mismatch))

        res["formula_mismatch_pixels"] = mismatch_count

        if np.all(finite_error):
            res["max_reconciliation_error"] = float(
                np.max(quadratic_error)
            )

        if mismatch_count:
            finding(
                "QA-020", "CRITICAL",
                f"CVA mathematical reconciliation failed at {mismatch_count} pixels.",
                "CVA Math Invariant", "FAIL",
                "Verify CVA magnitude equals the Euclidean norm of the three deltas.",
            )

        # Scale statistics to reduce overflow risk.
        maximum = float(np.max(magnitude))
        minimum = float(np.min(magnitude))

        if maximum == 0.0:
            mean = 0.0
            std = 0.0
        else:
            normalized = magnitude / maximum

            mean = float(
                np.mean(normalized) * maximum
            )

            std = float(
                np.std(normalized) * maximum
            )

        statistics = {
            "min": minimum,
            "max": maximum,
            "mean": mean,
            "std": std,
            "p1": float(np.percentile(magnitude, 1)),
            "p99": float(np.percentile(magnitude, 99)),
        }

        if not all(
            np.isfinite(value)
            for value in statistics.values()
        ):
            res["outliers"] = {
                key: value if np.isfinite(value) else None
                for key, value in statistics.items()
            }

            finding(
                "QA-020", "CRITICAL",
                "CVA statistics contain non-finite values.",
                "CVA Math Invariant", "FAIL",
                "Review numeric precision and raster scaling.",
            )

        else:
            res["outliers"] = statistics

        if maximum > 10.0:
            finding(
                "QA-010", "MEDIUM",
                f"Extreme CVA magnitude detected: {maximum:.2f}.",
                "CVA Engine", "WARN",
                "Verify raw band scaling and vegetation index inputs.",
            )

        return res

    # 8. WETLAND MASK QA
    def run_wetland_qa(
        self,
        wetland_mask: Optional[np.ndarray],
        valid_analysis_mask: np.ndarray,
        config: Dict[str, Any],
        wetland_result: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Validate wetland membership, source evidence and depletion safeguards."""

        res = {
            "status": "PASS",
            "source_status": "WETLAND_BOUNDARY_UNAVAILABLE",
            "source_type": "UNAVAILABLE",
            "boundary_verified": False,
            "physical_depletion_supported": False,
            "wetland_pixels": None,
            "valid_analysis_pixels": None,
            "invalid_membership_pixels": None,
            "invalid_analysis_pixels": None,
        }

        priority = {
            "PASS": 0,
            "NOT_AVAILABLE": 1,
            "WARN": 2,
            "FAIL": 3,
        }

        def finding(code, severity, message, status, action):
            self.add_flag(
                code, severity, message,
                "Wetland Analysis", status, action,
            )

            if priority[status] > priority[res["status"]]:
                res["status"] = status

        def config_section(parent, key):
            value = parent.get(key, {})

            if not isinstance(value, dict):
                finding(
                    "QA-011", "CRITICAL",
                    f"Invalid configuration section: {key}.",
                    "FAIL",
                    "Provide a valid configuration dictionary.",
                )
                return {}

            return value

        if not isinstance(config, dict):
            finding(
                "QA-011", "CRITICAL",
                "Wetland configuration is invalid.",
                "FAIL",
                "Provide a configuration dictionary.",
            )
            return res

        wetland_cfg = config_section(config, "wetland")
        boundary_cfg = config_section(wetland_cfg, "boundary")
        official_cfg = config_section(wetland_cfg, "official_dataset")
        claims_cfg = config_section(wetland_cfg, "depletion_claims")
        fusion_cfg = config_section(config, "fusion")

        claims_enabled = claims_cfg.get("enabled", False)
        independent_evidence = fusion_cfg.get(
            "independent_hydrological_source_accepted", False
        )

        if not isinstance(claims_enabled, bool):
            finding(
                "QA-011", "CRITICAL",
                "Depletion claims setting must be boolean.",
                "FAIL",
                "Use true or false for depletion claims.",
            )
            claims_enabled = False

        if not isinstance(independent_evidence, bool):
            finding(
                "QA-011", "CRITICAL",
                "Independent evidence setting must be boolean.",
                "FAIL",
                "Use a valid boolean evidence setting.",
            )
            independent_evidence = False

        authoritative_sources = {
            "PROJECT_BOUNDARY",
            "OFFICIAL_DATASET",
        }

        inferred_sources = {
            "JRC_WATER_CONTEXT",
            "OPTICAL_WATER_INFERENCE",
            "SAR_INFERENCE",
        }

        source_metadata = {}
        physical_area_available = False
        result_depletion_enabled = False

        # Prefer the source actually selected by the producer.
        if wetland_result is not None:
            source_type = getattr(
                wetland_result, "source_type", None
            )

            source_status = getattr(
                wetland_result, "status", None
            )

            source_metadata = getattr(
                wetland_result, "source_metadata", {}
            )

            metadata = getattr(
                wetland_result, "metadata", {}
            )

            if not isinstance(source_metadata, dict):
                finding(
                    "QA-011", "CRITICAL",
                    "Invalid wetland source metadata.",
                    "FAIL",
                    "Provide valid source provenance.",
                )
                source_metadata = {}

            if not isinstance(metadata, dict):
                finding(
                    "QA-011", "CRITICAL",
                    "Invalid wetland result metadata.",
                    "FAIL",
                    "Provide valid wetland result metadata.",
                )
                metadata = {}

            physical_area_available = (
                metadata.get("physical_area_available") is True
            )

            result_depletion_enabled = getattr(
                wetland_result,
                "physical_wetland_depletion_enabled",
                False,
            )

            if not isinstance(result_depletion_enabled, bool):
                finding(
                    "QA-011", "CRITICAL",
                    "Invalid physical depletion result setting.",
                    "FAIL",
                    "Use a boolean depletion setting.",
                )
                result_depletion_enabled = False

            if not isinstance(source_type, str):
                source_type = "INVALID"

            if source_type in authoritative_sources:
                expected_status = "WETLAND_BOUNDARY_AVAILABLE"
            elif source_type in inferred_sources:
                expected_status = "WETLAND_EXTENT_INFERRED"
            elif source_type == "UNAVAILABLE":
                expected_status = "WETLAND_BOUNDARY_UNAVAILABLE"
            else:
                expected_status = None

            if (
                expected_status is None
                or source_status != expected_status
            ):
                finding(
                    "QA-011", "CRITICAL",
                    "Wetland source and status are inconsistent.",
                    "FAIL",
                    "Correct wetland source provenance and status.",
                )
            else:
                res["source_type"] = source_type
                res["source_status"] = source_status

        else:
            finding(
                "QA-011", "INFO",
                "Actual wetland producer result was not provided.",
                "NOT_AVAILABLE",
                "Pass WetlandMaskResult to verify the selected source.",
            )

        # Inspect configured or selected boundary files.
        boundary_path = None
        official_source = False

        if res["source_type"] in authoritative_sources:
            boundary_path = source_metadata.get("source_path")
            official_source = (
                res["source_type"] == "OFFICIAL_DATASET"
            )
        elif wetland_result is None:
            boundary_path = boundary_cfg.get("path")

            if boundary_path is None:
                boundary_path = official_cfg.get("path")
                official_source = boundary_path is not None

        if (
            boundary_path is not None
            or res["source_type"] in authoritative_sources
        ):
            try:
                import json
                from rasterio.crs import CRS
                from rasterio.features import is_valid_geom
                from ..wetland.boundary import load_vector_boundary

                if not isinstance(boundary_path, (str, os.PathLike)):
                    raise ValueError("Missing or invalid boundary path.")

                file_path = Path(boundary_path)

                if not file_path.is_file():
                    raise FileNotFoundError(
                        "Boundary file does not exist."
                    )

                # Prevent silent CRS fallback for shapefiles.
                if file_path.suffix.lower() == ".shp":
                    prj = file_path.with_suffix(".prj")

                    if not prj.is_file():
                        raise ValueError(
                            "Shapefile CRS projection file is missing."
                        )

                    declared_crs = CRS.from_string(
                        prj.read_text(encoding="utf-8").strip()
                    )

                    if not declared_crs:
                        raise ValueError("Invalid shapefile CRS.")

                # GeoJSON without a CRS uses its standard WGS84 convention.
                elif file_path.suffix.lower() in (".json", ".geojson"):
                    data = json.loads(
                        file_path.read_text(encoding="utf-8")
                    )

                    if not isinstance(data, dict):
                        raise ValueError("Invalid GeoJSON structure.")

                    if "crs" in data:
                        crs_name = data["crs"]["properties"]["name"]

                        if not CRS.from_string(crs_name):
                            raise ValueError("Invalid declared GeoJSON CRS.")

                else:
                    raise ValueError("Unsupported boundary format.")

                if official_source:
                    classes = official_cfg.get("classes", [])
                    class_field = official_cfg.get("class_field")

                    if not isinstance(classes, list):
                        raise ValueError("Invalid official dataset classes.")

                    if classes and not isinstance(class_field, str):
                        raise ValueError("Missing official dataset class field.")

                    geometries, crs = load_vector_boundary(
                        file_path,
                        class_field=class_field,
                        select_classes=classes or None,
                    )
                else:
                    geometries, crs = load_vector_boundary(file_path)

                if not crs:
                    raise ValueError("Boundary CRS unavailable.")

                if not geometries:
                    raise ValueError("No polygon geometries available.")

                def validate_ring(ring):
                    coordinates = np.asarray(
                        ring, dtype=np.float64
                    )

                    if (
                        coordinates.ndim != 2
                        or coordinates.shape[0] < 4
                        or coordinates.shape[1] < 2
                        or not np.all(np.isfinite(coordinates))
                    ):
                        raise ValueError("Invalid polygon coordinates.")

                    xy = coordinates[:, :2]

                    if not np.array_equal(xy[0], xy[-1]):
                        raise ValueError("Polygon ring is not closed.")

                    if len(np.unique(xy[:-1], axis=0)) < 3:
                        raise ValueError("Degenerate polygon ring.")

                    shifted = xy - xy[0]

                    with np.errstate(over="ignore", invalid="ignore"):
                        twice_area = np.sum(
                            shifted[:-1, 0] * shifted[1:, 1]
                            - shifted[1:, 0] * shifted[:-1, 1]
                        )

                    if not np.isfinite(twice_area) or twice_area == 0:
                        raise ValueError("Polygon ring has invalid area.")

                for geometry in geometries:
                    if not is_valid_geom(geometry):
                        raise ValueError("Invalid boundary geometry.")

                    geometry_type = geometry.get("type")
                    coordinates = geometry.get("coordinates")

                    if geometry_type == "Polygon":
                        polygons = [coordinates]
                    elif geometry_type == "MultiPolygon":
                        polygons = coordinates
                    else:
                        raise ValueError("Unsupported boundary geometry.")

                    if not polygons:
                        raise ValueError("Empty polygon collection.")

                    for polygon in polygons:
                        if not polygon:
                            raise ValueError("Empty polygon.")

                        for ring in polygon:
                            validate_ring(ring)

                res["boundary_verified"] = True

            except Exception as exc:
                finding(
                    "QA-011", "CRITICAL",
                    f"Wetland boundary verification failed: {exc}",
                    "FAIL",
                    "Provide a readable polygon boundary with valid CRS.",
                )

        if res["source_type"] == "UNAVAILABLE":
            finding(
                "QA-011", "MEDIUM",
                "Authoritative wetland context is unavailable.",
                "WARN",
                "Provide a verified wetland boundary.",
            )

        elif res["source_type"] in inferred_sources:
            finding(
                "QA-011", "MEDIUM",
                "Wetland extent is inferred, not authoritative.",
                "WARN",
                "Do not interpret inferred water extent as physical depletion.",
            )

        # Validate the actual analysis mask.
        if (
            not isinstance(valid_analysis_mask, np.ndarray)
            or valid_analysis_mask.ndim != 2
            or valid_analysis_mask.size == 0
        ):
            finding(
                "QA-002", "CRITICAL",
                "Invalid wetland analysis mask dimensions.",
                "FAIL",
                "Provide a non-empty two-dimensional analysis mask.",
            )
            return res

        if (
            valid_analysis_mask.dtype.kind not in "biuf"
            or not np.all(np.isin(valid_analysis_mask, [0, 1]))
        ):
            finding(
                "QA-003", "CRITICAL",
                "Wetland analysis mask contains invalid values.",
                "FAIL",
                "Use a boolean or binary 0/1 analysis mask.",
            )
            return res

        analysis_mask = valid_analysis_mask.astype(
            bool, copy=False
        )

        res["valid_analysis_pixels"] = int(
            np.count_nonzero(analysis_mask)
        )

        if wetland_mask is None:
            finding(
                "QA-003", "INFO",
                "Wetland membership mask is unavailable.",
                "NOT_AVAILABLE",
                "Provide the generated wetland membership mask.",
            )

        elif (
            not isinstance(wetland_mask, np.ndarray)
            or wetland_mask.ndim != 2
            or wetland_mask.shape != analysis_mask.shape
            or wetland_mask.dtype.kind not in "biuf"
        ):
            finding(
                "QA-002", "CRITICAL",
                "Invalid wetland membership mask dimensions or dtype.",
                "FAIL",
                "Provide an aligned numeric wetland membership mask.",
            )

        else:
            invalid_membership = ~np.isin(
                wetland_mask, [0, 1, 255]
            )

            invalid_count = int(
                np.count_nonzero(invalid_membership)
            )

            res["invalid_membership_pixels"] = invalid_count

            if invalid_count:
                finding(
                    "QA-003", "CRITICAL",
                    f"{invalid_count} invalid wetland membership pixels.",
                    "FAIL",
                    "Use membership values 0, 1 or 255 only.",
                )

            res["wetland_pixels"] = int(
                np.count_nonzero(wetland_mask == 1)
            )

            # Fallback masks are not scientific wetland membership.
            if res["source_type"] != "UNAVAILABLE":
                invalid_analysis = (
                    analysis_mask & (wetland_mask != 1)
                )

                invalid_analysis_count = int(
                    np.count_nonzero(invalid_analysis)
                )

                res["invalid_analysis_pixels"] = invalid_analysis_count

                if invalid_analysis_count:
                    finding(
                        "QA-003", "CRITICAL",
                        f"{invalid_analysis_count} analysis pixels lack wetland membership.",
                        "FAIL",
                        "Restrict wetland analysis to valid membership pixels.",
                    )

        if res["valid_analysis_pixels"] == 0:
            finding(
                "QA-011", "INFO",
                "No valid wetland analysis pixels are available.",
                "NOT_AVAILABLE",
                "Verify boundary, AOI and temporal overlap.",
            )

        # A configured request is not proof of physical depletion.
        if claims_enabled or result_depletion_enabled:
            prerequisites_met = (
                res["source_type"] in authoritative_sources
                and res["boundary_verified"]
                and physical_area_available
                and independent_evidence
                and result_depletion_enabled
                and wetland_mask is not None
                and res["valid_analysis_pixels"] > 0
                and res["status"] != "FAIL"
            )

            if not prerequisites_met:
                finding(
                    "QA-011", "CRITICAL",
                    "Physical wetland depletion prerequisites are not satisfied.",
                    "FAIL",
                    "Disable depletion claims until authoritative boundary, physical area and independent evidence are verified.",
                )
            else:
                # The configuration flag alone does not verify evidence.
                finding(
                    "QA-011", "MEDIUM",
                    "Independent hydrological evidence requires verification.",
                    "WARN",
                    "Validate the actual independent evidence before reporting physical depletion.",
                )

        return res

    # 9. METRICS QA
    def run_metrics_qa(
        self,
        metrics: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate the actual MetricsResult data contract."""

        import math
        from numbers import Real, Integral

        res = {
            "status": "PASS",
            "physical_area_available": None,
            "area_basis": None,
            "checks": {},
            "severity_weights_disclaimer":
                "PROVISIONAL_ENGINEERING_PARAMETERS",
        }

        priority = {
            "PASS": 0,
            "NOT_AVAILABLE": 1,
            "WARN": 2,
            "FAIL": 3,
        }

        def finding(code, message, status="FAIL"):
            severity = {
                "FAIL": "CRITICAL",
                "WARN": "MEDIUM",
                "NOT_AVAILABLE": "INFO",
            }[status]

            self.add_flag(
                code,
                severity,
                message,
                "Metrics QA",
                status,
                "Verify metric inputs, formulas and output metadata.",
            )

            if priority[status] > priority[res["status"]]:
                res["status"] = status

        def section(parent, key, required=True):
            value = parent.get(key)

            if not isinstance(value, dict):
                if required or key in parent:
                    finding(
                        "QA-020",
                        f"Missing or invalid metrics section: {key}.",
                    )
                return {}

            return value

        def number(
            parent,
            key,
            label,
            nullable=False,
            signed=False,
            maximum=None,
        ):
            if key not in parent:
                finding(
                    "QA-020",
                    f"Missing metric: {label}.{key}.",
                )
                return None

            value = parent[key]

            if value is None and nullable:
                return None

            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
            ):
                finding(
                    "QA-020",
                    f"Invalid numeric metric: {label}.{key}.",
                )
                return None

            try:
                value = float(value)
            except (ValueError, TypeError, OverflowError):
                finding(
                    "QA-020",
                    f"Unrepresentable numeric metric: {label}.{key}.",
                )
                return None

            if not math.isfinite(value):
                finding(
                    "QA-020",
                    f"Non-finite metric: {label}.{key}.",
                )
                return None

            if not signed and value < 0:
                finding(
                    "QA-020",
                    f"Negative metric: {label}.{key}.",
                )
                return None

            if maximum is not None and value > maximum:
                finding(
                    "QA-016",
                    f"Metric exceeds allowed range: {label}.{key}.",
                )
                return None

            return value

        def count(parent, key, label):
            value = parent.get(key)

            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral)
                or value < 0
            ):
                finding(
                    "QA-020",
                    f"Invalid or missing pixel count: {label}.{key}.",
                )
                return None

            return int(value)

        def reconcile(
            actual,
            expected,
            label,
            atol=1e-8,
            rtol=1e-5,
        ):
            if actual is None or expected is None:
                return

            if not math.isclose(
                actual,
                expected,
                rel_tol=rtol,
                abs_tol=atol,
            ):
                finding(
                    "QA-015",
                    f"Metric reconciliation failed: {label}.",
                )
                res["checks"][label] = "FAIL"
            else:
                res["checks"][label] = "PASS"

        def upper_bound(value, limit, label):
            if value is None or limit is None:
                return

            if value > limit and not math.isclose(
                value,
                limit,
                rel_tol=1e-5,
                abs_tol=1e-3,
            ):
                finding(
                    "QA-020",
                    f"{label} exceeds its allowed baseline.",
                )

        if metrics is None or (
            isinstance(metrics, dict) and not metrics
        ):
            finding(
                "QA-020",
                "Metrics output is unavailable.",
                "NOT_AVAILABLE",
            )
            return res

        if not isinstance(metrics, dict):
            finding(
                "QA-020",
                "Metrics output must be a dictionary.",
            )
            return res

        if not isinstance(config, dict):
            finding(
                "QA-020",
                "Metrics configuration must be a dictionary.",
            )
            return res

        metadata = section(metrics, "metadata")
        producer_metadata = section(
            metadata, "metrics_metadata"
        )

        physical = metadata.get("physical_area_available")

        if not isinstance(physical, bool):
            finding(
                "QA-020",
                "Physical area availability must be boolean.",
            )
            return res

        res["physical_area_available"] = physical

        expected_basis = (
            "physical_crs"
            if physical
            else "pixel_count_only"
        )

        res["area_basis"] = expected_basis

        if metadata.get("area_basis") != expected_basis:
            finding(
                "QA-020",
                "Area basis conflicts with physical area availability.",
            )

        if producer_metadata.get(
            "physical_area_available"
        ) is not physical:
            finding(
                "QA-020",
                "Nested physical area metadata is inconsistent.",
            )

        if producer_metadata.get(
            "area_basis"
        ) != expected_basis:
            finding(
                "QA-020",
                "Nested area basis is inconsistent.",
            )

        if not physical:
            finding(
                "QA-020",
                "Physical areas are unavailable; validating pixel counts.",
                "NOT_AVAILABLE",
            )

        def area(
            parent,
            label,
            m2_key="area_m2",
            ha_key="area_ha",
            signed=False,
        ):
            m2 = number(
                parent,
                m2_key,
                label,
                nullable=not physical,
                signed=signed,
            )

            ha = number(
                parent,
                ha_key,
                label,
                nullable=not physical,
                signed=signed,
            )

            if not physical:
                if (
                    parent.get(m2_key) is not None
                    or parent.get(ha_key) is not None
                ):
                    finding(
                        "QA-020",
                        f"Unsupported physical area reported: {label}.",
                    )
                return None

            if m2 is not None and ha is not None:
                reconcile(
                    ha,
                    m2 / 10000.0,
                    f"{label}_unit_conversion",
                    atol=1e-7,
                )

            return m2

        baseline = section(metrics, "baseline_area")
        impact = section(metrics, "impact_area")
        weighted = section(metrics, "severity_weighted_area")
        recovery = section(metrics, "recovery_area")
        canopy = section(metrics, "canopy_loss_area")
        residual = section(metrics, "residual_stress_area")

        baseline_pixels = count(
            baseline, "pixel_count", "baseline"
        )

        baseline_m2 = area(baseline, "baseline")

        impact_m2 = area(
            impact,
            "impact",
            "impact_area_m2",
            "impact_area_ha",
        )

        swd_m2 = area(
            impact,
            "severity_weighted",
            "severity_weighted_area_m2",
            "severity_weighted_area_ha_deg",
        )

        duplicate_swd_m2 = area(
            weighted,
            "duplicate_severity_weighted",
            "severity_weighted_area_m2",
            "severity_weighted_area_ha_deg",
        )

        reconcile(
            duplicate_swd_m2,
            swd_m2,
            "severity_weighted_duplicate",
            atol=1e-3,
        )

        recovery_m2 = area(recovery, "recovery")
        canopy_m2 = area(canopy, "canopy_loss")
        residual_m2 = area(residual, "residual_stress")

        for label, value in (
            ("Impact area", impact_m2),
            ("Severity-weighted area", swd_m2),
            ("Recovery area", recovery_m2),
            ("Canopy loss area", canopy_m2),
            ("Residual stress area", residual_m2),
        ):
            upper_bound(value, baseline_m2, label)

        if baseline_pixels == 0:
            finding(
                "QA-016",
                "Baseline contains no valid pixels.",
                "WARN",
            )

            if baseline_m2 is not None and baseline_m2 != 0:
                finding(
                    "QA-020",
                    "Nonzero baseline area with zero pixels.",
                )

        if (
            physical
            and baseline_pixels is not None
            and baseline_pixels > 0
            and baseline_m2 == 0
        ):
            finding(
                "QA-020",
                "Positive baseline pixel count with zero physical area.",
            )

        def ratio(parent, key, numerator, denominator, label):
            unavailable = (
                not physical
                or denominator is None
                or denominator == 0
            )

            value = number(
                parent,
                key,
                label,
                nullable=unavailable,
                maximum=1.0,
            )

            if unavailable:
                if parent.get(key) is not None:
                    finding(
                        "QA-016",
                        f"{label} reported without a valid denominator.",
                    )
                return None

            if numerator is not None:
                reconcile(
                    value,
                    numerator / denominator,
                    label,
                )

            return value

        ndr = ratio(
            metrics,
            "normalized_degradation_ratio",
            swd_m2,
            baseline_m2,
            "ndr",
        )

        nested_ndr = ratio(
            impact,
            "ndr",
            swd_m2,
            baseline_m2,
            "nested_ndr",
        )

        reconcile(ndr, nested_ndr, "ndr_duplicate")

        def percentage(parent, key, value, label):
            result = number(
                parent,
                key,
                label,
                nullable=value is None,
                maximum=100.0,
            )

            if value is None:
                if parent.get(key) is not None:
                    finding(
                        "QA-016",
                        f"{label} reported without a valid ratio.",
                    )
            else:
                reconcile(
                    result,
                    value * 100.0,
                    label,
                    atol=1e-6,
                )

        percentage(
            impact, "ndr_percent", ndr, "ndr_percentage"
        )

        ratio(
            recovery,
            "ratio",
            recovery_m2,
            baseline_m2,
            "recovery_ratio",
        )

        canopy_ratio = ratio(
            canopy,
            "ratio",
            canopy_m2,
            baseline_m2,
            "canopy_loss_ratio",
        )

        percentage(
            canopy,
            "percentage",
            canopy_ratio,
            "canopy_loss_percentage",
        )

        # Validate the complete transition population.
        transitions = section(metrics, "transition_metrics")

        transition_names = [
            f"{a}_to_{b}"
            for a in ("B", "S", "M", "H")
            for b in ("B", "S", "M", "H")
        ]

        transition_counts = {}
        transition_areas = {}

        def population_entry(parent, name, label, pct_key):
            entry = section(parent, name)

            pixels = count(entry, "pixel_count", label)
            m2 = area(entry, label)

            pct = number(
                entry,
                pct_key,
                label,
                maximum=100.0,
            )

            if baseline_pixels is not None and pixels is not None:
                if pixels > baseline_pixels:
                    finding(
                        "QA-015",
                        f"{label} pixel count exceeds baseline.",
                    )

            upper_bound(m2, baseline_m2, label)

            if physical:
                if m2 is not None and baseline_m2 is not None:
                    expected_pct = (
                        m2 / baseline_m2 * 100.0
                        if baseline_m2 > 0 else 0.0
                    )
                    reconcile(
                        pct,
                        expected_pct,
                        f"{label}_percentage",
                        atol=1e-6,
                    )
            elif pixels is not None and baseline_pixels is not None:
                expected_pct = (
                    pixels / baseline_pixels * 100.0
                    if baseline_pixels > 0 else 0.0
                )
                reconcile(
                    pct,
                    expected_pct,
                    f"{label}_pixel_percentage",
                    atol=1e-6,
                )

            return pixels, m2

        for name in transition_names:
            pixels, m2 = population_entry(
                transitions,
                name,
                f"transition_{name}",
                "percentage_of_valid_wetland_analysis_area",
            )

            transition_counts[name] = pixels
            transition_areas[name] = m2

        def total(values):
            values = list(values)

            if any(value is None for value in values):
                return None

            try:
                result = sum(values)

                if not math.isfinite(result):
                    raise OverflowError

                return result
            except (OverflowError, ValueError):
                finding(
                    "QA-020",
                    "Metric total exceeds numeric limits.",
                )
                return None

        def reconcile_count(actual, expected, label):
            if actual is None or expected is None:
                return

            if actual != expected:
                finding(
                    "QA-015",
                    f"Pixel count reconciliation failed: {label}.",
                )
                res["checks"][label] = "FAIL"
            else:
                res["checks"][label] = "PASS"

        reconcile_count(
            total(transition_counts.values()),
            baseline_pixels,
            "transition_pixel_total",
        )

        reconcile(
            total(transition_areas.values()),
            baseline_m2,
            "transition_area_total",
            atol=1e-3,
        )

        stable_vegetation_m2 = total(
            transition_areas[name]
            for name in ("S_to_S", "M_to_M", "H_to_H")
        )

        upper_bound(
            residual_m2,
            stable_vegetation_m2,
            "Residual stress area",
        )

        ratio(
            residual,
            "ratio",
            residual_m2,
            stable_vegetation_m2,
            "residual_stress_ratio",
        )

        # Reconcile class totals and transition marginals.
        class_metrics = section(metrics, "class_area_metrics")

        class_names = {
            "background": "B",
            "severe": "S",
            "moderate": "M",
            "healthy": "H",
        }

        class_areas = {"T1": {}, "T2": {}}

        for date in ("T1", "T2"):
            date_data = section(class_metrics, date)
            date_counts = []
            date_areas = []

            for name, short in class_names.items():
                label = f"{date}_{name}"

                pixels, m2 = population_entry(
                    date_data,
                    name,
                    label,
                    "percentage",
                )

                date_counts.append(pixels)
                date_areas.append(m2)
                class_areas[date][name] = m2

                matching = [
                    transition
                    for transition in transition_names
                    if (
                        transition.startswith(short + "_to_")
                        if date == "T1"
                        else transition.endswith("_to_" + short)
                    )
                ]

                reconcile_count(
                    pixels,
                    total(
                        transition_counts[t]
                        for t in matching
                    ),
                    f"{label}_transition_pixels",
                )

                reconcile(
                    m2,
                    total(
                        transition_areas[t]
                        for t in matching
                    ),
                    f"{label}_transition_area",
                    atol=1e-3,
                )

            reconcile_count(
                total(date_counts),
                baseline_pixels,
                f"{date}_class_pixel_total",
            )

            reconcile(
                total(date_areas),
                baseline_m2,
                f"{date}_class_area_total",
                atol=1e-3,
            )

        changes = section(class_metrics, "changes")

        for name in class_names:
            entry = section(changes, name)

            change_m2 = area(
                entry,
                f"{name}_change",
                "area_change_m2",
                "area_change_ha",
                signed=True,
            )

            before = class_areas["T1"][name]
            after = class_areas["T2"][name]

            if before is not None and after is not None:
                reconcile(
                    change_m2,
                    after - before,
                    f"{name}_area_change",
                    atol=1e-3,
                )

            if name == "healthy":
                loss = area(
                    entry,
                    "healthy_loss",
                    "loss_m2",
                    "loss_ha",
                )

                if before is not None and after is not None:
                    reconcile(
                        loss,
                        max(before - after, 0.0),
                        "healthy_loss_formula",
                        atol=1e-3,
                    )

        # Validate severity weights against producer metadata.
        standard_weights = {
            "healthy": 0.0,
            "moderate": 0.35,
            "severe": 0.80,
            "background": 1.0,
        }

        metrics_cfg = section(
            config, "metrics", required=False
        )

        configured_weights = metrics_cfg.get(
            "severity_weights", standard_weights
        )

        actual_weights = section(
            producer_metadata, "severity_weights"
        )

        if not isinstance(configured_weights, dict):
            finding(
                "QA-020",
                "Severity weights must be a dictionary.",
            )
            configured_weights = {}

        for name, default in standard_weights.items():
            configured_value = number(
                {
                    name: configured_weights.get(
                        name, default
                    )
                },
                name,
                "configured_weights",
                maximum=1.0,
            )

            actual_value = number(
                {
                    name: actual_weights.get(
                        name, default
                    )
                },
                name,
                "producer_weights",
                maximum=1.0,
            )

            reconcile(
                actual_value,
                configured_value,
                f"severity_weight_{name}",
            )

        include_canopy = producer_metadata.get(
            "include_canopy_loss_in_swd"
        )

        if not isinstance(include_canopy, bool):
            finding(
                "QA-020",
                "Canopy-loss weighting setting must be boolean.",
            )
        else:
            configured_canopy = metrics_cfg.get(
                "include_canopy_loss_in_swd", False
            )

            if (
                not isinstance(configured_canopy, bool)
                or configured_canopy != include_canopy
            ):
                finding(
                    "QA-020",
                    "Canopy-loss weighting configuration is inconsistent.",
                )

            swd_limit = impact_m2

            if include_canopy:
                swd_limit = (
                    impact_m2 + canopy_m2
                    if impact_m2 is not None
                    and canopy_m2 is not None
                    else None
                )

            upper_bound(
                swd_m2,
                swd_limit,
                "Severity-weighted area",
            )

        # Wetland extent evidence safeguards.

        extent = section(
            metrics, "wetland_extent_metrics"
        )

        required_extent_fields = (
            "status",
            "extent_T1_ha",
            "extent_T2_ha",
            "extent_change_ha",
            "physical_wetland_depletion",
        )

        for field in required_extent_fields:
            if field not in extent:
                finding(
                    "QA-020",
                    f"Missing wetland extent field: {field}.",
                )

        extent_status = extent.get("status")

        if extent_status != "NOT_AVAILABLE":
            finding(
                "QA-020",
                "Wetland extent availability is unsupported "
                "without independently verified evidence.",
            )

        for field in (
            "extent_T1_ha",
            "extent_T2_ha",
            "extent_change_ha",
        ):
            if extent.get(field) is not None:
                finding(
                    "QA-020",
                    f"Unsupported wetland extent value: {field}.",
                )

        depletion = extent.get(
            "physical_wetland_depletion"
        )

        if not (
            isinstance(depletion, str)
            and depletion == "NOT_AVAILABLE"
        ):
            finding(
                "QA-020",
                "Physical wetland depletion must remain "
                "NOT_AVAILABLE until evidence is verified.",
            )


        # ADVANCED METRICS CROSS-FIELD QA

        def duplicate(
            parent,
            key,
            expected,
            label,
            maximum=None,
        ):
            actual = number(
                parent,
                key,
                label,
                nullable=expected is None,
                maximum=maximum,
            )

            if expected is None:
                if parent.get(key) is not None:
                    finding(
                        "QA-015",
                        f"Unexpected value in {label}.{key}.",
                    )
            else:
                reconcile(
                    actual,
                    expected,
                    f"{label}_{key}",
                )

        # 1. Reconcile duplicate degradation fields.

        duplicate_impact = area(
            weighted,
            "duplicate_impact",
            "impact_area_m2",
            "impact_area_ha",
        )

        reconcile(
            duplicate_impact,
            impact_m2,
            "duplicate_impact_area",
            atol=1e-3,
        )

        duplicate(
            weighted,
            "ndr",
            ndr,
            "weighted",
            maximum=1.0,
        )

        duplicate(
            weighted,
            "ndr_percent",
            ndr * 100.0 if ndr is not None else None,
            "weighted",
            maximum=100.0,
        )

        # 2. Reconcile the public summary.

        summary = section(metadata, "summary")

        def hectares(value):
            return (
                value / 10000.0
                if value is not None
                else None
            )

        summary_values = {
            "baseline_area_ha": hectares(baseline_m2),
            "affected_area_ha": hectares(impact_m2),
            "severity_weighted_area_ha_deg": hectares(swd_m2),
            "normalized_degradation_ratio": ndr,
            "recovery_area_ha": hectares(recovery_m2),
            "residual_stress_area_ha": hectares(residual_m2),
            "canopy_loss_area_ha": hectares(canopy_m2),
        }

        for key, expected in summary_values.items():
            duplicate(
                summary,
                key,
                expected,
                "summary",
            )

        # 3. Reject unexpected transition and class entries.

        unexpected_transitions = (
            set(transitions) - set(transition_names)
        )

        if unexpected_transitions:
            finding(
                "QA-015",
                "Unexpected transition entries detected.",
            )

        for date in ("T1", "T2"):
            date_data = section(class_metrics, date)

            if set(date_data) - set(class_names):
                finding(
                    "QA-015",
                    f"Unexpected class entries detected for {date}.",
                )

        if set(changes) - set(class_names):
            finding(
                "QA-015",
                "Unexpected class change entries detected.",
            )

        # 4. Validate pixel count and area relationships.

        for name in transition_names:
            pixels = transition_counts[name]
            m2 = transition_areas[name]

            if physical and pixels is not None and m2 is not None:
                if pixels == 0 and m2 != 0:
                    finding(
                        "QA-020",
                        f"Transition {name} has area but no pixels.",
                    )

                if pixels > 0 and m2 <= 0:
                    finding(
                        "QA-020",
                        f"Transition {name} has pixels but no area.",
                    )

        # 5. Validate eligible transition-area limits.
        # Filtered change areas may be smaller than raw transitions.

        degradation_limit = total(
            transition_areas[name]
            for name in (
                "H_to_M",
                "H_to_S",
                "M_to_S",
            )
        )

        recovery_limit = total(
            transition_areas[name]
            for name in (
                "M_to_H",
                "S_to_M",
                "S_to_H",
            )
        )

        canopy_limit = total(
            transition_areas[name]
            for name in (
                "H_to_B",
                "M_to_B",
                "S_to_B",
            )
        )

        upper_bound(
            impact_m2,
            degradation_limit,
            "Degradation transition area",
        )

        upper_bound(
            recovery_m2,
            recovery_limit,
            "Recovery transition area",
        )

        upper_bound(
            canopy_m2,
            canopy_limit,
            "Canopy loss transition area",
        )

        # 6. Reconcile raw recovery transition details.

        recovery_details = section(
            recovery, "details_m2"
        )

        for name in (
            "M_to_H",
            "S_to_M",
            "S_to_H",
        ):
            duplicate(
                recovery_details,
                name,
                transition_areas[name],
                "recovery_details",
            )

        # 7. Calculate maximum possible weighted area.
        # Do not assume filtered patches equal raw transitions.

        weighted_transitions = {
            "H_to_M": "moderate",
            "H_to_S": "severe",
            "M_to_S": "severe",
        }

        if include_canopy is True:
            weighted_transitions.update({
                "H_to_B": "background",
                "M_to_B": "background",
                "S_to_B": "background",
            })

        weighted_limits = []

        for name, weight_name in weighted_transitions.items():
            transition_area = transition_areas[name]

            weight = number(
                {
                    weight_name: actual_weights.get(
                        weight_name,
                        standard_weights[weight_name],
                    )
                },
                weight_name,
                "weighted_transition",
                maximum=1.0,
            )

            weighted_limits.append(
                transition_area * weight
                if transition_area is not None
                and weight is not None
                else None
            )

        upper_bound(
            swd_m2,
            total(weighted_limits),
            "Maximum transition-weighted area",
        )

        # 8. Verify producer reconciliation flags.

        producer_qa = section(metrics, "qa_flags")
        producer_flags = section(producer_qa, "flags")

        if producer_flags.get(
            "reconciliation_class_passes"
        ) is not True:
            finding(
                "QA-015",
                "Producer class reconciliation did not pass.",
            )

        transition_flag = producer_flags.get(
            "reconciliation_transition_passes"
        )

        if physical:
            if transition_flag is not True:
                finding(
                    "QA-015",
                    "Producer transition reconciliation did not pass.",
                )
        else:
            # The current producer returns None when
            # physical transition-area reconciliation is unavailable.
            if transition_flag is not None and transition_flag is not True:
                finding(
                    "QA-015",
                    "Invalid pixel-only transition reconciliation status.",
                )

        # 9. Reconcile wetland extent summary status.

        if summary.get(
            "wetland_extent_status"
        ) != extent.get("status"):
            finding(
                "QA-015",
                "Wetland extent summary status is inconsistent.",
            )


        # CVA AND DELTA STATISTICS QA

        def validate_statistics(
            parent,
            key,
            population,
            percentiles,
            signed=False,
        ):
            stats = section(parent, key)

            fields = [
                "min",
                "max",
                "mean",
                "median",
                "std",
            ] + [
                f"p{p}" for p in percentiles
            ]

            values = {}

            for field in fields:
                values[field] = number(
                    stats,
                    field,
                    key,
                    nullable=population is None
                    or population == 0,
                    signed=signed and field != "std",
                )

            if population == 0:
                if any(
                    stats.get(field) is not None
                    for field in fields
                ):
                    finding(
                        "QA-020",
                        f"{key}: empty population contains statistics.",
                    )

                return values

            if any(
                value is None
                for value in values.values()
            ):
                return values

            minimum = values["min"]
            maximum = values["max"]
            mean = values["mean"]
            median = values["median"]
            std = values["std"]

            if minimum > maximum:
                finding(
                    "QA-020",
                    f"{key}: minimum exceeds maximum.",
                )

                return values

            tolerance = 1e-7 * max(
                1.0,
                abs(minimum),
                abs(maximum),
            )

            for field in (
                ["mean", "median"]
                + [f"p{p}" for p in percentiles]
            ):
                value = values[field]

                if (
                    value < minimum - tolerance
                    or value > maximum + tolerance
                ):
                    finding(
                        "QA-020",
                        f"{key}: {field} is outside observed bounds.",
                    )

            # Population standard deviation cannot
            # exceed half the observed range.
            maximum_std = maximum / 2 - minimum / 2

            if std > maximum_std + tolerance:
                finding(
                    "QA-020",
                    f"{key}: standard deviation exceeds its mathematical bound.",
                )

            ordered = [(50, median)]

            ordered.extend(
                (p, values[f"p{p}"])
                for p in percentiles
            )

            ordered.sort(key=lambda item: item[0])

            for left, right in zip(
                ordered, ordered[1:]
            ):
                if left[1] > right[1] + tolerance:
                    finding(
                        "QA-020",
                        f"{key}: percentile ordering is invalid.",
                    )

            if population == 1:
                for field in fields:
                    expected = (
                        0.0
                        if field == "std"
                        else minimum
                    )

                    reconcile(
                        values[field],
                        expected,
                        f"{key}_single_pixel_{field}",
                        atol=tolerance,
                    )

            return values

        # Statistical populations are based on raw
        # transition classes, not MMU-filtered patches.

        population_groups = {
            "all_valid_wetland": tuple(
                transition_names
            ),
            "degradation": (
                "H_to_M",
                "H_to_S",
                "M_to_S",
            ),
            "recovery": (
                "M_to_H",
                "S_to_M",
                "S_to_H",
            ),
            "stable_vegetation": (
                "S_to_S",
                "M_to_M",
                "H_to_H",
            ),
            "canopy_loss": (
                "H_to_B",
                "M_to_B",
                "S_to_B",
            ),
            "background_to_vegetation": (
                "B_to_S",
                "B_to_M",
                "B_to_H",
            ),
        }

        populations = {}

        for name, members in population_groups.items():
            populations[name] = total(
                transition_counts[member]
                for member in members
            )

        # 1. Validate all CVA statistical subsets.

        cva_statistics = section(
            metrics, "cva_statistics"
        )

        validated_cva = {}

        for name, population in populations.items():
            validated_cva[name] = validate_statistics(
                cva_statistics,
                name,
                population,
                [95],
                signed=False,
            )

        # 2. Validate signed vegetation-index changes.

        delta_statistics = section(
            metrics, "delta_statistics"
        )

        for index in ("ndvi", "gndvi", "ndre"):
            index_stats = section(
                delta_statistics, index
            )

            for name in (
                "all_valid_wetland",
                "degradation",
            ):
                validate_statistics(
                    index_stats,
                    name,
                    populations[name],
                    [5, 95],
                    signed=True,
                )

        # 3. Reconcile public CVA summary values.

        all_cva = validated_cva[
            "all_valid_wetland"
        ]

        for summary_key, stats_key in (
            ("mean_cva", "mean"),
            ("median_cva", "median"),
            ("p95_cva", "p95"),
        ):
            duplicate(
                summary,
                summary_key,
                all_cva[stats_key],
                "summary_cva",
            )

        # 4. CVA subset ranges must remain within
        # the complete analysis population range.

        all_min = all_cva["min"]
        all_max = all_cva["max"]

        if all_min is not None and all_max is not None:
            tolerance = 1e-7 * max(
                1.0,
                abs(all_min),
                abs(all_max),
            )

            for name, values in validated_cva.items():

                if name == "all_valid_wetland":
                    continue

                subset_min = values["min"]
                subset_max = values["max"]

                if (
                    subset_min is not None
                    and subset_min < all_min - tolerance
                ):
                    finding(
                        "QA-020",
                        f"{name}: CVA subset minimum is below the global minimum.",
                    )

                if (
                    subset_max is not None
                    and subset_max > all_max + tolerance
                ):
                    finding(
                        "QA-020",
                        f"{name}: CVA subset maximum exceeds the global maximum.",
                    )

        return res

    # 10. UNCERTAINTY QA
    def run_uncertainty_qa(
        self,
        config: Dict[str, Any],
        reference_samples_exist: bool,
    ) -> Dict[str, Any]:
        """Report uncertainty without unsupported scientific claims."""

        res = {
            "model": "NOT_AVAILABLE",
            "classification": "NOT_AVAILABLE",
            "spatial": "NOT_AVAILABLE",
            "temporal": "NOT_AVAILABLE",
            "wetland_boundary": "NOT_AVAILABLE",
            "area_estimation": "NOT_AVAILABLE",
            "status": "NOT_AVAILABLE",
            "probability_calibration_verified": False,
            "independent_area_error_correction": False,
        }

        if not isinstance(config, dict):
            self.add_flag(
                "QA-012", "HIGH",
                "Invalid uncertainty configuration.",
                "Uncertainty QA", "FAIL",
                "Provide a valid configuration dictionary.",
            )
            res["status"] = "FAIL"
            return res

        if not isinstance(reference_samples_exist, bool):
            self.add_flag(
                "QA-012", "HIGH",
                "Invalid reference sample availability indicator.",
                "Uncertainty QA", "FAIL",
                "Provide a boolean availability indicator.",
            )
            res["status"] = "FAIL"
            return res

        wetland_cfg = config.get("wetland", {})

        if not isinstance(wetland_cfg, dict):
            self.add_flag(
                "QA-011", "HIGH",
                "Invalid wetland uncertainty configuration.",
                "Uncertainty QA", "FAIL",
                "Provide a valid wetland configuration.",
            )
            res["status"] = "FAIL"
            return res

        boundary_cfg = wetland_cfg.get("boundary", {})

        if not isinstance(boundary_cfg, dict):
            self.add_flag(
                "QA-011", "HIGH",
                "Invalid wetland boundary configuration.",
                "Uncertainty QA", "FAIL",
                "Provide a valid boundary configuration.",
            )
            res["status"] = "FAIL"
            return res

        findings = [
            (
                "QA-013",
                "Model probability calibration has not been independently verified.",
                "Evaluate probabilities against independent held-out labels.",
            ),
            (
                "QA-008",
                "Classification uncertainty has not been quantified.",
                "Obtain independent classification accuracy evidence.",
            ),
            (
                "QA-003",
                "Spatial uncertainty has not been quantified.",
                "Obtain positional accuracy and registration error evidence.",
            ),
            (
                "QA-006",
                "Temporal uncertainty has not been quantified.",
                "Evaluate acquisition conditions and temporal comparability.",
            ),
            (
                "QA-011",
                "Wetland boundary uncertainty has not been independently verified.",
                "Validate boundary provenance and positional accuracy.",
            ),
            (
                "QA-014",
                "Independent area-error correction has not been verified.",
                "Validate sampling design and area-estimation results.",
            ),
        ]

        for code, message, action in findings:
            self.add_flag(
                code, "INFO", message,
                "Uncertainty QA", "NOT_AVAILABLE",
                action,
            )

        if reference_samples_exist:
            message = (
                "Reference samples are reported as available, "
                "but their independence, sampling design and "
                "estimation results have not been verified."
            )
        else:
            message = (
                "Independent reference samples unavailable. "
                "Area-estimation uncertainty unavailable."
            )

        self.add_flag(
            "QA-012", "INFO", message,
            "Uncertainty QA", "NOT_AVAILABLE",
            "Provide independently validated reference samples.",
        )

        self.mark_category(
            "uncertainty", "NOT_AVAILABLE"
        )

        return res

    def generate_qa_report(self) -> Dict[str, Any]:
        """Aggregate QA outcomes and expose individual check coverage."""

        categories = self.aggregate_check_categories()

        statuses = [
            status for status in categories.values()
            if status != "NOT_APPLICABLE"
        ]

        not_run = sorted(
            name for name, status in categories.items()
            if status == "NOT_RUN"
        )

        unavailable = sorted(
            name for name, status in categories.items()
            if status == "NOT_AVAILABLE"
        )

        if "FAIL" in statuses:

            overall = "FAIL"

        elif not statuses or any(
            status != "PASS" for status in statuses
        ):

            overall = "WARN"

        else:

            overall = "PASS"

        result = {
            "overall_status": overall,
            "categories": categories,
            "flags": [dict(flag) for flag in self.flags],
            "not_run_categories": not_run,
            "unavailable_categories": unavailable,
        }

        if hasattr(self, "check_status"):

            result.update({
                "analysis_mode": self.qa_analysis_mode,
                "required_checks": list(self.required_checks),
                "checks": dict(self.check_status),
                "not_run_checks": sorted(
                    name for name, status
                    in self.check_status.items()
                    if status == "NOT_RUN"
                ),
                "unavailable_checks": sorted(
                    name for name, status
                    in self.check_status.items()
                    if status == "NOT_AVAILABLE"
                ),
                "not_applicable_categories": sorted(
                    name for name, status in categories.items()
                    if status == "NOT_APPLICABLE"
                ),
            })

        return result


    def run_reproducibility_qa(
        self,
        metadata,
        manifest=None,
        config_snapshot=None,
    ):
        """Validate recorded provenance without claiming rerun equivalence."""
        import re

        from ..core.config import config_hash

        checks = {}

        def valid_hash(value):

            return (
                isinstance(value, str)
                and re.fullmatch(
                    r"[0-9a-f]{64}", value
                ) is not None
            )

        def record(name, status, message):

            checks[name] = {
                "status": status,
                "message": message,
            }

            if status != "PASS":

                severity = {
                    "FAIL": "HIGH",
                    "WARN": "MEDIUM",
                    "NOT_AVAILABLE": "INFO",
                }[status]

                self.add_flag(
                    "QA017",
                    severity,
                    message,
                    "Reproducibility QA",
                    status,
                    "Review provenance and rerun verification.",
                )

        def finish():

            statuses = {
                item["status"]
                for item in checks.values()
            }

            if "FAIL" in statuses:
                status = "FAIL"
            elif "WARN" in statuses:
                status = "WARN"
            elif "NOT_AVAILABLE" in statuses:
                status = "NOT_AVAILABLE"
            else:
                status = "PASS"

            self.mark_category(
                "reproducibility", status
            )

            return {
                "status": status,
                "checks": checks,
                "recorded_provenance_consistent": (
                    all(
                        item["status"] == "PASS"
                        for item in checks.values()
                    )
                ),
                "reproducibility_verified": False,
            }

        if not isinstance(metadata, dict):

            record(
                "metadata",
                "FAIL",
                "Execution metadata must be a dictionary.",
            )

            return finish()

        record(
            "metadata",
            "PASS",
            "Execution metadata structure available.",
        )

        configuration_hash = metadata.get(
            "configuration_hash"
        )

        if not valid_hash(configuration_hash):

            record(
                "configuration",
                "FAIL",
                "Configuration checksum missing or invalid.",
            )

        elif config_snapshot is None:

            record(
                "configuration",
                "NOT_AVAILABLE",
                "Configuration snapshot not supplied.",
            )

        else:

            try:

                actual_hash = config_hash(config_snapshot)

                matches = (
                    actual_hash == configuration_hash
                )

            except Exception:

                matches = False

            record(
                "configuration",
                "PASS" if matches else "FAIL",
                "Configuration checksum reconciled."
                if matches else
                "Configuration snapshot checksum mismatch.",
            )

        record(
            "model",
            "PASS"
            if valid_hash(metadata.get("model_sha256"))
            else "FAIL",
            "Model checksum format validated."
            if valid_hash(metadata.get("model_sha256"))
            else "Model checksum missing or invalid.",
        )

        single = "input_scene" in metadata
        paired = (
            "input_t1" in metadata
            or "input_t2" in metadata
        )

        if single and not paired:

            expected_inputs = ("input_scene",)

        elif paired and not single:

            expected_inputs = ("input_t1", "input_t2")

        else:

            expected_inputs = ()

            record(
                "input_structure",
                "FAIL",
                "Missing or ambiguous input provenance.",
            )

        for name in expected_inputs:

            fingerprint = metadata.get(name)

            valid = (
                isinstance(fingerprint, dict)
                and fingerprint.get("status") == "PASS"
                and fingerprint.get("exists") is True
                and valid_hash(fingerprint.get("sha256"))
                and fingerprint.get("kind")
                    in ("file", "directory")
                and type(fingerprint.get("file_count")) is int
                and fingerprint["file_count"] > 0
                and type(fingerprint.get("size_bytes")) is int
                and fingerprint["size_bytes"] >= 0
                and fingerprint.get("fingerprint_version") == 1
            )

            if valid and fingerprint["kind"] == "file":

                valid = fingerprint["file_count"] == 1

            record(
                name,
                "PASS" if valid else "FAIL",
                f"{name} fingerprint metadata validated."
                if valid else
                f"{name} fingerprint missing or invalid.",
            )

        environment = metadata.get(
            "environment_provenance"
        )

        if environment is None:

            record(
                "environment",
                "NOT_AVAILABLE",
                "Environment provenance not supplied.",
            )

        elif not isinstance(environment, dict):

            record(
                "environment",
                "FAIL",
                "Invalid environment provenance.",
            )

        else:

            payload = {
                key: value
                for key, value in environment.items()
                if key not in (
                    "environment_hash",
                    "reproducibility_verified",
                )
            }

            try:

                environment_valid = (
                    valid_hash(
                        environment.get("environment_hash")
                    )
                    and config_hash(payload)
                    == environment["environment_hash"]
                    and environment.get(
                        "reproducibility_verified"
                    ) is False
                )

            except Exception:

                environment_valid = False

            record(
                "environment",
                "PASS" if environment_valid else "FAIL",
                "Environment checksum reconciled."
                if environment_valid else
                "Environment provenance checksum mismatch.",
            )

            dependencies = environment.get("dependencies")

            valid_dependencies = (
                isinstance(dependencies, dict)
                and bool(dependencies)
                and all(
                    isinstance(key, str)
                    and (
                        value is None
                        or (
                            isinstance(value, str)
                            and bool(value.strip())
                        )
                    )
                    for key, value in dependencies.items()
                )
            )

            record(
                "dependencies",
                "PASS" if valid_dependencies else "FAIL",
                "Dependency metadata structure validated."
                if valid_dependencies else
                "Invalid dependency metadata.",
            )

            dirty = environment.get(
                "git_working_tree_dirty"
            )

            record(
                "working_tree",
                "PASS" if dirty is False
                else "WARN" if dirty is True
                else "NOT_AVAILABLE",
                "Git working tree is clean."
                if dirty is False else
                "Git working tree contains changes."
                if dirty is True else
                "Git working-tree state unavailable.",
            )

        commit = metadata.get("git_commit")

        valid_commit = (
            isinstance(commit, str)
            and re.fullmatch(
                r"[0-9a-f]{40}|[0-9a-f]{64}",
                commit,
            ) is not None
        )

        record(
            "git_commit",
            "PASS" if valid_commit else "NOT_AVAILABLE",
            "Git commit recorded."
            if valid_commit else
            "Git commit unavailable or invalid.",
        )

        if isinstance(environment, dict):

            record(
                "git_consistency",
                "PASS"
                if valid_commit
                and environment.get("git_commit") == commit
                else "FAIL"
                if valid_commit
                else "NOT_AVAILABLE",
                "Git commit metadata reconciled."
                if valid_commit
                and environment.get("git_commit") == commit
                else "Git commit consistency not established.",
            )

        if manifest is None:

            record(
                "manifest",
                "NOT_AVAILABLE",
                "Output manifest not supplied.",
            )

        elif not isinstance(manifest, dict):

            record(
                "manifest",
                "FAIL",
                "Invalid output manifest structure.",
            )

        else:

            outputs = manifest.get("outputs")
            total = manifest.get("total_outputs")

            valid_manifest = (
                isinstance(outputs, list)
                and bool(outputs)
                and type(total) is int
                and total == len(outputs)
                and isinstance(metadata.get("run_id"), str)
                and bool(metadata["run_id"])
                and manifest.get("run_id")
                    == metadata["run_id"]
            )

            if not valid_manifest:

                record(
                    "manifest",
                    "FAIL",
                    "Manifest count or run identity mismatch.",
                )

            else:

                from pathlib import PurePosixPath
                from pathlib import PureWindowsPath

                seen = set()

                for index, output in enumerate(outputs):

                    valid_output = isinstance(output, dict)

                    if valid_output:

                        name = output.get("path")

                        valid_output = (
                            isinstance(name, str)
                            and bool(name.strip())
                            and output.get("status") == "PASS"
                            and valid_hash(output.get("sha256"))
                            and type(output.get("size_bytes"))
                                is int
                            and output["size_bytes"] >= 0
                        )

                    if valid_output:

                        normalized = name.replace(
                            chr(92), "/"
                        )

                        relative = PurePosixPath(normalized)
                        canonical = relative.as_posix().casefold()

                        valid_output = (
                            not relative.is_absolute()
                            and not PureWindowsPath(name).drive
                            and ".." not in relative.parts
                            and ":" not in normalized
                            and chr(0) not in normalized
                            and canonical != "."
                            and canonical not in seen
                        )

                        seen.add(canonical)

                    record(
                        f"output_{index}",
                        "PASS" if valid_output else "FAIL",
                        f"Output entry {index} validated."
                        if valid_output else
                        f"Output entry {index} invalid or duplicated.",
                    )

                record(
                    "manifest",
                    "PASS",
                    "Manifest count and run identity reconciled.",
                )

        return finish()


    CHECK_CATEGORIES = {
        "input": "input",
        "vegetation_index": "input",
        "model": "model",
        "class_distribution": "model",
        "spatial": "spatial",
        "nodata": "spatial",
        "temporal": "temporal",
        "transition": "change",
        "change_patch": "change",
        "cva": "change",
        "wetland": "wetland",
        "metrics": "metrics",
        "uncertainty": "uncertainty",
        "reproducibility": "reproducibility",
    }

    def configure_required_checks(self, analysis_mode):
        """Configure the required QA checks before validation begins."""

        if analysis_mode not in ("single_date", "two_date"):
            raise ValueError(
                f"Unsupported QA analysis mode: {analysis_mode}"
            )

        if self.flags or any(
            status != "NOT_RUN"
            for status in self.categories_status.values()
        ):
            raise RuntimeError(
                "Configure QA checks before validation begins."
            )

        if hasattr(self, "check_status"):
            raise RuntimeError(
                "Required QA checks are already configured."
            )

        required = set(self.CHECK_CATEGORIES)

        if analysis_mode == "single_date":

            required -= {
                "temporal",
                "transition",
                "change_patch",
                "cva",
                "metrics",
                "wetland",
            }

        self.qa_analysis_mode = analysis_mode

        self.required_checks = tuple(sorted(required))

        self.check_status = {
            name: "NOT_RUN"
            for name in self.required_checks
        }

    def record_check(self, check, status):
        """Record an individual required check outcome."""

        if not hasattr(self, "check_status"):
            raise RuntimeError(
                "Required QA checks are not configured."
            )

        if check not in self.check_status:
            raise ValueError(
                f"Unknown or inapplicable QA check: {check}"
            )

        if status not in self.VALID_STATUSES:
            raise ValueError(
                f"Invalid QA check status: {status}"
            )

        current = self.check_status[check]

        priority = {
            "NOT_RUN": 0,
            "PASS": 1,
            "NOT_AVAILABLE": 2,
            "WARN": 3,
            "FAIL": 4,
        }

        if priority[status] >= priority[current]:

            self.check_status[check] = status

    def aggregate_check_categories(self):
        """Aggregate required checks without hiding incomplete results."""

        if not hasattr(self, "check_status"):
            return dict(self.categories_status)

        categories = {}

        for category in sorted(self.VALID_CATEGORIES):

            statuses = [
                self.check_status[name]
                for name in self.required_checks
                if self.CHECK_CATEGORIES[name] == category
            ]

            legacy = self.categories_status[category]

            if legacy == "FAIL" or "FAIL" in statuses:

                result = "FAIL"

            elif legacy == "WARN" or "WARN" in statuses:

                result = "WARN"

            elif not statuses:

                result = "NOT_APPLICABLE"

            elif legacy == "NOT_AVAILABLE" and all(
                status == "PASS" for status in statuses
            ):

                result = "WARN"

            elif all(
                status == "PASS" for status in statuses
            ):

                result = "PASS"

            elif all(
                status == "NOT_RUN" for status in statuses
            ):

                result = "NOT_RUN"

            elif all(
                status == "NOT_AVAILABLE"
                for status in statuses
            ):

                result = "NOT_AVAILABLE"

            else:

                result = "WARN"

            categories[category] = result

        return categories


    CHECK_METHODS = {
        "input": "run_input_qa",
        "vegetation_index": "run_vegetation_index_qa",
        "model": "run_model_qa",
        "class_distribution": "run_class_distribution_qa",
        "spatial": "run_spatial_qa",
        "nodata": "run_nodata_propagation_qa",
        "temporal": "run_temporal_qa",
        "transition": "run_transition_qa",
        "change_patch": "run_change_patch_qa",
        "cva": "run_cva_qa",
        "wetland": "run_wetland_qa",
        "metrics": "run_metrics_qa",
        "uncertainty": "run_uncertainty_qa",
        "reproducibility": "run_reproducibility_qa",
    }

    def execute_check(self, check, *args, **kwargs):
        """Execute a required QA check and record its actual outcome."""

        if not hasattr(self, "check_status"):
            raise RuntimeError(
                "Configure required QA checks before execution."
            )

        if check not in self.check_status:
            raise ValueError(
                f"Unknown or inapplicable QA check: {check}"
            )

        method_name = self.CHECK_METHODS[check]
        method = getattr(self, method_name)

        category = self.CHECK_CATEGORIES[check]

        phases = {
            "input": "Input Ingestion",
            "model": "Model QA",
            "spatial": "Spatial Alignment",
            "temporal": "Temporal Alignment",
            "change": "Transition Mapping",
            "wetland": "Wetland Analysis",
            "metrics": "Metrics QA",
            "uncertainty": "Uncertainty QA",
            "reproducibility": "Reproducibility QA",
        }

        start = len(self.flags)

        def execution_failure(message):

            self.add_flag(
                "QA-EXECUTION",
                "HIGH",
                message,
                phases[category],
                "FAIL",
                "Inspect validation inputs and execution logs.",
            )

            self.record_check(check, "FAIL")

            return {
                "status": "FAIL",
                "execution_failed": True,
                "check": check,
                "message": message,
            }

        try:

            result = method(*args, **kwargs)

        except Exception as exc:

            return execution_failure(
                f"{check} QA execution failed: "
                f"{type(exc).__name__}."
            )

        if not isinstance(result, dict):

            return execution_failure(
                f"{check} QA returned an invalid result."
            )

        valid_outcomes = {
            "PASS",
            "WARN",
            "FAIL",
            "NOT_AVAILABLE",
        }

        returned_status = result.get("status")

        if (
            not isinstance(returned_status, str)
            or returned_status not in valid_outcomes
        ):

            return execution_failure(
                f"{check} QA returned an invalid or missing status."
            )

        outcomes = [returned_status]

        if check == "model":

            probability_status = result.get(
                "probability_status"
            )

            if probability_status in valid_outcomes:

                outcomes.append(probability_status)

            elif returned_status == "PASS":

                outcomes.append("NOT_AVAILABLE")

        for flag in self.flags[start:]:

            status = flag.get("status")

            if status in valid_outcomes:

                outcomes.append(status)

            else:

                return execution_failure(
                    f"{check} QA generated an invalid finding status."
                )

        if "FAIL" in outcomes:

            final_status = "FAIL"

        elif "WARN" in outcomes:

            final_status = "WARN"

        elif "NOT_AVAILABLE" in outcomes:

            final_status = "NOT_AVAILABLE"

        else:

            final_status = "PASS"

        self.record_check(check, final_status)

        # Associate findings with the method that produced them,
        # including findings belonging to another category.
        for flag in self.flags[start:]:

            flag["check"] = check

        return result
