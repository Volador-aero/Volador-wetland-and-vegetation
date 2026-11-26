from typing import Dict, Any, Optional

import numpy as np


def check_probability_calibration(
    predictions: Optional[Any] = None,
    ground_truth: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate calibration inputs without claiming verified calibration."""

    def result(status, message, count=0, validated=False):

        return {
            "status": status,
            "message": message,
            "data_count": int(count),
            "input_validation_passed": validated,
            "calibration_verified": False,
        }

    if config is not None and not isinstance(config, dict):

        return result(
            "FAIL",
            "Calibration configuration must be a dictionary.",
        )

    if predictions is None or ground_truth is None:

        return result(
            "NOT_AVAILABLE",
            "Probability calibration deferred to V2. "
            "Independent held-out dataset and ground-truth "
            "labels not available.",
        )

    try:

        preds = np.asarray(predictions)
        labels = np.asarray(ground_truth)

    except (TypeError, ValueError, OverflowError):

        return result(
            "FAIL",
            "Calibration inputs cannot be converted to arrays.",
        )

    if preds.ndim not in (1, 2):

        return result(
            "FAIL",
            "Predictions must have shape (N,) or (N, C).",
        )

    if labels.ndim != 1:

        return result(
            "FAIL",
            "Ground-truth labels must have shape (N,).",
        )

    if preds.shape[0] != labels.shape[0]:

        return result(
            "FAIL",
            "Calibration data shape mismatch between "
            "predictions and ground-truth labels.",
        )

    if preds.ndim == 2 and preds.shape[1] < 2:

        return result(
            "FAIL",
            "Multiclass probabilities require at least two classes.",
        )

    if preds.shape[0] == 0:

        return result(
            "NOT_AVAILABLE",
            "Calibration dataset is empty.",
        )

    if preds.dtype.kind not in "iuf":

        return result(
            "FAIL",
            "Predictions must contain real numeric probabilities.",
        )

    if labels.dtype.kind not in "iu":

        return result(
            "FAIL",
            "Ground-truth labels must contain integer class IDs.",
        )

    if not np.all(np.isfinite(preds)):

        return result(
            "FAIL",
            "Predictions contain NaN or infinite values.",
        )

    if np.any(preds < 0) or np.any(preds > 1):

        return result(
            "FAIL",
            "Predicted probabilities must be within [0, 1].",
        )

    if preds.ndim == 1:

        class_count = 2

    else:

        class_count = preds.shape[1]

        probability_sums = np.sum(
            preds,
            axis=1,
            dtype=np.float64,
        )

        if not np.allclose(
            probability_sums,
            1.0,
            rtol=0.0,
            atol=1e-4,
        ):

            return result(
                "FAIL",
                "Multiclass probabilities must sum to one.",
            )

    if np.any(labels < 0) or np.any(labels >= class_count):

        return result(
            "FAIL",
            "Ground-truth labels contain invalid class IDs.",
        )

    return result(
        "NOT_AVAILABLE",
        "Calibration inputs passed structural validation. "
        "Independent dataset provenance and actual probability "
        "calibration have not been verified.",
        count=preds.shape[0],
        validated=True,
    )
