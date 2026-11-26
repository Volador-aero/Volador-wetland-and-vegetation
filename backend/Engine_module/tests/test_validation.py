import pytest
import numpy as np
from src.gis_engine.validation.olofsson import estimate_area_with_reference_samples
from src.gis_engine.validation.calibration import check_probability_calibration
from src.gis_engine.validation.quality import QualityAssuranceEngine

def test_olofsson_no_samples():
    """Verify Olofsson returns NOT_AVAILABLE when no reference samples exist."""
    res = estimate_area_with_reference_samples(None)
    assert res["status"] == "NOT_AVAILABLE"
    assert res["model_derived_estimate"] is True
    assert res["independent_area_error_correction"] is False
    assert "Model-derived estimate" in res["message"]

def test_olofsson_with_samples():
    """Verify Olofsson calculation is correct with mock samples and strata."""
    # 4 classes: 0, 1, 2, 3
    # Strata definition area
    strata_def = {
        0: {"area": 40.0},
        1: {"area": 30.0},
        2: {"area": 20.0},
        3: {"area": 10.0}
    }
    # reference samples
    # format: list of dicts with mapped_class and reference_class
    ref_samples = []
    # Strata 0: 10 samples, 8 match class 0, 2 are class 1
    ref_samples.extend([{"mapped_class": 0, "reference_class": 0}] * 8)
    ref_samples.extend([{"mapped_class": 0, "reference_class": 1}] * 2)
    # Strata 1: 10 samples, 9 match class 1, 1 is class 2
    ref_samples.extend([{"mapped_class": 1, "reference_class": 1}] * 9)
    ref_samples.extend([{"mapped_class": 1, "reference_class": 2}] * 1)
    # Strata 2: 10 samples, 7 match class 2, 3 are class 0
    ref_samples.extend([{"mapped_class": 2, "reference_class": 2}] * 7)
    ref_samples.extend([{"mapped_class": 2, "reference_class": 0}] * 3)
    # Strata 3: 10 samples, 10 match class 3
    ref_samples.extend([{"mapped_class": 3, "reference_class": 3}] * 10)
    
    res = estimate_area_with_reference_samples(
        classification_map=None,
        reference_samples=ref_samples,
        strata_definition=strata_def
    )
    
    assert res["status"] == "WARN"
    assert res["model_derived_estimate"] is False
    assert res["independent_area_error_correction"] is False
    assert res["total_area"] == 100.0
    
    strata_res = res["strata_results"]
    assert len(strata_res) == 4
    assert strata_res[3]["adjusted_area_estimate"] == pytest.approx(10.0) # all 10 matched, area weight 10%
    assert strata_res[3]["confidence_interval_95_lower"] == 10.0
    assert strata_res[3]["confidence_interval_95_upper"] == 10.0
    assert res["calculation_completed"] is True
    assert res["independent_validation_verified"] is False

    expected_areas = {
        0: 38.0,
        1: 35.0,
        2: 17.0,
        3: 10.0,
    }

    for class_id, expected in expected_areas.items():

        item = strata_res[class_id]

        assert item["adjusted_area_estimate"] == pytest.approx(expected)

        assert item["confidence_interval_95_lower"] >= 0

        assert item["confidence_interval_95_upper"] <= 100.0

    assert sum(
        item["adjusted_area_estimate"]
        for item in strata_res.values()
    ) == pytest.approx(100.0)


def test_calibration_check():
    """Verify calibration input validation and evidence requirements."""

    # Missing independent validation data.
    res = check_probability_calibration(None)

    assert res["status"] == "NOT_AVAILABLE"
    assert "deferred" in res["message"].lower()
    assert res["calibration_verified"] is False

    # Valid deterministic multiclass probabilities.
    preds = np.tile(
        [0.1, 0.2, 0.3, 0.4],
        (10, 1),
    )

    gt = np.array(
        [0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
        dtype=int,
    )

    res_active = check_probability_calibration(
        preds, gt
    )

    assert res_active["status"] == "NOT_AVAILABLE"
    assert res_active["data_count"] == 10
    assert res_active["input_validation_passed"] is True
    assert res_active["calibration_verified"] is False

    # Invalid probabilities must be rejected.
    invalid_preds = np.full((10, 4), 0.5)

    res_invalid = check_probability_calibration(
        invalid_preds, gt
    )

    assert res_invalid["status"] == "FAIL"
    assert res_invalid["input_validation_passed"] is False
    assert res_invalid["calibration_verified"] is False

def test_qa_engine_cva_math():
    """Verify CVA math identity check triggers a CRITICAL flag on mismatch."""
    config = {
        "metrics": {"severity_weights": {"healthy": 0.0, "moderate": 0.35, "severe": 0.8, "background": 1.0}}
    }
    engine = QualityAssuranceEngine(config)
    
    # Correct CVA math
    cva_mag = np.array([5.0], dtype=np.float32)
    d_ndvi = np.array([3.0], dtype=np.float32)
    d_gndvi = np.array([4.0], dtype=np.float32)
    d_ndre = np.array([0.0], dtype=np.float32)
    valid_mask = np.array([True])
    
    res = engine.run_cva_qa(cva_mag, d_ndvi, d_gndvi, d_ndre, valid_mask)
    assert res["status"] == "PASS"
    assert len(engine.flags) == 0
    
    # Incorrect CVA math
    cva_mag_bad = np.array([10.0], dtype=np.float32) # should be 5
    res_bad = engine.run_cva_qa(cva_mag_bad, d_ndvi, d_gndvi, d_ndre, valid_mask)
    assert res_bad["status"] == "FAIL"
    assert len(engine.flags) == 1
    assert engine.flags[0]["code"] == "QA-020"
    assert engine.flags[0]["severity"] == "CRITICAL"


def test_qa_reproducibility_category_is_initialized():
    """QA-017 should update the reproducibility category consistently."""
    config = {}
    engine = QualityAssuranceEngine(config)

    engine.add_flag(
        code="QA-017",
        severity="CRITICAL",
        message="Reproducibility check failed.",
        phase="Reproducibility",
        status="FAIL",
        recommended_action="Review the generated outputs.",
    )

    report = engine.generate_qa_report()

    assert "reproducibility" in report["categories"]
    assert report["categories"]["reproducibility"] == "FAIL"
    assert report["overall_status"] == "FAIL"
    assert report["flags"][0]["code"] == "QA-017"



# QA STATUS LIFECYCLE REGRESSION TESTS

def test_qa_initial_categories_not_run():
    engine = QualityAssuranceEngine({})
    report = engine.generate_qa_report()

    assert len(report["categories"]) == 9
    assert all(
        status == "NOT_RUN"
        for status in report["categories"].values()
    )
    assert len(report["not_run_categories"]) == 9
    assert report["overall_status"] == "WARN"


def test_qa_all_categories_pass():
    engine = QualityAssuranceEngine({})

    for category in engine.categories_status:
        engine.mark_category(category, "PASS")

    report = engine.generate_qa_report()

    assert report["overall_status"] == "PASS"
    assert report["not_run_categories"] == []
    assert report["unavailable_categories"] == []


def test_qa_warning_status():
    engine = QualityAssuranceEngine({})

    for category in engine.categories_status:
        engine.mark_category(category, "PASS")

    engine.mark_category("spatial", "WARN")

    report = engine.generate_qa_report()

    assert report["categories"]["spatial"] == "WARN"
    assert report["overall_status"] == "WARN"


def test_qa_failure_status():
    engine = QualityAssuranceEngine({})

    engine.mark_category("model", "FAIL")

    report = engine.generate_qa_report()

    assert report["categories"]["model"] == "FAIL"
    assert report["overall_status"] == "FAIL"


def test_qa_unavailable_status():
    engine = QualityAssuranceEngine({})

    engine.mark_category("uncertainty", "NOT_AVAILABLE")

    report = engine.generate_qa_report()

    assert report["categories"]["uncertainty"] == "NOT_AVAILABLE"
    assert "uncertainty" in report["unavailable_categories"]
    assert report["overall_status"] == "WARN"


def test_qa_failure_cannot_be_overwritten():
    engine = QualityAssuranceEngine({})

    engine.mark_category("model", "FAIL")

    for status in ("PASS", "WARN", "NOT_RUN", "NOT_AVAILABLE"):
        engine.mark_category("model", status)

    assert engine.categories_status["model"] == "FAIL"
    assert engine.generate_qa_report()["overall_status"] == "FAIL"


def test_qa_warning_cannot_be_downgraded():
    engine = QualityAssuranceEngine({})

    engine.mark_category("spatial", "WARN")
    engine.mark_category("spatial", "PASS")

    assert engine.categories_status["spatial"] == "WARN"

    engine.mark_category("spatial", "FAIL")

    assert engine.categories_status["spatial"] == "FAIL"


def test_qa_invalid_category_rejected():
    engine = QualityAssuranceEngine({})

    with pytest.raises(ValueError):
        engine.mark_category("invalid_category", "PASS")


def test_qa_invalid_status_rejected():
    engine = QualityAssuranceEngine({})

    with pytest.raises(ValueError):
        engine.mark_category("input", "INVALID_STATUS")



# QA FLAG HANDLING REGRESSION TESTS

@pytest.mark.parametrize(
    "code,phase,expected_category",
    [
        ("QA-010", "Input Ingestion", "input"),
        ("QA-003", "Input Ingestion", "input"),
        ("QA-020", "Preprocessing", "input"),
        ("QA-001", "Model QA", "model"),
        ("QA-002", "Model QA", "model"),
        ("QA-020", "Model Inference", "model"),
        ("QA-002", "Spatial Alignment", "spatial"),
        ("QA-006", "Temporal Alignment", "temporal"),
        ("QA-020", "Transition Mapping", "change"),
        ("QA-015", "Transition Mapping", "change"),
        ("QA-020", "CVA Math Invariant", "change"),
        ("QA-003", "Wetland Analysis", "wetland"),
        ("QA-011", "Uncertainty QA", "uncertainty"),
        ("QA-020", "Metrics QA", "metrics"),
        ("QA-017", "Reproducibility QA", "reproducibility"),
    ],
)
def test_qa_flag_category_mapping(
    code, phase, expected_category
):
    engine = QualityAssuranceEngine({})

    engine.add_flag(
        code,
        "CRITICAL",
        "Test finding",
        phase,
        "FAIL",
        "Test action",
    )

    assert engine.flags[-1]["category"] == expected_category
    assert engine.categories_status[expected_category] == "FAIL"

    for category, status in engine.categories_status.items():
        if category != expected_category:
            assert status == "NOT_RUN"


def test_qa_high_severity_warning_remains_warning():
    engine = QualityAssuranceEngine({})

    engine.add_flag(
        "QA-003",
        "HIGH",
        "Invalid input values",
        "Input Ingestion",
        "WARN",
        "Review input values",
    )

    assert engine.categories_status["input"] == "WARN"
    assert engine.flags[-1]["severity"] == "HIGH"
    assert engine.generate_qa_report()["overall_status"] == "WARN"


def test_qa_legacy_warn_severity_normalized():
    engine = QualityAssuranceEngine({})

    engine.add_flag(
        "QA-009",
        "WARN",
        "Dominant transition",
        "Transition Mapping",
        "WARN",
        "Review transition map",
    )

    assert engine.flags[-1]["severity"] == "MEDIUM"
    assert engine.categories_status["change"] == "WARN"


def test_qa_unavailable_flag_not_failure():
    engine = QualityAssuranceEngine({})

    engine.add_flag(
        "QA-012",
        "INFO",
        "Reference samples unavailable",
        "Uncertainty QA",
        "NOT_AVAILABLE",
        "Collect reference samples",
    )

    assert engine.categories_status["uncertainty"] == "NOT_AVAILABLE"
    assert engine.generate_qa_report()["overall_status"] == "WARN"


def test_qa_invalid_severity_rejected():
    engine = QualityAssuranceEngine({})

    with pytest.raises(ValueError):
        engine.add_flag(
            "QA-001",
            "INVALID",
            "Test finding",
            "Input Ingestion",
            "FAIL",
            "Test action",
        )

    assert engine.flags == []


def test_qa_invalid_flag_status_rejected():
    engine = QualityAssuranceEngine({})

    with pytest.raises(ValueError):
        engine.add_flag(
            "QA-001",
            "HIGH",
            "Test finding",
            "Input Ingestion",
            "INVALID",
            "Test action",
        )

    assert engine.flags == []


def test_qa_unknown_category_mapping_rejected():
    engine = QualityAssuranceEngine({})

    with pytest.raises(ValueError):
        engine.add_flag(
            "QA-999",
            "HIGH",
            "Test finding",
            "Unknown Phase",
            "FAIL",
            "Test action",
        )

    assert engine.flags == []


def test_qa_code_fallback_mapping():
    engine = QualityAssuranceEngine({})

    engine.add_flag(
        "QA-017",
        "CRITICAL",
        "Reproducibility failure",
        "Custom Phase",
        "FAIL",
        "Verify output",
    )

    assert engine.flags[-1]["category"] == "reproducibility"
    assert engine.categories_status["reproducibility"] == "FAIL"


def test_qa_warning_flag_cannot_overwrite_failure():
    engine = QualityAssuranceEngine({})

    engine.add_flag(
        "QA-001",
        "CRITICAL",
        "Missing input",
        "Input Ingestion",
        "FAIL",
        "Provide input",
    )

    engine.add_flag(
        "QA-010",
        "MEDIUM",
        "Negative input values",
        "Input Ingestion",
        "WARN",
        "Review input",
    )

    assert engine.categories_status["input"] == "FAIL"
    assert engine.generate_qa_report()["overall_status"] == "FAIL"



# INPUT AND VEGETATION INDEX QA REGRESSION TESTS

def make_qa_bands():
    return {
        name: np.array([[0.1, 0.2], [0.3, 0.4]])
        for name in ("green", "red", "rededge", "nir")
    }


def make_qa_indices():
    return {
        name: np.array([[0.1, 0.2], [0.3, 0.4]])
        for name in ("ndvi", "gndvi", "ndre")
    }


@pytest.mark.parametrize(
    "array,expected",
    [
        (np.array([[1, 2], [3, 4]]), "PASS"),
        (np.empty((0, 0)), "FAIL"),
        (np.full((2, 2), np.nan), "FAIL"),
        (np.full((2, 2), np.inf), "FAIL"),
        (np.array([["a", "b"], ["c", "d"]]), "FAIL"),
        (np.ones((2, 2, 2)), "FAIL"),
        (np.ones((3, 3)), "FAIL"),
        (np.ones((2, 2)), "WARN"),
        (np.array([[0.1, np.nan], [0.3, 0.4]]), "WARN"),
        (np.array([[0.1, np.inf], [0.3, 0.4]]), "WARN"),
        (np.array([[-0.1, 0.2], [0.3, 0.4]]), "WARN"),
        (np.array([[1j, 2j], [3j, 4j]]), "FAIL"),
        ([[1, 2], [3, 4]], "FAIL"),
    ],
)
def test_input_qa_array_edge_cases(array, expected):
    engine = QualityAssuranceEngine({})
    t1 = make_qa_bands()
    t1["red"] = array

    result = engine.run_input_qa(
        t1, make_qa_bands(), None, None
    )

    assert result["status"] == expected

    if expected in ("FAIL", "WARN"):
        assert engine.categories_status["input"] == expected


@pytest.mark.parametrize(
    "band",
    ["green", "red", "rededge", "nir"],
)
def test_input_qa_missing_required_band(band):
    engine = QualityAssuranceEngine({})
    t1 = make_qa_bands()
    del t1[band]

    result = engine.run_input_qa(
        t1, make_qa_bands(), None, None
    )

    assert result["status"] == "FAIL"
    assert any(
        flag["code"] == "QA-001"
        for flag in engine.flags
    )


def test_input_qa_missing_dataset():
    engine = QualityAssuranceEngine({})

    result = engine.run_input_qa(
        None, make_qa_bands(), None, None
    )

    assert result["status"] == "FAIL"


def test_input_qa_finite_only_statistics():
    import json

    engine = QualityAssuranceEngine({})
    t1 = make_qa_bands()

    t1["red"] = np.array(
        [[0.1, np.nan], [0.3, np.inf]]
    )

    result = engine.run_input_qa(
        t1, make_qa_bands(), None, None
    )

    stats = result["bands"]["T1_red"]

    assert result["status"] == "WARN"
    assert stats["valid_pixels"] == 2
    assert stats["invalid_pixels"] == 2
    assert stats["min"] == pytest.approx(0.1)
    assert stats["max"] == pytest.approx(0.3)
    assert stats["mean"] == pytest.approx(0.2)

    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "mask,expected",
    [
        (np.ones((2, 2), dtype=bool), "PASS"),
        (np.ones((3, 3), dtype=bool), "FAIL"),
        (np.zeros((2, 2), dtype=bool), "NOT_AVAILABLE"),
        (np.full((2, 2), 255), "FAIL"),
        (np.ones((2, 2), dtype=np.uint8), "PASS"),
        (np.empty((0, 0)), "FAIL"),
        (np.ones((2, 2, 2)), "FAIL"),
        (None, "FAIL"),
        (np.full((2, 2), np.nan), "FAIL"),
    ],
)
def test_vegetation_qa_mask_edge_cases(mask, expected):
    engine = QualityAssuranceEngine({})

    result = engine.run_vegetation_index_qa(
        make_qa_indices(),
        make_qa_indices(),
        mask,
    )

    assert result["status"] == expected


@pytest.mark.parametrize(
    "array,expected",
    [
        (np.full((2, 2), np.nan), "FAIL"),
        (np.full((2, 2), np.inf), "FAIL"),
        (np.array([[0.1, np.nan], [0.3, 0.4]]), "WARN"),
        (np.array([[0.1, np.inf], [0.3, 0.4]]), "WARN"),
        (np.array([[0.1, 1.5], [0.3, 0.4]]), "WARN"),
        (np.array([[-1.5, 0.2], [0.3, 0.4]]), "WARN"),
        (np.ones((3, 3)), "FAIL"),
        (np.array([["a", "b"], ["c", "d"]]), "FAIL"),
    ],
)
def test_vegetation_qa_index_edge_cases(array, expected):
    engine = QualityAssuranceEngine({})
    t1 = make_qa_indices()
    t1["ndvi"] = array

    result = engine.run_vegetation_index_qa(
        t1,
        make_qa_indices(),
        np.ones((2, 2), dtype=bool),
    )

    assert result["status"] == expected


@pytest.mark.parametrize(
    "index",
    ["ndvi", "gndvi", "ndre"],
)
def test_vegetation_qa_missing_index(index):
    engine = QualityAssuranceEngine({})
    t1 = make_qa_indices()
    del t1[index]

    result = engine.run_vegetation_index_qa(
        t1,
        make_qa_indices(),
        np.ones((2, 2), dtype=bool),
    )

    assert result["status"] == "FAIL"


def test_vegetation_qa_ignores_pixels_outside_mask():
    engine = QualityAssuranceEngine({})
    t1 = make_qa_indices()

    t1["ndvi"] = np.array(
        [[0.1, np.inf], [0.3, 0.4]]
    )

    mask = np.array(
        [[True, False], [True, True]]
    )

    result = engine.run_vegetation_index_qa(
        t1, make_qa_indices(), mask
    )

    assert result["status"] == "PASS"
    assert result["T1"]["ndvi"]["valid_pixels"] == 3
    assert result["T1"]["ndvi"]["invalid_pixels"] == 0


def test_vegetation_qa_json_serialization():
    import json

    engine = QualityAssuranceEngine({})
    t1 = make_qa_indices()

    t1["ndvi"] = np.array(
        [[0.1, np.nan], [0.3, np.inf]]
    )

    result = engine.run_vegetation_index_qa(
        t1,
        make_qa_indices(),
        np.ones((2, 2), dtype=bool),
    )

    assert result["status"] == "WARN"
    assert result["T1"]["ndvi"]["valid_pixels"] == 2
    assert result["T1"]["ndvi"]["invalid_pixels"] == 2

    json.dumps(result, allow_nan=False)



# MODEL AND CLASS DISTRIBUTION QA REGRESSION TESTS

@pytest.fixture
def qa_mock_onnx(monkeypatch, tmp_path):
    from unittest.mock import MagicMock
    import sys

    module = sys.modules[QualityAssuranceEngine.__module__]

    model_path = tmp_path / "test_model.onnx"
    model_path.touch()

    session = MagicMock()

    input_details = MagicMock()
    input_details.name = "input"
    input_details.shape = [1, 5, 256, 256]

    output_details = MagicMock()
    output_details.name = "output"
    output_details.shape = [1, 4, 256, 256]

    session.get_inputs.return_value = [input_details]
    session.get_outputs.return_value = [output_details]
    session.get_providers.return_value = ["CPUExecutionProvider"]

    monkeypatch.setattr(
        module.ort,
        "InferenceSession",
        MagicMock(return_value=session),
    )

    return str(model_path), session


@pytest.mark.parametrize(
    "probabilities,expected",
    [
        (np.full((4, 2, 2), 0.25), "PASS"),
        (None, "NOT_AVAILABLE"),
        (np.empty((4, 0, 0)), "FAIL"),
        (np.ones((3, 2, 2)), "FAIL"),
        (np.ones((4, 2)), "FAIL"),
        (np.full((4, 2, 2), np.nan), "FAIL"),
        (np.full((4, 2, 2), np.inf), "FAIL"),
        (np.full((4, 2, 2), -0.25), "FAIL"),
        (np.full((4, 2, 2), 1.5), "FAIL"),
        (np.full((4, 2, 2), 0.5), "FAIL"),
        (np.ones((1, 4, 2, 2)), "FAIL"),
        (np.full((4, 2, 2), "invalid"), "FAIL"),
    ],
)
def test_model_qa_probability_edge_cases(
    qa_mock_onnx, probabilities, expected
):
    model_path, _ = qa_mock_onnx
    engine = QualityAssuranceEngine({})

    result = engine.run_model_qa(
        model_path, None, probabilities
    )

    assert result["status"] == expected


@pytest.mark.parametrize(
    "model_path",
    [None, "", 123],
)
def test_model_qa_invalid_path(model_path):
    engine = QualityAssuranceEngine({})

    result = engine.run_model_qa(
        model_path, None, None
    )

    assert result["status"] == "FAIL"


def test_model_qa_missing_file(tmp_path):
    engine = QualityAssuranceEngine({})

    result = engine.run_model_qa(
        str(tmp_path / "missing.onnx"),
        None,
        None,
    )

    assert result["status"] == "FAIL"
    assert engine.categories_status["model"] == "FAIL"


def test_model_qa_corrupted_model(
    qa_mock_onnx, monkeypatch
):
    import sys
    from unittest.mock import MagicMock

    model_path, _ = qa_mock_onnx
    module = sys.modules[QualityAssuranceEngine.__module__]

    monkeypatch.setattr(
        module.ort,
        "InferenceSession",
        MagicMock(
            side_effect=RuntimeError("Invalid ONNX model")
        ),
    )

    engine = QualityAssuranceEngine({})

    result = engine.run_model_qa(
        model_path, None, None
    )

    assert result["status"] == "FAIL"
    assert "error" in result


@pytest.mark.parametrize(
    "shape",
    [
        [1, 3, 256, 256],
        [1, 6, 256, 256],
        [1, 5],
        [],
    ],
)
def test_model_qa_invalid_input_shape(
    qa_mock_onnx, shape
):
    model_path, session = qa_mock_onnx
    session.get_inputs()[0].shape = shape

    engine = QualityAssuranceEngine({})

    result = engine.run_model_qa(
        model_path,
        None,
        np.full((4, 2, 2), 0.25),
    )

    assert result["status"] == "FAIL"


@pytest.mark.parametrize(
    "logits,expected",
    [
        (np.zeros((4, 2, 2)), "PASS"),
        (np.ones((3, 2, 2)), "FAIL"),
        (np.ones((4, 3, 3)), "FAIL"),
        (np.full((4, 2, 2), np.nan), "FAIL"),
        (np.full((4, 2, 2), np.inf), "FAIL"),
        (np.empty((4, 0, 0)), "FAIL"),
    ],
)
def test_model_qa_logits(
    qa_mock_onnx, logits, expected
):
    model_path, _ = qa_mock_onnx
    engine = QualityAssuranceEngine({})

    result = engine.run_model_qa(
        model_path,
        logits,
        np.full((4, 2, 2), 0.25),
    )

    assert result["status"] == expected


def test_model_qa_confidence_statistics(qa_mock_onnx):
    import json

    model_path, _ = qa_mock_onnx
    engine = QualityAssuranceEngine({})

    probabilities = np.array(
        [0.7, 0.1, 0.1, 0.1]
    ).reshape(4, 1, 1)

    result = engine.run_model_qa(
        model_path, None, probabilities
    )

    assert result["status"] == "PASS"
    assert result["probability_status"] == "PASS"
    assert result["calibration_status"] == "NOT_AVAILABLE"

    assert result["confidence"]["mean_max_prob"] == pytest.approx(0.7)
    assert result["confidence"]["median_max_prob"] == pytest.approx(0.7)

    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "health,expected",
    [
        (np.array([[0, 1], [2, 3]]), "PASS"),
        (np.full((2, 2), 4), "FAIL"),
        (np.full((2, 2), -1), "FAIL"),
        (np.full((2, 2), 255), "FAIL"),
        (np.full((2, 2), np.nan), "FAIL"),
        (np.full((2, 2), np.inf), "FAIL"),
        (np.full((2, 2), 1.5), "FAIL"),
        (np.ones((3, 3)), "FAIL"),
        (np.ones((2, 2)), "WARN"),
        (np.ones((2, 2, 2)), "FAIL"),
        (np.full((2, 2), "invalid"), "FAIL"),
        (None, "FAIL"),
    ],
)
def test_class_distribution_qa_edge_cases(
    health, expected
):
    engine = QualityAssuranceEngine({})

    result = engine.run_class_distribution_qa(
        health,
        np.array([[0, 1], [2, 3]]),
        np.ones((2, 2), dtype=bool),
    )

    assert result["status"] == expected


@pytest.mark.parametrize(
    "mask,expected",
    [
        (np.ones((2, 2), dtype=bool), "PASS"),
        (np.ones((2, 2), dtype=np.uint8), "PASS"),
        (np.zeros((2, 2), dtype=bool), "NOT_AVAILABLE"),
        (np.full((2, 2), 255), "FAIL"),
        (np.ones((3, 3)), "FAIL"),
        (np.empty((0, 0)), "FAIL"),
        (np.full((2, 2), np.nan), "FAIL"),
        (None, "FAIL"),
    ],
)
def test_class_distribution_qa_masks(mask, expected):
    engine = QualityAssuranceEngine({})
    health = np.array([[0, 1], [2, 3]])

    result = engine.run_class_distribution_qa(
        health, health, mask
    )

    assert result["status"] == expected


def test_class_distribution_percentages():
    engine = QualityAssuranceEngine({})
    health = np.array([[0, 1], [2, 3]])

    result = engine.run_class_distribution_qa(
        health,
        health,
        np.ones((2, 2), dtype=bool),
    )

    assert result["status"] == "PASS"
    assert result["T1"] == {
        0: 25.0,
        1: 25.0,
        2: 25.0,
        3: 25.0,
    }

    assert sum(result["T1"].values()) == pytest.approx(100.0)


def test_class_distribution_ignores_nodata_outside_mask():
    engine = QualityAssuranceEngine({})

    health = np.array([[0, 255], [2, 3]])
    mask = np.array([[True, False], [True, True]])

    result = engine.run_class_distribution_qa(
        health, health, mask
    )

    assert result["status"] == "PASS"
    assert 255 not in result["T1"]
    assert sum(result["T1"].values()) == pytest.approx(100.0)



# SPATIAL AND NODATA QA REGRESSION TESTS

def make_qa_metadata(
    crs="EPSG:32644",
    width=2,
    height=2,
    x=500000,
    y=2000000,
    pixel=10,
):
    from types import SimpleNamespace
    from rasterio.transform import from_origin

    return SimpleNamespace(
        crs=crs,
        width=width,
        height=height,
        transform=from_origin(x, y, pixel, pixel),
        resolution=(pixel, pixel),
        bounds=(
            x,
            y - height * pixel,
            x + width * pixel,
            y,
        ),
    )


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, "PASS"),
        ({"crs": None}, "FAIL"),
        ({"crs": "invalid"}, "FAIL"),
        ({"crs": "EPSG:32643"}, "FAIL"),
        ({"width": 3}, "FAIL"),
        ({"height": 3}, "FAIL"),
        ({"width": 0}, "FAIL"),
        ({"height": -1}, "FAIL"),
        ({"pixel": 20}, "FAIL"),
        ({"x": 500010}, "FAIL"),
        ({"y": 2000010}, "FAIL"),
        ({"x": 500000.0000001}, "PASS"),
        ({"pixel": 0}, "FAIL"),
    ],
)
def test_spatial_qa_alignment_cases(kwargs, expected):
    engine = QualityAssuranceEngine({})

    result = engine.run_spatial_qa({
        "T1": make_qa_metadata(),
        "T2": make_qa_metadata(**kwargs),
    })

    assert result["status"] == expected


@pytest.mark.parametrize(
    "case,expected",
    [
        ("empty", "NOT_AVAILABLE"),
        ("single", "NOT_AVAILABLE"),
        ("missing", "NOT_AVAILABLE"),
        ("missing_reference", "NOT_AVAILABLE"),
        ("all_missing", "NOT_AVAILABLE"),
        ("invalid_type", "FAIL"),
    ],
)
def test_spatial_qa_metadata_availability(case, expected):
    engine = QualityAssuranceEngine({})

    cases = {
        "empty": {},
        "single": {"T1": make_qa_metadata()},
        "missing": {
            "T1": make_qa_metadata(),
            "T2": None,
        },
        "missing_reference": {
            "T1": None,
            "T2": make_qa_metadata(),
        },
        "all_missing": {"T1": None, "T2": None},
        "invalid_type": [],
    }

    result = engine.run_spatial_qa(cases[case])

    assert result["status"] == expected


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("bounds", (0, 0, 100, 100)),
        ("bounds", (0, 0, 0, 0)),
        ("bounds", (0, 0, np.inf, 100)),
        ("transform", (0, 0, 0, 0, 0, 0)),
        ("transform", (1, 2, 3)),
        ("transform", (np.nan, 0, 0, 0, -10, 0)),
        ("resolution", (0, 10)),
        ("resolution", (-10, 10)),
        ("resolution", (np.inf, 10)),
        ("resolution", (20, 20)),
        ("width", True),
        ("height", 2.5),
    ],
)
def test_spatial_qa_invalid_metadata(attribute, value):
    engine = QualityAssuranceEngine({})

    bad = make_qa_metadata()
    setattr(bad, attribute, value)

    result = engine.run_spatial_qa({
        "T1": make_qa_metadata(),
        "T2": bad,
    })

    assert result["status"] == "FAIL"
    assert engine.categories_status["spatial"] == "FAIL"


def test_spatial_qa_json_serialization():
    import json

    engine = QualityAssuranceEngine({})

    result = engine.run_spatial_qa({
        "T1": make_qa_metadata(),
        "T2": make_qa_metadata(),
    })

    assert result["status"] == "PASS"
    assert result["reference"] == "T1"
    assert len(result["rasters"]) == 2

    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "output,expected",
    [
        (np.array([[0, 1], [2, 255]]), "PASS"),
        (np.array([[0.0, 0.5], [0.7, np.nan]]), "PASS"),
        (np.array([[0, 1], [2, 0]]), "FAIL"),
        (np.array([[0, 1], [2, 15]]), "FAIL"),
        (np.array([[0, 1], [2, -1]]), "FAIL"),
        (np.array([[0.0, 0.5], [0.7, 0.9]]), "FAIL"),
        (np.array([[0.0, 0.5], [0.7, 0.0]]), "FAIL"),
        (np.array([[0.0, 0.5], [0.7, np.inf]]), "FAIL"),
        (np.full((2, 2), 255), "FAIL"),
        (np.full((2, 2), np.nan), "FAIL"),
        (np.ones((3, 3)), "FAIL"),
        (np.ones((2, 2, 2)), "FAIL"),
        (np.full((2, 2), "invalid"), "FAIL"),
        (None, "NOT_AVAILABLE"),
    ],
)
def test_nodata_qa_output_cases(output, expected):
    engine = QualityAssuranceEngine({})

    health = np.array([[0, 1], [2, 255]])
    mask = np.array([[True, True], [True, False]])

    result = engine.run_nodata_propagation_qa(
        health,
        health,
        {"output": output},
        mask,
    )

    assert result["status"] == expected


@pytest.mark.parametrize(
    "mask,expected",
    [
        (np.array([[True, True], [True, False]]), "PASS"),
        (np.array([[1, 1], [1, 0]]), "PASS"),
        (np.full((2, 2), 255), "FAIL"),
        (np.full((2, 2), np.nan), "FAIL"),
        (np.ones((3, 3)), "FAIL"),
        (np.ones((2, 2, 2)), "FAIL"),
        (np.empty((0, 0)), "FAIL"),
        (None, "FAIL"),
    ],
)
def test_nodata_qa_mask_cases(mask, expected):
    engine = QualityAssuranceEngine({})
    health = np.array([[0, 1], [2, 255]])

    result = engine.run_nodata_propagation_qa(
        health,
        health,
        {"health": health},
        mask,
    )

    assert result["status"] == expected


@pytest.mark.parametrize(
    "health",
    [
        np.array([[255, 1], [2, 255]]),
        np.array([[4, 1], [2, 255]]),
        np.array([[np.nan, 1], [2, 255]]),
        np.array([[np.inf, 1], [2, 255]]),
        np.array([[1.5, 1], [2, 255]]),
        np.ones((3, 3)),
        None,
    ],
)
def test_nodata_qa_invalid_health_maps(health):
    engine = QualityAssuranceEngine({})

    valid_health = np.array([[0, 1], [2, 255]])
    mask = np.array([[True, True], [True, False]])

    result = engine.run_nodata_propagation_qa(
        health,
        valid_health,
        {"health": valid_health},
        mask,
    )

    assert result["status"] == "FAIL"


def test_nodata_qa_empty_outputs():
    engine = QualityAssuranceEngine({})
    health = np.array([[0, 1], [2, 255]])
    mask = np.array([[True, True], [True, False]])

    result = engine.run_nodata_propagation_qa(
        health, health, {}, mask
    )

    assert result["status"] == "NOT_AVAILABLE"


def test_nodata_qa_empty_valid_region():
    engine = QualityAssuranceEngine({})
    health = np.full((2, 2), 255)
    mask = np.zeros((2, 2), dtype=bool)

    result = engine.run_nodata_propagation_qa(
        health,
        health,
        {"health": health},
        mask,
    )

    assert result["status"] == "NOT_AVAILABLE"
    assert result["valid_pixels"] == 0
    assert result["nodata_pixels"] == 4


def test_nodata_qa_all_pixels_valid():
    engine = QualityAssuranceEngine({})
    health = np.array([[0, 1], [2, 3]])
    mask = np.ones((2, 2), dtype=bool)

    result = engine.run_nodata_propagation_qa(
        health,
        health,
        {"health": health},
        mask,
    )

    assert result["status"] == "PASS"
    assert result["valid_pixels"] == 4
    assert result["nodata_pixels"] == 0


def test_nodata_qa_checks_inside_when_all_pixels_valid():
    engine = QualityAssuranceEngine({})
    health = np.array([[0, 1], [2, 3]])
    mask = np.ones((2, 2), dtype=bool)

    result = engine.run_nodata_propagation_qa(
        health,
        health,
        {"health": np.full((2, 2), 255)},
        mask,
    )

    assert result["status"] == "FAIL"


def test_nodata_qa_json_serialization():
    import json

    engine = QualityAssuranceEngine({})
    health = np.array([[0, 1], [2, 255]])
    mask = np.array([[True, True], [True, False]])

    result = engine.run_nodata_propagation_qa(
        health,
        health,
        {
            "health": health,
            "cva": np.array([[0.0, 0.5], [0.7, np.nan]]),
        },
        mask,
    )

    assert result["status"] == "PASS"
    assert result["valid_pixels"] == 3
    assert result["nodata_pixels"] == 1

    assert result["outputs"]["health"]["invalid_outside_pixels"] == 0
    assert result["outputs"]["cva"]["invalid_inside_pixels"] == 0

    json.dumps(result, allow_nan=False)


def test_nodata_qa_failure_not_overwritten():
    engine = QualityAssuranceEngine({})
    health = np.array([[0, 1], [2, 255]])
    mask = np.array([[True, True], [True, False]])

    result = engine.run_nodata_propagation_qa(
        health,
        health,
        {
            "invalid": np.zeros((2, 2), dtype=int),
            "missing": None,
            "valid": health,
        },
        mask,
    )

    assert result["status"] == "FAIL"
    assert engine.categories_status["spatial"] == "FAIL"



# TEMPORAL QA REGRESSION TESTS

@pytest.mark.parametrize(
    "t1,t2,expected",
    [
        ("2023-08-14", "2024-08-14", "PASS"),
        ("2024-08-14", "2024-08-15", "PASS"),
        ("2024-08-14", "2023-08-14", "FAIL"),
        ("2024-08-14", "2024-08-14", "FAIL"),
        ("14-08-2023", "2024-08-14", "FAIL"),
        ("2023-13-14", "2024-08-14", "FAIL"),
        ("2023-08-32", "2024-08-14", "FAIL"),
        (None, "2024-08-14", "NOT_AVAILABLE"),
        ("2023-08-14", None, "NOT_AVAILABLE"),
        ("", "2024-08-14", "NOT_AVAILABLE"),
        (20230814, "2024-08-14", "FAIL"),
        ("2024-02-29", "2025-02-28", "PASS"),
        ("2023-02-29", "2024-02-28", "FAIL"),
        ("2023-12-15", "2024-01-15", "PASS"),
        ("2023-11-15", "2024-02-15", "WARN"),
        ("2023-01-15", "2023-07-15", "WARN"),
        ("2023-08-15", "2023-10-15", "PASS"),
        ("2023-08-15", "2023-11-15", "WARN"),
        ("2023-8-14", "2024-08-14", "FAIL"),
        ("   ", "2024-08-14", "NOT_AVAILABLE"),
        (None, None, "NOT_AVAILABLE"),
        ("2023-08-14", "invalid", "FAIL"),
    ],
)
def test_temporal_qa_date_cases(t1, t2, expected):
    import json

    engine = QualityAssuranceEngine({})
    mask = np.ones((2, 2), dtype=bool)

    result = engine.run_temporal_qa(
        t1, t2, None, None, mask
    )

    assert result["status"] == expected

    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "mask,expected",
    [
        (np.ones((2, 2), dtype=bool), "PASS"),
        (np.array([[1, 0], [1, 0]]), "PASS"),
        (np.zeros((2, 2), dtype=bool), "FAIL"),
        (None, "NOT_AVAILABLE"),
        (np.empty((0, 0)), "FAIL"),
        (np.ones((2, 2, 2)), "FAIL"),
        (np.full((2, 2), 255), "FAIL"),
        (np.full((2, 2), np.nan), "FAIL"),
        (np.full((2, 2), np.inf), "FAIL"),
        (np.full((2, 2), -1), "FAIL"),
        (np.full((2, 2), 0.5), "FAIL"),
        (np.full((2, 2), "invalid"), "FAIL"),
        ([[1, 0], [1, 0]], "FAIL"),
        (np.ones((2, 2), dtype=np.uint8), "PASS"),
        (np.ones((2, 2), dtype=np.complex128), "FAIL"),
    ],
)
def test_temporal_qa_overlap_cases(mask, expected):
    import json

    engine = QualityAssuranceEngine({})

    result = engine.run_temporal_qa(
        "2023-08-14",
        "2024-08-14",
        None,
        None,
        mask,
    )

    assert result["status"] == expected

    json.dumps(result, allow_nan=False)


def test_temporal_qa_calculation_accuracy():
    engine = QualityAssuranceEngine({})

    result = engine.run_temporal_qa(
        "2023-08-14",
        "2024-08-14",
        None,
        None,
        np.array([[1, 0], [1, 0]]),
    )

    assert result["status"] == "PASS"
    assert result["date_difference_days"] == 366
    assert result["valid_overlap_pixels"] == 2
    assert result["total_grid_pixels"] == 4
    assert result["grid_overlap_percentage"] == 50.0
    assert result["seasonal_month_difference"] == 0


def test_temporal_qa_full_overlap_percentage():
    engine = QualityAssuranceEngine({})

    result = engine.run_temporal_qa(
        "2023-08-14",
        "2024-08-14",
        None,
        None,
        np.ones((3, 3), dtype=bool),
    )

    assert result["valid_overlap_pixels"] == 9
    assert result["total_grid_pixels"] == 9
    assert result["grid_overlap_percentage"] == 100.0


def test_temporal_qa_zero_overlap_percentage():
    engine = QualityAssuranceEngine({})

    result = engine.run_temporal_qa(
        "2023-08-14",
        "2024-08-14",
        None,
        None,
        np.zeros((2, 2), dtype=bool),
    )

    assert result["status"] == "FAIL"
    assert result["valid_overlap_pixels"] == 0
    assert result["grid_overlap_percentage"] == 0.0
    assert engine.categories_status["spatial"] == "FAIL"


def test_temporal_qa_cross_year_seasonality():
    engine = QualityAssuranceEngine({})

    result = engine.run_temporal_qa(
        "2023-12-15",
        "2024-01-15",
        None,
        None,
        np.ones((2, 2), dtype=bool),
    )

    assert result["status"] == "PASS"
    assert result["seasonal_month_difference"] == 1
    assert result["phenology_warning"] == "NO"


def test_temporal_qa_seasonal_warning_category():
    engine = QualityAssuranceEngine({})

    result = engine.run_temporal_qa(
        "2023-11-15",
        "2024-02-15",
        None,
        None,
        np.ones((2, 2), dtype=bool),
    )

    assert result["status"] == "WARN"
    assert result["seasonal_month_difference"] == 3
    assert result["phenology_warning"] == "YES"
    assert engine.categories_status["temporal"] == "WARN"

    assert any(
        flag["code"] == "QA-007"
        and flag["category"] == "temporal"
        for flag in engine.flags
    )


def test_temporal_qa_invalid_date_category():
    engine = QualityAssuranceEngine({})

    result = engine.run_temporal_qa(
        "invalid",
        "2024-08-14",
        None,
        None,
        np.ones((2, 2), dtype=bool),
    )

    assert result["status"] == "FAIL"
    assert engine.categories_status["temporal"] == "FAIL"

    assert any(
        flag["code"] == "QA-006"
        and flag["category"] == "temporal"
        for flag in engine.flags
    )


def test_temporal_qa_failure_not_overwritten():
    engine = QualityAssuranceEngine({})

    result = engine.run_temporal_qa(
        "invalid",
        "2024-08-14",
        None,
        None,
        None,
    )

    assert result["status"] == "FAIL"
    assert engine.categories_status["temporal"] == "FAIL"
    assert engine.categories_status["spatial"] == "NOT_AVAILABLE"


def test_temporal_qa_warning_not_overwritten():
    engine = QualityAssuranceEngine({})

    result = engine.run_temporal_qa(
        "2023-01-15",
        "2023-07-15",
        None,
        None,
        None,
    )

    assert result["status"] == "WARN"
    assert engine.categories_status["temporal"] == "WARN"
    assert engine.categories_status["spatial"] == "NOT_AVAILABLE"



# TRANSITION AND CHANGE PATCH QA REGRESSION TESTS

@pytest.mark.parametrize(
    "case,expected",
    [
        ("valid", "PASS"),
        ("wrong_formula", "FAIL"),
        ("invalid_id", "FAIL"),
        ("negative_id", "FAIL"),
        ("fractional_id", "FAIL"),
        ("nan_id", "FAIL"),
        ("inf_id", "FAIL"),
        ("nodata_inside", "FAIL"),
        ("invalid_t1", "FAIL"),
        ("invalid_t2", "FAIL"),
        ("fractional_health", "FAIL"),
        ("wrong_shape", "FAIL"),
        ("missing_transition", "FAIL"),
        ("missing_mask", "FAIL"),
        ("invalid_mask", "FAIL"),
        ("integer_mask", "PASS"),
        ("partial_overlap", "PASS"),
        ("invalid_outside", "FAIL"),
        ("empty_overlap", "NOT_AVAILABLE"),
        ("dominant", "WARN"),
        ("nan_health", "FAIL"),
        ("string_transition", "FAIL"),
    ],
)
def test_transition_qa_cases(case, expected):
    import json

    engine = QualityAssuranceEngine({})

    t1 = np.array([[0, 1], [2, 3]])
    t2 = np.array([[1, 2], [3, 0]])

    trans = t1 * 4 + t2
    mask = np.ones((2, 2), dtype=bool)

    if case == "wrong_formula":
        trans = trans + 1
    elif case == "invalid_id":
        trans = np.full((2, 2), 16)
    elif case == "negative_id":
        trans = np.full((2, 2), -1)
    elif case == "fractional_id":
        trans = trans + 0.5
    elif case == "nan_id":
        trans = np.full((2, 2), np.nan)
    elif case == "inf_id":
        trans = np.full((2, 2), np.inf)
    elif case == "nodata_inside":
        trans = np.full((2, 2), 255)
    elif case == "invalid_t1":
        t1 = np.full((2, 2), 4)
    elif case == "invalid_t2":
        t2 = np.full((2, 2), 255)
    elif case == "fractional_health":
        t1 = t1 + 0.5
    elif case == "wrong_shape":
        trans = np.ones((3, 3))
    elif case == "missing_transition":
        trans = None
    elif case == "missing_mask":
        mask = None
    elif case == "invalid_mask":
        mask = np.full((2, 2), 255)
    elif case == "integer_mask":
        mask = mask.astype(int)
    elif case == "partial_overlap":
        mask[1, 1] = False
        trans[1, 1] = 255
    elif case == "invalid_outside":
        mask[1, 1] = False
    elif case == "empty_overlap":
        mask = np.zeros((2, 2), dtype=bool)
        trans = np.full((2, 2), 255)
    elif case == "dominant":
        t1 = np.zeros((2, 2))
        t2 = np.zeros((2, 2))
        trans = np.zeros((2, 2))
    elif case == "nan_health":
        t1 = np.full((2, 2), np.nan)
    elif case == "string_transition":
        trans = np.full((2, 2), "invalid")

    result = engine.run_transition_qa(
        trans, t1, t2, mask
    )

    assert result["status"] == expected

    json.dumps(result, allow_nan=False)


def test_transition_qa_pixel_reconciliation():
    engine = QualityAssuranceEngine({})

    t1 = np.array([[0, 1], [2, 3]])
    t2 = np.array([[1, 2], [3, 0]])
    trans = t1 * 4 + t2

    result = engine.run_transition_qa(
        trans,
        t1,
        t2,
        np.ones((2, 2), dtype=bool),
    )

    assert result["status"] == "PASS"
    assert result["valid_overlap_pixels"] == 4
    assert result["reported_transition_pixels"] == 4
    assert result["formula_mismatch_pixels"] == 0

    distribution = result["transition_distribution"]

    assert sum(
        item["pixel_count"]
        for item in distribution.values()
    ) == 4

    assert sum(
        item["percentage"]
        for item in distribution.values()
    ) == pytest.approx(100.0)


def test_transition_qa_formula_mismatch_count():
    engine = QualityAssuranceEngine({})

    t1 = np.array([[0, 1], [2, 3]])
    t2 = np.array([[1, 2], [3, 0]])
    trans = t1 * 4 + t2

    trans[0, 0] = 2

    result = engine.run_transition_qa(
        trans,
        t1,
        t2,
        np.ones((2, 2), dtype=bool),
    )

    assert result["status"] == "FAIL"
    assert result["formula_mismatch_pixels"] == 1
    assert engine.categories_status["change"] == "FAIL"


def test_transition_qa_invalid_outside_count():
    engine = QualityAssuranceEngine({})

    t1 = np.array([[0, 1], [2, 3]])
    t2 = np.array([[1, 2], [3, 0]])

    result = engine.run_transition_qa(
        t1 * 4 + t2,
        t1,
        t2,
        np.array([[True, True], [True, False]]),
    )

    assert result["status"] == "FAIL"
    assert result["invalid_outside_pixels"] == 1


def test_transition_qa_dominance_category():
    engine = QualityAssuranceEngine({})

    arr = np.zeros((2, 2))
    mask = np.ones((2, 2), dtype=bool)

    result = engine.run_transition_qa(
        arr, arr, arr, mask
    )

    assert result["status"] == "WARN"
    assert engine.categories_status["change"] == "WARN"

    assert any(
        flag["code"] == "QA-009"
        and flag["category"] == "change"
        for flag in engine.flags
    )


@pytest.mark.parametrize(
    "case,expected",
    [
        ("empty", "PASS"),
        ("full", "PASS"),
        ("boolean", "PASS"),
        ("invalid_binary", "FAIL"),
        ("negative", "FAIL"),
        ("fractional", "FAIL"),
        ("nan", "FAIL"),
        ("inf", "FAIL"),
        ("nodata_inside", "FAIL"),
        ("wrong_shape", "FAIL"),
        ("missing_map", "NOT_AVAILABLE"),
        ("missing_mask", "FAIL"),
        ("invalid_mask", "FAIL"),
        ("empty_overlap", "NOT_AVAILABLE"),
        ("outside_change", "FAIL"),
        ("outside_nodata", "PASS"),
        ("outside_zero", "PASS"),
        ("string_array", "FAIL"),
    ],
)
def test_change_patch_qa_cases(case, expected):
    import json

    engine = QualityAssuranceEngine({})

    arr = np.zeros((2, 2), dtype=np.uint8)
    mask = np.ones((2, 2), dtype=bool)

    if case == "full":
        arr = np.ones((2, 2), dtype=np.uint8)
    elif case == "boolean":
        arr = np.ones((2, 2), dtype=bool)
    elif case == "invalid_binary":
        arr = np.full((2, 2), 2)
    elif case == "negative":
        arr = np.full((2, 2), -1)
    elif case == "fractional":
        arr = np.full((2, 2), 0.5)
    elif case == "nan":
        arr = np.full((2, 2), np.nan)
    elif case == "inf":
        arr = np.full((2, 2), np.inf)
    elif case == "nodata_inside":
        arr = np.full((2, 2), 255)
    elif case == "wrong_shape":
        arr = np.ones((3, 3))
    elif case == "missing_map":
        arr = None
    elif case == "missing_mask":
        mask = None
    elif case == "invalid_mask":
        mask = np.full((2, 2), 255)
    elif case == "empty_overlap":
        mask = np.zeros((2, 2), dtype=bool)
    elif case == "outside_change":
        arr = np.ones((2, 2))
        mask[1, 1] = False
    elif case == "outside_nodata":
        arr = np.array([[1, 1], [0, 255]])
        mask[1, 1] = False
    elif case == "outside_zero":
        arr = np.array([[1, 1], [0, 0]])
        mask[1, 1] = False
    elif case == "string_array":
        arr = np.full((2, 2), "invalid")

    result = engine.run_change_patch_qa(
        arr, arr, arr, arr, mask
    )

    assert result["status"] == expected

    json.dumps(result, allow_nan=False)


def test_change_patch_qa_full_patch_statistics():
    engine = QualityAssuranceEngine({})

    arr = np.ones((2, 2))
    mask = np.ones((2, 2), dtype=bool)

    result = engine.run_change_patch_qa(
        arr, arr, arr, arr, mask
    )

    stats = result["degradation"]

    assert result["status"] == "PASS"
    assert stats["patch_count"] == 1
    assert stats["change_pixels"] == 4
    assert stats["min_size_px"] == 4
    assert stats["median_size_px"] == 4.0
    assert stats["max_size_px"] == 4
    assert stats["change_percentage"] == 100.0


def test_change_patch_qa_four_neighbor_connectivity():
    engine = QualityAssuranceEngine({})

    arr = np.array([[1, 0], [0, 1]])
    mask = np.ones((2, 2), dtype=bool)

    result = engine.run_change_patch_qa(
        arr, arr, arr, arr, mask
    )

    assert result["connectivity"] == 4
    assert result["degradation"]["patch_count"] == 2


def test_change_patch_qa_fractional_statistics():
    engine = QualityAssuranceEngine({})

    arr = np.array([[1, 0, 1], [0, 0, 1]])
    mask = np.ones(arr.shape, dtype=bool)

    result = engine.run_change_patch_qa(
        arr, arr, arr, arr, mask
    )

    stats = result["degradation"]

    assert stats["patch_count"] == 2
    assert stats["change_pixels"] == 3
    assert stats["min_size_px"] == 1
    assert stats["median_size_px"] == 1.5
    assert stats["max_size_px"] == 2
    assert stats["mean_size_px"] == 1.5
    assert stats["change_percentage"] == 50.0


def test_change_patch_qa_zero_change_statistics():
    engine = QualityAssuranceEngine({})

    arr = np.zeros((2, 2))
    mask = np.ones((2, 2), dtype=bool)

    result = engine.run_change_patch_qa(
        arr, arr, arr, arr, mask
    )

    stats = result["degradation"]

    assert result["status"] == "PASS"
    assert stats["patch_count"] == 0
    assert stats["change_pixels"] == 0
    assert stats["change_percentage"] == 0.0


def test_change_patch_qa_failure_priority():
    engine = QualityAssuranceEngine({})

    valid = np.zeros((2, 2))
    invalid = np.full((2, 2), 2)
    mask = np.ones((2, 2), dtype=bool)

    result = engine.run_change_patch_qa(
        invalid,
        None,
        valid,
        valid,
        mask,
    )

    assert result["status"] == "FAIL"
    assert result["degradation"]["status"] == "FAIL"
    assert result["recovery"]["status"] == "NOT_AVAILABLE"
    assert engine.categories_status["change"] == "FAIL"


def test_change_patch_qa_independent_map_statistics():
    engine = QualityAssuranceEngine({})

    empty = np.zeros((2, 2))
    full = np.ones((2, 2))
    diagonal = np.array([[1, 0], [0, 1]])
    single = np.array([[1, 0], [0, 0]])

    result = engine.run_change_patch_qa(
        empty,
        full,
        diagonal,
        single,
        np.ones((2, 2), dtype=bool),
    )

    assert result["status"] == "PASS"
    assert result["degradation"]["patch_count"] == 0
    assert result["recovery"]["patch_count"] == 1
    assert result["canopy_loss"]["patch_count"] == 2
    assert result["residual_stress"]["change_pixels"] == 1



# CVA QA REGRESSION TESTS

@pytest.mark.parametrize(
    "case,expected",
    [
        ("valid", "PASS"),
        ("incorrect", "FAIL"),
        ("negative_magnitude", "FAIL"),
        ("zero", "PASS"),
        ("negative_deltas", "PASS"),
        ("nan_magnitude", "FAIL"),
        ("inf_magnitude", "FAIL"),
        ("nan_ndvi", "FAIL"),
        ("inf_gndvi", "FAIL"),
        ("nan_ndre", "FAIL"),
        ("wrong_magnitude_shape", "FAIL"),
        ("wrong_delta_shape", "FAIL"),
        ("missing_magnitude", "FAIL"),
        ("missing_delta", "FAIL"),
        ("string_magnitude", "FAIL"),
        ("complex_magnitude", "FAIL"),
        ("missing_mask", "FAIL"),
        ("empty_mask", "FAIL"),
        ("wrong_mask_shape", "FAIL"),
        ("invalid_mask", "FAIL"),
        ("nan_mask", "FAIL"),
        ("integer_mask", "PASS"),
        ("empty_region", "NOT_AVAILABLE"),
        ("valid_extreme", "WARN"),
        ("invalid_extreme", "FAIL"),
        ("small_error", "PASS"),
        ("large_error", "FAIL"),
        ("integer_overflow", "WARN"),
    ],
)
def test_cva_qa_edge_cases(case, expected):
    import json

    engine = QualityAssuranceEngine({})

    shape = (2, 2)

    mag = np.full(shape, 5.0)
    a = np.full(shape, 3.0)
    b = np.full(shape, 4.0)
    c = np.zeros(shape)
    mask = np.ones(shape, dtype=bool)

    if case == "incorrect":
        mag *= 2
    elif case == "negative_magnitude":
        mag = -mag
    elif case == "zero":
        mag = a = b = c = np.zeros(shape)
    elif case == "negative_deltas":
        a = -a
        b = -b
    elif case == "nan_magnitude":
        mag = np.full(shape, np.nan)
    elif case == "inf_magnitude":
        mag = np.full(shape, np.inf)
    elif case == "nan_ndvi":
        a = np.full(shape, np.nan)
    elif case == "inf_gndvi":
        b = np.full(shape, np.inf)
    elif case == "nan_ndre":
        c = np.full(shape, np.nan)
    elif case == "wrong_magnitude_shape":
        mag = np.ones((3, 3))
    elif case == "wrong_delta_shape":
        a = np.ones((3, 3))
    elif case == "missing_magnitude":
        mag = None
    elif case == "missing_delta":
        a = None
    elif case == "string_magnitude":
        mag = np.full(shape, "invalid")
    elif case == "complex_magnitude":
        mag = mag.astype(complex)
    elif case == "missing_mask":
        mask = None
    elif case == "empty_mask":
        mask = np.empty((0, 0))
    elif case == "wrong_mask_shape":
        mask = np.ones((3, 3))
    elif case == "invalid_mask":
        mask = np.full(shape, 255)
    elif case == "nan_mask":
        mask = np.full(shape, np.nan)
    elif case == "integer_mask":
        mask = mask.astype(int)
    elif case == "empty_region":
        mask = np.zeros(shape, dtype=bool)
    elif case == "valid_extreme":
        mag *= 3
        a *= 3
        b *= 3
    elif case == "invalid_extreme":
        mag *= 4
    elif case == "small_error":
        mag += 1e-7
    elif case == "large_error":
        mag += 0.01
    elif case == "integer_overflow":
        mag = np.full(shape, 50000)
        a = np.full(shape, 30000, dtype=np.int32)
        b = np.full(shape, 40000, dtype=np.int32)

    result = engine.run_cva_qa(
        mag, a, b, c, mask
    )

    assert result["status"] == expected

    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((), "FAIL"),
        ((4,), "PASS"),
        ((2, 2), "PASS"),
        ((2, 2, 2), "FAIL"),
        ((1, 2, 2, 2), "FAIL"),
        ((0,), "FAIL"),
        ((0, 0), "FAIL"),
    ],
)
def test_cva_qa_dimension_compatibility(shape, expected):
    engine = QualityAssuranceEngine({})

    result = engine.run_cva_qa(
        np.full(shape, 5.0),
        np.full(shape, 3.0),
        np.full(shape, 4.0),
        np.zeros(shape),
        np.ones(shape, dtype=bool),
    )

    assert result["status"] == expected


def test_cva_qa_mismatched_array_shapes():
    engine = QualityAssuranceEngine({})

    result = engine.run_cva_qa(
        np.full((4,), 5.0),
        np.full((2, 2), 3.0),
        np.full((4,), 4.0),
        np.zeros((4,)),
        np.ones((4,), dtype=bool),
    )

    assert result["status"] == "FAIL"


def test_cva_qa_mathematical_statistics():
    engine = QualityAssuranceEngine({})

    shape = (2, 2)

    result = engine.run_cva_qa(
        np.full(shape, 5.0),
        np.full(shape, 3.0),
        np.full(shape, 4.0),
        np.zeros(shape),
        np.ones(shape, dtype=bool),
    )

    assert result["status"] == "PASS"
    assert result["valid_pixels"] == 4
    assert result["invalid_pixels"] == 0
    assert result["negative_magnitude_pixels"] == 0
    assert result["formula_mismatch_pixels"] == 0
    assert result["max_reconciliation_error"] == 0.0
    assert result["max_magnitude_error"] == 0.0

    stats = result["outliers"]

    assert stats["min"] == 5.0
    assert stats["max"] == 5.0
    assert stats["mean"] == 5.0
    assert stats["std"] == 0.0
    assert stats["p1"] == 5.0
    assert stats["p99"] == 5.0


def test_cva_qa_partial_mask():
    engine = QualityAssuranceEngine({})

    mag = np.full((2, 2), 5.0)
    mag[1, 1] = np.nan

    mask = np.array([
        [True, True],
        [True, False],
    ])

    result = engine.run_cva_qa(
        mag,
        np.full((2, 2), 3.0),
        np.full((2, 2), 4.0),
        np.zeros((2, 2)),
        mask,
    )

    assert result["status"] == "PASS"
    assert result["valid_pixels"] == 3
    assert result["invalid_pixels"] == 0


def test_cva_qa_large_magnitude_overflow():
    import json

    engine = QualityAssuranceEngine({})

    huge = np.full((2, 2), 1e200)
    zero = np.zeros((2, 2))

    result = engine.run_cva_qa(
        huge,
        huge,
        zero,
        zero,
        np.ones((2, 2), dtype=bool),
    )

    assert result["status"] == "WARN"
    assert result["formula_mismatch_pixels"] == 0
    assert result["max_reconciliation_error"] == 0.0

    json.dumps(result, allow_nan=False)


def test_cva_qa_failure_priority():
    engine = QualityAssuranceEngine({})

    result = engine.run_cva_qa(
        np.full((2, 2), 20.0),
        np.full((2, 2), 3.0),
        np.full((2, 2), 4.0),
        np.zeros((2, 2)),
        np.ones((2, 2), dtype=bool),
    )

    assert result["status"] == "FAIL"
    assert engine.categories_status["change"] == "FAIL"

    assert any(
        flag["code"] == "QA-020"
        and flag["status"] == "FAIL"
        for flag in engine.flags
    )

    assert any(
        flag["code"] == "QA-010"
        and flag["status"] == "WARN"
        for flag in engine.flags
    )



# WETLAND QA REGRESSION TESTS

import json
from types import SimpleNamespace


@pytest.fixture
def wetland_test_files(tmp_path):
    polygon = {
        "type": "Polygon",
        "coordinates": [[
            [80.0, 16.0],
            [80.1, 16.0],
            [80.1, 16.1],
            [80.0, 16.1],
            [80.0, 16.0],
        ]],
    }

    contents = {
        "valid": polygon,
        "empty": {
            "type": "FeatureCollection",
            "features": [],
        },
        "point": {
            "type": "Point",
            "coordinates": [80.0, 16.0],
        },
        "unclosed": {
            "type": "Polygon",
            "coordinates": [[
                [80.0, 16.0],
                [80.1, 16.0],
                [80.1, 16.1],
                [80.0, 16.1],
            ]],
        },
        "degenerate": {
            "type": "Polygon",
            "coordinates": [[
                [0, 0], [1, 1], [2, 2], [0, 0],
            ]],
        },
        "invalid_crs": {
            **polygon,
            "crs": {
                "type": "name",
                "properties": {
                    "name": "INVALID_CRS",
                },
            },
        },
    }

    files = {}

    for name, content in contents.items():
        file = tmp_path / f"{name}.geojson"
        file.write_text(
            json.dumps(content),
            encoding="utf-8",
        )
        files[name] = file

    files["missing"] = tmp_path / "missing.geojson"

    files["malformed"] = tmp_path / "malformed.geojson"
    files["malformed"].write_text(
        "invalid json", encoding="utf-8"
    )

    files["unsupported"] = tmp_path / "boundary.txt"
    files["unsupported"].write_text(
        "invalid boundary", encoding="utf-8"
    )

    files["missing_prj"] = tmp_path / "boundary.shp"
    files["missing_prj"].touch()

    files["missing_path"] = None

    return files


def make_wetland_qa_result(
    path=None,
    source_type="PROJECT_BOUNDARY",
    status="WETLAND_BOUNDARY_AVAILABLE",
    physical_area=True,
    depletion=False,
):
    return SimpleNamespace(
        source_type=source_type,
        status=status,
        source_metadata={
            "source_path": str(path)
        } if path is not None else {},
        metadata={
            "physical_area_available": physical_area,
        },
        physical_wetland_depletion_enabled=depletion,
    )


def make_wetland_qa_config(claims=False, evidence=False):
    return {
        "wetland": {
            "depletion_claims": {
                "enabled": claims,
            },
        },
        "fusion": {
            "independent_hydrological_source_accepted": evidence,
        },
    }


def make_wetland_qa_mask():
    return np.ones((2, 2), dtype=np.uint8)


def assert_wetland_qa(
    wetland, analysis, config, producer, expected
):
    engine = QualityAssuranceEngine({})

    result = engine.run_wetland_qa(
        wetland, analysis, config, producer
    )

    assert result["status"] == expected, engine.flags

    json.dumps(result, allow_nan=False)
    json.dumps(engine.flags, allow_nan=False)

    return result, engine


@pytest.mark.parametrize(
    "name,expected",
    [
        ("valid", "PASS"),
        ("missing", "FAIL"),
        ("empty", "FAIL"),
        ("point", "FAIL"),
        ("unclosed", "FAIL"),
        ("degenerate", "FAIL"),
        ("invalid_crs", "FAIL"),
        ("malformed", "FAIL"),
        ("unsupported", "FAIL"),
        ("missing_prj", "FAIL"),
        ("missing_path", "FAIL"),
    ],
)
def test_wetland_boundary_validation(
    wetland_test_files, name, expected
):
    mask = make_wetland_qa_mask()

    result, engine = assert_wetland_qa(
        mask,
        mask,
        make_wetland_qa_config(),
        make_wetland_qa_result(
            wetland_test_files[name]
        ),
        expected,
    )

    assert result["boundary_verified"] is (
        expected == "PASS"
    )


@pytest.mark.parametrize(
    "membership,expected",
    [
        (np.ones((2, 2)), "PASS"),
        (np.ones((2, 2), dtype=bool), "PASS"),
        (None, "NOT_AVAILABLE"),
        (np.ones((3, 3)), "FAIL"),
        (np.ones((2, 2, 2)), "FAIL"),
        (np.full((2, 2), -1), "FAIL"),
        (np.full((2, 2), 2), "FAIL"),
        (np.full((2, 2), 0.5), "FAIL"),
        (np.full((2, 2), np.nan), "FAIL"),
        (np.full((2, 2), np.inf), "FAIL"),
        (np.full((2, 2), "invalid"), "FAIL"),
        (np.full((2, 2), 255), "FAIL"),
        (np.zeros((2, 2)), "FAIL"),
    ],
)
def test_wetland_membership_validation(
    wetland_test_files, membership, expected
):
    assert_wetland_qa(
        membership,
        make_wetland_qa_mask(),
        make_wetland_qa_config(),
        make_wetland_qa_result(
            wetland_test_files["valid"]
        ),
        expected,
    )


@pytest.mark.parametrize(
    "analysis,expected",
    [
        (np.ones((2, 2)), "PASS"),
        (np.ones((2, 2), dtype=bool), "PASS"),
        (None, "FAIL"),
        (np.empty((0, 0)), "FAIL"),
        (np.ones((3, 3)), "FAIL"),
        (np.ones((2, 2, 2)), "FAIL"),
        (np.full((2, 2), 255), "FAIL"),
        (np.full((2, 2), 0.5), "FAIL"),
        (np.full((2, 2), np.nan), "FAIL"),
        (np.full((2, 2), np.inf), "FAIL"),
        (np.zeros((2, 2)), "NOT_AVAILABLE"),
    ],
)
def test_wetland_analysis_mask_validation(
    wetland_test_files, analysis, expected
):
    assert_wetland_qa(
        make_wetland_qa_mask(),
        analysis,
        make_wetland_qa_config(),
        make_wetland_qa_result(
            wetland_test_files["valid"]
        ),
        expected,
    )


@pytest.mark.parametrize(
    "source_type,status,expected",
    [
        (None, None, "WARN"),
        (
            "UNAVAILABLE",
            "WETLAND_BOUNDARY_UNAVAILABLE",
            "WARN",
        ),
        (
            "OPTICAL_WATER_INFERENCE",
            "WETLAND_EXTENT_INFERRED",
            "WARN",
        ),
        (
            "JRC_WATER_CONTEXT",
            "WETLAND_EXTENT_INFERRED",
            "WARN",
        ),
        (
            "SAR_INFERENCE",
            "WETLAND_EXTENT_INFERRED",
            "WARN",
        ),
        (
            "UNKNOWN",
            "WETLAND_BOUNDARY_AVAILABLE",
            "FAIL",
        ),
        (
            "PROJECT_BOUNDARY",
            "WETLAND_EXTENT_INFERRED",
            "FAIL",
        ),
        (
            "OFFICIAL_DATASET",
            "WETLAND_BOUNDARY_AVAILABLE",
            "PASS",
        ),
    ],
)
def test_wetland_source_consistency(
    wetland_test_files, source_type, status, expected
):
    producer = None

    if source_type is not None:
        producer = make_wetland_qa_result(
            wetland_test_files["valid"],
            source_type=source_type,
            status=status,
        )

    mask = make_wetland_qa_mask()

    assert_wetland_qa(
        mask,
        mask,
        make_wetland_qa_config(),
        producer,
        expected,
    )


@pytest.mark.parametrize(
    "claims,evidence,source_type,physical_area,depletion,expected",
    [
        (
            False, False, "PROJECT_BOUNDARY",
            True, False, "PASS",
        ),
        (
            True, True, "UNAVAILABLE",
            True, False, "FAIL",
        ),
        (
            True, True, "OPTICAL_WATER_INFERENCE",
            True, False, "FAIL",
        ),
        (
            True, False, "PROJECT_BOUNDARY",
            True, True, "FAIL",
        ),
        (
            True, True, "PROJECT_BOUNDARY",
            False, True, "FAIL",
        ),
        (
            True, True, "PROJECT_BOUNDARY",
            True, False, "FAIL",
        ),
        (
            True, True, "PROJECT_BOUNDARY",
            True, True, "WARN",
        ),
    ],
)
def test_wetland_depletion_safeguards(
    wetland_test_files,
    claims,
    evidence,
    source_type,
    physical_area,
    depletion,
    expected,
):
    statuses = {
        "PROJECT_BOUNDARY": "WETLAND_BOUNDARY_AVAILABLE",
        "UNAVAILABLE": "WETLAND_BOUNDARY_UNAVAILABLE",
        "OPTICAL_WATER_INFERENCE": "WETLAND_EXTENT_INFERRED",
    }

    producer = make_wetland_qa_result(
        wetland_test_files["valid"],
        source_type=source_type,
        status=statuses[source_type],
        physical_area=physical_area,
        depletion=depletion,
    )

    mask = make_wetland_qa_mask()

    result, engine = assert_wetland_qa(
        mask,
        mask,
        make_wetland_qa_config(claims, evidence),
        producer,
        expected,
    )

    assert result["physical_depletion_supported"] is False


@pytest.mark.parametrize(
    "config",
    [
        None,
        {"wetland": []},
        {"wetland": {"boundary": []}},
        {"wetland": {"official_dataset": []}},
        {"wetland": {"depletion_claims": []}},
        {"fusion": []},
        {
            "wetland": {
                "depletion_claims": {
                    "enabled": "true",
                },
            },
        },
        {
            "fusion": {
                "independent_hydrological_source_accepted": 1,
            },
        },
    ],
)
def test_wetland_invalid_configuration(
    wetland_test_files, config
):
    mask = make_wetland_qa_mask()

    assert_wetland_qa(
        mask,
        mask,
        config,
        make_wetland_qa_result(
            wetland_test_files["valid"]
        ),
        "FAIL",
    )


def test_wetland_membership_outside_analysis(
    wetland_test_files
):
    mask = make_wetland_qa_mask()

    analysis = np.array([
        [1, 0],
        [1, 0],
    ], dtype=np.uint8)

    result, engine = assert_wetland_qa(
        mask,
        analysis,
        make_wetland_qa_config(),
        make_wetland_qa_result(
            wetland_test_files["valid"]
        ),
        "PASS",
    )

    assert result["wetland_pixels"] == 4
    assert result["valid_analysis_pixels"] == 2
    assert result["invalid_analysis_pixels"] == 0


def test_wetland_nodata_outside_analysis(
    wetland_test_files
):
    membership = np.array([
        [1, 255],
        [1, 0],
    ], dtype=np.uint8)

    analysis = np.array([
        [1, 0],
        [1, 0],
    ], dtype=np.uint8)

    result, engine = assert_wetland_qa(
        membership,
        analysis,
        make_wetland_qa_config(),
        make_wetland_qa_result(
            wetland_test_files["valid"]
        ),
        "PASS",
    )

    assert result["wetland_pixels"] == 2
    assert result["valid_analysis_pixels"] == 2
    assert result["invalid_analysis_pixels"] == 0


def test_wetland_unavailable_fallback():
    producer = make_wetland_qa_result(
        source_type="UNAVAILABLE",
        status="WETLAND_BOUNDARY_UNAVAILABLE",
    )

    result, engine = assert_wetland_qa(
        np.zeros((2, 2)),
        make_wetland_qa_mask(),
        make_wetland_qa_config(),
        producer,
        "WARN",
    )

    assert result["boundary_verified"] is False
    assert result["physical_depletion_supported"] is False


def test_wetland_failure_priority():
    producer = make_wetland_qa_result(
        source_type="UNAVAILABLE",
        status="WETLAND_BOUNDARY_UNAVAILABLE",
    )

    result, engine = assert_wetland_qa(
        np.full((2, 2), 2),
        make_wetland_qa_mask(),
        make_wetland_qa_config(),
        producer,
        "FAIL",
    )

    assert engine.categories_status["wetland"] == "FAIL"


def test_wetland_backward_compatibility():
    engine = QualityAssuranceEngine({})
    mask = make_wetland_qa_mask()

    result = engine.run_wetland_qa(
        mask,
        mask,
        make_wetland_qa_config(),
    )

    assert result["status"] == "WARN"
    assert result["physical_depletion_supported"] is False

    json.dumps(result, allow_nan=False)
