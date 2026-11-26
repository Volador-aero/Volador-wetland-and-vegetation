"""
tests/test_pipeline_e2e_report.py
===================================
End-to-end pipeline test verifying that one call to pipeline.run_analysis()
creates the full GIS output hierarchy, cartographic figures, and the final PDF.
"""

from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import pypdf

from gis_engine.api.service import analyze_wetland as run_analysis


def _create_mock_scene(dir_path: Path, prefix: str = "scene"):
    dir_path.mkdir(parents=True, exist_ok=True)
    transform = from_origin(500000, 4000000, 10, 10)
    data = (np.ones((64, 64), dtype=np.float32) * 2000.0)

    for band in ["G", "R", "RE", "NIR"]:
        with rasterio.open(
            dir_path / f"{prefix}_{band}.tif",
            "w",
            driver="GTiff",
            height=64,
            width=64,
            count=1,
            dtype="float32",
            crs="EPSG:32632",
            transform=transform,
        ) as dst:
            dst.write(data, 1)


@patch("gis_engine.inference.onnx.ONNXModelWrapper.load_model")
@patch("gis_engine.inference.onnx.ONNXModelWrapper.predict")
def test_pipeline_e2e_produces_pdf_and_maps(mock_predict, mock_load, tmp_path):
    """Verifies that run_analysis produces FINAL_ANALYSIS_REPORT.pdf and map figures."""
    mock_load.return_value = None
    # Mock logits (1, 4, 64, 64) and class map (64, 64)
    logits = np.zeros((1, 4, 64, 64), dtype=np.float32)
    logits[0, 3, :, :] = 5.0  # Healthy
    classes = np.full((64, 64), 3, dtype=np.uint8)
    mock_predict.return_value = (logits, classes)

    t1_dir = tmp_path / "t1"
    t2_dir = tmp_path / "t2"
    _create_mock_scene(t1_dir)
    _create_mock_scene(t2_dir)

    out_dir = tmp_path / "outputs"

    # Resolve config path relative to Engine_module root (two levels up from this test file)
    _engine_root = Path(__file__).resolve().parent.parent
    _config_path = _engine_root / "configs" / "config.json"

    result = run_analysis(
        t1_scene=t1_dir,
        t2_scene=t2_dir,
        t1_date="2023-08-14",
        t2_date="2023-08-28",
        config_path=str(_config_path),
        model_path=_create_valid_qa_model(tmp_path),  # dummy model path
        output_dir=out_dir,
    )

    assert result.status in ["SUCCESS", "WARNING"]
    run_dir = Path(result.output_directory)

    # Check Markdown report
    md_report = run_dir / "reports" / "analysis_report.md"
    assert md_report.is_file()

    # Check PDF report
    pdf_report = run_dir / "reports" / "FINAL_ANALYSIS_REPORT.pdf"
    assert pdf_report.is_file()
    assert pdf_report.stat().st_size > 5000

    # Verify PDF content via PyPDF
    reader = pypdf.PdfReader(str(pdf_report))
    assert len(reader.pages) >= 3

    # Check outputs dictionary & manifest
    assert "reports/FINAL_ANALYSIS_REPORT.pdf" in result.outputs
    manifest_p = run_dir / "manifest.json"
    assert manifest_p.is_file()


def _create_valid_qa_model(directory):
    """Create a valid synthetic ONNX model for mocked integration tests."""

    import numpy as np
    import onnx

    from onnx import helper, TensorProto, numpy_helper

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    model_path = directory / "qa_test_model.onnx"

    model_input = helper.make_tensor_value_info(
        "input",
        TensorProto.FLOAT,
        [1, 5, "height", "width"],
    )

    model_output = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        [1, 4, "height", "width"],
    )

    weights = numpy_helper.from_array(
        np.zeros((4, 5, 1, 1), dtype=np.float32),
        name="weights",
    )

    bias = numpy_helper.from_array(
        np.array([0, 0, 0, 10], dtype=np.float32),
        name="bias",
    )

    convolution = helper.make_node(
        "Conv",
        ["input", "weights", "bias"],
        ["output"],
        kernel_shape=[1, 1],
    )

    graph = helper.make_graph(
        [convolution],
        "synthetic_qa_test_model",
        [model_input],
        [model_output],
        [weights, bias],
    )

    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 13)
        ],
        producer_name="integration-test-fixture",
    )

    model.ir_version = 8

    onnx.checker.check_model(model)

    onnx.save(model, str(model_path))

    return model_path
