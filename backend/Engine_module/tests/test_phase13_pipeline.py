"""
tests/test_phase13_pipeline.py
===============================
Phase 13 E2E and unit tests for the ProductionPipeline class.

Verifies:
1. Complete execution of the production pipeline on synthetic data.
2. Structure and content of the generated output directory (t1_health.tif, metadata.json, etc.).
3. Creation of config_used.json and manifest.json.
4. Validation and error handling inside the pipeline.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.gis_engine.core.pipeline import ProductionPipeline
from src.gis_engine.core.exceptions import InputValidationError, ConfigurationError

TEST_CONFIG = {
    "model": {
        "path": "dummy_model.onnx",
        "in_channels": 5,
        "num_classes": 4,
        "channel_order": ["ndvi", "gndvi", "ndre", "nir", "rededge"]
    },
    "preprocessing": {
        "mode": "inference",
        "band_scaling_factor": 10000.0,
        "require_crs": False,
        "epsilon": 1e-6,
        "min_denominator": 0.05,
        "min_band_value": 0.01,
        "gaussian_sigma": 1.0,
        "normalize": True
    },
    "ingestion": {
        "band_mapping": {
            "green": ["*_G.tif", "*green.tif"],
            "red": ["*_R.tif", "*red.tif"],
            "rededge": ["*_RE.tif", "*rededge.tif"],
            "nir": ["*_NIR.tif", "*nir.tif"]
        }
    },
    "water": {
        "ndwi_threshold": 0.0,
        "epsilon": 1e-6
    },
    "change": {
        "residual_stress": {
            "enabled": True,
            "threshold": {"method": "percentile", "value": 95.0}
        }
    },
    "fusion": {
        "independent_hydrological_source_accepted": False
    },
    "wetland": {
        "source": "none"
    },
    "metrics": {
        "severity_weights": {
            "healthy": 0.0,
            "moderate": 0.35,
            "severe": 0.8,
            "background": 1.0
        }
    },
    "output": {
        "colormap": {
            "0": [128, 128, 128, 255],
            "1": [255, 0, 0, 255],
            "2": [255, 165, 0, 255],
            "3": [0, 170, 0, 255]
        }
    }
}


def create_mock_bands(dir_path: Path, shape=(10, 10), crs="EPSG:4326", transform=None, fill_val=1000.0):
    dir_path.mkdir(parents=True, exist_ok=True)
    if transform is None:
        transform = from_origin(100.0, 10.0, 0.001, 0.001)

    bands = ["G", "R", "RE", "NIR"]
    for b in bands:
        path = dir_path / f"scene_{b}.tif"
        profile = {
            'driver': 'GTiff',
            'height': shape[0],
            'width': shape[1],
            'count': 1,
            'dtype': rasterio.float32,
            'crs': crs,
            'transform': transform,
            'nodata': -9999.0
        }
        arr = np.full(shape, fill_val, dtype=np.float32)
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(arr, 1)


@pytest.fixture(autouse=True)
def mock_onnx_wrapper():
    def mock_load(self):
        self.session = "mock_session"
        self.input_name = "input"
        self.output_name = "output"
        self.execution_provider = "CPUExecutionProvider"
        self.loading_time_ms = 0.0

    # Mock ONNX model Wrapper predict to return healthy vegetation class map
    def mock_predict(self, tensor):
        logits = np.zeros((1, 4, 10, 10), dtype=np.float32)
        logits[0, 3, :, :] = 10.0  # healthy class high
        class_map = np.full((10, 10), 3, dtype=np.uint8)
        return logits, class_map

    with patch('src.gis_engine.inference.onnx.ONNXModelWrapper.load_model', new=mock_load), \
         patch('src.gis_engine.inference.onnx.ONNXModelWrapper.predict', new=mock_predict):
        yield


def test_pipeline_successful_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        t1_dir = tmp_path / "t1"
        t2_dir = tmp_path / "t2"
        output_dir = tmp_path / "outputs"
        config_path = tmp_path / "config.json"

        # Write config file
        config_path.write_text(json.dumps(TEST_CONFIG))

        # Write bands
        transform = from_origin(100.0, 10.0, 0.001, 0.001)
        create_mock_bands(t1_dir, shape=(10, 10), crs="EPSG:4326", transform=transform, fill_val=2000.0)
        create_mock_bands(t2_dir, shape=(10, 10), crs="EPSG:4326", transform=transform, fill_val=2500.0)

        # Run pipeline
        pipeline = ProductionPipeline()
        result = pipeline.run_analysis(
            t1_scene=t1_dir,
            t2_scene=t2_dir,
            t1_date="2026-08-01",
            t2_date="2026-08-15",
            config_path=config_path,
            model_path=_create_valid_qa_model(tmp_path),  # dummy model path
            output_base_dir=output_dir,
            run_id="run_pipeline_test"
        )

        assert result.status in ["SUCCESS", "WARNING"]
        assert result.run_id == "run_pipeline_test"

        run_path = output_dir / "run_pipeline_test"
        assert run_path.is_dir()

        # Check required files exist in output structure
        assert (run_path / "config_used.json").is_file()
        assert (run_path / "metadata.json").is_file()
        assert (run_path / "manifest.json").is_file()
        assert (run_path / "analysis_result.json").is_file()
        assert (run_path / "vegetation" / "t1_health.tif").is_file()
        assert (run_path / "vegetation" / "t2_health.tif").is_file()
        assert (run_path / "change" / "transition_map.tif").is_file()
        assert (run_path / "change" / "cva_magnitude.tif").is_file()
        assert (run_path / "degradation" / "degradation_map.tif").is_file()
        assert (run_path / "degradation" / "recovery_map.tif").is_file()
        assert (run_path / "hydrology" / "water_change.tif").is_file()
        assert (run_path / "reports" / "analysis_report.md").is_file()
        assert (run_path / "logs" / "analysis.log").is_file()

        # Verify manifest matches actual file structure
        with open(run_path / "manifest.json", "r") as f:
            manifest = json.load(f)
            # Administrative records are intentionally excluded
            # from the immutable output manifest.

            from hashlib import sha256
            from pathlib import Path as ManifestPath

            excluded_outputs = {
                "manifest.json",
                "manifest_integrity.json",
                "run_status.json",
            }

            registered_outputs = {
                str(value).replace(chr(92), "/")
                for value in result.outputs.values()
            }

            assert excluded_outputs.issubset(registered_outputs)

            expected_outputs = registered_outputs - excluded_outputs

            manifest_entries = manifest["outputs"]

            assert isinstance(manifest_entries, list)

            assert manifest["total_outputs"] == len(manifest_entries)

            actual_outputs = {
                entry["path"] for entry in manifest_entries
            }

            assert len(actual_outputs) == len(manifest_entries)

            assert actual_outputs == expected_outputs

            assert not actual_outputs.intersection(excluded_outputs)

            run_directory = ManifestPath(result.output_directory)

            for entry in manifest_entries:

                output_path = run_directory / entry["path"]

                assert output_path.is_file()

                assert entry["status"] == "PASS"

                assert entry["size_bytes"] == output_path.stat().st_size

                digest = sha256()

                with output_path.open("rb") as stream:
                    for chunk in iter(
                        lambda: stream.read(1024 * 1024),
                        b"",
                    ):
                        digest.update(chunk)

                assert entry["sha256"] == digest.hexdigest()

            import json as manifest_json

            integrity = manifest_json.loads(
                (run_directory / "manifest_integrity.json").read_text(
                    encoding="utf-8"
                )
            )

            terminal = manifest_json.loads(
                (run_directory / "run_status.json").read_text(
                    encoding="utf-8"
                )
            )

            assert integrity["status"] == "PASS"

            assert integrity["integrity_verified"] is True

            assert terminal["status"] == "COMPLETED"

            assert terminal["execution_status"] == result.status

            assert terminal["manifest_integrity_verified"] is True

            assert terminal["manifest_sha256"] == sha256(
                (run_directory / "manifest.json").read_bytes()
            ).hexdigest()


def test_pipeline_validation_errors():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        t1_dir = tmp_path / "t1"
        t2_dir = tmp_path / "t2"
        output_dir = tmp_path / "outputs"
        config_path = tmp_path / "config.json"

        # Write config
        config_path.write_text(json.dumps(TEST_CONFIG))

        # Create mock directories so they pass the directory existence validation checks
        t1_dir.mkdir(parents=True, exist_ok=True)
        t2_dir.mkdir(parents=True, exist_ok=True)

        # Test case 1: T1 date >= T2 date
        pipeline = ProductionPipeline()
        result = pipeline.run_analysis(
            t1_scene=t1_dir,
            t2_scene=t2_dir,
            t1_date="2026-08-15",
            t2_date="2026-08-01",  # earlier
            config_path=config_path,
            model_path=config_path,
            output_base_dir=output_dir,
            run_id="test_date_fail"
        )
        assert result.status == "FAILED"
        assert any("date" in err for err in result.errors)

        # Test case 2: Missing directories
        result2 = pipeline.run_analysis(
            t1_scene=t1_dir / "missing",
            t2_scene=t2_dir,
            t1_date="2026-08-01",
            t2_date="2026-08-15",
            config_path=config_path,
            model_path=config_path,
            output_base_dir=output_dir,
            run_id="test_missing_dir"
        )
        assert result2.status == "FAILED"
        assert any("directory not found" in err for err in result2.errors)


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
