import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app import services
from app.routers.projects import ProjectDatesUpdate, update_project_dates


def _project(tmp_path):
    return {
        "id": "preflight-project",
        "t1_dir": str(tmp_path / "t1"),
        "t2_dir": str(tmp_path / "t2"),
        "analysis_mode": "wetland",
        "wetland_boundary": None,
        "t1_date": None,
        "t2_date": None,
    }


def _config(model_path):
    return {
        "model": {
            "path": str(model_path),
            "in_channels": 5,
            "num_classes": 4,
            "channel_order": ["ndvi", "gndvi", "ndre", "nir", "rededge"],
        },
        "preprocessing": {},
        "water": {},
        "ingestion": {},
    }


class _Shape:
    def __init__(self, input_shape=(1, 5, "height", "width"), output_shape=(1, 4, "height", "width")):
        self._input_shape = input_shape
        self._output_shape = output_shape

    def get_inputs(self):
        return [type("Input", (), {"shape": self._input_shape})()]

    def get_outputs(self):
        return [type("Output", (), {"shape": self._output_shape})()]


class _Model:
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = _Shape()


@pytest.fixture
def valid_preflight(monkeypatch, tmp_path):
    project = _project(tmp_path)
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")

    monkeypatch.setattr(services.ProjectStore, "get_project", lambda _: project)
    monkeypatch.setattr(services.ProjectStore, "load_preparation", lambda _: None)
    monkeypatch.setattr(services, "load_config", lambda _: _config(model_path))
    monkeypatch.setattr(services, "ONNXModelWrapper", _Model)
    monkeypatch.setattr(
        services.DatasetService,
        "inspect_and_validate",
        lambda **_: {"valid": True, "messages": []},
    )
    return project


def test_missing_model_is_rejected_before_run_state(valid_preflight, monkeypatch, tmp_path):
    missing = tmp_path / "missing.onnx"
    monkeypatch.setattr(services, "load_config", lambda _: _config(missing))

    with pytest.raises(ValueError, match="analysis model is missing") as error:
        services.RunWorker.preflight_run(valid_preflight["id"])

    assert str(tmp_path) not in str(error.value)
    assert not services.RUN_STATES


def test_model_output_shape_is_rejected(valid_preflight, monkeypatch):
    class WrongOutputModel(_Model):
        def __init__(self, model_path):
            self.model_path = model_path
            self.session = _Shape(output_shape=(1, 3, "height", "width"))

    monkeypatch.setattr(services, "ONNXModelWrapper", WrongOutputModel)

    with pytest.raises(ValueError, match="analysis model is incompatible"):
        services.RunWorker.preflight_run(valid_preflight["id"])

    assert not services.RUN_STATES


def test_equal_override_dates_are_rejected(valid_preflight):
    with pytest.raises(ValueError, match="must be different"):
        services.RunWorker.preflight_run(
            valid_preflight["id"],
            {
                "t1_date": "2023-08-14",
                "t2_date": "2023-08-14",
            },
        )

    assert not services.RUN_STATES


def test_valid_sample_fallback_completes_without_run_state(valid_preflight):
    services.RunWorker.preflight_run(valid_preflight["id"])

    assert not services.RUN_STATES


def test_project_date_update_rejects_equal_dates(monkeypatch):
    monkeypatch.setattr(
        services.ProjectStore,
        "get_project",
        lambda _: {"id": "preflight-project"},
    )

    with pytest.raises(HTTPException, match="must be different"):
        update_project_dates(
            "preflight-project",
            ProjectDatesUpdate(t1_date="2023-08-14", t2_date="2023-08-14"),
        )
