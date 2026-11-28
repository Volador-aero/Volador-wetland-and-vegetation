import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app import services
from app.routers.projects import ProjectDatesUpdate, update_project_dates


@pytest.fixture(autouse=True)
def isolate_run_states():
    services.RUN_STATES.clear()
    yield
    services.RUN_STATES.clear()


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


def test_disabled_observation_dates_allow_equal_override_dates(valid_preflight):
    services.RunWorker.preflight_run(
        valid_preflight["id"],
        {
            "use_observation_dates": False,
            "t1_date": "2023-08-14",
            "t2_date": "2023-08-14",
        },
    )

    assert not services.RUN_STATES


def test_t1_later_than_t2_is_rejected(valid_preflight):
    with pytest.raises(ValueError, match="must not be later"):
        services.RunWorker.preflight_run(
            valid_preflight["id"],
            {
                "t1_date": "2023-08-28",
                "t2_date": "2023-08-14",
            },
        )

    assert not services.RUN_STATES


def test_camel_case_analysis_mode_overrides_default(valid_preflight, monkeypatch):
    observed = {}

    def validate_dataset(**kwargs):
        observed.update(kwargs)
        return {"valid": True, "messages": []}

    monkeypatch.setattr(
        services.DatasetService,
        "inspect_and_validate",
        validate_dataset,
    )

    services.RunWorker.preflight_run(
        valid_preflight["id"],
        {"analysisMode": "vegetation"},
    )

    assert observed["analysis_mode"] == "vegetation"
    assert not services.RUN_STATES


def test_valid_sample_fallback_completes_without_run_state(valid_preflight):
    services.RunWorker.preflight_run(valid_preflight["id"])

    assert not services.RUN_STATES


def test_partial_wetland_upload_does_not_use_sample_fallback(valid_preflight, monkeypatch, tmp_path):
    t1_dir = Path(valid_preflight["t1_dir"])
    t1_dir.mkdir()
    (t1_dir / "scene_G.tif").write_bytes(b"test")

    with pytest.raises(ValueError, match="requires both T1 and T2"):
        services.RunWorker.preflight_run(valid_preflight["id"])


def test_malformed_model_config_is_rejected_as_value_error(valid_preflight, monkeypatch):
    monkeypatch.setattr(services, "load_config", lambda _: {"model": {}})

    with pytest.raises(ValueError, match="model path is missing"):
        services.RunWorker.preflight_run(valid_preflight["id"])


def test_reset_project_is_blocked_for_active_run(valid_preflight, monkeypatch, tmp_path):
    project = valid_preflight
    services.RUN_STATES["active-run"] = {
        "run_id": "active-run",
        "project_id": project["id"],
        "status": "VALIDATING",
    }
    monkeypatch.setattr(
        services.ProjectStore,
        "load_projects",
        lambda: {project["id"]: project},
    )

    with pytest.raises(ValueError, match="cannot be reset"):
        services.ProjectStore.reset_project(project["id"])


def test_readiness_reports_blocked_equal_dates(valid_preflight, monkeypatch):
    project = valid_preflight
    project["t1_date"] = "2023-08-14"
    project["t2_date"] = "2023-08-14"

    readiness = services.RunWorker.get_readiness(
        project["id"],
        {"analysis_mode": "wetland"},
    )

    date_check = next(check for check in readiness["checks"] if check["id"] == "observation_dates")
    assert date_check["status"] == "BLOCKED"
    assert readiness["ready"] is False
    assert readiness["status"] == "BLOCKED"


def test_readiness_missing_model_is_blocked(valid_preflight, monkeypatch, tmp_path):
    missing = tmp_path / "missing.onnx"
    monkeypatch.setattr(services, "load_config", lambda _: _config(missing))

    readiness = services.RunWorker.get_readiness(
        valid_preflight["id"],
        {"analysis_mode": "wetland"},
    )

    model_check = next(check for check in readiness["checks"] if check["id"] == "model_compatibility")
    assert model_check["status"] == "BLOCKED"
    assert readiness["ready"] is False


def test_readiness_wetland_boundary_is_warning(valid_preflight):
    readiness = services.RunWorker.get_readiness(
        valid_preflight["id"],
        {"analysis_mode": "wetland"},
    )

    boundary_check = next(check for check in readiness["checks"] if check["id"] == "wetland_boundary")
    assert boundary_check["status"] == "WARNING"
    assert boundary_check["blocking"] is False


def test_readiness_vegetation_does_not_require_t2(valid_preflight):
    readiness = services.RunWorker.get_readiness(
        valid_preflight["id"],
        {"analysis_mode": "vegetation"},
    )

    t2_check = next(check for check in readiness["checks"] if check["id"] == "t2_dataset")
    assert t2_check["status"] == "PASS"
    assert t2_check["blocking"] is False


def test_start_run_ids_are_unique_within_same_second(monkeypatch, tmp_path):
    project = {
        "id": "run-id-project",
        "analysis_mode": "wetland",
    }
    preparation = {
        "status": "COMMITTED",
        "is_locked": True,
        "inclusion": None,
        "manual_exclusions": [],
    }

    monkeypatch.setattr(
        services.ProjectStore,
        "get_project",
        lambda _: project,
    )
    monkeypatch.setattr(
        services.ProjectStore,
        "load_preparation",
        lambda _: preparation,
    )
    monkeypatch.setattr(services, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(services.executor, "submit", lambda *args: None)
    services.RUN_STATES.clear()

    first = services.RunWorker.start_run(project["id"])
    second = services.RunWorker.start_run(project["id"])

    assert first["run_id"] != second["run_id"]
    assert first["run_id"].startswith("run_")
    assert second["run_id"].startswith("run_")
    assert first["run_id"] in services.RUN_STATES
    assert second["run_id"] in services.RUN_STATES

    services.RUN_STATES.clear()


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
