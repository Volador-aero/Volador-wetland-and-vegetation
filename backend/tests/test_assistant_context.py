import json
from pathlib import Path

import pytest

from app import assistant_context_service as context_service


@pytest.fixture
def project_env(tmp_path, monkeypatch):
    project_id = "project_test123"
    project_dir = tmp_path / "projects" / project_id
    (project_dir / "datasets" / "t1").mkdir(parents=True)
    (project_dir / "datasets" / "t2").mkdir(parents=True)
    monkeypatch.setattr(context_service, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(context_service.ProjectStore, "get_project", lambda _id: {
        "name": "Test project", "analysis_mode": "vegetation", "latest_run_id": None
    })
    return project_id, project_dir


def test_invalid_project_id_is_rejected(project_env):
    result = context_service.build_assistant_context("../outside")
    assert result["project_available"] is False


def test_missing_project_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(context_service.ProjectStore, "get_project", lambda _id: None)
    monkeypatch.setattr(context_service, "STORAGE_DIR", tmp_path)
    result = context_service.build_assistant_context("project_test")
    assert result["project_available"] is False
    assert "not found" in result["message"]


def test_dataset_folders_are_counted(project_env):
    project_id, project_dir = project_env
    (project_dir / "datasets" / "t1" / "green.tif").write_bytes(b"raster")
    result = context_service.build_assistant_context(project_id)
    assert result["datasets"]["t1"]["file_count"] == 1
    assert result["datasets"]["t2"]["file_count"] == 0


@pytest.mark.parametrize("payload", ["[]", "null", '{"metrics": []}', '{"metrics": null}', '{"metrics": "bad"}'])
def test_malformed_preparation_json_does_not_crash(project_env, payload):
    project_id, project_dir = project_env
    (project_dir / "preparation.json").write_text(payload, encoding="utf-8")
    result = context_service.build_assistant_context(project_id)
    assert result["preparation"]["available"] is False
    assert result["preparation"]["read_error"] is True


def test_valid_preparation_with_bad_metric_values_is_sanitized(project_env):
    project_id, project_dir = project_env
    (project_dir / "preparation.json").write_text(json.dumps({
        "status": "DRAFT", "is_locked": False,
        "metrics": {"area_basis": "PIXEL_COUNT_ONLY", "source_pixels": True,
                    "effective_pixels": float("nan"), "effective_ha": 42}
    }), encoding="utf-8")
    result = context_service.build_assistant_context(project_id)
    prep = result["preparation"]
    assert prep["available"] is True
    assert prep["source_pixels"] is None
    assert prep["effective_pixels"] is None
    assert prep["effective_area_ha"] is None


def test_malformed_analysis_nested_objects_do_not_crash(project_env, monkeypatch):
    project_id, project_dir = project_env
    run_id = "run_test123"
    run_dir = project_dir / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "analysis_result.json").write_text(json.dumps({
        "run_id": run_id, "spectral_indices": [], "vegetation": None,
        "spatial": "invalid"
    }), encoding="utf-8")
    monkeypatch.setattr(context_service.RunWorker, "get_run_status", lambda _id: {
        "project_id": project_id, "status": "COMPLETED"
    })
    result = context_service.build_assistant_context(project_id, run_id)
    assert result["analysis"]["available"] is True
    assert result["analysis"]["spectral_indices"] == {}
    assert result["analysis"]["vegetation"] == {}
    assert result["analysis"]["spatial"]["total_pixels"] is None


def test_analysis_result_symlink_outside_run_is_rejected(project_env, monkeypatch, tmp_path):
    project_id, project_dir = project_env
    run_id = "run_test123"
    run_dir = project_dir / "runs" / run_id
    run_dir.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    try:
        (run_dir / "analysis_result.json").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are not available in this environment")
    monkeypatch.setattr(context_service.RunWorker, "get_run_status", lambda _id: {
        "project_id": project_id, "status": "COMPLETED"
    })
    result = context_service.build_assistant_context(project_id, run_id)
    assert result["analysis"]["available"] is False


def test_wrong_run_owner_is_rejected(project_env, monkeypatch):
    project_id, _ = project_env
    monkeypatch.setattr(context_service.RunWorker, "get_run_status", lambda _id: {
        "project_id": "another_project", "status": "COMPLETED"
    })
    result = context_service.build_assistant_context(project_id, "run_test123")
    assert result["analysis"]["available"] is False
