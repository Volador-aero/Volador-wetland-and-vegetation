import json
from pathlib import Path

import pytest

from app import services


def _project_record(project_id, old_root):
    return {
        "id": project_id,
        "name": "Legacy project",
        "t1_dir": str(old_root / "datasets" / "t1"),
        "t2_dir": str(old_root / "datasets" / "t2"),
        "wetland_boundary": str(old_root / "boundary.geojson"),
    }


def test_get_project_rebases_legacy_dataset_paths_to_current_storage(tmp_path, monkeypatch):
    storage = tmp_path / "storage"
    monkeypatch.setattr(services, "STORAGE_DIR", storage)
    (storage / "projects").mkdir(parents=True)
    old_root = tmp_path / "old-computer" / "project"
    old_root.mkdir(parents=True)
    project_id = "prj_12345"
    record = _project_record(project_id, old_root)
    (storage / "projects.json").write_text(json.dumps({project_id: record}))

    project = services.ProjectStore.get_project(project_id)

    assert Path(project["t1_dir"]) == storage / "projects" / project_id / "datasets" / "t1"
    assert Path(project["t2_dir"]) == storage / "projects" / project_id / "datasets" / "t2"
    assert project["wetland_boundary"] is None
    # Reading a project must not rewrite the legacy record unexpectedly.
    saved = json.loads((storage / "projects.json").read_text())[project_id]
    assert saved["t1_dir"] == record["t1_dir"]


def test_reset_project_never_deletes_legacy_external_dataset_paths(tmp_path, monkeypatch):
    storage = tmp_path / "storage"
    monkeypatch.setattr(services, "STORAGE_DIR", storage)
    (storage / "projects").mkdir(parents=True)
    project_id = "prj_12346"

    external_t1 = tmp_path / "external" / "t1"
    external_t2 = tmp_path / "external" / "t2"
    external_t1.mkdir(parents=True)
    external_t2.mkdir(parents=True)
    protected_file = external_t1 / "keep.tif"
    protected_file.write_bytes(b"do not delete")

    record = _project_record(project_id, tmp_path / "external")
    (storage / "projects.json").write_text(json.dumps({project_id: record}))

    current_t1 = storage / "projects" / project_id / "datasets" / "t1"
    current_t2 = storage / "projects" / project_id / "datasets" / "t2"
    current_t1.mkdir(parents=True)
    current_t2.mkdir(parents=True)
    (current_t1 / "remove.tif").write_bytes(b"remove")
    (current_t2 / "remove.tif").write_bytes(b"remove")

    result = services.ProjectStore.reset_project(project_id)

    assert protected_file.read_bytes() == b"do not delete"
    assert list(current_t1.iterdir()) == []
    assert list(current_t2.iterdir()) == []
    assert Path(result["t1_dir"]) == current_t1
    assert Path(result["t2_dir"]) == current_t2


def test_project_directory_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(services, "STORAGE_DIR", tmp_path / "storage")
    with pytest.raises(ValueError):
        services.ProjectStore.project_directory("prj_../../outside")
