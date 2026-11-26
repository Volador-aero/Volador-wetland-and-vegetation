import pytest
from fastapi import HTTPException

from app import services
from app.services import RunWorker


RUN_ID = "run_20261126_144103"

PROJECT_A = "prj_1795704062345"
PROJECT_B = "prj_1795704063051"


@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(
        services, "STORAGE_DIR", tmp_path
    )

    monkeypatch.setattr(
        services, "RUN_STATES", {}
    )

    projects = tmp_path / "projects"
    projects.mkdir()

    return projects


def create_run(projects, project_id):
    run_dir = projects / project_id / "runs" / RUN_ID
    run_dir.mkdir(parents=True)
    return run_dir


def test_duplicate_run_id_requires_project_identity(
    isolated_storage,
):
    create_run(isolated_storage, PROJECT_A)
    create_run(isolated_storage, PROJECT_B)

    with pytest.raises(HTTPException) as exc:
        RunWorker.get_run_status(RUN_ID)

    assert exc.value.status_code == 409


@pytest.mark.parametrize(
    "project_id",
    [PROJECT_A, PROJECT_B],
)
def test_scoped_run_never_resolves_another_project(
    isolated_storage,
    project_id,
):
    create_run(isolated_storage, PROJECT_A)
    create_run(isolated_storage, PROJECT_B)

    expected = (
        isolated_storage
        / project_id
        / "runs"
        / RUN_ID
    )

    result = RunWorker.get_run_status(
        RUN_ID,
        project_id=project_id,
    )

    assert result is not None
    assert result["project_id"] == project_id
    assert result["run_id"] == RUN_ID

    from pathlib import Path

    assert Path(
        result["output_directory"]
    ).resolve() == expected.resolve()


def test_missing_project_does_not_fall_back(
    isolated_storage,
):
    create_run(isolated_storage, PROJECT_A)

    result = RunWorker.get_run_status(
        RUN_ID,
        project_id=PROJECT_B,
    )

    assert result is None


@pytest.mark.parametrize(
    "project_id",
    [
        "../",
        "..",
        "CON",
        "NUL",
        "AUX",
        "prj_invalid/child",
        "prj_invalid\\child",
    ],
)
def test_invalid_project_identity_is_rejected(
    isolated_storage,
    project_id,
):
    create_run(isolated_storage, PROJECT_A)

    assert RunWorker.get_run_status(
        RUN_ID,
        project_id=project_id,
    ) is None


def test_missing_run_returns_none(
    isolated_storage,
):
    assert RunWorker.get_run_status(
        RUN_ID,
        project_id=PROJECT_A,
    ) is None


def test_incomplete_output_is_not_reported_completed(
    isolated_storage,
):
    create_run(isolated_storage, PROJECT_A)

    result = RunWorker.get_run_status(
        RUN_ID,
        project_id=PROJECT_A,
    )

    assert result is not None
    assert result["status"] != "COMPLETED"


def test_historical_recovery_does_not_pollute_run_cache(
    isolated_storage,
):
    create_run(isolated_storage, PROJECT_A)
    create_run(isolated_storage, PROJECT_B)

    RunWorker.get_run_status(
        RUN_ID,
        project_id=PROJECT_A,
    )

    RunWorker.get_run_status(
        RUN_ID,
        project_id=PROJECT_B,
    )

    assert RUN_ID not in services.RUN_STATES


def create_live_state(project_id, status):
    state = {
        "run_id": RUN_ID,
        "project_id": project_id,
        "status": status,
        "error": (
            "Execution failed."
            if status == "FAILED"
            else None
        ),
    }

    services.RUN_STATES[RUN_ID] = state
    return state


def test_live_failure_without_run_directory_is_preserved(
    isolated_storage,
):
    state = create_live_state(PROJECT_A, "FAILED")

    result = RunWorker.get_run_status(
        RUN_ID,
        project_id=PROJECT_A,
    )

    assert result == state
    assert result["status"] == "FAILED"


def test_live_failure_takes_precedence_over_disk(
    isolated_storage,
):
    create_run(isolated_storage, PROJECT_A)
    state = create_live_state(PROJECT_A, "FAILED")

    result = RunWorker.get_run_status(
        RUN_ID,
        project_id=PROJECT_A,
    )

    assert result == state
    assert result["error"] == "Execution failed."


def test_live_failure_does_not_leak_to_other_project(
    isolated_storage,
):
    create_live_state(PROJECT_A, "FAILED")

    assert RunWorker.get_run_status(
        RUN_ID,
        project_id=PROJECT_B,
    ) is None


def test_live_and_disk_identity_collision_is_ambiguous(
    isolated_storage,
):
    create_run(isolated_storage, PROJECT_B)
    create_live_state(PROJECT_A, "RUNNING")

    with pytest.raises(HTTPException) as exc:
        RunWorker.get_run_status(RUN_ID)

    assert exc.value.status_code == 409


def test_live_running_state_is_preserved(
    isolated_storage,
):
    state = create_live_state(PROJECT_A, "RUNNING")

    assert RunWorker.get_run_status(
        RUN_ID,
        project_id=PROJECT_A,
    ) == state


def test_live_completed_state_cannot_bypass_disk_checks(
    isolated_storage,
):
    create_run(isolated_storage, PROJECT_A)
    create_live_state(PROJECT_A, "COMPLETED")

    result = RunWorker.get_run_status(
        RUN_ID,
        project_id=PROJECT_A,
    )

    assert result is not None
    assert result["status"] != "COMPLETED"


def test_scoped_disk_run_does_not_use_other_live_state(
    isolated_storage,
):
    create_run(isolated_storage, PROJECT_B)
    create_live_state(PROJECT_A, "FAILED")

    result = RunWorker.get_run_status(
        RUN_ID,
        project_id=PROJECT_B,
    )

    assert result is not None
    assert result["project_id"] == PROJECT_B
    assert result["error"] != "Execution failed."
