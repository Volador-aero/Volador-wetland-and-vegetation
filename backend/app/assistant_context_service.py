
import json
import math
from pathlib import Path
from typing import Any

from app.services import ProjectStore, STORAGE_DIR, RunWorker


def _is_safe_id(value: str) -> bool:
    """
    Check that an ID is a safe folder name.
    """
    return (
        isinstance(value, str)
        and bool(value)
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value
    )


def _safe_number(value: Any):
    """
    Return a number only if it is valid and finite.
    """
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)) and math.isfinite(value):
        return value

    return None


def _get_dataset_status(project_dir: Path) -> dict:
    """
    Inspect the actual T1 and T2 dataset folders.
    """
    dataset_status = {}

    for name in ("t1", "t2"):
        folder = project_dir / "datasets" / name

        files = []

        if folder.is_dir():
            files = [
                item.name
                for item in folder.iterdir()
                if item.is_file()
            ]

        dataset_status[name] = {
            "folder_exists": folder.is_dir(),
            "file_count": len(files),
        }

    return dataset_status


def _get_preparation_status(project_dir: Path) -> dict:
    """
    Read selected preparation metadata without exposing file paths.
    """
    preparation_file = project_dir / "preparation.json"

    if not preparation_file.is_file():
        return {"available": False}

    try:
        with preparation_file.open("r", encoding="utf-8") as file:
            data = json.load(file)

        metrics = data.get("metrics", {})

        return {
            "available": True,
            "status": data.get("status"),
            "is_locked": data.get("is_locked"),
            "area_basis": metrics.get("area_basis"),
            "source_pixels": _safe_number(
                metrics.get("source_pixels")
            ),
            "effective_pixels": _safe_number(
                metrics.get("effective_pixels")
            ),
            "effective_area_ha": (
                _safe_number(metrics.get("effective_ha"))
                if metrics.get("area_basis") != "PIXEL_COUNT_ONLY"
                else None
            ),
        }

    except (OSError, json.JSONDecodeError):
        return {"available": False, "read_error": True}


def _get_analysis_context(
    project_id: str,
    run_id: str,
) -> dict:
    """
    Read analysis results only after verifying project/run ownership.
    """
    if not _is_safe_id(project_id) or not _is_safe_id(run_id):
        return {"available": False}

    run_info = RunWorker.get_run_status(run_id)

    if not run_info or run_info.get("project_id") != project_id:
        return {"available": False}

    if run_info.get("status") != "COMPLETED":
        return {
            "available": False,
            "run_status": run_info.get("status"),
        }

    run_dir = (
        STORAGE_DIR / "projects" / project_id / "runs" / run_id
    ).resolve()

    expected_root = (
        STORAGE_DIR / "projects" / project_id / "runs"
    ).resolve()

    if not run_dir.is_relative_to(expected_root):
        return {"available": False}

    candidates = [
        run_dir / "analysis_result.json",
        run_dir / run_id / "analysis_result.json",
    ]

    analysis_file = next(
        (path for path in candidates if path.is_file()),
        None,
    )

    if analysis_file is None:
        return {"available": False}

    try:
        with analysis_file.open("r", encoding="utf-8") as file:
            data = json.load(file)

    except (OSError, json.JSONDecodeError):
        return {"available": False}

    if not isinstance(data, dict):
        return {"available": False}

    if data.get("run_id") != run_id:
        return {"available": False}

    spectral_indices = {}

    for name in ("ndvi", "gndvi", "ndre"):
        index_data = data.get("spectral_indices", {}).get(name)

        if not isinstance(index_data, dict):
            continue

        spectral_indices[name] = {
            key: _safe_number(index_data.get(key))
            for key in ("mean", "median", "std", "p95")
            if _safe_number(index_data.get(key)) is not None
        }

    vegetation = {}

    classes = data.get("vegetation", {}).get("classes", {})

    if isinstance(classes, dict):
        for name, details in classes.items():
            if not isinstance(details, dict):
                continue

            percentage = _safe_number(details.get("percentage"))

            if percentage is not None:
                vegetation[str(name)] = {
                    "percentage": percentage,
                }

    spatial_data = data.get("spatial", {})

    spatial = {
        "dimensions": spatial_data.get("dimensions"),
        "total_pixels": _safe_number(
            spatial_data.get("total_pixels")
        ),
        "valid_pixels": _safe_number(
            spatial_data.get("valid_pixels")
        ),
        "physical_area_available": spatial_data.get(
            "physical_area_available"
        ),
        "area_basis": spatial_data.get("area_basis"),
        "total_area_ha": (
            _safe_number(spatial_data.get("total_area_ha"))
            if spatial_data.get("physical_area_available") is True
            else None
        ),
    }

    return {
        "available": True,
        "run_id": run_id,
        "analysis_mode": data.get("analysis_mode"),
        "status": data.get("status"),
        "spectral_indices": spectral_indices,
        "vegetation": vegetation,
        "spatial": spatial,
    }


def build_assistant_context(
    project_id: str | None,
    run_id: str | None = None,
) -> dict:
    """
    Build a compact, read-only context for the AETHRA Assistant.
    """
    if not project_id or not _is_safe_id(project_id):
        return {
            "project_available": False,
            "message": "No valid project was selected.",
        }

    project = ProjectStore.get_project(project_id)

    if not project:
        return {
            "project_available": False,
            "message": "The selected project was not found.",
        }

    project_dir = (
        STORAGE_DIR / "projects" / project_id
    ).resolve()

    expected_root = (
        STORAGE_DIR / "projects"
    ).resolve()

    if not project_dir.is_relative_to(expected_root):
        return {
            "project_available": False,
            "message": "The selected project is unavailable.",
        }

    selected_run_id = run_id or project.get("latest_run_id")

    context = {
        "project_available": True,
        "project": {
            "name": project.get("name"),
            "description": project.get("description"),
            "analysis_mode": project.get("analysis_mode"),
            "aoi_name": project.get("aoi_name"),
            "status": project.get("status"),
            "preparation_status": project.get(
                "preparation_status"
            ),
            "datasets_ready_flag": project.get("datasets_ready"),
        },
        "datasets": _get_dataset_status(project_dir),
        "preparation": _get_preparation_status(project_dir),
        "analysis": {"available": False},
    }

    if selected_run_id:
        context["analysis"] = _get_analysis_context(
            project_id,
            selected_run_id,
        )

    return context
