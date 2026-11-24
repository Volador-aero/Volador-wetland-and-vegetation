"""Build safe, compact, read-only context for the AETHRA assistant."""

import json
import math
from pathlib import Path
from typing import Any

from app.services import ProjectStore, STORAGE_DIR, RunWorker


def _is_safe_id(value: Any) -> bool:
    """Accept only simple folder identifiers, never paths."""
    return (
        isinstance(value, str)
        and bool(value.strip())
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value
        and "\x00" not in value
    )


def _safe_number(value: Any):
    """Return finite numeric values, excluding booleans."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    return None


def _is_within(path: Path, root: Path) -> bool:
    """Resolve paths (including symlinks) and verify containment."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def _get_dataset_status(project_dir: Path) -> dict:
    """Count regular files in the project's T1/T2 folders safely."""
    result = {}
    for name in ("t1", "t2"):
        folder = project_dir / "datasets" / name
        exists = False
        count = 0
        try:
            exists = folder.is_dir() and _is_within(folder, project_dir)
            if exists:
                count = sum(1 for item in folder.iterdir() if item.is_file())
        except OSError:
            # A missing, unreadable, or concurrently removed folder is unavailable.
            exists = False
            count = 0
        result[name] = {"folder_exists": exists, "file_count": count}
    return result


def _get_preparation_status(project_dir: Path) -> dict:
    """Read preparation metadata while tolerating malformed JSON structures."""
    preparation_file = project_dir / "preparation.json"
    if not _is_within(preparation_file, project_dir):
        return {"available": False}
    try:
        if not preparation_file.is_file():
            return {"available": False}
        with preparation_file.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"available": False, "read_error": True}

    if not isinstance(data, dict):
        return {"available": False, "read_error": True}

    metrics = data.get("metrics")
    # A preparation record is only considered available when its metrics
    # section has the expected object structure. Treat null, lists, strings,
    # and missing metrics as malformed rather than silently accepting them.
    if not isinstance(metrics, dict):
        return {"available": False, "read_error": True}

    area_basis = metrics.get("area_basis")
    physical_area = (
        _safe_number(metrics.get("effective_ha"))
        if area_basis != "PIXEL_COUNT_ONLY"
        else None
    )
    return {
        "available": True,
        "status": data.get("status") if isinstance(data.get("status"), str) else None,
        "is_locked": data.get("is_locked") if isinstance(data.get("is_locked"), bool) else None,
        "area_basis": area_basis if isinstance(area_basis, str) else None,
        "source_pixels": _safe_number(metrics.get("source_pixels")),
        "effective_pixels": _safe_number(metrics.get("effective_pixels")),
        "effective_area_ha": physical_area,
    }


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _get_analysis_context(project_id: str, run_id: str) -> dict:
    """Read a completed run's result only when ownership and paths are valid."""
    unavailable = {"available": False}
    if not _is_safe_id(project_id) or not _is_safe_id(run_id):
        return unavailable

    try:
        run_info = RunWorker.get_run_status(run_id)
    except Exception:
        return unavailable
    if not isinstance(run_info, dict) or run_info.get("project_id") != project_id:
        return unavailable

    run_status = run_info.get("status")
    if run_status != "COMPLETED":
        return {"available": False, "run_status": run_status if isinstance(run_status, str) else None}

    project_root = (Path(STORAGE_DIR) / "projects" / project_id).resolve()
    runs_root = project_root / "runs"
    run_dir = runs_root / run_id
    if not _is_within(run_dir, runs_root):
        return unavailable

    candidates = (run_dir / "analysis_result.json", run_dir / run_id / "analysis_result.json")
    analysis_file = next(
        (candidate for candidate in candidates if candidate.is_file() and _is_within(candidate, run_dir)),
        None,
    )
    if analysis_file is None:
        return unavailable

    try:
        with analysis_file.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return unavailable

    if not isinstance(data, dict) or data.get("run_id") != run_id:
        return unavailable

    spectral_indices = {}
    raw_indices = _as_dict(data.get("spectral_indices"))
    for name in ("ndvi", "gndvi", "ndre"):
        index_data = _as_dict(raw_indices.get(name))
        if index_data:
            values = {
                key: number
                for key in ("mean", "median", "std", "p95")
                if (number := _safe_number(index_data.get(key))) is not None
            }
            if values:
                spectral_indices[name] = values

    vegetation = {}
    classes = _as_dict(_as_dict(data.get("vegetation")).get("classes"))
    for name, details in classes.items():
        details = _as_dict(details)
        percentage = _safe_number(details.get("percentage"))
        if percentage is not None:
            vegetation[str(name)] = {"percentage": percentage}

    spatial_data = _as_dict(data.get("spatial"))
    physical_area_available = spatial_data.get("physical_area_available") is True
    dimensions = spatial_data.get("dimensions")
    if not isinstance(dimensions, (dict, list, str, int, float)):
        dimensions = None

    spatial = {
        "dimensions": dimensions,
        "total_pixels": _safe_number(spatial_data.get("total_pixels")),
        "valid_pixels": _safe_number(spatial_data.get("valid_pixels")),
        "physical_area_available": physical_area_available,
        "area_basis": spatial_data.get("area_basis") if isinstance(spatial_data.get("area_basis"), str) else None,
        "total_area_ha": _safe_number(spatial_data.get("total_area_ha")) if physical_area_available else None,
    }
    return {
        "available": True,
        "run_id": run_id,
        "analysis_mode": data.get("analysis_mode") if isinstance(data.get("analysis_mode"), str) else None,
        "status": data.get("status") if isinstance(data.get("status"), str) else None,
        "spectral_indices": spectral_indices,
        "vegetation": vegetation,
        "spatial": spatial,
    }


def build_assistant_context(project_id: str | None, run_id: str | None = None) -> dict:
    """Build a compact context; malformed/missing project data never crashes chat."""
    if not _is_safe_id(project_id):
        return {"project_available": False, "message": "No valid project was selected."}

    try:
        project = ProjectStore.get_project(project_id)
    except Exception:
        project = None
    if not isinstance(project, dict):
        return {"project_available": False, "message": "The selected project was not found."}

    projects_root = (Path(STORAGE_DIR) / "projects").resolve()
    project_dir = projects_root / project_id
    if not _is_within(project_dir, projects_root) or not project_dir.is_dir():
        return {"project_available": False, "message": "The selected project is unavailable."}

    selected_run_id = run_id or project.get("latest_run_id")
    context = {
        "project_available": True,
        "project": {
            "name": project.get("name") if isinstance(project.get("name"), str) else None,
            "description": project.get("description") if isinstance(project.get("description"), str) else None,
            "analysis_mode": project.get("analysis_mode") if isinstance(project.get("analysis_mode"), str) else None,
            "aoi_name": project.get("aoi_name") if isinstance(project.get("aoi_name"), str) else None,
            "status": project.get("status") if isinstance(project.get("status"), str) else None,
            "preparation_status": project.get("preparation_status") if isinstance(project.get("preparation_status"), str) else None,
            "datasets_ready_flag": project.get("datasets_ready") if isinstance(project.get("datasets_ready"), bool) else None,
        },
        "datasets": _get_dataset_status(project_dir),
        "preparation": _get_preparation_status(project_dir),
        "analysis": {"available": False},
    }
    if selected_run_id:
        context["analysis"] = _get_analysis_context(project_id, selected_run_id)
    return context
