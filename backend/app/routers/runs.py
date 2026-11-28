"""
backend/app/routers/runs.py
============================
Router for initiating analysis runs and checking run execution status.
"""

from fastapi import APIRouter, HTTPException, status
from ..schemas import RunCreateRequest, RunStatusResponse, AnalysisReadinessResponse
from ..services import RunWorker, ProjectStore

router = APIRouter(prefix="/api/runs", tags=["Runs"])


@router.get("/readiness/{project_id}", response_model=AnalysisReadinessResponse)
def get_analysis_readiness(
    project_id: str,
    analysis_mode: str = "wetland",
    use_observation_dates: bool = True,
):
    project = ProjectStore.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    try:
        return RunWorker.get_readiness(project_id, {
            "analysis_mode": analysis_mode,
            "use_observation_dates": use_observation_dates,
        })
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post("", response_model=RunStatusResponse, status_code=status.HTTP_202_ACCEPTED)
def launch_run(payload: RunCreateRequest):
    project = ProjectStore.get_project(payload.project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {payload.project_id} not found")

    effective_mode = "wetland"
    if payload.config:
        if payload.config.analysis_mode:
            effective_mode = payload.config.analysis_mode.strip().lower()
        if payload.config.analysisMode:
            effective_mode = payload.config.analysisMode.strip().lower()
    elif project.get("analysis_mode"):
        effective_mode = project["analysis_mode"].strip().lower()

    config_dict = payload.config.model_dump() if payload.config else None

    try:
        RunWorker.preflight_run(payload.project_id, config_dict)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        ) from exc

    print(
        "[GIS DEBUG] backend payload manual_exclusions:",
        payload.config.manual_exclusions if payload.config else None,
    )

    print(
        "[GIS DEBUG] backend config dict:",
        payload.config.model_dump() if payload.config else None,
    )
    try:
        run_info = RunWorker.start_run(payload.project_id, config_dict)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )
    return run_info


@router.get("/{run_id}", response_model=RunStatusResponse)
def get_run_status(run_id: str):
    run_info = RunWorker.get_run_status(run_id)
    if not run_info:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return run_info
