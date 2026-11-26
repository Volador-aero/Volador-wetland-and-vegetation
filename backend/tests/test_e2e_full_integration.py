"""
backend/tests/test_e2e_full_integration.py
============================================
Comprehensive End-to-End Integration and Scientific Regression Test Suite
for the GIS Wetland Degradation & Depletion Engine.
"""

import sys
import time
import json
from pathlib import Path
from fastapi.testclient import TestClient
import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
ENGINE_SRC = BACKEND_DIR / "Engine_module" / "src"
sys.path.insert(0, str(ENGINE_SRC))

from app.main import app
from gis_engine.api.service import analyze_wetland

client = TestClient(app)


def test_full_e2e_user_workflow():
    """
    Simulates complete user journey from project creation to dataset upload,
    dataset validation, background run execution, layer retrieval,
    results parsing, and PDF report retrieval.
    """
    # 1. User creates project
    res = client.post("/api/projects", json={
        "name": "E2E Test Project",
        "description": "Full end-to-end user workflow test",
        "aoi_name": "Sample Wetland"
    })
    assert res.status_code == 201
    project = res.json()
    project_id = project["id"]

    # 2. User validates datasets
    res = client.post(f"/api/projects/{project_id}/datasets/validate")
    assert res.status_code == 200
    val_data = res.json()
    assert val_data["status"] in ["PASS", "WARNING"]
    assert val_data["valid"] is True
    assert val_data["metadata"] is not None

    # 3. User launches background pipeline execution
    res = client.post("/api/runs", json={"project_id": project_id})
    assert res.status_code == 202
    run_data = res.json()
    run_id = run_data["run_id"]

    # 4. Frontend polls run status until completion
    max_wait = 300  # seconds
    start_time = time.time()
    completed = False

    while time.time() - start_time < max_wait:
        res = client.get(f"/api/runs/{run_id}")
        assert res.status_code == 200
        status_info = res.json()

        if status_info["status"] == "COMPLETED":
            completed = True
            break
        elif status_info["status"] == "FAILED":
            pytest.fail(f"Run failed: {status_info.get('error')}")

        time.sleep(0.5)

    assert completed is True, f"Run did not complete within {max_wait} seconds"

    # 5. User retrieves analysis results & scientific metrics
    res = client.get(f"/api/runs/{run_id}/results")
    assert res.status_code == 200
    results = res.json()
    assert results["status"] in ["SUCCESS", "WARNING"]
    assert "degradation" in results
    assert "cva" in results
    assert "validation" in results

    # 6. User retrieves layer catalog and requests PNG raster images
    res = client.get(f"/api/runs/{run_id}/layers")
    assert res.status_code == 200
    layers = res.json()
    assert len(layers) >= 5

    for layer in layers:
        if layer["ready"]:
            img_res = client.get(layer["url"])
            assert img_res.status_code == 200
            assert img_res.headers["content-type"] == "image/png"
            assert len(img_res.content) > 0

    # 7. User retrieves PDF cartographic report (simple and technical)
    res = client.get(f"/api/runs/{run_id}/report/pdf")
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"
    assert len(res.content) > 1000  # Non-empty PDF document

    res_tech = client.get(f"/api/runs/{run_id}/report/pdf/technical")
    assert res_tech.status_code == 200
    assert res_tech.headers["content-type"] == "application/pdf"
    assert len(res_tech.content) > 1000  # Non-empty technical PDF document

    # Clean up project
    _wait_for_project_workers(project_id)
    client.delete(f"/api/projects/{project_id}")


def test_scientific_regression_reproducibility(tmp_path):
    """
    Verifies that the integrated FastAPI execution reproduces the exact
    baseline scientific outputs of ProductionPipeline to within numerical tolerance.
    """
    sample_dir = BACKEND_DIR / "Engine_module" / "data" / "sample_scene"
    t1_dir = sample_dir / "t1"
    t2_dir = sample_dir / "t2"
    config_path = BACKEND_DIR / "Engine_module" / "configs" / "config.json"
    model_path = BACKEND_DIR / "Engine_module" / "models" / "best_finetuned_model.onnx"
    wetland_boundary = sample_dir / "wetland_boundary.geojson"

    # Baseline direct engine execution
    baseline_out = tmp_path / "baseline_out"
    res_direct = analyze_wetland(
        t1_scene=t1_dir,
        t2_scene=t2_dir,
        t1_date="2023-08-14",
        t2_date="2023-08-28",
        config_path=config_path,
        model_path=model_path,
        output_dir=baseline_out,
        wetland_boundary=wetland_boundary,
        run_id="baseline_run"
    )

    assert res_direct.status in ["SUCCESS", "WARNING"], (
        f"Direct engine failed: {res_direct.status}"
    )

    # API execution
    proj_res = client.post("/api/projects", json={"name": "Regression Test Proj"})
    proj_id = proj_res.json()["id"]
    client.post(f"/api/projects/{proj_id}/datasets/validate")
    run_res = client.post("/api/runs", json={"project_id": proj_id})
    run_id = run_res.json()["run_id"]

    for _ in range(60):
        s = client.get(f"/api/runs/{run_id}").json()
        if s["status"] in ["COMPLETED", "FAILED"]:
            break
        time.sleep(0.5)

    _wait_for_project_workers(proj_id)
    terminal_response = client.get(f"/api/runs/{run_id}")
    assert terminal_response.status_code == 200
    terminal = terminal_response.json()
    assert terminal["status"] == "COMPLETED", terminal

    results_response = client.get(f"/api/runs/{run_id}/results")
    assert results_response.status_code == 200
    api_results = results_response.json()
    assert api_results["status"] in ["SUCCESS", "WARNING"]

    # Read baseline analysis_result.json written by direct engine call
    baseline_json_path = baseline_out / "baseline_run" / "analysis_result.json"
    with open(baseline_json_path, "r") as f:
        direct_json = json.load(f)

    # Compare Metrics (Degradation & CVA)
    assert direct_json["degradation"]["area_ha"] == api_results["degradation"]["area_ha"]
    assert np.isclose(
        direct_json["cva"]["cva_magnitude"]["mean"],
        api_results["cva"]["cva_magnitude"]["mean"],
        atol=1e-4
    )
    assert direct_json["spatial"]["dimensions"] == api_results["spatial"]["dimensions"]

    _wait_for_project_workers(proj_id)
    client.delete(f"/api/projects/{proj_id}")


def _wait_for_project_workers(project_id, timeout=300):
    """Wait for project workers before removing test data."""
    from concurrent.futures import wait
    from app import services

    with services._PROJECT_LIFECYCLE_LOCK:
        futures = [
            future
            for run_id, future in services._RUN_FUTURES.items()
            if services.RUN_STATES.get(
                run_id, {}
            ).get("project_id") == project_id
        ]

    if futures:
        completed, pending = wait(
            futures, timeout=timeout
        )

        assert not pending, (
            f"Project {project_id} still has active workers."
        )

        for future in completed:
            future.result()

    services._assert_project_idle(project_id)
