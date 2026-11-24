"""Exercise the FastAPI boundary across two independent API processes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))


def serve(root, mode):
    from fastapi.testclient import TestClient
    from app import services
    services.STORAGE_DIR = root
    services.RUN_INDEX_DIR = root / "run_index"
    os.environ["ENABLE_STAGE_RESUME"] = "true"
    from app.main import app
    from gis_engine.core.checkpoints import Checkpoints
    if mode == "crash":
        original = Checkpoints.run

        def stop_at_metrics(self, stage, work, outputs, warnings):
            if stage == "METRICS_QA":
                os._exit(77)
            return original(self, stage, work, outputs, warnings)

        Checkpoints.run = stop_at_metrics
    with TestClient(app) as client:
        if mode == "crash":
            p = client.post("/api/projects", json={"name": "restart test", "analysis_mode": "vegetation"}).json()
            response = client.post("/api/runs", json={"project_id": p["id"], "config": {"analysis_mode": "vegetation"}})
            assert response.status_code == 202, response.text
            time.sleep(60)
            raise AssertionError("Worker never reached metrics")
        run_id = next(services.RUN_INDEX_DIR.glob("*.json")).stem
        url = "/api/runs/" + run_id
        state = client.get(url).json()
        assert state["status"] == "INTERRUPTED", state
        info = client.get(url + "/resume-check").json()
        assert info["next_stage"] == "METRICS_QA", info
        config = Path(services.RUN_STATES[run_id]["checkpoint_arguments"]["config_path"])
        original = config.read_bytes()
        config.write_bytes(original + b" ")
        assert client.post(url + "/resume").status_code == 409
        config.write_bytes(original)
        res = client.post(url + "/resume")
        assert res.status_code == 202, res.text
        assert res.json()["run_id"] == run_id
        assert client.post(url + "/resume").status_code == 409
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            state = client.get(url).json()
            if state["status"] in services.TERMINAL_RUN_STATUSES:
                break
            time.sleep(0.1)
        assert state["status"] == "COMPLETED" and state["resume_count"] == 1, state


class ResumeApiTest(unittest.TestCase):
    def test_restart_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            for mode, code in [("crash", 77), ("resume", 0)]:
                child = subprocess.run([sys.executable, __file__, temp, mode],
                                       timeout=150, capture_output=True)
                self.assertEqual(child.returncode, code, child.stderr[-4000:])


if __name__ == "__main__":
    if len(sys.argv) == 3:
        serve(Path(sys.argv[1]), sys.argv[2])
    else:
        unittest.main()
