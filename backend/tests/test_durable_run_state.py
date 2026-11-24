"""Focused tests for durable, constant-time run-state recovery."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "Engine_module" / "src"))

from app import services
from gis_engine.core.exceptions import OutputError
from gis_engine.core.pipeline import _prepare_run_directory


class RecordingExecutor:
    def __init__(self):
        self.submissions = []
        self.state_existed_at_submit = False

    def submit(self, function, run_id, project, output_dir, config_overrides):
        self.state_existed_at_submit = (output_dir / "run_state.json").is_file()
        self.submissions.append(
            (function, run_id, project, output_dir, config_overrides)
        )


class DurableRunStateTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.storage_root = Path(self.temporary_directory.name)
        self.storage_patch = patch.object(services, "STORAGE_DIR", self.storage_root)
        self.index_patch = patch.object(
            services, "RUN_INDEX_DIR", self.storage_root / "run_index"
        )
        self.storage_patch.start()
        self.index_patch.start()
        services.RUN_STATES.clear()

    def tearDown(self):
        services.RUN_STATES.clear()
        self.index_patch.stop()
        self.storage_patch.stop()
        self.temporary_directory.cleanup()

    def create_durable_run(self, status="PREPROCESSING"):
        run_id = "run_durable_test"
        project_id = "project_durable_test"
        run_dir = self.storage_root / "projects" / project_id / "runs" / run_id
        run_dir.mkdir(parents=True)
        state = {
            "run_id": run_id,
            "project_id": project_id,
            "status": status,
            "progress": 25,
            "current_step": "Spatial Alignment & Preprocessing",
            "created_at": "2026-01-01T00:00:00Z",
            "started_at": "2026-01-01T00:00:01Z",
            "completed_at": None,
            "message": "Analysis is processing",
            "error": None,
            "output_directory": str(run_dir),
            "config_overrides": {"analysis_mode": "wetland"},
        }
        services._write_json_atomically(run_dir / "run_state.json", state)
        services._write_json_atomically(
            self.storage_root / "run_index" / f"{run_id}.json",
            {
                "run_id": run_id,
                "project_id": project_id,
                "output_directory": str(run_dir),
            },
        )
        return run_id, run_dir

    def project_patches(self, executor):
        project = {
            "id": "project_durable_test",
            "t1_dir": "unused-t1",
            "t2_dir": "unused-t2",
        }
        preparation = {
            "status": "COMMITTED",
            "is_locked": True,
            "inclusion": None,
            "manual_exclusions": [],
        }
        return (
            patch.object(services.ProjectStore, "get_project", return_value=project),
            patch.object(
                services.ProjectStore, "load_preparation", return_value=preparation
            ),
            patch.object(services, "executor", executor),
        )

    def test_start_run_persists_queued_state_before_worker_submission(self):
        executor = RecordingExecutor()
        project_patch, preparation_patch, executor_patch = self.project_patches(executor)
        with project_patch, preparation_patch, executor_patch:
            started = services.RunWorker.start_run(
                "project_durable_test", {"analysis_mode": "wetland"}
            )

        run_dir = Path(started["output_directory"])
        persisted = json.loads(
            (run_dir / "run_state.json").read_text(encoding="utf-8")
        )
        index_path = self.storage_root / "run_index" / f"{started['run_id']}.json"
        self.assertEqual(persisted["status"], "QUEUED")
        self.assertEqual(
            persisted["config_overrides"], {"analysis_mode": "wetland"}
        )
        self.assertTrue(index_path.is_file())
        self.assertTrue(executor.state_existed_at_submit)
        self.assertEqual(len(executor.submissions), 1)

    def test_non_terminal_run_becomes_interrupted_after_memory_loss(self):
        run_id, run_dir = self.create_durable_run()
        recovered = services.RunWorker.get_run_status(run_id)

        self.assertEqual(recovered["status"], "INTERRUPTED")
        self.assertEqual(
            recovered["last_confirmed_step"], "Spatial Alignment & Preprocessing"
        )
        persisted = json.loads(
            (run_dir / "run_state.json").read_text(encoding="utf-8")
        )
        self.assertEqual(persisted["status"], "INTERRUPTED")
        self.assertTrue(persisted["completed_at"])

    def test_terminal_run_is_returned_without_status_guessing(self):
        run_id, run_dir = self.create_durable_run(status="COMPLETED")
        completed_state = json.loads(
            (run_dir / "run_state.json").read_text(encoding="utf-8")
        )
        completed_state.update(
            {
                "progress": 100,
                "current_step": "COMPLETED",
                "completed_at": "2026-01-01T00:02:00Z",
            }
        )
        services._write_json_atomically(run_dir / "run_state.json", completed_state)
        recovered = services.RunWorker.get_run_status(run_id)

        self.assertEqual(recovered["status"], "COMPLETED")
        self.assertEqual(recovered["progress"], 100)
        self.assertFalse((run_dir / "run_state.json.tmp").exists())

    def test_retry_creates_new_run_with_original_run_lineage(self):
        original_run_id, _ = self.create_durable_run(status="FAILED")
        executor = RecordingExecutor()
        project_patch, preparation_patch, executor_patch = self.project_patches(executor)
        with project_patch, preparation_patch, executor_patch:
            retried = services.RunWorker.retry_run(original_run_id)

        self.assertNotEqual(retried["run_id"], original_run_id)
        self.assertEqual(retried["retry_of"], original_run_id)
        self.assertEqual(
            retried["config_overrides"], {"analysis_mode": "wetland"}
        )
        self.assertTrue(executor.state_existed_at_submit)

    def test_project_paths_are_rebased_to_current_storage_root(self):
        project_id = "project_from_another_machine"
        stale_projects = {
            project_id: {
                "id": project_id,
                "t1_dir": r"C:\Users\OtherUser\old-project\datasets\t1",
                "t2_dir": r"C:\Users\OtherUser\old-project\datasets\t2",
            }
        }
        services._write_json_atomically(
            self.storage_root / "projects.json", stale_projects
        )

        loaded = services.ProjectStore.load_projects()

        expected_root = self.storage_root / "projects" / project_id / "datasets"
        self.assertEqual(loaded[project_id]["t1_dir"], str(expected_root / "t1"))
        self.assertEqual(loaded[project_id]["t2_dir"], str(expected_root / "t2"))
        self.assertTrue((expected_root / "t1").is_dir())
        self.assertTrue((expected_root / "t2").is_dir())

    def test_engine_accepts_directory_reserved_by_durable_state(self):
        run_dir = self.storage_root / "reserved-run"
        run_dir.mkdir()
        (run_dir / "run_state.json").write_text("{}", encoding="utf-8")

        _prepare_run_directory(run_dir)

        self.assertTrue((run_dir / "run_state.json").is_file())

    def test_engine_rejects_directory_with_previous_analysis_output(self):
        run_dir = self.storage_root / "existing-run"
        run_dir.mkdir()
        (run_dir / "run_state.json").write_text("{}", encoding="utf-8")
        (run_dir / "metrics").mkdir()

        with self.assertRaises(OutputError):
            _prepare_run_directory(run_dir)


if __name__ == "__main__":
    unittest.main()
