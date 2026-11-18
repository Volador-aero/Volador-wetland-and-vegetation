r"""Setup and validate the Wetland GIS repository (Python 3.11-3.13).

Run from any working directory:
    py -3.11 environment.py --setup       # fix paths, install dependencies, check/fix demo model
    python environment.py --fix-paths     # repair projects.json only
    .\.venv\Scripts\python.exe environment.py --check  # read-only environment and data checks
    .\.venv\Scripts\python.exe environment.py --check --deep-model
    python environment.py --fix-model     # repair only known generated demo ONNX model
    python environment.py --self-test     # built-in regression tests

Path migration is intentionally restricted to the known per-project paths.
An original backup of projects.json is created before writing any changes.
No GeoTIFF files or dataset contents are modified. --fix-model makes a
backup and adjusts ONNX IR only for the repository's known single-Conv demo model;
unknown or trained ONNX models are never automatically rewritten.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
import venv
import warnings
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
BANDS = ("scene_R.tif", "scene_G.tif", "scene_NIR.tif", "scene_RE.tif")
PROJECT_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+\Z")


class Report:
    def __init__(self, quiet: bool = False):
        self.counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
        self.quiet = quiet

    def add(self, status: str, message: str) -> None:
        self.counts[status] += 1
        if not self.quiet:
            print(f"[{status}] {message}")

    def summary(self) -> bool:
        if not self.quiet:
            print("\n" + "=" * 52)
            print("ENVIRONMENT VALIDATION SUMMARY")
            print("=" * 52)
            for name in ("PASS", "WARN", "FAIL"):
                print(f"{name:5}: {self.counts[name]}")
        return self.counts["FAIL"] == 0


def python_supported(version) -> bool:
    return (3, 11) <= tuple(version[:2]) < (3, 14)


def storage_dir(root: Path) -> Path:
    return root / "backend" / "storage"


def projects_file(root: Path) -> Path:
    return storage_dir(root) / "projects.json"


def project_dataset_dir(root: Path, project_id: str, date: str) -> Path:
    if not PROJECT_ID_PATTERN.fullmatch(project_id):
        raise ValueError(f"Unsafe project identifier: {project_id!r}")
    if date not in ("t1", "t2"):
        raise ValueError(f"Unexpected date label: {date!r}")
    return storage_dir(root) / "projects" / project_id / "datasets" / date


def read_projects(root: Path, report: Report) -> dict | None:
    path = projects_file(root)
    if not path.exists():
        report.add("WARN", "No saved projects.json; nothing to migrate or validate")
        return None
    try:
        records = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        report.add("FAIL", f"Cannot read projects.json: {exc}")
        return None
    if not isinstance(records, dict):
        report.add("FAIL", "projects.json must contain a JSON object")
        return None
    for project_id, record in records.items():
        if not isinstance(project_id, str) or not PROJECT_ID_PATTERN.fullmatch(project_id):
            report.add("FAIL", f"Unsafe project ID in projects.json: {project_id!r}")
            return None
        if not isinstance(record, dict):
            report.add("FAIL", f"Project {project_id!r} is not a JSON object")
            return None
    return records


def relocate_project_path(original: str, root: Path, project_id: str) -> Path | None:
    """Rebase only references physically under backend/storage/projects/<id>.

    PureWindowsPath handles saved Windows absolute paths even when running on
    macOS/Linux; PurePosixPath supports the opposite direction.
    """
    if not isinstance(original, str) or not original.strip():
        return None
    saved = PureWindowsPath(original) if "\\" in original else PurePosixPath(original)
    parts = saved.parts
    normalized = [p.casefold() for p in parts]
    anchor = ("backend", "storage", "projects", project_id.casefold())
    for index in range(len(normalized) - len(anchor) + 1):
        if tuple(normalized[index:index + 4]) != anchor:
            continue
        suffix = parts[index + 4:]
        if any(part in ("", ".", "..") for part in suffix):
            return None
        return storage_dir(root) / "projects" / project_id / Path(*suffix)
    return None


def write_projects_safely(path: Path, records: dict) -> Path:
    """Keep a copy of the exact old file; replace JSON atomically."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    backup = path.with_name(f"projects.json.backup-{stamp}-{suffix}")
    shutil.copy2(path, backup)
    temporary = path.with_name(f".projects.json.{suffix}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(records, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return backup


def fix_project_paths(root: Path, report: Report) -> bool:
    """Repair configured paths and create empty local upload destinations.

    No data are fabricated or copied from another computer. Returns False on
    unsafe data or IO failures. Repeating the command is safe (idempotent).
    """
    records = read_projects(root, report)
    if records is None:
        return report.counts["FAIL"] == 0

    updated = 0
    pending_dirs: list[Path] = []
    for project_id, project in records.items():
        for date in ("t1", "t2"):
            field = f"{date}_dir"
            target = project_dataset_dir(root, project_id, date)
            pending_dirs.append(target)
            before = project.get(field)
            if before != str(target):
                project[field] = str(target)
                updated += 1
                report.add("PASS", f"Rebased {project_id}/{field} -> {target}")

        # Optional wetland boundary paths may point to the old repository.
        # Don't guess an unrelated external path or a missing resource.
        boundary = project.get("wetland_boundary")
        if isinstance(boundary, str) and boundary:
            target = relocate_project_path(boundary, root, project_id)
            if target is not None and boundary != str(target):
                project["wetland_boundary"] = str(target)
                updated += 1
                report.add("PASS", f"Rebased {project_id}/wetland_boundary -> {target}")
                if not target.is_file():
                    report.add("WARN", f"Boundary not present locally: {target}")
            elif target is None and not Path(boundary).is_file():
                report.add("WARN", f"External boundary path not changed: {project_id}")

    try:
        # Create upload destinations even when no local TIFF files exist yet.
        # Do not invent band files or change datasets_ready/status metadata.
        for directory in pending_dirs:
            directory.mkdir(parents=True, exist_ok=True)
        if updated:
            backup = write_projects_safely(projects_file(root), records)
            report.add("PASS", f"Saved project path changes; backup: {backup}")
        else:
            report.add("PASS", "Project configuration paths are already correct")
        report.add("PASS", f"Local upload folders checked for {len(records)} projects")
        return True
    except OSError as exc:
        report.add("FAIL", f"Cannot repair local configuration: {exc}")
        return False


def check_layout(root: Path, report: Report) -> None:
    report.add("PASS", f"Repository root: {root}")
    for name in (
        "README.md", "backend/app/main.py", "backend/app/services.py",
        "backend/Engine_module/src/gis_engine",
        "backend/Engine_module/configs",
    ):
        present = (root / name).exists()
        report.add("PASS" if present else "FAIL", f"{'Found' if present else 'Missing'} {name}")
    # This directory may be ignored by Git and created on first setup.
    report.add("PASS" if storage_dir(root).is_dir() else "WARN",
               "backend/storage exists" if storage_dir(root).is_dir()
               else "backend/storage absent; --setup will create it")


def check_python(report: Report) -> None:
    version = sys.version.split()[0]
    report.add("PASS" if python_supported(sys.version_info) else "FAIL",
               f"Python {version} ({'supported' if python_supported(sys.version_info) else 'unsupported'})")
    if sys.prefix == sys.base_prefix:
        report.add("WARN", "Python is not running inside a virtual environment")
    else:
        report.add("PASS", f"Virtual environment: {sys.prefix}")


def check_requirements(root: Path, report: Report) -> None:
    path = root / "requirements.txt"
    if not path.is_file():
        report.add("FAIL", "requirements.txt not found")
        return
    try:
        from packaging.requirements import Requirement
    except ImportError:
        report.add("FAIL", "packaging is needed for checking dependency versions")
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            requirement = Requirement(stripped)
            if requirement.marker and not requirement.marker.evaluate():
                continue
            installed = importlib.metadata.version(requirement.name)
            compatible = not requirement.specifier or installed in requirement.specifier
            report.add("PASS" if compatible else "FAIL",
                       f"{requirement.name}=={installed}"
                       if compatible else f"{requirement.name}: installed {installed}; expected {requirement.specifier}")
        except importlib.metadata.PackageNotFoundError:
            report.add("FAIL", f"Missing package: {requirement.name}")
        except Exception as exc:
            report.add("FAIL", f"Cannot check requirement {stripped!r}: {exc}")

    result = subprocess.run([sys.executable, "-m", "pip", "check"],
                            capture_output=True, text=True, check=False)
    if result.returncode == 0:
        report.add("PASS", "pip dependency compatibility check")
    else:
        details = (result.stdout or result.stderr).strip()[:600]
        report.add("FAIL", f"pip check: {details}")


def inspect_raster_dir(folder: Path, name: str, report: Report,
                       required: bool = False):
    """Check GeoTIFF metadata and a small sampled pixel window."""
    severity = "FAIL" if required else "WARN"
    if not folder.is_dir():
        report.add(severity, f"{name}: dataset directory missing")
        return None
    if not any(folder.glob("*.tif")) and not any(folder.glob("*.TIF")):
        report.add(severity, f"{name}: no GeoTIFF files uploaded yet")
        return None
    try:
        import rasterio
        from rasterio.errors import NotGeoreferencedWarning
    except ImportError:
        report.add("FAIL", "rasterio is missing; cannot validate rasters")
        return None

    grid = None
    for band in BANDS:
        path = folder / band
        if not path.is_file():
            report.add(severity, f"{name}: missing {band}")
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NotGeoreferencedWarning)
                with rasterio.open(path) as src:
                    if src.count < 1 or src.width < 1 or src.height < 1:
                        report.add(severity, f"{name}/{band}: empty raster")
                        continue
                    if src.crs is None or src.transform.is_identity:
                        report.add(severity, f"{name}/{band}: not georeferenced")
                        continue
                    signature = (src.crs, src.width, src.height, src.transform)
                    if grid is None:
                        grid = signature
                    elif signature != grid:
                        report.add(severity, f"{name}/{band}: CRS or pixel grid mismatch")
                        continue
                    sample = src.read(1, masked=True, out_shape=(min(64, src.height), min(64, src.width)))
                    if sample.count() == 0:
                        report.add(severity, f"{name}/{band}: no valid sampled pixels")
                    else:
                        report.add("PASS", f"{name}/{band}: {src.width}x{src.height}, {src.crs}")
        except Exception as exc:
            report.add(severity, f"{name}/{band}: unable to open raster: {exc}")
    return grid


def check_projects(root: Path, report: Report) -> None:
    records = read_projects(root, report)
    if records is None:
        return
    for project_id, project in records.items():
        grids = {}
        for date in ("t1", "t2"):
            field = f"{date}_dir"
            expected = project_dataset_dir(root, project_id, date)
            configured = project.get(field)
            if configured != str(expected):
                report.add("WARN", f"{project_id}/{field}: path not local. Run --fix-paths")
            # Validate the directory that the backend is intended to use.
            grids[date] = inspect_raster_dir(expected, f"{project_id}/{date}", report, required=False)
        if grids["t1"] is not None and grids["t2"] is not None:
            if grids["t1"] == grids["t2"]:
                report.add("PASS", f"{project_id}: temporal grids match")
            else:
                report.add("WARN", f"{project_id}: temporal grids need alignment")
        boundary = project.get("wetland_boundary")
        if isinstance(boundary, str) and boundary and not Path(boundary).is_file():
            report.add("WARN", f"{project_id}: configured boundary file missing")


def model_path(root: Path) -> Path:
    return root / "backend" / "Engine_module" / "models" / "best_finetuned_model.onnx"


def is_known_generated_demo(model) -> bool:
    """Recognize exactly the tiny 1x1 Conv model from scripts/create_onnx_model.py.

    NEVER lower the IR version of a real/trained model merely to bypass an error.
    """
    graph = model.graph
    opsets = {entry.domain or "": entry.version for entry in model.opset_import}
    return (
        model.producer_name == "gis_wetland_engine"
        and graph.name == "WetlandHealthClassifier"
        and len(graph.node) == 1
        and graph.node[0].op_type == "Conv"
        and list(graph.node[0].input) == ["input", "conv_weight", "conv_bias"]
        and list(graph.node[0].output) == ["output"]
        and len(graph.initializer) == 2
        and {item.name for item in graph.initializer} == {"conv_weight", "conv_bias"}
        and not graph.sparse_initializer
        and not any(item.external_data for item in graph.initializer)
        and len(graph.input) == 1
        and graph.input[0].name == "input"
        and len(graph.output) == 1
        and graph.output[0].name == "output"
        and opsets == {"": 13}
    )


def probe_model_session(path: Path, infer: bool = False):
    """Use the same CPU runtime as the backend; structural ONNX checks aren't enough."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if not inputs or not outputs:
        raise ValueError("ONNX model has no input/output tensors")
    inp, out = inputs[0].shape, outputs[0].shape
    if len(inp) != 4 or inp[1] != 5:
        raise ValueError(f"Expected NCHW input with 5 channels, got {inp}")
    if len(out) != 4 or out[1] != 4:
        raise ValueError(f"Expected output with 4 classes, got {out}")

    if infer:
        import numpy as np

        # This is a smoke test, not a check of scientific model accuracy.
        batch = int(inp[0]) if isinstance(inp[0], int) else 1
        height = int(inp[2]) if isinstance(inp[2], int) else 64
        width = int(inp[3]) if isinstance(inp[3], int) else 64
        # Refuse unexpected fixed dimensions to avoid excessive allocations.
        if batch != 1 or not (1 <= height <= 1024) or not (1 <= width <= 1024):
            raise ValueError(f"Unsafe test shape: {inp}; runtime load succeeded")
        result = session.run([outputs[0].name], {
            inputs[0].name: np.zeros((batch, 5, height, width), dtype=np.float32)
        })
        if not result or result[0].shape != (batch, 4, height, width):
            raise ValueError(f"Unexpected inference output: {[x.shape for x in result]}")
    return inp, out


def repair_generated_model(path: Path, report: Report) -> bool:
    """Repair ONLY the repository's small generated demo model, with a backup.

    If a model uses newer IR features or is a trained model, it must be
    re-exported properly with a compatible runtime instead of relabeling it.
    """
    if not path.is_file():
        report.add("WARN", f"Model missing: {path}. Generate/restore a model before analysis")
        return True  # Setup can finish, but --check will fail.

    try:
        import onnx
        import onnxruntime  # noqa: F401 - needed for runtime check

        # First see whether it already works; don't edit successful models.
        try:
            probe_model_session(path)
            report.add("PASS", "ONNX model already loads in ONNX Runtime; no changes made")
            return True
        except Exception as error:
            original_error = str(error)

        if path.stat().st_size > 1_000_000:
            report.add("FAIL", "ONNX model cannot load and is not a tiny demo model. "
                       f"Re-export or use a compatible runtime. Reason: {original_error}")
            return False
        model = onnx.load_model(str(path), load_external_data=False)
        if not is_known_generated_demo(model):
            report.add("FAIL", "ONNX model is not the known generated single-Conv demo; "
                       f"refusing to change IR. Error: {original_error}")
            return False
        if model.ir_version <= 10:
            report.add("FAIL", f"Demo model IR={model.ir_version} already <=10, "
                       f"but runtime rejects it: {original_error}")
            return False

        # IR 10 supports the elementary Conv graph and opset 13 and is
        # supported by the installed runtime (including versions older than 1.31).
        old_ir = model.ir_version
        model.ir_version = 10
        temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.onnx")
        try:
            onnx.save_model(model, str(temporary))
            onnx.checker.check_model(str(temporary))
            probe_model_session(temporary, infer=True)  # Verify BEFORE overwrite.

            backup = path.with_name(
                f"{path.stem}.backup-{datetime.now():%Y%m%d-%H%M%S}-"
                f"{uuid.uuid4().hex[:8]}{path.suffix}"
            )
            shutil.copy2(path, backup)
            os.replace(temporary, path)
            report.add("PASS", f"Repaired generated demo ONNX IR {old_ir} -> 10")
            report.add("PASS", f"Previous ONNX file backed up to: {backup}")
            return True
        finally:
            if temporary.exists():
                temporary.unlink()
    except Exception as exc:
        report.add("FAIL", f"ONNX model repair failed, original preserved: {exc}")
        return False


def check_model(root: Path, report: Report, deep: bool) -> None:
    model = model_path(root)
    if not model.is_file():
        report.add("FAIL", f"ONNX model missing: {model}. "
                   "Restore a trained model or generate the bundled demo model")
        return
    try:
        import onnx

        # File-path based checker also works with external tensor data.
        onnx.checker.check_model(str(model))
        info = onnx.load_model(str(model), load_external_data=False)
        report.add("PASS", f"ONNX structurally valid; IR={info.ir_version}; "
                   f"opsets={[(i.domain or 'ai.onnx', i.version) for i in info.opset_import]}")
        if is_known_generated_demo(info):
            report.add("WARN", "Using bundled 1x1 Conv demo model, NOT a trained ResNet34")
    except Exception as exc:
        report.add("FAIL", f"ONNX structural validation failed: {exc}")
        return

    # Always load the model: --check must detect the IR mismatch before GIS execution.
    try:
        inp, out = probe_model_session(model, infer=deep)
        report.add("PASS", f"ONNX Runtime loaded model: input={inp}, output={out}")
        if deep:
            report.add("PASS", "ONNX CPU smoke inference succeeded")
    except Exception as exc:
        report.add("FAIL", f"ONNX Runtime cannot load/run model: {exc}. "
                   "Try environment.py --fix-model for the bundled demo only")


def check_backend_import(root: Path, report: Report) -> None:
    """Catch imports missing from requirements before starting Uvicorn."""
    proc = subprocess.run(
        [sys.executable, "-c", "import backend.app.main; print('Backend imported')"],
        cwd=root, capture_output=True, text=True, check=False,
    )
    if proc.returncode == 0:
        report.add("PASS", "FastAPI backend imports successfully")
    else:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        report.add("FAIL", "Backend import failed: " + " | ".join(detail[-5:])[:900])


def run_check(deep: bool) -> int:
    report = Report()
    check_python(report)
    check_layout(ROOT, report)
    check_requirements(ROOT, report)
    sample = ROOT / "backend" / "Engine_module" / "data" / "sample_scene"
    if sample.is_dir():
        first = inspect_raster_dir(sample / "t1", "sample/t1", report, required=True)
        second = inspect_raster_dir(sample / "t2", "sample/t2", report, required=True)
        if first is not None and second is not None and first != second:
            report.add("FAIL", "Sample T1 and T2 grids do not match")
    else:
        report.add("WARN", "Optional sample_scene dataset is absent")
    check_projects(ROOT, report)
    check_model(ROOT, report, deep)
    check_backend_import(ROOT, report)
    return 0 if report.summary() else 1


def setup() -> int:
    report = Report()
    if not python_supported(sys.version_info):
        report.add("FAIL", "Please run with Python 3.11-3.13")
        return 1
    check_layout(ROOT, report)
    if report.counts["FAIL"]:
        report.summary()
        return 1
    try:
        storage_dir(ROOT).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        report.add("FAIL", f"Unable to create backend/storage: {exc}")
        report.summary()
        return 1
    if not fix_project_paths(ROOT, report):
        report.summary()
        return 1
    if not REQUIREMENTS.is_file():
        report.add("FAIL", "requirements.txt missing")
        report.summary()
        return 1
    try:
        if not VENV.exists():
            print(f"Creating virtual environment: {VENV}")
            venv.EnvBuilder(with_pip=True).create(VENV)
        python = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.is_file():
            raise RuntimeError(f"Virtual environment interpreter not found: {python}")
        subprocess.run([str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS)], check=True)
        report.add("PASS", "Dependencies installed in local virtual environment")
        # Use the newly-installed virtual environment, not the global interpreter.
        repaired = subprocess.run(
            [str(python), str(ROOT / "environment.py"), "--fix-model"],
            cwd=ROOT, check=False,
        )
        if repaired.returncode:
            report.add("FAIL", "ONNX model incompatible. See --fix-model details above")
            report.summary()
            return 1
        report.add("PASS", "ONNX model compatibility checked / known demo repaired")
        report.summary()
        print(f"\nNext: {python} environment.py --check --deep-model")
        return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        report.add("FAIL", f"Environment setup failed: {exc}")
        report.summary()
        return 1


class SelfTests(unittest.TestCase):
    """No extra test framework; tests live inside this single script."""

    def test_python_version_bounds(self):
        self.assertTrue(python_supported((3, 11)))
        self.assertTrue(python_supported((3, 13)))
        self.assertFalse(python_supported((3, 10)))
        self.assertFalse(python_supported((3, 14)))

    def test_local_project_paths_are_portable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(
                project_dataset_dir(root, "prj_123", "t1"),
                root / "backend/storage/projects/prj_123/datasets/t1",
            )

    def test_windows_boundary_rebasing_on_any_os(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = relocate_project_path(
                r"C:\Users\HP\Desktop\repo\backend\storage\projects\prj_123\datasets\wetland.geojson",
                Path(tmp), "prj_123",
            )
            self.assertEqual(result, Path(tmp) / "backend/storage/projects/prj_123/datasets/wetland.geojson")

    def test_external_boundary_remains_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(relocate_project_path(r"D:\external\boundary.shp", Path(tmp), "prj_123"))

    def test_migration_repairs_paths_preserves_data_and_backs_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = projects_file(root)
            config.parent.mkdir(parents=True)
            before = {
                "prj_123": {
                    "id": "prj_123", "name": "Example", "datasets_ready": True,
                    "t1_dir": r"C:\Users\HP\Desktop\oldrepo\backend\storage\projects\prj_123\datasets\t1",
                    "t2_dir": r"C:\Users\HP\Desktop\oldrepo\backend\storage\projects\prj_123\datasets\t2",
                    "extra": {"preserve": "me"},
                }
            }
            config.write_text(json.dumps(before), encoding="utf-8")
            r = Report(quiet=True)
            self.assertTrue(fix_project_paths(root, r))
            after = json.loads(config.read_text(encoding="utf-8"))
            for date in ("t1", "t2"):
                path = project_dataset_dir(root, "prj_123", date)
                self.assertEqual(after["prj_123"][f"{date}_dir"], str(path))
                self.assertTrue(path.is_dir())
            self.assertTrue(after["prj_123"]["datasets_ready"])
            self.assertEqual(after["prj_123"]["extra"], {"preserve": "me"})
            backups = list(config.parent.glob("projects.json.backup-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(json.loads(backups[0].read_text(encoding="utf-8")), before)
            self.assertTrue(fix_project_paths(root, Report(quiet=True)))
            self.assertEqual(len(list(config.parent.glob("projects.json.backup-*"))), 1)

    def test_invalid_json_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = projects_file(root)
            config.parent.mkdir(parents=True)
            config.write_text("{invalid json", encoding="utf-8")
            report = Report(quiet=True)
            self.assertFalse(fix_project_paths(root, report))
            self.assertEqual(config.read_text(encoding="utf-8"), "{invalid json")
            self.assertEqual(list(config.parent.glob("projects.json.backup-*")), [])

    def test_unsafe_project_id_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = projects_file(root)
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps({"../../escape": {}}), encoding="utf-8")
            report = Report(quiet=True)
            self.assertFalse(fix_project_paths(root, report))
            self.assertFalse((root / "escape").exists())

    def test_saved_project_missing_data_is_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = projects_file(root)
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps({"prj_123": {"datasets_ready": True}}), encoding="utf-8")
            report = Report(quiet=True)
            check_projects(root, report)
            self.assertGreater(report.counts["WARN"], 0)
            self.assertEqual(report.counts["FAIL"], 0)

    def test_invalid_layout_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Report(quiet=True)
            check_layout(Path(tmp), report)
            self.assertGreater(report.counts["FAIL"], 0)


    def test_demo_model_ir_repair_and_idempotence(self):
        try:
            import numpy as np
            import onnx
            import onnxruntime
        except ImportError:
            self.skipTest("onnx, onnxruntime, numpy not installed")
        from onnx import TensorProto, helper
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "best_finetuned_model.onnx"
            x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 5, 8, 8])
            y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 8, 8])
            weights = helper.make_tensor("conv_weight", TensorProto.FLOAT,
                                         [4, 5, 1, 1], [0.1] * 20)
            bias = helper.make_tensor("conv_bias", TensorProto.FLOAT, [4], [0.0] * 4)
            node = helper.make_node("Conv", ["input", "conv_weight", "conv_bias"],
                                    ["output"], kernel_shape=[1, 1])
            graph = helper.make_graph([node], "WetlandHealthClassifier", [x], [y],
                                      initializer=[weights, bias])
            mdl = helper.make_model(graph, producer_name="gis_wetland_engine",
                                    opset_imports=[helper.make_operatorsetid("", 13)])
            mdl.ir_version = 14
            onnx.save_model(mdl, str(target))
            # A future ONNX Runtime may support IR14 natively; in that case
            # no repair is needed and the utility must not rewrite anything.
            try:
                probe_model_session(target)
                already_supported = True
            except Exception:
                already_supported = False
            report = Report(quiet=True)
            self.assertTrue(repair_generated_model(target, report))
            self.assertEqual(onnx.load_model(str(target)).ir_version,
                             14 if already_supported else 10)
            self.assertEqual(report.counts["FAIL"], 0)
            expected_backups = 0 if already_supported else 1
            self.assertEqual(len(list(Path(tmp).glob("*.backup-*.onnx"))), expected_backups)
            self.assertTrue(repair_generated_model(target, Report(quiet=True)))
            self.assertEqual(len(list(Path(tmp).glob("*.backup-*.onnx"))), expected_backups)

    def test_trained_or_unknown_model_is_not_rewritten(self):
        try:
            import onnx
            import onnxruntime
        except ImportError:
            self.skipTest("onnx and onnxruntime not installed")
        from onnx import TensorProto, helper
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "trained.onnx"
            x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 5, 8, 8])
            y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 5, 8, 8])
            graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])],
                                      "OtherGraph", [x], [y])
            mdl = helper.make_model(graph, producer_name="third_party",
                                    opset_imports=[helper.make_operatorsetid("", 13)])
            mdl.ir_version = 14
            onnx.save_model(mdl, str(target))
            before = target.read_bytes()
            report = Report(quiet=True)
            # Either refuse an incompatible unknown model, or leave it alone
            # if a future runtime happens to support its IR version.
            result = repair_generated_model(target, report)
            self.assertEqual(target.read_bytes(), before)
            self.assertEqual(list(Path(tmp).glob("*.backup-*.onnx")), [])
            if not result:
                self.assertGreater(report.counts["FAIL"], 0)


def self_test() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SelfTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Wetland GIS environment setup and checks")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--setup", action="store_true", help="Repair paths, create .venv, install requirements")
    actions.add_argument("--fix-paths", action="store_true", help="Backup and repair projects.json and upload directories")
    actions.add_argument("--fix-model", action="store_true", help="Backup and repair ONLY the bundled 1x1 Conv demo ONNX model")
    actions.add_argument("--check", action="store_true", help="Read-only environment validation")
    actions.add_argument("--self-test", action="store_true", help="Run embedded regression tests")
    parser.add_argument("--deep-model", action="store_true", help="Also run tiny ONNX CPU smoke inference during --check")
    args = parser.parse_args()
    if args.deep_model and not args.check:
        parser.error("--deep-model requires --check")
    if args.self_test:
        return self_test()
    if args.setup:
        return setup()
    if args.fix_paths:
        report = Report()
        success = fix_project_paths(ROOT, report)
        return 0 if success and report.summary() else 1
    if args.fix_model:
        report = Report()
        success = repair_generated_model(model_path(ROOT), report)
        return 0 if success and report.summary() else 1
    return run_check(args.deep_model)


if __name__ == "__main__":
    sys.exit(main())
