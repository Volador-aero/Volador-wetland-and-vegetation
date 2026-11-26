"""
src/gis_engine/core/pipeline.py
==================================
Canonical Production Orchestrator for the GIS Wetland Degradation &
Depletion Engine.

This module owns orchestration ONLY. Scientific calculations remain
in their respective modules. The pipeline calls them in sequence,
handles exceptions, manages the output directory structure, writes
provenance metadata, the output manifest, and the analysis report.

Architecture (Phase 13 locked):
    INPUT → INGESTION → ALIGNMENT → VEGETATION INFERENCE →
    TRANSITIONS → CVA → DEGRADATION/RECOVERY → WETLAND CONTEXT →
    HYDROLOGY → FUSION → METRICS → QA → REPORTING → OUTPUT PACKAGE
"""

import hashlib
import json
import logging
import os
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import rasterio

from .config import load_config, save_config_snapshot, config_hash
from .dispatcher import AnalysisDispatcher
from .constants import (
    CHANNEL_ORDER,
    NODATA_CATEGORICAL,
    NODATA_CONTINUOUS,
    PHYSICAL_DEPLETION_DEFAULT,
    SOFTWARE_VERSION,
)
from .exceptions import (
    AlignmentError,
    ConfigurationError,
    FusionError,
    GISProcessingError,
    HydrologyError,
    InferenceError,
    InputValidationError,
    MetricError,
    OutputError,
    ValidationError,
)
from .results import AnalysisResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_run_id(user_run_id: Optional[str] = None) -> str:
    """Generates a unique run ID in format YYYYMMDDTHHMMSSZ_<short-uuid>."""
    if user_run_id:
        return user_run_id
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short = uuid.uuid4().hex[:8]
    return f"{ts}_{short}"


def _setup_logging(run_id: str, log_dir: Path, log_level: str = "INFO") -> logging.FileHandler:
    """Configures structured logging for a production run."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "analysis.log"

    root_logger = logging.getLogger("gis_engine")
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove pre-existing handlers to avoid duplication across runs
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt=f"[%(levelname)s] [{run_id}] %(name)s — %(message)s",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    return fh


def _file_sha256(path: Path) -> str:
    """Hash a regular file with platform-compatible change detection."""
    import os
    import stat

    p = Path(path)

    def common_identity(info):

        return (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
        )

    def source_identity(info):

        return common_identity(info) + (
            info.st_ctime_ns,
        )

    def is_link(item):

        return (
            item.is_symlink()
            or getattr(
                item, "is_junction", lambda: False
            )()
        )

    try:

        if is_link(p):
            return ""

        before = p.stat()

        if not stat.S_ISREG(before.st_mode):
            return ""

        sha = hashlib.sha256()
        bytes_read = 0

        with p.open("rb") as handle:

            opened = os.fstat(handle.fileno())

            if not stat.S_ISREG(opened.st_mode):
                return ""

            if common_identity(before) != common_identity(opened):
                return ""

            while True:

                chunk = handle.read(1024 * 1024)

                if not chunk:
                    break

                sha.update(chunk)
                bytes_read += len(chunk)

            after = os.fstat(handle.fileno())

            if source_identity(opened) != source_identity(after):
                return ""

            if bytes_read != opened.st_size:
                return ""

            if is_link(p):
                return ""

            current = p.stat()

            if source_identity(before) != source_identity(current):
                return ""

            if common_identity(after) != common_identity(current):
                return ""

        return sha.hexdigest()

    except (OSError, ValueError, RuntimeError):
        return ""


def _file_fingerprint(path: Path) -> Dict[str, Any]:
    """Fingerprint files and directory contents deterministically."""
    import os
    import stat
    import json

    p = Path(path)

    result = {
        "path": str(p),
        "exists": False,
        "size_bytes": None,
        "sha256": None,
        "kind": "unknown",
        "file_count": 0,
        "fingerprint_version": 1,
        "status": "NOT_AVAILABLE",
    }

    def reject(message):

        result["status"] = "FAIL"
        result["sha256"] = None
        result["error"] = message

        return result

    def is_link(item):

        return (
            item.is_symlink()
            or getattr(
                item, "is_junction", lambda: False
            )()
        )

    try:

        if is_link(p):
            return reject(
                "Symbolic links and junctions are not supported."
            )

        if not p.exists():

            result["error"] = "Input path does not exist."

            return result

        result["exists"] = True

        if p.is_file():

            result["kind"] = "file"

            before = p.stat()
            digest = _file_sha256(p)
            after = p.stat()

            if not digest:
                return reject("File checksum calculation failed.")

            if (
                before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
                or before.st_ino != after.st_ino
            ):
                return reject("File changed during fingerprinting.")

            result.update({
                "size_bytes": after.st_size,
                "sha256": digest,
                "file_count": 1,
                "status": "PASS",
            })

            return result

        if not p.is_dir():
            return reject("Unsupported input path type.")

        result["kind"] = "directory"

        root = p.resolve(strict=True)

        def scan():

            inventory = {}

            def on_error(exc):
                raise exc

            for folder, directories, filenames in os.walk(
                root,
                followlinks=False,
                onerror=on_error,
            ):

                current = Path(folder)

                for name in directories + filenames:

                    item = current / name

                    if is_link(item):
                        raise ValueError(
                            "Directory contains a link or junction."
                        )

                    resolved = item.resolve(strict=True)

                    if not resolved.is_relative_to(root):
                        raise ValueError(
                            "Directory entry escapes input root."
                        )

                    info = item.stat()

                    if stat.S_ISDIR(info.st_mode):
                        kind = "directory"

                    elif stat.S_ISREG(info.st_mode):
                        kind = "file"

                    else:
                        raise ValueError(
                            "Unsupported directory entry type."
                        )

                    relative = item.relative_to(root).as_posix()

                    inventory[relative] = (
                        kind,
                        info.st_size,
                        info.st_mtime_ns,
                        info.st_ctime_ns,
                        info.st_dev,
                        info.st_ino,
                    )

            return inventory

        before = scan()

        entries = []
        total_size = 0
        file_count = 0

        for relative in sorted(before):

            kind, size, *_ = before[relative]

            if kind == "directory":

                entries.append({
                    "path": relative,
                    "kind": "directory",
                })

                continue

            digest = _file_sha256(root / relative)

            if not digest:
                return reject(
                    "Directory file checksum calculation failed."
                )

            entries.append({
                "path": relative,
                "kind": "file",
                "size_bytes": size,
                "sha256": digest,
            })

            total_size += size
            file_count += 1

        after = scan()

        if before != after:
            return reject(
                "Directory changed during fingerprinting."
            )

        if file_count == 0:

            result["size_bytes"] = 0
            result["error"] = "Input directory contains no files."

            return result

        serialized = json.dumps(
            entries,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

        result.update({
            "size_bytes": total_size,
            "file_count": file_count,
            "sha256": hashlib.sha256(
                serialized
            ).hexdigest(),
            "status": "PASS",
        })

        return result

    except (OSError, ValueError, RuntimeError):

        return reject(
            "Input fingerprint could not be safely calculated."
        )


def _write_json(path: Path, data: Any) -> None:
    """Writes JSON to path, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, default=str)


def _qa_error_report(exc: Exception) -> Dict[str, Any]:
    """Creates a structured QA report when QA execution fails."""
    return {
        "overall_status": "ERROR",
        "categories": {},
        "flags": [
            {
                "code": "QA-ERROR",
                "severity": "ERROR",
                "message": f"QA execution failed: {exc}",
                "phase": "Quality Assurance",
                "status": "ERROR",
                "recommended_action": (
                    "Review the pipeline logs, investigate the QA error, "
                    "and rerun the analysis."
                ),
            }
        ],
    }


def _output_entry(
    run_dir: Path,
    rel_path: str,
) -> Dict[str, Any]:
    """Create a manifest entry restricted to the run directory."""
    from pathlib import PureWindowsPath

    result = {
        "path": rel_path,
        "type": "File",
        "size_bytes": None,
        "sha256": None,
        "status": "FAIL",
    }

    def reject(message):

        result["error"] = message

        return result

    if not isinstance(rel_path, str) or not rel_path.strip():

        return reject("Invalid output path.")

    normalized = rel_path.replace(chr(92), "/")

    relative = Path(normalized)

    windows_path = PureWindowsPath(normalized)

    if (
        relative.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or normalized.startswith("/")
        or ".." in relative.parts
        or ":" in normalized
        or chr(0) in normalized
    ):

        return reject("Unsafe output path.")

    try:

        root = Path(run_dir).resolve(strict=True)

        if not root.is_dir():
            return reject("Invalid run directory.")

        current = root

        for part in relative.parts:

            current = current / part

            if (
                current.is_symlink()
                or getattr(
                    current, "is_junction", lambda: False
                )()
            ):

                return reject(
                    "Output path contains a link or junction."
                )

        full = current.resolve(strict=True)

        if not full.is_relative_to(root):

            return reject(
                "Output path escapes the run directory."
            )

        if not full.is_file():

            return reject(
                "Manifest output is not a regular file."
            )

        types = {
            ".tif": "GeoTIFF",
            ".tiff": "GeoTIFF",
            ".json": "JSON",
            ".csv": "CSV",
            ".md": "Markdown",
            ".pdf": "PDF",
            ".png": "PNG",
        }

        fingerprint = _file_fingerprint(full)

        if fingerprint["status"] != "PASS":

            return reject(
                "Output fingerprint verification failed."
            )

        result.update({
            "path": normalized,
            "type": types.get(
                full.suffix.lower(), "File"
            ),
            "size_bytes": fingerprint["size_bytes"],
            "sha256": fingerprint["sha256"],
            "status": "PASS",
        })

        return result

    except (OSError, ValueError, RuntimeError):

        return reject(
            "Output file could not be safely fingerprinted."
        )


def _capture_run_provenance(
    model_path,
    input_paths,
    config,
    config_snapshot_path,
):
    """Capture provenance without claiming reproducibility."""

    from .config import collect_environment_provenance

    if not isinstance(input_paths, dict):
        raise ValueError("Input paths must be a dictionary.")

    if set(input_paths) not in (
        {"input_scene"},
        {"input_t1", "input_t2"},
    ):
        raise ValueError("Invalid input provenance structure.")

    configuration_hash = config_hash(config)

    snapshot = None
    snapshot_error = None

    try:

        snapshot_path = Path(config_snapshot_path)

        if not snapshot_path.is_file():
            raise FileNotFoundError(
                "Configuration snapshot unavailable."
            )

        with snapshot_path.open(
            "r", encoding="utf-8"
        ) as stream:

            snapshot = json.load(stream)

        if config_hash(snapshot) != configuration_hash:

            snapshot_error = (
                "Configuration snapshot checksum mismatch."
            )

    except Exception as exc:

        snapshot_error = (
            "Configuration snapshot validation failed: "
            + type(exc).__name__
        )

    environment = collect_environment_provenance()

    provenance = {
        "model_path": str(model_path),
        "model_sha256": _file_sha256(Path(model_path)),
        "configuration_hash": configuration_hash,
        "config_snapshot": str(config_snapshot_path),
        "config_snapshot_error": snapshot_error,
        "environment_provenance": environment,
        "git_commit": environment.get("git_commit"),
        "captured_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "reproducibility_verified": False,
    }

    for name, input_path in input_paths.items():

        provenance[name] = _file_fingerprint(
            Path(input_path)
        )

    return provenance, snapshot


def _compare_run_provenance(initial, current):
    """Detect changes between two provenance observations."""

    checks = {}

    def record(name, status, message):

        checks[name] = {
            "status": status,
            "message": message,
        }

    def valid_hash(value):

        return (
            isinstance(value, str)
            and len(value) == 64
            and all(
                character in "0123456789abcdef"
                for character in value
            )
        )

    def finish():

        statuses = {
            item["status"]
            for item in checks.values()
        }

        if "FAIL" in statuses:
            status = "FAIL"

        elif "WARN" in statuses:
            status = "WARN"

        elif "NOT_AVAILABLE" in statuses:
            status = "NOT_AVAILABLE"

        else:
            status = "PASS"

        return {
            "status": status,
            "checks": checks,
            "observed_provenance_unchanged": (
                bool(checks)
                and all(
                    item["status"] == "PASS"
                    for item in checks.values()
                )
            ),
            "reproducibility_verified": False,
        }

    if (
        not isinstance(initial, dict)
        or not isinstance(current, dict)
    ):

        record(
            "structure",
            "FAIL",
            "Invalid provenance observations.",
        )

        return finish()

    for name in (
        "model_sha256",
        "configuration_hash",
    ):

        before = initial.get(name)
        after = current.get(name)

        matches = (
            valid_hash(before)
            and valid_hash(after)
            and before == after
        )

        record(
            name,
            "PASS" if matches else "FAIL",
            f"{name} unchanged."
            if matches else
            f"{name} missing, invalid or changed.",
        )

    snapshot_valid = (
        "config_snapshot_error" in initial
        and "config_snapshot_error" in current
        and initial["config_snapshot_error"] is None
        and current["config_snapshot_error"] is None
        and isinstance(initial.get("config_snapshot"), str)
        and bool(initial["config_snapshot"])
        and initial["config_snapshot"]
            == current.get("config_snapshot")
    )

    record(
        "configuration_snapshot",
        "PASS" if snapshot_valid else "FAIL",
        "Configuration snapshots reconciled."
        if snapshot_valid else
        "Configuration snapshot missing, invalid or changed.",
    )

    input_keys = (
        "input_scene",
        "input_t1",
        "input_t2",
    )

    before_keys = {
        name for name in input_keys
        if name in initial
    }

    after_keys = {
        name for name in input_keys
        if name in current
    }

    structure_valid = (
        before_keys == after_keys
        and before_keys in (
            {"input_scene"},
            {"input_t1", "input_t2"},
        )
    )

    record(
        "input_structure",
        "PASS" if structure_valid else "FAIL",
        "Input structure unchanged."
        if structure_valid else
        "Input structure missing or changed.",
    )

    for name in sorted(before_keys | after_keys):

        before = initial.get(name)
        after = current.get(name)

        def valid_fingerprint(value):

            return (
                isinstance(value, dict)
                and value.get("status") == "PASS"
                and value.get("exists") is True
                and valid_hash(value.get("sha256"))
                and value.get("kind")
                    in ("file", "directory")
                and type(value.get("file_count")) is int
                and value["file_count"] > 0
                and type(value.get("size_bytes")) is int
                and value["size_bytes"] >= 0
                and type(
                    value.get("fingerprint_version")
                ) is int
                and value["fingerprint_version"] == 1
                and (
                    value["kind"] != "file"
                    or value["file_count"] == 1
                )
            )

        matches = (
            valid_fingerprint(before)
            and valid_fingerprint(after)
            and all(
                before.get(key) == after.get(key)
                for key in (
                    "sha256",
                    "size_bytes",
                    "file_count",
                    "kind",
                    "fingerprint_version",
                )
            )
        )

        record(
            name,
            "PASS" if matches else "FAIL",
            f"{name} fingerprint unchanged."
            if matches else
            f"{name} missing, invalid or changed.",
        )

    before_env = initial.get(
        "environment_provenance"
    )

    after_env = current.get(
        "environment_provenance"
    )

    if (
        not isinstance(before_env, dict)
        or not isinstance(after_env, dict)
    ):

        record(
            "environment",
            "NOT_AVAILABLE",
            "Environment comparison unavailable.",
        )

    else:

        def valid_environment(value):

            payload = {
                key: item
                for key, item in value.items()
                if key not in (
                    "environment_hash",
                    "reproducibility_verified",
                )
            }

            try:

                return (
                    valid_hash(
                        value.get("environment_hash")
                    )
                    and config_hash(payload)
                        == value["environment_hash"]
                    and value.get(
                        "reproducibility_verified"
                    ) is False
                )

            except Exception:

                return False

        if (
            not valid_environment(before_env)
            or not valid_environment(after_env)
        ):

            record(
                "environment",
                "FAIL",
                "Environment checksum invalid.",
            )

        else:

            stable_fields = (
                "python_version",
                "python_implementation",
                "os",
                "architecture",
                "dependencies",
                "git_commit",
            )

            changed = [
                name for name in stable_fields
                if before_env.get(name)
                    != after_env.get(name)
            ]

            record(
                "environment",
                "WARN" if changed else "PASS",
                "Environment fields changed: "
                + ", ".join(changed)
                if changed else
                "Recorded environment fields unchanged.",
            )

        dirty_before = before_env.get(
            "git_working_tree_dirty"
        )

        dirty_after = after_env.get(
            "git_working_tree_dirty"
        )

        if dirty_before is True or dirty_after is True:

            record(
                "working_tree",
                "WARN",
                "Git working tree contains changes.",
            )

        elif dirty_before is False and dirty_after is False:

            record(
                "working_tree",
                "PASS",
                "Git working tree reported clean.",
            )

        else:

            record(
                "working_tree",
                "NOT_AVAILABLE",
                "Git working-tree state unavailable.",
            )

    return finish()


def _finalize_production_provenance_qa(
    qa_engine,
    previous_report,
    initial,
    model_path,
    input_paths,
    config,
    snapshot_path,
    warnings,
):
    """Finalize observed provenance before report generation."""

    try:

        current, snapshot = _capture_run_provenance(
            model_path,
            input_paths,
            config,
            snapshot_path,
        )

        comparison = _compare_run_provenance(
            initial,
            current,
        )

        if qa_engine is None:

            raise RuntimeError(
                "QA engine unavailable."
            )

        for name, finding in comparison["checks"].items():

            status = finding["status"]

            if status == "PASS":
                continue

            severity = {
                "FAIL": "HIGH",
                "WARN": "MEDIUM",
                "NOT_AVAILABLE": "INFO",
            }[status]

            qa_engine.add_flag(
                "QA017",
                severity,
                finding["message"],
                "Reproducibility QA",
                status,
                "Review recorded provenance before accepting results.",
            )

            qa_engine.flags[-1]["check"] = "reproducibility"

        qa_engine.record_check(
            "reproducibility",
            comparison["status"],
        )

        report = qa_engine.generate_qa_report()

        if (
            isinstance(previous_report, dict)
            and previous_report.get("overall_status") == "ERROR"
        ):

            report["overall_status"] = "ERROR"

            report["previous_qa_error"] = previous_report

        elif (
            isinstance(previous_report, dict)
            and previous_report.get("overall_status") == "FAIL"
        ):

            report["overall_status"] = "FAIL"

        report["reproducibility"] = {
            "status": comparison["status"],
            "checks": comparison["checks"],
            "observed_provenance_unchanged": comparison[
                "observed_provenance_unchanged"
            ],
            "reproducibility_verified": False,
            "manifest_integrity_status": "NOT_RUN",
        }

        if comparison["status"] != "PASS":

            warnings.append(
                "Reproducibility provenance status: "
                + comparison["status"]
            )

        if report.get("overall_status") != "PASS":

            message = (
                "QA validation is incomplete or contains findings."
            )

            if message not in warnings:
                warnings.append(message)

        return report, current

    except Exception as exc:

        message = (
            "Final provenance QA execution failed: "
            + type(exc).__name__
        )

        warnings.append(message)

        if qa_engine is not None:

            try:

                qa_engine.add_flag(
                    "QA017",
                    "HIGH",
                    message,
                    "Reproducibility QA",
                    "FAIL",
                    "Inspect provenance inputs and execution logs.",
                )

                qa_engine.record_check(
                    "reproducibility",
                    "FAIL",
                )

                report = qa_engine.generate_qa_report()

            except Exception:

                report = _qa_error_report(exc)

        else:

            report = _qa_error_report(exc)

        report["overall_status"] = "ERROR"

        report["previous_qa_report"] = previous_report

        report["reproducibility"] = {
            "status": "FAIL",
            "execution_failed": True,
            "message": message,
            "reproducibility_verified": False,
            "manifest_integrity_status": "NOT_RUN",
        }

        return report, None


def _verify_output_manifest(
    run_dir,
    manifest,
    expected_run_id,
    expected_paths=None,
):
    """Verify recorded output checksums against actual files."""

    import re

    from pathlib import PurePosixPath, PureWindowsPath

    checks = []

    def record(name, status, message):

        checks.append({
            "path": name,
            "status": status,
            "message": message,
        })

    def finish():

        passed = sum(
            item["status"] == "PASS"
            for item in checks
        )

        failed = sum(
            item["status"] == "FAIL"
            for item in checks
        )

        verified = bool(checks) and failed == 0

        return {
            "status": "PASS" if verified else "FAIL",
            "integrity_verified": verified,
            "checks": checks,
            "passed_checks": passed,
            "failed_checks": failed,
            "scope": "REGISTERED_IMMUTABLE_OUTPUTS",
            "authenticity_verified": False,
        }

    def canonical_path(value):

        if not isinstance(value, str):
            return None

        if not value.strip():
            return None

        normalized = value.replace(chr(92), "/")

        relative = PurePosixPath(normalized)

        windows = PureWindowsPath(normalized)

        if (
            relative.is_absolute()
            or windows.drive
            or windows.root
            or ":" in normalized
            or chr(0) in normalized
            or relative.as_posix() == "."
        ):
            return None

        parts = normalized.split("/")

        reserved = {
            "CON", "PRN", "AUX", "NUL",
            "CONIN$", "CONOUT$",
        }

        reserved.update(
            f"COM{number}"
            for number in range(1, 10)
        )

        reserved.update(
            f"LPT{number}"
            for number in range(1, 10)
        )

        for part in parts:

            if (
                not part
                or part in (".", "..")
                or part.endswith((" ", "."))
                or any(ord(char) < 32 for char in part)
                or any(
                    char in '<>"|?*'
                    for char in part
                )
                or part.split(".")[0].upper() in reserved
            ):
                return None

        return relative.as_posix().casefold()

    if not isinstance(manifest, dict):

        record(
            "manifest",
            "FAIL",
            "Manifest must be a dictionary.",
        )

        return finish()

    if (
        not isinstance(expected_run_id, str)
        or not expected_run_id.strip()
        or manifest.get("run_id") != expected_run_id
    ):

        record(
            "run_id",
            "FAIL",
            "Manifest run identity mismatch.",
        )

    else:

        record(
            "run_id",
            "PASS",
            "Manifest run identity reconciled.",
        )

    entries = manifest.get("outputs")
    total = manifest.get("total_outputs")

    if (
        not isinstance(entries, list)
        or not entries
        or type(total) is not int
        or total != len(entries)
    ):

        record(
            "manifest_count",
            "FAIL",
            "Manifest output count is invalid.",
        )

        return finish()

    record(
        "manifest_count",
        "PASS",
        "Manifest output count reconciled.",
    )

    try:

        root = Path(run_dir)

        if (
            not root.is_dir()
            or root.is_symlink()
            or (
                hasattr(root, "is_junction")
                and root.is_junction()
            )
        ):

            raise ValueError(
                "Invalid output directory."
            )

        root = root.resolve(strict=True)

    except (OSError, TypeError, ValueError):

        record(
            "run_directory",
            "FAIL",
            "Run output directory is unavailable or unsafe.",
        )

        return finish()

    seen = set()

    excluded = {
        "manifest.json",
        "manifest_integrity.json",
        "analysis.log",
        "logs/analysis.log",
    }

    for index, entry in enumerate(entries):

        if not isinstance(entry, dict):

            record(
                f"entry_{index}",
                "FAIL",
                "Manifest entry must be a dictionary.",
            )

            continue

        name = entry.get("path")

        canonical = canonical_path(name)

        if canonical is None:

            record(
                f"entry_{index}",
                "FAIL",
                "Invalid or unsafe output path.",
            )

            continue

        if (
            canonical in excluded
            or canonical.endswith("/analysis.log")
        ):

            record(
                name,
                "FAIL",
                "Mutable or self-referential output included.",
            )

            continue

        if canonical in seen:

            record(
                name,
                "FAIL",
                "Duplicate output path.",
            )

            continue

        seen.add(canonical)

        expected_hash = entry.get("sha256")
        expected_size = entry.get("size_bytes")

        if (
            entry.get("status") != "PASS"
            or not isinstance(expected_hash, str)
            or re.fullmatch(
                r"[0-9a-f]{64}",
                expected_hash,
            ) is None
            or type(expected_size) is not int
            or expected_size < 0
        ):

            record(
                name,
                "FAIL",
                "Invalid recorded output fingerprint.",
            )

            continue

        try:

            actual = _output_entry(root, name)

            if (
                not isinstance(actual, dict)
                or actual.get("status") != "PASS"
            ):

                record(
                    name,
                    "FAIL",
                    "Output missing, unreadable or unsafe.",
                )

                continue

            if actual.get("size_bytes") != expected_size:

                record(
                    name,
                    "FAIL",
                    "Output file size mismatch.",
                )

                continue

            if actual.get("sha256") != expected_hash:

                record(
                    name,
                    "FAIL",
                    "Output SHA-256 checksum mismatch.",
                )

                continue

            record(
                name,
                "PASS",
                "Output size and checksum verified.",
            )

        except Exception as exc:

            record(
                name,
                "FAIL",
                "Output verification failed: "
                + type(exc).__name__,
            )

    if expected_paths is not None:

        if not isinstance(
            expected_paths,
            (list, tuple, set),
        ):

            record(
                "output_inventory",
                "FAIL",
                "Invalid expected output inventory.",
            )

        else:

            expected = set()
            inventory_valid = True

            for name in expected_paths:

                canonical = canonical_path(name)

                if (
                    canonical is None
                    or canonical in expected
                ):

                    inventory_valid = False
                    break

                expected.add(canonical)

            if not inventory_valid:

                record(
                    "output_inventory",
                    "FAIL",
                    "Invalid or duplicated expected output.",
                )

            elif expected != seen:

                record(
                    "output_inventory",
                    "FAIL",
                    "Manifest and expected output inventory differ.",
                )

            else:

                record(
                    "output_inventory",
                    "PASS",
                    "Registered output inventory reconciled.",
                )

    return finish()


def _finalize_output_manifest(
    run_dir,
    run_id,
    outputs,
    config_snapshot_path,
    warnings,
    errors,
):
    """Persist and independently verify registered output files."""

    import json

    root = Path(run_dir)

    manifest_path = root / "manifest.json"

    integrity_path = root / "manifest_integrity.json"

    excluded = []

    expected_paths = []

    manifest = None

    try:

        snapshot = Path(config_snapshot_path)

        relative_snapshot = snapshot.resolve(
            strict=True
        ).relative_to(
            root.resolve(strict=True)
        ).as_posix()

        outputs[relative_snapshot] = relative_snapshot

        # Register source-observation validity when produced.

        validity_path = (
            root / "wetland" / "wetland_source_validity.tif"
        )

        if validity_path.is_file():

            relative = (
                "wetland/wetland_source_validity.tif"
            )

            outputs[relative] = relative

        # Exclude mutable logs and self-referential files.

        for relative in outputs.values():

            if not isinstance(relative, str):

                raise ValueError(
                    "Invalid registered output path."
                )

            normalized = relative.replace(
                chr(92), "/"
            ).casefold()

            if (
                normalized in {
                    "manifest.json",
                    "manifest_integrity.json",
                    "analysis.log",
                }
                or normalized.endswith("/analysis.log")
            ):

                excluded.append(relative)

                continue

            expected_paths.append(relative)

        manifest = {
            "run_id": run_id,
            "generated_utc": (
                datetime.now(timezone.utc).isoformat()
            ),
            "total_outputs": len(expected_paths),
            "outputs": [
                _output_entry(root, relative)
                for relative in expected_paths
            ],
            "integrity_scope": (
                "REGISTERED_IMMUTABLE_OUTPUTS"
            ),
            "excluded_outputs": excluded,
            "verification_report": (
                "manifest_integrity.json"
            ),
        }

        _write_json(
            manifest_path,
            manifest,
        )

        # Verify the persisted manifest, not only memory.

        with manifest_path.open(
            "r",
            encoding="utf-8",
        ) as stream:

            persisted = json.load(stream)

        if persisted != manifest:

            raise ValueError(
                "Persisted manifest content mismatch."
            )

        integrity = _verify_output_manifest(
            root,
            persisted,
            run_id,
            expected_paths,
        )

    except Exception as exc:

        integrity = {
            "status": "FAIL",
            "integrity_verified": False,
            "authenticity_verified": False,
            "scope": (
                "REGISTERED_IMMUTABLE_OUTPUTS"
            ),
            "checks": [{
                "path": "manifest",
                "status": "FAIL",
                "message": (
                    "Manifest finalization failed: "
                    + type(exc).__name__
                ),
            }],
            "passed_checks": 0,
            "failed_checks": 1,
        }

    integrity["run_id"] = run_id

    integrity["verified_utc"] = (
        datetime.now(timezone.utc).isoformat()
    )

    integrity["excluded_outputs"] = excluded

    integrity["manifest_sha256"] = (
        _file_sha256(manifest_path)
        if manifest_path.is_file()
        else None
    )

    try:

        _write_json(
            integrity_path,
            integrity,
        )

        with integrity_path.open(
            "r",
            encoding="utf-8",
        ) as stream:

            persisted_integrity = json.load(stream)

        if persisted_integrity != integrity:

            raise ValueError(
                "Integrity report persistence mismatch."
            )

        outputs["manifest_integrity.json"] = (
            "manifest_integrity.json"
        )

    except Exception as exc:

        integrity["status"] = "FAIL"

        integrity["integrity_verified"] = False

        integrity["persistence_failed"] = True

        errors.append(
            "Manifest integrity report persistence failed: "
            + type(exc).__name__
        )

    if manifest_path.is_file():

        outputs["manifest.json"] = "manifest.json"

    if integrity.get("status") != "PASS":

        message = "Output manifest integrity verification failed."

        if message not in errors:

            errors.append(message)

        if message not in warnings:

            warnings.append(message)

    return manifest, integrity


def _resolve_final_execution_status(
    qa_report,
    manifest_integrity,
    warnings,
    errors,
):
    """Resolve execution status without hiding critical failures."""

    def add_error(message):

        if message not in errors:
            errors.append(message)

    def add_warning(message):

        if message not in warnings:
            warnings.append(message)

    if not isinstance(qa_report, dict):

        add_error(
            "Quality assurance report is unavailable or invalid."
        )

    else:

        qa_status = qa_report.get("overall_status")

        if qa_status == "FAIL":

            add_error(
                "Quality assurance validation failed."
            )

        elif qa_status == "ERROR":

            add_error(
                "Quality assurance execution failed."
            )

        elif qa_status == "WARN":

            add_warning(
                "Quality assurance contains unresolved findings."
            )

        elif qa_status != "PASS":

            add_error(
                "Quality assurance status is missing or invalid."
            )

    if not isinstance(manifest_integrity, dict):

        add_error(
            "Output manifest integrity report is unavailable."
        )

    elif (
        manifest_integrity.get("status") != "PASS"
        or manifest_integrity.get("integrity_verified") is not True
    ):

        add_error(
            "Output manifest integrity verification failed."
        )

    if errors:

        return "FAILED"

    if warnings:

        return "WARNING"

    return "SUCCESS"


def _persist_terminal_run_status(
    run_dir,
    run_id,
    status,
    qa_report,
    manifest_integrity,
    warnings,
    errors,
):
    """Persist terminal execution status without changing hashed outputs."""

    import json
    import os
    import tempfile

    temporary_path = None

    try:

        if (
            not isinstance(run_id, str)
            or not run_id.strip()
        ):

            raise ValueError(
                "Invalid run identity."
            )

        if status not in (
            "SUCCESS",
            "WARNING",
            "FAILED",
        ):

            raise ValueError(
                "Invalid terminal execution status."
            )

        root = Path(run_dir)

        if (
            not root.is_dir()
            or root.is_symlink()
            or getattr(
                root, "is_junction", lambda: False
            )()
        ):

            raise ValueError(
                "Unsafe run directory."
            )

        root = root.resolve(strict=True)

        destination = root / "run_status.json"

        if (
            destination.is_symlink()
            or getattr(
                destination, "is_junction", lambda: False
            )()
        ):

            raise ValueError(
                "Unsafe terminal status destination."
            )

        resolved = _resolve_final_execution_status(
            qa_report,
            manifest_integrity,
            warnings,
            errors,
        )

        if status == "FAILED" or resolved == "FAILED":

            status = "FAILED"

        elif status == "WARNING" or resolved == "WARNING":

            status = "WARNING"

        else:

            status = "SUCCESS"

        completed = status in (
            "SUCCESS",
            "WARNING",
        )

        qa_status = (
            qa_report.get("overall_status")
            if isinstance(qa_report, dict)
            else "ERROR"
        )

        integrity_status = (
            manifest_integrity.get("status")
            if isinstance(manifest_integrity, dict)
            else "FAIL"
        )

        payload = {
            "schema_version": 1,
            "run_id": run_id,
            "status": (
                "COMPLETED"
                if completed
                else "FAILED"
            ),
            "execution_status": status,
            "terminal": True,
            "progress": 100 if completed else None,
            "current_step": (
                "COMPLETED"
                if completed
                else "FAILED"
            ),
            "completed_at": (
                datetime.now(timezone.utc).isoformat()
            ),
            "qa_status": qa_status,
            "manifest_integrity_status": integrity_status,
            "manifest_integrity_verified": (
                isinstance(manifest_integrity, dict)
                and manifest_integrity.get(
                    "integrity_verified"
                ) is True
                and integrity_status == "PASS"
            ),
            "manifest_sha256": (
                manifest_integrity.get("manifest_sha256")
                if isinstance(manifest_integrity, dict)
                else None
            ),
            "warnings": list(warnings),
            "errors": list(errors),
            "error": (
                "; ".join(str(item) for item in errors)
                or "Pipeline execution failed."
                if not completed
                else None
            ),
            "message": {
                "SUCCESS": (
                    "Analysis completed successfully."
                ),
                "WARNING": (
                    "Analysis completed with warnings."
                ),
                "FAILED": (
                    "Analysis failed validation or execution."
                ),
            }[status],
        }

        # Atomic replacement prevents partially written JSON.

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=root,
            prefix=".run_status_",
            suffix=".tmp",
            delete=False,
        ) as stream:

            temporary_path = Path(stream.name)

            json.dump(
                payload,
                stream,
                indent=4,
                ensure_ascii=False,
                allow_nan=False,
            )

            stream.flush()

            os.fsync(stream.fileno())

        os.replace(
            temporary_path,
            destination,
        )

        temporary_path = None

        with destination.open(
            "r",
            encoding="utf-8",
        ) as stream:

            persisted = json.load(stream)

        if persisted != payload:

            raise ValueError(
                "Terminal status persistence mismatch."
            )

        return status

    except Exception as exc:

        message = (
            "Terminal run status persistence failed: "
            + type(exc).__name__
        )

        if message not in errors:

            errors.append(message)

        return "FAILED"

    finally:

        if temporary_path is not None:

            try:

                temporary_path.unlink(
                    missing_ok=True
                )

            except OSError:

                pass








# ---------------------------------------------------------------------------
# ProductionPipeline
# ---------------------------------------------------------------------------


def _execute_production_model_qa(
    qa_engine,
    model_path,
    result,
    observation_label,
):
    """Validate model outputs without modifying inference arrays."""

    def invalid(message):

        qa_engine.add_flag(
            "QA008",
            "HIGH",
            f"{observation_label}: {message}",
            "Model QA",
            "FAIL",
            "Inspect model output dimensions and validity.",
        )

        qa_engine.record_check("model", "FAIL")

    logits = result.logits
    probabilities = result.probabilities
    mask = result.valid_mask

    if not isinstance(mask, np.ndarray):
        invalid("Model validity mask is not an array.")
        return

    if (
        mask.ndim != 2
        or mask.size == 0
        or mask.dtype.kind not in "biuf"
        or not np.all(np.isin(mask, [0, 1]))
    ):
        invalid("Model validity mask is not binary 2D data.")
        return

    if not isinstance(logits, np.ndarray):
        invalid("Model logits are not an array.")
        return

    if not isinstance(probabilities, np.ndarray):
        invalid("Model probabilities are not an array.")
        return

    if logits.ndim == 4:

        if logits.shape[0] != 1:
            invalid("Unexpected model batch dimension.")
            return

        logits = logits[0]

    expected_shape = (4,) + mask.shape

    if logits.shape != expected_shape:
        invalid("Unexpected model logits shape.")
        return

    if probabilities.shape != expected_shape:
        invalid("Unexpected model probability shape.")
        return

    if (
        logits.dtype.kind not in "iuf"
        or probabilities.dtype.kind not in "iuf"
    ):
        invalid("Model outputs must contain real numbers.")
        return

    valid = mask.astype(bool)

    if not np.any(valid):

        qa_engine.execute_check(
            "model",
            str(model_path),
            None,
            None,
        )

        qa_engine.record_check(
            "model",
            "NOT_AVAILABLE",
        )

        return

    # Preserve the four-class axis and select valid pixels.
    # Result shape: (4, 1, valid_pixel_count).

    selected_logits = logits[:, valid][:, None, :]

    selected_probabilities = (
        probabilities[:, valid][:, None, :]
    )

    start = len(qa_engine.flags)

    qa_engine.execute_check(
        "model",
        str(model_path),
        selected_logits,
        selected_probabilities,
    )

    for flag in qa_engine.flags[start:]:
        flag["observation"] = observation_label



def _execute_production_spatial_qa(
    qa_engine,
    reference_grid,
    raster_paths,
):
    """Validate actual raster geometry against the reference grid."""

    from types import SimpleNamespace

    metadata = {
        "reference_grid": reference_grid,
    }

    for name, raster_path in raster_paths.items():

        if raster_path is None:
            metadata[name] = None
            continue

        try:

            raster_path = Path(raster_path)

            with rasterio.open(raster_path) as src:

                if src.count != 1:
                    raise ValueError(
                        "Expected a single-band raster."
                    )

                metadata[name] = SimpleNamespace(
                    crs=src.crs,
                    transform=src.transform,
                    width=src.width,
                    height=src.height,
                    resolution=src.res,
                    bounds=tuple(src.bounds),
                )

        except Exception as exc:

            metadata[name] = None

            qa_engine.add_flag(
                "QA-005",
                "HIGH",
                f"Cannot inspect raster {name}: "
                f"{type(exc).__name__}",
                "Spatial Alignment",
                "FAIL",
                "Inspect missing or unreadable raster outputs.",
            )

            qa_engine.record_check(
                "spatial",
                "FAIL",
            )

    return qa_engine.execute_check(
        "spatial",
        metadata,
    )



def _execute_production_nodata_qa(
    qa_engine,
    specifications,
):
    """Validate output-specific NoData and validity contracts."""

    statuses = []

    def finding(name, message, status):

        qa_engine.add_flag(
            "QA-019",
            "HIGH" if status == "FAIL" else "INFO",
            f"{name}: {message}",
            "NoData QA",
            status,
            "Inspect output validity masks and NoData propagation.",
        )

        qa_engine.flags[-1]["check"] = "nodata"
        statuses.append(status)

    for name, specification in specifications.items():

        values, mask, sentinel, allowed, bounds = specification

        if values is None or mask is None:

            finding(
                name,
                "Output or validity mask unavailable.",
                "NOT_AVAILABLE",
            )
            continue

        if (
            not isinstance(values, np.ndarray)
            or values.ndim != 2
            or values.size == 0
            or values.dtype.kind not in "iuf"
        ):

            finding(
                name,
                "Invalid output array.",
                "FAIL",
            )
            continue

        if (
            not isinstance(mask, np.ndarray)
            or mask.ndim != 2
            or mask.shape != values.shape
            or mask.dtype.kind not in "biuf"
            or not np.all(np.isin(mask, [0, 1]))
        ):

            finding(
                name,
                "Invalid or mismatched validity mask.",
                "FAIL",
            )
            continue

        valid = mask.astype(bool)

        inside = values[valid]
        outside = values[~valid]

        failed = False

        if outside.size and not np.all(
            outside == sentinel
        ):

            finding(
                name,
                "Incorrect NoData values outside valid region.",
                "FAIL",
            )
            failed = True

        if inside.size:

            if not np.all(np.isfinite(inside)):

                finding(
                    name,
                    "Non-finite values inside valid region.",
                    "FAIL",
                )
                failed = True

            # Zero confidence can be a legitimate value.
            # Do not treat it as missing inside valid pixels.

            if sentinel != 0 and np.any(
                inside == sentinel
            ):

                finding(
                    name,
                    "NoData sentinel inside valid region.",
                    "FAIL",
                )
                failed = True

            if allowed is not None and not np.all(
                np.isin(inside, allowed)
            ):

                finding(
                    name,
                    "Unsupported categorical values.",
                    "FAIL",
                )
                failed = True

            if bounds is not None:

                lower, upper = bounds

                invalid_range = (
                    (lower is not None and np.any(inside < lower))
                    or
                    (upper is not None and np.any(inside > upper))
                )

                if invalid_range:

                    finding(
                        name,
                        "Values outside permitted numeric range.",
                        "FAIL",
                    )
                    failed = True

        if not failed:

            if inside.size:
                statuses.append("PASS")

            else:
                finding(
                    name,
                    "No valid observations available.",
                    "NOT_AVAILABLE",
                )

    if "FAIL" in statuses:
        status = "FAIL"

    elif not statuses:
        status = "NOT_AVAILABLE"

    elif all(item == "PASS" for item in statuses):
        status = "PASS"

    elif all(
        item == "NOT_AVAILABLE"
        for item in statuses
    ):
        status = "NOT_AVAILABLE"

    else:
        status = "WARN"

    qa_engine.record_check("nodata", status)

    return {"status": status}



def _inspect_single_date_source_validity(
    qa_engine,
    band_paths,
    reference_grid,
):
    """Inspect source observations independently of model validity."""

    bands = {}
    common_valid = None
    complete = True

    expected_shape = (
        reference_grid.height,
        reference_grid.width,
    )

    for name in ("green", "red", "rededge", "nir"):

        try:

            path = band_paths.get(name)

            if path is None:
                raise FileNotFoundError(name)

            with rasterio.open(path) as src:

                if (
                    src.count != 1
                    or (src.height, src.width) != expected_shape
                    or src.crs != reference_grid.crs
                    or not np.allclose(
                        tuple(src.transform),
                        tuple(reference_grid.transform),
                        rtol=0,
                        atol=min(reference_grid.resolution) * 1e-6,
                    )
                ):
                    raise ValueError(
                        "Source raster grid mismatch."
                    )

                values = src.read(1)

                if values.dtype.kind not in "iuf":
                    raise ValueError(
                        "Unsupported source raster data type."
                    )

                valid = (
                    (src.read_masks(1) > 0)
                    & np.isfinite(values)
                    & (values != NODATA_CONTINUOUS)
                )

                if src.nodata is not None:
                    valid &= values != src.nodata

                bands[name] = np.where(
                    valid,
                    values,
                    np.nan,
                )

                if common_valid is None:
                    common_valid = valid.copy()

                else:
                    common_valid &= valid

        except Exception as exc:

            complete = False
            bands[name] = None

            qa_engine.add_flag(
                "QA-001",
                "HIGH",
                f"Cannot validate source band {name}: "
                f"{type(exc).__name__}",
                "Input Ingestion",
                "FAIL",
                "Inspect required source bands and raster alignment.",
            )

            qa_engine.flags[-1]["check"] = "input"

            qa_engine.record_check(
                "input",
                "FAIL",
            )

    if not complete:
        common_valid = None

    return bands, common_valid


class ProductionPipeline:
    """
    Canonical production orchestrator for the GIS Wetland Degradation
    & Depletion Engine.

    Usage::

        pipeline = ProductionPipeline()
        result = pipeline.run_analysis(
            t1_scene=Path("data/T1"),
            t2_scene=Path("data/T2"),
            t1_date="2023-08-14",
            t2_date="2023-08-28",
            config_path=Path("configs/config.json"),
            model_path=Path("models/best_finetuned_model.onnx"),
            output_base_dir=Path("outputs/runs"),
        )
    """

    def run_analysis(
        self,
        t1_scene: Union[str, Path],
        t2_scene: Optional[Union[str, Path]],
        t1_date: Optional[str],
        t2_date: Optional[str],
        config_path: Union[str, Path],
        model_path: Union[str, Path],
        output_base_dir: Union[str, Path],
        wetland_boundary: Optional[Union[str, Path]] = None,
        run_id: Optional[str] = None,
        log_level: str = "INFO",
    ) -> AnalysisResult:
        """
        Executes the complete production analysis pipeline.

        Returns an AnalysisResult regardless of success/failure.
        Check result.status for "SUCCESS", "WARNING", or "FAILED".
        """
        # ------------------------------------------------------------------
        # 0. Run ID & directory setup
        # ------------------------------------------------------------------
        run_id = _generate_run_id(run_id)
        output_base_dir = Path(output_base_dir)
        run_dir = output_base_dir / run_id

        if run_dir.exists():
            raise OutputError(
                f"Run directory already exists: {run_dir}. "
                "Use a unique run_id or remove the existing directory."
            )

        run_dir.mkdir(parents=True, exist_ok=True)

        veg_dir     = run_dir / "vegetation"
        change_dir  = run_dir / "change"
        deg_dir     = run_dir / "degradation"
        wetland_dir = run_dir / "wetland"
        hydro_dir   = run_dir / "hydrology"
        fusion_dir  = run_dir / "fusion"
        metrics_dir = run_dir / "metrics"
        val_dir     = run_dir / "validation"
        reports_dir = run_dir / "reports"
        logs_dir    = run_dir / "logs"

        for d in [veg_dir, change_dir, deg_dir, wetland_dir, hydro_dir,
                  fusion_dir, metrics_dir, val_dir, reports_dir, logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        fh = None
        try:
            fh = _setup_logging(run_id, logs_dir, log_level)
            run_logger = logging.getLogger("gis_engine.pipeline")

            warnings: List[str] = []
            errors:   List[str] = []
            outputs:  Dict[str, str] = {}

            run_start = time.perf_counter()
            run_logger.info("Phase 13 — Production pipeline START. run_id=%s", run_id)

            # Keep refs for later stages (populated as pipeline proceeds)
            t1_health = t2_health = overlap_mask = None
            manual_exclusion_mask = None
            effective_overlap_mask = None
            ref_grid = aligned_pair = pair_result = None
            transition_map = delta_ndvi = delta_gndvi = delta_ndre = cva_magnitude = None
            deg_result = wetland_result = None
            hydro_metrics: Dict[str, Any] = {}
            fusion_metrics: Dict[str, Any] = {}
            metrics_result_dict: Dict[str, Any] = {}
            qa_dict: Dict[str, Any] = {
                "overall_status": "NOT_REPORTED",
                "categories": {},
                "flags": [],
            }

            # ==============================================================
            # STEP 1 — Load & validate configuration
            # ==============================================================
            run_logger.info("Step 1 — Loading configuration: %s", config_path)
            try:
                config = load_config(config_path)
                config["model"]["path"] = str(model_path)

                config["model"]["path"] = str(model_path)

                dispatcher = AnalysisDispatcher.from_config(config)

                run_logger.info(
                    "Analysis mode: %s",
                    dispatcher.analysis_mode,
                )
            except ConfigurationError:
                raise
            except Exception as exc:
                raise ConfigurationError(f"Unexpected config error: {exc}") from exc

            config_snapshot_path = save_config_snapshot(config, run_dir)
            cfg_hash = config_hash(config)

            # Capture provenance before scientific processing.

            provenance_input_paths = (
                {"input_scene": t1_scene}
                if dispatcher.is_vegetation
                else {
                    "input_t1": t1_scene,
                    "input_t2": t2_scene,
                }
            )

            initial_run_provenance, initial_config_snapshot = (
                _capture_run_provenance(
                    model_path,
                    provenance_input_paths,
                    config,
                    config_snapshot_path,
                )
            )

            run_logger.info("Step 1 — Configuration loaded. hash=%s…", cfg_hash[:16])

            # ==============================================================
            # STEP 2 — Validate inputs
            # ==============================================================
            run_logger.info("Step 2 — Validating inputs")

            t1_scene = Path(t1_scene)
            model_path = Path(model_path)

            if not t1_scene.is_dir():
                raise InputValidationError(
                    f"Primary scene directory not found: {t1_scene}"
                )

            if not model_path.is_file():
                raise InputValidationError(
                    f"ONNX model file not found: {model_path}"
                )

            # --------------------------------------------------------------
            # Vegetation mode:
            #   Single-date analysis. Only one scene is required.
            # --------------------------------------------------------------
            if dispatcher.is_vegetation:
                t2_scene = Path(t2_scene) if t2_scene else None

                t1_date = t1_date or "single-date"

                t2_date = t2_date or None
                interval_days = None

                run_logger.info(
                    "Step 2 — Vegetation mode: single scene=%s",
                    t1_scene,
                )

            # --------------------------------------------------------------
            # Wetland mode:
            #   Temporal analysis. T1 + T2 and both dates are required.
            # --------------------------------------------------------------
            else:
                if not t2_scene:
                    raise InputValidationError(
                        "Wetland mode requires a T2 scene directory."
                    )

                if not t1_date:
                    raise InputValidationError(
                        "Wetland mode requires a T1 date."
                    )

                if not t2_date:
                    raise InputValidationError(
                        "Wetland mode requires a T2 date."
                    )

                t2_scene = Path(t2_scene)

                if not t2_scene.is_dir():
                    raise InputValidationError(
                        f"T2 scene directory not found: {t2_scene}"
                    )

                from datetime import datetime as _dt

                try:
                    d1 = _dt.strptime(t1_date, "%Y-%m-%d")
                    d2 = _dt.strptime(t2_date, "%Y-%m-%d")
                except ValueError as exc:
                    raise InputValidationError(
                        f"Invalid date format: {exc}"
                    ) from exc

                if d1 > d2:
                    raise InputValidationError(
                        f"T1 date ({t1_date}) must not be later than "
                        f"T2 date ({t2_date})."
                    )

                if d1 == d2:
                    warnings.append(
                        f"T1 and T2 dates are identical ({t1_date}). "
                        "This is valid for same-acquisition baseline testing. "
                        "Temporal interval is 0 days; no genuine temporal "
                        "change can exist."
                    )

                interval_days = (d2 - d1).days

                run_logger.info(
                    "Step 2 — Wetland inputs valid. "
                    "T1=%s T2=%s interval=%d days",
                    t1_date,
                    t2_date,
                    interval_days,
                )

            # ==============================================================
            # VEGETATION MODE BRANCH — Single-Date Pipeline Execution
            # ==============================================================
            if dispatcher.is_vegetation:
                run_logger.info("Step 3 — Vegetation mode: single-date execution (bypassing temporal alignment)")
                from ..alignment.spatial import ReferenceGrid
                from ..ingestion.loader import RasterLoader
                band_mapping = config.get("ingestion", {}).get("band_mapping", {})
                loader = RasterLoader(band_mapping=band_mapping)
                t1_paths = {}
                for b_name in ["green", "red", "rededge", "nir"]:
                    try:
                        t1_paths[b_name] = loader._find_band_file(t1_scene, b_name)
                    except FileNotFoundError:
                        pass
                if not t1_paths:
                    raise FileNotFoundError(f"No valid spectral bands found in scene: {t1_scene}")
                first_band_path = next(iter(t1_paths.values()))
                with rasterio.open(first_band_path) as src:
                    ref_grid = ReferenceGrid(
                        crs=src.crs,
                        transform=src.transform,
                        width=src.width,
                        height=src.height,
                        resolution=src.res,
                        bounds=(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top),
                    )
                    h, w = src.height, src.width

                overlap_mask = np.ones((h, w), dtype=bool)
                effective_overlap_mask = overlap_mask
                overlap_pixel_count = int(h * w)
                inclusion_pixel_count = overlap_pixel_count
                excluded_pixel_count = 0
                effective_pixel_count = overlap_pixel_count
                total_overlap_area_ha = None
                inclusion_area_ha = None
                excluded_area_ha = None
                effective_analysis_area_ha = None
                physical_area_available = False
                area_basis = "PIXEL_COUNT_ONLY"
                if ref_grid.crs is not None:
                    try:
                        from ..metrics.area import calculate_pixel_areas
                        pixel_areas = calculate_pixel_areas(
                            ref_grid.transform,
                            (ref_grid.height, ref_grid.width),
                            ref_grid.crs,
                        )
                        effective_analysis_area_ha = float(np.sum(pixel_areas)) / 10000.0
                        total_overlap_area_ha = effective_analysis_area_ha
                        inclusion_area_ha = effective_analysis_area_ha
                        excluded_area_ha = 0.0
                        physical_area_available = True
                        area_basis = "RASTER_CRS_AND_TRANSFORM"
                    except Exception:
                        pass

                band_statistics = {"scene": {}}
                for b_name in ["green", "red", "rededge", "nir"]:
                    p = t1_paths.get(b_name)
                    if p and Path(p).is_file():
                        with rasterio.open(p) as src:
                            arr = src.read(1).astype(np.float32)
                            valid = ~np.isnan(arr) & ~np.isinf(arr) & (arr != (src.nodata or -9999.0))
                            v = arr[valid]
                            if v.size > 0:
                                band_statistics["scene"][b_name] = {
                                    "valid_count": int(v.size),
                                    "min": float(np.min(v)),
                                    "max": float(np.max(v)),
                                    "mean": float(np.mean(v)),
                                    "median": float(np.median(v)),
                                    "std": float(np.std(v)),
                                    "p95": float(np.percentile(v, 95.0)),
                                }

                # Step 4 — Run Single-Date Vegetation Analysis
                run_logger.info("Steps 4–7 — Vegetation analysis (single-date)")
                try:
                    from ..vegetation.pipeline import VegetationAnalysisPipeline
                    veg_pipeline = VegetationAnalysisPipeline()
                    veg_result = veg_pipeline.run(
                        scene_dir=t1_scene,
                        model_path=model_path,
                        veg_dir=veg_dir,
                        config=config,
                        run_logger=run_logger,
                    )
                except Exception as exc:
                    raise GISProcessingError(f"Vegetation analysis failed: {exc}") from exc

                outputs["vegetation/health.tif"] = "vegetation/health.tif"
                outputs["vegetation/confidence.tif"] = "vegetation/confidence.tif"
                outputs["vegetation/statistics.json"] = "vegetation/statistics.json"

                # Spectral indices on single scene
                eps = 1e-7
                index_stats = {}
                nir_p = t1_paths.get("nir")
                red_p = t1_paths.get("red")
                green_p = t1_paths.get("green")
                re_p = t1_paths.get("rededge")

                def _calc_s_idx(arr: np.ndarray, formula_str: str) -> Dict[str, Any]:
                    v = arr[~np.isnan(arr) & ~np.isinf(arr)]
                    return {
                        "formula": formula_str,
                        "mean": float(np.mean(v)) if v.size else 0.0,
                        "median": float(np.median(v)) if v.size else 0.0,
                        "std": float(np.std(v)) if v.size else 0.0,
                        "p95": float(np.percentile(v, 95.0)) if v.size else 0.0,
                    }

                if nir_p and red_p:
                    with rasterio.open(nir_p) as s_nir, rasterio.open(red_p) as s_red:
                        n_arr = s_nir.read(1).astype(np.float32)
                        r_arr = s_red.read(1).astype(np.float32)
                        ndvi = (n_arr - r_arr) / (n_arr + r_arr + eps)
                        index_stats["ndvi"] = _calc_s_idx(ndvi, "(NIR - Red) / (NIR + Red)")

                if nir_p and green_p:
                    with rasterio.open(nir_p) as s_nir, rasterio.open(green_p) as s_green:
                        n_arr = s_nir.read(1).astype(np.float32)
                        g_arr = s_green.read(1).astype(np.float32)
                        gndvi = (n_arr - g_arr) / (n_arr + g_arr + eps)
                        index_stats["gndvi"] = _calc_s_idx(gndvi, "(NIR - Green) / (NIR + Green)")

                if nir_p and re_p:
                    with rasterio.open(nir_p) as s_nir, rasterio.open(re_p) as s_re:
                        n_arr = s_nir.read(1).astype(np.float32)
                        re_arr = s_re.read(1).astype(np.float32)
                        ndre = (n_arr - re_arr) / (n_arr + re_arr + eps)
                        index_stats["ndre"] = _calc_s_idx(ndre, "(NIR - RedEdge) / (NIR + RedEdge)")

                # Single-date QA
                try:
                    from ..validation.quality import QualityAssuranceEngine

                    qa_engine = QualityAssuranceEngine(config)

                    qa_engine.configure_required_checks("single_date")

                    qa_engine.execute_check(
                        "uncertainty",
                        config,
                        reference_samples_exist=False,
                    )

                    _execute_production_model_qa(
                        qa_engine,
                        model_path,
                        veg_result,
                        "T1",
                    )

                    spatial_paths = {
                        f"scene/{name}": t1_paths.get(name)
                        for name in ("green", "red", "rededge", "nir")
                    }

                    spatial_paths.update({
                        "vegetation/health": veg_dir / "health.tif",
                        "vegetation/confidence": veg_dir / "confidence.tif",
                    })

                    _execute_production_spatial_qa(
                        qa_engine,
                        ref_grid,
                        spatial_paths,
                    )

                    _execute_production_nodata_qa(
                        qa_engine,
                        {
                            "T1 health": (
                                veg_result.health_map,
                                veg_result.valid_mask,
                                255,
                                (0, 1, 2, 3),
                                None,
                            ),
                            "T1 confidence": (
                                veg_result.confidence,
                                veg_result.valid_mask,
                                0.0,
                                None,
                                (0.0, 1.0),
                            ),
                        },
                    )

                    source_bands, source_valid = (
                        _inspect_single_date_source_validity(
                            qa_engine,
                            t1_paths,
                            ref_grid,
                        )
                    )

                    qa_engine.execute_check(
                        "input",
                        source_bands,
                        None,
                        {},
                        None,
                    )

                    if source_valid is not None:

                        scene_indices = {
                            "ndvi": ndvi if "ndvi" in index_stats else None,
                            "gndvi": gndvi if "gndvi" in index_stats else None,
                            "ndre": ndre if "ndre" in index_stats else None,
                        }

                        qa_engine.execute_check(
                            "vegetation_index",
                            scene_indices,
                            None,
                            source_valid,
                        )

                        model_valid = veg_result.valid_mask

                        if (
                            not isinstance(model_valid, np.ndarray)
                            or model_valid.shape != source_valid.shape
                            or model_valid.dtype.kind not in "biuf"
                            or not np.all(np.isin(model_valid, [0, 1]))
                        ):

                            qa_engine.add_flag(
                                "QA-019",
                                "HIGH",
                                "Model validity mask is invalid or misaligned.",
                                "NoData QA",
                                "FAIL",
                                "Reconcile model validity with source observations.",
                            )

                            qa_engine.flags[-1]["check"] = "nodata"

                            qa_engine.record_check(
                                "nodata",
                                "FAIL",
                            )

                            qa_engine.record_check(
                                "class_distribution",
                                "NOT_AVAILABLE",
                            )

                        else:

                            model_valid = model_valid.astype(bool)

                            unsupported = int(np.count_nonzero(
                                model_valid & ~source_valid
                            ))

                            omitted = int(np.count_nonzero(
                                source_valid & ~model_valid
                            ))

                            if unsupported:

                                qa_engine.add_flag(
                                    "QA-019",
                                    "HIGH",
                                    f"Model classified {unsupported} pixels "
                                    "without valid source observations.",
                                    "NoData QA",
                                    "FAIL",
                                    "Propagate source observation validity into inference.",
                                )

                                qa_engine.flags[-1]["check"] = "nodata"

                                qa_engine.record_check(
                                    "nodata",
                                    "FAIL",
                                )

                            if omitted:

                                qa_engine.add_flag(
                                    "QA-019",
                                    "MEDIUM",
                                    f"Model omitted {omitted} valid source pixels.",
                                    "NoData QA",
                                    "WARN",
                                    "Inspect preprocessing and inference coverage.",
                                )

                                qa_engine.flags[-1]["check"] = "nodata"

                                qa_engine.record_check(
                                    "nodata",
                                    "WARN",
                                )

                            qa_engine.execute_check(
                                "class_distribution",
                                veg_result.health_map,
                                None,
                                source_valid & model_valid,
                            )

                    else:

                        qa_engine.record_check(
                            "vegetation_index",
                            "NOT_AVAILABLE",
                        )

                        qa_engine.record_check(
                            "class_distribution",
                            "NOT_AVAILABLE",
                        )

                        qa_engine.record_check(
                            "nodata",
                            "NOT_AVAILABLE",
                        )

                    qa_dict = qa_engine.generate_qa_report()

                    if qa_dict.get("overall_status") != "PASS":
                        warnings.append(
                            "QA validation is incomplete or contains findings."
                        )

                except Exception as exc:

                    qa_dict = _qa_error_report(exc)

                    warnings.append(f"QA: {exc}")

                    run_logger.warning(
                        "Quality assurance issue: %s",
                        exc,
                    )

                # Finalize provenance before exporting QA and reports.

                qa_dict, final_run_provenance = (
                    _finalize_production_provenance_qa(
                        locals().get("qa_engine"),
                        qa_dict,
                        initial_run_provenance,
                        model_path,
                        provenance_input_paths,
                        config,
                        config_snapshot_path,
                        warnings,
                    )
                )

                _write_json(
                    val_dir / "qa_report.json",
                    qa_dict,
                )

                outputs["validation/qa_report.json"] = (
                    "validation/qa_report.json"
                )

                run_logger.info(
                    "Quality assurance status: %s",
                    qa_dict.get("overall_status", "UNKNOWN"),
                )

                total_runtime = time.perf_counter() - run_start

                veg_classes = veg_result.class_statistics.get("classes", {})
                healthy_pct = veg_classes.get("3", {}).get("percentage", 0.0)
                moderate_pct = veg_classes.get("2", {}).get("percentage", 0.0)
                severe_pct = veg_classes.get("1", {}).get("percentage", 0.0)
                background_pct = veg_classes.get("0", {}).get("percentage", 0.0)
                veg_coverage_pct = healthy_pct + moderate_pct + severe_pct

                mean_conf = float(np.nanmean(veg_result.confidence[veg_result.valid_mask])) if np.any(veg_result.valid_mask) else 0.0

                result_content = {
                    "run_id": run_id,
                    "software_version": SOFTWARE_VERSION,
                    "analysis_mode": "vegetation",
                    "status": "WARNING" if warnings else "SUCCESS",
                    "inputs": {
                        "scene_path": str(t1_scene),
                        "model_path": str(model_path),
                        "band_statistics": band_statistics,
                    },
                    "temporal": {
                        "date": t1_date,
                    },
                    "spectral_indices": index_stats,
                    "spatial": {
                        "crs": str(ref_grid.crs) if ref_grid and ref_grid.crs is not None else None,
                        "dimensions": [ref_grid.height, ref_grid.width],
                        "resolution": list(ref_grid.resolution),
                        "extent": list(ref_grid.bounds),
                        "total_area_ha": total_overlap_area_ha,
                        "effective_analysis_area_ha": effective_analysis_area_ha,
                        "physical_area_available": physical_area_available,
                        "area_basis": area_basis,
                        "total_pixels": int(ref_grid.width * ref_grid.height),
                        "valid_pixels": int(veg_result.class_statistics.get("total_valid_pixels", 0)),
                    },
                    "vegetation": veg_result.class_statistics,
                    "metrics": {
                        "class_distribution": veg_result.class_statistics,
                        "healthy_vegetation_percentage": healthy_pct,
                        "moderate_stress_percentage": moderate_pct,
                        "severe_stress_percentage": severe_pct,
                        "background_percentage": background_pct,
                        "vegetation_coverage_percentage": veg_coverage_pct,
                        "mean_confidence": mean_conf,
                        "analysis_extent": {
                            "valid_analysis_area_ha": effective_analysis_area_ha or 0.0,
                        },
                    },
                    "validation": qa_dict,
                    "uncertainty": {
                        "mean_confidence": mean_conf,
                    },
                    "warnings": warnings,
                    "errors": errors,
                    "outputs": outputs,
                    "performance": {
                        "total_runtime_seconds": total_runtime,
                        "inference": veg_result.performance,
                    },
                }

                _write_json(run_dir / "analysis_result.json", result_content)
                outputs["analysis_result.json"] = "analysis_result.json"

                # Step 14 — Reports (Vegetation Health Markdown, Maps & PDF)
                run_logger.info("Step 14 — Generating vegetation analysis reports")
                try:
                    from ..output.report import generate_analysis_report
                    generate_analysis_report(
                        result_dict=result_content,
                        output_path=reports_dir / "analysis_report.md",
                        run_id=run_id,
                    )
                    outputs["reports/analysis_report.md"] = "reports/analysis_report.md"
                except Exception as exc:
                    warnings.append(f"Markdown Report: {exc}")
                    run_logger.warning("Step 14 — Markdown report warning: %s", exc)

                map_assets = {}
                try:
                    from ..output.cartography import generate_report_maps
                    map_assets = generate_report_maps(
                        run_dir=run_dir,
                        figures_dir=reports_dir / "figures",
                    )
                    for fig_key, fig_path in map_assets.items():
                        rel_fig = str(fig_path.relative_to(run_dir)).replace("\\", "/")
                        outputs[f"reports/figures/{fig_path.name}"] = rel_fig
                except Exception as exc:
                    warnings.append(f"Map Rendering: {exc}")
                    run_logger.warning("Step 14 — Map rendering warning: %s", exc)

                try:
                    from ..output.pdf import generate_all_reports
                    generated_pdfs = generate_all_reports(
                        result_dict=result_content,
                        run_dir=run_dir,
                        reports_dir=reports_dir,
                        map_assets=map_assets,
                        run_id=run_id,
                    )
                    # Register all generated PDFs in the output manifest
                    for pdf_key, pdf_path in generated_pdfs.items():
                        rel = str(pdf_path.relative_to(run_dir)).replace("\\", "/")
                        outputs[rel] = rel
                except Exception as exc:
                    warnings.append(f"PDF Report: {exc}")
                    run_logger.warning("Step 14 — PDF report warning: %s", exc)

                # Provenance metadata.json & manifest
                model_sha256 = _file_sha256(model_path)
                try:
                    import subprocess
                    git_commit = subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                    ).decode().strip()
                except Exception:
                    git_commit = "unavailable"

                metadata = {
                    "run_id": run_id,
                    "software_version": SOFTWARE_VERSION,
                    "analysis_mode": "vegetation",
                    "git_commit": git_commit,
                    "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "python_version": sys.version,
                    "os": platform.platform(),
                    "model_path": str(model_path),
                    "model_sha256": model_sha256,
                    "configuration_hash": cfg_hash,
                    "config_snapshot": str(config_snapshot_path),
                    "input_scene": _file_fingerprint(t1_scene),
                    "date": t1_date,
                    "crs": str(ref_grid.crs) if ref_grid and ref_grid.crs is not None else None,
                    "resolution": list(ref_grid.resolution) if ref_grid else None,
                    "dimensions": [ref_grid.height, ref_grid.width] if ref_grid else None,
                    "total_runtime_seconds": time.perf_counter() - run_start,
                }
                # Preserve observed initial and final run provenance.

                metadata["initial_provenance"] = initial_run_provenance

                metadata["final_provenance"] = final_run_provenance

                metadata["reproducibility_verified"] = False

                metadata["provenance_comparison"] = qa_dict.get(
                    "reproducibility",
                    {},
                )

                # Retain legacy metadata fields using initial fingerprints.

                for provenance_key in (
                    "git_commit",
                    "model_sha256",
                    "configuration_hash",
                    "config_snapshot",
                    "input_scene",
                    "input_t1",
                    "input_t2",
                ):

                    if provenance_key in initial_run_provenance:

                        metadata[provenance_key] = (
                            initial_run_provenance[provenance_key]
                        )

                metadata["manifest_integrity_report"] = (
                    "manifest_integrity.json"
                )

                _write_json(run_dir / "metadata.json", metadata)
                outputs["metadata.json"] = "metadata.json"

                manifest, manifest_integrity = (
                    _finalize_output_manifest(
                        run_dir,
                        run_id,
                        outputs,
                        config_snapshot_path,
                        warnings,
                        errors,
                    )
                )

                final_status = _resolve_final_execution_status(
                    qa_dict,
                    manifest_integrity,
                    warnings,
                    errors,
                )

                # Persist terminal status outside the immutable manifest.

                final_status = _persist_terminal_run_status(
                    run_dir,
                    run_id,
                    final_status,
                    qa_dict,
                    manifest_integrity,
                    warnings,
                    errors,
                )

                if (run_dir / "run_status.json").is_file():

                    outputs["run_status.json"] = "run_status.json"
                run_logger.info(
                    "Phase 13 — Vegetation Pipeline COMPLETE. status=%s run_id=%s runtime=%.1fs",
                    final_status, run_id, time.perf_counter() - run_start,
                )

                return AnalysisResult(
                    run_id=run_id,
                    status=final_status,
                    output_directory=str(run_dir),
                    outputs=outputs,
                    metrics=result_content.get("metrics", {}),
                    validation=qa_dict,
                    warnings=warnings,
                    errors=errors,
                    metadata=metadata,
                )

            # ==============================================================
            # STEP 3 — Spatial / Temporal Alignment (Wetland Mode)
            # ==============================================================
            run_logger.info("Step 3 — Spatial/temporal alignment")
            try:
                from ..alignment.temporal import TemporalAligner
                aligner = TemporalAligner(config)
                aligned_pair = aligner.align_pair(t1_scene, t1_date, t2_scene, t2_date)
            except Exception as exc:
                raise AlignmentError(f"Alignment failed: {exc}") from exc

            ref_grid     = aligned_pair.reference_grid
            overlap_mask = aligned_pair.overlap_mask          # bool ndarray (H, W)
            run_logger.info(
                "Step 3 — Alignment complete. overlap=%.2f%%",
                aligned_pair.overlap_percentage,
            )

            # Create alignment output directories
            alignment_dir = run_dir / "alignment"
            t1_aligned_out = alignment_dir / "t1_aligned"
            t2_aligned_out = alignment_dir / "t2_aligned"
            t1_aligned_out.mkdir(parents=True, exist_ok=True)
            t2_aligned_out.mkdir(parents=True, exist_ok=True)

            # Write aligned bands to disk as green.tif, red.tif, rededge.tif, nir.tif
            align_profile = {
                "driver": "GTiff",
                "dtype": "float32",
                "nodata": NODATA_CONTINUOUS,
                "width": ref_grid.width,
                "height": ref_grid.height,
                "count": 1,
                "crs": ref_grid.crs,
                "transform": ref_grid.transform,
                "compress": "lzw",
            }

            for band_name, band_arr in aligned_pair.t1_data.items():
                with rasterio.open(t1_aligned_out / f"{band_name}.tif", "w", **align_profile) as dst:
                    dst.write(band_arr.astype(np.float32), 1)

            for band_name, band_arr in aligned_pair.t2_data.items():
                with rasterio.open(t2_aligned_out / f"{band_name}.tif", "w", **align_profile) as dst:
                    dst.write(band_arr.astype(np.float32), 1)

            # Write overlap mask to alignment folder
            mask_profile = {
                "driver": "GTiff",
                "dtype": "uint8",
                "nodata": 0,
                "width": ref_grid.width,
                "height": ref_grid.height,
                "count": 1,
                "crs": ref_grid.crs,
                "transform": ref_grid.transform,
                "compress": "lzw",
            }
            with rasterio.open(alignment_dir / "overlap_mask.tif", "w", **mask_profile) as dst:
                dst.write(overlap_mask.astype(np.uint8), 1)

            # Record outputs
            outputs["alignment/t1_aligned/green.tif"] = "alignment/t1_aligned/green.tif"
            outputs["alignment/t1_aligned/red.tif"] = "alignment/t1_aligned/red.tif"
            outputs["alignment/t1_aligned/rededge.tif"] = "alignment/t1_aligned/rededge.tif"
            outputs["alignment/t1_aligned/nir.tif"] = "alignment/t1_aligned/nir.tif"
            outputs["alignment/t2_aligned/green.tif"] = "alignment/t2_aligned/green.tif"
            outputs["alignment/t2_aligned/red.tif"] = "alignment/t2_aligned/red.tif"
            outputs["alignment/t2_aligned/rededge.tif"] = "alignment/t2_aligned/rededge.tif"
            outputs["alignment/t2_aligned/nir.tif"] = "alignment/t2_aligned/nir.tif"
            outputs["alignment/overlap_mask.tif"] = "alignment/overlap_mask.tif"

            # ==============================================================
            # STEP 3A — Preparation Masking (Inclusion AOI & Manual Exclusions)
            # ==============================================================
            run_logger.info("Step 3A — Building authoritative preparation mask")

            try:
                from ..wetland.exclusion import build_manual_exclusion_mask, build_inclusion_mask

                inclusion_geometry = config.get("inclusion_geometry")
                manual_exclusions = config.get("manual_exclusions", [])

                exclusion_dir = run_dir / "exclusions"
                exclusion_dir.mkdir(parents=True, exist_ok=True)

                inclusion_mask_path = (
                    exclusion_dir / "inclusion_mask.tif"
                )
                manual_exclusion_mask_path = (
                    exclusion_dir / "manual_exclusion_mask.tif"
                )

                run_logger.info(
                    "[EXCLUSION TRACE] reference grid: "
                    f"crs={ref_grid.crs}, "
                    f"transform={ref_grid.transform}, "
                    f"width={ref_grid.width}, "
                    f"height={ref_grid.height}, "
                    f"bounds={ref_grid.bounds}"
                )

                run_logger.info(
                    "[EXCLUSION TRACE] inclusion geometry: "
                    f"{'Provided' if inclusion_geometry else 'Full Grid'}"
                )
                run_logger.info(
                    "[EXCLUSION TRACE] manual exclusions: "
                    f"{len(manual_exclusions)} exclusions"
                )

                # 1. Build inclusion mask (1 = inside AOI, 0 = outside)
                inclusion_mask = build_inclusion_mask(
                    geometry=inclusion_geometry,
                    reference_grid=ref_grid,
                    output_path=inclusion_mask_path,
                )

                # 2. Build manual exclusion mask (1 = excluded, 0 = not excluded)
                manual_exclusion_mask = build_manual_exclusion_mask(
                    exclusions=manual_exclusions,
                    reference_grid=ref_grid,
                    output_path=manual_exclusion_mask_path,
                )

                # 3. Form authoritative effective analysis mask:
                # effective = overlap & inclusion & ~exclusion
                effective_overlap_mask = (
                    overlap_mask.astype(bool)
                    & inclusion_mask.astype(bool)
                    & ~manual_exclusion_mask.astype(bool)
                )

                overlap_pixel_count = int(
                    np.sum(overlap_mask.astype(bool))
                )

                inclusion_pixel_count = int(
                    np.sum(
                        overlap_mask.astype(bool)
                        & inclusion_mask.astype(bool)
                    )
                )

                excluded_pixel_count = int(
                    np.sum(
                        overlap_mask.astype(bool)
                        & inclusion_mask.astype(bool)
                        & manual_exclusion_mask.astype(bool)
                    )
                )

                effective_pixel_count = int(
                    np.sum(effective_overlap_mask)
                )

                outputs[
                    "exclusions/inclusion_mask.tif"
                ] = "exclusions/inclusion_mask.tif"
                outputs[
                    "exclusions/manual_exclusion_mask.tif"
                ] = "exclusions/manual_exclusion_mask.tif"

                # Physical hectare areas are only authoritative when the
                # analysis grid is georeferenced. For the current pixel-space
                # DJI test imagery (CRS=None + identity transform), pixel counts
                # remain authoritative and physical area is explicitly unavailable.
                total_overlap_area_ha = None
                inclusion_area_ha = None
                excluded_area_ha = None
                effective_analysis_area_ha = None
                physical_area_available = False
                area_basis = "PIXEL_COUNT_ONLY"

                try:
                    if ref_grid.crs is not None:
                        from ..metrics.area import calculate_pixel_areas

                        pixel_areas = calculate_pixel_areas(
                            ref_grid.transform,
                            (ref_grid.height, ref_grid.width),
                            ref_grid.crs,
                        )

                        total_overlap_area_ha = float(
                            np.sum(
                                pixel_areas[
                                    overlap_mask.astype(bool)
                                ]
                            )
                        ) / 10000.0

                        inclusion_area_ha = float(
                            np.sum(
                                pixel_areas[
                                    overlap_mask.astype(bool)
                                    & inclusion_mask.astype(bool)
                                ]
                            )
                        ) / 10000.0

                        excluded_area_ha = float(
                            np.sum(
                                pixel_areas[
                                    overlap_mask.astype(bool)
                                    & inclusion_mask.astype(bool)
                                    & manual_exclusion_mask.astype(bool)
                                ]
                            )
                        ) / 10000.0

                        effective_analysis_area_ha = float(
                            np.sum(
                                pixel_areas[
                                    effective_overlap_mask
                                ]
                            )
                        ) / 10000.0

                        physical_area_available = True
                        area_basis = "RASTER_CRS_AND_TRANSFORM"
                    else:
                        run_logger.warning(
                            "Step 3A — Physical area unavailable: analysis grid "
                            "has no CRS. Reporting pixel counts only."
                        )
                except Exception as exc:
                    run_logger.warning(
                        "Failed to calculate physical exclusion areas: %s",
                        exc,
                    )

                if physical_area_available:
                    run_logger.info(
                        "Step 3A — Analysis footprint: total_overlap=%.4f ha, "
                        "inclusion=%.4f ha, excluded=%.4f ha, effective=%.4f ha",
                        total_overlap_area_ha,
                        inclusion_area_ha,
                        excluded_area_ha,
                        effective_analysis_area_ha,
                    )
                else:
                    run_logger.info(
                        "Step 3A — Analysis footprint: total_overlap_pixels=%d, "
                        "inclusion_pixels=%d, excluded_pixels=%d, effective_pixels=%d; "
                        "physical area unavailable.",
                        overlap_pixel_count,
                        inclusion_pixel_count,
                        excluded_pixel_count,
                        effective_pixel_count,
                    )
                # Preserve the original alignment footprint separately from the
                # effective analysis footprint after manual exclusions.
                effective_mask_path = alignment_dir / "effective_analysis_mask.tif"

                effective_mask_profile = {
                    "driver": "GTiff",
                    "height": int(ref_grid.height),
                    "width": int(ref_grid.width),
                    "count": 1,
                    "dtype": "uint8",
                    "crs": ref_grid.crs,
                    "transform": ref_grid.transform,
                    "nodata": 0,
                    "compress": "lzw",
                }

                with rasterio.open(
                    effective_mask_path,
                    "w",
                    **effective_mask_profile,
                ) as dst:
                    dst.write(
                        effective_overlap_mask.astype(np.uint8),
                        1,
                    )

                outputs[
                    "alignment/effective_analysis_mask.tif"
                ] = "alignment/effective_analysis_mask.tif"
                run_logger.info(
                    "Step 3A — Preparation complete: inclusion=%s, %d exclusions, "
                    "excluded_pixels=%d, effective_pixels=%d",
                    "custom" if inclusion_geometry else "full",
                    len(manual_exclusions),
                    excluded_pixel_count,
                    effective_pixel_count,
                )

            except Exception as exc:
                raise GISProcessingError(
                    f"Manual exclusion / preparation processing failed: {exc}"
                ) from exc

            # Authoritative input band radiometric statistics
            band_statistics: Dict[str, Any] = {"t1": {}, "t2": {}}
            for b_name in ["green", "red", "rededge", "nir"]:
                arr1 = aligned_pair.t1_data.get(b_name)
                if arr1 is not None:
                    v1 = arr1[effective_overlap_mask & ~np.isnan(arr1) & ~np.isinf(arr1)]
                    if v1.size > 0:
                        band_statistics["t1"][b_name] = {
                            "valid_count": int(v1.size),
                            "min": float(np.min(v1)),
                            "max": float(np.max(v1)),
                            "mean": float(np.mean(v1)),
                            "median": float(np.median(v1)),
                            "std": float(np.std(v1)),
                            "p95": float(np.percentile(v1, 95.0)),
                        }
                arr2 = aligned_pair.t2_data.get(b_name)
                if arr2 is not None:
                    v2 = arr2[effective_overlap_mask & ~np.isnan(arr2) & ~np.isinf(arr2)]
                    if v2.size > 0:
                        band_statistics["t2"][b_name] = {
                            "valid_count": int(v2.size),
                            "min": float(np.min(v2)),
                            "max": float(np.max(v2)),
                            "mean": float(np.mean(v2)),
                            "median": float(np.median(v2)),
                            "std": float(np.std(v2)),
                            "p95": float(np.percentile(v2, 95.0)),
                        }
            # ==============================================================
            # STEP 4–7 — Reusable Vegetation Analysis Pipeline (Two-Date Wetland)
            # ==============================================================
            run_logger.info("Steps 4–7 — Vegetation analysis on T1 & T2")

            try:
                from ..vegetation.pipeline import VegetationAnalysisPipeline
                from ..core.results import TwoDateInferenceResult
                from ..core.pair import calculate_class_stats
                from ..change.transition import calculate_transitions
                from ..change.cva import calculate_cva
                from ..change.degradation import analyze_degradation

                vegetation_pipeline = VegetationAnalysisPipeline()

                veg_t1_dir = veg_dir / "T1"
                veg_t2_dir = veg_dir / "T2"

                run_logger.info("Steps 4–7 — Vegetation inference T1: %s", t1_aligned_out)
                t1_veg = vegetation_pipeline.run(
                    scene_dir=t1_aligned_out,
                    model_path=model_path,
                    veg_dir=veg_t1_dir,
                    config=config,
                    run_logger=run_logger,
                )

                run_logger.info("Steps 4–7 — Vegetation inference T2: %s", t2_aligned_out)
                t2_veg = vegetation_pipeline.run(
                    scene_dir=t2_aligned_out,
                    model_path=model_path,
                    veg_dir=veg_t2_dir,
                    config=config,
                    run_logger=run_logger,
                )

                t1_health = t1_veg.health_map
                t2_health = t2_veg.health_map

                veg_profile = {
                    "driver": "GTiff",
                    "dtype": "uint8",
                    "nodata": NODATA_CATEGORICAL,
                    "width": ref_grid.width,
                    "height": ref_grid.height,
                    "count": 1,
                    "crs": ref_grid.crs,
                    "transform": ref_grid.transform,
                    "compress": "lzw",
                }
                with rasterio.open(veg_dir / "t1_health.tif", "w", **veg_profile) as dst:
                    dst.write(t1_health, 1)
                with rasterio.open(veg_dir / "t2_health.tif", "w", **veg_profile) as dst:
                    dst.write(t2_health, 1)

                mask_profile = {**veg_profile, "nodata": 0}
                with rasterio.open(veg_dir / "overlap_mask.tif", "w", **mask_profile) as dst:
                    dst.write(overlap_mask.astype(np.uint8), 1)

                outputs["vegetation/t1_health.tif"] = "vegetation/t1_health.tif"
                outputs["vegetation/t2_health.tif"] = "vegetation/t2_health.tif"
                outputs["vegetation/T1/confidence.tif"] = "vegetation/T1/confidence.tif"
                outputs["vegetation/T2/confidence.tif"] = "vegetation/T2/confidence.tif"
                outputs["vegetation/overlap_mask.tif"] = "vegetation/overlap_mask.tif"

                class_statistics_t1 = calculate_class_stats(t1_health, effective_overlap_mask)
                class_statistics_t2 = calculate_class_stats(t2_health, effective_overlap_mask)

                pair_result = TwoDateInferenceResult(
                    t1_health_map=t1_health,
                    t2_health_map=t2_health,
                    t1_logits=t1_veg.logits,
                    t2_logits=t2_veg.logits,
                    t1_probabilities=t1_veg.probabilities,
                    t2_probabilities=t2_veg.probabilities,
                    t1_confidence=t1_veg.confidence,
                    t2_confidence=t2_veg.confidence,
                    t1_valid_mask=t1_veg.valid_mask,
                    t2_valid_mask=t2_veg.valid_mask,
                    overlap_mask=overlap_mask,
                    reference_grid=ref_grid,
                    t1_metadata=t1_veg.metadata.__dict__ if hasattr(t1_veg.metadata, "__dict__") else {},
                    t2_metadata=t2_veg.metadata.__dict__ if hasattr(t2_veg.metadata, "__dict__") else {},
                    class_statistics_t1=class_statistics_t1,
                    class_statistics_t2=class_statistics_t2,
                    performance={
                        "t1_inference": t1_veg.performance,
                        "t2_inference": t2_veg.performance,
                    },
                )
                perf = pair_result.performance

                from ..output.json import write_pair_metadata_json
                pair_meta_dict = {
                    "pair_name": f"{t1_date}_{t2_date}",
                    "t1_date": t1_date,
                    "t2_date": t2_date,
                    "class_statistics_t1": class_statistics_t1,
                    "class_statistics_t2": class_statistics_t2,
                    "performance": perf,
                }
                write_pair_metadata_json(str(veg_dir / "pair_metadata.json"), pair_meta_dict)
                outputs["vegetation/pair_metadata.json"] = "vegetation/pair_metadata.json"

                # Step 5: Transition Analysis
                run_logger.info("Vegetation analysis — transition analysis")
                trans_result = calculate_transitions(
                    t1_health_map=t1_health,
                    t2_health_map=t2_health,
                    valid_overlap_mask=effective_overlap_mask.astype(np.uint8),
                    reference_grid=ref_grid,
                    output_dir=change_dir,
                    pair_name=f"{t1_date}_{t2_date}",
                )
                transition_map = trans_result.transition_map
                outputs["change/transition_map.tif"] = "change/transition_map.tif"

                # Step 6: CVA
                run_logger.info("Vegetation analysis — CVA")
                cva_result = calculate_cva(
                    t1_indices=t1_aligned_out,
                    t2_indices=t2_aligned_out,
                    valid_overlap_mask=effective_overlap_mask,
                    reference_grid=ref_grid,
                    output_dir=change_dir,
                    pair_name=f"{t1_date}_{t2_date}",
                    transition_map=transition_map,
                    config=config,
                )
                delta_ndvi = cva_result.delta_ndvi
                delta_gndvi = cva_result.delta_gndvi
                delta_ndre = cva_result.delta_ndre
                cva_magnitude = cva_result.cva_magnitude

                for fname in ["delta_ndvi.tif", "delta_gndvi.tif", "delta_ndre.tif", "cva_magnitude.tif"]:
                    outputs[f"change/{fname}"] = f"change/{fname}"

                # Step 7: Degradation
                run_logger.info("Vegetation analysis — degradation analysis")
                deg_result = analyze_degradation(
                    transition_map=transition_map,
                    cva_magnitude=cva_magnitude,
                    delta_ndvi=delta_ndvi,
                    delta_gndvi=delta_gndvi,
                    delta_ndre=delta_ndre,
                    valid_mask=effective_overlap_mask,
                    config=config,
                    reference_grid=ref_grid,
                    output_dir=deg_dir,
                )
                for fname in ["degradation_map.tif", "recovery_map.tif", "residual_stress.tif", "canopy_loss_map.tif"]:
                    outputs[f"degradation/{fname}"] = f"degradation/{fname}"

            except Exception as exc:
                raise GISProcessingError(
                    f"Vegetation analysis failed: {exc}"
                ) from exc

            run_logger.info(
                "Steps 4–7 — Vegetation analysis complete"
            )
            # ==============================================================
            if dispatcher.requires_wetland_context:
                # STEP 8 — Wetland Context
                # ==============================================================
                run_logger.info("Step 8 — Wetland context")
                try:
                    from ..wetland.mask import build_wetland_mask
                    wetland_source = (
                        str(wetland_boundary)
                        if wetland_boundary and Path(wetland_boundary).is_file()
                        else None
                    )
                    wetland_result = build_wetland_mask(
                        source=wetland_source,
                        reference_grid=ref_grid,
                        config=config,
                        output_dir=wetland_dir,
                        # Pass dynamic paths — never hardcoded date strings
                        t1_aligned_dir=str(t1_aligned_out),
                        t2_aligned_dir=str(t2_aligned_out),
                        overlap_mask=effective_overlap_mask.astype(np.uint8),
                    )
                except Exception as exc:
                    warnings.append(f"Wetland context: {exc}")
                    run_logger.warning("Step 8 — Wetland context issue: %s", exc)
                    from ..wetland.mask import WetlandMaskResult
                    zero = np.zeros((ref_grid.height, ref_grid.width), dtype=np.uint8)
                    wetland_result = WetlandMaskResult(
                        wetland_mask=zero.copy(),
                        valid_analysis_mask=zero.copy(),
                        source_type="UNAVAILABLE",
                        source_metadata={
                            "status": "Wetland context unavailable."
                        },
                        wetland_area_m2=None,
                        wetland_area_ha=None,
                        valid_analysis_area_m2=None,
                        valid_analysis_area_ha=None,
                        status="WETLAND_BOUNDARY_UNAVAILABLE",
                        physical_wetland_depletion_enabled=False,
                        metadata={
                            "status": "WETLAND_BOUNDARY_UNAVAILABLE",
                            "source_type": "UNAVAILABLE",
                            "boundary_available": False,
                            "wetland_context_available": False,
                            "wetland_metrics_available": False,
                            "wetland_membership_verified": False,
                            "physical_area_available": False,
                            "area_basis": "unavailable",
                            "analysis_scope": "UNAVAILABLE",
                            "wetland_pixel_count": None,
                            "valid_wetland_analysis_pixel_count": None,
                            "wetland_area_m2": None,
                            "wetland_area_ha": None,
                            "valid_wetland_analysis_area_m2": None,
                            "valid_wetland_analysis_area_ha": None,
                            "physical_wetland_depletion_enabled": False,
                        },
                    )

                wetland_context_available = (
                    wetland_result is not None
                    and wetland_result.source_type != "UNAVAILABLE"
                )

                if not wetland_context_available:
                    warnings.append(
                        "Wetland context unavailable; "
                        "general vegetation metrics will be calculated "
                        "without verified wetland membership. Fusion skipped."
                    )

                if (
                    wetland_context_available
                    and (wetland_dir / "wetland_mask.tif").is_file()
                ):
                    outputs["wetland/wetland_mask.tif"] = "wetland/wetland_mask.tif"
                run_logger.info("Step 8 — Wetland context: %s", wetland_result.status)

                # ==============================================================
                # STEP 9 — NDWI Hydrology
                # ==============================================================
                run_logger.info("Step 9 — NDWI hydrology")
                try:
                    from ..wetland.water import analyze_hydrology
                    overlap_mask_tif = alignment_dir / "effective_analysis_mask.tif"
                    wetland_mask_tif  = wetland_dir / "wetland_mask.tif"

                    hydro_metrics = analyze_hydrology(
                        t1_aligned_dir=t1_aligned_out,
                        t2_aligned_dir=t2_aligned_out,
                        output_dir=hydro_dir,
                        config=config,
                        wetland_mask_path=(
                            wetland_mask_tif
                            if wetland_context_available
                            and wetland_mask_tif.is_file()
                            else None
                        ),
                        overlap_mask_path=overlap_mask_tif,
                    )
                    for fname in ["ndwi_t1.tif", "ndwi_t2.tif", "delta_ndwi.tif",
                                  "water_mask_t1.tif", "water_mask_t2.tif", "water_change.tif"]:
                        if (hydro_dir / fname).is_file():
                            outputs[f"hydrology/{fname}"] = f"hydrology/{fname}"
                except Exception as exc:
                    warnings.append(f"Hydrology: {exc}")
                    run_logger.warning("Step 9 — Hydrology issue: %s", exc)

                run_logger.info("Step 9 — Hydrology complete")

                # ==============================================================
                # STEP 10 — Multi-Evidence Fusion
                # ==============================================================
                run_logger.info("Step 10 — Multi-evidence fusion")
                water_change_path = hydro_dir / "water_change.tif"
                if (
                    wetland_context_available
                    and water_change_path.is_file()
                ):
                    try:
                        from ..change.fusion import fuse_evidence
                        fusion_metrics = fuse_evidence(
                            transition_map=transition_map,
                            water_change=water_change_path,
                            valid_analysis_mask=wetland_result.valid_analysis_mask,
                            config=config,
                            delta_ndvi=delta_ndvi,
                            delta_gndvi=delta_gndvi,
                            delta_ndre=delta_ndre,
                            cva_magnitude=cva_magnitude,
                            wetland_mask=wetland_result.wetland_mask,
                            output_dir=fusion_dir,
                            reference_grid=ref_grid,
                        )
                        # Copy fusion metrics JSON to metrics dir
                        fusion_metrics_src = fusion_dir / "fusion_metrics.json"
                        if fusion_metrics_src.is_file():
                            import shutil
                            shutil.copy2(fusion_metrics_src, metrics_dir / "fusion_metrics.json")
                            outputs["metrics/fusion_metrics.json"] = "metrics/fusion_metrics.json"

                        # Copy evidence matrix CSV
                        em_src = fusion_dir / "evidence_matrix.csv"
                        if em_src.is_file():
                            shutil.copy2(em_src, metrics_dir / "evidence_matrix.csv")
                            outputs["metrics/evidence_matrix.csv"] = "metrics/evidence_matrix.csv"

                        if (fusion_dir / "evidence_fusion.tif").is_file():
                            outputs["fusion/evidence_fusion.tif"] = "fusion/evidence_fusion.tif"
                    except Exception as exc:
                        warnings.append(f"Fusion: {exc}")
                        run_logger.warning("Step 10 — Fusion issue: %s", exc)
                else:
                    warnings.append("water_change.tif unavailable; fusion skipped.")
                    run_logger.warning("Step 10 — water_change.tif missing; fusion skipped")

                run_logger.info("Step 10 — Fusion complete")

                # ==============================================================
                # STEP 11 — Metrics
                # ==============================================================
                run_logger.info("Step 11 — Metrics calculation")
                deg_map_path    = deg_dir / "degradation_map.tif"
                rec_map_path    = deg_dir / "recovery_map.tif"
                res_map_path    = deg_dir / "residual_stress.tif"
                canopy_map_path = deg_dir / "canopy_loss_map.tif"
                cva_mag_path    = change_dir / "cva_magnitude.tif"
                trans_path      = change_dir / "transition_map.tif"
                d_ndvi_path     = change_dir / "delta_ndvi.tif"
                d_gndvi_path    = change_dir / "delta_gndvi.tif"
                d_ndre_path     = change_dir / "delta_ndre.tif"
                wetland_mask_tif = wetland_dir / "wetland_mask.tif"

                required = [deg_map_path, rec_map_path, res_map_path, canopy_map_path,
                            cva_mag_path, trans_path, d_ndvi_path, d_gndvi_path, d_ndre_path]
                if all(p.is_file() for p in required):
                    try:
                        from ..metrics.degradation import calculate_metrics
                        if wetland_context_available:
                            wm = wetland_result.wetland_mask
                            vm = wetland_result.valid_analysis_mask
                            metrics_scope = 'WETLAND_CONTEXT'
                        else:
                            vm = np.asarray(effective_overlap_mask)
                            if vm.ndim != 2 or not np.all(
                                np.isin(vm, [0, 1])
                            ):
                                raise ValueError(
                                    'Invalid effective analysis mask.'
                                )
                            vm = vm.astype(np.uint8, copy=True)
                            wm = vm.copy()
                            metrics_scope = 'GENERAL_VEGETATION'

                        metrics_result = calculate_metrics(
                            transition_map=trans_path,
                            cva_magnitude=cva_mag_path,
                            delta_ndvi=d_ndvi_path,
                            delta_gndvi=d_gndvi_path,
                            delta_ndre=d_ndre_path,
                            degradation_map=deg_map_path,
                            recovery_map=rec_map_path,
                            residual_stress_map=res_map_path,
                            canopy_loss_map=canopy_map_path,
                            wetland_mask=wm,
                            valid_analysis_mask=vm,
                            config=config,
                            reference_grid=ref_grid,
                        )
                        metrics_result_dict = metrics_result.to_dict()
                        metrics_result_dict['analysis_scope'] = metrics_scope
                        metrics_result_dict['wetland_context_available'] = bool(
                            wetland_context_available
                        )
                        metrics_result_dict['wetland_membership_verified'] = False
                        metrics_result_dict['metadata']['analysis_scope'] = metrics_scope
                        if not wetland_context_available:
                            metrics_result_dict['metadata']['scope_disclaimer'] = (
                                'General vegetation metrics within the effective '
                                'analysis footprint; not verified wetland metrics.'
                            )
                        _write_json(metrics_dir / "metrics.json", metrics_result_dict)
                        outputs["metrics/metrics.json"] = "metrics/metrics.json"

                        # Copy transition matrix CSV if produced by calculate_transitions
                        trans_csv_src = change_dir / "transition_matrix.csv"
                        if trans_csv_src.is_file():
                            import shutil
                            shutil.copy2(trans_csv_src, metrics_dir / "transition_matrix.csv")
                            outputs["metrics/transition_matrix.csv"] = "metrics/transition_matrix.csv"
                    except Exception as exc:
                        warnings.append(f"Metrics: {exc}")
                        run_logger.warning("Step 11 — Metrics issue: %s", exc)
                else:
                    warnings.append(
                        "Some raster outputs missing; full metrics skipped."
                    )

                # Water metrics JSON
                water_metrics_src = hydro_dir / "water_metrics.json"
                if water_metrics_src.is_file():
                    import shutil
                    shutil.copy2(water_metrics_src, metrics_dir / "water_metrics.json")
                    outputs["metrics/water_metrics.json"] = "metrics/water_metrics.json"

                run_logger.info("Step 11 — Metrics complete")

                # ==============================================================
                # STEP 12 — Quality Assurance
                # ==============================================================

            # QA applies to every two-date run.
            try:
                from ..validation.quality import QualityAssuranceEngine

                qa_engine = QualityAssuranceEngine(config)

                qa_engine.configure_required_checks("two_date")

                qa_engine.execute_check(
                    "uncertainty",
                    config,
                    reference_samples_exist=False,
                )

                # Production class distribution QA.
                if (
                    t1_health is not None
                    and t2_health is not None
                    and effective_overlap_mask is not None
                ):
                    qa_engine.execute_check(
                        "class_distribution",
                        t1_health,
                        t2_health,
                        effective_overlap_mask & pair_result.t1_valid_mask & pair_result.t2_valid_mask,
                    )
                else:
                    qa_engine.record_check(
                        "class_distribution",
                        "NOT_AVAILABLE",
                    )


                # Production temporal QA.
                if overlap_mask is not None:
                    qa_engine.execute_check(
                        "temporal",
                        t1_date,
                        t2_date,
                        None,
                        None,
                        overlap_mask,
                    )
                else:
                    qa_engine.record_check(
                        "temporal",
                        "NOT_AVAILABLE",
                    )


                # Production transition QA.
                if all(
                    value is not None
                    for value in (
                        transition_map,
                        t1_health,
                        t2_health,
                        effective_overlap_mask,
                    )
                ):
                    qa_engine.execute_check(
                        "transition",
                        transition_map,
                        t1_health,
                        t2_health,
                        trans_result.valid_mask,
                    )
                else:
                    qa_engine.record_check(
                        "transition",
                        "NOT_AVAILABLE",
                    )


                # Production CVA mathematical QA.
                if all(
                    value is not None
                    for value in (
                        cva_magnitude,
                        delta_ndvi,
                        delta_gndvi,
                        delta_ndre,
                        effective_overlap_mask,
                    )
                ):
                    qa_engine.execute_check(
                        "cva",
                        cva_magnitude,
                        delta_ndvi,
                        delta_gndvi,
                        delta_ndre,
                        cva_result.valid_mask,
                    )
                else:
                    qa_engine.record_check(
                        "cva",
                        "NOT_AVAILABLE",
                    )


                # Production wetland context QA.
                if wetland_result is not None:
                    qa_engine.execute_check(
                        "wetland",
                        wetland_result.wetland_mask,
                        wetland_result.valid_analysis_mask,
                        config,
                        wetland_result=wetland_result,
                    )
                else:
                    qa_engine.record_check(
                        "wetland",
                        "NOT_AVAILABLE",
                    )


                # Production metrics reconciliation QA.
                if metrics_result_dict:
                    qa_engine.execute_check(
                        "metrics",
                        metrics_result_dict,
                        config,
                    )
                else:
                    qa_engine.record_check(
                        "metrics",
                        "NOT_AVAILABLE",
                    )

                # Validate actual aligned input bands.
                if aligned_pair is not None:

                    def _qa_normalize_bands(bands):
                        normalized = {}

                        for name, values in bands.items():

                            if (
                                isinstance(values, np.ndarray)
                                and values.dtype.kind in "iuf"
                            ):
                                normalized[name] = np.where(
                                    values == NODATA_CONTINUOUS,
                                    np.nan,
                                    values,
                                )
                            else:
                                normalized[name] = values

                        return normalized

                    qa_engine.execute_check(
                        "input",
                        _qa_normalize_bands(aligned_pair.t1_data),
                        _qa_normalize_bands(aligned_pair.t2_data),
                        aligned_pair.t1_metadata,
                        aligned_pair.t2_aligned_metadata,
                    )

                else:
                    qa_engine.record_check(
                        "input",
                        "NOT_AVAILABLE",
                    )


                # Validate actual degradation and recovery outputs.
                if (
                    deg_result is not None
                    and effective_overlap_mask is not None
                ):

                    qa_engine.execute_check(
                        "change_patch",
                        deg_result.degradation_map,
                        deg_result.recovery_map,
                        deg_result.canopy_loss,
                        deg_result.residual_stress,
                        trans_result.valid_mask & cva_result.valid_mask,
                    )

                else:
                    qa_engine.record_check(
                        "change_patch",
                        "NOT_AVAILABLE",
                    )

                if pair_result is not None:

                    _execute_production_model_qa(
                        qa_engine,
                        model_path,
                        t1_veg,
                        "T1",
                    )

                    _execute_production_model_qa(
                        qa_engine,
                        model_path,
                        t2_veg,
                        "T2",
                    )

                else:

                    qa_engine.record_check(
                        "model",
                        "NOT_AVAILABLE",
                    )

                spatial_paths = {}

                for band_name in ("green", "red", "rededge", "nir"):

                    spatial_paths[f"T1/{band_name}"] = (
                        t1_aligned_out / f"{band_name}.tif"
                    )

                    spatial_paths[f"T2/{band_name}"] = (
                        t2_aligned_out / f"{band_name}.tif"
                    )

                # Inspect registered raster outputs only.
                for relative_path in outputs.values():

                    relative_path = Path(relative_path)

                    if relative_path.suffix.lower() in (".tif", ".tiff"):

                        spatial_paths[str(relative_path)] = (
                            run_dir / relative_path
                        )

                _execute_production_spatial_qa(
                    qa_engine,
                    ref_grid,
                    spatial_paths,
                )

                nodata_specs = {}

                if pair_result is not None:

                    for label, vegetation in (
                        ("T1", t1_veg),
                        ("T2", t2_veg),
                    ):

                        nodata_specs[f"{label} health"] = (
                            vegetation.health_map,
                            vegetation.valid_mask,
                            255,
                            (0, 1, 2, 3),
                            None,
                        )

                        nodata_specs[f"{label} confidence"] = (
                            vegetation.confidence,
                            vegetation.valid_mask,
                            0.0,
                            None,
                            (0.0, 1.0),
                        )

                else:
                    nodata_specs["Vegetation"] = (
                        None, None, 255, None, None
                    )

                if transition_map is not None:

                    nodata_specs["Transition"] = (
                        transition_map,
                        trans_result.valid_mask,
                        255,
                        tuple(range(16)),
                        None,
                    )

                else:
                    nodata_specs["Transition"] = (
                        None, None, 255, None, None
                    )

                if cva_magnitude is not None:

                    for label, values, bounds in (
                        ("CVA magnitude", cva_magnitude, (0.0, None)),
                        ("Delta NDVI", delta_ndvi, None),
                        ("Delta GNDVI", delta_gndvi, None),
                        ("Delta NDRE", delta_ndre, None),
                    ):

                        nodata_specs[label] = (
                            values,
                            cva_result.valid_mask,
                            NODATA_CONTINUOUS,
                            None,
                            bounds,
                        )

                else:
                    nodata_specs["CVA"] = (
                        None, None, NODATA_CONTINUOUS, None, None
                    )

                if (
                    deg_result is not None
                    and transition_map is not None
                    and cva_magnitude is not None
                ):

                    patch_valid = (
                        trans_result.valid_mask
                        & cva_result.valid_mask
                    )

                    for label, values in (
                        ("Degradation", deg_result.degradation_map),
                        ("Recovery", deg_result.recovery_map),
                        ("Canopy loss", deg_result.canopy_loss),
                        ("Residual stress", deg_result.residual_stress),
                    ):

                        nodata_specs[label] = (
                            values,
                            patch_valid,
                            255,
                            (0, 1),
                            None,
                        )

                else:
                    nodata_specs["Change patches"] = (
                        None, None, 255, None, None
                    )

                _execute_production_nodata_qa(
                    qa_engine,
                    nodata_specs,
                )

                qa_dict = qa_engine.generate_qa_report()

                if qa_dict.get("overall_status") != "PASS":
                    warnings.append(
                        "QA validation is incomplete or contains findings."
                    )

            except Exception as exc:

                qa_dict = _qa_error_report(exc)

                warnings.append(f"QA: {exc}")

                run_logger.warning(
                    "Quality assurance issue: %s",
                    exc,
                )

            _write_json(
                val_dir / "qa_report.json",
                qa_dict,
            )

            outputs["validation/qa_report.json"] = (
                "validation/qa_report.json"
            )

            run_logger.info(
                "Quality assurance status: %s",
                qa_dict.get("overall_status", "UNKNOWN"),
            )

                # ==============================================================

            # STEP 13 — Assemble result package
            # ==============================================================
            run_logger.info("Step 13 — Assembling result package")
            total_runtime = time.perf_counter() - run_start

            veg_stats: Dict[str, Any] = {}
            if pair_result:
                veg_stats = {
                    "t1": pair_result.class_statistics_t1,
                    "t2": pair_result.class_statistics_t2,
                }

            deg_stats: Dict[str, Any] = {}
            rec_stats: Dict[str, Any] = {}
            if deg_result:
                stats = deg_result.statistics
                deg_count = stats.get("degradation_pixel_count", 0)
                rec_count = stats.get("recovery_pixel_count", 0)
                valid_cnt = stats.get("valid_pixel_count", 1) or 1
                deg_pct   = stats.get("degradation_percentage", 0.0)
                rec_pct   = stats.get("recovery_percentage", 0.0)
                # Area from metrics if available — use the actual field names from MetricsResult
                deg_area_ha = metrics_result_dict.get("impact_area", {}).get("impact_area_ha", 0.0) if metrics_result_dict else 0.0
                rec_area_ha = metrics_result_dict.get("recovery_area", {}).get("area_ha", 0.0) if metrics_result_dict else 0.0
                deg_stats = {"pixel_count": deg_count, "area_ha": deg_area_ha, "percentage": deg_pct}
                rec_stats = {"pixel_count": rec_count, "area_ha": rec_area_ha, "percentage": rec_pct}

            cva_stats: Dict[str, Any] = {}
            if cva_result:
                cva_stats = cva_result.statistics

            change_summary: Dict[str, Any] = {}
            if trans_result:
                change_summary = {
                    cat: {
                        "count": trans_result.category_counts.get(cat, 0),
                        "area_ha": trans_result.category_areas_ha.get(cat, 0.0),
                        "percentage": trans_result.category_percentages.get(cat, 0.0),
                    }
                    for cat in trans_result.category_counts
                }

            # Authoritative spectral index statistics (NDVI, GNDVI, NDRE, NDWI)
            index_stats: Dict[str, Any] = {}
            if aligned_pair is not None and effective_overlap_mask is not None:
                nir1, red1 = aligned_pair.t1_data.get("nir"), aligned_pair.t1_data.get("red")
                green1, re1 = aligned_pair.t1_data.get("green"), aligned_pair.t1_data.get("rededge")
                nir2, red2 = aligned_pair.t2_data.get("nir"), aligned_pair.t2_data.get("red")
                green2, re2 = aligned_pair.t2_data.get("green"), aligned_pair.t2_data.get("rededge")

                eps = 1e-7

                def _calc_idx_stats(arr1: np.ndarray, arr2: np.ndarray, formula_str: str) -> Dict[str, Any]:
                    v1 = arr1[effective_overlap_mask & ~np.isnan(arr1) & ~np.isinf(arr1)] if arr1 is not None else np.array([])
                    v2 = arr2[effective_overlap_mask & ~np.isnan(arr2) & ~np.isinf(arr2)] if arr2 is not None else np.array([])
                    d_arr = (arr2 - arr1) if (arr1 is not None and arr2 is not None) else None
                    vd = d_arr[effective_overlap_mask & ~np.isnan(d_arr) & ~np.isinf(d_arr)] if d_arr is not None else np.array([])
                    return {
                        "formula": formula_str,
                        "t1_mean": float(np.mean(v1)) if v1.size else 0.0,
                        "t1_median": float(np.median(v1)) if v1.size else 0.0,
                        "t1_std": float(np.std(v1)) if v1.size else 0.0,
                        "t1_p95": float(np.percentile(v1, 95.0)) if v1.size else 0.0,
                        "t2_mean": float(np.mean(v2)) if v2.size else 0.0,
                        "t2_median": float(np.median(v2)) if v2.size else 0.0,
                        "t2_std": float(np.std(v2)) if v2.size else 0.0,
                        "t2_p95": float(np.percentile(v2, 95.0)) if v2.size else 0.0,
                        "delta_mean": float(np.mean(vd)) if vd.size else 0.0,
                        "delta_median": float(np.median(vd)) if vd.size else 0.0,
                        "delta_std": float(np.std(vd)) if vd.size else 0.0,
                        "delta_p95": float(np.percentile(vd, 95.0)) if vd.size else 0.0,
                    }

                if nir1 is not None and red1 is not None and nir2 is not None and red2 is not None:
                    ndvi1 = (nir1 - red1) / (nir1 + red1 + eps)
                    ndvi2 = (nir2 - red2) / (nir2 + red2 + eps)
                    index_stats["ndvi"] = _calc_idx_stats(ndvi1, ndvi2, "(NIR - Red) / (NIR + Red)")

                if nir1 is not None and green1 is not None and nir2 is not None and green2 is not None:
                    gndvi1 = (nir1 - green1) / (nir1 + green1 + eps)
                    gndvi2 = (nir2 - green2) / (nir2 + green2 + eps)
                    index_stats["gndvi"] = _calc_idx_stats(gndvi1, gndvi2, "(NIR - Green) / (NIR + Green)")

                if nir1 is not None and re1 is not None and nir2 is not None and re2 is not None:
                    ndre1 = (nir1 - re1) / (nir1 + re1 + eps)
                    ndre2 = (nir2 - re2) / (nir2 + re2 + eps)
                    index_stats["ndre"] = _calc_idx_stats(ndre1, ndre2, "(NIR - RedEdge) / (NIR + RedEdge)")

                if dispatcher.is_wetland:
                    if green1 is not None and nir1 is not None and green2 is not None and nir2 is not None:
                        ndwi1 = (green1 - nir1) / (green1 + nir1 + eps)
                        ndwi2 = (green2 - nir2) / (green2 + nir2 + eps)
                        index_stats["ndwi"] = _calc_idx_stats(
                            ndwi1,
                            ndwi2,
                            "(Green - NIR) / (Green + NIR)",
                        )

            # Physical depletion
            # Configuration flags alone do not verify independent
            # hydrological evidence.
            phys_depl = PHYSICAL_DEPLETION_DEFAULT

            # Validate the existing reporting indices without
            # recomputing or replacing scientific outputs.

            _production_index_qa_completed = False

            try:

                if (
                    "qa_engine" in locals()
                    and effective_overlap_mask is not None
                ):

                    available_indices = locals()

                    t1_indices_qa = {
                        "ndvi": available_indices.get("ndvi1"),
                        "gndvi": available_indices.get("gndvi1"),
                        "ndre": available_indices.get("ndre1"),
                    }

                    t2_indices_qa = {
                        "ndvi": available_indices.get("ndvi2"),
                        "gndvi": available_indices.get("gndvi2"),
                        "ndre": available_indices.get("ndre2"),
                    }

                    def _normalize_qa_index(values):

                        if (
                            isinstance(values, np.ndarray)
                            and values.dtype.kind in "iuf"
                        ):
                            return np.where(
                                values == NODATA_CONTINUOUS,
                                np.nan,
                                values,
                            )

                        return values

                    t1_indices_qa = {
                        name: _normalize_qa_index(values)
                        for name, values in t1_indices_qa.items()
                    }

                    t2_indices_qa = {
                        name: _normalize_qa_index(values)
                        for name, values in t2_indices_qa.items()
                    }

                    qa_engine.execute_check(
                        "vegetation_index",
                        t1_indices_qa,
                        t2_indices_qa,
                        effective_overlap_mask,
                    )

                    _production_index_qa_completed = True

                elif "qa_engine" in locals():

                    qa_engine.record_check(
                        "vegetation_index",
                        "NOT_AVAILABLE",
                    )

                if "qa_engine" in locals():

                    qa_dict = qa_engine.generate_qa_report()

            except Exception as index_qa_error:

                if "qa_engine" in locals():

                    qa_engine.add_flag(
                        "QA-EXECUTION",
                        "HIGH",
                        "Vegetation index QA execution failed: "
                        + type(index_qa_error).__name__,
                        "Preprocessing",
                        "FAIL",
                        "Inspect reporting index availability and validity.",
                    )

                    qa_engine.record_check(
                        "vegetation_index",
                        "FAIL",
                    )

                    qa_dict = qa_engine.generate_qa_report()

                else:

                    qa_dict = _qa_error_report(index_qa_error)

                warnings.append(
                    "Vegetation index QA execution failed."
                )

            # Keep the exported QA report synchronized with
            # the validation data included in analysis results.

            # Finalize provenance before exporting QA and reports.

            qa_dict, final_run_provenance = (
                _finalize_production_provenance_qa(
                    locals().get("qa_engine"),
                    qa_dict,
                    initial_run_provenance,
                    model_path,
                    provenance_input_paths,
                    config,
                    config_snapshot_path,
                    warnings,
                )
            )

            _write_json(
                val_dir / "qa_report.json",
                qa_dict,
            )

            result_content: Dict[str, Any] = {
                "run_id": run_id,
                "software_version": SOFTWARE_VERSION,
                "analysis_mode": dispatcher.analysis_mode,
                "status": "WARNING" if warnings else "SUCCESS",

                "inputs": {
                    "t1_path": str(t1_scene),
                    "t2_path": str(t2_scene),
                    "model_path": str(model_path),
                    "wetland_boundary": str(wetland_boundary) if wetland_boundary else None,
                    "band_statistics": band_statistics,
                },

                "temporal": {
                    "t1_date": t1_date,
                    "t2_date": t2_date,
                    "interval_days": interval_days,
                },

                "spectral_indices": index_stats,

                "spatial": {
                    "crs": str(ref_grid.crs),
                    "dimensions": [ref_grid.height, ref_grid.width],
                    "resolution": list(ref_grid.resolution),
                    "extent": list(ref_grid.bounds),
                    "overlap_percentage": aligned_pair.overlap_percentage if aligned_pair else None,
                    "total_overlap_area_ha": total_overlap_area_ha,
                    "inclusion_area_ha": inclusion_area_ha,
                    "manual_excluded_area_ha": excluded_area_ha,
                    "effective_analysis_area_ha": effective_analysis_area_ha,
                    "physical_area_available": physical_area_available,
                    "area_basis": area_basis,
                    "overlap_pixel_count": overlap_pixel_count,
                    "inclusion_pixel_count": inclusion_pixel_count,
                    "excluded_pixel_count": excluded_pixel_count,
                    "effective_pixel_count": effective_pixel_count,
                },

                # Common vegetation-analysis results.
                "vegetation": veg_stats,
                "change": {
                    "category_summary": change_summary,
                },
                "cva": cva_stats,
                "degradation": deg_stats,
                "recovery": rec_stats,

                # Common analysis metrics.
                "metrics": {
                    "normalized_degradation_ratio": (
                        metrics_result_dict.get("normalized_degradation_ratio")
                        if metrics_result_dict
                        else None
                    ),

                    "severity_weighted_degradation_area_ha": (
                        metrics_result_dict.get("severity_weighted_area", {})
                        .get("severity_weighted_area_ha_deg")
                        if metrics_result_dict
                        else None
                    ),

                    # Stable public API fields.
                    "impact_area": {
                        "degradation_ha": (
                            metrics_result_dict.get("impact_area", {})
                            .get("impact_area_ha", 0.0)
                            if metrics_result_dict
                            else 0.0
                        ),
                    },

                    "recovery_area": {
                        "recovery_ha": (
                            metrics_result_dict.get("recovery_area", {})
                            .get("area_ha", 0.0)
                            if metrics_result_dict
                            else 0.0
                        ),
                    },

                    "cva_magnitude": {
                        "mean": (
                            (
                                metrics_result_dict
                                .get("cva_statistics", {})
                                .get("all_valid_wetland", {})
                                .get("mean")
                            )
                            if (
                                metrics_result_dict
                                and metrics_result_dict
                                .get("cva_statistics", {})
                                .get("all_valid_wetland", {})
                                .get("mean") is not None
                            )
                            else 0.0
                        ),
                    },

                    "analysis_extent": {
                        "valid_analysis_area_ha": (
                            metrics_result_dict.get("baseline_area", {})
                            .get("area_ha", 0.0)
                            if metrics_result_dict
                            else 0.0
                        ),
                    },

                    "transition_matrix": {
                        "matrix_m2": [
                            [float(trans_result.transition_areas_m2.get(i * 4 + j) or 0.0) for j in range(4)]
                            for i in range(4)
                        ],
                        "matrix_pixels": [
                            [int(trans_result.transition_counts.get(i * 4 + j) or 0) for j in range(4)]
                            for i in range(4)
                        ],
                    } if trans_result else None,
                },

                "validation": qa_dict,

                "uncertainty": {
                    "mean_confidence_t1": (
                        float(np.nanmean(pair_result.t1_confidence))
                        if pair_result and hasattr(pair_result, "t1_confidence")
                        else None
                    ),
                },

                "warnings": warnings,
                "errors": errors,
                "outputs": outputs,

                "performance": {
                    "total_runtime_seconds": total_runtime,
                    "inference": perf if pair_result else {},
                },
            }

            _write_json(run_dir / "analysis_result.json", result_content)

            if dispatcher.is_wetland:
                result_content["wetland"] = {
                    "status": wetland_result.status if wetland_result else "UNAVAILABLE",
                    "source_type": wetland_result.source_type if wetland_result else "UNAVAILABLE",
                    "area_ha": wetland_result.wetland_area_ha if wetland_result else 0.0,
                    "physical_depletion_enabled": (
                        wetland_result.physical_wetland_depletion_enabled
                        if wetland_result
                        else False
                    ),
                }

                result_content["hydrology"] = {
                    "ndwi_threshold": config.get("water", {}).get("ndwi_threshold", 0.0),
                    "water_gain_area_ha": hydro_metrics.get("water_gain_ha"),
                    "water_loss_area_ha": hydro_metrics.get("water_loss_ha"),
                    "water_area_t1_ha": hydro_metrics.get("water_area_t1_ha"),
                    "water_area_t2_ha": hydro_metrics.get("water_area_t2_ha"),
                }

                result_content["fusion"] = {
                    "evidence_level": fusion_metrics.get(
                        "physical_depletion_assessment",
                        "N/A",
                    ),
                    "hydro_ecological_degradation_area_ha": fusion_metrics.get(
                        "hydro_ecological_degradation_area_ha"
                    ),
                    "hydro_ecological_recovery_area_ha": fusion_metrics.get(
                        "hydro_ecological_recovery_area_ha"
                    ),
                }

                result_content["physical_depletion_assessment"] = phys_depl
            _write_json(run_dir / "analysis_result.json", result_content)
            outputs["analysis_result.json"] = "analysis_result.json"

            # ==============================================================
            # STEP 14 — Generate Analysis Reports (Markdown, Figures & PDF)
            # ==============================================================
            run_logger.info("Step 14 — Generating analysis reports")
            try:
                from ..output.report import generate_analysis_report
                generate_analysis_report(
                    result_dict=result_content,
                    output_path=reports_dir / "analysis_report.md",
                    run_id=run_id,
                )
                outputs["reports/analysis_report.md"] = "reports/analysis_report.md"
            except Exception as exc:
                warnings.append(f"Markdown Report: {exc}")
                run_logger.warning("Step 14 — Markdown report warning: %s", exc)

            map_assets: Dict[str, Path] = {}
            try:
                from ..output.cartography import generate_report_maps
                map_assets = generate_report_maps(
                    run_dir=run_dir,
                    figures_dir=reports_dir / "figures",
                )
                for fig_key, fig_path in map_assets.items():
                    rel_fig = str(fig_path.relative_to(run_dir)).replace("\\", "/")
                    outputs[f"reports/figures/{fig_path.name}"] = rel_fig
            except Exception as exc:
                warnings.append(f"Map Rendering: {exc}")
                run_logger.warning("Step 14 — Map rendering warning: %s", exc)

            try:
                from ..output.pdf import generate_all_reports
                generated_pdfs = generate_all_reports(
                    result_dict=result_content,
                    run_dir=run_dir,
                    reports_dir=reports_dir,
                    map_assets=map_assets,
                    run_id=run_id,
                )
                # Register all generated PDFs in the output manifest
                for pdf_key, pdf_path in generated_pdfs.items():
                    rel = str(pdf_path.relative_to(run_dir)).replace("\\", "/")
                    outputs[rel] = rel
            except Exception as exc:
                warnings.append(f"PDF Report: {exc}")
                run_logger.warning("Step 14 — PDF report warning: %s", exc)

            # ==============================================================
            # STEP 15 — Provenance metadata.json
            # ==============================================================
            run_logger.info("Step 15 — Writing provenance metadata")
            model_sha256 = _file_sha256(model_path)
            try:
                import subprocess
                git_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                ).decode().strip()
            except Exception:
                git_commit = "unavailable"

            metadata: Dict[str, Any] = {
                "run_id": run_id,
                "software_version": SOFTWARE_VERSION,
                "analysis_mode": dispatcher.analysis_mode,
                "git_commit": git_commit,
                "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "python_version": sys.version,
                "os": platform.platform(),
                "model_path": str(model_path),
                "model_sha256": model_sha256,
                "configuration_hash": cfg_hash,
                "config_snapshot": str(config_snapshot_path),
                "input_t1": _file_fingerprint(t1_scene),
                "input_t2": _file_fingerprint(t2_scene),
                "t1_date": t1_date,
                "t2_date": t2_date,
                "crs": str(ref_grid.crs) if ref_grid else None,
                "resolution": list(ref_grid.resolution) if ref_grid else None,
                "dimensions": [ref_grid.height, ref_grid.width] if ref_grid else None,
                "total_runtime_seconds": time.perf_counter() - run_start,
            }
            # Preserve observed initial and final run provenance.

            metadata["initial_provenance"] = initial_run_provenance

            metadata["final_provenance"] = final_run_provenance

            metadata["reproducibility_verified"] = False

            metadata["provenance_comparison"] = qa_dict.get(
                "reproducibility",
                {},
            )

            # Retain legacy metadata fields using initial fingerprints.

            for provenance_key in (
                "git_commit",
                "model_sha256",
                "configuration_hash",
                "config_snapshot",
                "input_scene",
                "input_t1",
                "input_t2",
            ):

                if provenance_key in initial_run_provenance:

                    metadata[provenance_key] = (
                        initial_run_provenance[provenance_key]
                    )

            metadata["manifest_integrity_report"] = (
                "manifest_integrity.json"
            )

            _write_json(run_dir / "metadata.json", metadata)
            outputs["metadata.json"] = "metadata.json"

            # ==============================================================
            # STEP 16 — Output manifest.json
            # ==============================================================
            run_logger.info("Step 16 — Writing output manifest")
            manifest, manifest_integrity = (
                _finalize_output_manifest(
                    run_dir,
                    run_id,
                    outputs,
                    config_snapshot_path,
                    warnings,
                    errors,
                )
            )

            # ==============================================================
            # Done
            # ==============================================================
            final_status = _resolve_final_execution_status(
                qa_dict,
                manifest_integrity,
                warnings,
                errors,
            )

            # Persist terminal status outside the immutable manifest.

            final_status = _persist_terminal_run_status(
                run_dir,
                run_id,
                final_status,
                qa_dict,
                manifest_integrity,
                warnings,
                errors,
            )

            if (run_dir / "run_status.json").is_file():

                outputs["run_status.json"] = "run_status.json"
            run_logger.info(
                "Phase 13 — Pipeline COMPLETE. status=%s run_id=%s runtime=%.1fs",
                final_status, run_id, time.perf_counter() - run_start,
            )

            return AnalysisResult(
                run_id=run_id,
                status=final_status,
                output_directory=str(run_dir),
                outputs=outputs,
                metrics=result_content.get("metrics", {}),
                validation=qa_dict,
                warnings=warnings,
                errors=errors,
                metadata=metadata,
            )

        except (InputValidationError, ConfigurationError, AlignmentError,
                InferenceError, HydrologyError, FusionError, MetricError,
                OutputError, ValidationError, GISProcessingError) as exc:
            errors.append(f"{exc.__class__.__name__}: {exc}")
            run_logger.error("Phase 13 — Pipeline FAILED: %s", exc)
            return AnalysisResult(
                run_id=run_id,
                status="FAILED",
                output_directory=str(run_dir),
                outputs=outputs,
                metrics={},
                validation={},
                warnings=warnings,
                errors=errors,
                metadata={"run_id": run_id, "error": str(exc)},
            )
        except Exception as exc:
            errors.append(f"Internal error: {exc}")
            run_logger.error("Phase 13 — Unexpected error: %s", exc, exc_info=True)
            return AnalysisResult(
                run_id=run_id,
                status="FAILED",
                output_directory=str(run_dir),
                outputs=outputs,
                metrics={},
                validation={},
                warnings=warnings,
                errors=errors,
                metadata={"run_id": run_id, "error": str(exc)},
            )
        finally:
            if fh is not None:
                fh.close()
                logging.getLogger("gis_engine").removeHandler(fh)


def run_analysis(
    t1_scene: Union[str, Path],
    t2_scene: Union[str, Path],
    config_path: Union[str, Path],
    model_path: Union[str, Path],
    output_base_dir: Union[str, Path],
    t1_date: str,
    t2_date: str,
    wetland_boundary: Optional[Union[str, Path]] = None,
    run_id: Optional[str] = None,
    log_level: str = "INFO",
) -> AnalysisResult:
    """Convenience helper invoking ProductionPipeline().run_analysis()."""
    pipeline = ProductionPipeline()
    return pipeline.run_analysis(
        t1_scene=t1_scene,
        t2_scene=t2_scene,
        t1_date=t1_date,
        t2_date=t2_date,
        config_path=config_path,
        model_path=model_path,
        output_base_dir=output_base_dir,
        wetland_boundary=wetland_boundary,
        run_id=run_id,
        log_level=log_level,
    )


