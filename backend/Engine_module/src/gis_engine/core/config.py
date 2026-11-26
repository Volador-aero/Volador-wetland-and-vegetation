"""
src/gis_engine/core/config.py
================================
Configuration loader and snapshot utilities for the production engine.

The authoritative configuration format is JSON (configs/config.json).
Each production run saves a verbatim snapshot of the configuration used
as config_used.json inside the run output directory.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Union

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Required top-level configuration keys
_REQUIRED_KEYS = {"model", "preprocessing", "water", "ingestion"}

# Required model sub-keys
_REQUIRED_MODEL_KEYS = {"path", "channel_order", "in_channels", "num_classes"}


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads and validates the production configuration from a JSON file.

    Args:
        config_path: Path to the config JSON file.

    Returns:
        Configuration dictionary.

    Raises:
        ConfigurationError: If the file is missing, malformed, or missing
            required keys.
    """
    config_path = Path(config_path)

    if not config_path.is_file():
        raise ConfigurationError(
            f"Configuration file not found: {config_path}"
        )

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"Configuration file is not valid JSON: {config_path}. "
            f"Detail: {exc}"
        ) from exc

    if not isinstance(config, dict):
        raise ConfigurationError(
            f"Configuration must be a JSON object (dict), got {type(config).__name__}."
        )

    # Validate required top-level keys
    missing_top = _REQUIRED_KEYS - set(config.keys())
    if missing_top:
        raise ConfigurationError(
            f"Configuration is missing required top-level keys: {sorted(missing_top)}"
        )

    # Validate required model sub-keys
    model_cfg = config.get("model", {})
    missing_model = _REQUIRED_MODEL_KEYS - set(model_cfg.keys())
    if missing_model:
        raise ConfigurationError(
            f"Configuration 'model' section is missing required keys: {sorted(missing_model)}"
        )

    # Validate channel order
    channel_order = model_cfg.get("channel_order", [])
    if not isinstance(channel_order, list) or len(channel_order) == 0:
        raise ConfigurationError(
            f"Configuration 'model.channel_order' must be a non-empty list, got: {channel_order!r}"
        )

    logger.debug("Configuration loaded from: %s", config_path)
    return config


def config_hash(config: Dict[str, Any]) -> str:
    """Hash a valid JSON configuration deterministically."""
    import math

    if not isinstance(config, dict):
        raise ConfigurationError(
            "Configuration must be a dictionary."
        )

    def validate(value, location="config"):

        if value is None or isinstance(value, (str, bool, int)):
            return

        if isinstance(value, float):

            if not math.isfinite(value):
                raise ConfigurationError(
                    f"Non-finite configuration value at {location}."
                )

            return

        if isinstance(value, list):

            for index, item in enumerate(value):
                validate(item, f"{location}[{index}]")

            return

        if isinstance(value, dict):

            for key, item in value.items():

                if not isinstance(key, str):
                    raise ConfigurationError(
                        f"Invalid configuration key at {location}."
                    )

                validate(item, f"{location}.{key}")

            return

        raise ConfigurationError(
            f"Unsupported configuration value at {location}."
        )

    try:

        validate(config)

        serialized = json.dumps(
            config,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")

        return hashlib.sha256(serialized).hexdigest()

    except (TypeError, ValueError, RecursionError) as exc:

        raise ConfigurationError(
            "Configuration cannot be safely serialized."
        ) from exc


def save_config_snapshot(config: Dict[str, Any], output_dir: Union[str, Path]) -> Path:
    """
    Saves a verbatim JSON snapshot of the configuration used in this run.
    The snapshot file is named ``config_used.json`` and is written inside
    the provided output directory.

    Args:
        config: Configuration dictionary to snapshot.
        output_dir: Run output directory path.

    Returns:
        Path to the written snapshot file.

    Raises:
        OutputError: If writing fails.
    """
    from .exceptions import OutputError  # local import to avoid circular deps

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "config_used.json"

    try:
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    except OSError as exc:
        raise OutputError(
            f"Failed to write configuration snapshot to {snapshot_path}: {exc}"
        ) from exc

    logger.debug("Configuration snapshot saved: %s", snapshot_path)
    return snapshot_path

def collect_environment_provenance() -> Dict[str, Any]:
    """Collect execution environment metadata without changing it."""
    import sys
    import platform
    import subprocess
    from importlib import metadata

    dependencies = (
        "numpy",
        "scipy",
        "rasterio",
        "affine",
        "pyproj",
        "onnxruntime",
        "onnxruntime-gpu",
        "opencv-python",
        "opencv-python-headless",
        "shapely",
        "pandas",
        "matplotlib",
        "reportlab",
        "pypdf",
        "fastapi",
        "pydantic",
        "uvicorn",
    )

    versions = {}

    for package in dependencies:

        try:

            versions[package] = metadata.version(package)

        except metadata.PackageNotFoundError:

            versions[package] = None

    repo_location = Path(__file__).resolve().parent

    def git_command(arguments):

        try:

            result = subprocess.run(
                ["git", "-C", str(repo_location), *arguments],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
                shell=False,
            )

            return result.stdout.strip()

        except (
            OSError,
            subprocess.SubprocessError,
            UnicodeError,
        ):

            return None

    commit = git_command(
        ["rev-parse", "HEAD"]
    )

    working_tree = git_command([
        "status",
        "--porcelain",
        "--untracked-files=normal",
    ])

    environment = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "os": platform.platform(),
        "architecture": platform.machine(),
        "dependencies": versions,
        "git_commit": commit,
        "git_working_tree_dirty": (
            bool(working_tree)
            if working_tree is not None
            else None
        ),
    }

    environment["environment_hash"] = config_hash(
        environment
    )

    environment["reproducibility_verified"] = False

    return environment
