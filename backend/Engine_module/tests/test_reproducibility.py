import hashlib
import json
from copy import deepcopy

import pytest

from src.gis_engine.core.config import (
    config_hash,
    collect_environment_provenance,
)
from src.gis_engine.core.exceptions import ConfigurationError
from src.gis_engine.core.pipeline import (
    _file_sha256,
    _file_fingerprint,
    _output_entry,
)
from src.gis_engine.validation.quality import (
    QualityAssuranceEngine,
)


def test_configuration_hash_legacy_compatibility():
    config = {"model": {"channels": 5}, "threshold": 0.5}

    expected = hashlib.sha256(
        json.dumps(
            config,
            sort_keys=True,
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()

    assert config_hash(config) == expected


def test_configuration_order_independence():
    assert config_hash({"a": 1, "b": 2}) == config_hash(
        {"b": 2, "a": 1}
    )


@pytest.mark.parametrize(
    "config",
    [
        None,
        [],
        True,
        "invalid",
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": float("-inf")},
        {"nested": {"value": float("nan")}},
        {"values": [float("nan")]},
        {1: "invalid"},
        {"value": {1, 2}},
        {"value": (1, 2)},
        {"value": b"invalid"},
        {"value": object()},
    ],
)
def test_invalid_configuration_rejected(config):
    with pytest.raises(ConfigurationError):
        config_hash(config)


def test_circular_configuration_rejected():
    config = {}
    config["self"] = config

    with pytest.raises(ConfigurationError):
        config_hash(config)


def test_file_hash_after_modification(tmp_path):
    path = tmp_path / "green.tif"

    for content in (b"green", b"GREEN", b""):
        path.write_bytes(content)

        assert _file_sha256(path) == hashlib.sha256(
            content
        ).hexdigest()

        fingerprint = _file_fingerprint(path)

        assert fingerprint["status"] == "PASS"
        assert fingerprint["size_bytes"] == len(content)
        assert fingerprint["file_count"] == 1


def verified_directory(path):
    result = _file_fingerprint(path)

    assert result["status"] == "PASS", result
    assert isinstance(result["sha256"], str)
    assert len(result["sha256"]) == 64

    return result


def test_directory_content_changes(tmp_path):
    scene = tmp_path / "scene"
    scene.mkdir()

    green = scene / "green.tif"
    red = scene / "red.tif"

    green.write_bytes(b"green")
    red.write_bytes(b"red")

    original = verified_directory(scene)

    green.write_bytes(b"GREEN")

    modified = verified_directory(scene)

    assert modified["sha256"] != original["sha256"]
    assert modified["size_bytes"] == original["size_bytes"]

    nir = scene / "nir.tif"
    nir.write_bytes(b"nir")

    added = verified_directory(scene)

    assert added["sha256"] != modified["sha256"]
    assert added["file_count"] == 3

    nir.unlink()

    assert verified_directory(scene)["sha256"] == modified["sha256"]

    green.rename(scene / "renamed.tif")

    assert verified_directory(scene)["sha256"] != modified["sha256"]


def test_directory_hash_deterministic(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"

    first.mkdir()
    second.mkdir()

    (first / "a.txt").write_bytes(b"a")
    (first / "b.txt").write_bytes(b"b")

    (second / "b.txt").write_bytes(b"b")
    (second / "a.txt").write_bytes(b"a")

    assert (
        verified_directory(first)["sha256"]
        == verified_directory(second)["sha256"]
    )


def test_missing_and_empty_inputs(tmp_path):
    missing = tmp_path / "missing"

    assert _file_sha256(missing) == ""
    assert _file_fingerprint(missing)["status"] == "NOT_AVAILABLE"
    assert _file_fingerprint(tmp_path)["status"] == "NOT_AVAILABLE"


@pytest.mark.parametrize(
    "path",
    [
        "../outside.json",
        "nested/../../outside.json",
        r"..\outside.json",
        "/absolute.json",
        r"C:\outside.json",
        r"\\server\share\file.json",
        "file.json:alternate",
        "",
    ],
)
def test_unsafe_output_paths(tmp_path, path):
    assert _output_entry(tmp_path, path)["status"] == "FAIL"


def test_valid_output_checksum(tmp_path):
    path = tmp_path / "result.json"
    path.write_bytes(b"{}")

    result = _output_entry(tmp_path, "result.json")

    assert result["status"] == "PASS"
    assert result["size_bytes"] == 2
    assert result["sha256"] == hashlib.sha256(b"{}").hexdigest()


def test_missing_output_rejected(tmp_path):
    assert _output_entry(
        tmp_path, "missing.json"
    )["status"] == "FAIL"


def test_symbolic_links_rejected(tmp_path):
    target = tmp_path / "target.txt"
    target.write_bytes(b"data")

    link = tmp_path / "link.txt"

    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("Symbolic-link creation unavailable.")

    assert _file_sha256(link) == ""
    assert _file_fingerprint(link)["status"] == "FAIL"
    assert _output_entry(tmp_path, "link.txt")["status"] == "FAIL"
    assert _file_fingerprint(tmp_path)["status"] == "FAIL"


def test_environment_hash_reconciliation():
    result = collect_environment_provenance()

    payload = {
        key: value
        for key, value in result.items()
        if key not in (
            "environment_hash",
            "reproducibility_verified",
        )
    }

    assert config_hash(payload) == result["environment_hash"]
    assert result["reproducibility_verified"] is False

    json.dumps(result, allow_nan=False)


@pytest.fixture
def provenance():
    config = {"model": {"channels": 5}}

    digest = "a" * 64
    commit = "b" * 40

    environment = {
        "python_version": "3.14.6",
        "dependencies": {"numpy": "2.5.4"},
        "git_commit": commit,
        "git_working_tree_dirty": False,
    }

    environment["environment_hash"] = config_hash(environment)
    environment["reproducibility_verified"] = False

    fingerprint = {
        "status": "PASS",
        "exists": True,
        "sha256": digest,
        "kind": "directory",
        "file_count": 4,
        "size_bytes": 100,
        "fingerprint_version": 1,
    }

    metadata = {
        "run_id": "test_run",
        "configuration_hash": config_hash(config),
        "model_sha256": digest,
        "input_t1": deepcopy(fingerprint),
        "input_t2": deepcopy(fingerprint),
        "git_commit": commit,
        "environment_provenance": environment,
    }

    manifest = {
        "run_id": "test_run",
        "total_outputs": 1,
        "outputs": [
            {
                "path": "result.json",
                "status": "PASS",
                "sha256": digest,
                "size_bytes": 100,
            }
        ],
    }

    return config, metadata, manifest


def evaluate(provenance, expected):
    config, metadata, manifest = provenance

    engine = QualityAssuranceEngine(config)

    result = engine.run_reproducibility_qa(
        metadata, manifest, config
    )

    assert result["status"] == expected
    assert result["reproducibility_verified"] is False

    assert (
        engine.generate_qa_report()["categories"]["reproducibility"]
        == expected
    )


def test_valid_provenance(provenance):
    evaluate(provenance, "PASS")


@pytest.mark.parametrize(
    "key,value,expected",
    [
        ("configuration_hash", None, "FAIL"),
        ("configuration_hash", "a" * 64, "FAIL"),
        ("model_sha256", None, "FAIL"),
        ("model_sha256", "invalid", "FAIL"),
        ("input_t1", None, "FAIL"),
        ("input_t2", None, "FAIL"),
        ("input_t1", {"status": "FAIL"}, "FAIL"),
        ("environment_provenance", None, "NOT_AVAILABLE"),
        ("environment_provenance", [], "FAIL"),
        ("git_commit", None, "NOT_AVAILABLE"),
        ("git_commit", "c" * 40, "FAIL"),
    ],
)
def test_invalid_metadata(provenance, key, value, expected):
    provenance[1][key] = value

    evaluate(provenance, expected)


@pytest.mark.parametrize(
    "key,value",
    [
        ("total_outputs", 2),
        ("run_id", "wrong_run"),
        ("outputs", []),
    ],
)
def test_invalid_manifest(provenance, key, value):
    provenance[2][key] = value

    evaluate(provenance, "FAIL")


@pytest.mark.parametrize(
    "key,value",
    [
        ("status", "FAIL"),
        ("sha256", "invalid"),
        ("size_bytes", -1),
        ("path", "../outside.json"),
        ("path", "/absolute.json"),
        ("path", r"C:\outside.json"),
        ("path", "file.json:alternate"),
        ("path", ""),
    ],
)
def test_invalid_manifest_output(provenance, key, value):
    provenance[2]["outputs"][0][key] = value

    evaluate(provenance, "FAIL")


def test_duplicate_manifest_output(provenance):
    manifest = provenance[2]

    manifest["outputs"].append(
        deepcopy(manifest["outputs"][0])
    )

    manifest["total_outputs"] = 2

    evaluate(provenance, "FAIL")


def test_dirty_working_tree_warning(provenance):
    environment = provenance[1]["environment_provenance"]

    environment["git_working_tree_dirty"] = True

    payload = {
        key: value
        for key, value in environment.items()
        if key not in (
            "environment_hash",
            "reproducibility_verified",
        )
    }

    environment["environment_hash"] = config_hash(payload)

    evaluate(provenance, "WARN")


def test_single_date_provenance(provenance):
    metadata = provenance[1]

    metadata["input_scene"] = metadata.pop("input_t1")
    metadata.pop("input_t2")

    evaluate(provenance, "PASS")


def test_missing_manifest(provenance):
    config, metadata, _ = provenance

    evaluate((config, metadata, None), "NOT_AVAILABLE")
