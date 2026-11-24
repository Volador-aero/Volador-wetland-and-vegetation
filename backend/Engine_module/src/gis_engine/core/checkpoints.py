"""File-backed stage boundaries shared by the API and the standalone engine."""
import hashlib
import json
import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from affine import Affine

from ..alignment.spatial import ReferenceGrid


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(arguments):
    """Bind the effective input bytes, including ONNX external data, to the run."""
    files = set()
    for key in ("t1_scene", "t2_scene", "config_path", "model_path", "wetland_boundary"):
        if arguments.get(key):
            p = Path(arguments[key])
            if p.is_dir():
                files.update(x for x in p.rglob("*") if x.is_file())
            else:
                files.add(p)
                if key == "model_path":
                    files.update(p.parent.glob("*.data"))
                if p.suffix.lower() == ".shp":
                    files.update(p.parent.glob(p.stem + ".*"))
    evidence = [(str(p), digest(p)) for p in sorted(files)]
    # Bind compatibility to the installed scientific code, including local edits.
    source = Path(__file__).resolve().parents[1]
    evidence.extend((p.relative_to(source).as_posix(), digest(p)) for p in sorted(source.rglob("*.py")))
    return hashlib.sha256(json.dumps([arguments, evidence], sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    temp = path.with_name("." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("w", encoding="utf8") as stream:
            json.dump(value, stream, allow_nan=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def contained(root, relative):
    if Path(relative).is_absolute():
        raise ValueError("Expected a relative checkpoint path")
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("Checkpoint path outside run directory")
    return path


def stamp(path, *, checksum=False):
    stat = path.stat()
    values = [stat.st_size, stat.st_mtime_ns]
    if checksum:
        values.append(digest(path))
    return values


def validate_artifact(path):
    if path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(path) as ds:
            if not (ds.count and ds.width and ds.height):
                raise ValueError("Empty raster")
            ds.read(1, window=((0, 1), (0, 1)))
    elif path.suffix == ".json":
        json.loads(path.read_text(encoding="utf8"))
    elif path.suffix == ".npy":
        np.load(path, mmap_mode="r", allow_pickle=False)


PLANS = {
    "vegetation": ["INPUT_VALIDATION", "PREPROCESSING", "VEGETATION_INFERENCE", "METRICS_QA", "REPORTING"],
    "wetland": ["INPUT_VALIDATION", "ALIGNMENT", "VEGETATION_INFERENCE", "CHANGE_ANALYSIS",
                "WETLAND_ANALYSIS", "METRICS_QA", "REPORTING"],
}


class Checkpoints:
    def __init__(self, root, mode, signature, *, resume=False, notify=None):
        self.root = Path(root).resolve()
        self.path = self.root / "checkpoint_manifest.json"
        self.plan = PLANS[mode]
        self.notify = notify
        self.known = {}
        self.records = {}
        self.manifest = {"version": 1, "run_id": self.root.name, "signature": signature,
                         "mode": mode, "stages": []}
        if resume:
            saved = json.loads(self.path.read_text(encoding="utf8"))
            if not isinstance(saved, dict) or not isinstance(saved.get("stages"), list):
                raise ValueError("Unreadable checkpoint manifest")
            if any(saved[k] != self.manifest[k] for k in ("version", "run_id", "signature", "mode")):
                raise ValueError("Inputs, configuration, model or pipeline version changed")
            self.manifest = saved
            self._validate_prefix()
        else:
            atomic_json(self.path, self.manifest)

    def _validate_prefix(self):
        valid = []
        for stage, entry in zip(self.plan, self.manifest["stages"]):
            try:
                if entry["stage"] != stage:
                    break
                p = contained(self.root, f"checkpoints/{stage}.json")
                if digest(p) != entry["sha256"]:
                    break
                record = json.loads(p.read_text(encoding="utf8"))
                for name, expected in record["files"].items():
                    artifact = contained(self.root, name)
                    if stamp(artifact, checksum=len(expected) == 3) != expected:
                        raise ValueError("Changed artifact")
                    validate_artifact(artifact)
                self.known.update(record["files"])
                self.records[stage] = record
                valid.append(entry)
            except (ValueError, OSError, KeyError, TypeError, EOFError):
                break
        self.manifest["stages"] = valid

    def prepare_resume(self):
        # The prefix is the commit boundary. Keep partial scientific outputs as
        # evidence outside the live output tree, including optional outputs.
        atomic_json(self.path, self.manifest)
        archive = self.root / "checkpoints" / ("partial-" + uuid.uuid4().hex)
        for name in self.artifacts():
            if name not in self.known:
                dest = archive / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(contained(self.root, name), dest)

    def artifacts(self):
        for folder in ("alignment", "exclusions", "vegetation", "change", "degradation",
                       "wetland", "hydrology", "fusion", "metrics", "validation", "reports"):
            for p in contained(self.root, folder).rglob("*"):
                if p.is_file():
                    yield p.relative_to(self.root).as_posix()
        for name in ("analysis_result.json", "manifest.json", "metadata.json"):
            if (self.root / name).is_file():
                yield name

    def run(self, stage, work, outputs, warnings):
        record = self.records.get(stage)
        if record:
            outputs.update(record["outputs"])
            warnings[:] = record["warnings"]
            return self._decode(record["value"])
        if self.notify:
            self.notify(stage)
        prior_outputs = set(outputs)
        value = work()
        for key in outputs.keys() - prior_outputs:
            if not contained(self.root, outputs[key]).is_file():
                raise ValueError(f"Stage {stage} did not produce {outputs[key]}")
        folder = contained(self.root, "checkpoints/" + uuid.uuid4().hex)
        folder.mkdir(parents=True)
        files = {}
        encoded = self._encode(value, folder, files)
        for name in self.artifacts():
            path = contained(self.root, name)
            current = stamp(path)
            if name not in self.known:
                validate_artifact(path)
                files[name] = self.artifact_stamp(path)
            elif self.known[name][:2] != current:
                raise ValueError(f"Stage {stage} modified committed output {name}")
        record = {"files": files, "value": encoded, "outputs": dict(outputs), "warnings": list(warnings)}
        target = contained(self.root, f"checkpoints/{stage}.json")
        atomic_json(target, record)
        self.manifest["stages"].append({"stage": stage, "sha256": digest(target)})
        atomic_json(self.path, self.manifest)
        self.known.update(files)
        return value

    @staticmethod
    def artifact_stamp(path):
        strong = path.suffix == ".json" or os.getenv("CHECKPOINT_FULL_HASH") == "1"
        return stamp(path, checksum=strong)

    def _encode(self, v, folder, files):
        if isinstance(v, np.ndarray):
            p = folder / f"{len(files)}.npy"
            np.save(p, v, allow_pickle=False)
            name = p.relative_to(self.root).as_posix()
            files[name] = self.artifact_stamp(p)
            return {"array": name}
        if isinstance(v, ReferenceGrid):
            return {"grid": v.to_dict()}
        if isinstance(v, SimpleNamespace):
            return {"fields": self._encode(vars(v), folder, files)}
        if isinstance(v, Path):
            return {"path": v.relative_to(self.root).as_posix()}
        if isinstance(v, dict):
            return {"items": [[self._encode(k, folder, files), self._encode(x, folder, files)] for k, x in v.items()]}
        if isinstance(v, (tuple, list)):
            return [self._encode(x, folder, files) for x in v]
        return v.item() if isinstance(v, np.generic) else v

    def _decode(self, v):
        if isinstance(v, list):
            return [self._decode(x) for x in v]
        if not isinstance(v, dict):
            return v
        if "array" in v:
            return np.load(contained(self.root, v["array"]), mmap_mode="r", allow_pickle=False)
        if "grid" in v:
            g = dict(v["grid"])
            g["transform"] = Affine(*g["transform"])
            g["crs"] = rasterio.crs.CRS.from_string(g["crs"]) if g["crs"] else None
            return ReferenceGrid(**g)
        if "fields" in v:
            return SimpleNamespace(**self._decode(v["fields"]))
        if "path" in v:
            return contained(self.root, v["path"])
        return {self._decode(k): self._decode(x) for k, x in v["items"]}


def fields(obj, *names):
    return SimpleNamespace(**{name: getattr(obj, name) for name in names})
