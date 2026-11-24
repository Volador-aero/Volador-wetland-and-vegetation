"""Kill a real sample-scene worker, then resume its run in a separate process."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1] / "Engine_module"
sys.path.insert(0, str(ENGINE / "src"))


def worker(root, mode, stop, resume):
    from gis_engine.core.pipeline import ProductionPipeline
    from gis_engine.core.checkpoints import fingerprint
    cfg = root / "config.json"
    if not cfg.exists():
        data = json.loads((ENGINE / "configs/config.json").read_text())
        data["analysis_mode"] = mode
        cfg.write_text(json.dumps(data))
    scene = ENGINE / "data/sample_scene"
    args = dict(t1_scene=str(scene / "t1"), t1_date="2023-08-14",
                t2_scene=str(scene / "t2") if mode == "wetland" else None,
                t2_date="2023-08-28" if mode == "wetland" else None,
                model_path=str(ENGINE / "models/best_finetuned_model.onnx"),
                wetland_boundary=str(scene / "wetland_boundary.geojson"), config_path=str(cfg))

    def event(stage):
        with (root / "events").open("a") as stream:
            stream.write(stage + "\n")
        if stage == stop:
            os._exit(77)

    result = ProductionPipeline().run_analysis(**args, run_id="sample", output_base_dir=root,
                  checkpoint_signature=None if stop == "baseline" else fingerprint(args),
                  resume=resume, stage_callback=event)
    if result.status == "FAILED":
        raise RuntimeError(result.errors)


class CheckpointTests(unittest.TestCase):
    def test_damaged_stage_rolls_back_and_preserves_partial_artifacts(self):
        from gis_engine.core.checkpoints import Checkpoints, contained
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = Checkpoints(root, "wetland", "same-input")
            store.run("INPUT_VALIDATION", lambda: None, {}, [])
            (root / "alignment").mkdir()
            output = root / "alignment/grid.json"
            def align():
                output.write_text('{"width": 100}')
                return 100
            store.run("ALIGNMENT", align, {}, [])
            output.write_text("partial")
            reopened = Checkpoints(root, "wetland", "same-input", resume=True)
            self.assertEqual(list(reopened.records), ["INPUT_VALIDATION"])
            reopened.prepare_resume()
            self.assertFalse(output.exists())
            self.assertEqual(len(list(root.glob("checkpoints/partial-*/alignment/grid.json"))), 1)
            with self.assertRaises(ValueError):
                Checkpoints(root, "wetland", "changed", resume=True)
            with self.assertRaises(ValueError):
                contained(root, "../outside.json")

    def test_real_restart(self):
        for mode in ("vegetation", "wetland"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                def launch(stop, resume):
                    return subprocess.run([sys.executable, __file__, temp, mode, stop, str(resume)],
                                          capture_output=True, timeout=240)
                stops = (["ALIGNMENT", "VEGETATION_INFERENCE", "METRICS_QA", "REPORTING"]
                         if mode == "wetland" else ["VEGETATION_INFERENCE", "REPORTING"])
                for index, stage in enumerate(stops):
                    interrupted = launch(stage, index > 0)
                    self.assertEqual(interrupted.returncode, 77, interrupted.stderr[-4000:])
                before = (root / "events").read_text().splitlines()
                resumed = launch("", True)
                self.assertEqual(resumed.returncode, 0, resumed.stderr[-4000:])
                self.assertEqual((root / "events").read_text().splitlines()[len(before):], ["REPORTING"])
                self.assertTrue((root / "sample/manifest.json").is_file())
                import numpy as np
                import rasterio
                clean = root / "clean"
                clean.mkdir()
                baseline = subprocess.run([sys.executable, __file__, str(clean), mode, "baseline", "False"],
                                          capture_output=True, timeout=240)
                self.assertEqual(baseline.returncode, 0, baseline.stderr[-4000:])
                for expected in (clean / "sample").rglob("*.tif"):
                    restored = root / "sample" / expected.relative_to(clean / "sample")
                    with rasterio.open(expected) as left, rasterio.open(restored) as right:
                        self.assertEqual(left.transform, right.transform)
                        self.assertEqual(left.crs, right.crs)
                        np.testing.assert_allclose(left.read(), right.read(), equal_nan=True)
                def result_fields(folder):
                    data = json.loads((folder / "sample/analysis_result.json").read_text())
                    return {k: data[k] for k in ("metrics", "vegetation", "uncertainty")}
                self.assertEqual(result_fields(root), result_fields(clean))


if __name__ == "__main__":
    if len(sys.argv) == 5:
        worker(Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4] == "True")
    else:
        unittest.main()
