# Environment Setup and Validation — Initial Feature

## Purpose

This is a self-contained development-environment improvement for the **GIS Wetland Degradation & Depletion Engine**. It adds three repository-root files only:

| File | Responsibility |
| --- | --- |
| `environment.py` | Python virtual environment bootstrap, configuration-path migration, dependency, raster and ONNX validation, and embedded regression tests |
| `requirements.txt` | Windows/Python 3.11 dependency baseline for the FastAPI/GIS runtime and existing Python tests |
| `SETUP_NOTES.md` | Setup instructions, testing, assumptions, limits and PR scope |

No existing `backend/` Python source file needs to be edited for this initial improvement.

## Prerequisites

- **Python 3.11** recommended (the utility accepts Python 3.11–3.13, but the exact dependency versions were observed working with Python 3.11 on Windows).
- Git and Windows PowerShell; Node.js/npm only if also running the frontend.
- The original repository, preferably obtained with **`git clone`**, not a GitHub Download ZIP. ZIP archives lack the `.git` metadata required to create/push branches.
- Run all commands below from the repository root (where `README.md` and `environment.py` are located).

## One-time setup on a fresh clone

```powershell
py -3.11 environment.py --self-test
py -3.11 environment.py --setup
.\.venv\Scripts\python.exe environment.py --self-test
.\.venv\Scripts\python.exe environment.py --check --deep-model
```

`--setup` performs the following actions, in order:

1. Validates expected repository directories and Python version.
2. Loads `backend/storage/projects.json` **if it exists**, backs up the original before rewriting, and replaces old computer-specific `t1_dir` and `t2_dir` with paths under this clone's actual `backend/storage/projects/<id>/datasets/` directory. Known project-local boundary paths can be rebased; unrelated external paths are left alone. Empty upload directories are created as necessary. **No raster files are fabricated.**
3. Creates `.venv` if absent and installs the pinned `requirements.txt` packages.
4. Checks the existing `backend/Engine_module/models/best_finetuned_model.onnx` against the installed ONNX Runtime. If the file is specifically the repository's generated **single-Conv demonstration model** and uses an unsupported higher IR version, it creates a backup and changes **only that demo model** to compatible IR version 10, validating it through CPU inference before replacing. Unknown/trained models are never rewritten. The script does **not** run `backend/scripts/create_onnx_model.py`.

`--check` is read-only and verifies structure, installed versions, `pip check`, sample raster metadata (if present), saved-project data, ONNX structural and runtime load, and backend import. `--deep-model` additionally runs a tiny CPU inference smoke test.

### Optional standalone commands

```powershell
.\.venv\Scripts\python.exe environment.py --fix-paths
.\.venv\Scripts\python.exe environment.py --fix-model
.\.venv\Scripts\python.exe environment.py --check
```

You **do not** need to rerun `--setup` for every backend start. The `--fix-paths` and `--fix-model` commands are intended for explicit maintenance; rerunning them when already correct is safe. Do **not** rerun the original `create_onnx_model.py` unnecessarily, as it can regenerate a newer, incompatible ONNX IR file; if you deliberately regenerate the demo model, repeat `--fix-model` and `--check --deep-model`.

## Starting the application

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

In a separate terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

For the React frontend, run `npm install` and `npm run dev` **from `frontend/`**. Frontend dependencies remain in the existing `frontend/package.json` / `package-lock.json`.

## Existing regression tests

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_fastapi_server.py
.\.venv\Scripts\python.exe -m pytest backend/tests/test_e2e_full_integration.py
Push-Location .\backend\Engine_module
..\..\.venv\Scripts\python.exe -m pytest
Pop-Location
```

These commands exercise the original project test suites, not just the setup helper. They should be run as appropriate for the available data and project state; their complete results have **not** been claimed here.

## Validation performed while developing this initial feature

On the prior Windows project copy, the user reported:

- 9 setup self-tests passing before the ONNX-specific tests were added.
- Portable project-path migration backed up and corrected the stale `C:\Users\HP\...` locations, allowing dataset uploads to return HTTP 200.
- The generated ONNX demo file initially failed because its IR was 14 and the installed runtime supported up to 13. `--fix-model` successfully created a backup and repaired the known demo file to IR 10; `--check --deep-model` then loaded it and completed CPU smoke inference.
- Latest reported environment check: **43 PASS, 4 WARN, 0 FAIL**. Remaining warnings were incomplete saved-project datasets and the fact that the model is a demo, not a trained ResNet34.

Those are results from the **previous local working copy**, not a guarantee about every fresh clone. Re-run checks on the new clone before opening the PR.

## Expected warnings versus failures

- **FAIL**: unsupported Python, missing required packages or backend modules, invalid core layout, unreadable required sample raster, or a model that ONNX Runtime cannot load.
- **WARN**: missing/incomplete *user project* datasets, missing temporal scenes, outdated configured paths before migration, or the use of a demonstration model.
- Successful environment checks **do not** imply that incomplete scientific datasets are suitable for analysis. The backend must still reject invalid analysis inputs.

## Known limitations

- `--setup` changes **local generated configuration and the known demo ONNX model** where necessary. Both are backed up. No original backend `.py` source file is modified.
- A repaired ONNX demo file remains an untrained, simplified model. This is an environment-compatibility workaround and not evidence of production scientific accuracy.
- Some auxiliary scripts (`torch` / `segmentation_models_pytorch` model export, specialized validation scripts) may need additional optional packages; they are outside the supported backend/test baseline here.
- `ProjectStore` concurrency and simultaneous project writes are not addressed by this utility; no backend source change is included.
- A fresh repository may contain old project records without their underlying raster datasets. The tool fixes file-system paths, not missing source data.

## Pull request and Git hygiene

The intended PR changes **only** `environment.py`, `requirements.txt`, and `SETUP_NOTES.md` at the repository root. Never stage `.venv`, `backend/storage/projects.json`, `backend/storage/projects/`, generated results, raster inputs, model backups, or the repaired `best_finetuned_model.onnx` file. Even when these appear as local changes, use explicit file names with `git add`, **not `git add .`**.

Create the feature branch before adding the files; after local validation, inspect what is staged:

```powershell
git add -- environment.py requirements.txt SETUP_NOTES.md
git diff --cached --name-only
git diff --cached --check
git commit -m "feat: add portable GIS environment setup and validation"
```

The `--setup` process is expected to produce local files; these are **not** part of this PR. If the upstream repository is not writable by you, fork it and open a PR from your fork to `Volador-aero/Volador-wetland-and-vegetation:main`.

## Acceptance criteria

- [ ] Bootstrap `.venv` from one root requirements file on a clean clone.
- [ ] Repair stale absolute paths in `projects.json` with a backup and no fabricated rasters.
- [ ] Detect/repair only the known incompatible generated ONNX demo file and verify runtime load.
- [ ] Report meaningful PASS/WARN/FAIL outcomes and pass embedded self-tests.
- [ ] Start the FastAPI backend and verify `/api/health`.
- [ ] Submit a clean feature branch containing only the three intended files.
