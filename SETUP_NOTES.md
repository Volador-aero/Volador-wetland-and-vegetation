# Environment Setup and Validation — Initial Feature

## Purpose

This is a self-contained development-environment improvement for the **GIS Wetland Degradation & Depletion Engine**. It adds three repository-root files only:

| File | Responsibility |
| --- | --- |
| `environment.py` | Python virtual environment bootstrap, configuration-path migration, dependency, raster and ONNX validation, project readiness checks, a runner for the existing tests, and embedded regression tests |
| `requirements.txt` | Windows/Python 3.11 dependency baseline for the FastAPI/GIS runtime and existing Python tests |
| `SETUP_NOTES.md` | Setup instructions, testing, assumptions, limits and PR scope |

No existing `backend/` Python source file needs to be edited for this initial improvement.

## What's new in this update

This update extends the same three files. No existing `backend/` or `frontend/` source file is changed.

### New commands

Run these with the project's virtual environment, for example `.\.venv\Scripts\python.exe environment.py --readiness`.

| Command | What it does |
| --- | --- |
| `--readiness` | Read-only. Reports **READY** or **NOT READY** for each saved project in vegetation and wetland mode and names exactly what is missing or wrong. |
| `--readiness --project ID` | Same, for one project. |
| `--readiness --strict` | Exit code 1 if any project is not ready (useful in scripts). |
| `--run-tests` | Runs the existing `backend/tests` suite with this interpreter and prints a compact result with the names of failing tests. Add `--include-engine` to also run the `backend/Engine_module` tests and `--test-timeout N` to change the per-suite limit (default 1200 s). |
| `--verify` | One command: `--check --deep-model`, then `--readiness`, then `--run-tests`, followed by a combined summary. |
| `--setup --recreate` | Deletes and rebuilds `.venv`. Required when the existing `.venv` was built with an unsupported Python. |

### Project readiness rules

The rules mirror the existing dataset validation tests (`test_dataset_validation_modes.py`). **Vegetation** mode needs the four bands at T1. **Wetland** mode needs the four bands at both T1 and T2. The mode checked is the project's `analysis_mode`; when it is not set, both modes are shown. The wetland boundary is optional in both modes.

| Finding | Effect |
| --- | --- |
| A band file is missing (`scene_R`, `scene_G`, `scene_NIR`, `scene_RE`) | Blocks every mode that needs that date |
| A band file is empty, corrupt, unreadable, has no pixels, or contains only NoData | Blocks |
| Bands of one date have different dimensions | Blocks (the channels cannot be stacked) |
| A band has no CRS or no geotransform | Note only. The existing identical-input test treats an undefined CRS as valid drone imagery |
| T1 and T2 have different dimensions or CRS | Note only. The engine has its own alignment step |
| Wetland boundary missing, unreadable or without a polygon | Note only. The boundary is optional |
| An upload folder exists with no matching project in `projects.json` | Warning |

The check only reads files. It does not call the backend's `/datasets/validate` endpoint on purpose: according to the existing tests that endpoint changes the project status to `STAGED` and falls back to the sample scene when nothing was uploaded, which would hide a missing dataset.

### Edge cases now handled

| Case | Behaviour |
| --- | --- |
| Band files named `scene_R.TIF` or `scene_R.tiff` | Found (case-insensitive) |
| Empty or whitespace-only `projects.json` | Clear FAIL message; the file is not touched |
| Repository path with spaces (for example `C:\Users\Anjan P Manoj\...`) | Covered by a self-test |
| `.venv` created with an unsupported Python, or whose interpreter is missing | `--setup` stops before changing any file and says to use `--setup --recreate`; the folder is never deleted unless it contains `pyvenv.cfg` |
| Unsupported Python running `--setup` | Message names the supported range and shows the `py -3.11` command |
| `pip install` fails (offline, proxy, no wheel) | Message with the pip exit code and what to check |
| Non-UTF-8 pytest output on Windows | Output is decoded safely |
| Tests hang | Per-suite timeout |
| `.pytest_cache` clutter | Not created by `--run-tests` |

### Verification

- `--self-test` now contains 30 tests (the 11 from the first version plus 19 new ones covering the rules and edge cases above).
- The new commands were also run end to end in a Linux sandbox against a synthetic repository (paths with spaces, six projects in different states, a corrupt project, an orphan folder, a stale `.venv`). `--readiness` left `projects.json` byte-for-byte unchanged.
- Windows results for this update are listed in the pull request description.

### Notes on the existing tests

- `--run-tests` runs the real backend tests. They create and delete projects in the real `backend/storage`, so run them on a development copy. The end-to-end test can take a minute.
- `backend/tests/test_fastapi_server.py::test_real_tiff_upload_and_validation` reads files from `C:\Users\HP\Desktop` and returns early when they are missing, so it reports as passed on any other machine without testing anything. It is not changed in this PR.

### Assumptions to confirm with the team

1. The project record key is `analysis_mode` with the values `vegetation` and `wetland`.
2. The readiness rules above match what the backend accepts, especially "no CRS is allowed" and "T1/T2 mismatch is not blocking".

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
- Readiness is a file-level check. It does not prove that a run will succeed, and it does not replace the backend's own validation.
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
- [ ] `--readiness` names the missing bands or problems for each saved project and leaves `projects.json` unchanged.
- [ ] `--run-tests` runs the existing backend tests and reports passing and failing tests.
- [ ] `--verify` runs the check, readiness and tests in one command.
- [ ] `--setup` stops cleanly on a stale `.venv` and rebuilds it with `--recreate`.
- [ ] Submit a clean feature branch containing only the three intended files.
