# Stage-Level Checkpoint Recovery Implementation Plan

## Using checkpoint recovery

In PowerShell, set `$env:ENABLE_STAGE_RESUME = 'true'` in the backend terminal, then start the API with the repository's usual Uvicorn command. The setting applies to newly launched runs. Each storage directory supports one API process. Set `$env:CHECKPOINT_FULL_HASH = '1'` before a new run if every intermediate raster and array needs a byte-for-byte integrity check on recovery.

After an interruption, open the same project and its Analysis page. The backend checks the saved inputs, preparation, model, engine code and consecutive stage records. **Resume from checkpoint** continues the same run. **Retry Analysis** creates a new run. Older status-only runs support Retry.

The implementation uses an OS-managed storage lock and a per-process resume mutex. The OS releases ownership on process death. The checkpoint manifest is the commit point. A resumed worker moves uncommitted scientific outputs into a private partial-output folder and memory-maps saved NumPy arrays. Raster validation defaults to file size, modification time, a readable header and a sample read. Small JSON outputs are hashed. The input signature uses content hashes.

Run the backend regression checks from the repository root with a Python environment containing the engine dependencies:

```powershell
python -m pytest backend/tests -q
cd frontend
npm.cmd test -- src/features/analysis/AnalysisPage.test.tsx
npm.cmd run build
```

In the default validation mode, recovery costs include hashing the input and model bytes, checking each committed artifact, reloading intermediate data used by the remaining stages, and executing those stages. Writing array checkpoints also costs time and disk space proportional to the bytes saved. The constant-time run index accelerates status lookup only.

## 1. Objective

Extend durable run-state recovery so an analysis interrupted by a backend crash or restart can continue from its last safely completed pipeline stage.

Python memory, the original background thread, open Rasterio handles, and a model call that stopped halfway cannot be restored after the process dies. The solution is to save validated intermediate outputs and checkpoints, then start a new worker that reconstructs its context and continues with the next incomplete stage.

The first version provides stage-level recovery, not tile-level inference recovery.

## 2. Current Codebase and Gap

The existing foundation includes:

- `RunWorker`, `run_state.json`, restart detection, retry, and indexed lookup in `backend/app/services.py`.
- Launch, status, and retry routes in `backend/app/routers/runs.py`.
- `ProductionPipeline.run_analysis()` in `backend/Engine_module/src/gis_engine/core/pipeline.py`.
- Status polling in `frontend/src/context/ProjectContext.tsx`.
- Progress and interrupted-run UI in `frontend/src/features/analysis/AnalysisPage.tsx`.

The main limitation is that `RunWorker._execute_pipeline()` displays several stages but then calls `analyze_wetland(...)` as one long operation. `ProductionPipeline.run_analysis()` keeps important aligned rasters, arrays, masks, model results, and metrics in local variables. A new worker cannot recreate those values from `run_state.json` alone.

True recovery therefore requires the engine to expose stage boundaries with explicit disk inputs, outputs, validators, and loaders.

## 3. Scope

### Included

- Durable, atomic stage checkpoints.
- Validation of intermediate output before reuse.
- Input, AOI, configuration, model, and pipeline-version fingerprints.
- A new worker that resumes the same logical run.
- `RESUMING` status and Resume API.
- Resume controls in Pipeline Execution Progress.
- Duplicate-resume protection.
- Backend, engine, API, frontend, and process-restart tests.

### Excluded from the first version

- Recovering destroyed Python objects or the original thread.
- Continuing from an exact Python instruction.
- Continuing an ONNX call halfway through execution.
- Tile-level checkpoints.
- Resuming with changed or incompatible inputs.
- Distributed recovery across machines.

## 4. Recovery Flow

```text
Worker completes stages and commits checkpoints
                     |
                     v
Backend crashes during a later stage
                     |
                     v
Backend restarts and reads durable metadata
                     |
                     v
Run becomes INTERRUPTED; checkpoints are validated
                     |
                     v
UI offers Resume when resume_available=true
                     |
                     v
New worker loads the last completed checkpoint
                     |
                     v
Pipeline continues with the next incomplete stage
```

If a crash occurs during a stage, that stage runs again from its beginning. A stage is skipped only after its complete checkpoint has been atomically committed.

## 5. Recoverable Stage Design

Use a small number of reliable boundaries rather than checkpointing every internal function.

| Stage | Current work | Required reusable outputs |
|---|---|---|
| `INPUT_VALIDATION` | Configuration, mode, input and date validation | Config snapshot, input manifest and fingerprints |
| `ALIGNMENT` | Raster loading, reference grid, spatial alignment and analysis masks | Aligned T1/T2, masks and reference-grid JSON |
| `VEGETATION_INFERENCE` | T1/T2 model execution | Health maps, confidence, required probabilities/logits and metadata |
| `CHANGE_ANALYSIS` | Transition, CVA, degradation and recovery | Transition/delta/CVA and degradation/recovery rasters |
| `WETLAND_ANALYSIS` | Wetland context, hydrology and evidence fusion | Wetland, NDWI, water and fusion outputs |
| `METRICS_QA` | Metrics and quality assurance | Metrics JSON/CSV and QA report |
| `REPORTING` | Result assembly, provenance, reports and manifest | Reports, metadata, manifest and final result |

Vegetation-only mode records `WETLAND_ANALYSIS` as `SKIPPED`. Stage order must be selected from the configured analysis mode.

## 6. Storage Layout

```text
backend/storage/projects/<project_id>/runs/<run_id>/
├── run_state.json
├── checkpoint_manifest.json
├── checkpoints/
│   ├── input_validation.json
│   ├── alignment.json
│   ├── vegetation_inference.json
│   ├── change_analysis.json
│   ├── wetland_analysis.json
│   ├── metrics_qa.json
│   └── reporting.json
├── alignment/
├── vegetation/
├── change/
├── degradation/
├── wetland/
├── hydrology/
├── fusion/
├── metrics/
├── validation/
└── reports/
```

All checkpoint paths must be relative to the run directory. Reject any path that resolves outside that directory.

## 7. Checkpoint Contracts

Example `checkpoint_manifest.json`:

```json
{
  "schema_version": 1,
  "run_id": "run_20261120_112629_265438",
  "project_id": "prj_1790921535755",
  "analysis_mode": "wetland",
  "pipeline_version": "1",
  "input_fingerprint": "sha256:...",
  "config_fingerprint": "sha256:...",
  "model_fingerprint": "sha256:...",
  "last_completed_stage": "ALIGNMENT",
  "next_stage": "VEGETATION_INFERENCE",
  "completed_stages": ["INPUT_VALIDATION", "ALIGNMENT"],
  "skipped_stages": [],
  "revision": 2,
  "updated_at": "2026-11-22T10:30:00+05:30"
}
```

Example `checkpoints/alignment.json`:

```json
{
  "schema_version": 1,
  "stage": "ALIGNMENT",
  "status": "COMPLETED",
  "inputs_fingerprint": "sha256:...",
  "completed_at": "2026-11-22T10:30:00+05:30",
  "outputs": [
    {
      "path": "alignment/effective_analysis_mask.tif",
      "type": "geotiff",
      "size_bytes": 1048576,
      "sha256": "..."
    },
    {
      "path": "alignment/reference_grid.json",
      "type": "json",
      "size_bytes": 420,
      "sha256": "..."
    }
  ],
  "loader": "load_alignment_checkpoint"
}
```

Extend `run_state.json` with:

```json
{
  "status": "INTERRUPTED",
  "last_completed_stage": "ALIGNMENT",
  "next_stage": "VEGETATION_INFERENCE",
  "resume_available": true,
  "resume_blocked_reason": null,
  "resume_count": 0,
  "attempt_id": "attempt_1",
  "worker_instance_id": null,
  "lease_expires_at": null
}
```

The backend calculates `resume_available`; the frontend must not guess it.

## 8. Engine Refactoring

### 8.1 Serializable pipeline context

Add a context type to the engine:

```python
@dataclass
class PipelineContext:
    run_id: str
    run_dir: Path
    analysis_mode: str
    config: dict[str, Any]
    input_manifest: dict[str, Any]
    artifacts: dict[str, str]
    warnings: list[str]
    outputs: dict[str, str]
```

Do not serialize live Rasterio objects or ONNX sessions. Store their files and reopen them in a stage loader.

### 8.2 Extract stage runners

Refactor `ProductionPipeline.run_analysis()` into methods such as:

```python
def run_input_validation(ctx: PipelineContext) -> StageResult: ...
def run_alignment(ctx: PipelineContext) -> StageResult: ...
def run_vegetation_inference(ctx: PipelineContext) -> StageResult: ...
def run_change_analysis(ctx: PipelineContext) -> StageResult: ...
def run_wetland_analysis(ctx: PipelineContext) -> StageResult: ...
def run_metrics_qa(ctx: PipelineContext) -> StageResult: ...
def run_reporting(ctx: PipelineContext) -> StageResult: ...
```

Every stage needs explicit input artifacts, output artifacts, output validation, and a loader for the next stage. It must not depend on an undocumented local variable left by an earlier stage.

### 8.3 Rehydrate values from artifacts

- Reopen aligned bands using Rasterio instead of expecting `aligned_pair` in memory.
- Reconstruct `ReferenceGrid` from JSON.
- Reload health, confidence, transition, CVA, degradation, wetland and mask rasters.
- Rebuild small result objects from JSON metadata when later code needs them.
- Keep large arrays on disk and load them only for the stage that requires them.

### 8.4 Stage orchestrator

```python
for stage in stage_plan:
    if checkpoint_store.is_valid(stage, fingerprints):
        context = stage.load_checkpoint(context)
        continue

    checkpoint_store.invalidate_from(stage)
    result = stage.run(context)
    stage.validate(result)
    checkpoint_store.commit(stage, result)
    context = stage.apply(result, context)
```

If a checkpoint is invalid, invalidate it and all later checkpoints because downstream results may depend on it.

## 9. Safe Checkpoint Commit

Use this order:

1. Mark the stage running in `run_state.json`.
2. Write outputs to a temporary stage directory where practical.
3. Flush and close all output files.
4. Validate required files, JSON structure and GeoTIFF readability.
5. Record metadata and optional checksums.
6. Atomically promote temporary output to final output.
7. Atomically write the stage checkpoint.
8. Atomically update the checkpoint manifest.
9. Update `run_state.json` with the last and next stages.

A file merely existing does not prove completion. If the backend stops before the checkpoint commit, the stage must run again.

For large rasters, default validation should check path, size, CRS, dimensions, transform, band count, dtype and nodata. Full SHA-256 hashing should be configurable because it reads the entire raster. Small JSON/configuration files should use full checksums.

## 10. Compatibility Fingerprints

Checkpoint reuse requires matching fingerprints for:

- T1 dataset files.
- T2 files when required.
- AOI/wetland boundary and manual exclusions.
- Analysis mode and observation dates.
- Effective configuration snapshot.
- ONNX model file.
- Pipeline and checkpoint schema versions.

Rules:

```text
All fingerprints match
    -> Resume is allowed

Input, AOI, config or model changed
    -> Resume is blocked; start a new run

Checkpoint output is missing/corrupt
    -> Fall back to the previous valid stage
```

Never mix old intermediate results with changed scientific inputs.

## 11. Run Directory Changes

`_prepare_run_directory()` currently accepts only a new reservation containing `run_state.json`. Add an explicit engine option:

```python
run_analysis(..., resume=False)
```

- `resume=False`: retain current protection and reject unexpected existing output.
- `resume=True`: require an existing matching state, checkpoint manifest and valid resume lease.
- Never infer Resume simply because output files exist.
- Never delete a previous run directory to make Resume work.

## 12. Backend Service Implementation

Create `backend/app/checkpoints.py`, or keep the first version near `RunWorker` if project conventions require it.

### `CheckpointStore`

- Atomically read/write the manifest and stage records.
- Validate path containment and artifacts.
- Find the last valid stage.
- Invalidate one stage and all later stages.
- Return a clear blocked reason.

### `RunWorker.inspect_resume(run_id)`

Returns resume availability, last completed stage, next stage, and any blocked reason.

### `RunWorker.resume_run(run_id)`

1. Locate state through `run_index`.
2. Require `INTERRUPTED` or an explicitly recoverable `FAILED` state.
3. Acquire a per-run resume lock.
4. Confirm no live worker lease exists.
5. Validate fingerprints and checkpoints.
6. Increment `resume_count` and create a new `attempt_id`.
7. Persist `RESUMING`.
8. Submit a new worker with the same `run_id` and directory.
9. Continue from the next incomplete stage.

Resume retains the same run ID because it continues the same output package. Existing Retry remains the safe fallback and creates a new run ID.

## 13. Duplicate-Resume Protection

Persist a worker lease:

```json
{
  "worker_instance_id": "api-process-123:thread-7",
  "lease_expires_at": "2026-11-22T10:35:00+05:30",
  "attempt_id": "attempt_2"
}
```

Resume is permitted only when the old lease expired, no in-process worker owns the run, and the caller holds the resume lock. Renew the lease using a lightweight heartbeat or at bounded stage transitions.

For the first version, document and enforce a single Uvicorn worker. Multiple Uvicorn processes require a cross-process file lock or transactional database; an ordinary Python lock is insufficient.

## 14. API Changes

Add to `backend/app/routers/runs.py`:

```text
GET  /api/runs/{run_id}/resume-check
POST /api/runs/{run_id}/resume
```

Successful Resume returns `202 Accepted` with the same run ID, `RESUMING`, last/next stages and updated resume count.

Structured conflict codes should include:

- `RUN_NOT_INTERRUPTED`
- `RUN_ALREADY_ACTIVE`
- `RESUME_ALREADY_REQUESTED`
- `CHECKPOINT_NOT_FOUND`
- `CHECKPOINT_CORRUPT`
- `CHECKPOINT_INCOMPATIBLE`
- `STATE_STORAGE_UNAVAILABLE`

Extend `RunStatusResponse` in `backend/app/schemas.py` with optional resume fields so existing clients remain compatible.

## 15. Frontend Changes

### `frontend/src/api/client.ts`

- Add `RESUMING` to `RunStatus.status`.
- Add checkpoint/resume fields.
- Add `checkRunResume()` and `resumeRun()`.
- Preserve structured backend errors.

### `frontend/src/context/ProjectContext.tsx`

- Add `resumeActiveRun()`.
- Poll while `RESUMING` and during processing stages.
- Prevent simultaneous Resume and Retry submissions.
- Ignore a late response after the active project changes.
- Reload outputs after completion.

### `frontend/src/features/analysis/AnalysisPage.tsx`

For a resumable interrupted run, show:

```text
Analysis interrupted
Last completed stage: Spatial Alignment
Next stage: Vegetation Inference

[Resume from checkpoint] [Restart as new run]
```

If Resume is blocked, show the backend reason and offer only a clean Restart. Explain that Resume uses validated intermediate output, while Restart creates a new run.

## 16. Edge Cases

| Situation | Required behaviour |
|---|---|
| Crash while calculating a stage | Rerun that stage |
| Crash while writing temporary output | Ignore/remove temporary output and rerun |
| Crash before checkpoint commit | Rerun the stage |
| Crash after checkpoint commit | Load it and start at the next stage |
| Truncated checkpoint JSON | Reject it and fall back to the previous valid stage |
| Missing/unreadable GeoTIFF | Invalidate its stage and all later stages |
| Input, AOI, config or model changed | Block Resume and require a new run |
| Two Resume clicks | Start at most one worker |
| Repeated backend restarts | Reuse the last valid checkpoint and increment resume count |
| Report output alone is corrupt | Resume from `REPORTING`, not inference |
| Vegetation mode | Mark the wetland stage skipped |
| Project changes during request | Do not display the old project's response |

## 17. Test Plan

### Unit tests

- Atomic checkpoint writes preserve the previous valid record after failure.
- Paths cannot escape the run directory.
- Missing, truncated and invalid checkpoints are rejected.
- GeoTIFF validation detects unreadable or incompatible output.
- Fingerprints change with input, AOI, config or model changes.
- Invalidating one stage invalidates later stages.
- Vegetation and wetland modes generate correct stage plans.

### Engine tests

- Each stage runs from documented inputs only.
- Every stage produces its declared artifacts.
- Each loader reconstructs the next stage's context.
- A resumed run produces the same scientific output as a clean run within defined numerical tolerances.
- A reporting-only Resume does not rerun inference.

### Real restart test

1. Launch the backend as a subprocess.
2. Start a run and wait for a checkpoint.
3. Terminate the backend during the next stage.
4. Restart the backend.
5. Verify `INTERRUPTED` and `resume_available=true`.
6. Call Resume.
7. Verify the same run ID and a new attempt ID.
8. Verify completed stages were loaded rather than rerun.
9. Verify complete final output and manifest.

Repeat during Alignment, Inference, Metrics, and Reporting.

### API and frontend tests

- Resume-check for resumable and blocked runs.
- Duplicate Resume creates one worker.
- Retry creates a new run ID; Resume preserves it.
- Resume button appears only when allowed and disables after one click.
- `RESUMING` continues polling.
- Blocked reason is displayed.
- Project switching rejects stale responses.

## 18. Acceptance Criteria

- Every stage has explicit inputs, outputs, validation and loading logic.
- Checkpoints are committed only after output validation.
- A real backend termination recovers the run as interrupted.
- Resume availability and blocked reasons are accurate.
- Resume keeps the run ID, increments the attempt, and starts a new worker.
- Completed stages are skipped; the interrupted stage restarts from its beginning.
- Changed inputs cannot reuse incompatible checkpoints.
- Duplicate requests cannot start duplicate workers.
- Retry continues to provide a clean new run.
- Resumed output passes the same validation as clean output.
- All backend, engine, API, frontend and process-restart tests pass.

## 19. Time Complexity

Let `S` be the number of stages, `F` the number of checkpoint artifacts, `B` the bytes fully hashed, and `C_remaining` the work after the last checkpoint.

| Operation | Complexity |
|---|---:|
| Find last checkpoint from manifest | `O(1)` |
| Validate stage order | `O(S)`; `S` is small and fixed |
| Validate artifact metadata | `O(F)` |
| Full checksum validation | `O(B)` |
| Write small checkpoint JSON | Effectively `O(1)` |
| Resume computation | `O(C_remaining)` |
| Clean restart | `O(C_total)` |

Scientific processing complexity does not increase. Checkpointing adds small JSON writes and artifact validation at stage boundaries. Metadata validation keeps overhead low; full raster checksums should be optional.

## 20. Effort Estimate

| Work | Estimate |
|---|---:|
| Stage dependency and artifact design | 1-2 days |
| Checkpoint store, validation and fingerprints | 2-3 days |
| Alignment/context refactor | 2-3 days |
| Inference/change refactor | 3-4 days |
| Wetland/metrics/QA/reporting refactor | 3-4 days |
| Resume service, locking and API | 2-3 days |
| Frontend integration | 1-2 days |
| Automated and real-restart tests | 3-5 days |
| Documentation and rollout | 1 day |
| **Total for one developer** | **18-27 working days** |

Recommended milestones:

1. Checkpoint storage, schemas, fingerprints and tests.
2. Resume late stages such as Reporting and Metrics first.
3. Refactor Alignment, Inference and Change Analysis.
4. Add Resume API and frontend.
5. Complete process-kill verification and rollout controls.

## 21. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Hidden in-memory stage dependencies | Explicit artifact contracts and loader tests |
| Partial output treated as complete | Validate and atomically commit checkpoint last |
| Inputs changed | Fingerprint all scientific inputs |
| Multiple workers resume one run | Lease, lock, attempt ID and single-worker deployment |
| Large checksum cost | Metadata validation by default; optional full hash |
| Scientific changes from refactoring | Compare clean and resumed outputs within tolerances |
| Increased disk usage | Document retained artifacts and later add safe retention |
| Old checkpoint incompatibility | Schema versioning and explicit migration/blocking |

## 22. Rollout Notes

1. Keep existing durable status and Retry behaviour working during development.
2. Put checkpoint recovery behind `ENABLE_STAGE_RESUME=false` initially.
3. Enable it first for sample datasets.
4. Initially use one Uvicorn worker.
5. Test vegetation and wetland modes independently.
6. Compare output manifests and scientific results between clean and resumed runs.
7. Show Resume only when the backend confirms validity.
8. Never delete interrupted evidence automatically when validation fails.

Suggested verification commands from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest backend\tests\test_durable_run_state.py -q
.\.venv\Scripts\python.exe -m pytest backend\tests\test_checkpoint_recovery.py -q
Set-Location frontend
npm.cmd run build
```

The checkpoint test file is created during implementation. Use the project's actual virtual-environment Python if it differs.

## 23. Definition of Done

The feature is not complete merely because a Resume button appears. It is complete when a backend process is deliberately terminated during real analysis, a new backend process validates stored checkpoints, a new worker continues from the correct stage, completed stages are demonstrably skipped, and final outputs match a clean run within accepted scientific tolerances.
