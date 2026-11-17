# Durable Run-State Recovery

## Purpose

Durable run-state recovery keeps background GIS analysis status available after
a FastAPI restart, crash, or development reload. Active status remains cached in
the in-memory `RUN_STATES` dictionary for fast polling, while every run also has
a small authoritative record on disk.

The implementation is located in:

- `backend/app/services.py` - persistence, direct lookup, recovery and retry
- `backend/app/routers/runs.py` - status and retry endpoints
- `backend/app/schemas.py` - API response schema
- `frontend/src/api/client.ts` - interrupted status and retry request
- `frontend/src/context/ProjectContext.tsx` - polling and retry behavior
- `frontend/src/features/analysis/AnalysisPage.tsx` - interruption UI
- `backend/tests/test_durable_run_state.py` - focused tests

## Storage Layout

Each accepted run creates two small files:

```text
backend/storage/
|-- run_index/
|   `-- <run_id>.json
`-- projects/<project_id>/runs/<run_id>/
    `-- run_state.json
```

The index contains the direct path to the run directory. `run_state.json`
contains identifiers, lifecycle status, progress, processing step, timestamps,
message, error, output directory, configuration overrides and retry lineage.

The backend writes JSON to a temporary file, flushes it to disk and atomically
replaces the previous file with `os.replace`. This prevents readers from seeing
a partially written state record.

## Lifecycle

```text
QUEUED
  -> VALIDATING
  -> PREPROCESSING
  -> INFERENCE
  -> CHANGE_ANALYSIS
  -> REPORTING
  -> COMPLETED
```

Any active stage can become `FAILED`. When the backend finds a non-terminal
durable state after memory loss, it changes the status to `INTERRUPTED` and
preserves `last_confirmed_step`.

The backend writes state only when the lifecycle stage changes. It does not
write state for individual raster pixels, tiles, model batches or frontend
polling requests.

## Time Complexity

Time performance is the primary design constraint. Let:

- `B` be the size of one bounded run-state JSON record.
- `P` be the number of project directories.
- `R` be the total number of durable runs.
- `Rp` be the number of runs in one project.

| Operation | Formal complexity | Practical behavior |
|---|---:|---|
| Active status lookup | `O(1)` average | One `RUN_STATES` dictionary lookup |
| Locate a durable run | `O(1)` path lookup | One small index-file read |
| Read durable state | `O(B)` | Effectively constant for the bounded JSON record |
| Save a lifecycle transition | `O(B)` | One small atomic JSON replacement |
| Restart recovery for a new run | `O(B)` | One index read, one state read and at most one write |
| Retry setup | `O(B)` | Reads old state and creates a queued state |
| Legacy recovery without an index | `O(P)` plus metadata reads | Compatibility path for older runs only |
| Reset/delete project indexes | `O(Rp)` | Runs only during reset or deletion |

Because `B` is small and bounded, normal status lookup, persistence and restart
recovery are operationally `O(1)` per run. Their cost does not grow with raster
width, raster height, pixel count, model tensor size, map tiles or the total
number of indexed runs.

A successful run performs a fixed number of small writes: the initial `QUEUED`
record, the major processing transitions and the final terminal state. No state
write is performed inside raster processing or ONNX inference loops.

`flush`, `fsync` and `os.replace` have constant algorithmic complexity for the
bounded record. Their measured latency still depends on the local disk and
operating system.

### Space Complexity

Each run stores one bounded state record and one bounded index pointer:

```text
O(R * B), effectively O(R)
```

Storage grows with the number of runs, not with raster size.

## Implementation Plan and Status

### Phase 1 - Durable foundation

- Add atomic JSON read/write helpers.
- Create `run_state.json` before executor submission.
- Create `run_index/<run_id>.json` for direct lookup.
- Use microseconds in run IDs to prevent same-second collisions.

Status: implemented.

### Phase 2 - Worker integration

- Persist each meaningful processing transition.
- Persist `COMPLETED` only after successful pipeline completion.
- Persist `FAILED` with the original error and completion time.
- Keep `RUN_STATES` as the active polling cache.

Status: implemented.

### Phase 3 - Restart recovery

- Read new runs through the direct index.
- Treat `run_state.json` as authoritative after memory loss.
- Mark unfinished runs as `INTERRUPTED`.
- Preserve the last confirmed step.
- Retain the directory scan for legacy runs only.

Status: implemented.

### Phase 4 - Retry and frontend behavior

- Add `POST /api/runs/<run_id>/retry`.
- Allow only `FAILED` and `INTERRUPTED` runs to be retried.
- Create a new run ID with `retry_of` lineage.
- Preserve the original run folder.
- Stop polling interrupted runs and display Retry Analysis.

Status: implemented.

### Phase 5 - Follow-up validation

- Add a process-level integration test that interrupts and restarts the real API.
- Add optional timing telemetry if production latency measurements are required.
- Add schema migration only if the durable record changes later.

Status: focused tests implemented; process-level restart testing and telemetry
remain follow-up work.

## Setup

No database, queue or additional service is required. Use a Python environment
that contains the existing backend dependencies such as FastAPI, NumPy and
Rasterio.

From the project root, start the backend:

```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Start the frontend in a second terminal:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

The state and index directories are created automatically when the first run is
accepted.

## API

```text
POST /api/runs
GET  /api/runs/<run_id>
POST /api/runs/<run_id>/retry
```

Retry returns `202 Accepted` with a new run ID and saves
`retry_of: <original_run_id>`.

## Tests

The focused tests use Python's built-in `unittest`, temporary directories and
do not modify `backend/storage`.

```powershell
python -m unittest discover -s backend/tests -p "test_durable_run_state.py" -v
```

The suite verifies:

1. Queued state exists before background-worker submission.
2. An unfinished run becomes `INTERRUPTED` after memory loss.
3. A saved completed run remains completed without status guessing.
4. Retry creates a new run and preserves original-run lineage.

The same tests can be collected by `pytest` when it is installed.

## Operational Notes

- `RUN_STATES` remains the fastest active-status path.
- New-format recovery does not scan all projects.
- Legacy runs continue to use the previous fallback.
- Reset and deletion remove associated index files.
- Unreadable durable state is reported as a recovery failure.
- Retry never overwrites the original run folder.
