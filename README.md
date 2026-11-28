# GIS Wetland Degradation & Depletion Engine

Integrated geospatial analysis application for Sentinel-2 temporal vegetation degradation, 4x4 transition analysis, Change Vector Analysis (CVA), NDWI surface-water monitoring, and cartographic PDF report generation.

---

## 1. System Architecture

```
React 19 Frontend (Vite + OpenLayers + ECharts)
       │
       ▼  HTTP / REST API (JSON + Tile/Raster Stream)
FastAPI Application Layer (backend/app)
       │
       ├── Project Service & Dataset Manager (Local FS Storage)
       ├── Async Background Run Manager (Worker Thread Pool)
       └── Tile & Layer Renderer (Rasterio / GeoTIFF Server)
       │
       ▼
GIS Core Engine (backend/Engine_module/src/gis_engine)
       ├── Ingestion & Input Validation
       ├── Spatial & Temporal Alignment
       ├── Preprocessing & Index Computation (NDVI, GNDVI, NDRE, NDWI)
       ├── ONNX Inference (5-Channel U-Net / ResNet34)
       ├── Change Analysis (4×4 Transition, CVA, Degradation, Recovery)
       ├── Hydrological Evidence & Wetland Masking
       ├── Metrics & Area Calculations
       ├── QA Validation (Input QA, Temporal QA, Model QA)
       └── PDF Report Generator (ReportLab / Matplotlib Cartography)
```

---

## 2. Quick Start

### Step 1: Generate Sample Dataset (Optional Out-of-the-Box Setup)
```bash
python backend/scripts/generate_sample_scene.py
python backend/scripts/create_onnx_model.py
```

### Step 2: Start the Backend FastAPI Server
```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
The API server starts at `http://127.0.0.1:8000` (Swagger docs available at `http://127.0.0.1:8000/docs`).

### Step 3: Start the React Frontend Application
```bash
cd frontend
npm install
npm run dev
```
The web app opens at `http://localhost:5173`.

---

## 3. Usage Guide

### Create a project

1. Open `http://localhost:5173`.
2. Create a project from the project workflow.
3. Select **Vegetation** for a single-date analysis or **Wetland** for a
   T1-to-T2 temporal comparison.

### Prepare satellite data

Each scene needs separate GeoTIFF bands for:

- Green
- Red
- Red Edge
- NIR

Vegetation mode requires the T1 scene. Wetland mode requires both T1 and T2
scenes, plus valid observation dates. T1 and T2 rasters must use compatible
spatial grids. A wetland boundary or independent hydrological evidence is
optional, but its absence is displayed as a warning rather than proof of
physical wetland depletion.

### Prepare the analysis area

Use the Preparation page to:

- review the scene extent;
- include or exclude the analysis area;
- commit and lock the preparation state.

The Run Analysis button remains blocked until the preparation area is committed
and locked and every other required readiness check passes.

### Review analysis readiness

The Analysis page displays a readiness checklist before run launch:

- **PASS**: the prerequisite is satisfied.
- **WARNING**: the run can continue, but scientific or contextual evidence is
  limited.
- **BLOCKED**: the run cannot start until the prerequisite is corrected.

Readiness is refreshed after project changes, dataset validation, date edits,
preparation changes, analysis-mode changes, and observation-date setting
changes.

Backend launch validation is enforced independently of the checklist. Rejected
preflight requests do not create run IDs, persisted run state, output folders,
or background tasks.

### Configure observation dates

Use **Edit Observation Dates** to update T1 and T2 after uploading the
datasets. In wetland mode, T1 must precede T2. Equal or missing dates are
blocked when observation-date validation is enabled.

The observation-date setting can be disabled for an analysis configuration
where calendar sequencing is not required. The backend uses the same setting
and continues to apply the rest of the run validation.

### Run and monitor analysis

Start the run from the Analysis page. The backend performs synchronous
preflight validation, then starts a background worker. Progress and status
can be monitored in the frontend.

Vegetation-only runs produce:

- the T1 vegetation health classification;
- class, area, and confidence metrics;
- QA and validation information;
- Markdown, GeoTIFF, JSON, and PDF outputs where supported.

Wetland runs additionally produce:

- 4 x 4 vegetation transitions;
- Change Vector Analysis;
- degradation and recovery evidence;
- NDWI surface-water change;
- wetland-context and combined-evidence maps.

Physical wetland depletion is reported only when the configured independent
evidence requirements are satisfied. Otherwise, the status remains
`NOT_ASSESSED`.

### Review results and reports

Use the Results, Validation, and Report pages to inspect maps and metrics,
download output files, and open or export the generated analysis report.

The API documentation is available while the backend is running at
`http://127.0.0.1:8000/docs`.

---

## 4. Running Automated Tests

### 1. Scientific GIS Engine Unit Tests (265 Tests)
```bash
cd backend/Engine_module
python -m pytest
```

### 2. FastAPI Application Server Integration Tests
```bash
python -m pytest backend/tests/test_fastapi_server.py
```

### 3. End-to-End Workflow & Scientific Regression Test Suite
```bash
python -m pytest backend/tests/test_e2e_full_integration.py
```

---

## 5. Scientific Principles

- **5-Channel Model Input**: Channels `0=NDVI`, `1=GNDVI`, `2=NDRE`, `3=NIR`, `4=RedEdge`.
- **4 Vegetation Classes**: `0=Background`, `1=Severe Stress`, `2=Moderate Stress`, `3=Healthy Vegetation`.
- **Authoritative Change Analysis**: 4×4 Transition Matrix, Change Vector Analysis (CVA), Vegetation Degradation/Recovery, and NDWI Surface Water extent.
- **Physical Wetland Depletion**: Claimed ONLY when independent temporal wetland spatial evidence is accepted via configuration.

## 6. Changelog

### Unreleased

- Added synchronous backend preflight validation for project state,
  preparation, analysis mode, ONNX model shape, required datasets and bands,
  spatial compatibility, and observation dates.
- Blocked analysis requests now fail before creating run state or output
  directories.
- Added a structured `GET /api/runs/readiness/{project_id}` endpoint.
- Added the Analysis Readiness Checklist with PASS, WARNING, and BLOCKED states.
- Added frontend run-button gating based on readiness.
- Added editable observation dates and support for disabling
  `use_observation_dates` validation.
- Added consistent sample-dataset fallback handling for vegetation and wetland
  modes.
- Added user-friendly model and validation errors that avoid exposing internal
  filesystem paths.
- Improved global error visibility and documented the complete user workflow.
- Added focused backend readiness/preflight tests and frontend checklist tests.

## 7. Complete Documentation

See [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md) for the
complete project guide, including:

- architecture and repository structure;
- vegetation and wetland workflows;
- input data requirements and scientific processing;
- backend API endpoints;
- configuration and output package details;
- frontend workflow;
- testing, validation, limitations, and troubleshooting.
