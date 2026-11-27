# GIS Wetland Degradation & Depletion Engine

Integrated geospatial analysis application for Sentinel-2 temporal vegetation degradation, 4×4 transition analysis, Change Vector Analysis (CVA), NDWI surface-water monitoring, and cartographic PDF report generation.

---

## 1. System Architecture

```text
React 19 Frontend (Vite + OpenLayers + ECharts)
       |
       ▼ HTTP / REST API (JSON + Tile/Raster Stream)
FastAPI Application Layer (backend/app)
       |
       ├── Project Service & Dataset Manager (Local FS Storage)
       ├── Async Background Run Manager (Worker Thread Pool)
       └── Tile & Layer Renderer (Rasterio / GeoTIFF Server)
       |
       ▼
GIS Core Engine (backend/Engine_module/src/gis_engine)
       |
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

### Step 1: Generate Sample Dataset (Optional)

```bash
python backend/scripts/generate_sample_scene.py
python backend/scripts/create_onnx_model.py
```

### Step 2: Start the Backend FastAPI Server

From the project root:

```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

The API server starts at:

- Backend: http://127.0.0.1:8000
- Swagger API documentation: http://127.0.0.1:8000/docs

### Step 3: Start the React Frontend Application

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

The web application opens at:

http://localhost:5173

---

## 3. Running Automated Tests

### 3.1 Scientific GIS Engine Unit Tests

The existing project documentation records 265 GIS engine tests.

```bash
cd backend/Engine_module
python -m pytest
```

### 3.2 FastAPI Application Server Integration Tests

From the project root:

```bash
python -m pytest backend/tests/test_fastapi_server.py
```

### 3.3 End-to-End Workflow and Scientific Regression Tests

```bash
python -m pytest backend/tests/test_e2e_full_integration.py
```

### 3.4 AI Assistant Testing

The AI Assistant enhancement was also validated through backend and frontend tests.

Recorded results:

- Backend test suite: 64 passed, 1 skipped.
- Frontend tests: 12/12 passed.
- Targeted API-client tests: 4/4 passed.
- Production frontend build completed successfully.

The skipped backend test was related to a Windows symlink-check limitation.

---

## 4. Scientific Principles

- **5-Channel Model Input:** Channels `0=NDVI`, `1=GNDVI`, `2=NDRE`, `3=NIR`, `4=RedEdge`.
- **4 Vegetation Classes:** `0=Background`, `1=Severe Stress`, `2=Moderate Stress`, `3=Healthy Vegetation`.
- **Authoritative Change Analysis:** 4×4 Transition Matrix, Change Vector Analysis (CVA), Vegetation Degradation/Recovery, and NDWI Surface Water extent.
- **Physical Wetland Depletion:** Claimed only when independent temporal wetland spatial evidence is accepted via configuration.

---

## 5. AETHRA AI Assistant

The AETHRA AI Assistant is integrated into the GIS application to help users understand GIS and remote-sensing concepts and ask questions using available project, dataset-preparation and completed-analysis context.

The Assistant uses OpenRouter for language-model responses. The existing GIS engine remains responsible for scientific calculations. The Assistant explains available information and must not invent missing measurements.

### 5.1 Backend Dependencies

The Assistant integration uses the OpenAI-compatible SDK and environment configuration.

The required packages include:

```bash
pip install openai python-dotenv
```

If these packages are already included in the backend dependency manifest, install them through that manifest instead.

### 5.2 Environment Configuration

Create a `.env` file inside the `backend` directory:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openrouter/free
```

Replace the placeholder with your own OpenRouter API key.

**Security:** Never commit the `.env` file or expose the API key in frontend code. Keep secrets in the backend environment.

### 5.3 Frontend Environment Configuration

The frontend supports the following environment variable:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

This specifies the FastAPI backend URL. The implementation includes a localhost fallback.

### 5.4 Starting the Backend with the Assistant

From the project root, use the Quick Start backend command in Section 2:

```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Ensure the backend `.env` configuration is available before starting the application.

### 5.5 Assistant API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/assistant/status` | Checks Assistant configuration status. |
| POST | `/api/assistant/chat` | Accepts a question and returns an AI-generated response. |

The Assistant can receive verified project context when available. Its status endpoint indicates configuration state; it does not guarantee that the external provider is reachable.

### 5.6 Assistant Workflow

1. Open the AETHRA application.
2. Select or open the required project.
3. Navigate to the AI Assistant section.
4. Enter a question in the chat interface.
5. Review the generated response.
6. Ask follow-up questions when clarification is needed.

Example questions:

- What is NDVI?
- Explain the difference between NDVI and NDWI.
- What information is available in my selected project?
- Are vegetation analysis metrics available for this project?

Project-specific answers depend on the information available in the selected project and its analysis outputs.

### 5.7 Error Handling and Limitations

- Missing API configuration can prevent Assistant requests from being processed.
- External provider or network failures may result in service errors.
- `READY` indicates that configuration exists; it does not guarantee provider availability.
- Missing project data or metrics should be reported as unavailable rather than treated as zero.
- Code-block rendering was not browser-tested during the recorded validation.
- Some path-safety scenarios involving uploaded datasets, legacy absolute paths and reset operations require further verification.

---

## 6. Additional Documentation

The following documents provide further information about the AI Assistant enhancement and project handover:

| Document | Description |
|---|---|
| `CHANGELOG.md` | Records the main implementation changes, enhancements and known limitations. |
| `USAGE_GUIDE.md` | Explains how to use the AETHRA AI Assistant. |
| `HANDOVER_NOTES.md` | Describes architecture changes, configuration, dependencies, deployment considerations and pending work. |

---

## 7. GitHub Links

- **Main Repository:** https://github.com/Volador-aero/Volador-wetland-and-vegetation
- **AI Assistant Feature Branch:** https://github.com/PranavJha2003/Volador-wetland-and-vegetation/tree/feature/aethra-ai-assistant
- **AI Assistant Pull Request:** https://github.com/Volador-aero/Volador-wetland-and-vegetation/pull/5

