# GIS Wetland and Vegetation Frontend

React 19, TypeScript, and Vite frontend for the GIS Wetland Degradation &
Depletion Engine.

## Prerequisites

- Node.js 20 or later
- npm
- The backend API running on `http://127.0.0.1:8000`

## Usage

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Open `http://localhost:5173`.

Create a production build:

```bash
npm run build
```

Run the test suite:

```bash
npm test
```

Run lint checks:

```bash
npm run lint
```

## Application workflow

1. Create or select a project.
2. Upload T1 Green, Red, RedEdge, and NIR GeoTIFF bands.
3. For wetland mode, upload the T2 band set and set T1/T2 observation dates.
4. Prepare and lock the analysis area.
5. Review the Analysis Readiness Checklist:
   - **PASS** means the prerequisite is satisfied.
   - **WARNING** means the run can continue with limited evidence.
   - **BLOCKED** means the prerequisite must be corrected before launching.
6. Run vegetation or wetland analysis.
7. Review maps and metrics on the Results and Validation pages.
8. Download or preview the generated report.

The frontend is configured to call `/api` through the backend. If the API URL
changes, update the API client configuration in
`src/api/client.ts`.

## Observation dates

Use **Edit Observation Dates** in the application shell to update T1 and T2
after uploading datasets.

When observation-date validation is enabled, wetland runs require:

- both dates in `YYYY-MM-DD` format;
- T1 earlier than T2;
- no equal dates.

Disabling the observation-date setting omits this validation. Run preflight
and readiness still enforce all remaining model, dataset, spatial, and
preparation requirements.

## Key directories

- `src/api/` - typed backend API client.
- `src/context/` - project and workflow state.
- `src/features/analysis/` - analysis configuration, run progress, and
  readiness checklist.
- `src/features/preparation/` - analysis-area preparation.
- `src/features/results/` - result maps and metrics.
- `src/features/validation/` - QA and validation displays.
- `src/features/report/` - report preview and downloads.
- `src/components/maps/` - OpenLayers map components.

## Changelog

### Unreleased

- Replaced the default Vite template README with project-specific setup and
  usage documentation.
- Added the Analysis Readiness Checklist for PASS, WARNING, and BLOCKED states.
- Added run-button readiness gating.
- Added automatic readiness refresh after project, dataset, date,
  preparation, and analysis-mode changes.
- Added editable observation-date support and `use_observation_dates`
  handling.
- Added user-facing launch and preflight error visibility.

For complete project documentation, see the root
[README](../README.md) and [project documentation](../docs/PROJECT_DOCUMENTATION.md).
