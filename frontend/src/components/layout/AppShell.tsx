import { lazy, Suspense, useEffect, useState } from "react";
import Sidebar from "./Sidebar";
import TopBar from "./TopBar";
import MapViewport from "../maps/MapViewport";
import MetricCard from "../results/MetricCard";
import ValidationStatus from "../datasets/ValidationStatus";
import { useProject } from "../../context/useProject";
import type { DatasetValidationState } from "@/types/dataset";
import { updateProjectDates } from "../../api/client";

const AnalysisPage = lazy(() => import("../../features/analysis/AnalysisPage"));
const ResultsPage = lazy(() => import("../../features/results/ResultsPage"));
const ValidationPage = lazy(() => import("../../features/validation/ValidationPage"));
const ReportPage = lazy(() => import("../../features/report/ReportPage"));
const PrepareAreaPage = lazy(() => import("../../features/preparation/PrepareAreaPage"));

function PageLoadingFallback() {
  return (
    <div
      role="status"
      aria-live="polite"
      className="flex min-h-48 items-center justify-center text-sm text-[var(--text-secondary)]"
    >
      Loading page...
    </div>
  );
}

type AnalysisMode = "vegetation" | "wetland";

type OverviewResultFields = {
  vegetation?: {
    classes?: Record<string, {
      percentage?: number;
    }>;
  };
  uncertainty?: {
    mean_confidence?: number;
  };
};


export default function AppShell() {
  const [activeItem, setActiveItem] = useState("Overview");
  // analysisMode is lifted here so both the Dataset page and the Analysis
  // page share the same mode without needing an extra context value.
  const [modeSelection, setModeSelection] = useState<{
    projectId: string | undefined;
    mode: AnalysisMode;
  } | null>(null);

  const {
    project,
    validation,
    triggerValidation,
    uploadBand,
    results,
    runStatus,
    loading,
    error,
  } = useProject();

  const projectId = project?.id;

  const projectMode: AnalysisMode =
    project?.analysis_mode === "vegetation"
      ? "vegetation"
      : "wetland";

  const analysisMode: AnalysisMode =
    modeSelection && modeSelection.projectId === projectId
      ? modeSelection.mode
      : projectMode;


  const [showDateModal, setShowDateModal] = useState(false);
  const [t1DateInput, setT1DateInput] = useState("");
  const [t2DateInput, setT2DateInput] = useState("");
  const [savingDates, setSavingDates] = useState(false);
  const [dateSaveError, setDateSaveError] = useState<string | null>(null);

  const datePromptKey = JSON.stringify([
    projectId,
    validation?.dates_required === true,
    project?.t1_date ?? "",
    project?.t2_date ?? "",
    analysisMode,
  ]);

  const [previousDatePromptKey, setPreviousDatePromptKey] =
    useState("");

  if (previousDatePromptKey !== datePromptKey) {
    setPreviousDatePromptKey(datePromptKey);

    const needsDates =
      Boolean(projectId) &&
      validation?.dates_required === true &&
      analysisMode === "wetland";

    setShowDateModal(needsDates);
    setT1DateInput(project?.t1_date ?? "");
    setT2DateInput(project?.t2_date ?? "");
    setDateSaveError(null);
  }

  const handleModeChange = (newMode: AnalysisMode) => {
    setModeSelection({
      projectId,
      mode: newMode,
    });
  };

  useEffect(() => {
    if (projectId) {
      void triggerValidation(analysisMode);
    }
  }, [analysisMode, projectId, triggerValidation]);

  const handleSaveDates = async () => {
    if (!project) return;

    if (!t1DateInput || !t2DateInput) {
      setDateSaveError("Please enter both T1 and T2 observation dates.");
      return;
    }

    if (t1DateInput > t2DateInput) {
      setDateSaveError("T1 date must not be later than T2 date.");
      return;
    }

    try {
      setSavingDates(true);
      setDateSaveError(null);

      await updateProjectDates(project.id, t1DateInput, t2DateInput);

      setShowDateModal(false);
      await triggerValidation(analysisMode);
    } catch (err: unknown) {
      setDateSaveError(
        err instanceof Error
          ? err.message
          : "Failed to save observation dates."
      );
    } finally {
      setSavingDates(false);
    }
  };

  const handleFileUpload = (
    timePoint: "t1" | "t2",
    band: "green" | "red" | "rededge" | "nir" | "boundary",
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = e.target.files?.[0];
    if (!file) return;
    uploadBand(timePoint, band, file, analysisMode);
  };

  // ─────────────────────────────────────────────────────────────────────────
  // Shared band upload row
  // ─────────────────────────────────────────────────────────────────────────
  const bandRow = (
    tp: "t1" | "t2",
    b: "green" | "red" | "rededge" | "nir"
  ) => (
    <label
      key={b}
      className="flex items-center justify-between rounded-md border border-[var(--border)] p-2.5 text-xs"
    >
      <span className="capitalize font-medium text-[var(--text)]">{b} Band</span>
      <input
        type="file"
        accept=".tif,.tiff,.TIF"
        className="text-xs text-[var(--text-secondary)]"
        onChange={(e) => handleFileUpload(tp, b, e)}
      />
    </label>
  );

  const renderWorkspace = () => {
    // ─────────────────────────────────────────────────────────────────────
    // OVERVIEW
    // ─────────────────────────────────────────────────────────────────────
    if (activeItem === "Overview") {
      const isVegMode = analysisMode === "vegetation";

      // Vegetation-mode overview metrics derived from results
      const vegClasses =
        (results as OverviewResultFields | null)?.vegetation?.classes ||
        results?.metrics?.class_distribution?.classes ||
        {};
      const healthyPct =
        typeof vegClasses["3"]?.percentage === "number"
          ? `${vegClasses["3"].percentage.toFixed(1)}%`
          : "—";
      const stressPct =
        typeof vegClasses["1"]?.percentage === "number" &&
        typeof vegClasses["2"]?.percentage === "number"
          ? `${(vegClasses["1"].percentage + vegClasses["2"].percentage).toFixed(1)}%`
          : "—";
      const vegConf =
        typeof (results as OverviewResultFields | null)?.uncertainty?.mean_confidence === "number"
          ? `${(((results as OverviewResultFields | null)?.uncertainty?.mean_confidence ?? 0) * 100).toFixed(1)}%`
          : "—";

      return (
        <>
          {/* ── Scene info cards ───────────────────────────────────────────── */}
          {isVegMode ? (
            /* Vegetation: single scene */
            <section className="grid grid-cols-1 gap-4">
              <div className="rounded-lg border border-[var(--accent)]/30 bg-[var(--surface)] p-5">
                <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--accent)]">
                  Vegetation Scene — {project?.t1_date || "—"}
                </div>

                <div className="mt-2 text-lg font-medium text-[var(--text)]">
                  {validation?.valid ? "Dataset Loaded & Validated" : "Not available"}
                </div>

                <div className="mt-1 text-xs text-[var(--text-secondary)]">
                  CRS: {validation?.metadata?.crs || "Not available"} · Resolution:{" "}
                  {validation?.metadata?.resolution || "Not available"} · Single-date vegetation health analysis
                </div>
              </div>
            </section>
          ) : (
            /* Wetland: T1 + T2 */
            <section className="grid grid-cols-2 gap-4">
              <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
                <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">
                  T1 Baseline — {project?.t1_date || "Not available"}
                </div>

                <div className="mt-2 text-lg font-medium text-[var(--text)]">
                  {validation?.valid ? "Dataset Loaded & Validated" : "Not available"}
                </div>

                <div className="mt-1 text-xs text-[var(--text-secondary)]">
                  CRS: {validation?.metadata?.crs || "Not available"} · Resolution:{" "}
                  {validation?.metadata?.resolution || "Not available"}
                </div>
              </div>

              <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
                <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">
                  T2 Comparison — {project?.t2_date || "Not available"}
                </div>

                <div className="mt-2 text-lg font-medium text-[var(--text)]">
                  {validation?.valid ? "Dataset Loaded & Validated" : "Not available"}
                </div>

                <div className="mt-1 text-xs text-[var(--text-secondary)]">
                  CRS: {validation?.metadata?.crs || "Not available"} · Resolution:{" "}
                  {validation?.metadata?.resolution || "Not available"}
                </div>
              </div>
            </section>
          )}

          <MapViewport />

          {/* ── Metric cards — mode-aware ──────────────────────────────────── */}
          {isVegMode ? (
            <section className="grid grid-cols-4 gap-4">
              <MetricCard
                label="Healthy Vegetation"
                value={healthyPct}
              />
              <MetricCard
                label="Total Stress"
                value={stressPct}
              />
              <MetricCard
                label="Mean Confidence"
                value={vegConf}
              />
              <MetricCard
                label="Quality Status"
                value={
                  results?.validation?.overall_status ||
                  runStatus?.status ||
                  (validation?.valid ? "PASS" : "READY")
                }
              />
            </section>
          ) : (
            <section className="grid grid-cols-4 gap-4">
              <MetricCard
                label="Baseline Area"
                value={
                  typeof results?.wetland?.area_ha === "number" && results.wetland.area_ha > 0
                    ? `${results.wetland.area_ha.toFixed(2)} ha`

                    : "—"
                }
              />

              <MetricCard
                label="Affected Area"
                value={
                  typeof results?.degradation?.area_ha === "number"
                    ? `${results.degradation.area_ha.toFixed(2)} ha`
                    : "—"
                }
              />

              <MetricCard
                label="Change Area"
                value={
                  typeof results?.degradation?.area_ha === "number" &&
                  typeof results?.recovery?.area_ha === "number"
                    ? `${(results.degradation.area_ha + results.recovery.area_ha).toFixed(2)} ha`
                    : "—"
                }
              />

              <MetricCard
                label="Quality Status"
                value={
                  results?.validation?.overall_status ||
                  runStatus?.status ||
                  (validation?.valid ? "PASS" : "READY")
                }
              />
            </section>
          )}
        </>
      );
    }

    // ─────────────────────────────────────────────────────────────────────
    // DATASETS  —  mode-aware upload panels
    // ─────────────────────────────────────────────────────────────────────
    if (activeItem === "Datasets") {
      return (
        <div className="space-y-5">
          <section className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="text-2xl font-semibold tracking-tight text-[var(--text)]">
                Datasets Management
              </div>

              <div className="mt-1 text-sm text-[var(--text-secondary)]">
                {analysisMode === "vegetation"
                  ? "Register a single Sentinel-2 scene for single-date vegetation health analysis."
                  : "Register baseline (T1) and comparison (T2) multispectral Sentinel-2 band files and optional wetland boundary vector."}
              </div>
            </div>

            <div className="flex rounded-lg border border-[var(--border)] bg-[var(--surface)] p-1 text-xs">
              <button
                type="button"
                onClick={() => handleModeChange("vegetation")}
                className={`rounded px-3 py-1.5 font-medium transition-colors ${
                  analysisMode === "vegetation"
                    ? "bg-[var(--accent)] text-white shadow-sm"
                    : "text-[var(--text-secondary)] hover:text-[var(--text)]"
                }`}
              >
                Vegetation Mode
              </button>
              <button
                type="button"
                onClick={() => handleModeChange("wetland")}
                className={`rounded px-3 py-1.5 font-medium transition-colors ${
                  analysisMode === "wetland"
                    ? "bg-[var(--accent)] text-white shadow-sm"
                    : "text-[var(--text-secondary)] hover:text-[var(--text)]"
                }`}
              >
                Wetland Mode
              </button>
            </div>
          </section>

          {/* ──────────────────────────────────────────────────────────── */}
          {/* VEGETATION MODE — single scene panel                         */}
          {/* ──────────────────────────────────────────────────────────── */}
          {analysisMode === "vegetation" && (
            <section className="rounded-lg border border-[var(--accent)]/40 bg-[var(--surface)] p-5">
              <div className="mb-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--accent)]">
                Vegetation Scene
              </div>
              <div className="text-sm font-semibold text-[var(--text)]">
                Single-Date Scene
              </div>
              <div className="mt-1 text-xs text-[var(--text-secondary)]">
                Required bands: Green, Red, RedEdge, NIR
              </div>
              <div className="mt-4 flex flex-col gap-2">
                {(["green", "red", "rededge", "nir"] as const).map((b) =>
                  bandRow("t1", b)
                )}
              </div>
            </section>
          )}

          {/* ──────────────────────────────────────────────────────────── */}
          {/* WETLAND MODE — T1 + T2 panels                                */}
          {/* ──────────────────────────────────────────────────────────── */}
          {analysisMode === "wetland" && (
            <section className="grid grid-cols-2 gap-5">
              {/* T1 BASELINE */}
              <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
                <div className="mb-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">
                  T1 Baseline
                </div>
                <div className="text-sm font-semibold text-[var(--text)]">
                  Baseline Scene
                </div>
                <div className="mt-1 text-xs text-[var(--text-secondary)]">
                  Required bands: Green, Red, RedEdge, NIR
                </div>
                <div className="mt-4 flex flex-col gap-2">
                  {(["green", "red", "rededge", "nir"] as const).map((b) =>
                    bandRow("t1", b)
                  )}
                </div>
              </div>

              {/* T2 COMPARISON */}
              <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
                <div className="mb-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">
                  T2 Comparison
                </div>
                <div className="text-sm font-semibold text-[var(--text)]">
                  Comparison Scene
                </div>
                <div className="mt-1 text-xs text-[var(--text-secondary)]">
                  Required bands: Green, Red, RedEdge, NIR
                </div>
                <div className="mt-4 flex flex-col gap-2">
                  {(["green", "red", "rededge", "nir"] as const).map((b) =>
                    bandRow("t2", b)
                  )}
                </div>
              </div>
            </section>
          )}

          {/* Optional Wetland Boundary (Available in both modes) */}
          <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
            <div className="text-sm font-semibold text-[var(--text)]">
              Optional Wetland Boundary
              <span className="ml-2 text-[10px] font-normal text-[var(--text-secondary)]">
                (Optional)
              </span>
            </div>
            <div className="mt-1 text-xs text-[var(--text-secondary)]">
              Upload GeoJSON or Shapefile vector boundary to enable restricted wetland
              spatial analysis and physical depletion assessment.
            </div>
            <div className="mt-4">
              <label className="flex items-center justify-between rounded-md border border-[var(--border)] p-2.5 text-xs">
                <span className="font-medium text-[var(--text)]">
                  Wetland Spatial Boundary (.geojson / .shp / .tif)
                </span>
                <input
                  type="file"
                  accept=".geojson,.json,.shp,.tif,.tiff"
                  className="text-xs text-[var(--text-secondary)]"
                  onChange={(e) => handleFileUpload("t1", "boundary", e)}
                />
              </label>
            </div>
          </section>

          {/* Validation Action */}
          <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm font-medium text-[var(--text)]">
                  Dataset Validation & Spatial Inspection
                </div>
                <div className="mt-1 text-xs text-[var(--text-secondary)]">
                  Verify band matching, spatial dimensions, CRS compatibility, and NoData
                  values before analysis.
                </div>
              </div>

              <button
                type="button"
                disabled={loading}
                onClick={() => triggerValidation(analysisMode)}
                className="rounded-md bg-[var(--accent)] px-4 py-2.5 text-xs font-medium text-white transition-opacity hover:opacity-90 disabled:opacity-50"
              >
                {loading ? "VALIDATING..." : "Validate Datasets"}
              </button>
            </div>

            {validation && (
              <div className="mt-5 rounded-md border border-[var(--border)] bg-[var(--background)] p-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-[var(--text)]">
                    Validation Status
                  </span>
                  <ValidationStatus
                    status={
                      (validation.status === "PASS"
                        ? "READY"
                        : validation.status === "ERROR"
                        ? "INVALID"
                        : validation.status === "WARNING"
                        ? "WARNING"
                        : "NOT_VALIDATED") as DatasetValidationState
                    }
                  />
                </div>
                <ul className="mt-3 flex flex-col gap-1 text-xs text-[var(--text-secondary)]">
                  {validation.messages.map((m, idx) => (
                    <li key={idx}>• {m}</li>
                  ))}
                </ul>
                {validation.metadata && (
                  <div className="mt-3 grid grid-cols-4 gap-2 border-t border-[var(--border)] pt-3 text-[11px] text-[var(--text)]">
                    <div>CRS: {validation.metadata.crs}</div>
                    <div>
                      Dimensions: {validation.metadata.width} x {validation.metadata.height}
                    </div>
                    <div>Resolution: {validation.metadata.resolution}</div>
                    <div>DataType: {validation.metadata.dtype}</div>
                  </div>
                )}
              </div>
            )}
          </section>

          {/* Observation dates modal — only relevant in wetland (temporal) mode */}
          {showDateModal && analysisMode === "wetland" && (
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
              <div className="w-full max-w-md rounded-lg border border-[var(--border)] bg-[var(--surface)] p-6 shadow-2xl">
                <div className="text-lg font-semibold text-[var(--text)]">
                  Observation Dates Required
                </div>

                <div className="mt-2 text-sm text-[var(--text-secondary)]">
                  The acquisition dates could not be detected automatically from the uploaded
                  datasets. Enter the observation date for each scene.
                </div>

                <div className="mt-5 space-y-4">
                  <label className="block">
                    <span className="text-xs font-medium text-[var(--text)]">
                      T1 — Baseline Observation Date
                    </span>
                    <input
                      type="date"
                      value={t1DateInput}
                      onChange={(e) => setT1DateInput(e.target.value)}
                      className="mt-1.5 w-full rounded-md border border-[var(--border)] bg-[var(--background)] px-3 py-2 text-sm text-[var(--text)] outline-none focus:border-[var(--accent)]"
                    />
                  </label>

                  <label className="block">
                    <span className="text-xs font-medium text-[var(--text)]">
                      T2 — Comparison Observation Date
                    </span>
                    <input
                      type="date"
                      value={t2DateInput}
                      onChange={(e) => setT2DateInput(e.target.value)}
                      className="mt-1.5 w-full rounded-md border border-[var(--border)] bg-[var(--background)] px-3 py-2 text-sm text-[var(--text)] outline-none focus:border-[var(--accent)]"
                    />
                  </label>

                  {dateSaveError && (
                    <div className="rounded-md border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-400">
                      {dateSaveError}
                    </div>
                  )}
                </div>

                <div className="mt-6 flex justify-end gap-3">
                  <button
                    type="button"
                    disabled={savingDates}
                    onClick={() => setShowDateModal(false)}
                    className="rounded-md border border-[var(--border)] px-4 py-2 text-xs font-medium text-[var(--text-secondary)] hover:bg-[var(--background)] disabled:opacity-50"
                  >
                    Cancel
                  </button>

                  <button
                    type="button"
                    disabled={savingDates}
                    onClick={handleSaveDates}
                    className="rounded-md bg-[var(--accent)] px-4 py-2 text-xs font-medium text-white hover:opacity-90 disabled:opacity-50"
                  >
                    {savingDates ? "Saving..." : "Save Dates"}
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      );
    }

    if (activeItem === "Analysis")
      return (
        <Suspense fallback={<PageLoadingFallback />}>
<AnalysisPage
          analysisMode={analysisMode}
          onAnalysisModeChange={handleModeChange}
        />
</Suspense>
      );
    if (activeItem === "Prepare") return <Suspense fallback={<PageLoadingFallback />}>
<PrepareAreaPage analysisMode={analysisMode} />
</Suspense>;
    if (activeItem === "Results") return <Suspense fallback={<PageLoadingFallback />}>
<ResultsPage />
</Suspense>;
    if (activeItem === "Validation") return <Suspense fallback={<PageLoadingFallback />}>
<ValidationPage />
</Suspense>;
    if (activeItem === "Report") return <Suspense fallback={<PageLoadingFallback />}>
<ReportPage analysisMode={analysisMode} />
</Suspense>;

    return null;
  };

  return (
    <div className="flex h-screen overflow-hidden bg-[var(--background)]">
      <Sidebar activeItem={activeItem} onNavigate={setActiveItem} />

      <main className="flex min-w-0 flex-1 flex-col">
        <TopBar activeItem={activeItem} />

        <div className="flex min-h-0 flex-1 flex-col gap-5 overflow-auto p-7">
          {error && (
            <div className="rounded-md border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-400">
              {error}
            </div>
          )}
          {renderWorkspace()}
        </div>
      </main>
    </div>
  );
}