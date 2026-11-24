import { useEffect, useState } from "react";
import { checkResume, type ResumeCheck } from "../../api/client";
import AnalysisConfig from "./AnalysisConfig";
import AnalysisInputs from "./AnalysisInputs";
import { useProject } from "../../context/useProject";
import type { RunStatus } from "../../api/client";

type AnalysisMode = "vegetation" | "wetland";

type AnalysisPageProps = {
  analysisMode: AnalysisMode;
  onAnalysisModeChange: (mode: AnalysisMode) => void;
};

const runStatePresentation: Record<
  RunStatus["status"],
  { label: string; description: string; badgeClass: string; dotClass: string }
> = {
  QUEUED: {
    label: "Queued",
    description: "The run is saved and waiting for a background worker.",
    badgeClass: "border-slate-400/40 bg-slate-500/10 text-slate-600 dark:text-slate-300",
    dotClass: "bg-slate-500 animate-pulse",
  },
  RESUMING: {
    label: "Resuming",
    description: "Loading saved stages for this run.",
    badgeClass: "border-blue-500/40 bg-blue-500/10 text-blue-700",
    dotClass: "bg-blue-500 animate-pulse",
  },
  VALIDATING: {
    label: "Validating",
    description: "Checking the uploaded raster inputs and run configuration.",
    badgeClass: "border-blue-500/40 bg-blue-500/10 text-blue-700 dark:text-blue-300",
    dotClass: "bg-blue-500 animate-pulse",
  },
  ALIGNING: {
    label: "Aligning rasters",
    description: "Aligning source rasters to a common spatial grid.",
    badgeClass: "border-cyan-500/40 bg-cyan-500/10 text-cyan-700 dark:text-cyan-300",
    dotClass: "bg-cyan-500 animate-pulse",
  },
  PREPROCESSING: {
    label: "Preprocessing",
    description: "Preparing aligned input data for model inference.",
    badgeClass: "border-cyan-500/40 bg-cyan-500/10 text-cyan-700 dark:text-cyan-300",
    dotClass: "bg-cyan-500 animate-pulse",
  },
  INFERENCE: {
    label: "Running model",
    description: "Executing vegetation classification with the ONNX model.",
    badgeClass: "border-violet-500/40 bg-violet-500/10 text-violet-700 dark:text-violet-300",
    dotClass: "bg-violet-500 animate-pulse",
  },
  CHANGE_ANALYSIS: {
    label: "Analysing change",
    description: "Calculating transitions and change-vector statistics.",
    badgeClass: "border-violet-500/40 bg-violet-500/10 text-violet-700 dark:text-violet-300",
    dotClass: "bg-violet-500 animate-pulse",
  },
  HYDROLOGY: {
    label: "Analysing hydrology",
    description: "Calculating water and wetland evidence.",
    badgeClass: "border-sky-500/40 bg-sky-500/10 text-sky-700 dark:text-sky-300",
    dotClass: "bg-sky-500 animate-pulse",
  },
  METRICS: {
    label: "Calculating metrics",
    description: "Producing area and degradation measurements.",
    badgeClass: "border-indigo-500/40 bg-indigo-500/10 text-indigo-700 dark:text-indigo-300",
    dotClass: "bg-indigo-500 animate-pulse",
  },
  VALIDATION: {
    label: "Checking outputs",
    description: "Running quality checks on generated outputs.",
    badgeClass: "border-blue-500/40 bg-blue-500/10 text-blue-700 dark:text-blue-300",
    dotClass: "bg-blue-500 animate-pulse",
  },
  REPORTING: {
    label: "Generating report",
    description: "Creating final reports and output metadata.",
    badgeClass: "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300",
    dotClass: "bg-amber-500 animate-pulse",
  },
  COMPLETED: {
    label: "Completed",
    description: "The analysis and all output files completed successfully.",
    badgeClass: "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300",
    dotClass: "bg-emerald-500",
  },
  FAILED: {
    label: "Failed",
    description: "The run stopped because an execution error occurred.",
    badgeClass: "border-red-500/40 bg-red-500/10 text-red-700 dark:text-red-300",
    dotClass: "bg-red-500",
  },
  INTERRUPTED: {
    label: "Interrupted",
    description: "The backend restarted before this run could finish.",
    badgeClass: "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300",
    dotClass: "bg-amber-500",
  },
};

export default function AnalysisPage({
  analysisMode,
  onAnalysisModeChange,
}: AnalysisPageProps) {
  const { project, validation, runStatus, triggerRun, retryActiveRun, resumeActiveRun, loading } = useProject();
  const [check, setCheck] = useState<ResumeCheck | null>(null);
  useEffect(() => {
    let active = true;
    setCheck(null);
    if (runStatus && ["FAILED", "INTERRUPTED"].includes(runStatus.status)) {
      checkResume(runStatus.run_id).then(info => { if (active) setCheck(info); })
        .catch(() => { if (active) setCheck({resume_available: false, resume_blocked_reason: "Could not check saved stages."}); });
    }
    return () => { active = false; };
  }, [runStatus?.run_id, runStatus?.status]);

  const [config, setConfig] = useState({
    observationDatesEnabled: true,
    transitionEnabled: true,
    cvaEnabled: true,
    wetlandMaskEnabled: true,
    degradationEnabled: true,
    recoveryEnabled: true,
    areaStatisticsEnabled: true,
    inputQaEnabled: true,
    temporalQaEnabled: true,
    changeQaEnabled: true,
  });

  const toggleConfig = <K extends keyof typeof config>(key: K) => {
    setConfig((current) => ({
      ...current,
      [key]:
        typeof current[key] === "boolean"
          ? !current[key]
          : current[key],
    }));
  };

  const handleChange = (
    key:
      | "analysisMode"
      | "observationDatesEnabled"
      | "transitionEnabled"
      | "cvaEnabled"
      | "wetlandMaskEnabled"
      | "degradationEnabled"
      | "recoveryEnabled"
      | "areaStatisticsEnabled"
      | "inputQaEnabled"
      | "temporalQaEnabled"
      | "changeQaEnabled",
    value?: any
  ) => {
    if (key === "analysisMode") {
      const targetMode: AnalysisMode =
        value === "vegetation" || value === "wetland"
          ? value
          : analysisMode === "vegetation"
          ? "wetland"
          : "vegetation";
      onAnalysisModeChange(targetMode);
    } else {
      toggleConfig(key);
    }
  };

  const isRunning =
    runStatus && !["COMPLETED", "FAILED", "INTERRUPTED"].includes(runStatus.status);

  const isDatasetValid = Boolean(
    project?.datasets_ready || validation?.valid
  );

  const fullConfig = { ...config, analysisMode, analysis_mode: analysisMode };
  const displayedRunState = runStatus
    ? runStatePresentation[runStatus.status]
    : null;

  return (
    <div className="space-y-5">
      <section>
        <div className="text-2xl font-semibold tracking-tight text-[var(--text)]">
          Analysis Pipeline Execution
        </div>

        <div className="mt-1 text-sm text-[var(--text-secondary)]">
          {analysisMode === "vegetation"
            ? "Single-date vegetation health analysis — NDVI, GNDVI, NDRE classification."
            : "Two-date wetland degradation & depletion analysis — temporal change, CVA, hydrology, fusion."}
        </div>
      </section>

      <AnalysisInputs
        t1Date={project?.t1_date ?? null}
        t2Date={project?.t2_date ?? null}
        analysisMode={analysisMode}
      />

      <AnalysisConfig
        analysisMode={analysisMode}
        observationDatesEnabled={config.observationDatesEnabled}
        transitionEnabled={config.transitionEnabled}
        cvaEnabled={config.cvaEnabled}
        wetlandMaskEnabled={config.wetlandMaskEnabled}
        degradationEnabled={config.degradationEnabled}
        recoveryEnabled={config.recoveryEnabled}
        areaStatisticsEnabled={config.areaStatisticsEnabled}
        inputQaEnabled={config.inputQaEnabled}
        temporalQaEnabled={config.temporalQaEnabled}
        changeQaEnabled={config.changeQaEnabled}
        onModeChange={onAnalysisModeChange}
        onChange={handleChange}
      />

      {/* Real Step-by-Step Run Status */}
      <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-medium text-[var(--text)]">
              Pipeline Execution Progress
            </div>

            <div className="mt-1 text-xs text-[var(--text-secondary)]">
              {runStatus
                ? runStatus.current_step
                : "Ready to launch backend execution."}
            </div>
          </div>

          <div
            className={`flex items-center gap-2 rounded-md border px-3 py-1.5 text-xs font-semibold ${
              displayedRunState
                ? displayedRunState.badgeClass
                : "border-[var(--border)] text-[var(--text-secondary)]"
            }`}
          >
            <span
              className={`h-2 w-2 rounded-full ${
                displayedRunState?.dotClass ?? "bg-[var(--text-secondary)]"
              }`}
            />
            {displayedRunState?.label ?? "Not started"}
          </div>
        </div>

        <div className="mt-3 text-xs text-[var(--text-secondary)]">
          {displayedRunState?.description ??
            "No run state has been created for this project."}
        </div>

        {/* Progress Bar */}
        {runStatus && (
          <div className="mt-4">
            <div className="mb-1 flex justify-between text-xs text-[var(--text-secondary)]">
              <span>{runStatus.current_step}</span>
              <span>{runStatus.progress}%</span>
            </div>

            <div className="h-2.5 w-full overflow-hidden rounded-full bg-[var(--border)]">
              <div
                className="h-full bg-[var(--accent)] transition-all duration-300"
                style={{ width: `${runStatus.progress}%` }}
              />
            </div>
          </div>
        )}

        {runStatus && (
          <div className="mt-2 text-[11px] text-[var(--text-secondary)]">
            Run {runStatus.run_id}{runStatus.resume_count ? ` · Resumed ${runStatus.resume_count} time(s)` : ""}.
          </div>
        )}

        {runStatus?.error && (
          <div className="mt-3 rounded-md border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-400">
            Execution Failure: {runStatus.error}
          </div>
        )}

        {runStatus && ["INTERRUPTED", "FAILED"].includes(runStatus.status) && (
          <div className="mt-3 rounded-md border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-700 dark:text-amber-300">
            <div>{runStatus.message || "The backend stopped before analysis completed."}</div>
            <button
              disabled={loading}
              type="button"
              onClick={retryActiveRun}
              className="mt-2 rounded bg-amber-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-amber-700"
            >
              Retry Analysis
            </button>
            {check?.resume_available ? (
              <div className="mt-2">
                <div>Last saved stage: {check.last_completed_stage}. Continue at {check.next_stage}.</div>
                <button type="button" disabled={loading} onClick={resumeActiveRun}
                  className="mt-2 rounded bg-blue-700 px-3 py-1.5 text-white">Resume from checkpoint</button>
              </div>
            ) : <div className="mt-2">{check?.resume_blocked_reason || "Checking saved stages…"}</div>}
          </div>
        )}
      </section>

      <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-medium text-[var(--text)]">
              Run Pipeline
            </div>

            <div className="mt-1 text-xs text-[var(--text-secondary)]">
              {!isDatasetValid && !isRunning
                ? "Dataset validation required before launching analysis run."
                : analysisMode === "vegetation"
                ? "Launch single-date vegetation health inference (Green, Red, RedEdge, NIR → NDVI/GNDVI/NDRE)."
                : "Launch background execution of Sentinel-2 inference, 4×4 transitions, CVA, NDWI, and PDF cartography."}
            </div>
          </div>

          <button
            type="button"
            disabled={Boolean(isRunning) || !isDatasetValid}
            onClick={() => triggerRun(fullConfig)}
            className="rounded-md bg-[var(--accent)] px-5 py-2.5 text-xs font-medium text-white transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {isRunning
              ? `Executing (${runStatus.progress}%)...`
              : !isDatasetValid
              ? "Validation Required"
              : "Run Analysis"}
          </button>
        </div>
      </section>
    </div>
  );
}
