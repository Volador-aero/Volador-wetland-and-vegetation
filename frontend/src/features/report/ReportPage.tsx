import { useProject } from "../../context/useProject";
import { getPdfReportUrl, getTechnicalPdfReportUrl } from "../../api/client";

type ReportPageProps = {
  analysisMode: "vegetation" | "wetland";
};

export default function ReportPage({ analysisMode }: ReportPageProps) {
  const { project, runStatus, results } = useProject();

  const projectId = project?.id ?? "";
  const runId = runStatus?.run_id ?? "";

  const isCompleted = Boolean(
    projectId &&
    runId &&
    runStatus?.project_id === projectId &&
    runStatus.status === "COMPLETED"
  );

  const pdfUrl = isCompleted
    ? getPdfReportUrl(runId, projectId)
    : "";

  const techPdfUrl = isCompleted
    ? getTechnicalPdfReportUrl(runId, projectId)
    : "";

  // Prefer the prop (selected mode) over what's stored in results.
  // results?.analysis_mode is only available after a completed run;
  // the prop reflects the user's current UI selection immediately.
  const effectiveMode =
    results?.analysis_mode ?? analysisMode ?? "wetland";
  const isVegetation = effectiveMode === "vegetation";

  const simpleReportLabel = isVegetation
    ? "Vegetation Health Report"
    : "Wetland Degradation Report";
  const simpleReportDesc = isVegetation
    ? "Executive summary: single-date canopy health, classification statistics, and key observations."
    : "Executive summary: temporal vegetation transitions, surface-water dynamics, and multi-evidence wetland assessment.";
  const technicalReportLabel = isVegetation
    ? "Vegetation Health Technical Report"
    : "Wetland Degradation Technical Report";
  const technicalReportDesc = isVegetation
    ? "Full technical audit: band statistics, index formulas, model architecture, class metrics, QA, and provenance."
    : "Full technical audit: 4×4 transition matrix, CVA displacements, NDWI dynamics, fusion rules, QA, and provenance.";

  const qa = isCompleted ? results?.validation : undefined;

  const getCheck = (key: string): string => {
    if (!isCompleted) return "PENDING";

    const checks = qa?.checks;

    if (!checks || typeof checks !== "object") {
      return "NOT_REPORTED";
    }

    const value = (checks as Record<string, unknown>)[key];

    if (typeof value === "string") return value;

    if (value && typeof value === "object") {
      const status = (value as Record<string, unknown>).status;
      if (typeof status === "string") return status;
    }

    return "NOT_REPORTED";
  };

  const reportChecks = [
    { label: "Input Validation", state: getCheck("input") },
    { label: "Model Validation", state: getCheck("model") },
    { label: "Class Statistics", state: getCheck("class_distribution") },
    { label: "Vegetation Indices", state: getCheck("vegetation_index") },
    ...(!isVegetation ? [
      { label: "Transition Matrix", state: getCheck("transition") },
      { label: "Change Vector Analysis", state: getCheck("cva") },
      { label: "Area Metrics", state: getCheck("metrics") },
    ] : []),
    {
      label: "Quality Assurance Validation",
      state: isCompleted
        ? qa?.overall_status ?? qa?.status ?? "NOT_REPORTED"
        : "PENDING",
    },
    { label: "Uncertainty", state: getCheck("uncertainty") },
    { label: "Reproducibility", state: getCheck("reproducibility") },
  ];

  return (
    <div className="space-y-5">
      <section>
        <div className="text-2xl font-semibold tracking-tight text-[var(--text)]">
          {isVegetation ? "Vegetation Health Report" : "Cartographic Analysis Report"}
        </div>
        <div className="mt-1 text-sm text-[var(--text-secondary)]">
          Review report readiness, download the compiled ReportLab PDF reports, and preview the
          executive summary.
        </div>
      </section>

      {/* Report Readiness Section */}
      <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-medium text-[var(--text)]">Report Generation Readiness</div>
            <div className="mt-1 text-xs text-[var(--text-secondary)]">
              Automated validation check of required scientific figures, tables, and provenance metadata.
            </div>
          </div>

          <div
            className={`rounded-md px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.12em] ${
              isCompleted
                ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                : "bg-amber-500/20 text-amber-400 border border-amber-500/30"
            }`}
          >
            {isCompleted ? "READY" : "AWAITING ENGINE OUTPUT"}
          </div>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-3">
          {reportChecks.map((check) => (
            <div
              key={check.label}
              className="flex items-center justify-between rounded-md border border-[var(--border)] p-3 text-xs"
            >
              <span className="text-[var(--text)]">{check.label}</span>
              <span
                className={`font-semibold text-[10px] uppercase tracking-[0.1em] ${
                  check.state === "PASS" || check.state === "AVAILABLE"
                    ? "text-emerald-400"
                    : "text-[var(--text-secondary)]"
                }`}
              >
                {check.state}
              </span>
            </div>
          ))}
        </div>
      </section>

      {/* PDF Download Action Bar — Simple Report */}
      <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-medium text-[var(--text)]">{simpleReportLabel}</div>
            <div className="mt-1 text-xs text-[var(--text-secondary)]">{simpleReportDesc}</div>
          </div>

          <a
            href={isCompleted ? pdfUrl : "#"}
            target="_blank"
            rel="noreferrer"
            className={`rounded-md bg-[var(--accent)] px-5 py-2.5 text-xs font-medium text-white transition-opacity hover:opacity-90 ${
              !isCompleted ? "pointer-events-none opacity-40" : ""
            }`}
          >
            Download Report
          </a>
        </div>
      </section>

      {/* PDF Download Action Bar — Technical Report */}
      <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-medium text-[var(--text)]">{technicalReportLabel}</div>
            <div className="mt-1 text-xs text-[var(--text-secondary)]">{technicalReportDesc}</div>
          </div>

          <a
            href={isCompleted ? techPdfUrl : "#"}
            target="_blank"
            rel="noreferrer"
            className={`rounded-md border border-[var(--accent)] bg-transparent px-5 py-2.5 text-xs font-medium text-[var(--accent)] transition-opacity hover:opacity-90 ${
              !isCompleted ? "pointer-events-none opacity-40" : ""
            }`}
          >
            Download Technical Report
          </a>
        </div>
      </section>

      {/* PDF Embedded Viewport — previews the simple report */}
      {isCompleted && (
        <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
          <div className="mb-3 text-sm font-medium text-[var(--text)]">
            PDF Report Preview &nbsp;
            <span className="text-xs font-normal text-[var(--text-secondary)]">
              ({simpleReportLabel})
            </span>
          </div>
          <div className="h-[650px] w-full overflow-hidden rounded-md border border-[var(--border)] bg-[#525659]">
            <iframe
              src={pdfUrl}
              title={`${simpleReportLabel} Preview`}
              className="h-full w-full border-none"
            />
          </div>
        </section>
      )}
    </div>
  );
}