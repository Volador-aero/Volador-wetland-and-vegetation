import { useMemo, useState } from "react";
import { useProject } from "../../context/useProject";

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function normalizeStatus(
  value: unknown,
  fallback = "NOT_REPORTED"
): string {
  if (typeof value !== "string" || !value.trim()) {
    return fallback;
  }

  return value.trim().toUpperCase();
}

function readStatus(value: unknown): string {
  if (typeof value === "string") {
    return normalizeStatus(value);
  }

  return normalizeStatus(asRecord(value).status);
}

function formatLabel(value: string): string {
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

const CHECK_LABELS: Record<string, string> = {
  input: "Input Band Validation",
  vegetation_index: "Vegetation Index Validation",
  model: "Model Execution Validation",
  class_distribution: "Class Distribution Validation",
  spatial: "Spatial Grid Alignment",
  nodata: "NoData Propagation",
  temporal: "Temporal Comparability",
  transition: "Transition Matrix Validation",
  change_patch: "Change Patch Validation",
  cva: "Change Vector Analysis",
  wetland: "Wetland Evidence Validation",
  metrics: "Area and Degradation Metrics",
  uncertainty: "Scientific Uncertainty",
  reproducibility: "Reproducibility and Provenance",
};

export default function ValidationPage() {
  const { results, validation: dsValidation } = useProject();

  const qa = results?.validation ?? {};
  const overallStatus = normalizeStatus(
    qa.overall_status || qa.status,
    results ? "NOT_REPORTED" : "PENDING"
  );

  // Keep findings stable between renders to avoid unnecessary useMemo recalculation.
  const rawFindings = results?.validation?.flags;

  const findings = useMemo(
    () => (Array.isArray(rawFindings) ? rawFindings : []),
    [rawFindings]
  );

  const [searchQuery, setSearchQuery] = useState("");
  const [severityFilter, setSeverityFilter] = useState("ALL");
  const [statusFilter, setStatusFilter] = useState("ALL");

  const hasActiveFilters =
    searchQuery.trim() !== "" ||
    severityFilter !== "ALL" ||
    statusFilter !== "ALL";

  const filteredFindings = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();

    return findings.filter((finding) => {
      const severity = normalizeStatus(finding.severity, "UNSPECIFIED");
      const status = normalizeStatus(finding.status);

      const searchableText = [
        finding.code,
        finding.message,
        finding.phase,
        finding.recommended_action,
        asRecord(finding).category,
        asRecord(finding).check,
        severity,
        status,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();

      const matchesSearch = !query || searchableText.includes(query);

      const matchesSeverity =
        severityFilter === "ALL" || severity === severityFilter;

      const matchesStatus =
        statusFilter === "ALL" || status === statusFilter;

      return matchesSearch && matchesSeverity && matchesStatus;
    });
  }, [findings, searchQuery, severityFilter, statusFilter]);

  const clearFilters = () => {
    setSearchQuery("");
    setSeverityFilter("ALL");
    setStatusFilter("ALL");
  };

  const structuredChecks = asRecord(qa.checks);
  const structuredCategories = asRecord(qa.categories);
  const qaRecord = asRecord(qa);

  const requiredChecks = Array.isArray(qaRecord.required_checks)
    ? qaRecord.required_checks.filter(
        (item): item is string => typeof item === "string"
      )
    : [];

  const checkKeys = Array.from(
    new Set([
      ...requiredChecks,
      ...Object.keys(structuredChecks),
    ])
  );

  const hasStructuredChecks = checkKeys.length > 0;

  const fallbackStatus = results ? "NOT_REPORTED" : "PENDING";

  const qaChecks = hasStructuredChecks
    ? checkKeys.map((key) => ({
        label: CHECK_LABELS[key] ?? formatLabel(key),
        status: readStatus(structuredChecks[key]),
      }))
    : [
        {
          label: "Input Band Completeness",
          status: fallbackStatus,
        },
        {
          label: "Spatial Grid Alignment",
          status: normalizeStatus(
            qa.qa_flags?.alignment,
            fallbackStatus
          ),
        },
        {
          label: "Temporal Phenology Interval",
          status: normalizeStatus(
            qa.qa_flags?.temporal,
            fallbackStatus
          ),
        },
        {
          label: "5-Channel Model Execution",
          status: normalizeStatus(
            qa.qa_flags?.inference,
            fallbackStatus
          ),
        },
        {
          label: "Raster Integrity & Georeferencing",
          status: fallbackStatus,
        },
        {
          label: "Olofsson Sample Validation",
          status: normalizeStatus(
            qa.olofsson_status,
            fallbackStatus
          ),
        },
        {
          label: "Independent Hydrological Evidence",
          status: normalizeStatus(
            qa.hydro_status,
            fallbackStatus
          ),
        },
      ];

  const categoryChecks = Object.entries(
    structuredCategories
  ).map(([key, value]) => ({
    key,
    label: formatLabel(key),
    status: readStatus(value),
  }));

  const countStatuses = (statuses: string[]) => {
    const counts: Record<string, number> = {};

    for (const status of statuses) {
      counts[status] = (counts[status] ?? 0) + 1;
    }

    return Object.entries(counts);
  };

  const checkSummary = countStatuses(
    qaChecks.map((item) => item.status)
  );

  const findingSummary = countStatuses(
    findings.map((item) => normalizeStatus(item.status))
  );

  const getStatusClass = (status: string) => {
    const normalized = status.toUpperCase();

    if (normalized === "PASS") {
      return "bg-emerald-500/10 text-emerald-400";
    }

    if (normalized === "WARN" || normalized === "WARNING") {
      return "bg-amber-500/10 text-amber-400";
    }

    if (normalized === "FAIL" || normalized === "ERROR") {
      return "bg-red-500/10 text-red-400";
    }

    return "bg-gray-500/10 text-gray-400";
  };

  const getSeverityClass = (severity: string) => {
    switch (severity.toUpperCase()) {
      case "CRITICAL":
      case "HIGH":
        return "bg-red-500/10 text-red-400";

      case "MEDIUM":
        return "bg-amber-500/10 text-amber-400";

      case "LOW":
        return "bg-blue-500/10 text-blue-400";

      default:
        return "bg-gray-500/10 text-gray-400";
    }
  };

  return (
    <div className="space-y-5">
      {/* Page Header */}
      <section>
        <div className="text-2xl font-semibold tracking-tight text-[var(--text)]">
          Quality Assurance & Scientific Validation
        </div>

        <div className="mt-1 text-sm text-[var(--text-secondary)]">
          Review scientific input validation, QA flags, and independent
          reference sample availability.
        </div>
      </section>

      {/* Overall Pipeline Status */}
      <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="text-sm font-medium text-[var(--text)]">
              Overall Pipeline QA Status
            </div>

            <div className="mt-1 text-xs text-[var(--text-secondary)]">
              {results
                ? `Overall QA State: ${overallStatus}`
                : "Pending execution."}
            </div>
          </div>

          <div
            className={`rounded-md border px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.12em] ${
              overallStatus === "PASS"
                ? "border-emerald-500/30 bg-emerald-500/20 text-emerald-400"
                : overallStatus === "WARN" || overallStatus === "WARNING"
                  ? "border-amber-500/30 bg-amber-500/20 text-amber-400"
                  : overallStatus === "FAIL" || overallStatus === "ERROR"
                    ? "border-red-500/30 bg-red-500/20 text-red-400"
                    : "border-gray-500/30 bg-gray-500/20 text-gray-400"
            }`}
          >
            {overallStatus}
          </div>
        </div>
      </section>

      {/* Systematic Validation Checks */}
      <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
        <div className="text-sm font-medium text-[var(--text)]">
          Systematic Validation Checks
        </div>

        <p className="mt-1 text-xs text-[var(--text-secondary)]">
          {hasStructuredChecks
            ? `${qaChecks.length} engine-reported checks`
            : "Structured engine checks are not available. Showing legacy fields."}
        </p>

        {dsValidation && (
          <p className="mt-2 text-xs text-[var(--text-secondary)]">
            Dataset readiness: {dsValidation.valid ? "READY" : "NOT READY"}.
            Dataset readiness does not establish scientific QA success.
          </p>
        )}

        {hasStructuredChecks && (
          <div
            aria-label="Validation check summary"
            className="mt-3 flex flex-wrap gap-2"
          >
            {checkSummary.map(([status, count]) => (
              <span
                key={status}
                className={`rounded px-2.5 py-1 text-xs font-semibold ${getStatusClass(status)}`}
              >
                {count} {status}
              </span>
            ))}
          </div>
        )}

        <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
          {qaChecks.map((item) => (
            <div
              key={item.label}
              className="flex items-center justify-between gap-3 rounded-md border border-[var(--border)] p-4"
            >
              <span className="text-xs font-medium text-[var(--text)]">
                {item.label}
              </span>

              <span
                className={`shrink-0 rounded px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.1em] ${getStatusClass(item.status)}`}
              >
                {item.status}
              </span>
            </div>
          ))}
        </div>
      </section>

      {categoryChecks.length > 0 && (
        <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
          <div className="text-sm font-medium text-[var(--text)]">
            Validation Categories
          </div>

          <p className="mt-1 text-xs text-[var(--text-secondary)]">
            {categoryChecks.length} engine-reported categories
          </p>

          <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
            {categoryChecks.map((item) => (
              <div
                key={item.key}
                className="flex items-center justify-between gap-3 rounded-md border border-[var(--border)] p-3"
              >
                <span className="text-xs font-medium text-[var(--text)]">
                  {item.label}
                </span>

                <span className={`rounded px-2 py-1 text-[10px] font-semibold ${getStatusClass(item.status)}`}>
                  {item.status}
                </span>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Actionable QA Findings */}
      <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="text-sm font-medium text-[var(--text)]">
              Actionable QA Findings
            </div>

            <div className="mt-1 text-xs text-[var(--text-secondary)]">
              Search and filter findings to identify issues requiring attention.
            </div>
          </div>

          <span className="rounded-md bg-gray-500/10 px-3 py-1.5 text-xs font-semibold text-gray-400">
            {filteredFindings.length} of {findings.length} Findings
          </span>
        </div>

        {findingSummary.length > 0 && (
          <div
            aria-label="QA finding summary"
            className="mt-3 flex flex-wrap gap-2"
          >
            {findingSummary.map(([status, count]) => (
              <span
                key={status}
                className={`rounded px-2.5 py-1 text-xs font-semibold ${getStatusClass(status)}`}
              >
                {count} {status}
              </span>
            ))}
          </div>
        )}

        {/* Search and Filters */}
        <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-4">
          <div className="md:col-span-2">
            <label
              htmlFor="qa-search"
              className="mb-1.5 block text-xs font-medium text-[var(--text-secondary)]"
            >
              Search Findings
            </label>

            <input
              id="qa-search"
              type="search"
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              placeholder="Search code, category, check, message or action..."
              className="w-full rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm text-[var(--text)] outline-none transition focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            />
          </div>

          <div>
            <label
              htmlFor="qa-severity"
              className="mb-1.5 block text-xs font-medium text-[var(--text-secondary)]"
            >
              Severity
            </label>

            <select
              id="qa-severity"
              value={severityFilter}
              onChange={(event) => setSeverityFilter(event.target.value)}
              className="w-full rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm text-[var(--text)] outline-none focus:border-blue-500"
            >
              <option value="ALL">All Severities</option>
              <option value="CRITICAL">Critical</option>
              <option value="HIGH">High</option>
              <option value="MEDIUM">Medium</option>
              <option value="LOW">Low</option>
              <option value="INFO">Info</option>
              <option value="UNSPECIFIED">Unspecified</option>
            </select>
          </div>

          <div>
            <label
              htmlFor="qa-status"
              className="mb-1.5 block text-xs font-medium text-[var(--text-secondary)]"
            >
              Status
            </label>

            <select
              id="qa-status"
              value={statusFilter}
              onChange={(event) => setStatusFilter(event.target.value)}
              className="w-full rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm text-[var(--text)] outline-none focus:border-blue-500"
            >
              <option value="ALL">All Statuses</option>
              <option value="FAIL">Fail</option>
              <option value="ERROR">Error</option>
              <option value="WARN">Warn</option>
              <option value="WARNING">Warning</option>
              <option value="PASS">Pass</option>
              <option value="NOT_AVAILABLE">Not Available</option>
              <option value="NOT_RUN">Not Run</option>
              <option value="NOT_APPLICABLE">Not Applicable</option>
              <option value="NOT_REPORTED">Not Reported</option>
              {Array.from(
                new Set(findings.map((item) => normalizeStatus(item.status)))
              )
                .filter((status) => ![
                  "ALL", "FAIL", "ERROR", "WARN", "WARNING", "PASS",
                  "NOT_AVAILABLE", "NOT_RUN", "NOT_APPLICABLE", "NOT_REPORTED",
                ].includes(status))
                .map((status) => (
                  <option key={status} value={status}>
                    {formatLabel(status.toLowerCase())}
                  </option>
                ))}
            </select>
          </div>
        </div>

        {/* Filter Summary and Reset */}
        {hasActiveFilters && (
          <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
            <p className="text-xs text-[var(--text-secondary)]">
              Showing {filteredFindings.length} matching{" "}
              {filteredFindings.length === 1 ? "finding" : "findings"}.
            </p>

            <button
              type="button"
              onClick={clearFilters}
              className="rounded-md border border-[var(--border)] px-3 py-1.5 text-xs font-medium text-[var(--text)] transition hover:bg-gray-500/10"
            >
              Clear Filters
            </button>
          </div>
        )}

        {/* Findings List */}
        <div className="mt-4 space-y-3">
          {filteredFindings.length > 0 ? (
            filteredFindings.map((finding, index) => {
              const severity = normalizeStatus(
                finding.severity,
                "UNSPECIFIED"
              );

              const status = normalizeStatus(finding.status);

              const category = asRecord(finding).category;
              const check = asRecord(finding).check;

              return (
                <article
                  key={`${finding.code || "finding"}-${index}`}
                  className="rounded-md border border-[var(--border)] p-4 transition-colors hover:border-gray-500/40"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-sm font-semibold text-[var(--text)]">
                        {finding.code || "QA Finding"}
                      </span>

                      <span
                        className={`rounded px-2 py-1 text-[10px] font-semibold ${getSeverityClass(severity)}`}
                      >
                        {severity}
                      </span>
                    </div>

                    <span
                      className={`rounded px-2 py-1 text-[10px] font-semibold ${getStatusClass(status)}`}
                    >
                      {status}
                    </span>
                  </div>

                  {finding.phase && (
                    <div className="mt-2 text-xs text-[var(--text-secondary)]">
                      Phase: {finding.phase}
                    </div>
                  )}

                  {typeof category === "string" && (
                    <div className="mt-1 text-xs text-[var(--text-secondary)]">
                      Category: {formatLabel(category)}
                    </div>
                  )}

                  {typeof check === "string" && (
                    <div className="mt-1 text-xs text-[var(--text-secondary)]">
                      Check: {CHECK_LABELS[check] ?? formatLabel(check)}
                    </div>
                  )}

                  <p className="mt-2 text-sm text-[var(--text)]">
                    {finding.message || "No finding description was provided."}
                  </p>

                  {finding.recommended_action && (
                    <div className="mt-3 rounded-md border border-blue-500/20 bg-blue-500/5 p-3">
                      <div className="text-xs font-semibold text-blue-400">
                        Recommended Action
                      </div>

                      <p className="mt-1 text-xs leading-relaxed text-[var(--text-secondary)]">
                        {finding.recommended_action}
                      </p>
                    </div>
                  )}
                </article>
              );
            })
          ) : (
            <div className="rounded-md border border-dashed border-[var(--border)] p-6 text-center">
              <p className="text-sm font-medium text-[var(--text)]">
                {findings.length === 0
                  ? results
                    ? "No structured QA findings were returned for this analysis."
                    : "Run an analysis to view detailed QA findings."
                  : "No findings match your search or selected filters."}
              </p>

              {findings.length > 0 && hasActiveFilters && (
                <p className="mt-1 text-xs text-[var(--text-secondary)]">
                  Try adjusting the search terms or filters.
                </p>
              )}

              {hasActiveFilters && (
                <button
                  type="button"
                  onClick={clearFilters}
                  className="mt-3 rounded-md border border-[var(--border)] px-3 py-1.5 text-xs font-medium text-[var(--text)] transition hover:bg-gray-500/10"
                >
                  Clear Filters
                </button>
              )}
            </div>
          )}
        </div>
      </section>

      {/* Execution Warnings */}
      {results?.warnings && results.warnings.length > 0 && (
        <section className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-5">
          <div className="text-xs font-medium text-amber-400">
            Non-blocking Execution Warnings
          </div>

          <ul className="mt-2 flex flex-col gap-1 text-xs text-amber-300/80">
            {results.warnings.map((warning, index) => (
              <li key={index} className="flex gap-2">
                <span aria-hidden="true">•</span>
                <span>{warning}</span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}