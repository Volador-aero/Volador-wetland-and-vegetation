import type { AnalysisReadiness } from "../../api/client";

type Props = {
  readiness: AnalysisReadiness | null;
  loading: boolean;
};

const statusIcon = {
  PASS: "\u2713",
  WARNING: "\u26A0",
  BLOCKED: "\u2717",
};

export default function AnalysisReadinessChecklist({ readiness, loading }: Props) {
  if (loading) {
    return (
      <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
        <div className="text-sm font-medium text-[var(--text)]">Analysis Readiness</div>
        <div className="mt-2 text-xs text-[var(--text-secondary)]">Checking prerequisites...</div>
      </section>
    );
  }

  if (!readiness) {
    return (
      <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
        <div className="text-sm font-medium text-[var(--text)]">Analysis Readiness</div>
        <div className="mt-2 text-xs text-[var(--text-secondary)]">Select a project to view readiness.</div>
      </section>
    );
  }

  const blocked = readiness.status === "BLOCKED";

  return (
    <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
      <div className="flex items-center justify-between gap-3">
        <div className="text-sm font-medium text-[var(--text)]">Analysis Readiness</div>
        <div
          className={
            blocked
              ? "rounded-md border border-red-500/30 bg-red-500/10 px-3 py-1 text-xs font-semibold text-red-400"
              : readiness.status === "WARNING"
              ? "rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-1 text-xs font-semibold text-amber-400"
              : "rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 text-xs font-semibold text-emerald-400"
          }
        >
          {blocked ? "ANALYSIS BLOCKED" : readiness.status === "WARNING" ? "READY WITH WARNINGS" : "READY TO RUN"}
        </div>
      </div>

      {blocked && (
        <div className="mt-3 text-xs text-red-400">Fix the failed checks before starting.</div>
      )}

      <div className="mt-4 space-y-2">
        {readiness.checks.map((check) => (
          <div
            key={check.id}
            className="grid grid-cols-[1.25rem_minmax(0,1fr)] gap-2 rounded-md border border-[var(--border)] px-3 py-2"
          >
            <div
              className={
                check.status === "PASS"
                  ? "text-emerald-400"
                  : check.status === "WARNING"
                  ? "text-amber-400"
                  : "text-red-400"
              }
              aria-label={check.status}
            >
              {statusIcon[check.status]}
            </div>
            <div>
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-medium text-[var(--text)]">{check.label}</span>
                <span className="text-[10px] font-semibold tracking-wide text-[var(--text-secondary)]">
                  {check.status}
                </span>
              </div>
              <div className="mt-1 text-xs text-[var(--text-secondary)]">{check.message}</div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
