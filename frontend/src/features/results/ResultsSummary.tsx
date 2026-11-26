import MetricCard from "../../components/results/MetricCard";
import { useProject } from "../../context/useProject";

export default function ResultsSummary() {
  const { results } = useProject();

  const deg = results?.degradation || results?.metrics?.degradation || {};
  const rec = results?.recovery || {};
  const metrics = results?.metrics || {};

  const analysisScope = metrics.analysis_scope;

  const baseline = metrics.baseline_area;

  const totalAreaHa =
    baseline &&
    typeof baseline === "object" &&
    "area_ha" in baseline
      ? baseline.area_ha
      : undefined;

  const baselineHa = typeof totalAreaHa === "number" && Number.isFinite(totalAreaHa) ? `${totalAreaHa.toFixed(2)} ha` : "—";

  const degHa = deg.area_ha ?? metrics.degradation?.degraded_area_ha ?? metrics.impact_area?.degradation_ha;
  const affectedHa = typeof degHa === "number" && Number.isFinite(degHa) ? `${degHa.toFixed(2)} ha` : "—";

  const swdVal = metrics.severity_weighted_degradation_area_ha ?? metrics.degradation?.severity_weighted_degradation_ha;
  const swdHa = typeof swdVal === "number" && Number.isFinite(swdVal) ? `${swdVal.toFixed(2)} ha_deg` : "—";

  const ndrVal = metrics.normalized_degradation_ratio ?? metrics.degradation?.normalized_degradation_ratio;
  const ratioPct = typeof ndrVal === "number" && Number.isFinite(ndrVal) ? `${(ndrVal * 100).toFixed(1)}%` : "—";

  const recHaVal = rec.area_ha ?? metrics.degradation?.recovered_area_ha ?? metrics.recovery_area?.recovery_ha;
  const recoveryHa = typeof recHaVal === "number" && Number.isFinite(recHaVal) ? `${recHaVal.toFixed(2)} ha` : "—";

  return (
    <section>
      <div className="mb-3">
        <div className="text-sm font-medium text-[var(--text)]">
          Analysis Area Summary
        </div>

        <div className="mt-1 text-xs text-[var(--text-secondary)]">
          Quantified metrics computed by the GIS Engine.
        </div>
      </div>

      <div className="grid grid-cols-5 gap-4">
        <MetricCard
          label="Baseline Area"
          value={baselineHa}
          description={
            analysisScope === "GENERAL_VEGETATION"
              ? "Valid general vegetation analysis area."
              : analysisScope === "WETLAND_CONTEXT"
                ? "Valid wetland-context analysis area."
                : "Engine-reported baseline; scope not established."
          }
        />

        <MetricCard
          label="Affected Area"
          value={affectedHa}
          description="Area experiencing degradation."
        />

        <MetricCard
          label="Severity-Weighted"
          value={swdHa}
          description="Area weighted by degradation severity."
        />

        <MetricCard
          label="Degradation Ratio"
          value={ratioPct}
          description="Normalized degradation ratio."
        />

        <MetricCard
          label="Recovery Area"
          value={recoveryHa}
          description="Area showing recovery transitions."
        />
      </div>
    </section>
  );
}