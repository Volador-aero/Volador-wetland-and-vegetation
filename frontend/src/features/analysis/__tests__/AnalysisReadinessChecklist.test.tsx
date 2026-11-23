import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import AnalysisReadinessChecklist from "../AnalysisReadinessChecklist";

describe("AnalysisReadinessChecklist", () => {
  it("renders pass, warning, and blocked states", () => {
    render(
      <AnalysisReadinessChecklist
        loading={false}
        readiness={{
          project_id: "project-1",
          analysis_mode: "wetland",
          ready: false,
          status: "BLOCKED",
          checks: [
            { id: "pass", label: "Datasets ready", status: "PASS", message: "Datasets validated.", blocking: false },
            { id: "warning", label: "Boundary", status: "WARNING", message: "Boundary unavailable.", blocking: false },
            { id: "blocked", label: "Preparation", status: "BLOCKED", message: "Lock the preparation area.", blocking: true },
          ],
        }}
      />
    );

    expect(screen.getByText("ANALYSIS BLOCKED")).toBeTruthy();
    expect(screen.getByText("Datasets ready")).toBeTruthy();
    expect(screen.getByText("Boundary")).toBeTruthy();
    expect(screen.getByText("Preparation")).toBeTruthy();
    expect(screen.getByText("Lock the preparation area.")).toBeTruthy();
  });

  it("shows ready state when no blocking checks exist", () => {
    render(
      <AnalysisReadinessChecklist
        loading={false}
        readiness={{
          project_id: "project-1",
          analysis_mode: "vegetation",
          ready: true,
          status: "READY",
          checks: [
            { id: "pass", label: "Model ready", status: "PASS", message: "Model compatible.", blocking: false },
          ],
        }}
      />
    );

    expect(screen.getByText("READY TO RUN")).toBeTruthy();
  });
});
