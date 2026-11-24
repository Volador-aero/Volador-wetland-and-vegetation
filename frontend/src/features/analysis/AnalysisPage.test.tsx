// @vitest-environment jsdom
import { fireEvent, render, screen, cleanup } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import AnalysisPage from "./AnalysisPage";

const { resume, check, context } = vi.hoisted(() => ({
  resume: vi.fn(), check: vi.fn(),
  context: {runStatus: {run_id: "run_1", status: "INTERRUPTED", progress: 45}, loading: false},
}));
vi.mock("../../context/useProject", () => ({useProject: () => ({...context, resumeActiveRun: resume})}));
vi.mock("../../api/client", () => ({checkResume: check}));
vi.mock("./AnalysisInputs", () => ({default: () => null}));
vi.mock("./AnalysisConfig", () => ({default: () => null}));
afterEach(cleanup);

it("offers Resume at the backend-confirmed boundary", async () => {
  check.mockResolvedValue({resume_available: true, last_completed_stage: "ALIGNMENT", next_stage: "VEGETATION_INFERENCE"});
  render(<AnalysisPage analysisMode="wetland" onAnalysisModeChange={() => {}} />);
  fireEvent.click(await screen.findByRole("button", {name: "Resume from checkpoint"}));
  expect(resume).toHaveBeenCalledOnce();
  expect(screen.getByText(/Last saved stage: ALIGNMENT/)).toBeTruthy();
});

it("explains why a legacy run needs a retry", async () => {
  check.mockResolvedValue({resume_available: false, resume_blocked_reason: "No saved stages"});
  render(<AnalysisPage analysisMode="wetland" onAnalysisModeChange={() => {}} />);
  expect(await screen.findByText("No saved stages")).toBeTruthy();
  expect(screen.queryByRole("button", {name: "Resume from checkpoint"})).toBeNull();
});
