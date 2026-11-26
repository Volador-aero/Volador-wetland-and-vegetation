import {
  render,
  screen,
  cleanup,
  fireEvent,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ValidationPage from "../ValidationPage";

const { mockUseProject } = vi.hoisted(() => ({
  mockUseProject: vi.fn(),
}));

vi.mock("../../../context/useProject", () => ({
  useProject: mockUseProject,
}));

const sampleFindings = [
  {
    code: "QA-011",
    severity: "HIGH",
    status: "FAIL",
    phase: "Alignment",
    message: "Spatial alignment check failed.",
    recommended_action: "Check raster coordinate systems.",
  },
  {
    code: "QA-012",
    severity: "MEDIUM",
    status: "WARN",
    phase: "Temporal",
    message: "Temporal interval needs review.",
    recommended_action: "Review the selected observation dates.",
  },
  {
    code: "QA-013",
    severity: "LOW",
    status: "PASS",
    phase: "Inference",
    message: "Model inference completed successfully.",
  },
];

const mockResults = {
  validation: {
    overall_status: "WARN",
    flags: sampleFindings,
  },
  warnings: [],
};

describe("ValidationPage - Actionable QA Findings", () => {
  beforeEach(() => {
    mockUseProject.mockReset();
  });

  afterEach(() => {
    cleanup();
  });

  it("shows the pending message when no analysis results exist", () => {
    mockUseProject.mockReturnValue({
      results: null,
      validation: null,
    });

    render(<ValidationPage />);

    expect(
      screen.getByText("Run an analysis to view detailed QA findings."),
    ).toBeInTheDocument();

    expect(screen.getByText("0 of 0 Findings")).toBeInTheDocument();
  });

  it("displays finding code, severity, status and phase", () => {
    mockUseProject.mockReturnValue({
      results: mockResults,
      validation: null,
    });

    render(<ValidationPage />);

    expect(screen.getByText("QA-011")).toBeInTheDocument();
    expect(screen.getByText("HIGH")).toBeInTheDocument();
    expect(screen.getByText("FAIL")).toBeInTheDocument();
    expect(screen.getByText("Phase: Alignment")).toBeInTheDocument();
    expect(screen.getByText("Spatial alignment check failed.")).toBeInTheDocument();

    expect(screen.getByText("3 of 3 Findings")).toBeInTheDocument();
  });

  it("displays the recommended action", () => {
    mockUseProject.mockReturnValue({
      results: mockResults,
      validation: null,
    });

    render(<ValidationPage />);

    expect(screen.getAllByText("Recommended Action").length).toBe(2);

    expect(
      screen.getByText("Check raster coordinate systems."),
    ).toBeInTheDocument();
  });

  it("searches findings by code", () => {
    mockUseProject.mockReturnValue({
      results: mockResults,
      validation: null,
    });

    render(<ValidationPage />);

    fireEvent.change(screen.getByLabelText("Search Findings"), {
      target: { value: "QA-012" },
    });

    expect(screen.getByText("QA-012")).toBeInTheDocument();
    expect(screen.queryByText("QA-011")).not.toBeInTheDocument();
    expect(screen.queryByText("QA-013")).not.toBeInTheDocument();

    expect(screen.getByText("1 of 3 Findings")).toBeInTheDocument();
  });

  it("searches findings by message", () => {
    mockUseProject.mockReturnValue({
      results: mockResults,
      validation: null,
    });

    render(<ValidationPage />);

    fireEvent.change(screen.getByLabelText("Search Findings"), {
      target: { value: "coordinate systems" },
    });

    expect(screen.getByText("QA-011")).toBeInTheDocument();
    expect(screen.queryByText("QA-012")).not.toBeInTheDocument();
  });

  it("filters findings by severity", () => {
    mockUseProject.mockReturnValue({
      results: mockResults,
      validation: null,
    });

    render(<ValidationPage />);

    fireEvent.change(screen.getByLabelText("Severity"), {
      target: { value: "HIGH" },
    });

    expect(screen.getByText("QA-011")).toBeInTheDocument();
    expect(screen.queryByText("QA-012")).not.toBeInTheDocument();
    expect(screen.queryByText("QA-013")).not.toBeInTheDocument();
  });

  it("filters findings by status", () => {
    mockUseProject.mockReturnValue({
      results: mockResults,
      validation: null,
    });

    render(<ValidationPage />);

    fireEvent.change(screen.getByLabelText("Status"), {
      target: { value: "WARN" },
    });

    expect(screen.getByText("QA-012")).toBeInTheDocument();
    expect(screen.queryByText("QA-011")).not.toBeInTheDocument();
    expect(screen.queryByText("QA-013")).not.toBeInTheDocument();
  });

  it("combines search and severity filters", () => {
    mockUseProject.mockReturnValue({
      results: mockResults,
      validation: null,
    });

    render(<ValidationPage />);

    fireEvent.change(screen.getByLabelText("Search Findings"), {
      target: { value: "alignment" },
    });

    fireEvent.change(screen.getByLabelText("Severity"), {
      target: { value: "HIGH" },
    });

    expect(screen.getByText("QA-011")).toBeInTheDocument();
    expect(screen.queryByText("QA-012")).not.toBeInTheDocument();
  });

  it("shows an empty state when no filters match", () => {
    mockUseProject.mockReturnValue({
      results: mockResults,
      validation: null,
    });

    render(<ValidationPage />);

    fireEvent.change(screen.getByLabelText("Search Findings"), {
      target: { value: "something that does not exist" },
    });

    expect(
      screen.getByText(
        "No findings match your search or selected filters.",
      ),
    ).toBeInTheDocument();

    expect(screen.getByText("0 of 3 Findings")).toBeInTheDocument();
  });

  it("clears all active filters", () => {
    mockUseProject.mockReturnValue({
      results: mockResults,
      validation: null,
    });

    render(<ValidationPage />);

    fireEvent.change(screen.getByLabelText("Search Findings"), {
      target: { value: "QA-011" },
    });

    fireEvent.change(screen.getByLabelText("Severity"), {
      target: { value: "HIGH" },
    });

    fireEvent.click(screen.getByRole("button", { name: "Clear Filters" }));

    expect(screen.getByLabelText("Search Findings")).toHaveValue("");
    expect(screen.getByLabelText("Severity")).toHaveValue("ALL");
    expect(screen.getByLabelText("Status")).toHaveValue("ALL");
    expect(screen.getByText("3 of 3 Findings")).toBeInTheDocument();
  });

  it("shows an empty state when results contain no QA findings", () => {
    mockUseProject.mockReturnValue({
      results: {
        validation: {
          overall_status: "PASS",
          flags: [],
        },
        warnings: [],
      },
      validation: null,
    });

    render(<ValidationPage />);

    expect(
      screen.getByText(
        "No structured QA findings were returned for this analysis.",
      ),
    ).toBeInTheDocument();

    expect(screen.getByText("0 of 0 Findings")).toBeInTheDocument();
  });

  it("uses safe fallback values when finding details are missing", () => {
    mockUseProject.mockReturnValue({
      results: {
        validation: {
          flags: [{ code: "QA-014" }],
        },
        warnings: [],
      },
      validation: null,
    });

    render(<ValidationPage />);

    expect(screen.getByText("QA-014")).toBeInTheDocument();
    expect(screen.getByText("UNSPECIFIED")).toBeInTheDocument();
    expect(screen.getAllByText("NOT_REPORTED").length).toBeGreaterThan(0);

    expect(
      screen.getByText("No finding description was provided."),
    ).toBeInTheDocument();
  });
});


describe("ValidationPage - Structured Engine QA", () => {
  const structuredResults = {
    validation: {
      overall_status: "WARN",
      categories: {
        input: "PASS",
        model: "WARN",
        uncertainty: "NOT_AVAILABLE",
      },
      required_checks: [
        "input",
        "model",
        "uncertainty",
        "reproducibility",
      ],
      checks: {
        input: "PASS",
        model: "WARN",
        uncertainty: "NOT_AVAILABLE",
        reproducibility: "NOT_RUN",
      },
      flags: [
        {
          code: "QA-101",
          severity: "INFO",
          status: "NOT_AVAILABLE",
          category: "uncertainty",
          check: "uncertainty",
          message: "Independent evidence unavailable.",
          recommended_action: "Provide independent samples.",
        },
        {
          code: "QA-102",
          severity: "MEDIUM",
          status: "WARN",
          category: "model",
          check: "class_distribution",
          message: "Class distribution needs review.",
        },
      ],
    },
    warnings: [],
  };

  beforeEach(() => {
    mockUseProject.mockReset();

    mockUseProject.mockReturnValue({
      results: structuredResults,
      validation: { valid: true },
    });
  });

  afterEach(() => {
    cleanup();
  });

  it("renders actual engine-reported validation checks", () => {
    render(<ValidationPage />);

    expect(
      screen.getByText("4 engine-reported checks"),
    ).toBeInTheDocument();

    expect(
      screen.getByText("Input Band Validation"),
    ).toBeInTheDocument();

    expect(
      screen.getByText("Reproducibility and Provenance"),
    ).toBeInTheDocument();
  });

  it("renders the actual validation categories", () => {
    render(<ValidationPage />);

    expect(
      screen.getByText("Validation Categories"),
    ).toBeInTheDocument();

    expect(
      screen.getByText("3 engine-reported categories"),
    ).toBeInTheDocument();
  });

  it("filters unavailable findings without hiding warnings by default", () => {
    render(<ValidationPage />);

    expect(screen.getByText("2 of 2 Findings")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Status"), {
      target: { value: "NOT_AVAILABLE" },
    });

    expect(screen.getByText("QA-101")).toBeInTheDocument();
    expect(screen.queryByText("QA-102")).not.toBeInTheDocument();
    expect(screen.getByText("1 of 2 Findings")).toBeInTheDocument();
  });

  it("searches findings using structured category identifiers", () => {
    render(<ValidationPage />);

    fireEvent.change(screen.getByLabelText("Search Findings"), {
      target: { value: "uncertainty" },
    });

    expect(screen.getByText("QA-101")).toBeInTheDocument();
    expect(screen.queryByText("QA-102")).not.toBeInTheDocument();
  });

  it("does not fabricate PASS for a missing required check", () => {
    mockUseProject.mockReturnValue({
      results: {
        validation: {
          overall_status: "WARN",
          required_checks: ["input", "spatial"],
          checks: { input: "PASS" },
          flags: [],
        },
        warnings: [],
      },
      validation: { valid: true },
    });

    render(<ValidationPage />);

    const spatialRow = screen.getByText(
      "Spatial Grid Alignment",
    ).parentElement;

    expect(spatialRow).toHaveTextContent("NOT_REPORTED");
    expect(spatialRow).not.toHaveTextContent("PASS");
  });

  it("does not convert dataset readiness into scientific validation", () => {
    mockUseProject.mockReturnValue({
      results: null,
      validation: { valid: true },
    });

    render(<ValidationPage />);

    expect(screen.queryByText("PASS")).not.toBeInTheDocument();

    expect(
      screen.getByText(/Dataset readiness: READY/),
    ).toBeInTheDocument();
  });
});
