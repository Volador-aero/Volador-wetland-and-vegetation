import { StrictMode } from "react";
import {
  act,
  cleanup,
  renderHook,
  waitFor,
} from "@testing-library/react";
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";

import { ProjectProvider } from "../ProjectContext";
import { useProject } from "../useProject";
import * as api from "../../api/client";

import type {
  Project,
  ValidationResult,
  PreparationState,
  RunStatus,
} from "../../api/client";

vi.mock("../../api/client", () => ({
  getProjects: vi.fn(),
  createProject: vi.fn(),
  validateDatasets: vi.fn(),
  uploadBandFile: vi.fn(),
  launchRun: vi.fn(),
  getRunStatus: vi.fn(),
  getRunResults: vi.fn(),
  getRunLayers: vi.fn(),
  resetProject: vi.fn(),
  getPreparation: vi.fn(),
  savePreparationDraft: vi.fn(),
  commitPreparation: vi.fn(),
  unlockPreparation: vi.fn(),
}));

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;

  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });

  return { promise, resolve, reject };
}

function makeProject(id: string): Project {
  return {
    id,
    name: `Project ${id}`,
    analysis_mode: "wetland",
    datasets_ready: false,
    status: "INVALID",
    preparation_status: "DRAFT",
    is_locked: false,
  } as Project;
}

function makeValidation(valid: boolean): ValidationResult {
  return { valid } as ValidationResult;
}

const projectA = makeProject("project-a");
const projectB = makeProject("project-b");

const preparationA = {
  status: "DRAFT",
  inclusion: null,
} as PreparationState;

const preparationB = {
  status: "COMMITTED",
  inclusion: null,
} as PreparationState;

async function setup() {
  const hook = renderHook(() => useProject(), {
    wrapper: ProjectProvider,
  });

  await waitFor(() => {
    expect(hook.result.current.project?.id)
      .toBe(projectA.id);

    expect(hook.result.current.loading)
      .toBe(false);
  });

  return hook;
}

describe("ProjectContext lifecycle reliability", () => {
  beforeEach(() => {
    vi.resetAllMocks();

    vi.mocked(api.getProjects)
      .mockResolvedValue([projectA, projectB]);

    vi.mocked(api.getPreparation)
      .mockResolvedValue(preparationA);

    vi.mocked(api.validateDatasets)
      .mockResolvedValue(makeValidation(true));

    vi.mocked(api.getRunLayers)
      .mockResolvedValue([]);
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it("initializes the default project", async () => {
    const { result } = await setup();

    expect(result.current.project?.id)
      .toBe(projectA.id);

    expect(result.current.projects)
      .toHaveLength(2);
  });

  it("shares initialization under StrictMode", async () => {
    const { result } = renderHook(() => useProject(), {
      wrapper: ({ children }) => (
        <StrictMode>
          <ProjectProvider>
            {children}
          </ProjectProvider>
        </StrictMode>
      ),
    });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
      expect(result.current.project?.id)
        .toBe(projectA.id);
    });

    expect(api.getProjects).toHaveBeenCalledTimes(1);
  });

  it("does not create duplicate default projects", async () => {
    vi.mocked(api.getProjects).mockResolvedValue([]);
    vi.mocked(api.createProject)
      .mockResolvedValue(projectA);

    const { result } = renderHook(() => useProject(), {
      wrapper: ({ children }) => (
        <StrictMode>
          <ProjectProvider>
            {children}
          </ProjectProvider>
        </StrictMode>
      ),
    });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
      expect(result.current.project?.id)
        .toBe(projectA.id);
    });

    expect(api.createProject)
      .toHaveBeenCalledTimes(1);
  });

  it("ignores validation from a previous project", async () => {
    const { result } = await setup();

    const pending = deferred<ValidationResult>();

    vi.mocked(api.validateDatasets)
      .mockReturnValueOnce(pending.promise);

    let request!: Promise<void>;

    act(() => {
      request = result.current.triggerValidation();
    });

    await act(async () => {
      await result.current.switchProject(projectB.id);
    });

    await act(async () => {
      pending.resolve(makeValidation(false));
      await request;
    });

    expect(result.current.project?.id)
      .toBe(projectB.id);

    expect(result.current.validation).toBeNull();
  });

  it("keeps the newest validation response", async () => {
    const { result } = await setup();

    const older = deferred<ValidationResult>();
    const newer = deferred<ValidationResult>();

    vi.mocked(api.validateDatasets)
      .mockReturnValueOnce(older.promise)
      .mockReturnValueOnce(newer.promise);

    let first!: Promise<void>;
    let second!: Promise<void>;

    act(() => {
      first = result.current.triggerValidation("wetland");
      second = result.current.triggerValidation("vegetation");
    });

    await act(async () => {
      newer.resolve(makeValidation(true));
      await second;
    });

    await act(async () => {
      older.resolve(makeValidation(false));
      await first;
    });

    expect(result.current.validation?.valid)
      .toBe(true);

    expect(result.current.project?.analysis_mode)
      .toBe("vegetation");
  });

  it("ignores stale validation errors", async () => {
    const { result } = await setup();

    const pending = deferred<ValidationResult>();

    vi.mocked(api.validateDatasets)
      .mockReturnValueOnce(pending.promise);

    let request!: Promise<void>;

    act(() => {
      request = result.current.triggerValidation();
    });

    await act(async () => {
      await result.current.switchProject(projectB.id);
    });

    await act(async () => {
      pending.reject(new Error("Old project error"));
      await request;
    });

    expect(result.current.error).toBeNull();
  });

  it("rejects stale responses after switching away and back", async () => {
    const { result } = await setup();

    const pending = deferred<ValidationResult>();

    vi.mocked(api.validateDatasets)
      .mockReturnValueOnce(pending.promise);

    let request!: Promise<void>;

    act(() => {
      request = result.current.triggerValidation();
    });

    await act(async () => {
      await result.current.switchProject(projectB.id);
    });

    await act(async () => {
      await result.current.switchProject(projectA.id);
    });

    await act(async () => {
      pending.resolve(makeValidation(true));
      await request;
    });

    expect(result.current.project?.id)
      .toBe(projectA.id);

    expect(result.current.validation).toBeNull();
  });

  it("ignores stale preparation restoration", async () => {
    const { result } = await setup();

    const pending = deferred<PreparationState>();

    vi.mocked(api.getPreparation)
      .mockReturnValueOnce(pending.promise)
      .mockResolvedValueOnce(preparationA);

    let switching!: Promise<void>;

    act(() => {
      switching = result.current.switchProject(projectB.id);
    });

    await act(async () => {
      await result.current.switchProject(projectA.id);
    });

    await act(async () => {
      pending.resolve(preparationB);
      await switching;
    });

    expect(result.current.project?.id)
      .toBe(projectA.id);

    expect(result.current.preparation)
      .toEqual(preparationA);
  });

  it("ignores stale commit responses", async () => {
    const { result } = await setup();

    const pending = deferred<PreparationState>();

    vi.mocked(api.commitPreparation)
      .mockReturnValueOnce(pending.promise);

    let committing!: Promise<void>;

    act(() => {
      committing = result.current.commitArea();
    });

    await act(async () => {
      await result.current.switchProject(projectB.id);
    });

    await act(async () => {
      pending.resolve(preparationB);
      await committing;
    });

    expect(result.current.project?.id)
      .toBe(projectB.id);

    expect(result.current.project?.is_locked)
      .toBe(false);
  });

  it("prevents duplicate pending analysis launches", async () => {
    const { result } = await setup();

    const pending = deferred<RunStatus>();

    vi.mocked(api.launchRun)
      .mockReturnValueOnce(pending.promise);

    let launching!: Promise<void>;

    act(() => {
      launching = result.current.triggerRun();
    });

    await act(async () => {
      await result.current.triggerRun();
    });

    expect(api.launchRun).toHaveBeenCalledTimes(1);

    await act(async () => {
      pending.resolve({
        run_id: "test-run",
        status: "FAILED",
      } as RunStatus);

      await launching;
    });
  });

  it("clears preparation and analysis state after reset", async () => {
    const { result } = await setup();

    vi.mocked(api.resetProject)
      .mockResolvedValue(projectA);

    await act(async () => {
      await result.current.clearAll();
    });

    expect(result.current.preparation).toBeNull();
    expect(result.current.validation).toBeNull();
    expect(result.current.results).toBeNull();
    expect(result.current.runStatus).toBeNull();
    expect(result.current.layers).toEqual([]);
    expect(result.current.manualExclusions).toEqual([]);
  });

  it("does not replace the active project with a stale reset", async () => {
    const { result } = await setup();

    const pending = deferred<Project>();

    vi.mocked(api.resetProject)
      .mockReturnValueOnce(pending.promise);

    let resetting!: Promise<void>;

    act(() => {
      resetting = result.current.clearAll();
    });

    await act(async () => {
      await result.current.switchProject(projectB.id);
    });

    await act(async () => {
      pending.resolve(projectA);
      await resetting;
    });

    expect(result.current.project?.id)
      .toBe(projectB.id);
  });
});
