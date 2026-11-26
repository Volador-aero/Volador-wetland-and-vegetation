import React, { useState, useEffect, useCallback, useRef } from "react";
import type {
  Project,
  ValidationResult,
  RunStatus,
  MapLayer,
  AnalysisResults,
  PreparationState,
} from "../api/client";
import {
  getProjects,
  createProject,
  validateDatasets,
  uploadBandFile,
  launchRun,
  getRunStatus,
  getRunResults,
  getRunLayers,
  resetProject,
  getPreparation,
  savePreparationDraft,
  commitPreparation,
  unlockPreparation,
} from "../api/client";
import { ProjectContext } from "./projectContextObject";

export type ManualExclusion = {
  id: string;
  geometry: {
    type: "Polygon" | "MultiPolygon";
    coordinates: number[][][] | number[][][][];
  };
  category: string;
  comment: string;
};

export const ProjectProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [project, setProject] = useState<Project | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [runStatus, setRunStatus] = useState<RunStatus | null>(null);
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [layers, setLayers] = useState<MapLayer[]>([]);
  const [activeLayerId, setActiveLayerId] = useState<string>("t2_health");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [manualExclusions, setManualExclusions] = useState<ManualExclusion[]>(
    []
  );
  const [preparation, setPreparation] = useState<PreparationState | null>(null);

  const projectSessionRef = useRef(0);
  const activeProjectIdRef = useRef<string | null>(null);
  const validationRequestRef = useRef(0);
  const resultRequestRef = useRef(0);
  const runLaunchesRef = useRef(new Set<string>());
  const initializationRef = useRef<Promise<Project[]> | null>(null);
  const creationPendingRef = useRef(false);
  const resettingProjectsRef = useRef(new Set<string>());
  const preparationRequestRef = useRef(0);
  const preparationMutationRef = useRef<number | null>(null);

  const activateProjectSession = useCallback((projectId: string) => {
    projectSessionRef.current += 1;
    validationRequestRef.current += 1;
    resultRequestRef.current += 1;
    preparationRequestRef.current += 1;
    preparationMutationRef.current = null;
    activeProjectIdRef.current = projectId;
  }, []);

  const isCurrentSession = useCallback((
    projectId: string,
    session: number
  ) => {
    return (
      activeProjectIdRef.current === projectId &&
      projectSessionRef.current === session
    );
  }, []);


  // Load outputs only for the active project session.
  const loadRunData = useCallback(async (
    runId: string,
    expectedProjectId?: string,
    expectedSession?: number
  ) => {
    const projectId =
      expectedProjectId ?? activeProjectIdRef.current;

    const session =
      expectedSession ?? projectSessionRef.current;

    if (!projectId || !isCurrentSession(projectId, session)) {
      return;
    }

    const request = ++resultRequestRef.current;

    const isCurrentRequest = () =>
      isCurrentSession(projectId, session) &&
      resultRequestRef.current === request;

    try {
      const layerRequest = getRunLayers(runId, projectId)
        .then((value) => ({ ok: true as const, value }))
        .catch((reason: unknown) => ({
          ok: false as const,
          reason,
        }));

      const resData = await getRunResults(runId, projectId);

      if (!isCurrentRequest()) return;

      if (resData.run_id !== runId) {
        throw new Error("Unexpected analysis result identity.");
      }

      setResults(resData);

      const layerResponse = await layerRequest;

      if (!isCurrentRequest()) return;

      if (!layerResponse.ok) {
        setLayers([]);
        setActiveLayerId("");
        setError("Analysis results loaded, but map layers are unavailable.");
        return;
      }

      const layerList = layerResponse.value;
      setLayers(layerList);

      setActiveLayerId((prev) =>
        layerList.some((layer) => layer.id === prev)
          ? prev
          : layerList[0]?.id ?? ""
      );
    } catch (err: unknown) {
      if (!isCurrentRequest()) return;

      console.error("Failed to load run outputs:", err);

      setError(
        err instanceof Error
          ? err.message
          : "Failed to load run outputs."
      );
    }
  }, [isCurrentSession]);

  // Share initialization across StrictMode effect replays.
  useEffect(() => {
    let cancelled = false;

    const initialSession = projectSessionRef.current;

    const initializeProjects = async (): Promise<Project[]> => {
      const existing = await getProjects();

      if (existing.length > 0) {
        return existing;
      }

      const created = await createProject(
        "Wetland Degradation Study",
        "Sentinel-2 5-channel temporal degradation analysis",
        "Central Reserve"
      );

      return [created];
    };

    const init = async () => {
      let activeSession = initialSession;

      const isActive = () =>
        !cancelled &&
        projectSessionRef.current === activeSession;

      try {
        setLoading(true);
        setError(null);

        if (!initializationRef.current) {
          initializationRef.current = initializeProjects();
        }

        const existing = await initializationRef.current;

        if (!isActive()) return;

        const currentProj = existing[0];

        if (!currentProj) {
          throw new Error("No project available.");
        }

        setProjects(existing);

        activateProjectSession(currentProj.id);

        activeSession = projectSessionRef.current;

        setProject(currentProj);

        const preparationRequest =
          ++preparationRequestRef.current;

        try {
          const state = await getPreparation(currentProj.id);

          if (!isActive()) return;

          if (
            preparationRequestRef.current === preparationRequest
          ) {
            setPreparation(state);
          }
        } catch (err) {
          if (isActive()) {
            console.warn(
              "Could not restore project preparation:",
              err
            );
          }
        }

        if (!isActive()) return;

        if (currentProj.latest_run_id) {
          try {
            const existingRun = await getRunStatus(
              currentProj.latest_run_id,
              currentProj.id
            );

            if (!isActive()) return;

            if (
              existingRun.run_id !== currentProj.latest_run_id ||
              existingRun.project_id !== currentProj.id
            ) {
              throw new Error("Unexpected restored run identity.");
            }

            setRunStatus(existingRun);

            if (existingRun.status === "COMPLETED") {
              await loadRunData(
                existingRun.run_id,
                currentProj.id,
                activeSession
              );
            }
          } catch (err) {
            if (isActive()) {
              console.warn(
                "Could not restore the latest run:",
                err
              );
            }
          }
        }

        // AppShell triggers validation for the active project.
        // Avoid issuing a competing initialization request.

      } catch (err: unknown) {
        if (isActive()) {
          const message =
            err instanceof Error
              ? err.message
              : "Unknown error";

          setError(`Backend connection error: ${message}`);
        }
      } finally {
        if (isActive()) {
          setLoading(false);
        }
      }
    };

    void init();

    return () => {
      cancelled = true;
    };
  }, [loadRunData, activateProjectSession]);

  const pollingRunId = runStatus?.run_id;
  const pollingStatus = runStatus?.status;
  const pollingProjectId = project?.id;

  // Schedule the next poll only after the current request finishes.
  useEffect(() => {
    if (
      !pollingRunId ||
      !pollingProjectId ||
      !pollingStatus ||
      ["COMPLETED", "FAILED"].includes(pollingStatus)
    ) {
      return;
    }

    const session = projectSessionRef.current;

    let cancelled = false;

    let timer: ReturnType<typeof setTimeout> | undefined;

    const isActive = () =>
      !cancelled &&
      isCurrentSession(pollingProjectId, session);

    const poll = async () => {
      if (!isActive()) return;

      let terminal = false;

      try {
        const updated = await getRunStatus(pollingRunId, pollingProjectId);

        if (!isActive()) return;

        if (updated.run_id !== pollingRunId || updated.project_id !== pollingProjectId) {
          throw new Error("Unexpected run identity.");
        }

        setRunStatus(updated);

        terminal = ["COMPLETED", "FAILED"].includes(
          updated.status
        );

        if (updated.status === "COMPLETED") {
          await loadRunData(
            pollingRunId,
            pollingProjectId,
            session
          );
        }
      } catch (err) {
        if (isActive()) {
          console.error("Status poll error:", err);
        }
      } finally {
        if (isActive() && !terminal) {
          timer = setTimeout(poll, 1000);
        }
      }
    };

    timer = setTimeout(poll, 1000);

    return () => {
      cancelled = true;

      if (timer !== undefined) {
        clearTimeout(timer);
      }
    };
  }, [
    pollingRunId,
    pollingStatus,
    pollingProjectId,
    loadRunData,
    isCurrentSession,
  ]);

  const validationProjectId = project?.id;
  const validationProjectMode = project?.analysis_mode;

  const triggerValidation = useCallback(async (
    mode?: "vegetation" | "wetland"
  ) => {
    if (
      !validationProjectId ||
      activeProjectIdRef.current !== validationProjectId
    ) return;

    const session = projectSessionRef.current;
    const request = ++validationRequestRef.current;

    const isCurrentRequest = () =>
      isCurrentSession(validationProjectId, session) &&
      validationRequestRef.current === request;

    try {
      setLoading(true);
      setError(null);

      const resolvedMode =
        mode ??
        (validationProjectMode === "vegetation"
          ? "vegetation"
          : "wetland");

      const val = await validateDatasets(
        validationProjectId,
        resolvedMode
      );

      if (!isCurrentRequest()) return;

      setValidation(val);

      setProject((prev) =>
        prev?.id === validationProjectId
          ? {
              ...prev,
              analysis_mode: resolvedMode,
              datasets_ready: val.valid,
              status: val.valid ? "READY" : "INVALID",
            }
          : prev
      );
    } catch (err: unknown) {
      if (!isCurrentRequest()) return;

      setError(
        err instanceof Error
          ? err.message
          : "Validation failed"
      );
    } finally {
      if (isCurrentRequest()) {
        setLoading(false);
      }
    }
  }, [
    validationProjectId,
    validationProjectMode,
    isCurrentSession,
  ]);

  const uploadBand = async (
    timePoint: "t1" | "t2",
    band: "green" | "red" | "rededge" | "nir" | "boundary",
    file: File,
    mode?: "vegetation" | "wetland"
  ) => {
    if (!project) return;

    const pid = project.id;
    const session = projectSessionRef.current;

    const isCurrent = () =>
      isCurrentSession(pid, session);

    if (!isCurrent()) return;

    try {
      setLoading(true);
      setError(null);

      await uploadBandFile(pid, timePoint, band, file);

      if (!isCurrent()) return;

      await triggerValidation(mode);
    } catch (err: unknown) {
      if (isCurrent()) {
        setError(
          err instanceof Error
            ? err.message
            : "Upload failed"
        );
      }
    } finally {
      if (isCurrent()) {
        setLoading(false);
      }
    }
  };

  // ------------------------------------------------------------
  // Manual exclusion management
  // ------------------------------------------------------------

  const addManualExclusion = (exclusion: ManualExclusion) => {
    setManualExclusions((current) => [...current, exclusion]);
  };

  const updateManualExclusion = (
    updatedExclusion: ManualExclusion
  ) => {
    setManualExclusions((current) =>
      current.map((exclusion) =>
        exclusion.id === updatedExclusion.id
          ? updatedExclusion
          : exclusion
      )
    );
  };

  const removeManualExclusion = (id: string) => {
    setManualExclusions((current) =>
      current.filter((exclusion) => exclusion.id !== id)
    );
  };

  const clearManualExclusions = () => {
    setManualExclusions([]);
  };

  // ------------------------------------------------------------
  // Preparation / staging workflow
  // ------------------------------------------------------------

  const preparationProjectId = project?.id;

  const fetchPreparation = useCallback(
    async (projectId?: string): Promise<PreparationState | undefined> => {
      const pid = projectId ?? preparationProjectId;

      if (
        !pid ||
        activeProjectIdRef.current !== pid ||
        preparationMutationRef.current !== null
      ) {
        return undefined;
      }

      const session = projectSessionRef.current;
      const request = ++preparationRequestRef.current;

      const isCurrent = () =>
        isCurrentSession(pid, session) &&
        preparationRequestRef.current === request;

      try {
        const state = await getPreparation(pid);

        if (!isCurrent()) return undefined;

        setPreparation(state);

        return state;
      } catch (err) {
        if (isCurrent()) {
          console.warn("fetchPreparation failed:", err);
        }

        return undefined;
      }
    },
    [preparationProjectId, isCurrentSession]
  );

  const mutatePreparation = useCallback(async (
    projectId: string,
    operation: () => Promise<PreparationState>,
    failureMessage: string,
    onSuccess?: () => void
  ) => {
    if (activeProjectIdRef.current !== projectId) {
      return;
    }

    if (
      resettingProjectsRef.current.has(projectId) ||
      preparationMutationRef.current !== null
    ) {
      setError("Another preparation operation is in progress.");
      throw new Error("Preparation operation already in progress.");
    }

    const session = projectSessionRef.current;
    const request = ++preparationRequestRef.current;

    preparationMutationRef.current = request;

    const isCurrent = () =>
      isCurrentSession(projectId, session) &&
      preparationRequestRef.current === request;

    try {
      setError(null);

      const state = await operation();

      if (!isCurrent()) return;

      setPreparation(state);

      onSuccess?.();
    } catch (err: unknown) {
      if (!isCurrent()) return;

      setError(
        err instanceof Error
          ? err.message
          : failureMessage
      );

      throw err;
    } finally {
      if (
        isCurrentSession(projectId, session) &&
        preparationMutationRef.current === request
      ) {
        preparationMutationRef.current = null;
      }
    }
  }, [isCurrentSession]);

  const saveDraft = useCallback(
    async (
      inclusion: Record<string, unknown> | null,
      exclusions: ManualExclusion[]
    ) => {
      if (!preparationProjectId) return;

      const pid = preparationProjectId;

      await mutatePreparation(
        pid,
        () => savePreparationDraft(pid, inclusion, exclusions),
        "Failed to save draft",
        () => setManualExclusions(exclusions)
      );
    },
    [preparationProjectId, mutatePreparation]
  );

  const commitArea = useCallback(async () => {
    if (!preparationProjectId) return;

    const pid = preparationProjectId;

    await mutatePreparation(
      pid,
      () => commitPreparation(pid),
      "Failed to commit preparation",
      () => {
        setProject((prev) =>
          prev?.id === pid
            ? {
                ...prev,
                preparation_status: "COMMITTED",
                is_locked: true,
              }
            : prev
        );
      }
    );
  }, [preparationProjectId, mutatePreparation]);

  const unlockArea = useCallback(async () => {
    if (!preparationProjectId) return;

    const pid = preparationProjectId;

    await mutatePreparation(
      pid,
      () => unlockPreparation(pid),
      "Failed to unlock preparation",
      () => {
        setProject((prev) =>
          prev?.id === pid
            ? {
                ...prev,
                preparation_status: "DRAFT",
                is_locked: false,
              }
            : prev
        );
      }
    );
  }, [preparationProjectId, mutatePreparation]);

  // ------------------------------------------------------------
  // Switch between existing projects
  // ------------------------------------------------------------

  const switchProject = async (projectId: string) => {
    const selectedProject = projects.find(
      (item) => item.id === projectId
    );

    if (!selectedProject) {
      setError("Selected project could not be found.");
      return;
    }

    if (activeProjectIdRef.current === projectId) return;

    activateProjectSession(projectId);

    const session = projectSessionRef.current;

    const isCurrent = () =>
      isCurrentSession(projectId, session);

    try {
      setLoading(true);
      setError(null);

      setValidation(null);
      setRunStatus(null);
      setResults(null);
      setLayers([]);
      setManualExclusions([]);
      setPreparation(null);
      setActiveLayerId("");

      setProject(selectedProject);

      const preparationRequest =
        ++preparationRequestRef.current;

      try {
        const state = await getPreparation(projectId);

        if (!isCurrent()) return;

        if (
          preparationRequestRef.current === preparationRequest
        ) {
          setPreparation(state);
        }
      } catch (err) {
        if (isCurrent()) {
          console.warn(
            "Could not restore project preparation:",
            err
          );
        }
      }

      if (!isCurrent()) return;

      if (selectedProject.latest_run_id) {
        try {
          const existingRun = await getRunStatus(
              selectedProject.latest_run_id,
              selectedProject.id
            );

          if (!isCurrent()) return;

          if (
            existingRun.run_id !== selectedProject.latest_run_id ||
              existingRun.project_id !== selectedProject.id
          ) {
            throw new Error("Unexpected restored run identity.");
          }

          setRunStatus(existingRun);

          if (existingRun.status === "COMPLETED") {
            await loadRunData(
              existingRun.run_id,
              projectId,
              session
            );
          }
        } catch (err) {
          if (isCurrent()) {
            console.warn(
              "Could not restore the project's latest run:",
              err
            );
          }
        }
      }
    } catch (err: unknown) {
      if (isCurrent()) {
        setError(
          err instanceof Error
            ? err.message
            : "Failed to switch project."
        );
      }
    } finally {
      if (isCurrent()) {
        setLoading(false);
      }
    }
  };

  // ------------------------------------------------------------
  // Create a new project
  // ------------------------------------------------------------

  const createNewProject = async (
    name: string,
    description: string,
    aoiName: string
  ) => {
    if (creationPendingRef.current) {
      throw new Error("Project creation already in progress.");
    }

    creationPendingRef.current = true;

    let session = projectSessionRef.current;

    const isCurrent = () =>
      projectSessionRef.current === session;

    try {
      setLoading(true);
      setError(null);

      const newProject = await createProject(
        name.trim(),
        description.trim(),
        aoiName.trim()
      );

      setProjects((current) =>
        current.some((item) => item.id === newProject.id)
          ? current
          : [...current, newProject]
      );

      if (!isCurrent()) return;

      activateProjectSession(newProject.id);

      session = projectSessionRef.current;

      setValidation(null);
      setRunStatus(null);
      setResults(null);
      setLayers([]);
      setManualExclusions([]);
      setPreparation(null);
      setActiveLayerId("");

      setProject(newProject);

      const request = ++preparationRequestRef.current;

      try {
        const state = await getPreparation(newProject.id);

        if (
          isCurrent() &&
          preparationRequestRef.current === request
        ) {
          setPreparation(state);
        }
      } catch (err) {
        if (isCurrent()) {
          console.warn(
            "Could not load new project preparation:",
            err
          );
        }
      }
    } catch (err: unknown) {
      if (isCurrent()) {
        setError(
          err instanceof Error
            ? err.message
            : "Failed to create project."
        );
      }

      throw err;
    } finally {
      creationPendingRef.current = false;

      if (isCurrent()) {
        setLoading(false);
      }
    }
  };

  // ------------------------------------------------------------
  // Clear all analysis data and reset the current project session
  // ------------------------------------------------------------

  const clearAll = async () => {
    if (!project) return;

    const pid = project.id;

    let session = projectSessionRef.current;

    const isCurrent = () =>
      isCurrentSession(pid, session);

    if (!isCurrent()) return;

    if (
      resettingProjectsRef.current.has(pid) ||
      runLaunchesRef.current.has(pid) ||
      preparationMutationRef.current !== null ||
      (
        runStatus &&
        !["COMPLETED", "FAILED"].includes(runStatus.status)
      )
    ) {
      setError(
        "Wait for the current project operation to finish."
      );
      return;
    }

    resettingProjectsRef.current.add(pid);

    try {
      setLoading(true);
      setError(null);

      const reset = await resetProject(pid);

      if (reset.id !== pid) {
        throw new Error("Unexpected reset project identity.");
      }

      setProjects((current) =>
        current.map((item) =>
          item.id === pid ? reset : item
        )
      );

      if (!isCurrent()) return;

      activateProjectSession(pid);

      session = projectSessionRef.current;

      setProject(reset);
      setValidation(null);
      setRunStatus(null);
      setResults(null);
      setLayers([]);
      setManualExclusions([]);
      setPreparation(null);
      setActiveLayerId("");

    } catch (err: unknown) {
      if (isCurrent()) {
        setError(
          err instanceof Error
            ? err.message
            : "Failed to clear project session"
        );
      }
    } finally {
      resettingProjectsRef.current.delete(pid);

      if (isCurrent()) {
        setLoading(false);
      }
    }
  };

  // ------------------------------------------------------------
  // Launch analysis
  // ------------------------------------------------------------

  const triggerRun = async (
    config?: Record<string, unknown>
  ) => {
    if (!project) return;

    const pid = project.id;
    const session = projectSessionRef.current;

    const isCurrent = () =>
      isCurrentSession(pid, session);

    if (!isCurrent()) return;

    if (
      resettingProjectsRef.current.has(pid) ||
      runLaunchesRef.current.has(pid) ||
      (
        runStatus &&
        !["COMPLETED", "FAILED"].includes(runStatus.status)
      )
    ) {
      setError("An analysis is already in progress.");
      return;
    }

    runLaunchesRef.current.add(pid);

    try {
      setLoading(true);
      setError(null);

      const runConfig = {
        ...(config || {}),
        manual_exclusions: manualExclusions,
        inclusion_geometry: preparation?.inclusion ?? null,
      };

      const initialRun = await launchRun(pid, runConfig);

      if (!isCurrent()) return;

      if (!initialRun.run_id) {
        throw new Error("Backend returned an invalid run ID.");
      }

      resultRequestRef.current += 1;

      setResults(null);
      setLayers([]);
      setActiveLayerId("");
      setRunStatus(initialRun);

      setProject((prev) =>
        prev?.id === pid
          ? {
              ...prev,
              latest_run_id: initialRun.run_id,
            }
          : prev
      );

      setProjects((current) =>
        current.map((item) =>
          item.id === pid
            ? {
                ...item,
                latest_run_id: initialRun.run_id,
              }
            : item
        )
      );

      if (initialRun.status === "COMPLETED") {
        await loadRunData(
          initialRun.run_id,
          pid,
          session
        );
      }
    } catch (err: unknown) {
      if (isCurrent()) {
        setError(
          err instanceof Error
            ? err.message
            : "Run failed"
        );
      }
    } finally {
      runLaunchesRef.current.delete(pid);

      if (isCurrent()) {
        setLoading(false);
      }
    }
  };

  const refreshResults = async () => {
    if (runStatus?.run_id) {
      await loadRunData(runStatus.run_id);
    }
  };

  return (
    <ProjectContext.Provider
      value={{
        project,
        projects,
        createNewProject,
        switchProject,
        validation,
        runStatus,
        results,
        layers,
        activeLayerId,
        loading,
        error,

        manualExclusions,
        preparation,

        setActiveLayerId,
        triggerValidation,
        uploadBand,
        triggerRun,
        refreshResults,
        updateManualExclusion,
        addManualExclusion,
        removeManualExclusion,
        clearManualExclusions,
        clearAll,
        fetchPreparation,
        saveDraft,
        commitArea,
        unlockArea,
      }}
    >
      {children}
    </ProjectContext.Provider>
  );
};