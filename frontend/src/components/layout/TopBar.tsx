
import { useState } from "react";
import { useProject } from "../../context/useProject";

type TopBarProps = {
  activeItem: string;
};

export default function TopBar({ activeItem }: TopBarProps) {
  const {
    project,
    projects,
    validation,
    clearAll,
    loading,
    createNewProject,
    switchProject,
  } = useProject();

  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [showNewProject, setShowNewProject] = useState(false);
  const [showProjectList, setShowProjectList] = useState(false);

  const [projectName, setProjectName] = useState("");
  const [description, setDescription] = useState("");
  const [aoiName, setAoiName] = useState("");
  const [formError, setFormError] = useState("");

  const isReady = project?.datasets_ready || validation?.valid;

  const handleClearAll = async () => {
    try {
      await clearAll();
      setShowClearConfirm(false);
    } catch (err) {
      console.error("Failed to clear session:", err);
    }
  };

  const handleCreateProject = async (
    event: React.FormEvent<HTMLFormElement>
  ) => {
    event.preventDefault();
    setFormError("");

    if (!projectName.trim() || !description.trim() || !aoiName.trim()) {
      setFormError("Please complete all three fields.");
      return;
    }

    try {
      await createNewProject(projectName, description, aoiName);

      setProjectName("");
      setDescription("");
      setAoiName("");
      setShowNewProject(false);
    } catch (err) {
      setFormError(
        err instanceof Error
          ? err.message
          : "Unable to create the project. Please try again."
      );
    }
  };

  return (
    <>
      <header className="flex h-16 shrink-0 items-center justify-between border-b border-[var(--border)] bg-[var(--surface)] px-7">
        <div>
          <div className="text-xs text-[var(--text-secondary)]">
            Wetland Change Analysis
          </div>

          <div className="mt-0.5 text-sm font-medium text-[var(--text)]">
            {activeItem}
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Active Project Dropdown */}
          <div className="relative flex flex-col">
            <label
              htmlFor="project-selector"
              className="mb-1 text-[10px] uppercase tracking-[0.12em] text-[var(--text-secondary)]"
            >
              Active Project
            </label>

            <button
              id="project-selector"
              type="button"
              onClick={() => setShowProjectList((prev) => !prev)}
              disabled={loading || projects.length === 0}
              aria-expanded={showProjectList}
              aria-haspopup="listbox"
              className="flex max-w-[230px] items-center justify-between gap-3 rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-left text-xs font-medium text-[var(--text)] disabled:opacity-50"
            >
              <span className="truncate">
                {project?.name || "Select Project"}
              </span>

              <span aria-hidden="true">
                {showProjectList ? "▴" : "▾"}
              </span>
            </button>

            {showProjectList && (
              <div
                role="listbox"
                aria-label="Select a project"
                className="absolute right-0 top-full z-[100] mt-2 max-h-64 w-80 overflow-y-auto rounded-lg border border-[var(--border)] bg-[var(--surface)] shadow-xl"
              >
                {projects.map((item) => {
                  const isActive = project?.id === item.id;

                  return (
                    <button
                      key={item.id}
                      type="button"
                      role="option"
                      aria-selected={isActive}
                      onClick={() => {
                        setShowProjectList(false);

                        if (!isActive) {
                          void switchProject(item.id);
                        }
                      }}
                      className={`block w-full border-b border-[var(--border)] px-3 py-3 text-left transition hover:bg-[var(--background)] ${
                        isActive ? "bg-[var(--background)]" : ""
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="truncate text-xs font-semibold text-[var(--text)]">
                          {item.name}
                        </span>

                        {isActive && (
                          <span className="shrink-0 text-[10px] font-medium text-[var(--success)]">
                            ACTIVE
                          </span>
                        )}
                      </div>

                      <div className="mt-1 text-[10px] text-[var(--text-secondary)]">
                        {item.aoi_name || "Unnamed Area"} -{" "}
                        {item.analysis_mode || "Unknown"}
                      </div>
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {/* New Project Button */}
          <button
            type="button"
            onClick={() => {
              setFormError("");
              setShowNewProject(true);
            }}
            disabled={loading}
            className="mt-4 rounded-md bg-[var(--text)] px-3 py-2 text-xs font-medium text-[var(--surface)] transition hover:opacity-80 disabled:cursor-not-allowed disabled:opacity-50"
          >
            + New Project
          </button>

          <div className="h-8 w-px bg-[var(--border)]" />

          {/* Project Readiness */}
          <div className="flex items-center gap-2 text-xs text-[var(--text-secondary)]">
            <span
              className={`h-2 w-2 rounded-full ${
                isReady ? "bg-[var(--success)]" : "bg-amber-500"
              }`}
            />

            {project
              ? isReady
                ? "READY"
                : "CREATED"
              : "NO PROJECT"}
          </div>

          <div className="h-8 w-px bg-[var(--border)]" />

          {/* Clear All Button */}
          <button
            type="button"
            onClick={() => setShowClearConfirm(true)}
            disabled={!project || loading}
            className="rounded-md border border-[var(--border)] px-3 py-1.5 text-xs font-medium text-[var(--text-secondary)] transition hover:border-red-400 hover:text-red-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Clear All
          </button>
        </div>
      </header>

      {/* New Project Modal */}
      {showNewProject && (
        <div className="fixed inset-0 z-[1000] flex items-center justify-center bg-black/40 px-4">
          <div className="w-full max-w-lg rounded-lg border border-[var(--border)] bg-[var(--surface)] p-6 shadow-xl">
            <h2 className="text-lg font-semibold text-[var(--text)]">
              Create New Project
            </h2>

            <p className="mt-2 text-sm leading-6 text-[var(--text-secondary)]">
              Create a separate workspace for a new wetland study.
              Your existing projects and their data will remain available.
            </p>

            <form onSubmit={handleCreateProject} className="mt-5 space-y-4">
              <div>
                <label
                  htmlFor="new-project-name"
                  className="mb-1.5 block text-xs font-medium text-[var(--text)]"
                >
                  Project Name
                </label>

                <input
                  id="new-project-name"
                  type="text"
                  value={projectName}
                  onChange={(event) => setProjectName(event.target.value)}
                  placeholder="e.g. Wetland Degradation Study"
                  required
                  maxLength={100}
                  disabled={loading}
                  className="w-full rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm text-[var(--text)] outline-none focus:border-blue-500"
                />
              </div>

              <div>
                <label
                  htmlFor="new-project-description"
                  className="mb-1.5 block text-xs font-medium text-[var(--text)]"
                >
                  Description
                </label>

                <textarea
                  id="new-project-description"
                  value={description}
                  onChange={(event) => setDescription(event.target.value)}
                  placeholder="Describe the purpose of this study"
                  required
                  maxLength={500}
                  rows={3}
                  disabled={loading}
                  className="w-full resize-y rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm text-[var(--text)] outline-none focus:border-blue-500"
                />
              </div>

              <div>
                <label
                  htmlFor="new-project-aoi"
                  className="mb-1.5 block text-xs font-medium text-[var(--text)]"
                >
                  Study Area Name (AOI)
                </label>

                <input
                  id="new-project-aoi"
                  type="text"
                  value={aoiName}
                  onChange={(event) => setAoiName(event.target.value)}
                  placeholder="e.g. Central Reserve"
                  required
                  maxLength={150}
                  disabled={loading}
                  className="w-full rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm text-[var(--text)] outline-none focus:border-blue-500"
                />
              </div>

              {formError && (
                <p className="text-xs text-red-500" role="alert">
                  {formError}
                </p>
              )}

              <div className="flex justify-end gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => {
                    setShowNewProject(false);
                    setFormError("");
                  }}
                  disabled={loading}
                  className="rounded-md border border-[var(--border)] px-4 py-2 text-xs font-medium text-[var(--text-secondary)] hover:text-[var(--text)] disabled:opacity-50"
                >
                  Cancel
                </button>

                <button
                  type="submit"
                  disabled={loading}
                  className="rounded-md bg-blue-600 px-4 py-2 text-xs font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {loading ? "Creating..." : "Create Project"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Clear All Confirmation */}
      {showClearConfirm && (
        <div className="fixed inset-0 z-[1000] flex items-center justify-center bg-black/40 px-4">
          <div className="w-full max-w-md rounded-lg border border-[var(--border)] bg-[var(--surface)] p-6 shadow-xl">
            <div className="text-base font-semibold text-[var(--text)]">
              Clear current session?
            </div>

            <p className="mt-3 text-sm leading-6 text-[var(--text-secondary)]">
              This will clear uploaded datasets, validation state,
              analysis runs and results, and manual exclusions.
              Your project will remain available.
            </p>

            <p className="mt-2 text-xs text-red-500">
              This action cannot be undone.
            </p>

            <div className="mt-6 flex justify-end gap-3">
              <button
                type="button"
                onClick={() => setShowClearConfirm(false)}
                disabled={loading}
                className="rounded-md border border-[var(--border)] px-4 py-2 text-xs font-medium text-[var(--text-secondary)] hover:text-[var(--text)]"
              >
                Cancel
              </button>

              <button
                type="button"
                onClick={handleClearAll}
                disabled={loading}
                className="rounded-md bg-red-600 px-4 py-2 text-xs font-medium text-white hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading ? "Clearing..." : "Clear All"}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
