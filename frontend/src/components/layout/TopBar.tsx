import { useState } from "react";
import { useProject } from "../../context/useProject";

type TopBarProps = {
  activeItem: string;
};

export default function TopBar({ activeItem }: TopBarProps) {
  const {
    projects,
    project,
    validation,
    clearAll,
    loading,
    switchProject,
    createNewProject,
  } = useProject();
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [showNewProject, setShowNewProject] = useState(false);
  const [projectName, setProjectName] = useState("");
  const [aoiName, setAoiName] = useState("");
  const [createError, setCreateError] = useState<string | null>(null);

  const isReady = project?.datasets_ready || validation?.valid;

  const handleClearAll = async () => {
    try {
      await clearAll();
      setShowClearConfirm(false);
    } catch (err) {
      console.error("Failed to clear session:", err);
    }
  };

  const closeNewProject = () => {
    if (loading) return;
    setShowNewProject(false);
    setProjectName("");
    setAoiName("");
    setCreateError(null);
  };

  const handleCreateProject = async (event: React.FormEvent) => {
    event.preventDefault();
    const name = projectName.trim();
    const aoi = aoiName.trim();
    if (!name || !aoi) {
      setCreateError("Project name and AOI name are required.");
      return;
    }

    try {
      setCreateError(null);
      await createNewProject({
        name,
        aoiName: aoi,
      });
      closeNewProject();
    } catch (err: unknown) {
      setCreateError(
        err instanceof Error ? err.message : "Failed to create project"
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

        <div className="flex items-center gap-5">
          <div className="min-w-56">
            <div className="text-[10px] uppercase tracking-[0.12em] text-[var(--text-secondary)]">
              Project
            </div>

            <div className="mt-1 flex items-center gap-2">
              <select
                aria-label="Active project"
                value={project?.id ?? ""}
                onChange={(event) => switchProject(event.target.value)}
                disabled={loading || projects.length === 0}
                className="min-w-52 rounded-md border border-[var(--border)] bg-[var(--background)] px-2.5 py-1.5 text-xs font-medium text-[var(--text)] outline-none focus:border-[var(--accent)] disabled:opacity-50"
              >
                {projects.length === 0 && (
                  <option value="">No project loaded</option>
                )}
                {projects.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name} — {item.aoi_name || "Unnamed AOI"}
                  </option>
                ))}
              </select>

              <button
                type="button"
                onClick={() => setShowNewProject(true)}
                disabled={loading}
                className="whitespace-nowrap rounded-md bg-[var(--accent)] px-3 py-1.5 text-xs font-medium text-white hover:opacity-90 disabled:opacity-50"
              >
                + New Project
              </button>
            </div>
          </div>

          <div className="h-8 w-px bg-[var(--border)]" />

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

      {showNewProject && (
        <div className="fixed inset-0 z-[1000] flex items-center justify-center bg-black/50 px-4">
          <form
            onSubmit={handleCreateProject}
            className="w-full max-w-lg rounded-lg border border-[var(--border)] bg-[var(--surface)] p-6 shadow-xl"
          >
            <div className="text-base font-semibold text-[var(--text)]">
              Create New Project
            </div>
            <p className="mt-2 text-sm text-[var(--text-secondary)]">
              Create an independent workspace for datasets, analysis runs and reports.
            </p>

            <div className="mt-5 space-y-4">
              <label className="block">
                <span className="text-xs font-medium text-[var(--text)]">
                  Project name <span className="text-red-500">*</span>
                </span>
                <input
                  autoFocus
                  value={projectName}
                  onChange={(event) => setProjectName(event.target.value)}
                  placeholder="Coastal Wetland Assessment"
                  className="mt-1.5 w-full rounded-md border border-[var(--border)] bg-[var(--background)] px-3 py-2 text-sm text-[var(--text)] outline-none focus:border-[var(--accent)]"
                />
              </label>

              <label className="block">
                <span className="text-xs font-medium text-[var(--text)]">
                  AOI name <span className="text-red-500">*</span>
                </span>
                <input
                  value={aoiName}
                  onChange={(event) => setAoiName(event.target.value)}
                  placeholder="Central Reserve"
                  className="mt-1.5 w-full rounded-md border border-[var(--border)] bg-[var(--background)] px-3 py-2 text-sm text-[var(--text)] outline-none focus:border-[var(--accent)]"
                />
              </label>

              {createError && (
                <div className="rounded-md border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-500">
                  {createError}
                </div>
              )}
            </div>

            <div className="mt-6 flex justify-end gap-3">
              <button
                type="button"
                onClick={closeNewProject}
                disabled={loading}
                className="rounded-md border border-[var(--border)] px-4 py-2 text-xs font-medium text-[var(--text-secondary)] hover:text-[var(--text)] disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={loading}
                className="rounded-md bg-[var(--accent)] px-4 py-2 text-xs font-medium text-white hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading ? "Creating..." : "Create Project"}
              </button>
            </div>
          </form>
        </div>
      )}
    </>
  );
}
