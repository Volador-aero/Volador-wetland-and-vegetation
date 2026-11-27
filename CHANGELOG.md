# Changelog

This is a feature-oriented summary of the AETHRA AI Assistant work documented during the internship. It is not a complete commit-by-commit history.

## Initial implementation
- Added an Assistant API router and registered it with the FastAPI application.
- Added an initial Assistant status endpoint and its test.
- Created the `feature/aethra-ai-assistant` branch to isolate the enhancement.
- Reviewed existing project, dataset, analysis, results and reporting APIs before integration.

## LLM provider integration
- Explored Google Gemini during the initial implementation.
- Switched response generation to OpenRouter after the initial Gemini approach did not work as expected.
- Added project-context retrieval to provide available AETHRA project information to the Assistant.
- Retained the GIS engine as the source of scientific calculations; the Assistant explains available outputs rather than fabricating measurements.

## Reliability and security refinements
- Improved project path resolution and validation to restrict paths to the intended project storage area.
- Hardened context handling for invalid project IDs, missing project directories or datasets, malformed preparation data and incomplete analysis results.
- Added an Assistant availability/status endpoint to distinguish configured and unconfigured states.
- Improved handling of missing configuration, provider errors and empty responses.

## Frontend and response handling
- Added environment-based frontend API URL configuration through `VITE_API_BASE_URL`.
- Improved HTTP and network error messages and loading/status feedback.
- Improved Markdown response presentation, including headings, tables, lists and mathematical expressions.
- Added targeted API-client tests.

## Recorded validation
- Backend suite: 64 passed, 1 skipped.
- Frontend suite: 12/12 passed.
- Targeted API-client tests: 4/4 passed.
- Production build completed successfully.

## Known limitations
- One Windows symlink-related test was skipped.
- Code-block rendering was not browser-tested.
- Some path-safety cases involving uploaded datasets, legacy absolute paths and reset operations were not fully verified.
- An Assistant `READY` status indicates configuration is present; it does not guarantee that the external provider is currently available.
- The frontend build reported a non-blocking large-bundle warning.

For exact code changes and review history, see the feature branch and pull request linked in `README.md`.
