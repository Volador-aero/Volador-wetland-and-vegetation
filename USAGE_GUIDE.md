# AETHRA AI Assistant – Usage Guide

## Purpose
The AETHRA AI Assistant provides a conversational interface for asking questions about GIS concepts and information available in the selected AETHRA project. It explains available project context and analysis outputs. The GIS engine remains the source of scientific calculations.

## Before using the Assistant
1. Start the AETHRA backend and frontend using the verified setup instructions in `README.md`.
2. Ensure the backend environment contains a valid `OPENROUTER_API_KEY` and the configured `OPENROUTER_MODEL`.
3. Ensure the frontend points to the correct backend through `VITE_API_BASE_URL`, if a non-default address is used.
4. Open the AETHRA application in a browser.

## Basic workflow
1. Open the AETHRA application.
2. Select or open the project you want to work with.
3. Navigate to the AI Assistant section.
4. Enter a clear question in the chat input and submit it.
5. Review the generated response and the project information it references.
6. Ask a follow-up question if clarification is needed.

## Example questions
- "What is NDVI?"
- "Explain the difference between NDVI and NDWI."
- "What is the current status of the selected project?"
- "Are NDVI values available for this project?"

Project-specific answers depend on the information present in the selected project and its analysis outputs.

## Understanding status and errors
- `READY` indicates that the required Assistant configuration is present. It does not guarantee that OpenRouter is reachable or that a response will always be generated.
- A not-configured status means the required API key or model configuration is missing.
- A service-unavailable or connection message indicates that the Assistant backend or external provider could not complete the request. Check the backend process, environment configuration and network connection.
- If project data or analysis metrics are missing, the Assistant may report that the information is unavailable. Do not treat missing values as zero or as a completed analysis result.

## Good practices
- Select the correct project before asking project-specific questions.
- Ask focused questions and specify the metric or analysis stage.
- Check important scientific conclusions against AETHRA's original analysis outputs and reports.
- Do not use the Assistant as a replacement for GIS validation or scientific review.
- Do not enter secrets or sensitive information into chat prompts.

## Current limitations
- Responses depend on available project context and external model availability.
- The Assistant should not be assumed to know metrics absent from project data.
- Code-block rendering was not browser-tested in the recorded validation.
- The status endpoint reports configuration state, not guaranteed provider uptime.
