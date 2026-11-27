import logging
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from openai import APIError

from app.assistant_service import (
    generate_assistant_response,
    OPENROUTER_MODEL,
)
from app.assistant_context_service import build_assistant_context

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/assistant",
    tags=["AETHRA Assistant"],
)


class AssistantChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question asked by the user.",
    )
    project_id: str | None = Field(
        default=None,
        max_length=128,
        description="Currently selected AETHRA project ID.",
    )
    run_id: str | None = Field(
        default=None,
        max_length=128,
        description="Optional analysis run ID.",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be empty.")
        return value

    @field_validator("project_id", "run_id")
    @classmethod
    def validate_optional_ids(cls, value: str | None) -> str | None:
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("Project and run IDs cannot be empty.")
        return value


class AssistantChatResponse(BaseModel):
    answer: str
    model: str
    status: str


def _assistant_configuration():
    """Report local configuration only; never contact the AI provider here."""
    api_key_configured = bool(os.getenv("OPENROUTER_API_KEY", "").strip())
    model_configured = bool(str(OPENROUTER_MODEL or "").strip())
    configured = api_key_configured and model_configured

    return {
        "status": "READY" if configured else "NOT_CONFIGURED",
        "configured": configured,
        "model": OPENROUTER_MODEL if model_configured else None,
        "message": (
            "Assistant configuration is present."
            if configured
            else "OpenRouter API key or model configuration is missing."
        ),
    }


@router.get("/status")
def assistant_status():
    """
    Report whether the backend has the local settings needed for AI requests.

    READY means the API key and model are configured locally. It does not
    verify provider uptime or make a billable/external provider request.
    """
    return _assistant_configuration()


@router.post("/chat", response_model=AssistantChatResponse)
def assistant_chat(request: AssistantChatRequest):
    """
    Accept a user question, retrieve verified project context,
    and generate an OpenRouter response.
    """
    try:
        context = None
        if request.project_id:
            context = build_assistant_context(
                project_id=request.project_id,
                run_id=request.run_id,
            )

        answer = generate_assistant_response(
            question=request.question,
            context=context,
        )

        return AssistantChatResponse(
            answer=answer,
            model=OPENROUTER_MODEL,
            status="SUCCESS",
        )

    except RuntimeError:
        logger.exception("Assistant configuration or response error.")
        raise HTTPException(
            status_code=503,
            detail="The AI Assistant is not currently configured or available.",
        )
    except APIError:
        logger.exception("OpenRouter API request failed.")
        raise HTTPException(
            status_code=502,
            detail="The AI service could not complete your request.",
        )
    except Exception:
        logger.exception("Unexpected Assistant error.")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred in the AI Assistant.",
        )
