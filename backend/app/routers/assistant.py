
import logging

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
        description="Question asked by the user."
    )

    project_id: str | None = Field(
        default=None,
        max_length=128,
        description="Currently selected AETHRA project ID."
    )

    run_id: str | None = Field(
        default=None,
        max_length=128,
        description="Optional analysis run ID."
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


@router.get("/status")
def assistant_status():
    """
    Check whether the Assistant backend is reachable.
    """
    return {
        "status": "INITIALIZED",
        "feature": "AETHRA Assistant",
        "message": "Assistant backend is reachable."
    }


@router.post("/chat", response_model=AssistantChatResponse)
def assistant_chat(request: AssistantChatRequest):
    """
    Accept a user question, retrieve verified project context,
    and generate an OpenRouter response.
    """

    try:
        # Build context only when a project ID is supplied.
        context = None

        if request.project_id:
            context = build_assistant_context(
                project_id=request.project_id,
                run_id=request.run_id,
            )

        # Generate the answer using the question and project context.
        answer = generate_assistant_response(
            question=request.question,
            context=context,
        )

        return AssistantChatResponse(
            answer=answer,
            model=OPENROUTER_MODEL,
            status="SUCCESS"
        )

    except RuntimeError:
        logger.exception("Assistant configuration or response error.")

        raise HTTPException(
            status_code=503,
            detail="The AI Assistant is not currently configured or available."
        )

    except APIError:
        logger.exception("OpenRouter API request failed.")

        raise HTTPException(
            status_code=502,
            detail="The AI service could not complete your request."
        )

    except Exception:
        logger.exception("Unexpected Assistant error.")

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred in the AI Assistant."
        )
