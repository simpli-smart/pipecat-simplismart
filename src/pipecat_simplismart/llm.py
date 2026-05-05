"""OpenAI-compatible LLM service for Simplismart inference."""

from dataclasses import dataclass

from openai import NOT_GIVEN

from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService


@dataclass
class SimplismartLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for SimplismartLLMService."""


class SimplismartLLMService(OpenAILLMService):
    """LLM service using Simplismart's OpenAI-compatible Chat Completions API."""

    Settings = SimplismartLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.simplismart.live",
        model: str | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize Simplismart LLM with API key and optional model override.

        Args:
            api_key: Simplismart API key (JWT bearer token).
            base_url: API base URL (no trailing path; client uses ``/v1/...``).
            model: Deprecated convenience override; prefer ``settings=``.
            settings: Runtime-updatable LLM settings (merged into defaults).
            **kwargs: Passed to :class:`OpenAILLMService` (e.g. ``retry_timeout_secs``).
        """
        default_settings = self.Settings(
            model="google/gemma-4-31B-it",
            system_instruction=None,
            frequency_penalty=NOT_GIVEN,
            presence_penalty=NOT_GIVEN,
            seed=NOT_GIVEN,
            temperature=NOT_GIVEN,
            top_p=NOT_GIVEN,
            top_k=None,
            max_tokens=NOT_GIVEN,
            max_completion_tokens=NOT_GIVEN,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
            extra={},
        )
        if model is not None:
            default_settings.model = model
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            settings=default_settings,
            **kwargs,
        )
