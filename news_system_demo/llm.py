"""LangChain-backed OpenRouter client used by the didactic demo.

Este módulo encapsula OpenRouter detrás de LangChain y la validación estricta
de respuestas estructuradas. Si el modelo falla, la demo aborta de forma
explícita.
"""

from __future__ import annotations

import os
from typing import Any, Protocol, TypeVar

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from pydantic import BaseModel, SecretStr

DEFAULT_CHAT_MODEL = "deepseek/deepseek-v4-flash"
ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


class LlmClientProtocol(Protocol):
    """Subset of LLM behavior used by the demo workflow."""

    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[Any],
        temperature: float = 0.2,
    ) -> Any:
        """Return a structured JSON completion."""


class OpenRouterClient:
    """Small async LangChain adapter dedicated to the educational demo."""

    def __init__(
        self,
        *,
        env_path: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        """Initialize the demo client from environment variables or explicit args."""

        # Si nos pasan un `.env`, lo cargamos para que la demo sea autónoma y
        # no dependa de variables exportadas globalmente.
        if env_path is not None:
            load_dotenv(env_path)
        api_key_value = (
            api_key
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPENROUTER_KEY")
        )
        self.model = (
            model
            or os.getenv("DEMO_OPENROUTER_MODEL")
            or os.getenv("OPENROUTER_MODEL")
            or DEFAULT_CHAT_MODEL
        )
        self.timeout_seconds = timeout_seconds
        if not api_key_value:
            raise ValueError(
                "OPENROUTER_API_KEY is required for the news_system_demo workflow "
                "(OPENROUTER_KEY is still accepted for compatibility)."
            )
        self.api_key = api_key_value
        self._models: dict[float, ChatOpenRouter] = {}

    def _build_model(self, *, temperature: float) -> ChatOpenRouter:
        """Create a LangChain OpenRouter chat model for one temperature."""

        return ChatOpenRouter(
            model=self.model,
            api_key=SecretStr(self.api_key),
            temperature=temperature,
            timeout=int(self.timeout_seconds * 1000),
            max_retries=2,
            app_url="https://localhost/news-system-demo",
            app_title="news-system-demo",
            openrouter_provider={"require_parameters": True},
        )

    def _get_model(self, *, temperature: float) -> ChatOpenRouter:
        """Return a cached LangChain model for the requested temperature."""

        if temperature not in self._models:
            self._models[temperature] = self._build_model(temperature=temperature)
        return self._models[temperature]

    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseModelT],
        temperature: float = 0.2,
    ) -> ResponseModelT:
        """Request structured JSON and validate it against a response model."""

        try:
            model = self._get_model(temperature=temperature)
            structured_model = model.with_structured_output(
                response_model,
                method="json_schema",
                strict=True,
            )
            response = await structured_model.ainvoke(
                [
                    ("system", system_prompt),
                    ("human", user_prompt),
                ]
            )
            if not isinstance(response, response_model):
                response = response_model.model_validate(response)
            return response
        except Exception as exc:
            raise RuntimeError(
                f"OpenRouter completion failed for model {self.model}: {exc}"
            ) from exc
