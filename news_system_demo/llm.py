"""Minimal OpenRouter client used by the didactic demo."""

from __future__ import annotations

import json
import os
from typing import Any, TypeVar, cast

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel
from tenacity import AsyncRetrying, stop_after_attempt, wait_fixed

DEFAULT_CHAT_MODEL = "minimax/minimax-m2.7"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


def extract_text_content(content: Any) -> str:
    """Normalize an OpenRouter content payload into plain text."""

    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict)).strip()
    return str(content).strip()


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response."""

    stripped = raw_text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if "\n" in stripped:
            stripped = stripped.split("\n", maxsplit=1)[1]
        stripped = stripped.rsplit("```", maxsplit=1)[0].strip()

    try:
        return cast(dict[str, Any], json.loads(stripped))
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return cast(dict[str, Any], json.loads(stripped[start : end + 1]))


class DemoOpenRouterClient:
    """Small async client dedicated to the educational demo."""

    def __init__(
        self,
        *,
        env_path: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        """Initialize the demo client from environment variables or explicit args."""

        if env_path is not None:
            load_dotenv(env_path)
        self.api_key = api_key or os.getenv("OPENROUTER_KEY")
        self.model = (
            model
            or os.getenv("DEMO_OPENROUTER_MODEL")
            or os.getenv("OPENROUTER_MODEL")
            or DEFAULT_CHAT_MODEL
        )
        self.timeout_seconds = timeout_seconds
        if not self.api_key:
            raise ValueError("OPENROUTER_KEY is required for the news_system_demo workflow.")

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one chat completion request to OpenRouter."""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost/news-system-demo",
            "X-Title": "news-system-demo",
        }
        async for attempt in AsyncRetrying(stop=stop_after_attempt(3), wait=wait_fixed(2)):
            with attempt:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.post(
                        OPENROUTER_CHAT_URL,
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    return cast(dict[str, Any], response.json())
        raise RuntimeError("OpenRouter request exhausted retries in news_system_demo.")

    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseModelT],
        temperature: float = 0.2,
    ) -> ResponseModelT:
        """Request structured JSON and validate it against a response model."""

        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        response_payload = await self._post(payload)
        raw_text = extract_text_content(response_payload["choices"][0]["message"]["content"])
        parsed = extract_json_object(raw_text)
        return response_model.model_validate(parsed)
