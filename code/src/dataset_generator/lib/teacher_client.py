from __future__ import annotations

import os
import time
from typing import Any, Dict

import httpx
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import TEACHER_SYSTEM_PROMPT, TEACHER_USER_PROMPT
from src.dataset_generator.models import TeacherResponse


class APIError(Exception):
    pass


class TeacherClient:
    def __init__(
        self,
        *,
        model_name: str,
        timeout: int,
        temperature: float,
        top_p: float,
        base_url: str,
        allow_fallbacks: bool,
        provider_only: list[str],
    ) -> None:
        load_dotenv()

        self._base_url = base_url
        self._headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}

        if (
            not self._headers["Authorization"].endswith(" ")
            and self._headers["Authorization"] == "Bearer "
        ):
            raise APIError(
                f"Missing API key in env var {os.environ['OPENROUTER_API_KEY']}"
            )

        self._client = httpx.AsyncClient(
            base_url=self._base_url, headers=self._headers, timeout=timeout
        )
        self._model_name = model_name
        self._temperature = temperature
        self._top_p = top_p
        self._provider_only = provider_only
        self._allow_fallbacks = allow_fallbacks

    async def aclose(self) -> None:
        await self._client.aclose()

    @retry(
        retry=retry_if_exception_type(httpx.RequestError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )
    async def generate_cot(self, question: str) -> TeacherResponse:
        payload: Dict[str, Any] = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": TEACHER_USER_PROMPT.format(question=question),
                },
            ],
            "provider": {
                "only": self._provider_only,
                "allow_fallbacks": self._allow_fallbacks,
            },
            "temperature": self._temperature,
            "top_p": self._top_p,
            "tool_choice": "none",
        }

        start = time.perf_counter()
        resp = await self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        latency_ms = int((time.perf_counter() - start) * 1000)

        choice = data["choices"][0]

        return TeacherResponse(
            answer=choice["message"]["content"],
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage"),
            latency_ms=latency_ms,
            model=data.get("model"),
            request_id=data.get("id"),
        )
