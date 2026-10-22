"""Small OpenAI-compatible generation client with auditable usage totals."""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


class TextGenerator(Protocol):
    def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
        """Return one generated text completion."""


@dataclass
class GenerationUsage:
    requests: int = 0
    reported_requests: int = 0
    complete_usage_requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reported: bool = False


class OpenAICompatibleGenerator:
    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 120,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.usage = GenerationUsage()

    def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.endpoint, data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"generation request failed: {exc}") from exc

        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("generation response has no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("generation response has empty message content")

        usage = payload.get("usage") or {}
        self.usage.requests += 1
        self.usage.reported = self.usage.reported or bool(usage)
        self.usage.reported_requests += int(bool(usage))
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        complete_usage = all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
            for value in (prompt_tokens, completion_tokens)
        )
        self.usage.complete_usage_requests += int(complete_usage)
        if complete_usage:
            self.usage.prompt_tokens += int(prompt_tokens)
            self.usage.completion_tokens += int(completion_tokens)
        return content.strip()
