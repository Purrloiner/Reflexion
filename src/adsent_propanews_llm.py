"""Lazy LLM adapters for the AdSent PropaNews experiment.

The repository's existing ``llm.call_llm`` remains available for compatibility.  A
separate deterministic Vertex adapter is provided because the AdSent zero-shot setting
uses greedy decoding, while the existing client samples at temperature 0.7.
"""

from __future__ import annotations

import importlib
import math
import os
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class VertexGeminiConfig:
    model: str = "gemini-2.5-flash"
    project: str | None = None
    location: str = "global"
    temperature: float = 0.0
    seed: int = 42
    max_output_tokens: int | None = None
    max_retries: int = 10

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be a finite non-negative number")
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive when provided")


class VertexGeminiLLM:
    """Callable Gemini client constructed only on the first real model call."""

    def __init__(self, config: VertexGeminiConfig | None = None):
        self.config = config or VertexGeminiConfig()
        self._client: Any | None = None
        self._last_call_metadata: dict[str, object] | None = None

    @property
    def provenance(self) -> dict[str, object]:
        return {"backend": "vertex_gemini", **asdict(self.config)}

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is required for the vertex backend; install the project "
                "requirements or use --llm-backend existing"
            ) from exc

        project = self.config.project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise RuntimeError(
                "Vertex project is missing; pass --vertex-project or set "
                "GOOGLE_CLOUD_PROJECT"
            )
        self._client = genai.Client(
            vertexai=True,
            project=project,
            location=self.config.location,
        )
        return self._client

    def take_last_call_metadata(self) -> dict[str, object] | None:
        """Return and clear metadata for the most recent logical generation call."""

        metadata = self._last_call_metadata
        self._last_call_metadata = None
        return metadata

    @staticmethod
    def _finish_reason(response: object) -> str | None:
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return None
        value = getattr(candidates[0], "finish_reason", None)
        if value is None:
            return None
        name = getattr(value, "name", None)
        return str(name if name is not None else value).split(".")[-1].upper()

    @staticmethod
    def _usage(response: object) -> dict[str, int]:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return {}
        result: dict[str, int] = {}
        for field_name in (
            "prompt_token_count",
            "candidates_token_count",
            "total_token_count",
            "cached_content_token_count",
            "thoughts_token_count",
        ):
            value = getattr(usage, field_name, None)
            if type(value) is int:
                result[field_name] = value
        return result

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        message = str(exc).upper()
        code = str(getattr(exc, "code", "")).upper()
        combined = f"{code} {message}"
        return any(
            marker in combined
            for marker in (
                "429",
                "500",
                "502",
                "503",
                "504",
                "RESOURCE_EXHAUSTED",
                "UNAVAILABLE",
                "DEADLINE_EXCEEDED",
                "INTERNAL",
            )
        )

    def __call__(self, prompt: str) -> str:
        client = self._get_client()
        self._last_call_metadata = None
        attempt_log: list[dict[str, object]] = []
        for attempt in range(self.config.max_retries):
            try:
                generation_config: dict[str, object] = {
                    "temperature": self.config.temperature,
                    "seed": self.config.seed,
                    "candidate_count": 1,
                }
                if self.config.max_output_tokens is not None:
                    generation_config["max_output_tokens"] = self.config.max_output_tokens
                response = client.models.generate_content(
                    model=self.config.model,
                    contents=prompt,
                    config=generation_config,
                )
                finish_reason = self._finish_reason(response)
                usage = self._usage(response)
                metadata: dict[str, object] = {
                    "transport_attempts": attempt + 1,
                    "finish_reason": finish_reason,
                    "usage": usage,
                    "retry_events": attempt_log,
                }
                self._last_call_metadata = metadata
                if finish_reason is not None and finish_reason != "STOP":
                    raise RuntimeError(
                        "model response did not finish normally: "
                        f"finish_reason={finish_reason}"
                    )
                text = getattr(response, "text", None)
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError("model returned an empty response")
                return text
            except Exception as exc:
                retryable = self._retryable(exc)
                attempt_log.append(
                    {
                        "attempt": attempt + 1,
                        "error_type": type(exc).__name__,
                        "retryable": retryable,
                    }
                )
                existing = self._last_call_metadata or {}
                self._last_call_metadata = {
                    **existing,
                    "transport_attempts": attempt + 1,
                    "retry_events": list(attempt_log),
                }
                if not retryable or attempt + 1 >= self.config.max_retries:
                    raise
                wait_seconds = min(60.0, (2**attempt) + random.uniform(0, 3))
                time.sleep(wait_seconds)
        raise RuntimeError("model retry loop exhausted")


class ImportedCallableLLM:
    """Lazy wrapper around ``module:function`` (default: existing ``llm.call_llm``)."""

    def __init__(self, spec: str = "llm:call_llm"):
        if ":" not in spec:
            raise ValueError("callable spec must have the form module:function")
        module_name, attribute = spec.split(":", 1)
        if not module_name or not attribute:
            raise ValueError("callable spec must have the form module:function")
        self.spec = spec
        self._callable: Callable[[str], str] | None = None

    @property
    def provenance(self) -> dict[str, object]:
        return {"backend": "imported_callable", "spec": self.spec}

    def _load(self) -> Callable[[str], str]:
        if self._callable is not None:
            return self._callable
        module_name, attribute = self.spec.split(":", 1)
        module = importlib.import_module(module_name)
        candidate = getattr(module, attribute, None)
        if not callable(candidate):
            raise RuntimeError(f"{self.spec!r} did not resolve to a callable")
        self._callable = candidate
        return candidate

    def __call__(self, prompt: str) -> str:
        response = self._load()(prompt)
        if not isinstance(response, str):
            raise RuntimeError(f"{self.spec!r} returned a non-string response")
        return response
