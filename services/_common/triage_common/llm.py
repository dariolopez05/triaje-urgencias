from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


DEFAULT_PROMPTS_DIR = Path(os.getenv("PROMPTS_DIR", "/app/data/prompts"))


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    default_model: str
    timeout_seconds: float
    max_retries: int

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            base_url=os.getenv("OLLAMA_HOST", "http://ollama:11434"),
            default_model=os.getenv("OLLAMA_DEFAULT_MODEL", "llama3"),
            timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT", "120")),
            max_retries=int(os.getenv("OLLAMA_MAX_RETRIES", "3")),
        )


class LLMError(RuntimeError):
    pass


class LLMInvalidJSON(LLMError):
    pass


class LLMClient:
    def __init__(
        self,
        config: LLMConfig | None = None,
        client: Optional[httpx.Client] = None,
        prompts_dir: Path | None = None,
    ) -> None:
        self._config = config or LLMConfig.from_env()
        self._client = client or httpx.Client(timeout=self._config.timeout_seconds)
        self._env = Environment(
            loader=FileSystemLoader(str(prompts_dir or DEFAULT_PROMPTS_DIR)),
            autoescape=select_autoescape(default=False),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, template_name: str, **context: Any) -> str:
        template = self._env.get_template(template_name)
        return template.render(**context)

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        json_mode: bool = False,
        options: Optional[dict[str, Any]] = None,
        think: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or self._config.default_model,
            "prompt": prompt,
            "stream": False,
            "think": think,
        }
        if json_mode:
            payload["format"] = "json"
        if options:
            payload["options"] = options
        return self._call_generate(payload)

    def generate_json(
        self,
        prompt: str,
        model: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        raw = self.generate(prompt=prompt, model=model, json_mode=True, options=options)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMInvalidJSON(f"Respuesta LLM no es JSON valido: {raw[:200]}") from exc

    def render_and_generate(
        self,
        template_name: str,
        context: dict[str, Any],
        model: Optional[str] = None,
        json_mode: bool = False,
        options: Optional[dict[str, Any]] = None,
    ) -> str:
        prompt = self.render(template_name, **context)
        return self.generate(prompt=prompt, model=model, json_mode=json_mode, options=options)

    def render_and_generate_json(
        self,
        template_name: str,
        context: dict[str, Any],
        model: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        prompt = self.render(template_name, **context)
        return self.generate_json(prompt=prompt, model=model, options=options)

    def list_models(self) -> list[str]:
        response = self._client.get(f"{self._config.base_url}/api/tags")
        response.raise_for_status()
        data = response.json()
        return [model["name"] for model in data.get("models", [])]

    def close(self) -> None:
        self._client.close()

    def _call_generate(self, payload: dict[str, Any]) -> str:
        attempts = max(1, self._config.max_retries)

        @retry(
            reraise=True,
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((httpx.HTTPError,)),
        )
        def _do_call() -> str:
            response = self._client.post(
                f"{self._config.base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            if "response" not in data:
                raise LLMError(f"Respuesta Ollama sin campo 'response': {data}")
            return str(data["response"])

        return _do_call()
