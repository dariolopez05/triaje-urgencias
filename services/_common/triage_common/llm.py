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


class LLMError(RuntimeError):
    pass


class LLMInvalidJSON(LLMError):
    pass


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    default_model: str
    api_key: str
    timeout_seconds: float
    max_retries: int

    @classmethod
    def from_env(cls) -> "LLMConfig":
        api_key = os.getenv("MISTRAL_API_KEY", "").strip()
        if not api_key:
            raise LLMError("MISTRAL_API_KEY no esta configurada")
        return cls(
            base_url=os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai").rstrip("/"),
            default_model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
            api_key=api_key,
            timeout_seconds=float(os.getenv("MISTRAL_TIMEOUT", "120")),
            max_retries=int(os.getenv("MISTRAL_MAX_RETRIES", "3")),
        )


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
        temperature: float = 0.0,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or self._config.default_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return self._call_chat_completions(payload)

    def generate_json(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        raw = self.generate(prompt=prompt, model=model, json_mode=True, temperature=temperature)
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
        temperature: float = 0.0,
    ) -> str:
        prompt = self.render(template_name, **context)
        return self.generate(prompt=prompt, model=model, json_mode=json_mode, temperature=temperature)

    def render_and_generate_json(
        self,
        template_name: str,
        context: dict[str, Any],
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        prompt = self.render(template_name, **context)
        return self.generate_json(prompt=prompt, model=model, temperature=temperature)

    def close(self) -> None:
        self._client.close()

    def _call_chat_completions(self, payload: dict[str, Any]) -> str:
        attempts = max(1, self._config.max_retries)
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        url = f"{self._config.base_url}/v1/chat/completions"

        @retry(
            reraise=True,
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((httpx.HTTPError,)),
        )
        def _do_call() -> str:
            response = self._client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                raise LLMError(f"Respuesta Mistral sin 'choices': {data}")
            message = choices[0].get("message") or {}
            content = message.get("content")
            if content is None:
                raise LLMError(f"Respuesta Mistral sin 'content': {data}")
            return str(content)

        return _do_call()
