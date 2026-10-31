"""OpenAI-compatible HTTP backend; local inference packages are optional."""

import os
import threading
from contextlib import contextmanager

import requests

from model_config import ModelConfig


class APIBackend:
    def __init__(self, config: ModelConfig):
        self.config = config.validate()
        self.model_path = config.model
        self.backend_name = "api"
        self.deterministic = False  # A remote provider controls reproducibility.
        self.max_new_tokens = config.max_tokens
        self.max_concurrent_generations = max(1, int(os.environ.get("MND_MAX_CONCURRENT_GENERATIONS", "4")))
        self._generation_slots = threading.BoundedSemaphore(self.max_concurrent_generations)
        self._request_options = threading.local()

    @contextmanager
    def generation_options(self, max_new_tokens=None):
        previous = getattr(self._request_options, "max_new_tokens", None)
        self._request_options.max_new_tokens = max_new_tokens
        try:
            yield
        finally:
            self._request_options.max_new_tokens = previous

    def chat(self, messages):
        token_limit = getattr(self._request_options, "max_new_tokens", None) or self.max_new_tokens
        payload = {"model": self.config.model, "messages": messages,
                   self.config.token_parameter: token_limit}
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        with self._generation_slots:
            try:
                response = requests.post(self.config.api_base.rstrip("/") + "/chat/completions",
                                         headers=headers, json=payload, timeout=self.config.timeout)
                response.raise_for_status()
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else "unknown"
                raise RuntimeError(f"The {self.config.role} model API returned HTTP {status}.") from None
            except requests.RequestException:
                raise RuntimeError(f"The {self.config.role} model API could not be reached.") from None
        try:
            content = response.json()["choices"][0]["message"].get("content")
            if content is None:
                return ""
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                return "\n".join(part["text"] for part in content
                                 if isinstance(part, dict) and isinstance(part.get("text"), str)).strip()
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            pass
        raise RuntimeError(f"The {self.config.role} model API returned an invalid chat response.")
