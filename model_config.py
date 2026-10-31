"""Portable, independent model settings for each part of the workflow."""

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit


APP_DIR = Path(__file__).resolve().parent
ROLES = {"agent", "judge", "proposal", "roleplay"}


def load_environment(path=None) -> None:
    """Read an optional .env without interpolation or replacing exported variables.

    Supports KEY=value, export KEY=value, single/double quoted values, blank
    lines, and comments. Values such as $HOME are kept literally. A malformed
    line raises an error naming its line number, never its value.
    """
    source = Path(path) if path is not None else APP_DIR / ".env"
    if not source.is_file():
        return
    values = {}
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, raw_value = line.partition("=")
        key = key.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid .env setting on line {line_number}.")
        value = raw_value.strip()
        if value.startswith(("'", '"')):
            quote = value[0]
            end = value.find(quote, 1)
            if end < 0 or (value[end + 1:].strip() and not value[end + 1:].lstrip().startswith("#")):
                raise ValueError(f"Invalid quoted .env setting on line {line_number}.")
            value = value[1:end]
        else:
            value = "" if value.startswith("#") else re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        values[key] = value
    for key, value in values.items():
        os.environ.setdefault(key, value)


def _setting(role, name, fallback=None):
    for key in (f"MND_{role.upper()}_{name}", f"{role.upper()}_{name}"):
        if os.environ.get(key, "").strip():
            return os.environ[key].strip()
    return fallback


def _validate_api_base(api_base, role):
    try:
        parsed = urlsplit(api_base)
    except (TypeError, ValueError):
        raise ValueError(f"The {role} API base must be a valid HTTP(S) URL.") from None
    if (parsed.scheme not in {"http", "https"} or not parsed.netloc
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError(f"The {role} API base must be an HTTP(S) URL without credentials, query, or fragment.")


@dataclass(frozen=True)
class ModelConfig:
    role: str
    model: str = ""
    api_base: str = "https://api.openai.com/v1"
    api_key: str = field(default="", repr=False)
    timeout: float = 180.0
    max_tokens: int = 512
    token_parameter: str = "max_tokens"
    temperature: Optional[float] = None

    @classmethod
    def from_env(cls, role: str, **overrides):
        if role not in ROLES:
            raise ValueError(f"Unknown model role: {role}")
        load_environment()
        model_fallback = os.environ.get("LLM", "") if role == "agent" else ""
        token_fallback = os.environ.get("MND_MAX_NEW_TOKENS", "512") if role == "agent" else ("300" if role == "roleplay" else "4096")
        values = {
            "role": role,
            "model": _setting(role, "MODEL", model_fallback),
            "api_base": _setting(role, "API_BASE", os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"),
            "api_key": _setting(role, "API_KEY", os.environ.get("OPENAI_API_KEY", "")),
            "timeout": _setting(role, "TIMEOUT", "180"),
            "max_tokens": _setting(role, "MAX_TOKENS", token_fallback),
            "token_parameter": _setting(role, "TOKEN_PARAMETER", "max_tokens"),
            "temperature": _setting(role, "TEMPERATURE"),
        }
        for name, value in overrides.items():
            if name not in values or name == "role":
                raise TypeError(f"Unknown model setting: {name}")
            if value is not None:
                values[name] = value
        try:
            values["timeout"] = float(values["timeout"])
            values["max_tokens"] = int(values["max_tokens"])
            if values["temperature"] is not None:
                values["temperature"] = float(values["temperature"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric setting for the {role} model.") from exc
        return cls(**values)

    def validate(self):
        if not self.model or not self.model.strip():
            raise ValueError(f"Set MND_{self.role.upper()}_MODEL or pass a model name.")
        _validate_api_base(self.api_base, self.role)
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Model timeout must be positive and finite.")
        if self.max_tokens <= 0:
            raise ValueError("Model max_tokens must be positive.")
        if self.token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("token_parameter must be max_tokens or max_completion_tokens.")
        if self.temperature is not None and (not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2):
            raise ValueError("Model temperature must be between 0 and 2.")
        return self

    def public_dict(self):
        _validate_api_base(self.api_base, self.role)
        return {"role": self.role, "model": self.model, "api_base": self.api_base,
                "timeout": self.timeout, "max_tokens": self.max_tokens,
                "token_parameter": self.token_parameter, "temperature": self.temperature}
