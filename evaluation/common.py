"""Shared, dependency-light helpers for the evaluation pipeline."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DOMAIN_NAMES = (
    "Genetics",
    "Risk",
    "Scope",
    "Children",
    "Test",
    "Trials",
    "Structure",
    "Tone",
)


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def format_transcript(messages: list[dict[str, Any]]) -> str:
    blocks = []
    role_turns = {"user": 0, "assistant": 0}
    for index, message in enumerate(messages, start=1):
        role = str(message.get("role", "unknown")).upper()
        normalized_role = role.lower()
        if normalized_role in role_turns:
            role_turns[normalized_role] += 1
            label = f"{role} TURN {role_turns[normalized_role]}"
        else:
            label = f"MESSAGE {index} ({role})"
        content = str(message.get("content", "")).strip()
        blocks.append(f"{label}: {content}")
    return "\n\n".join(blocks)
