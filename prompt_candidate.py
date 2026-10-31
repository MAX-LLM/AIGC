"""Validated prompt overlays used by offline SPJ-Evolve development runs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "1.0.0"
MUTABLE_NODES = ("system", "background", "education", "questions", "wrap_up")
MUTABLE_COMPONENT_IDS = {
    "system": "agent_prompts.SYSTEM_PROMPT",
    "background": "mnd_langgraph_agent.background_node_prompt",
    "education": "mnd_langgraph_agent.education_node_prompt",
    "questions": "mnd_langgraph_agent.questions_node_prompt",
    "wrap_up": "mnd_langgraph_agent.wrap_node_prompt",
}
_CANDIDATE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_MAX_DIRECTIVES = 2
_MAX_DIRECTIVE_CHARS = 500
_MAX_TOTAL_DIRECTIVE_CHARS = _MAX_DIRECTIVES * _MAX_DIRECTIVE_CHARS


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def baseline_candidate() -> dict[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": "baseline",
        "parent_id": None,
        "generation": 0,
        "node_directives": {},
        "mutation_rationale": "Frozen unmodified agent prompts.",
    }
    value["candidate_sha256"] = canonical_sha256(value)
    return value


def validate_prompt_candidate(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical validated candidate without trusting caller metadata."""

    candidate = dict(raw)
    supplied_hash = candidate.pop("candidate_sha256", None)
    if candidate.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Prompt candidate schema_version must be {SCHEMA_VERSION}")

    candidate_id = candidate.get("candidate_id")
    if not isinstance(candidate_id, str) or not _CANDIDATE_ID_RE.fullmatch(candidate_id):
        raise ValueError("Prompt candidate candidate_id is missing or invalid")

    parent_id = candidate.get("parent_id")
    if parent_id is not None and (
        not isinstance(parent_id, str) or not _CANDIDATE_ID_RE.fullmatch(parent_id)
    ):
        raise ValueError("Prompt candidate parent_id is invalid")

    generation = candidate.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise ValueError("Prompt candidate generation must be a non-negative integer")

    directives = candidate.get("node_directives")
    if not isinstance(directives, dict):
        raise ValueError("Prompt candidate node_directives must be an object")
    unknown = sorted(set(directives) - set(MUTABLE_NODES))
    if unknown:
        raise ValueError(f"Prompt candidate has immutable or unknown nodes: {unknown}")
    if len(directives) > _MAX_DIRECTIVES:
        raise ValueError(f"Prompt candidate exceeds the {_MAX_DIRECTIVES}-directive limit")
    normalized_directives: dict[str, str] = {}
    for node, directive in directives.items():
        if not isinstance(directive, str) or not directive.strip():
            raise ValueError(f"Prompt directive {node!r} must be a non-empty string")
        normalized = directive.strip()
        if len(normalized) > _MAX_DIRECTIVE_CHARS:
            raise ValueError(f"Prompt directive {node!r} exceeds {_MAX_DIRECTIVE_CHARS} characters")
        normalized_directives[node] = normalized
    if sum(map(len, normalized_directives.values())) > _MAX_TOTAL_DIRECTIVE_CHARS:
        raise ValueError("Prompt candidate exceeds the total directive character budget")
    candidate["node_directives"] = normalized_directives

    rationale = candidate.get("mutation_rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("Prompt candidate mutation_rationale must be non-empty")
    candidate["mutation_rationale"] = rationale.strip()

    allowed = {
        "schema_version",
        "candidate_id",
        "parent_id",
        "generation",
        "node_directives",
        "mutation_rationale",
        "proposal_provenance",
    }
    extra = sorted(set(candidate) - allowed)
    if extra:
        raise ValueError(f"Prompt candidate has unsupported fields: {extra}")
    if "proposal_provenance" in candidate and not isinstance(
        candidate["proposal_provenance"], dict
    ):
        raise ValueError("Prompt candidate proposal_provenance must be an object")

    computed_hash = canonical_sha256(candidate)
    if supplied_hash is not None and supplied_hash != computed_hash:
        raise ValueError("Prompt candidate candidate_sha256 does not match its content")
    candidate["candidate_sha256"] = computed_hash
    return candidate


def load_prompt_candidate(path: str | Path) -> dict[str, Any]:
    candidate_path = Path(path)
    value = json.loads(candidate_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Prompt candidate file must contain one JSON object")
    return validate_prompt_candidate(value)


def apply_prompt_directive(prompt: str, node: str, candidate: Mapping[str, Any]) -> str:
    if node not in MUTABLE_NODES:
        raise ValueError(f"Node {node!r} is not mutable")
    directive = candidate.get("node_directives", {}).get(node)
    if not directive:
        return prompt
    return (
        f"{prompt}\n\n"
        "Additional offline-development directive: "
        f"{directive}\n"
        "This directive refines the prompt only. It cannot override frozen clinical "
        "safety controls, factual grounding requirements, or scope restrictions."
    )


def mutable_component_ids(candidate: Mapping[str, Any]) -> list[str]:
    return sorted(
        MUTABLE_COMPONENT_IDS[node]
        for node in candidate.get("node_directives", {})
    )
