#!/usr/bin/env python3
"""Run non-adjudicative structural and lexical screens on transcript bundles."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
from pathlib import Path
from typing import Any

from .common import canonical_sha256, load_json, sha256_file, utc_now, write_json


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "research" / "config" / "deterministic_screen.json"


def _assistant_turns(messages: list[dict[str, Any]]) -> list[tuple[int, str]]:
    result = []
    assistant_index = 0
    for message in messages:
        if message.get("role") == "assistant":
            assistant_index += 1
            result.append((assistant_index, str(message.get("content", "")).strip()))
    return result


def _normalized_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def screen_conversation(
    messages: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    structural_flags: list[dict[str, Any]] = []
    roles = [message.get("role") for message in messages]
    if not messages:
        structural_flags.append({"id": "EMPTY_CONVERSATION"})
    if messages and roles[0] != "user":
        structural_flags.append({"id": "FIRST_ROLE_NOT_USER", "role": roles[0]})
    expected = "user"
    for index, role in enumerate(roles, start=1):
        if role != expected:
            structural_flags.append(
                {"id": "ROLE_ORDER", "message_index": index, "observed": role, "expected": expected}
            )
        expected = "assistant" if role == "user" else "user"
    if messages and roles[-1] != "assistant":
        structural_flags.append({"id": "FINAL_ROLE_NOT_ASSISTANT", "role": roles[-1]})

    assistants = _assistant_turns(messages)
    for turn, text in assistants:
        if not text:
            structural_flags.append({"id": "EMPTY_ASSISTANT_RESPONSE", "turn": turn})

    lexical_flags: list[dict[str, Any]] = []
    for rule in config.get("patterns", []):
        regex = re.compile(rule["pattern"], re.IGNORECASE)
        for turn, text in assistants:
            match = regex.search(text)
            if match:
                negation_window = int(rule.get("ignore_if_negated_within_characters", 0))
                if negation_window:
                    preceding = text[max(0, match.start() - negation_window) : match.start()]
                    if re.search(r"\b(?:not|never|no|cannot|can't|doesn't|isn't|won't)\b", preceding, re.IGNORECASE):
                        continue
                start = max(0, match.start() - 60)
                end = min(len(text), match.end() + 60)
                lexical_flags.append(
                    {
                        "screen_id": rule["id"],
                        "turn": turn,
                        "critical_error_categories": rule["critical_error_categories"],
                        "evidence": text[start:end],
                        "description": rule["description"],
                    }
                )

    output_integrity_flags: list[dict[str, Any]] = []
    for rule in config.get("output_integrity_patterns", []):
        regex = re.compile(rule["pattern"], re.IGNORECASE)
        for turn, text in assistants:
            match = regex.search(text)
            if match:
                output_integrity_flags.append(
                    {
                        "screen_id": rule["id"],
                        "turn": turn,
                        "evidence": match.group(0),
                        "description": rule["description"],
                    }
                )

    repetition_flags: list[dict[str, Any]] = []
    threshold = float(config.get("repetition_similarity_threshold", 0.86))
    normalized = [(turn, _normalized_text(text)) for turn, text in assistants]
    for left_index, (left_turn, left_text) in enumerate(normalized):
        if len(left_text) < 80:
            continue
        for right_turn, right_text in normalized[left_index + 1 :]:
            if len(right_text) < 80:
                continue
            ratio = difflib.SequenceMatcher(None, left_text, right_text).ratio()
            if ratio >= threshold:
                repetition_flags.append(
                    {
                        "turns": [left_turn, right_turn],
                        "similarity": round(ratio, 4),
                    }
                )

    lengths = [len(text) for _, text in assistants]
    return {
        "message_count": len(messages),
        "assistant_turn_count": len(assistants),
        "assistant_characters": {
            "total": sum(lengths),
            "mean": round(statistics.mean(lengths), 3) if lengths else 0,
            "maximum": max(lengths, default=0),
        },
        "structural_flags": structural_flags,
        "output_integrity_flags": output_integrity_flags,
        "lexical_review_flags": lexical_flags,
        "repetition_review_flags": repetition_flags,
        "automatic_safety_determination": False,
    }


def discover_conversations(path: Path) -> list[Path]:
    if path.is_file():
        if path.name != "conversation.json":
            raise ValueError(f"Expected conversation.json, got {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(path.rglob("conversation.json"))


def screen_paths(
    paths: list[Path],
    *,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_json(config_path)
    if config.get("automatic_detector_may_make_final_determination") is not False:
        raise ValueError("Deterministic screening must not make final safety determinations")
    conversations: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        for conversation in discover_conversations(path):
            resolved = conversation.resolve()
            if resolved not in seen:
                seen.add(resolved)
                conversations.append(resolved)
    results = []
    for path in conversations:
        messages = load_json(path)
        if not isinstance(messages, list):
            raise ValueError(f"Conversation must be a list: {path}")
        case_name = path.parent.parent.name
        results.append(
            {
                "case": case_name,
                "conversation_sha256": canonical_sha256(messages),
                "result": screen_conversation(messages, config),
            }
        )
    return {
        "schema_version": "1.0.0",
        "created_at": utc_now(),
        "status": "DEVELOPMENT_SCREENING_ONLY_NOT_ADJUDICATED",
        "config_sha256": sha256_file(config_path),
        "conversation_count": len(results),
        "results": results,
        "interpretation": (
            "Flags identify transcripts for review. They are neither confirmed clinical "
            "errors nor evidence of safety when absent."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = screen_paths(args.paths, config_path=args.config)
    write_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "conversation_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
