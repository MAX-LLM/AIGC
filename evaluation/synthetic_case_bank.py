"""Load generated synthetic patient surfaces as immutable scripted case banks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .common import canonical_sha256, sha256_file


CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def load_case_bank(case_dir: Path) -> dict[str, dict[str, Any]]:
    paths = sorted(case_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"No synthetic case JSON files found in {case_dir}")
    cases: dict[str, dict[str, Any]] = {}
    for path in paths:
        case = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(case, dict):
            raise ValueError(f"Synthetic case must be an object: {path}")
        supplied_hash = case.get("content_sha256")
        unsigned = {key: value for key, value in case.items() if key != "content_sha256"}
        if not isinstance(supplied_hash, str) or supplied_hash != canonical_sha256(unsigned):
            raise ValueError(f"Synthetic case content hash mismatch: {path}")
        blueprint = case.get("immutable_blueprint")
        surface = case.get("generated_surface")
        if not isinstance(blueprint, dict) or not isinstance(surface, dict):
            raise ValueError(f"Synthetic case is missing blueprint or surface: {path}")
        case_id = blueprint.get("case_id")
        if (
            not isinstance(case_id, str)
            or not CASE_ID_RE.fullmatch(case_id)
            or surface.get("case_id") != case_id
        ):
            raise ValueError(f"Synthetic case identity is invalid: {path}")
        if case_id in cases:
            raise ValueError(f"Duplicate synthetic case ID: {case_id}")
        turn_plan = surface.get("turn_plan")
        if not isinstance(turn_plan, list) or not turn_plan:
            raise ValueError(f"Synthetic case has no patient turns: {path}")
        turns = []
        for index, turn in enumerate(turn_plan):
            if not isinstance(turn, dict):
                raise ValueError(f"Synthetic case turn {index + 1} is invalid: {path}")
            utterance = turn.get("patient_utterance")
            reveal_after = turn.get("reveal_only_after")
            if not isinstance(utterance, str) or not utterance.strip():
                raise ValueError(f"Synthetic case turn {index + 1} has no text: {path}")
            if isinstance(reveal_after, bool) or not isinstance(reveal_after, int):
                raise ValueError(f"Synthetic case turn {index + 1} has invalid reveal rule: {path}")
            if reveal_after < 0 or reveal_after > index:
                raise ValueError(f"Synthetic case turn {index + 1} reveals too late: {path}")
            turns.append(utterance.strip())
        if surface.get("opening_message") != turns[0]:
            raise ValueError(f"Synthetic case opening differs from turn one: {path}")
        vignette = {
            "case_id": case_id,
            "phase": blueprint.get("phase"),
            "scenario": blueprint.get("scenario"),
            "fixed_facts": blueprint.get("fixed_facts"),
            "persona_surface": surface.get("persona_surface"),
        }
        cases[case_id] = {
            "case_id": case_id,
            "vignette": json.dumps(vignette, indent=2, ensure_ascii=False),
            "turns": turns,
            "source_path": str(path),
            "source_sha256": sha256_file(path),
            "source_case": case,
        }
    return cases


def select_case_ids(cases: dict[str, dict[str, Any]], raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return sorted(cases)
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if not names:
        raise ValueError("At least one synthetic case is required")
    if len(names) != len(set(names)):
        raise ValueError("Synthetic case IDs must be unique")
    unknown = sorted(set(names) - set(cases))
    if unknown:
        raise ValueError(f"Unknown synthetic cases: {', '.join(unknown)}")
    return names


def case_hash_payload(
    cases: dict[str, dict[str, Any]], names: list[str]
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "vignette": cases[name]["vignette"],
            "turns": cases[name]["turns"],
            "source_sha256": cases[name]["source_sha256"],
        }
        for name in names
    }
