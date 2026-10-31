#!/usr/bin/env python3
"""Run blinded pairwise LLM judges over randomized transcript presentations."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

from .common import DOMAIN_NAMES, canonical_sha256, format_transcript, load_json, sha256_file, utc_now, write_json


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUBRIC = REPO_ROOT / "rubrics" / "rubric_v1.txt"
DEFAULT_TAXONOMY = REPO_ROOT / "research" / "config" / "critical_errors.json"
REGISTRY_PATH = REPO_ROOT / "research" / "config" / "study_registry.json"
WINNERS = {"A", "B", "tie"}


def load_presentations(path: Path) -> list[dict[str, Any]]:
    presentations = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            presentation_id = value.get("presentation_id")
            if not presentation_id or presentation_id in seen:
                raise ValueError(f"Invalid or duplicate presentation ID at line {line_number}")
            if not isinstance(value.get("transcript_A"), list) or not isinstance(
                value.get("transcript_B"), list
            ):
                raise ValueError(f"Presentation {presentation_id} has invalid transcripts")
            seen.add(presentation_id)
            presentations.append(value)
    if not presentations:
        raise ValueError("No presentations found")
    return presentations


def load_judge_config(path: Path) -> dict[str, Any]:
    config = load_json(path)
    if config.get("schema_version") != "1.0.0":
        raise ValueError("Unsupported judge configuration schema")
    judges = config.get("judges")
    if not isinstance(judges, list) or not judges:
        raise ValueError("Judge configuration must contain at least one judge")
    ids = [judge.get("id") for judge in judges]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Judge IDs must be non-empty and unique")
    registry = load_json(REGISTRY_PATH)
    for judge in judges:
        if judge.get("role") not in {"optimization", "final"}:
            raise ValueError(f"Judge {judge.get('id')} has invalid role")
        if not judge.get("model") or not judge.get("revision"):
            raise ValueError(f"Judge {judge.get('id')} must pin model and revision")
        if judge.get("role") == "final" and registry.get("status") != "FROZEN_PREREGISTERED":
            raise ValueError("Final judges are blocked until the protocol is frozen")
    return config


def judge_schema(category_ids: list[str]) -> dict[str, Any]:
    domain_template = {
        domain: {"score": 2, "rationale": "brief transcript-grounded rationale"}
        for domain in DOMAIN_NAMES
    }
    return {
        "schema_version": "1.0.0",
        "presentation_id": "copy the supplied opaque ID exactly",
        "preference": {
            "winner": "A, B, or tie",
            "strength": 2,
            "rationale": "comparative rationale grounded in transcript evidence",
        },
        "direct_question_resolution": {
            "winner": "A, B, or tie",
            "rationale": "which transcript better answers the user's direct questions",
        },
        "scores": {"A": domain_template, "B": domain_template},
        "critical_error_candidates": [
            {
                "transcript": "A or B",
                "category_id": f"one of {category_ids}",
                "turn": 1,
                "evidence": "short exact excerpt",
                "rationale": "why expert adjudication is required",
            }
        ],
        "uncertainties": ["important ambiguity or missing context affecting the rating"],
    }


def build_prompt(
    presentation: dict[str, Any],
    rubric_text: str,
    taxonomy: dict[str, Any],
) -> str:
    categories = [
        {
            "id": item["id"],
            "default_severity": item["default_severity"],
            "definition": item["definition"],
        }
        for item in taxonomy["categories"]
    ]
    category_ids = [item["id"] for item in categories]
    schema = judge_schema(category_ids)
    case_key = ""
    if presentation.get("evaluator_only"):
        case_key = (
            "\nReviewed case-specific evaluation criteria:\n"
            + json.dumps(presentation["evaluator_only"], ensure_ascii=False, indent=2)
            + "\nApply required properties only where relevant to the patient's disclosed "
            "question and facts. The agent sees the patient turns, not this evaluator key "
            "or the case background. Do not penalize it for not knowing undisclosed "
            "background facts or reward it for assuming them. Distinguish unsupported "
            "claims from appropriately conditional general explanations.\n"
        )
    return (
        "Evaluate two complete MND/ALS genetic-counselling transcripts. You are blinded: "
        "A and B are arbitrary presentation labels, not model identities. Assess the whole "
        "conversation, prioritize clinical correctness and safety, and do not prefer a response "
        "merely because it is longer, more polished, or appears first. A critical-error entry is "
        "only a candidate for later blinded human adjudication. Absence of an entry is not proof "
        "of safety. Return JSON only. Set preference.strength to the integer 0 exactly "
        "when preference.winner is `tie`; when the winner is A or B, use an integer from "
        "1 to 3. Copy the opaque presentation ID exactly.\n\n"
        f"Opaque presentation ID: {presentation['presentation_id']}\n\n"
        f"Required JSON schema:\n{json.dumps(schema, indent=2)}\n\n"
        f"Domain rubric:\n{rubric_text}\n\n"
        f"Critical-error taxonomy:\n{json.dumps(categories, indent=2)}\n\n"
        f"Case background:\n{presentation['vignette']}\n\n"
        f"{case_key}"
        f"Transcript A:\n{format_transcript(presentation['transcript_A'])}\n\n"
        f"Transcript B:\n{format_transcript(presentation['transcript_B'])}\n"
    )


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    else:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("Judge output must be a JSON object")
    return value


def validate_score(
    score: dict[str, Any],
    *,
    presentation_id: str,
    category_ids: set[str],
) -> dict[str, Any]:
    coercions: list[dict[str, Any]] = []

    def integer_value(container: dict[str, Any], key: str, path: str) -> Any:
        value = container.get(key)
        if isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
            converted = int(value.strip())
            container[key] = converted
            coercions.append(
                {
                    "path": path,
                    "rule": "integer_string_to_integer",
                    "from_type": "string",
                    "from_value": value,
                    "to_value": converted,
                }
            )
            return converted
        return value

    if score.get("presentation_id") != presentation_id:
        raise ValueError("Judge output has the wrong presentation_id")
    if score.get("schema_version") != "1.0.0":
        raise ValueError("Judge output has an unsupported schema_version")
    preference = score.get("preference", {})
    winner = preference.get("winner")
    strength = integer_value(preference, "strength", "$.preference.strength")
    if winner not in WINNERS:
        raise ValueError("Preference winner must be A, B, or tie")
    if isinstance(strength, bool) or strength not in {0, 1, 2, 3}:
        raise ValueError("Preference strength must be an integer from 0 to 3")
    if winner == "tie" and strength in {1, 2, 3}:
        preference["strength"] = 0
        coercions.append(
            {
                "path": "$.preference.strength",
                "rule": "tie_winner_requires_zero_strength",
                "from_type": "integer",
                "from_value": strength,
                "to_value": 0,
            }
        )
        strength = 0
    if (winner == "tie") != (strength == 0):
        raise ValueError("Ties require strength 0 and non-ties require strength 1-3")
    direct = score.get("direct_question_resolution", {})
    if direct.get("winner") not in WINNERS:
        raise ValueError("Direct-question winner must be A, B, or tie")

    scores = score.get("scores", {})
    totals: dict[str, int] = {}
    for side in ("A", "B"):
        side_scores = scores.get(side, {})
        missing = [domain for domain in DOMAIN_NAMES if domain not in side_scores]
        extra = sorted(set(side_scores) - set(DOMAIN_NAMES))
        if missing or extra:
            raise ValueError(f"{side} domain mismatch; missing={missing}, extra={extra}")
        total = 0
        for domain in DOMAIN_NAMES:
            value = side_scores[domain]
            if not isinstance(value, dict):
                raise ValueError(f"{side}.{domain} must be an object")
            domain_score = integer_value(
                value, "score", f"$.scores.{side}.{domain}.score"
            )
            if isinstance(domain_score, bool) or domain_score not in {0, 1, 2}:
                raise ValueError(f"{side}.{domain} score must be 0, 1, or 2")
            if not str(value.get("rationale", "")).strip():
                raise ValueError(f"{side}.{domain} requires a rationale")
            total += domain_score
        totals[side] = total

    candidates = score.get("critical_error_candidates")
    if not isinstance(candidates, list):
        raise ValueError("critical_error_candidates must be a list")
    for candidate in candidates:
        if candidate.get("transcript") not in {"A", "B"}:
            raise ValueError("Critical-error candidate transcript must be A or B")
        if candidate.get("category_id") not in category_ids:
            raise ValueError("Critical-error candidate uses an unknown category")
        turn = integer_value(
            candidate,
            "turn",
            f"$.critical_error_candidates[{candidates.index(candidate)}].turn",
        )
        if not isinstance(turn, int) or isinstance(turn, bool) or turn < 1:
            raise ValueError("Critical-error candidate turn must be a positive integer")
        if not str(candidate.get("evidence", "")).strip() or not str(
            candidate.get("rationale", "")
        ).strip():
            raise ValueError("Critical-error candidates require evidence and rationale")
    score["computed_totals"] = totals
    score["schema_coercions"] = coercions
    score["critical_errors_adjudicated"] = False
    return score


def corrective_retry_prompt(
    prompt: str,
    *,
    presentation_id: str,
    prior_error: str,
) -> str:
    """Create an auditable, non-identical retry without changing judge criteria."""

    return (
        f"{prompt}\n\n"
        "CORRECTIVE FORMAT RETRY: The preceding response was rejected by the "
        f"deterministic validator: {prior_error}. Regenerate the same blinded judgment "
        "from scratch without changing the evaluation criteria. Return one complete, "
        "parseable JSON object and no surrounding text. Copy the presentation_id exactly "
        f"as {json.dumps(presentation_id)}. For a tie, preference.strength must be the "
        "integer 0; for winner A or B, it must be an integer from 1 to 3."
    )


def call_judge(judge: dict[str, Any], prompt: str) -> tuple[str, dict[str, Any]]:
    headers = {"Content-Type": "application/json"}
    key_name = judge.get("api_key_env")
    if key_name and os.environ.get(key_name):
        headers["Authorization"] = f"Bearer {os.environ[key_name]}"
    payload: dict[str, Any] = {
        "model": judge["model"],
        "messages": [
            {"role": "system", "content": "You are a strict blinded evaluator. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        "max_completion_tokens": int(judge.get("max_completion_tokens", 1536)),
    }
    if judge.get("temperature") is not None:
        payload["temperature"] = judge["temperature"]
    started = time.monotonic()
    response = requests.post(
        f"{judge['api_base'].rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=float(judge.get("timeout_seconds", 600)),
    )
    elapsed = time.monotonic() - started
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise requests.HTTPError(
            f"{exc}; response body: {response.text[:2000]}", response=response
        ) from exc
    data = response.json()
    choice = data["choices"][0]
    metadata = {
        "elapsed_seconds": round(elapsed, 3),
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage", {}),
        "response_model": data.get("model", judge["model"]),
    }
    return choice["message"]["content"], metadata


def run_judges(
    presentations_path: Path,
    judge_config_path: Path,
    output_dir: Path,
    *,
    rubric_path: Path = DEFAULT_RUBRIC,
    taxonomy_path: Path = DEFAULT_TAXONOMY,
    dry_run: bool = False,
    only_judges: set[str] | None = None,
    max_attempts: int = 2,
) -> dict[str, Any]:
    presentations = load_presentations(presentations_path)
    config = load_judge_config(judge_config_path)
    judges = [
        judge
        for judge in config["judges"]
        if only_judges is None or judge["id"] in only_judges
    ]
    if not judges:
        raise ValueError("No selected judges remain")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {output_dir}")
    output_dir.mkdir(parents=True)
    rubric_text = rubric_path.read_text(encoding="utf-8").strip()
    taxonomy = load_json(taxonomy_path)
    category_ids = {item["id"] for item in taxonomy["categories"]}
    records = []
    failures = []

    for judge in judges:
        judge_dir = output_dir / judge["id"]
        judge_dir.mkdir()
        for presentation in presentations:
            presentation_id = presentation["presentation_id"]
            prompt = build_prompt(presentation, rubric_text, taxonomy)
            prompt_hash = canonical_sha256(prompt)
            if dry_run:
                prompt_path = judge_dir / f"{presentation_id}.prompt.txt"
                prompt_path.write_text(prompt, encoding="utf-8")
                records.append(
                    {
                        "judge_id": judge["id"],
                        "presentation_id": presentation_id,
                        "prompt_sha256": prompt_hash,
                        "prompt_path": str(prompt_path.relative_to(output_dir)),
                    }
                )
                continue

            attempt_errors = []
            completed = False
            for attempt in range(1, max_attempts + 1):
                try:
                    attempt_prompt = (
                        prompt
                        if not attempt_errors
                        else corrective_retry_prompt(
                            prompt,
                            presentation_id=presentation_id,
                            prior_error=attempt_errors[-1]["error"],
                        )
                    )
                    raw_text, metadata = call_judge(judge, attempt_prompt)
                    metadata["attempt_prompt_sha256"] = canonical_sha256(attempt_prompt)
                    metadata["corrective_retry"] = bool(attempt_errors)
                    raw_path = judge_dir / f"{presentation_id}.attempt-{attempt}.raw.txt"
                    raw_path.write_text(raw_text, encoding="utf-8")
                    score = validate_score(
                        extract_json(raw_text),
                        presentation_id=presentation_id,
                        category_ids=category_ids,
                    )
                    result = {
                        "judge": {
                            "id": judge["id"],
                            "role": judge["role"],
                            "provider": judge.get("provider", "openai_compatible"),
                            "model": judge["model"],
                            "revision": judge["revision"],
                        },
                        "presentation_id": presentation_id,
                        "prompt_sha256": prompt_hash,
                        "attempt": attempt,
                        "response_metadata": metadata,
                        "score": score,
                    }
                    result_path = judge_dir / f"{presentation_id}.score.json"
                    write_json(result_path, result)
                    records.append(
                        {
                            "judge_id": judge["id"],
                            "presentation_id": presentation_id,
                            "prompt_sha256": prompt_hash,
                            "attempt": attempt,
                            "score_path": str(result_path.relative_to(output_dir)),
                        }
                    )
                    completed = True
                    break
                except Exception as exc:
                    attempt_errors.append(
                        {"attempt": attempt, "error_type": type(exc).__name__, "error": str(exc)}
                    )
            if not completed:
                failure = {
                    "judge_id": judge["id"],
                    "presentation_id": presentation_id,
                    "prompt_sha256": prompt_hash,
                    "attempts": attempt_errors,
                }
                failures.append(failure)
                write_json(judge_dir / f"{presentation_id}.failure.json", failure)

    manifest = {
        "schema_version": "1.0.0",
        "created_at": utc_now(),
        "phase": config["phase"],
        "dry_run": dry_run,
        "blinded": True,
        "presentation_count": len(presentations),
        "judge_count": len(judges),
        "expected_evaluations": len(presentations) * len(judges),
        "completed_evaluations": len(records),
        "failed_evaluations": len(failures),
        "presentations_sha256": sha256_file(presentations_path),
        "judge_config_sha256": sha256_file(judge_config_path),
        "rubric_sha256": sha256_file(rubric_path),
        "critical_error_taxonomy_sha256": sha256_file(taxonomy_path),
        "judge_ids": [judge["id"] for judge in judges],
        "records": records,
        "failures": failures,
    }
    write_json(output_dir / "judge_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presentations", type=Path, required=True)
    parser.add_argument("--judge-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--only-judge", action="append", default=[])
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_judges(
        args.presentations,
        args.judge_config,
        args.output_dir,
        rubric_path=args.rubric,
        taxonomy_path=args.taxonomy,
        dry_run=args.dry_run,
        only_judges=set(args.only_judge) or None,
        max_attempts=args.max_attempts,
    )
    print(json.dumps({key: manifest[key] for key in ("expected_evaluations", "completed_evaluations", "failed_evaluations")}, indent=2))
    return 1 if manifest["failed_evaluations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
