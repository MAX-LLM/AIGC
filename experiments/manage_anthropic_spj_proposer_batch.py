#!/usr/bin/env python3
"""Prepare and import a fully independent Anthropic SPJ proposer lineage.

The reflection stage is one provider-native batch. Proposal requests are prepared
one at a time so candidate N can see candidates 1..N-1, matching the dependency
structure of the local SPJ proposer while retaining Anthropic Batch pricing.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.common import canonical_sha256, sha256_file, utc_now, write_json
from evaluation.frontier_pairwise_batch import build_provider_request
from evaluation.provider_config import native_config
from prompt_candidate import validate_prompt_candidate
from experiments.spj_lesson_register import proposer_context


RUBRIC_PATH = REPO_ROOT / "rubrics" / "rubric_v1.txt"
TAXONOMY_PATH = REPO_ROOT / "research" / "config" / "critical_errors.json"
REGISTRY_PATH = (
    REPO_ROOT / "research" / "config" / "development_experiment_registry.json"
)
SYSTEM_REFLECTION = (
    "Act as a strict development-only patient-alignment and clinical-counselling "
    "reviewer."
)
SYSTEM_PROPOSAL = "Act as the prompt-mutation component of SPJ-Evolve."
MUTABLE_NODES = ("system", "background", "education", "questions", "wrap_up")
RUBRIC_DOMAINS = (
    "Genetics",
    "Risk",
    "Scope",
    "Children",
    "Test",
    "Trials",
    "Structure",
    "Tone",
)


def repo_relative(path: Path) -> str:
    """Return a stable repository-relative path for provenance."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def git_state() -> dict[str, Any]:
    def run(command: list[str]) -> str:
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        return {
            "branch": run(["git", "branch", "--show-current"]),
            "commit": run(["git", "rev-parse", "HEAD"]),
            "tracked_dirty": bool(
                run(["git", "status", "--porcelain", "--untracked-files=no"])
            ),
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"branch": None, "commit": None, "tracked_dirty": None,
                "source": "source_archive_without_git_metadata"}


def closed_object(
    properties: dict[str, Any], required: list[str] | None = None
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required if required is not None else list(properties),
        "additionalProperties": False,
    }


def nonempty_string() -> dict[str, Any]:
    return {"type": "string", "minLength": 1}


def reflection_schema(case_id: str, category_ids: list[str]) -> dict[str, Any]:
    return closed_object(
        {
            "case_id": {"type": "string", "enum": [case_id]},
            "strengths": {
                "type": "array",
                "items": nonempty_string(),
            },
            "failure_modes": {
                "type": "array",
                "items": closed_object(
                    {
                        "turn": {"type": "integer"},
                        "rubric_domain": {
                            "type": "string",
                            "enum": list(RUBRIC_DOMAINS),
                        },
                        "evidence": nonempty_string(),
                        "problem": nonempty_string(),
                        "target_node": {
                            "type": "string",
                            "enum": list(MUTABLE_NODES),
                        },
                        "recommended_prompt_change": nonempty_string(),
                    }
                ),
            },
            "potential_critical_errors": {
                "type": "array",
                "items": closed_object(
                    {
                        "category_id": {
                            "type": "string",
                            "enum": category_ids,
                        },
                        "turn": {"type": "integer"},
                        "evidence": nonempty_string(),
                        "rationale": nonempty_string(),
                    }
                ),
            },
            "patient_alignment": closed_object(
                {
                    "answered_latest_questions": {
                        "type": "string",
                        "enum": ["yes", "partly", "no"],
                    },
                    "unnecessary_burden": {
                        "type": "array",
                        "items": nonempty_string(),
                    },
                    "agency_issue": {
                        "type": "array",
                        "items": nonempty_string(),
                    },
                }
            ),
        }
    )


def proposal_schema() -> dict[str, Any]:
    return closed_object(
        {
            "node_directives": {
                "type": "object",
                "properties": {
                    node: nonempty_string() for node in MUTABLE_NODES
                },
                "required": [],
                "additionalProperties": False,
            },
            "mutation_rationale": nonempty_string(),
        }
    )


def format_conversation(messages: list[dict[str, Any]], max_chars: int = 52000) -> str:
    formatted = "\n\n".join(
        f"{str(message.get('role', '')).upper()}: {message.get('content', '')}"
        for message in messages
    )
    if len(formatted) > max_chars:
        raise ValueError(
            f"Parent transcript has {len(formatted)} characters; exceeds {max_chars}"
        )
    return formatted


def discover_parent_cases(parent_root: Path, expected_count: int) -> list[dict[str, Any]]:
    cases = []
    for conversation_path in sorted(
        (parent_root / "transcripts" / "scripted").glob("*/*/conversation.json")
    ):
        case_id = conversation_path.parent.parent.name
        vignette_path = conversation_path.parent / "vignette.txt"
        conversation = json.loads(conversation_path.read_text(encoding="utf-8"))
        if not isinstance(conversation, list) or not vignette_path.is_file():
            raise ValueError(f"Incomplete parent transcript: {conversation_path}")
        cases.append(
            {
                "case_id": case_id,
                "vignette": vignette_path.read_text(encoding="utf-8").strip(),
                "conversation": conversation,
                "conversation_sha256": sha256_file(conversation_path),
            }
        )
    if len(cases) != expected_count:
        raise ValueError(
            f"Expected {expected_count} cases under {parent_root}, found {len(cases)}"
        )
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError(f"Duplicate case IDs under {parent_root}")
    return cases


def reflection_prompt(
    case: dict[str, Any],
    rubric: str,
    taxonomy: dict[str, Any],
    prior_judge_feedback: list[dict[str, Any]] | None = None,
) -> str:
    categories = [
        {"id": item["id"], "definition": item["definition"]}
        for item in taxonomy["categories"]
    ]
    schema_example = {
        "case_id": case["case_id"],
        "strengths": ["transcript-grounded strength"],
        "failure_modes": [
            {
                "turn": 1,
                "rubric_domain": "one of the eight rubric domains",
                "evidence": "short transcript excerpt",
                "problem": "specific observed problem",
                "target_node": "system, background, education, questions, or wrap_up",
                "recommended_prompt_change": "bounded prompt-level remedy",
            }
        ],
        "potential_critical_errors": [
            {
                "category_id": "one listed critical-error category",
                "turn": 1,
                "evidence": "short excerpt",
                "rationale": "why human review would be needed",
            }
        ],
        "patient_alignment": {
            "answered_latest_questions": "yes, partly, or no",
            "unnecessary_burden": ["specific burden or repetition"],
            "agency_issue": ["specific autonomy or preference issue"],
        },
    }
    feedback_text = ""
    if prior_judge_feedback:
        feedback_text = (
            "\n\nPrior blinded development-judge nominations for this exact case "
            "(unadjudicated signals, not confirmed errors):\n"
            f"{json.dumps(prior_judge_feedback, ensure_ascii=False, indent=2)}\n"
            "Assess these signals against the transcript and case facts. Preserve a "
            "useful warning even if you would phrase or categorize it differently."
        )
    return (
        "Act as a strict development-only patient-alignment and clinical-counselling "
        "reviewer. Review the complete synthetic transcript. Identify concrete failures "
        "that can be addressed by changing one of five mutable prompt components. "
        "Do not infer that absence of a flag proves safety. Return JSON only, following "
        "the schema exactly.\n\n"
        f"Required schema:\n{json.dumps(schema_example, indent=2)}\n\n"
        f"Rubric:\n{rubric}\n\n"
        f"Potential critical-error categories:\n{json.dumps(categories, indent=2)}\n\n"
        f"Case background:\n{case['vignette']}\n\n"
        f"Complete transcript:\n{format_conversation(case['conversation'])}"
        f"{feedback_text}"
    )


def proposal_prompt(
    reflections: list[dict[str, Any]],
    previous_candidates: list[dict[str, Any]],
    candidate_id: str,
    parent_id: str,
    generation: int,
    prior_judge_feedback: list[dict[str, Any]] | None = None,
) -> str:
    schema_example = {
        "node_directives": {"questions": "one concise imperative directive"},
        "mutation_rationale": "specific failure-to-mutation causal hypothesis",
    }
    prior = [
        {
            "candidate_id": candidate["candidate_id"],
            "node_directives": candidate["node_directives"],
            "mutation_rationale": candidate["mutation_rationale"],
        }
        for candidate in previous_candidates
    ]
    feedback_text = ""
    if prior_judge_feedback:
        feedback_text = (
            "\n\nPrior blinded development-judge nominations for this parent "
            "(unadjudicated signals, not confirmed errors):\n"
            f"{json.dumps(prior_judge_feedback, ensure_ascii=False)}"
        )
    return (
        "You are the prompt-mutation component of SPJ-Evolve. Propose exactly one "
        "testable prompt overlay for an offline synthetic development experiment. The "
        "overlay may add directives to system, background, education, questions, or wrap_up. "
        "It cannot weaken deterministic safety controls or modify the knowledge base, graph, "
        "rubric, or evaluator. Make the smallest plausible mutation linked to observed "
        "patient/judge feedback. Do not add clinical facts. Do not duplicate a previous "
        "candidate. Use at most two node directives and at most 500 characters per directive. "
        "Return JSON only.\n\n"
        f"Candidate identity assigned by the runner: {candidate_id}; parent={parent_id}; "
        f"generation={generation}.\n\n"
        f"Required output schema:\n{json.dumps(schema_example, indent=2)}\n\n"
        f"Patient-judge reflections:\n{json.dumps(reflections, ensure_ascii=False)}\n\n"
        f"Already proposed candidates:\n{json.dumps(prior, ensure_ascii=False)}"
        f"{feedback_text}"
    )


def load_lesson_context(path: Path, parent_id: str, generation: int) -> dict[str, Any]:
    """Load a frozen proposer-only register; never return runtime directives."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Lesson register must contain a JSON object")
    if "inputs" in payload:
        register = payload["inputs"]
        if not isinstance(register, dict) or payload.get("sha256") != canonical_sha256(register):
            raise ValueError("Lesson register authentication hash does not match its content")
    else:
        register = payload
    if register.get("register_type") != "proposer_only_development_memory":
        raise ValueError("Lesson register must have type proposer_only_development_memory")
    active_round = register.get("active_from_round", 1)
    if isinstance(active_round, bool) or not isinstance(active_round, int) or active_round < 1:
        raise ValueError("Lesson register active_from_round must be a positive integer")
    if generation < active_round:
        raise ValueError("Lesson register is not active for this proposal generation")
    if not isinstance(register.get("lessons"), list):
        raise ValueError("Lesson register lessons must be a list")
    try:
        return proposer_context(register, parent_id)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Lesson register contains malformed lesson observations") from exc


def load_feedback_bundle(
    path: Path,
    *,
    parent_ids: list[str],
    analysis_path: Path,
) -> dict[str, Any]:
    bundle = json.loads(path.read_text(encoding="utf-8"))
    if bundle.get("status") != "FROZEN_DEVELOPMENT_FEEDBACK_NOT_HUMAN_ADJUDICATED":
        raise ValueError("Feedback bundle is not frozen at the required boundary")
    if bundle.get("parent_ids") != parent_ids:
        raise ValueError("Feedback parent order differs from the frozen parent plan")
    if set(bundle.get("parents", {})) != set(parent_ids):
        raise ValueError("Feedback bundle parent roster is inconsistent")
    inputs = bundle.get("inputs", {})
    if inputs.get("analysis_sha256") != sha256_file(analysis_path):
        raise ValueError("Feedback bundle does not authenticate the parent analysis")
    for parent_id in parent_ids:
        parent = bundle["parents"][parent_id]
        nominations = parent.get("judge_nominations")
        if not isinstance(nominations, list):
            raise ValueError(f"Feedback nominations are invalid for {parent_id}")
        if parent.get("nomination_count") != len(nominations):
            raise ValueError(f"Feedback nomination count mismatch for {parent_id}")
    return bundle


def write_prepared_batch(
    *,
    batch_dir: Path,
    requests: list[dict[str, Any]],
    records: list[dict[str, Any]],
    model: str,
    effort: str,
    max_output_tokens: int,
    study_id: str,
    stage: str,
    metadata: dict[str, Any],
    api_base: str | None = None,
) -> dict[str, Any]:
    config = native_config("proposal", provider="anthropic", model=model, effort=effort,
                           max_tokens=max_output_tokens, api_base=api_base).validate()
    if not requests or len(requests) != len(records):
        raise ValueError("Prepared requests and provenance records must have equal nonzero counts")
    if batch_dir.exists():
        raise FileExistsError(f"Refusing to overwrite prepared batch: {batch_dir}")
    batch_dir.mkdir(parents=True)
    requests_path = batch_dir / "requests.jsonl"
    requests_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in requests),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "1.0.0",
        "created_at": utc_now(),
        "status": "PREPARED_NOT_SUBMITTED",
        "study_id": study_id,
        "method": "SPJ-Evolve",
        "stage": stage,
        "provider": "anthropic",
        "model_role": "proposal",
        "api_base": config.api_base,
        "generator": {
            "id": "spj-proposer",
            "role": "development_reflector_and_prompt_mutator",
            "model": model,
            "effort": effort,
            "structured_outputs": True,
        },
        "request_count": len(requests),
        "max_output_tokens_per_request": max_output_tokens,
        "batch_input_path": requests_path.name,
        "batch_input_sha256": sha256_file(requests_path),
        "records": records,
        "independence": {
            "prior_lineage_reflections_reused": False,
            "prior_lineage_candidates_visible_to_generator": False,
            "frontier_judge_outputs_visible_to_generator": False,
        },
        **metadata,
        "git": git_state(),
    }
    write_json(batch_dir / "batch_manifest.json", manifest)
    return manifest


def load_results(results_path: Path) -> dict[str, dict[str, Any]]:
    supplied = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        custom_id = item.get("custom_id")
        if not isinstance(custom_id, str) or custom_id in supplied:
            raise ValueError("Batch results contain a missing or duplicate custom_id")
        supplied[custom_id] = item
    return supplied


def extract_anthropic_json(line: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    result = line.get("result") or {}
    if result.get("type") != "succeeded":
        raise ValueError(f"Anthropic result type is {result.get('type')!r}")
    message = result.get("message") or {}
    texts = [
        block["text"]
        for block in message.get("content", [])
        if block.get("type") == "text" and isinstance(block.get("text"), str)
    ]
    if len(texts) != 1:
        raise ValueError(f"Expected one text block, found {len(texts)}")
    parsed = json.loads(texts[0])
    if not isinstance(parsed, dict):
        raise ValueError("Structured result is not a JSON object")
    metadata = {
        "response_id": message.get("id"),
        "response_model": message.get("model"),
        "stop_reason": message.get("stop_reason"),
        "usage": message.get("usage", {}),
        "raw_text_sha256": canonical_sha256(texts[0]),
        "execution_mode": line.get(
            "execution_mode", "anthropic_message_batches_api"
        ),
    }
    return parsed, metadata


def validate_reflection(
    value: dict[str, Any], case_id: str, category_ids: set[str]
) -> dict[str, Any]:
    expected = {
        "case_id",
        "strengths",
        "failure_modes",
        "potential_critical_errors",
        "patient_alignment",
    }
    if set(value) != expected or value.get("case_id") != case_id:
        raise ValueError(f"Reflection schema or case_id mismatch for {case_id}")
    if not isinstance(value["strengths"], list) or not all(
        isinstance(item, str) and item.strip() for item in value["strengths"]
    ):
        raise ValueError(f"Reflection strengths are invalid for {case_id}")
    if not isinstance(value["failure_modes"], list):
        raise ValueError(f"Reflection failure_modes are invalid for {case_id}")
    failure_keys = {
        "turn",
        "rubric_domain",
        "evidence",
        "problem",
        "target_node",
        "recommended_prompt_change",
    }
    for failure in value["failure_modes"]:
        if not isinstance(failure, dict) or set(failure) != failure_keys:
            raise ValueError(f"Reflection failure mode schema is invalid for {case_id}")
        if (
            isinstance(failure["turn"], bool)
            or not isinstance(failure["turn"], int)
            or failure["turn"] < 1
            or failure["rubric_domain"] not in RUBRIC_DOMAINS
            or failure["target_node"] not in MUTABLE_NODES
        ):
            raise ValueError(f"Reflection failure mode values are invalid for {case_id}")
        for field in ("evidence", "problem", "recommended_prompt_change"):
            if not isinstance(failure[field], str) or not failure[field].strip():
                raise ValueError(f"Reflection failure mode text is invalid for {case_id}")
    if not isinstance(value["potential_critical_errors"], list):
        raise ValueError(f"Reflection critical errors are invalid for {case_id}")
    critical_keys = {"category_id", "turn", "evidence", "rationale"}
    for critical in value["potential_critical_errors"]:
        if not isinstance(critical, dict) or set(critical) != critical_keys:
            raise ValueError(f"Reflection critical-error schema is invalid for {case_id}")
        if (
            critical["category_id"] not in category_ids
            or isinstance(critical["turn"], bool)
            or not isinstance(critical["turn"], int)
            or critical["turn"] < 1
        ):
            raise ValueError(f"Reflection critical-error values are invalid for {case_id}")
        for field in ("evidence", "rationale"):
            if not isinstance(critical[field], str) or not critical[field].strip():
                raise ValueError(f"Reflection critical-error text is invalid for {case_id}")
    alignment = value["patient_alignment"]
    if (
        not isinstance(alignment, dict)
        or set(alignment)
        != {"answered_latest_questions", "unnecessary_burden", "agency_issue"}
        or alignment.get("answered_latest_questions") not in {"yes", "partly", "no"}
    ):
        raise ValueError(f"Reflection patient_alignment is invalid for {case_id}")
    for field in ("unnecessary_burden", "agency_issue"):
        if not isinstance(alignment[field], list) or not all(
            isinstance(item, str) and item.strip() for item in alignment[field]
        ):
            raise ValueError(f"Reflection patient_alignment text is invalid for {case_id}")
    return value


def command_prepare_reflections(args: argparse.Namespace) -> int:
    config = native_config("proposal", model=args.model, effort=args.effort,
                           max_tokens=args.max_output_tokens)
    if config.provider != "anthropic":
        raise ValueError("This proposer uses Anthropic Messages. Set MND_PROPOSAL_PROVIDER=anthropic.")
    config.validate()
    args.model, args.effort, args.max_output_tokens = config.model, config.effort, config.max_tokens
    analysis_path = args.parent_analysis_dir / "summary.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    available = {item["candidate_id"] for item in analysis["candidates"]}
    if not 1 <= len(args.parent_ids) <= 2 or len(set(args.parent_ids)) != len(args.parent_ids):
        raise ValueError("One or two distinct --parent-id values are required")
    if args.expected_parent_case_count < 1:
        raise ValueError("expected-parent-case-count must be positive")
    if not set(args.parent_ids) <= available:
        raise ValueError("A requested parent is absent from the frozen analysis")
    rubric = RUBRIC_PATH.read_text(encoding="utf-8").strip()
    taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    category_ids = sorted(item["id"] for item in taxonomy["categories"])
    feedback_bundle = None
    if args.feedback_bundle is not None:
        feedback_bundle = load_feedback_bundle(
            args.feedback_bundle,
            parent_ids=args.parent_ids,
            analysis_path=analysis_path,
        )
    if not args.study_registry.is_file():
        raise FileNotFoundError(f"Study registry does not exist: {args.study_registry}")
    requests = []
    records = []
    parent_case_hashes = {}
    for parent_id in args.parent_ids:
        parent_root = args.parent_sweep_dir / "candidates" / parent_id
        cases = discover_parent_cases(parent_root, args.expected_parent_case_count)
        parent_case_hashes[parent_id] = {
            case["case_id"]: case["conversation_sha256"] for case in cases
        }
        for case in cases:
            custom_id = f"reflection-{parent_id}-{case['case_id']}"
            case_feedback = None
            if feedback_bundle is not None:
                case_feedback = [
                    item
                    for item in feedback_bundle["parents"][parent_id][
                        "judge_nominations"
                    ]
                    if item["case_id"] == case["case_id"]
                ]
            prompt = reflection_prompt(
                case,
                rubric,
                taxonomy,
                prior_judge_feedback=case_feedback,
            )
            schema = reflection_schema(case["case_id"], category_ids)
            request = build_provider_request(
                provider="anthropic",
                custom_id=custom_id,
                prompt=prompt,
                schema=schema,
                model=args.model,
                effort=args.effort,
                max_output_tokens=args.max_output_tokens,
            )
            request["params"]["system"] = SYSTEM_REFLECTION
            requests.append(request)
            records.append(
                {
                    "custom_id": custom_id,
                    "parent_id": parent_id,
                    "case_id": case["case_id"],
                    "prompt_sha256": canonical_sha256(prompt),
                    "output_json_schema_sha256": canonical_sha256(schema),
                    "conversation_sha256": case["conversation_sha256"],
                    "request_sha256": canonical_sha256(request),
                }
            )
    manifest = write_prepared_batch(
        batch_dir=args.output_root / "reflection-batch-prepared",
        requests=requests,
        records=records,
        model=args.model,
        effort=args.effort,
        max_output_tokens=args.max_output_tokens,
        study_id=args.study_id,
        stage="independent_reflection",
        api_base=config.api_base,
        metadata={
            "generation": args.generation,
            "proposal_configuration": config.public_dict(),
            "candidate_prefix": args.candidate_prefix,
            "parent_ids": args.parent_ids,
            "parent_analysis_path": repo_relative(analysis_path),
            "parent_analysis_sha256": sha256_file(analysis_path),
            "parent_sweep_path": repo_relative(args.parent_sweep_dir),
            "parent_case_hashes": parent_case_hashes,
            "rubric_sha256": sha256_file(RUBRIC_PATH),
            "critical_error_taxonomy_sha256": sha256_file(TAXONOMY_PATH),
            "registry_path": repo_relative(args.study_registry),
            "registry_sha256": sha256_file(args.study_registry),
            "feedback_bundle_path": (
                repo_relative(args.feedback_bundle)
                if args.feedback_bundle is not None
                else None
            ),
            "feedback_bundle_sha256": (
                sha256_file(args.feedback_bundle)
                if args.feedback_bundle is not None
                else None
            ),
            "independence": {
                "prior_lineage_reflections_reused": False,
                "prior_lineage_candidates_visible_to_generator": False,
                "frontier_judge_outputs_visible_to_generator": (
                    feedback_bundle is not None
                ),
                "judge_feedback_use": (
                    "intentional_adaptive_development_feedback"
                    if feedback_bundle is not None
                    else "none"
                ),
            },
        },
    )
    print(json.dumps({"request_count": len(requests), "batch_input_sha256": manifest["batch_input_sha256"]}, indent=2))
    return 0


def command_import_reflections(args: argparse.Namespace) -> int:
    batch_dir = args.output_root / "reflection-batch-prepared"
    manifest_path = batch_dir / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(batch_dir / manifest["batch_input_path"]) != manifest["batch_input_sha256"]:
        raise ValueError("Reflection batch input hash mismatch")
    if manifest.get("critical_error_taxonomy_sha256") != sha256_file(TAXONOMY_PATH):
        raise ValueError("Critical-error taxonomy changed after reflection preparation")
    expected = {record["custom_id"]: record for record in manifest["records"]}
    supplied = load_results(args.results)
    if set(supplied) != set(expected):
        raise ValueError("Reflection result IDs do not match prepared requests")
    result_source_by_id = {custom_id: args.results for custom_id in supplied}
    # Preserve the original attempt separately so corrective usage is counted once.
    all_result_sets = [dict(supplied)]
    for corrective_path in args.corrective_results:
        corrective = load_results(corrective_path)
        if not set(corrective) <= set(expected):
            raise ValueError("Corrective reflection results contain unknown request IDs")
        supplied.update(corrective)
        all_result_sets.append(corrective)
        result_source_by_id.update(
            {custom_id: corrective_path for custom_id in corrective}
        )
    reflection_dir = args.output_root / "reflections"
    if reflection_dir.exists():
        raise FileExistsError(f"Refusing to overwrite reflections: {reflection_dir}")
    category_ids = {
        item["id"]
        for item in json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))["categories"]
    }
    validated = []
    for custom_id, source in expected.items():
        value, response = extract_anthropic_json(supplied[custom_id])
        value = validate_reflection(value, source["case_id"], category_ids)
        validated.append((custom_id, source, value, response))

    # Validate every expected reflection before creating any output files.
    reflection_dir.mkdir(parents=True)
    records = []
    aggregate_selected_usage: dict[str, int] = {}
    aggregate_all_attempt_usage: dict[str, int] = {}
    for result_set in all_result_sets:
        for line in result_set.values():
            message = (line.get("result") or {}).get("message") or {}
            for key, amount in (message.get("usage") or {}).items():
                if isinstance(amount, int) and not isinstance(amount, bool):
                    aggregate_all_attempt_usage[key] = (
                        aggregate_all_attempt_usage.get(key, 0) + amount
                    )
    for custom_id, source, value, response in validated:
        output_path = reflection_dir / f"{source['parent_id']}-{source['case_id']}.json"
        write_json(output_path, value)
        for key, amount in response["usage"].items():
            if isinstance(amount, int) and not isinstance(amount, bool):
                aggregate_selected_usage[key] = (
                    aggregate_selected_usage.get(key, 0) + amount
                )
        records.append(
            {
                **source,
                **response,
                "result_source_path": repo_relative(result_source_by_id[custom_id]),
                "reflection_path": repo_relative(output_path),
                "reflection_sha256": sha256_file(output_path),
            }
        )
    import_manifest = {
        "schema_version": "1.0.0",
        "created_at": utc_now(),
        "status": "COMPLETE",
        "provider": "anthropic",
        "model": manifest["generator"]["model"],
        "effort": manifest["generator"]["effort"],
        "completed_reflections": len(records),
        "prepared_batch_manifest_sha256": sha256_file(manifest_path),
        "batch_results_sha256": sha256_file(args.results),
        "corrective_batch_results": [
            {"path": repo_relative(path), "sha256": sha256_file(path)}
            for path in args.corrective_results
        ],
        "aggregate_selected_usage": aggregate_selected_usage,
        "aggregate_all_attempt_usage": aggregate_all_attempt_usage,
        "attempts": [
            {"results_path": repo_relative(path), "results_sha256": sha256_file(path),
             "usage": _usage_sum([((line.get("result") or {}).get("message") or {}).get("usage") or {}
                                  for line in result_set.values()])}
            for path, result_set in zip([args.results, *args.corrective_results], all_result_sets)
        ],
        "records": records,
        "git": git_state(),
    }
    write_json(args.output_root / "reflection_import_manifest.json", import_manifest)
    print(
        json.dumps(
            {
                "completed_reflections": len(records),
                "aggregate_selected_usage": aggregate_selected_usage,
                "aggregate_all_attempt_usage": aggregate_all_attempt_usage,
            },
            indent=2,
        )
    )
    return 0


def command_prepare_corrective_reflections(args: argparse.Namespace) -> int:
    original_batch_dir = args.output_root / "reflection-batch-prepared"
    original_manifest_path = original_batch_dir / "batch_manifest.json"
    original = json.loads(original_manifest_path.read_text(encoding="utf-8"))
    if args.max_output_tokens is None:
        args.max_output_tokens = original["max_output_tokens_per_request"]
    requests = {
        item["custom_id"]: item
        for item in (
            json.loads(line)
            for line in (original_batch_dir / original["batch_input_path"])
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }
    expected = {record["custom_id"]: record for record in original["records"]}
    supplied = load_results(args.results)
    if set(supplied) != set(expected):
        raise ValueError("Original reflection result IDs do not match prepared requests")
    category_ids = {
        item["id"]
        for item in json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))["categories"]
    }
    corrective_requests = []
    corrective_records = []
    for custom_id, source in expected.items():
        prior_error = None
        try:
            value, _ = extract_anthropic_json(supplied[custom_id])
            validate_reflection(value, source["case_id"], category_ids)
        except Exception as exc:
            prior_error = f"{type(exc).__name__}: {exc}"
        if prior_error is None:
            continue
        request = json.loads(json.dumps(requests[custom_id]))
        original_prompt = request["params"]["messages"][0]["content"]
        corrected_prompt = original_prompt + (
            "\n\nCorrective retry: the preceding response exhausted its output budget or "
            "failed validation. Return one concise JSON object matching the required "
            "schema. Use no more than three strengths, five failure modes, and three "
            "potential critical errors. Keep each evidence, problem, rationale, and "
            "recommendation to at most two concise sentences. Do not emit commentary "
            "or markdown."
        )
        request["params"]["messages"][0]["content"] = corrected_prompt
        request["params"]["max_tokens"] = args.max_output_tokens
        corrective_requests.append(request)
        message = (supplied[custom_id].get("result") or {}).get("message") or {}
        corrective_records.append(
            {
                **source,
                "original_prompt_sha256": source["prompt_sha256"],
                "prompt_sha256": canonical_sha256(corrected_prompt),
                "original_request_sha256": source["request_sha256"],
                "request_sha256": canonical_sha256(request),
                "prior_stop_reason": message.get("stop_reason"),
                "prior_error": prior_error,
            }
        )
    if not corrective_requests:
        raise ValueError("No invalid reflection results require a corrective batch")
    manifest = write_prepared_batch(
        batch_dir=args.output_root / "reflection-corrective-1-prepared",
        requests=corrective_requests,
        records=corrective_records,
        model=original["generator"]["model"],
        effort=original["generator"]["effort"],
        max_output_tokens=args.max_output_tokens,
        study_id=original["study_id"],
        stage="independent_reflection_corrective_retry",
        api_base=original.get("api_base"),
        metadata={
            "generation": original["generation"],
            "candidate_prefix": original["candidate_prefix"],
            "parent_ids": original["parent_ids"],
            "parent_analysis_path": original["parent_analysis_path"],
            "parent_analysis_sha256": original["parent_analysis_sha256"],
            "parent_case_hashes": original["parent_case_hashes"],
            "feedback_bundle_path": original.get("feedback_bundle_path"),
            "feedback_bundle_sha256": original.get("feedback_bundle_sha256"),
            "independence": original["independence"],
            "registry_sha256": original["registry_sha256"],
            "corrective_retry": {
                "criteria_changed_from_original": False,
                "schema_changed_from_original": False,
                "prompt_unchanged": False,
                "only_invalid_requests": True,
                "prompt_change_scope": "concise-output and format correction only",
                "original_max_output_tokens": original[
                    "max_output_tokens_per_request"
                ],
                "corrective_max_output_tokens": args.max_output_tokens,
                "original_batch_manifest_sha256": sha256_file(original_manifest_path),
                "original_results_sha256": sha256_file(args.results),
            },
        },
    )
    print(
        json.dumps(
            {
                "corrective_request_count": len(corrective_requests),
                "batch_input_sha256": manifest["batch_input_sha256"],
            },
            indent=2,
        )
    )
    return 0


def candidate_plan(
    parent_ids: list[str], candidate_prefix: str, offspring_per_parent: int
) -> list[tuple[str, str]]:
    if offspring_per_parent < 1:
        raise ValueError("offspring-per-parent must be positive")
    return [
        (f"{candidate_prefix}-c{index + 1:02d}", parent_id)
        for index, parent_id in enumerate(
            parent_id
            for parent_id in parent_ids
            for _ in range(offspring_per_parent)
        )
    ]


def command_prepare_proposal(args: argparse.Namespace) -> int:
    reflection_manifest = json.loads(
        (args.output_root / "reflection_import_manifest.json").read_text(encoding="utf-8")
    )
    reflection_batch = json.loads(
        (args.output_root / "reflection-batch-prepared/batch_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if reflection_manifest.get("prepared_batch_manifest_sha256") != sha256_file(
        args.output_root / "reflection-batch-prepared/batch_manifest.json"
    ):
        raise ValueError("Reflection import does not authenticate the prepared batch")
    for record in reflection_manifest["records"]:
        if sha256_file(REPO_ROOT / record["reflection_path"]) != record["reflection_sha256"]:
            raise ValueError("Imported reflection changed after validation")
    expected_reflections = len(reflection_batch["parent_ids"]) * args.expected_parent_case_count
    if (reflection_manifest.get("completed_reflections") != expected_reflections
            or reflection_batch["request_count"] != expected_reflections):
        raise ValueError(f"Exactly {expected_reflections} imported independent reflections are required")
    if args.max_output_tokens is None:
        args.max_output_tokens = reflection_batch["max_output_tokens_per_request"]
    plan = candidate_plan(
        reflection_batch["parent_ids"],
        reflection_batch["candidate_prefix"],
        args.offspring_per_parent,
    )
    if args.candidate_index < 1 or args.candidate_index > len(plan):
        raise ValueError("candidate-index is outside the frozen candidate plan")
    candidate_id, parent_id = plan[args.candidate_index - 1]
    for prior_candidate_id, _ in plan[: args.candidate_index - 1]:
        if not (args.output_root / "candidates" / f"{prior_candidate_id}.json").is_file():
            raise ValueError(f"Required earlier candidate is missing: {prior_candidate_id}")
    if (args.output_root / "candidates" / f"{candidate_id}.json").exists():
        raise FileExistsError(f"Candidate already imported: {candidate_id}")

    reflections = []
    for path in sorted((args.output_root / "reflections").glob(f"{parent_id}-*.json")):
        reflections.append(json.loads(path.read_text(encoding="utf-8")))
    if len(reflections) != args.expected_parent_case_count:
        raise ValueError(
            f"Expected {args.expected_parent_case_count} reflections for {parent_id}, "
            f"found {len(reflections)}"
        )
    parent_candidate_path = (
        args.parent_sweep_dir / "candidates" / parent_id / "prompt_candidate.json"
    )
    parent_candidate = validate_prompt_candidate(
        json.loads(parent_candidate_path.read_text(encoding="utf-8"))
    )
    parent_feedback = None
    feedback_path_value = reflection_batch.get("feedback_bundle_path")
    if feedback_path_value is not None:
        feedback_path = REPO_ROOT / feedback_path_value
        if sha256_file(feedback_path) != reflection_batch["feedback_bundle_sha256"]:
            raise ValueError("Feedback bundle hash changed after reflection preparation")
        feedback_bundle = load_feedback_bundle(
            feedback_path,
            parent_ids=reflection_batch["parent_ids"],
            analysis_path=REPO_ROOT / reflection_batch["parent_analysis_path"],
        )
        parent_feedback = feedback_bundle["parents"][parent_id][
            "judge_nominations"
        ]
    previous_candidates = [parent_candidate]
    for prior_candidate_id, _ in plan[: args.candidate_index - 1]:
        previous_candidates.append(
            validate_prompt_candidate(json.loads(
                (args.output_root / "candidates" / f"{prior_candidate_id}.json").read_text(
                    encoding="utf-8"
                )
            ))
        )
    prompt = proposal_prompt(
        reflections,
        previous_candidates,
        candidate_id,
        parent_id,
        reflection_batch["generation"],
        prior_judge_feedback=parent_feedback,
    )
    lesson_context = None
    if args.lesson_register is not None:
        lesson_context = load_lesson_context(
            args.lesson_register, parent_id, reflection_batch["generation"]
        )
        prompt += (
            "\n\nHistorical proposer-only development memory (tentative repair context; "
            "this does not change the mutation scope or directive budget):\n"
            + json.dumps(lesson_context, ensure_ascii=False)
        )
    schema = proposal_schema()
    request = build_provider_request(
        provider="anthropic",
        custom_id=candidate_id,
        prompt=prompt,
        schema=schema,
        model=reflection_batch["generator"]["model"],
        effort=reflection_batch["generator"]["effort"],
        max_output_tokens=args.max_output_tokens,
    )
    request["params"]["system"] = SYSTEM_PROPOSAL
    batch_dir = args.output_root / "proposal-batches" / f"{candidate_id}-prepared"
    manifest = write_prepared_batch(
        batch_dir=batch_dir,
        requests=[request],
        records=[
            {
                "custom_id": candidate_id,
                "candidate_id": candidate_id,
                "candidate_index": args.candidate_index,
                "parent_id": parent_id,
                "prompt_sha256": canonical_sha256(prompt),
                "output_json_schema_sha256": canonical_sha256(schema),
                "request_sha256": canonical_sha256(request),
            }
        ],
        model=reflection_batch["generator"]["model"],
        effort=reflection_batch["generator"]["effort"],
        max_output_tokens=args.max_output_tokens,
        study_id=reflection_batch["study_id"],
        stage="sequential_prompt_mutation",
        api_base=reflection_batch.get("api_base"),
        metadata={
            "generation": reflection_batch["generation"],
            "candidate_prefix": reflection_batch["candidate_prefix"],
            "candidate_index": args.candidate_index,
            "candidate_id": candidate_id,
            "parent_id": parent_id,
            "parent_ids": reflection_batch["parent_ids"],
            "parent_analysis_sha256": reflection_batch["parent_analysis_sha256"],
            "parent_case_hashes": reflection_batch["parent_case_hashes"][parent_id],
            "parent_candidate_sha256": sha256_file(parent_candidate_path),
            "reflection_import_manifest_sha256": sha256_file(
                args.output_root / "reflection_import_manifest.json"
            ),
            "previous_candidate_ids": [
                candidate["candidate_id"] for candidate in previous_candidates
            ],
            "proposal_schedule": "sequential_dependency_preserving_provider_batches",
            "lesson_register_path": repo_relative(args.lesson_register) if args.lesson_register is not None else None,
            "lesson_register_sha256": sha256_file(args.lesson_register) if args.lesson_register is not None else None,
            "lesson_context_sha256": canonical_sha256(lesson_context) if lesson_context is not None else None,
            "feedback_bundle_path": reflection_batch.get("feedback_bundle_path"),
            "feedback_bundle_sha256": reflection_batch.get(
                "feedback_bundle_sha256"
            ),
            "parent_judge_nomination_count": (
                len(parent_feedback) if parent_feedback is not None else 0
            ),
        },
    )
    print(json.dumps({"candidate_id": candidate_id, "batch_input_sha256": manifest["batch_input_sha256"]}, indent=2))
    return 0


def command_import_proposal(args: argparse.Namespace) -> int:
    batch_dir = args.output_root / "proposal-batches" / f"{args.candidate_id}-prepared"
    manifest_path = batch_dir / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("candidate_id") != args.candidate_id:
        raise ValueError("Candidate ID does not match its prepared batch")
    if sha256_file(batch_dir / manifest["batch_input_path"]) != manifest["batch_input_sha256"]:
        raise ValueError("Proposal batch input hash mismatch")
    supplied = load_results(args.results)
    if set(supplied) != {args.candidate_id}:
        raise ValueError("Proposal batch result ID mismatch")
    proposal, response = extract_anthropic_json(supplied[args.candidate_id])
    if set(proposal) != {"node_directives", "mutation_rationale"}:
        raise ValueError("Proposal output has unexpected fields")
    delta = proposal["node_directives"]
    if not isinstance(delta, dict) or not 1 <= len(delta) <= 2:
        raise ValueError("Proposal must contain one or two node directives")
    for node, directive in delta.items():
        if node not in MUTABLE_NODES or not isinstance(directive, str):
            raise ValueError("Proposal contains an invalid node directive")
        if not directive.strip() or len(directive) > 500:
            raise ValueError("Proposal directive violates the 1..500 character bound")

    parent_path = (
        args.parent_sweep_dir
        / "candidates"
        / manifest["parent_id"]
        / "prompt_candidate.json"
    )
    if sha256_file(parent_path) != manifest["parent_candidate_sha256"]:
        raise ValueError("Parent candidate changed after proposal preparation")
    parent = validate_prompt_candidate(json.loads(parent_path.read_text(encoding="utf-8")))
    lesson_path_value = manifest.get("lesson_register_path")
    if lesson_path_value is not None:
        lesson_path = REPO_ROOT / lesson_path_value
        if sha256_file(lesson_path) != manifest["lesson_register_sha256"]:
            raise ValueError("Lesson register snapshot changed after proposal preparation")
        context = load_lesson_context(lesson_path, manifest["parent_id"], manifest["generation"])
        if canonical_sha256(context) != manifest["lesson_context_sha256"]:
            raise ValueError("Lesson register context changed after proposal preparation")
    inherited = dict(parent["node_directives"])
    inherited.update(delta)
    if inherited == parent["node_directives"]:
        raise ValueError("Proposal duplicates the unchanged parent directives")
    candidate = validate_prompt_candidate(
        {
            "schema_version": "1.0.0",
            "candidate_id": args.candidate_id,
            "parent_id": manifest["parent_id"],
            "generation": manifest["generation"],
            "node_directives": inherited,
            "mutation_rationale": proposal["mutation_rationale"],
            "proposal_provenance": {
                "provider": manifest["provider"],
                "model_key": manifest["generator"]["model"],
                "requested_model": manifest["generator"]["model"],
                "response_model": response["response_model"],
                "effort": manifest["generator"]["effort"],
                "execution_mode": response["execution_mode"],
                "proposal_schedule": (
                    "sequential_dependency_preserving_standard_api"
                    if response["execution_mode"]
                    == "anthropic_messages_standard_api"
                    else manifest["proposal_schedule"]
                ),
                "candidate_index": manifest["candidate_index"],
                "request_sha256": manifest["records"][0]["request_sha256"],
                "response_id": response["response_id"],
                "parent_case_hashes": manifest["parent_case_hashes"],
                "parent_analysis_sha256": manifest["parent_analysis_sha256"],
                "feedback_bundle_path": manifest.get("feedback_bundle_path"),
                "feedback_bundle_sha256": manifest.get(
                    "feedback_bundle_sha256"
                ),
                "parent_judge_nomination_count": manifest.get(
                    "parent_judge_nomination_count", 0
                ),
                "lesson_register_path": manifest.get("lesson_register_path"),
                "lesson_register_sha256": manifest.get("lesson_register_sha256"),
                "lesson_context_sha256": manifest.get("lesson_context_sha256"),
                "mutation_delta": delta,
            },
        }
    )
    candidate_dir = args.output_root / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for prior_path in sorted(candidate_dir.glob("*.json")):
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        if prior["node_directives"] == candidate["node_directives"]:
            raise ValueError(f"Proposal duplicates earlier candidate {prior['candidate_id']}")
    candidate_path = candidate_dir / f"{args.candidate_id}.json"
    if candidate_path.exists():
        raise FileExistsError(f"Refusing to overwrite candidate: {candidate_path}")
    write_json(candidate_path, candidate)
    import_manifest = {
        "schema_version": "1.0.0",
        "created_at": utc_now(),
        "status": "COMPLETE",
        "candidate_id": args.candidate_id,
        "candidate_sha256": candidate["candidate_sha256"],
        "prepared_batch_manifest_sha256": sha256_file(manifest_path),
        "batch_results_sha256": sha256_file(args.results),
        "response": response,
        "git": git_state(),
    }
    write_json(batch_dir / "import_manifest.json", import_manifest)
    print(json.dumps({"candidate_id": args.candidate_id, "candidate_sha256": candidate["candidate_sha256"], "usage": response["usage"]}, indent=2))
    return 0


def _usage_sum(usages: list[dict[str, Any]]) -> dict[str, int]:
    aggregate: dict[str, int] = {}
    for usage in usages:
        for key, amount in usage.items():
            if isinstance(amount, int) and not isinstance(amount, bool):
                aggregate[key] = aggregate.get(key, 0) + amount
    return aggregate


def _batch_usage_cost(usage: dict[str, Any], pricing: dict[str, Any]) -> float:
    """Use rates recorded for this prepared batch, never a model-name price table."""
    rates = {
        "input_tokens": pricing["effective_batch_input"],
        "output_tokens": pricing["effective_batch_output"],
    }
    for token, rate in (("cache_read_input_tokens", "effective_batch_cache_read_input"),
                        ("cache_creation_input_tokens", "effective_batch_cache_creation_input")):
        if usage.get(token, 0):
            if rate not in pricing:
                raise ValueError(f"Recorded pricing lacks the rate for {token}")
            rates[token] = pricing[rate]
    total = 0.0
    for token, rate in rates.items():
        amount = usage.get(token, 0)
        if (isinstance(amount, bool) or not isinstance(amount, int) or amount < 0
                or isinstance(rate, bool) or not isinstance(rate, (int, float))
                or not math.isfinite(rate) or rate < 0):
            raise ValueError("Token usage and recorded prices must be finite and non-negative")
        total += amount / 1_000_000 * rate
    return total


def _accounting_record(batch_dir: Path) -> dict[str, Any]:
    sync_path = batch_dir / "sync_execution_manifest.json"
    path = sync_path if sync_path.is_file() else batch_dir / "token_count_and_cost.json"
    if not path.is_file():
        raise ValueError(f"Missing recorded token pricing or execution costs: {batch_dir}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("prepared_batch_manifest_sha256") != sha256_file(batch_dir / "batch_manifest.json"):
        raise ValueError(f"Cost record does not authenticate its prepared batch: {path}")
    return {"path": repo_relative(path), "sha256": sha256_file(path),
            "mode": record.get("execution_mode", "anthropic_message_batches_api"),
            "standard_api": path == sync_path, "record": record}


def _accounted_cost(account: dict[str, Any], usage: dict[str, Any]) -> tuple[float, float]:
    record = account["record"]
    if account["standard_api"]:
        actual, maximum = record["actual_cost_usd"], record["hard_maximum_cost_usd"]
    else:
        actual = _batch_usage_cost(usage, record["pricing_usd_per_million_tokens"])
        maximum = record["cost_estimate_usd"]["hard_maximum_total"]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value < 0 for value in (actual, maximum)):
        raise ValueError("Recorded costs must be finite and non-negative")
    return actual, maximum


def command_finalize(args: argparse.Namespace) -> int:
    reflection_batch_path = args.output_root / "reflection-batch-prepared/batch_manifest.json"
    reflection_batch = json.loads(reflection_batch_path.read_text(encoding="utf-8"))
    reflection_import = json.loads(
        (args.output_root / "reflection_import_manifest.json").read_text(encoding="utf-8")
    )
    plan = candidate_plan(reflection_batch["parent_ids"], reflection_batch["candidate_prefix"],
                          args.offspring_per_parent)
    archive_path = args.output_root / "candidate_archive.jsonl"
    manifest_path = args.output_root / "proposal_manifest.json"
    if archive_path.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite a finalized proposal archive")

    candidates = []
    proposal_usages = []
    proposal_accounts = []
    for candidate_id, _ in plan:
        path = args.output_root / "candidates" / f"{candidate_id}.json"
        candidate = validate_prompt_candidate(json.loads(path.read_text(encoding="utf-8")))
        batch_dir = args.output_root / "proposal-batches" / f"{candidate_id}-prepared"
        imported = json.loads((batch_dir / "import_manifest.json").read_text(encoding="utf-8"))
        if (imported.get("candidate_sha256") != candidate["candidate_sha256"]
                or imported.get("prepared_batch_manifest_sha256") != sha256_file(batch_dir / "batch_manifest.json")):
            raise ValueError(f"Candidate import no longer matches prepared inputs: {candidate_id}")
        candidates.append(candidate)
        proposal_usages.append(imported["response"]["usage"])
        proposal_accounts.append(_accounting_record(batch_dir))

    reflection_dirs = [args.output_root / "reflection-batch-prepared"] + sorted(
        args.output_root.glob("reflection-corrective-*-prepared")
    )
    reflection_accounts = [_accounting_record(path) for path in reflection_dirs]
    attempts = reflection_import.get("attempts")
    if attempts is not None:
        if len(attempts) != len(reflection_accounts):
            raise ValueError("Reflection cost records do not match imported attempts")
        reflection_pairs = [
            _accounted_cost(account, attempt["usage"])
            for account, attempt in zip(reflection_accounts, attempts)
        ]
    elif all(account["standard_api"] for account in reflection_accounts):
        reflection_pairs = [_accounted_cost(account, {}) for account in reflection_accounts]
    else:
        # Older import manifests expose only aggregate usage. Their batch rates
        # must agree before one rate may be applied to the combined usage.
        if any(account["standard_api"] for account in reflection_accounts):
            raise ValueError("Legacy reflection imports cannot mix execution modes")
        rates = [account["record"]["pricing_usd_per_million_tokens"] for account in reflection_accounts]
        if any(rate != rates[0] for rate in rates[1:]):
            raise ValueError("Legacy reflection imports require matching recorded rates")
        reflection_pairs = [(_batch_usage_cost(reflection_import["aggregate_all_attempt_usage"], rates[0]),
                             sum(account["record"]["cost_estimate_usd"]["hard_maximum_total"]
                                 for account in reflection_accounts))]
    proposal_pairs = [_accounted_cost(account, usage)
                      for account, usage in zip(proposal_accounts, proposal_usages)]
    reflection_actual = sum(pair[0] for pair in reflection_pairs)
    reflection_maximum = sum(pair[1] for pair in reflection_pairs)
    proposal_actual = sum(pair[0] for pair in proposal_pairs)
    proposal_maximum = sum(pair[1] for pair in proposal_pairs)

    first_batch_dir = args.output_root / "proposal-batches" / f"{plan[0][0]}-prepared"
    unused_path = first_batch_dir / "canceled-batch-results-unused.jsonl"
    unused_usage = {}
    unused_actual = 0.0
    if unused_path.is_file():
        unused_usage = _usage_sum([
            ((line.get("result") or {}).get("message") or {}).get("usage") or {}
            for line in load_results(unused_path).values()
        ])
        # A canceled batch may finish after standard-API execution started.
        # Its own batch-rate record is required to count that unused spend.
        count = json.loads((first_batch_dir / "token_count_and_cost.json").read_text(encoding="utf-8"))
        if count.get("prepared_batch_manifest_sha256") != sha256_file(first_batch_dir / "batch_manifest.json"):
            raise ValueError("Unused batch pricing does not match its prepared input")
        unused_actual = _batch_usage_cost(unused_usage, count["pricing_usd_per_million_tokens"])

    archive = "".join(json.dumps({**candidate, "evaluation_status": "proposed_not_evaluated"},
                                  ensure_ascii=False) + "\n" for candidate in candidates)
    # All candidates, imports, and accounting are validated before writing output.
    archive_path.write_text(archive, encoding="utf-8")
    manifest = {
        "schema_version": "1.0.0", "phase": "development", "method": "SPJ-Evolve",
        "stage": "independent_reflection_and_sequential_mutation_proposal",
        "completed_at": utc_now(), "status": "COMPLETE_PROPOSED_NOT_EVALUATED",
        "generation": reflection_batch["generation"], "parent_ids": reflection_batch["parent_ids"],
        "parent_selection_mode": "explicit_parent_roster",
        "offspring_per_parent": args.offspring_per_parent,
        "candidate_prefix": reflection_batch["candidate_prefix"],
        "requested_candidate_count": len(plan), "valid_candidate_count": len(candidates),
        "reflection_count": reflection_import["completed_reflections"],
        "proposal_model": reflection_batch["generator"],
        "proposal_schedule": "sequential_dependency_preserving_provider_requests",
        "execution_modes": {
            "reflections": sorted({account["mode"] for account in reflection_accounts}),
            "proposals": sorted({account["mode"] for account in proposal_accounts}),
            "unused_cancellation_race_result": "anthropic_message_batches_api" if unused_path.is_file() else None,
        },
        "unused_cancellation_race": {
            "present": unused_path.is_file(), "candidate_id": plan[0][0], "used_for_candidate": False,
            "results_path": repo_relative(unused_path) if unused_path.is_file() else None,
            "usage": unused_usage,
        },
        "independence": reflection_batch["independence"],
        "parent_analysis_sha256": reflection_batch["parent_analysis_sha256"],
        "registry_sha256": reflection_batch["registry_sha256"],
        "feedback_bundle_path": reflection_batch.get("feedback_bundle_path"),
        "feedback_bundle_sha256": reflection_batch.get("feedback_bundle_sha256"),
        "candidate_archive_sha256": sha256_file(archive_path),
        "provider_reported_usage": _usage_sum([reflection_import["aggregate_all_attempt_usage"],
                                                *proposal_usages, unused_usage]),
        "cost": {
            "reflection_actual_usd": round(reflection_actual, 6),
            "proposal_actual_usd": round(proposal_actual, 6),
            "unused_cancellation_race_batch_actual_usd": round(unused_actual, 6),
            "combined_actual_usd": round(reflection_actual + proposal_actual + unused_actual, 6),
            "reflection_hard_maximum_usd": round(reflection_maximum, 6),
            "proposal_hard_maximum_usd": round(proposal_maximum, 6),
            "combined_hard_maximum_usd": round(reflection_maximum + proposal_maximum + unused_actual, 6),
            "accounting_records": [{key: account[key] for key in ("path", "sha256", "mode")}
                                   for account in reflection_accounts + proposal_accounts],
            "pricing_source": "authenticated_per_batch_records",
        },
        "git": git_state(),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    config = native_config("proposal")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_reflections = subparsers.add_parser("prepare-reflections")
    prepare_reflections.add_argument("--parent-analysis-dir", type=Path, required=True)
    prepare_reflections.add_argument("--parent-sweep-dir", type=Path, required=True)
    prepare_reflections.add_argument("--output-root", type=Path, required=True)
    prepare_reflections.add_argument("--parent-id", dest="parent_ids", action="append", required=True)
    prepare_reflections.add_argument("--generation", type=int, default=1)
    prepare_reflections.add_argument("--candidate-prefix", default="spj-g1")
    prepare_reflections.add_argument("--expected-parent-case-count", type=int, default=24)
    prepare_reflections.add_argument("--model", default=config.model, help="MND_PROPOSAL_MODEL or an explicit Anthropic model ID.")
    prepare_reflections.add_argument("--effort", default=config.effort, choices=("low", "medium", "high", "max"))
    prepare_reflections.add_argument("--max-output-tokens", type=int, default=config.max_tokens)
    prepare_reflections.add_argument("--study-id", default="spj-proposer-v1")
    prepare_reflections.add_argument("--feedback-bundle", type=Path)
    prepare_reflections.add_argument(
        "--study-registry", type=Path, default=REGISTRY_PATH
    )
    prepare_reflections.set_defaults(func=command_prepare_reflections)

    import_reflections = subparsers.add_parser("import-reflections")
    import_reflections.add_argument("--output-root", type=Path, required=True)
    import_reflections.add_argument("--results", type=Path, required=True)
    import_reflections.add_argument(
        "--corrective-results", type=Path, action="append", default=[]
    )
    import_reflections.set_defaults(func=command_import_reflections)

    corrective_reflections = subparsers.add_parser(
        "prepare-corrective-reflections"
    )
    corrective_reflections.add_argument("--output-root", type=Path, required=True)
    corrective_reflections.add_argument("--results", type=Path, required=True)
    corrective_reflections.add_argument("--max-output-tokens", type=int, default=None, help="Defaults to the frozen reflection request limit.")
    corrective_reflections.set_defaults(func=command_prepare_corrective_reflections)

    prepare_proposal = subparsers.add_parser("prepare-proposal")
    prepare_proposal.add_argument("--output-root", type=Path, required=True)
    prepare_proposal.add_argument("--parent-sweep-dir", type=Path, required=True)
    prepare_proposal.add_argument("--candidate-index", type=int, required=True)
    prepare_proposal.add_argument("--offspring-per-parent", type=int, default=3)
    prepare_proposal.add_argument("--expected-parent-case-count", type=int, default=24)
    prepare_proposal.add_argument("--lesson-register", type=Path,
                                  help="Optional frozen proposer-only lesson register, including freeze_register envelopes.")
    prepare_proposal.add_argument("--max-output-tokens", type=int, default=None, help="Defaults to the frozen reflection request limit.")
    prepare_proposal.set_defaults(func=command_prepare_proposal)

    import_proposal = subparsers.add_parser("import-proposal")
    import_proposal.add_argument("--output-root", type=Path, required=True)
    import_proposal.add_argument("--parent-sweep-dir", type=Path, required=True)
    import_proposal.add_argument("--candidate-id", required=True)
    import_proposal.add_argument("--results", type=Path, required=True)
    import_proposal.set_defaults(func=command_import_proposal)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--output-root", type=Path, required=True)
    finalize.add_argument("--offspring-per-parent", type=int, default=3)
    finalize.set_defaults(func=command_finalize)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
