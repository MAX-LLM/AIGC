#!/usr/bin/env python3
"""Prepare and import blinded frontier-model pairwise-judge batches."""

from __future__ import annotations

import json
import shutil
import statistics
from pathlib import Path
from typing import Any

from .common import canonical_sha256, load_json, sha256_file, utc_now, write_json
from .judge_pairwise import (
    DEFAULT_RUBRIC,
    DEFAULT_TAXONOMY,
    build_prompt,
    corrective_retry_prompt,
    load_presentations,
    validate_score,
)


SYSTEM_INSTRUCTION = (
    "Act as a strict blinded clinical evaluator. Judge only the supplied transcripts "
    "under the supplied rubric and return only the requested structured JSON."
)
PROVIDERS = {"openai", "anthropic"}


def _nonempty_string_schema() -> dict[str, Any]:
    return {"type": "string", "minLength": 1}


def _closed_object(
    properties: dict[str, Any], required: list[str] | None = None
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required if required is not None else list(properties),
        "additionalProperties": False,
    }


def strict_pairwise_json_schema(
    presentation_id: str, category_ids: list[str]
) -> dict[str, Any]:
    """Return one provider-neutral strict JSON schema for a presentation."""

    domain_schema = _closed_object(
        {
            "score": {"type": "integer", "enum": [0, 1, 2]},
            "rationale": _nonempty_string_schema(),
        }
    )
    side_schema = _closed_object(
        {
            domain: json.loads(json.dumps(domain_schema))
            for domain in (
                "Genetics",
                "Risk",
                "Scope",
                "Children",
                "Test",
                "Trials",
                "Structure",
                "Tone",
            )
        }
    )
    return _closed_object(
        {
            "schema_version": {"type": "string", "enum": ["1.0.0"]},
            "presentation_id": {"type": "string", "enum": [presentation_id]},
            "preference": _closed_object(
                {
                    "winner": {"type": "string", "enum": ["A", "B", "tie"]},
                    "strength": {"type": "integer", "enum": [0, 1, 2, 3]},
                    "rationale": _nonempty_string_schema(),
                }
            ),
            "direct_question_resolution": _closed_object(
                {
                    "winner": {"type": "string", "enum": ["A", "B", "tie"]},
                    "rationale": _nonempty_string_schema(),
                }
            ),
            "scores": _closed_object(
                {
                    "A": side_schema,
                    "B": json.loads(json.dumps(side_schema)),
                }
            ),
            "critical_error_candidates": {
                "type": "array",
                "items": _closed_object(
                    {
                        "transcript": {"type": "string", "enum": ["A", "B"]},
                        "category_id": {"type": "string", "enum": category_ids},
                        "turn": {"type": "integer"},
                        "evidence": _nonempty_string_schema(),
                        "rationale": _nonempty_string_schema(),
                    }
                ),
            },
            "uncertainties": {
                "type": "array",
                "items": _nonempty_string_schema(),
            },
        }
    )


def build_provider_request(
    *,
    provider: str,
    custom_id: str,
    prompt: str,
    schema: dict[str, Any],
    model: str,
    effort: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    if effort not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
        raise ValueError(f"Unsupported reasoning effort: {effort}")
    if provider == "anthropic" and effort not in {"low", "medium", "high", "max"}:
        raise ValueError("Anthropic reasoning effort must be low, medium, high or max")
    if max_output_tokens < 1:
        raise ValueError("max_output_tokens must be positive")
    if provider == "openai":
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": model,
                "instructions": SYSTEM_INSTRUCTION,
                "input": prompt,
                "reasoning": {"effort": effort},
                "max_output_tokens": max_output_tokens,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "spj_pairwise_score",
                        "strict": True,
                        "schema": schema,
                    }
                },
                "store": False,
            },
        }
    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": max_output_tokens,
            "system": SYSTEM_INSTRUCTION,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {
                "effort": effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        },
    }


def prepare_batch(
    *,
    provider: str,
    presentations_path: Path,
    output_dir: Path,
    judge_id: str,
    model: str,
    effort: str,
    max_output_tokens: int,
    rubric_path: Path = DEFAULT_RUBRIC,
    taxonomy_path: Path = DEFAULT_TAXONOMY,
    git_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create immutable provider requests without contacting either API."""

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite prepared batch: {output_dir}")
    output_dir.mkdir(parents=True)
    presentations = load_presentations(presentations_path)
    rubric_text = rubric_path.read_text(encoding="utf-8").strip()
    taxonomy = load_json(taxonomy_path)
    category_ids = sorted(item["id"] for item in taxonomy["categories"])
    records = []
    requests = []
    for presentation in presentations:
        presentation_id = presentation["presentation_id"]
        prompt = build_prompt(presentation, rubric_text, taxonomy)
        schema = strict_pairwise_json_schema(presentation_id, category_ids)
        request = build_provider_request(
            provider=provider,
            custom_id=presentation_id,
            prompt=prompt,
            schema=schema,
            model=model,
            effort=effort,
            max_output_tokens=max_output_tokens,
        )
        requests.append(request)
        records.append(
            {
                "custom_id": presentation_id,
                "presentation_id": presentation_id,
                "prompt_sha256": canonical_sha256(prompt),
                "output_json_schema_sha256": canonical_sha256(schema),
                "request_sha256": canonical_sha256(request),
            }
        )
    requests_path = output_dir / "requests.jsonl"
    requests_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in requests),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "1.0.0",
        "created_at": utc_now(),
        "status": "PREPARED_NOT_SUBMITTED",
        "provider": provider,
        "judge": {
            "id": judge_id,
            "role": "development_frontier_replication",
            "model": model,
            "effort": effort,
            "structured_outputs": True,
        },
        "request_count": len(requests),
        "expected_evaluations": len(requests),
        "max_output_tokens_per_request": max_output_tokens,
        "blinded": True,
        "candidate_roster_withheld_from_provider": True,
        "allocation_key_withheld_from_provider": True,
        "batch_input_path": requests_path.name,
        "batch_input_sha256": sha256_file(requests_path),
        "presentations_sha256": sha256_file(presentations_path),
        "rubric_sha256": sha256_file(rubric_path),
        "critical_error_taxonomy_sha256": sha256_file(taxonomy_path),
        "records": records,
        "git": git_state or {},
    }
    write_json(output_dir / "batch_manifest.json", manifest)
    return manifest


def prepare_corrective_batch(
    *,
    original_batch_dir: Path,
    failed_import_dir: Path,
    output_dir: Path,
    git_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare one format-only batch containing exactly the failed requests."""

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite prepared batch: {output_dir}")
    original_manifest_path = original_batch_dir / "batch_manifest.json"
    import_manifest_path = failed_import_dir / "judge_manifest.json"
    original = load_json(original_manifest_path)
    failed_import = load_json(import_manifest_path)
    if failed_import.get("status") != "INCOMPLETE_REQUIRES_CORRECTIVE_BATCH":
        raise ValueError("Source import does not require a corrective batch")
    if not failed_import.get("failures"):
        raise ValueError("Source import contains no failed evaluations")
    if failed_import.get("prepared_batch_manifest_sha256") != sha256_file(
        original_manifest_path
    ):
        raise ValueError("Failed import does not authenticate the original batch")
    if failed_import.get("judge") != original.get("judge"):
        raise ValueError("Failed import judge differs from the original batch")

    requests_path = original_batch_dir / original["batch_input_path"]
    if sha256_file(requests_path) != original["batch_input_sha256"]:
        raise ValueError("Original batch input hash mismatch")
    original_requests = {
        item["custom_id"]: item
        for item in (
            json.loads(line)
            for line in requests_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    original_records = {item["custom_id"]: item for item in original["records"]}
    failed_ids = [item["presentation_id"] for item in failed_import["failures"]]
    if len(failed_ids) != len(set(failed_ids)):
        raise ValueError("Failed import contains duplicate presentation IDs")
    if not set(failed_ids) <= set(original_requests):
        raise ValueError("Failed import contains an unknown presentation ID")

    provider = original["provider"]
    judge = original["judge"]
    requests = []
    records = []
    failures_by_id = {
        item["presentation_id"]: item for item in failed_import["failures"]
    }
    for presentation_id in failed_ids:
        source = original_requests[presentation_id]
        if provider == "openai":
            source_body = source["body"]
            base_prompt = source_body["input"]
            schema = source_body["text"]["format"]["schema"]
        else:
            source_params = source["params"]
            base_prompt = source_params["messages"][0]["content"]
            schema = source_params["output_config"]["format"]["schema"]
        prior_error = failures_by_id[presentation_id]["error"]
        corrected_prompt = corrective_retry_prompt(
            base_prompt,
            presentation_id=presentation_id,
            prior_error=prior_error,
        )
        request = build_provider_request(
            provider=provider,
            custom_id=presentation_id,
            prompt=corrected_prompt,
            schema=schema,
            model=judge["model"],
            effort=judge["effort"],
            max_output_tokens=original["max_output_tokens_per_request"],
        )
        requests.append(request)
        records.append(
            {
                "custom_id": presentation_id,
                "presentation_id": presentation_id,
                "base_prompt_sha256": original_records[presentation_id]["prompt_sha256"],
                "prompt_sha256": canonical_sha256(corrected_prompt),
                "prior_error": prior_error,
                "output_json_schema_sha256": canonical_sha256(schema),
                "request_sha256": canonical_sha256(request),
            }
        )

    output_dir.mkdir(parents=True)
    corrective_requests_path = output_dir / "requests.jsonl"
    corrective_requests_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in requests),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "1.0.0",
        "created_at": utc_now(),
        "status": "FORMAT_ONLY_CORRECTIVE_BATCH_PREPARED_NOT_SUBMITTED",
        "provider": provider,
        "judge": judge,
        "request_count": len(requests),
        "expected_evaluations": len(requests),
        "max_output_tokens_per_request": original["max_output_tokens_per_request"],
        "blinded": True,
        "candidate_roster_withheld_from_provider": True,
        "allocation_key_withheld_from_provider": True,
        "criteria_changed_from_original": False,
        "corrective_retry_number": 1,
        "batch_input_path": corrective_requests_path.name,
        "batch_input_sha256": sha256_file(corrective_requests_path),
        "presentations_sha256": original["presentations_sha256"],
        "rubric_sha256": original["rubric_sha256"],
        "critical_error_taxonomy_sha256": original[
            "critical_error_taxonomy_sha256"
        ],
        "original_batch_manifest_sha256": sha256_file(original_manifest_path),
        "failed_import_manifest_sha256": sha256_file(import_manifest_path),
        "records": records,
        "git": git_state or {},
    }
    write_json(output_dir / "batch_manifest.json", manifest)
    return manifest


def summarize_token_counts(values: list[int]) -> dict[str, Any]:
    if not values:
        raise ValueError("No token counts supplied")
    return {
        "request_count": len(values),
        "total_input_tokens": sum(values),
        "minimum_input_tokens": min(values),
        "median_input_tokens": statistics.median(values),
        "maximum_input_tokens": max(values),
    }


def _openai_result(line: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if line.get("error"):
        raise ValueError(f"OpenAI batch request error: {line['error']}")
    response = line.get("response") or {}
    if response.get("status_code") != 200:
        raise ValueError(f"OpenAI batch HTTP status: {response.get('status_code')}")
    body = response.get("body") or {}
    texts = []
    for item in body.get("output", []):
        if item.get("type") != "message":
            continue
        for block in item.get("content", []):
            if block.get("type") == "output_text" and isinstance(block.get("text"), str):
                texts.append(block["text"])
    if len(texts) != 1:
        raise ValueError(f"Expected one OpenAI output_text block, found {len(texts)}")
    return texts[0], {
        "response_id": body.get("id"),
        "response_model": body.get("model"),
        "status": body.get("status"),
        "usage": body.get("usage", {}),
        "execution_mode": line.get(
            "execution_mode", "openai_responses_batch_api"
        ),
    }


def _anthropic_result(line: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    result = line.get("result") or {}
    if result.get("type") != "succeeded":
        raise ValueError(f"Anthropic batch result type: {result.get('type')}")
    message = result.get("message") or {}
    texts = [
        block["text"]
        for block in message.get("content", [])
        if block.get("type") == "text" and isinstance(block.get("text"), str)
    ]
    if len(texts) != 1:
        raise ValueError(f"Expected one Anthropic text block, found {len(texts)}")
    return texts[0], {
        "response_id": message.get("id"),
        "response_model": message.get("model"),
        "stop_reason": message.get("stop_reason"),
        "usage": message.get("usage", {}),
    }


def import_batch_results(
    *,
    provider: str,
    results_path: Path,
    batch_dir: Path,
    output_dir: Path,
    git_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate downloaded provider results into the standard single-judge layout."""

    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite imported judge run: {output_dir}")
    batch_manifest = load_json(batch_dir / "batch_manifest.json")
    if batch_manifest["provider"] != provider:
        raise ValueError("Provider does not match prepared batch manifest")
    if sha256_file(batch_dir / batch_manifest["batch_input_path"]) != batch_manifest[
        "batch_input_sha256"
    ]:
        raise ValueError("Prepared batch input hash mismatch")
    expected = {item["custom_id"]: item for item in batch_manifest["records"]}
    supplied: dict[str, dict[str, Any]] = {}
    with results_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            custom_id = value.get("custom_id")
            if custom_id not in expected:
                raise ValueError(f"Unknown custom_id at result line {line_number}")
            if custom_id in supplied:
                raise ValueError(f"Duplicate custom_id at result line {line_number}")
            supplied[custom_id] = value

    output_dir.mkdir(parents=True)
    taxonomy = load_json(DEFAULT_TAXONOMY)
    category_ids = {item["id"] for item in taxonomy["categories"]}
    judge = batch_manifest["judge"]
    records = []
    failures = []
    aggregate_usage: dict[str, int] = {}
    for custom_id, source_record in expected.items():
        line = supplied.get(custom_id)
        if line is None:
            failures.append(
                {
                    "judge_id": judge["id"],
                    "presentation_id": custom_id,
                    "error_type": "MissingBatchResult",
                    "error": "No downloaded result matched the prepared request.",
                }
            )
            continue
        write_json(output_dir / f"{custom_id}.api.json", line)
        try:
            raw_text, metadata = (
                _openai_result(line) if provider == "openai" else _anthropic_result(line)
            )
            (output_dir / f"{custom_id}.attempt-1.raw.txt").write_text(
                raw_text, encoding="utf-8"
            )
            score = validate_score(
                json.loads(raw_text),
                presentation_id=custom_id,
                category_ids=category_ids,
            )
            usage = metadata.get("usage", {})
            for key, value in usage.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    aggregate_usage[key] = aggregate_usage.get(key, 0) + value
            result = {
                "judge": {
                    "id": judge["id"],
                    "role": judge["role"],
                    "provider": (
                        metadata.get("execution_mode", "openai_responses_batch_api")
                        if provider == "openai"
                        else "anthropic_message_batches_api"
                    ),
                    "model": metadata.get("response_model") or judge["model"],
                    "requested_model": judge["model"],
                    "effort": judge["effort"],
                },
                "presentation_id": custom_id,
                "prompt_sha256": source_record["prompt_sha256"],
                "attempt": 1,
                "response_metadata": metadata,
                "score": score,
            }
            score_path = output_dir / f"{custom_id}.score.json"
            write_json(score_path, result)
            records.append(
                {
                    "judge_id": judge["id"],
                    "presentation_id": custom_id,
                    "prompt_sha256": source_record["prompt_sha256"],
                    "attempt": 1,
                    "score_path": score_path.name,
                }
            )
        except Exception as exc:
            failure = {
                "judge_id": judge["id"],
                "presentation_id": custom_id,
                "prompt_sha256": source_record["prompt_sha256"],
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            write_json(output_dir / f"{custom_id}.failure.json", failure)
    manifest = {
        "schema_version": "1.0.0",
        "created_at": utc_now(),
        "phase": "development",
        "status": "COMPLETE" if not failures else "INCOMPLETE_REQUIRES_CORRECTIVE_BATCH",
        "dry_run": False,
        "blinded": True,
        "judge_count": 1,
        "judge_ids": [judge["id"]],
        "expected_evaluations": len(expected),
        "completed_evaluations": len(records),
        "failed_evaluations": len(failures),
        "records": records,
        "failures": failures,
        "judge": judge,
        "aggregate_usage": aggregate_usage,
        "provider_results_sha256": sha256_file(results_path),
        "prepared_batch_manifest_sha256": sha256_file(batch_dir / "batch_manifest.json"),
        "batch_input_sha256": batch_manifest["batch_input_sha256"],
        "presentations_sha256": batch_manifest["presentations_sha256"],
        "rubric_sha256": batch_manifest["rubric_sha256"],
        "critical_error_taxonomy_sha256": batch_manifest[
            "critical_error_taxonomy_sha256"
        ],
        "git": git_state or {},
    }
    write_json(output_dir / "judge_manifest.json", manifest)
    return manifest


def merge_corrective_imports(
    *,
    original_import_dir: Path,
    corrective_batch_dir: Path,
    corrective_import_dir: Path,
    output_dir: Path,
    git_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge successful first attempts with exactly one format-only retry."""

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite merged judge run: {output_dir}")
    original_manifest_path = original_import_dir / "judge_manifest.json"
    corrective_batch_manifest_path = corrective_batch_dir / "batch_manifest.json"
    corrective_import_manifest_path = corrective_import_dir / "judge_manifest.json"
    original = load_json(original_manifest_path)
    corrective_batch = load_json(corrective_batch_manifest_path)
    corrective = load_json(corrective_import_manifest_path)
    if original.get("status") != "INCOMPLETE_REQUIRES_CORRECTIVE_BATCH":
        raise ValueError("Original import does not require correction")
    if corrective.get("status") != "COMPLETE" or corrective.get(
        "failed_evaluations"
    ):
        raise ValueError("Corrective import is not complete")
    if corrective_batch.get("corrective_retry_number") != 1 or corrective_batch.get(
        "criteria_changed_from_original"
    ) is not False:
        raise ValueError("Corrective batch is not an authenticated format-only retry")
    if original.get("prepared_batch_manifest_sha256") != corrective_batch.get(
        "original_batch_manifest_sha256"
    ):
        raise ValueError("Corrective batch does not authenticate the original batch")
    if corrective.get("prepared_batch_manifest_sha256") != sha256_file(
        corrective_batch_manifest_path
    ):
        raise ValueError("Corrective import does not authenticate its prepared batch")
    if corrective_batch.get("failed_import_manifest_sha256") != sha256_file(
        original_manifest_path
    ):
        raise ValueError("Corrective batch does not authenticate the failed import")
    if original.get("judge") != corrective.get("judge"):
        raise ValueError("Original and corrective imports use different judges")

    failed_ids = {item["presentation_id"] for item in original.get("failures", [])}
    corrective_ids = {
        item["presentation_id"] for item in corrective.get("records", [])
    }
    if not failed_ids or corrective_ids != failed_ids:
        raise ValueError("Corrective results do not exactly replace the failed IDs")
    original_ids = {item["presentation_id"] for item in original.get("records", [])}
    if original_ids & corrective_ids:
        raise ValueError("Original and corrective successful IDs overlap")
    expected_count = original.get("expected_evaluations")
    if len(original_ids | corrective_ids) != expected_count:
        raise ValueError("Merged IDs do not cover the complete original batch")

    corrective_records = {
        item["presentation_id"]: item for item in corrective_batch["records"]
    }
    output_dir.mkdir(parents=True)
    records = []
    for source_record in original["records"]:
        presentation_id = source_record["presentation_id"]
        source_path = original_import_dir / source_record["score_path"]
        destination = output_dir / source_path.name
        shutil.copy2(source_path, destination)
        records.append(
            {
                **source_record,
                "score_sha256": sha256_file(destination),
                "rating_source": "initial_batch",
            }
        )
    for source_record in corrective["records"]:
        presentation_id = source_record["presentation_id"]
        source_path = corrective_import_dir / source_record["score_path"]
        result = load_json(source_path)
        result["attempt"] = 2
        result["corrective_retry"] = {
            "format_only": True,
            "criteria_changed": False,
            "base_prompt_sha256": corrective_records[presentation_id][
                "base_prompt_sha256"
            ],
            "corrective_prompt_sha256": corrective_records[presentation_id][
                "prompt_sha256"
            ],
            "prior_error": corrective_records[presentation_id]["prior_error"],
        }
        destination = output_dir / source_path.name
        write_json(destination, result)
        records.append(
            {
                **source_record,
                "attempt": 2,
                "base_prompt_sha256": corrective_records[presentation_id][
                    "base_prompt_sha256"
                ],
                "score_sha256": sha256_file(destination),
                "rating_source": "format_only_corrective_batch",
            }
        )
    records.sort(key=lambda item: item["presentation_id"])

    aggregate_usage: dict[str, int] = {}
    for source in (original, corrective):
        for key, value in source.get("aggregate_usage", {}).items():
            if isinstance(value, int) and not isinstance(value, bool):
                aggregate_usage[key] = aggregate_usage.get(key, 0) + value
    manifest = {
        "schema_version": "1.0.0",
        "created_at": utc_now(),
        "phase": "development",
        "status": "COMPLETE_AFTER_FORMAT_ONLY_CORRECTIVE_BATCH",
        "dry_run": False,
        "blinded": True,
        "judge_count": 1,
        "judge_ids": original["judge_ids"],
        "expected_evaluations": expected_count,
        "completed_evaluations": len(records),
        "failed_evaluations": 0,
        "records": records,
        "failures": [],
        "judge": original["judge"],
        "aggregate_usage": aggregate_usage,
        "presentations_sha256": original["presentations_sha256"],
        "rubric_sha256": original["rubric_sha256"],
        "critical_error_taxonomy_sha256": original[
            "critical_error_taxonomy_sha256"
        ],
        "git": git_state or {},
        "corrective_batch": {
            "retry_number": 1,
            "format_only": True,
            "criteria_changed": False,
            "replacement_evaluations": len(corrective_ids),
            "original_import_manifest_sha256": sha256_file(original_manifest_path),
            "corrective_batch_manifest_sha256": sha256_file(
                corrective_batch_manifest_path
            ),
            "corrective_import_manifest_sha256": sha256_file(
                corrective_import_manifest_path
            ),
        },
    }
    write_json(output_dir / "judge_manifest.json", manifest)
    return manifest
