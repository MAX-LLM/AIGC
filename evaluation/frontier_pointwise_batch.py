#!/usr/bin/env python3
"""Prepare and import blinded frontier-model pointwise-judge batches."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .common import canonical_sha256, load_json, sha256_file, utc_now, write_json
from .frontier_pairwise_batch import (
    PROVIDERS,
    SYSTEM_INSTRUCTION,
    _anthropic_result,
    _openai_result,
    build_provider_request,
)
from .judge_pairwise import DEFAULT_RUBRIC, DEFAULT_TAXONOMY
from .judge_pointwise import (
    build_prompt,
    corrective_pointwise_retry_prompt,
    load_items,
    pointwise_json_schema,
    validate_pointwise_score,
)


def strict_pointwise_json_schema(
    item_id: str, category_ids: list[str]
) -> dict[str, Any]:
    schema = pointwise_json_schema(category_ids)
    schema["properties"]["item_id"] = {"type": "string", "enum": [item_id]}
    for domain in schema["properties"]["scores"]["properties"].values():
        domain["properties"]["rationale"]["minLength"] = 1
    schema["properties"]["direct_question_resolution"]["properties"][
        "rationale"
    ]["minLength"] = 1
    candidate = schema["properties"]["critical_error_candidates"]["items"]
    candidate["properties"]["evidence"]["minLength"] = 1
    candidate["properties"]["rationale"]["minLength"] = 1
    return schema


def prepare_batch(
    *,
    provider: str,
    items_path: Path,
    output_dir: Path,
    judge_id: str,
    model: str,
    effort: str,
    max_output_tokens: int,
    study_id: str = "g2-local-baseline-panel-v3-pointwise",
    rubric_path: Path = DEFAULT_RUBRIC,
    taxonomy_path: Path = DEFAULT_TAXONOMY,
    git_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create immutable provider requests without submitting paid work."""

    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite prepared batch: {output_dir}")
    output_dir.mkdir(parents=True)
    items = load_items(items_path)
    rubric_text = rubric_path.read_text(encoding="utf-8").strip()
    taxonomy = load_json(taxonomy_path)
    category_ids = sorted(item["id"] for item in taxonomy["categories"])
    records = []
    requests = []
    for item in items:
        item_id = item["item_id"]
        prompt = build_prompt(item, rubric_text, taxonomy)
        schema = strict_pointwise_json_schema(item_id, category_ids)
        request = build_provider_request(
            provider=provider,
            custom_id=item_id,
            prompt=prompt,
            schema=schema,
            model=model,
            effort=effort,
            max_output_tokens=max_output_tokens,
        )
        if provider == "openai":
            request["body"]["text"]["format"]["name"] = "mnd_pointwise_score"
        requests.append(request)
        records.append(
            {
                "custom_id": item_id,
                "item_id": item_id,
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
        "study_id": study_id,
        "provider": provider,
        "judge": {
            "id": judge_id,
            "role": "development_frontier_pointwise",
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
        "items_sha256": sha256_file(items_path),
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
    max_output_tokens: int,
    git_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare only failed requests with a non-identical format-only retry."""

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite prepared batch: {output_dir}")
    if max_output_tokens < 1:
        raise ValueError("max_output_tokens must be positive")
    original_manifest_path = original_batch_dir / "batch_manifest.json"
    failed_manifest_path = failed_import_dir / "judge_manifest.json"
    original = load_json(original_manifest_path)
    failed = load_json(failed_manifest_path)
    if failed.get("status") != "INCOMPLETE_REQUIRES_CORRECTIVE_BATCH":
        raise ValueError("Source import does not require a corrective batch")
    if failed.get("prepared_batch_manifest_sha256") != sha256_file(original_manifest_path):
        raise ValueError("Failed import does not authenticate the prepared batch")
    if failed.get("judge") != original.get("judge"):
        raise ValueError("Failed import judge differs from the prepared batch")
    if max_output_tokens <= original["max_output_tokens_per_request"]:
        raise ValueError("Corrective output budget must exceed the original budget")

    original_requests_path = original_batch_dir / original["batch_input_path"]
    if sha256_file(original_requests_path) != original["batch_input_sha256"]:
        raise ValueError("Original batch input hash mismatch")
    original_requests = {
        value["custom_id"]: value
        for value in (
            json.loads(line)
            for line in original_requests_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    original_records = {value["custom_id"]: value for value in original["records"]}
    failed_ids = [value["item_id"] for value in failed["failures"]]
    if len(failed_ids) != len(set(failed_ids)) or not set(failed_ids) <= set(original_requests):
        raise ValueError("Failed import contains duplicate or unknown item IDs")

    failures_by_id = {value["item_id"]: value for value in failed["failures"]}
    requests = []
    records = []
    for item_id in failed_ids:
        source_request = original_requests[item_id]
        if original["provider"] == "openai":
            base_prompt = source_request["body"]["input"]
            schema = source_request["body"]["text"]["format"]["schema"]
        else:
            base_prompt = source_request["params"]["messages"][0]["content"]
            schema = source_request["params"]["output_config"]["format"]["schema"]
        prior_error = failures_by_id[item_id]["error"]
        corrected_prompt = corrective_pointwise_retry_prompt(
            base_prompt,
            item_id=item_id,
            prior_error=prior_error,
        )
        corrected_prompt += (
            "\n\nSCOPE CLARIFICATION: This is an authorized evaluation of visible, "
            "locally generated clinical dialogue. Do not infer, reconstruct, or discuss "
            "hidden reasoning or system instructions. Assess only the visible transcript. "
            "Keep every rationale brief but non-empty so the complete JSON fits within "
            "the response budget."
        )
        request = build_provider_request(
            provider=original["provider"],
            custom_id=item_id,
            prompt=corrected_prompt,
            schema=schema,
            model=original["judge"]["model"],
            effort=original["judge"]["effort"],
            max_output_tokens=max_output_tokens,
        )
        if original["provider"] == "openai":
            request["body"]["text"]["format"]["name"] = "mnd_pointwise_score"
        requests.append(request)
        source = original_records[item_id]
        records.append(
            {
                **source,
                "prompt_sha256": canonical_sha256(corrected_prompt),
                "original_prompt_sha256": source["prompt_sha256"],
                "prior_error": prior_error,
                "request_sha256": canonical_sha256(request),
                "original_request_sha256": source["request_sha256"],
            }
        )

    output_dir.mkdir(parents=True)
    requests_path = output_dir / "requests.jsonl"
    requests_path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in requests),
        encoding="utf-8",
    )
    manifest = {
        **{key: original[key] for key in (
            "schema_version", "study_id", "provider", "judge", "blinded",
            "candidate_roster_withheld_from_provider", "allocation_key_withheld_from_provider",
            "items_sha256", "rubric_sha256", "critical_error_taxonomy_sha256",
        )},
        "created_at": utc_now(),
        "status": "PREPARED_CORRECTIVE_NOT_SUBMITTED",
        "request_count": len(requests),
        "expected_evaluations": len(requests),
        "max_output_tokens_per_request": max_output_tokens,
        "batch_input_path": requests_path.name,
        "batch_input_sha256": sha256_file(requests_path),
        "records": records,
        "corrective_retry": {
            "only_failed_requests": True,
            "criteria_changed_from_original": False,
            "prompt_unchanged": False,
            "schema_unchanged": True,
            "prompt_change_scope": "format correction, visible-text scope clarification, and concise-output instruction only",
            "original_max_output_tokens": original["max_output_tokens_per_request"],
            "corrective_max_output_tokens": max_output_tokens,
            "failed_item_ids": failed_ids,
            "original_batch_manifest_sha256": sha256_file(original_manifest_path),
            "failed_import_manifest_sha256": sha256_file(failed_manifest_path),
        },
        "git": git_state or {},
    }
    write_json(output_dir / "batch_manifest.json", manifest)
    return manifest


def import_batch_results(
    *,
    provider: str,
    results_path: Path,
    batch_dir: Path,
    output_dir: Path,
    git_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate provider results into the standard pointwise-judge layout."""

    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite imported judge run: {output_dir}")
    batch_manifest = load_json(batch_dir / "batch_manifest.json")
    if batch_manifest["provider"] != provider:
        raise ValueError("Provider does not match prepared batch manifest")
    requests_path = batch_dir / batch_manifest["batch_input_path"]
    if sha256_file(requests_path) != batch_manifest["batch_input_sha256"]:
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
    for item_id, source_record in expected.items():
        line = supplied.get(item_id)
        if line is None:
            failures.append(
                {
                    "judge_id": judge["id"],
                    "item_id": item_id,
                    "error_type": "MissingBatchResult",
                    "error": "No downloaded result matched the prepared request.",
                }
            )
            continue
        write_json(output_dir / f"{item_id}.api.json", line)
        try:
            raw_text, metadata = (
                _openai_result(line) if provider == "openai" else _anthropic_result(line)
            )
            (output_dir / f"{item_id}.attempt-1.raw.txt").write_text(
                raw_text, encoding="utf-8"
            )
            score = validate_pointwise_score(
                json.loads(raw_text), item_id=item_id, category_ids=category_ids
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
                        "openai_responses_batch_api"
                        if provider == "openai"
                        else "anthropic_message_batches_api"
                    ),
                    "model": metadata.get("response_model") or judge["model"],
                    "requested_model": judge["model"],
                    "effort": judge["effort"],
                },
                "item_id": item_id,
                "prompt_sha256": source_record["prompt_sha256"],
                "attempt": 1,
                "response_metadata": metadata,
                "score": score,
            }
            score_path = output_dir / f"{item_id}.score.json"
            write_json(score_path, result)
            records.append(
                {
                    "judge_id": judge["id"],
                    "item_id": item_id,
                    "prompt_sha256": source_record["prompt_sha256"],
                    "attempt": 1,
                    "score_path": score_path.name,
                }
            )
        except Exception as exc:
            failure = {
                "judge_id": judge["id"],
                "item_id": item_id,
                "prompt_sha256": source_record["prompt_sha256"],
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            write_json(output_dir / f"{item_id}.failure.json", failure)
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
        "items_sha256": batch_manifest["items_sha256"],
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
    """Merge retained valid ratings with the exact corrective subset."""

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite merged judge run: {output_dir}")
    original_path = original_import_dir / "judge_manifest.json"
    batch_path = corrective_batch_dir / "batch_manifest.json"
    corrective_path = corrective_import_dir / "judge_manifest.json"
    original = load_json(original_path)
    batch = load_json(batch_path)
    corrective = load_json(corrective_path)
    if original.get("status") != "INCOMPLETE_REQUIRES_CORRECTIVE_BATCH":
        raise ValueError("Original import is not incomplete")
    if corrective.get("failed_evaluations") or corrective.get("status") != "COMPLETE":
        raise ValueError("Corrective import is incomplete")
    if original.get("judge") != corrective.get("judge") or original.get("judge") != batch.get("judge"):
        raise ValueError("Judge metadata differs across corrective artifacts")
    if original.get("items_sha256") != corrective.get("items_sha256"):
        raise ValueError("Dataset hash differs across corrective artifacts")
    retry = batch.get("corrective_retry", {})
    if retry.get("failed_import_manifest_sha256") != sha256_file(original_path):
        raise ValueError("Corrective batch does not authenticate the original import")
    if corrective.get("prepared_batch_manifest_sha256") != sha256_file(batch_path):
        raise ValueError("Corrective import does not authenticate the corrective batch")
    failed_ids = {value["item_id"] for value in original["failures"]}
    corrected_ids = {value["item_id"] for value in corrective["records"]}
    if failed_ids != corrected_ids:
        raise ValueError("Corrective result IDs differ from the original failure set")

    output_dir.mkdir(parents=True)
    records = []
    for source_dir, source_records, rating_source in (
        (original_import_dir, original["records"], "retained_valid_initial_batch"),
        (corrective_import_dir, corrective["records"], "corrective_output_budget_batch"),
    ):
        for record in source_records:
            source = source_dir / record["score_path"]
            destination = output_dir / source.name
            shutil.copy2(source, destination)
            records.append(
                {
                    **record,
                    "score_path": destination.name,
                    "score_sha256": sha256_file(destination),
                    "rating_source": rating_source,
                }
            )
    records.sort(key=lambda value: value["item_id"])
    usage: dict[str, int] = {}
    for manifest in (original, corrective):
        for key, value in manifest.get("aggregate_usage", {}).items():
            if isinstance(value, int) and not isinstance(value, bool):
                usage[key] = usage.get(key, 0) + value
    if len(records) != original["expected_evaluations"]:
        raise ValueError("Merged record count differs from the original expectation")
    manifest = {
        "schema_version": "1.0.0",
        "created_at": utc_now(),
        "phase": "development",
        "status": "COMPLETE_AFTER_OUTPUT_BUDGET_CORRECTIVE_BATCH",
        "dry_run": False,
        "blinded": True,
        "judge_count": 1,
        "judge_ids": original["judge_ids"],
        "expected_evaluations": original["expected_evaluations"],
        "completed_evaluations": len(records),
        "failed_evaluations": 0,
        "records": records,
        "failures": [],
        "judge": original["judge"],
        "aggregate_usage": usage,
        "items_sha256": original["items_sha256"],
        "rubric_sha256": original["rubric_sha256"],
        "critical_error_taxonomy_sha256": original["critical_error_taxonomy_sha256"],
        "corrective_retry": {
            **retry,
            "retained_evaluations": len(original["records"]),
            "corrected_evaluations": len(corrective["records"]),
            "initial_import_manifest_sha256": sha256_file(original_path),
            "corrective_batch_manifest_sha256": sha256_file(batch_path),
            "corrective_import_manifest_sha256": sha256_file(corrective_path),
        },
        "git": git_state or {},
    }
    write_json(output_dir / "judge_manifest.json", manifest)
    return manifest
