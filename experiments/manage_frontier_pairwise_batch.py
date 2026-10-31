#!/usr/bin/env python3
"""Prepare, price, submit, retrieve, and import frontier pairwise-judge batches."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.common import load_json, sha256_file, utc_now, write_json
from evaluation.provider_config import make_client, model_role, native_config
from evaluation.frontier_pairwise_batch import (
    import_batch_results,
    merge_corrective_imports,
    prepare_batch,
    prepare_corrective_batch,
    summarize_token_counts,
)


PAID_APPROVAL_PHRASE = "I_APPROVE_PAID_BATCH"


def _transport_client(provider, args=None, manifest=None):
    role = model_role(manifest, getattr(args, "role", None))
    return make_client(
        provider, role,
        api_base=getattr(args, "api_base", None) or (manifest or {}).get("api_base"),
        api_key=getattr(args, "api_key", None),
        timeout=getattr(args, "timeout", None),
        workspace_id=getattr(args, "anthropic_workspace_id", None),
    )


def _anthropic_client(workspace_id=None, *, role="judge", api_base=None, api_key=None):
    """Compatibility adapter; role keys and endpoints remain independent."""
    return make_client("anthropic", role, workspace_id=workspace_id, api_base=api_base, api_key=api_key)


def prepare_settings(args):
    config = native_config(
        getattr(args, "role", None) or "judge", provider=args.provider,
        model=args.model, effort=args.effort, max_tokens=args.max_output_tokens,
        api_base=getattr(args, "api_base", None),
        api_key=getattr(args, "api_key", None), timeout=getattr(args, "timeout", None),
    ).validate()
    args.provider, args.model, args.effort = config.provider, config.model, config.effort
    args.max_output_tokens = config.max_tokens
    return config


def _stamp_spj_policy(manifest, output_dir, *, force=False):
    # Historical model rosters are provenance, not restrictions on new runs.
    return manifest


def _enforce_spj_submission_policy(manifest):
    actor = manifest.get("judge") or manifest.get("generator") or {}
    native_config(model_role(manifest), provider=manifest.get("provider"),
                  model=actor.get("model", ""), effort=actor.get("effort", "high")).validate()


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
            "tracked_dirty": bool(run(["git", "status", "--porcelain", "--untracked-files=no"])),
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"branch": None, "commit": None, "tracked_dirty": None, "source_checkout": False}


def load_prepared_batch(batch_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = batch_dir / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    requests_path = batch_dir / manifest["batch_input_path"]
    if sha256_file(requests_path) != manifest["batch_input_sha256"]:
        raise ValueError("Prepared batch input hash mismatch")
    requests = [
        json.loads(line)
        for line in requests_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(requests) != manifest["request_count"]:
        raise ValueError("Prepared request count does not match batch manifest")
    return manifest, requests


def command_prepare(args: argparse.Namespace) -> int:
    config = prepare_settings(args)
    manifest = prepare_batch(
        provider=args.provider,
        presentations_path=args.presentations,
        output_dir=args.output_dir,
        judge_id=args.judge_id,
        model=args.model,
        effort=args.effort,
        max_output_tokens=args.max_output_tokens,
        git_state=git_state(),
    )
    manifest.update({"model_role": config.role, "api_base": config.api_base})
    write_json(args.output_dir / "batch_manifest.json", manifest)
    print(
        json.dumps(
            {
                "provider": manifest["provider"],
                "request_count": manifest["request_count"],
                "batch_input_sha256": manifest["batch_input_sha256"],
                "status": manifest["status"],
            },
            indent=2,
        )
    )
    return 0


def command_prepare_corrective(args: argparse.Namespace) -> int:
    manifest = prepare_corrective_batch(
        original_batch_dir=args.original_batch_dir,
        failed_import_dir=args.failed_import_dir,
        output_dir=args.output_dir,
        git_state=git_state(),
    )
    manifest = inherit_connection_metadata(manifest, args.original_batch_dir, args.output_dir)
    print(
        json.dumps(
            {
                "provider": manifest["provider"],
                "request_count": manifest["request_count"],
                "batch_input_sha256": manifest["batch_input_sha256"],
                "status": manifest["status"],
            },
            indent=2,
        )
    )
    return 0


def _openai_count(requests: list[dict[str, Any]], client=None) -> list[dict[str, Any]]:
    client = client or _transport_client("openai")
    records = []
    for index, request in enumerate(requests, start=1):
        body = request["body"]
        response = client.responses.input_tokens.count(
            model=body["model"],
            instructions=body["instructions"],
            input=body["input"],
            reasoning=body["reasoning"],
            text=body["text"],
        )
        records.append(
            {"custom_id": request["custom_id"], "input_tokens": response.input_tokens}
        )
        print(f"openai token count {index}/{len(requests)}", flush=True)
    return records


def _anthropic_count(requests: list[dict[str, Any]], client=None) -> list[dict[str, Any]]:
    client = client or _transport_client("anthropic")
    records = []
    for index, request in enumerate(requests, start=1):
        params = request["params"]
        response = client.messages.count_tokens(
            model=params["model"],
            system=params["system"],
            messages=params["messages"],
            output_config=params["output_config"],
        )
        records.append(
            {"custom_id": request["custom_id"], "input_tokens": response.input_tokens}
        )
        print(f"anthropic token count {index}/{len(requests)}", flush=True)
    return records


def command_count(args: argparse.Namespace) -> int:
    manifest, requests = load_prepared_batch(args.batch_dir)
    provider = manifest["provider"]
    actor = manifest.get("judge") or manifest.get("generator")
    if not isinstance(actor, dict):
        raise ValueError("Prepared batch manifest lacks a judge or generator actor")
    output_path = args.output or args.batch_dir / "token_count_and_cost.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite token count: {output_path}")
    client = _transport_client(provider, args, manifest)
    records = _openai_count(requests, client) if provider == "openai" else _anthropic_count(requests, client)
    token_summary = summarize_token_counts([item["input_tokens"] for item in records])
    effective_input_price = (
        args.standard_input_price_per_million * args.batch_price_multiplier
    )
    effective_output_price = (
        args.standard_output_price_per_million * args.batch_price_multiplier
    )
    maximum_output_tokens = (
        manifest["request_count"] * manifest["max_output_tokens_per_request"]
    )
    planning_output_tokens = (
        manifest["request_count"] * args.planning_output_tokens_per_request
    )
    input_cost = token_summary["total_input_tokens"] / 1_000_000 * effective_input_price
    planning_output_cost = planning_output_tokens / 1_000_000 * effective_output_price
    maximum_output_cost = maximum_output_tokens / 1_000_000 * effective_output_price
    result = {
        "schema_version": "1.0.0",
        "created_at": utc_now(),
        "status": "TOKEN_COUNT_COMPLETE_PAID_BATCH_NOT_SUBMITTED",
        "provider": provider,
        "model": actor["model"],
        "effort": actor["effort"],
        "prepared_batch_manifest_sha256": sha256_file(
            args.batch_dir / "batch_manifest.json"
        ),
        "batch_input_sha256": manifest["batch_input_sha256"],
        "token_count_method": "provider_token_count_api_no_generation",
        "input_tokens": token_summary,
        "output_budget": {
            "planning_tokens_per_request": args.planning_output_tokens_per_request,
            "planning_total_tokens": planning_output_tokens,
            "hard_maximum_tokens_per_request": manifest["max_output_tokens_per_request"],
            "hard_maximum_total_tokens": maximum_output_tokens,
        },
        "pricing_usd_per_million_tokens": {
            "standard_input": args.standard_input_price_per_million,
            "standard_output": args.standard_output_price_per_million,
            "batch_price_multiplier": args.batch_price_multiplier,
            "effective_batch_input": effective_input_price,
            "effective_batch_output": effective_output_price,
            "source_url": args.pricing_source_url,
            "checked_on": args.pricing_checked_on,
        },
        "cost_estimate_usd": {
            "known_input": round(input_cost, 6),
            "planning_output": round(planning_output_cost, 6),
            "planning_total": round(input_cost + planning_output_cost, 6),
            "hard_maximum_output": round(maximum_output_cost, 6),
            "hard_maximum_total": round(input_cost + maximum_output_cost, 6),
            "reasoning_tokens_are_billed_as_output": True,
        },
        "records": records,
    }
    write_json(output_path, result)
    print(json.dumps(result["cost_estimate_usd"], indent=2))
    return 0


def command_submit(args: argparse.Namespace) -> int:
    if args.confirm_paid_submission != PAID_APPROVAL_PHRASE:
        raise ValueError(
            "Paid submission blocked. Pass --confirm-paid-submission "
            f"{PAID_APPROVAL_PHRASE} only after recorded user approval."
        )
    manifest, requests = load_prepared_batch(args.batch_dir)
    _enforce_spj_submission_policy(manifest)
    cost_path = args.batch_dir / "token_count_and_cost.json"
    if not cost_path.exists():
        raise ValueError("Token count and cost gate must be completed before submission")
    cost = json.loads(cost_path.read_text(encoding="utf-8"))
    if cost["batch_input_sha256"] != manifest["batch_input_sha256"]:
        raise ValueError("Cost estimate does not authenticate this prepared batch")
    receipt_path = args.receipt or args.batch_dir / "submission_receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"Refusing to overwrite submission receipt: {receipt_path}")
    provider = manifest["provider"]
    if provider == "openai":
        client = _transport_client("openai", args, manifest)
        with (args.batch_dir / manifest["batch_input_path"]).open("rb") as handle:
            uploaded = client.files.create(file=handle, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={"study": manifest.get("study_id", "spj-g1-frontier-replication-v1")},
        )
        provider_receipt = {
            "uploaded_file": uploaded.model_dump(mode="json"),
            "batch": batch.model_dump(mode="json"),
        }
        batch_id = batch.id
    else:
        client = _transport_client("anthropic", args, manifest)
        batch = client.messages.batches.create(requests=requests)
        provider_receipt = {"batch": batch.model_dump(mode="json")}
        batch_id = batch.id
    receipt = {
        "schema_version": "1.0.0",
        "submitted_at": utc_now(),
        "status": "PAID_BATCH_SUBMITTED",
        "provider": provider,
        "batch_id": batch_id,
        "prepared_batch_manifest_sha256": sha256_file(
            args.batch_dir / "batch_manifest.json"
        ),
        "batch_input_sha256": manifest["batch_input_sha256"],
        "cost_gate_sha256": sha256_file(cost_path),
        "provider_receipt": provider_receipt,
        "git": git_state(),
    }
    write_json(receipt_path, receipt)
    print(json.dumps({"provider": provider, "batch_id": batch_id}, indent=2))
    return 0


def command_retrieve(args: argparse.Namespace) -> int:
    manifest = None
    if getattr(args, "batch_dir", None) is not None:
        manifest, _ = load_prepared_batch(args.batch_dir)
    role = model_role(manifest, getattr(args, "role", None))
    if manifest and args.provider and args.provider != manifest["provider"]:
        raise ValueError("Provider does not match the prepared batch manifest.")
    args.provider = args.provider or (manifest or {}).get("provider") or native_config(role).provider
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite downloaded results: {args.output}")
    if args.provider == "openai":
        client = _transport_client("openai", args, manifest)
        batch = client.batches.retrieve(args.batch_id)
        if batch.status != "completed" or not batch.output_file_id:
            print(json.dumps(batch.model_dump(mode="json"), indent=2, default=str))
            return 2
        response = client.files.content(batch.output_file_id)
        args.output.write_bytes(response.content)
    else:
        client = _transport_client("anthropic", args, manifest)
        batch = client.messages.batches.retrieve(args.batch_id)
        if batch.processing_status != "ended":
            print(json.dumps(batch.model_dump(mode="json"), indent=2, default=str))
            return 2
        with args.output.open("w", encoding="utf-8") as handle:
            for item in client.messages.batches.results(args.batch_id):
                handle.write(json.dumps(item.model_dump(mode="json"), sort_keys=True) + "\n")
    print(json.dumps({"results_path": str(args.output), "sha256": sha256_file(args.output)}, indent=2))
    return 0


def command_import(args: argparse.Namespace) -> int:
    manifest = import_batch_results(
        provider=args.provider,
        results_path=args.results,
        batch_dir=args.batch_dir,
        output_dir=args.output_dir,
        git_state=git_state(),
    )
    print(
        json.dumps(
            {
                "completed_evaluations": manifest["completed_evaluations"],
                "failed_evaluations": manifest["failed_evaluations"],
                "status": manifest["status"],
            },
            indent=2,
        )
    )
    return 1 if manifest["failed_evaluations"] else 0


def command_merge_corrective(args: argparse.Namespace) -> int:
    manifest = merge_corrective_imports(
        original_import_dir=args.original_import_dir,
        corrective_batch_dir=args.corrective_batch_dir,
        corrective_import_dir=args.corrective_import_dir,
        output_dir=args.output_dir,
        git_state=git_state(),
    )
    print(
        json.dumps(
            {
                "completed_evaluations": manifest["completed_evaluations"],
                "failed_evaluations": manifest["failed_evaluations"],
                "status": manifest["status"],
            },
            indent=2,
        )
    )
    return 0


def inherit_connection_metadata(manifest, source_dir, output_dir):
    source = load_json(source_dir / "batch_manifest.json")
    manifest["model_role"] = model_role(source)
    if source.get("api_base"):
        config = native_config(manifest["model_role"], provider=source["provider"], api_base=source["api_base"])
        manifest["api_base"] = config.api_base
    write_json(output_dir / "batch_manifest.json", manifest)
    return manifest


def add_connection_arguments(parser):
    parser.add_argument("--role", choices=("judge", "proposal"), default=None,
                        help="Credential role; inferred from a prepared batch when available.")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None, help="Prefer a role-specific environment variable.")
    parser.add_argument("--timeout", type=float, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--provider", choices=("openai", "anthropic"))
    prepare.add_argument("--presentations", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--judge-id", default="spj-configured-judge")
    prepare.add_argument("--model")
    prepare.add_argument(
        "--effort", choices=("none", "minimal", "low", "medium", "high", "xhigh", "max")
    )
    prepare.add_argument("--max-output-tokens", type=int)
    prepare.set_defaults(func=command_prepare)

    corrective = subparsers.add_parser("prepare-corrective")
    corrective.add_argument("--original-batch-dir", type=Path, required=True)
    corrective.add_argument("--failed-import-dir", type=Path, required=True)
    corrective.add_argument("--output-dir", type=Path, required=True)
    corrective.set_defaults(func=command_prepare_corrective)

    count = subparsers.add_parser("count")
    count.add_argument("--batch-dir", type=Path, required=True)
    count.add_argument("--output", type=Path)
    count.add_argument("--standard-input-price-per-million", type=float, required=True)
    count.add_argument("--standard-output-price-per-million", type=float, required=True)
    count.add_argument("--batch-price-multiplier", type=float, default=0.5)
    count.add_argument("--planning-output-tokens-per-request", type=int, default=4096)
    count.add_argument("--pricing-source-url", required=True)
    count.add_argument("--pricing-checked-on", required=True)
    count.set_defaults(func=command_count)

    submit = subparsers.add_parser("submit")
    submit.add_argument("--batch-dir", type=Path, required=True)
    submit.add_argument("--receipt", type=Path)
    submit.add_argument("--confirm-paid-submission", required=True)
    submit.set_defaults(func=command_submit)

    retrieve = subparsers.add_parser("retrieve")
    retrieve.add_argument("--provider", choices=("openai", "anthropic"))
    retrieve.add_argument("--batch-dir", type=Path, help="Optional prepared batch to select its provider and credential role.")
    retrieve.add_argument("--batch-id", required=True)
    retrieve.add_argument("--output", type=Path, required=True)
    retrieve.add_argument(
        "--anthropic-workspace-id",
        help=(
            "Anthropic workspace for an unscoped API key; otherwise uses "
            "ANTHROPIC_WORKSPACE_ID when set"
        ),
    )
    retrieve.set_defaults(func=command_retrieve)

    import_parser = subparsers.add_parser("import")
    import_parser.add_argument("--provider", choices=("openai", "anthropic"), required=True)
    import_parser.add_argument("--batch-dir", type=Path, required=True)
    import_parser.add_argument("--results", type=Path, required=True)
    import_parser.add_argument("--output-dir", type=Path, required=True)
    import_parser.set_defaults(func=command_import)

    merge = subparsers.add_parser("merge-corrective")
    merge.add_argument("--original-import-dir", type=Path, required=True)
    merge.add_argument("--corrective-batch-dir", type=Path, required=True)
    merge.add_argument("--corrective-import-dir", type=Path, required=True)
    merge.add_argument("--output-dir", type=Path, required=True)
    merge.set_defaults(func=command_merge_corrective)
    for command in (prepare, count, submit, retrieve):
        add_connection_arguments(command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
