#!/usr/bin/env python3
"""Prepare and import frontier pointwise batches; reuse audited batch gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.frontier_pointwise_batch import (
    import_batch_results,
    merge_corrective_imports,
    prepare_batch,
    prepare_corrective_batch,
)
from experiments.manage_frontier_pairwise_batch import (
    PAID_APPROVAL_PHRASE,
    add_connection_arguments,
    prepare_settings,
    inherit_connection_metadata,
    command_count,
    command_retrieve,
    command_submit,
    git_state,
)


def command_prepare(args: argparse.Namespace) -> int:
    config = prepare_settings(args)
    manifest = prepare_batch(
        provider=args.provider,
        items_path=args.items,
        output_dir=args.output_dir,
        judge_id=args.judge_id,
        model=args.model,
        effort=args.effort,
        max_output_tokens=args.max_output_tokens,
        study_id=args.study_id,
        git_state=git_state(),
    )
    manifest.update({"model_role": config.role, "api_base": config.api_base})
    from evaluation.common import write_json
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


def command_prepare_corrective(args: argparse.Namespace) -> int:
    manifest = prepare_corrective_batch(
        original_batch_dir=args.original_batch_dir,
        failed_import_dir=args.failed_import_dir,
        output_dir=args.output_dir,
        max_output_tokens=args.max_output_tokens,
        git_state=git_state(),
    )
    manifest = inherit_connection_metadata(manifest, args.original_batch_dir, args.output_dir)
    print(json.dumps({
        "provider": manifest["provider"],
        "request_count": manifest["request_count"],
        "batch_input_sha256": manifest["batch_input_sha256"],
        "status": manifest["status"],
    }, indent=2))
    return 0


def command_merge_corrective(args: argparse.Namespace) -> int:
    manifest = merge_corrective_imports(
        original_import_dir=args.original_import_dir,
        corrective_batch_dir=args.corrective_batch_dir,
        corrective_import_dir=args.corrective_import_dir,
        output_dir=args.output_dir,
        git_state=git_state(),
    )
    print(json.dumps({
        "status": manifest["status"],
        "completed_evaluations": manifest["completed_evaluations"],
        "corrected_evaluations": manifest["corrective_retry"]["corrected_evaluations"],
    }, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--provider", choices=("openai", "anthropic"))
    prepare.add_argument("--items", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--judge-id", default="spj-configured-pointwise-judge")
    prepare.add_argument("--model")
    prepare.add_argument(
        "--effort", choices=("none", "minimal", "low", "medium", "high", "xhigh", "max")
    )
    prepare.add_argument("--max-output-tokens", type=int)
    prepare.add_argument(
        "--study-id", default="g2-local-baseline-panel-v3-pointwise"
    )
    prepare.set_defaults(func=command_prepare)

    corrective = subparsers.add_parser("prepare-corrective")
    corrective.add_argument("--original-batch-dir", type=Path, required=True)
    corrective.add_argument("--failed-import-dir", type=Path, required=True)
    corrective.add_argument("--output-dir", type=Path, required=True)
    corrective.add_argument("--max-output-tokens", type=int, required=True)
    corrective.set_defaults(func=command_prepare_corrective)

    count = subparsers.add_parser("count")
    count.add_argument("--batch-dir", type=Path, required=True)
    count.add_argument("--output", type=Path)
    count.add_argument("--standard-input-price-per-million", type=float, required=True)
    count.add_argument("--standard-output-price-per-million", type=float, required=True)
    count.add_argument("--batch-price-multiplier", type=float, default=0.5)
    count.add_argument("--planning-output-tokens-per-request", type=int, default=1024)
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
