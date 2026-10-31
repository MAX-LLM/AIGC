#!/usr/bin/env python3
"""Build randomized, mirrored A/B transcript presentations from two run folders.

The judge-visible JSONL never contains model names, run IDs, file paths, or the
allocation seed. The allocation key is written separately with a ``.key`` suffix,
which is excluded from Git by the research data policy.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from .common import canonical_sha256, load_json, sha256_file, utc_now, write_json


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_repo_path(value: str, run_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = REPO_ROOT / path
    if candidate.exists():
        return candidate
    return run_dir / path


def load_completed_run(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "experiment_manifest.json"
    validation_path = run_dir / "manifest_validation.json"
    if not manifest_path.is_file() or not validation_path.is_file():
        raise ValueError(f"Run is not sealed with manifest and validation: {run_dir}")
    manifest = load_json(manifest_path)
    validation = load_json(validation_path)
    if not validation.get("ok") or validation.get("errors"):
        raise ValueError(f"Run manifest did not pass validation: {run_dir}")
    if (run_dir / "failure.json").exists():
        raise ValueError(f"Run contains a failure record: {run_dir}")
    transcript_root = _resolve_repo_path(manifest["outputs"]["transcript_root"], run_dir)
    if not transcript_root.is_dir():
        raise ValueError(f"Transcript root does not exist: {transcript_root}")
    return {
        "run_dir": run_dir.resolve(),
        "manifest_path": manifest_path.resolve(),
        "manifest": manifest,
        "transcript_root": transcript_root.resolve(),
    }


def assert_matched_design(left: dict[str, Any], right: dict[str, Any]) -> None:
    left_manifest = left["manifest"]
    right_manifest = right["manifest"]
    matched_fields = {
        "phase": lambda value: value["phase"],
        "prompt bundle": lambda value: value["intervention"]["prompt_bundle_sha256"],
        "knowledge": lambda value: value["intervention"]["knowledge_sha256"],
        "case set": lambda value: value["data"]["case_set_sha256"],
        "case count": lambda value: value["data"]["case_count"],
        "maximum turns": lambda value: value["generation"]["max_turns"],
        "simulator": lambda value: value["generation"]["simulator_parameters"],
    }
    mismatches = [
        name
        for name, getter in matched_fields.items()
        if getter(left_manifest) != getter(right_manifest)
    ]
    left_parameters = dict(left_manifest["generation"]["agent_parameters"])
    right_parameters = dict(right_manifest["generation"]["agent_parameters"])
    left_parameters.pop("health", None)
    right_parameters.pop("health", None)
    if left_parameters != right_parameters:
        mismatches.append("agent generation parameters")
    if mismatches:
        raise ValueError("Runs are not matched on: " + ", ".join(mismatches))


def discover_cases(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for path in sorted(run["transcript_root"].glob("*/*/conversation.json")):
        case_name = path.parent.parent.name
        if case_name in cases:
            raise ValueError(f"Run has multiple transcript bundles for case {case_name}")
        messages = load_json(path)
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Conversation must be a non-empty list: {path}")
        vignette_path = path.parent / "vignette.txt"
        metrics_path = path.parent / "metrics.json"
        if not vignette_path.is_file() or not metrics_path.is_file():
            raise ValueError(f"Incomplete transcript bundle: {path.parent}")
        cases[case_name] = {
            "conversation": messages,
            "conversation_sha256": canonical_sha256(messages),
            "vignette": vignette_path.read_text(encoding="utf-8").strip(),
            "metrics": load_json(metrics_path),
        }
        source_path = path.parent / "source_case.json"
        if source_path.is_file():
            cases[case_name]["evaluator_only"] = load_json(source_path).get("evaluator_only", {})
    expected = int(run["manifest"]["data"]["case_count"])
    if len(cases) != expected:
        raise ValueError(f"Expected {expected} cases but found {len(cases)}")
    return cases


def build_dataset(
    run_a_dir: Path,
    run_b_dir: Path,
    output_dir: Path,
    *,
    label_a: str,
    label_b: str,
    seed: int,
    mirror_order: bool = True,
) -> dict[str, Any]:
    if not label_a or not label_b or label_a == label_b:
        raise ValueError("System labels must be non-empty and distinct")
    run_a = load_completed_run(run_a_dir)
    run_b = load_completed_run(run_b_dir)
    assert_matched_design(run_a, run_b)
    cases_a = discover_cases(run_a)
    cases_b = discover_cases(run_b)
    if set(cases_a) != set(cases_b):
        raise ValueError("Runs contain different case names")
    for case_name in cases_a:
        if cases_a[case_name]["vignette"] != cases_b[case_name]["vignette"]:
            raise ValueError(f"Vignette differs between runs for {case_name}")

    output_dir.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    case_names = sorted(cases_a)
    rng.shuffle(case_names)
    presentations: list[dict[str, Any]] = []
    allocation: dict[str, Any] = {}

    for case_index, case_name in enumerate(case_names, start=1):
        case_id = f"C{case_index:04d}"
        first_a_is_run_a = bool(rng.getrandbits(1))
        orders = [first_a_is_run_a]
        if mirror_order:
            orders.append(not first_a_is_run_a)
        for order_index, a_is_run_a in enumerate(orders, start=1):
            presentation_id = f"{case_id}-P{order_index}"
            side_a = cases_a[case_name] if a_is_run_a else cases_b[case_name]
            side_b = cases_b[case_name] if a_is_run_a else cases_a[case_name]
            label_side_a = label_a if a_is_run_a else label_b
            label_side_b = label_b if a_is_run_a else label_a
            presentations.append(
                {
                    "schema_version": "1.0.0",
                    "presentation_id": presentation_id,
                    "case_id": case_id,
                    "order_replicate": order_index,
                    "vignette": side_a["vignette"],
                    "transcript_A": side_a["conversation"],
                    "transcript_B": side_b["conversation"],
                    "transcript_sha256": {
                        "A": side_a["conversation_sha256"],
                        "B": side_b["conversation_sha256"],
                    },
                }
            )
            allocation[presentation_id] = {
                "case_id": case_id,
                "case_name": case_name,
                "A": label_side_a,
                "B": label_side_b,
            }

    presentations_path = output_dir / "presentations.jsonl"
    with presentations_path.open("w", encoding="utf-8") as handle:
        for presentation in presentations:
            handle.write(json.dumps(presentation, ensure_ascii=False) + "\n")

    key_path = output_dir / "allocation.key"
    key = {
        "schema_version": "1.0.0",
        "created_at": utc_now(),
        "blinding": "JUDGE_BLINDED_ALLOCATION_KEY_PRIVATE",
        "seed": seed,
        "systems": {
            label_a: {
                "run_id": run_a["manifest"]["experiment_id"],
                "model_id": run_a["manifest"]["models"]["agent"]["model_id"],
                "manifest_sha256": sha256_file(run_a["manifest_path"]),
            },
            label_b: {
                "run_id": run_b["manifest"]["experiment_id"],
                "model_id": run_b["manifest"]["models"]["agent"]["model_id"],
                "manifest_sha256": sha256_file(run_b["manifest_path"]),
            },
        },
        "presentations": allocation,
    }
    write_json(key_path, key)

    manifest = {
        "schema_version": "1.0.0",
        "created_at": utc_now(),
        "phase": "development",
        "blinding": {
            "judge_visible_system_identity": False,
            "model_names_removed": True,
            "run_paths_removed": True,
            "mirrored_order": mirror_order,
            "allocation_key_separate": True,
        },
        "case_count": len(case_names),
        "presentation_count": len(presentations),
        "presentations_file": presentations_path.name,
        "presentations_sha256": sha256_file(presentations_path),
        "allocation_key_file": key_path.name,
        "allocation_key_sha256": sha256_file(key_path),
        "matched_design": {
            "prompt_bundle_sha256": run_a["manifest"]["intervention"]["prompt_bundle_sha256"],
            "knowledge_sha256": run_a["manifest"]["intervention"]["knowledge_sha256"],
            "case_set_sha256": run_a["manifest"]["data"]["case_set_sha256"],
        },
    }
    write_json(output_dir / "blinding_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--label-a", required=True)
    parser.add_argument("--label-b", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--no-mirror-order", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_dataset(
        args.run_a,
        args.run_b,
        args.output_dir,
        label_a=args.label_a,
        label_b=args.label_b,
        seed=args.seed,
        mirror_order=not args.no_mirror_order,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
