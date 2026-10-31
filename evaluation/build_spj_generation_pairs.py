#!/usr/bin/env python3
"""Build one blinded, mirrored judge set for every candidate in an SPJ sweep."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from .build_blinded_pairs import assert_matched_design, discover_cases, load_completed_run
from .common import sha256_file, utc_now, write_json


def build_generation_dataset(
    sweep_dir: Path,
    output_dir: Path,
    *,
    seed: int,
    include_evaluator_keys: bool = False,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite blinded generation set: {output_dir}")
    sweep_manifest_path = sweep_dir / "sweep_manifest.json"
    sweep_manifest = json.loads(sweep_manifest_path.read_text(encoding="utf-8"))
    if sweep_manifest.get("failed_candidates"):
        raise ValueError("Candidate sweep contains failed candidates")
    records = [
        record
        for record in sweep_manifest["records"]
        if record["candidate_id"] != "baseline"
    ]
    if not records:
        raise ValueError("Candidate sweep contains no mutations")

    baseline = load_completed_run(sweep_dir / "candidates" / "baseline")
    baseline_cases = discover_cases(baseline)
    rng = random.Random(seed)
    presentations: list[dict[str, Any]] = []
    allocation: dict[str, Any] = {}
    systems = {
        "baseline": {
            "candidate_sha256": baseline["manifest"]["intervention"]["candidate_sha256"],
            "model_id": baseline["manifest"]["models"]["agent"]["model_id"],
            "manifest_sha256": sha256_file(baseline["manifest_path"]),
        }
    }
    comparisons = []
    used_ids: set[str] = set()

    for candidate_index, record in enumerate(sorted(records, key=lambda item: item["candidate_id"]), start=1):
        candidate_id = record["candidate_id"]
        candidate = load_completed_run(sweep_dir / "candidates" / candidate_id)
        assert_matched_design(baseline, candidate)
        candidate_cases = discover_cases(candidate)
        if set(candidate_cases) != set(baseline_cases):
            raise ValueError(f"Candidate {candidate_id} has a mismatched case set")
        candidate_model = dict(candidate["manifest"]["models"]["agent"])
        baseline_model = dict(baseline["manifest"]["models"]["agent"])
        candidate_model.pop("accessed_at", None)
        baseline_model.pop("accessed_at", None)
        if candidate_model != baseline_model:
            raise ValueError("Generation candidates must use the same model as baseline")
        systems[candidate_id] = {
            "candidate_sha256": candidate["manifest"]["intervention"]["candidate_sha256"],
            "model_id": candidate["manifest"]["models"]["agent"]["model_id"],
            "manifest_sha256": sha256_file(candidate["manifest_path"]),
        }
        comparison_id = f"K{candidate_index:02d}"
        comparisons.append({"comparison_id": comparison_id, "candidate_id": candidate_id})
        case_names = sorted(baseline_cases)
        rng.shuffle(case_names)
        for case_index, case_name in enumerate(case_names, start=1):
            case_id = f"{comparison_id}-C{case_index:03d}"
            first_a_is_candidate = bool(rng.getrandbits(1))
            for order_index, a_is_candidate in enumerate(
                (first_a_is_candidate, not first_a_is_candidate), start=1
            ):
                while True:
                    presentation_id = f"P{rng.getrandbits(64):016x}"
                    if presentation_id not in used_ids:
                        used_ids.add(presentation_id)
                        break
                side_a = candidate_cases[case_name] if a_is_candidate else baseline_cases[case_name]
                side_b = baseline_cases[case_name] if a_is_candidate else candidate_cases[case_name]
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
                if include_evaluator_keys:
                    key_a = side_a.get("evaluator_only")
                    if not key_a or key_a != side_b.get("evaluator_only"):
                        raise ValueError("Missing or mismatched reviewed evaluator key")
                    presentations[-1]["evaluator_only"] = {
                        field: key_a[field] for field in (
                            "required_response_properties", "prohibited_claims",
                            "bidirectional_constraints", "balanced_response_requirements",
                        ) if field in key_a
                    }
                allocation[presentation_id] = {
                    "comparison_id": comparison_id,
                    "candidate_id": candidate_id,
                    "case_id": case_id,
                    "case_name": case_name,
                    "A": candidate_id if a_is_candidate else "baseline",
                    "B": "baseline" if a_is_candidate else candidate_id,
                }

    output_dir.mkdir(parents=True)
    presentations_path = output_dir / "presentations.jsonl"
    with presentations_path.open("w", encoding="utf-8") as handle:
        for presentation in presentations:
            handle.write(json.dumps(presentation, ensure_ascii=False) + "\n")
    key = {
        "schema_version": "1.0.0",
        "created_at": utc_now(),
        "blinding": "JUDGE_BLINDED_ALLOCATION_KEY_PRIVATE",
        "seed": seed,
        "systems": systems,
        "comparisons": comparisons,
        "presentations": allocation,
    }
    write_json(output_dir / "allocation.key", key)
    manifest = {
        "schema_version": "1.0.0",
        "created_at": utc_now(),
        "phase": "development",
        "method": "SPJ-Evolve",
        "candidate_count": len(records),
        "case_count_per_candidate": len(baseline_cases),
        "presentation_count": len(presentations),
        "mirrored_order": True,
        "judge_visible_system_identity": False,
        "evaluation_version": "reviewed-case-keys-v1" if include_evaluator_keys else "general-rubric-v1",
        "presentations_sha256": sha256_file(presentations_path),
        "allocation_key_sha256": sha256_file(output_dir / "allocation.key"),
        "sweep_manifest_sha256": sha256_file(sweep_manifest_path),
    }
    write_json(output_dir / "blinding_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_generation_dataset(args.sweep_dir, args.output_dir, seed=args.seed)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
