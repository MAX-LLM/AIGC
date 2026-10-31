#!/usr/bin/env python3
"""Unblind and summarize mirrored pairwise judge outputs without pseudoreplication."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .common import DOMAIN_NAMES, load_json, sha256_file, utc_now, write_json
from .judge_pairwise import load_presentations


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.mean(values) if values else None


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot take percentile of empty values")
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def cohens_kappa(left: list[str], right: list[str]) -> dict[str, float | int | None]:
    if len(left) != len(right):
        raise ValueError("Kappa rating vectors must have equal length")
    if not left:
        return {"n": 0, "agreement": None, "kappa": None}
    categories = sorted(set(left) | set(right))
    n = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / n
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum((left_counts[c] / n) * (right_counts[c] / n) for c in categories)
    kappa = None if math.isclose(expected, 1.0) else (observed - expected) / (1 - expected)
    return {"n": n, "agreement": observed, "kappa": kappa}


def cluster_bootstrap_preference(
    case_values: dict[str, float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    cases = sorted(case_values)
    estimate = _mean(case_values.values())
    if len(cases) < 2 or samples < 1:
        return {
            "case_count": len(cases),
            "estimate": estimate,
            "interval_95": None,
            "reason": "At least two case clusters and one bootstrap sample are required",
        }
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        sampled = [rng.choice(cases) for _ in cases]
        estimates.append(statistics.mean(case_values[case] for case in sampled))
    return {
        "case_count": len(cases),
        "estimate": estimate,
        "interval_95": [_percentile(estimates, 0.025), _percentile(estimates, 0.975)],
        "bootstrap_samples": samples,
        "seed": seed,
        "unit": "case cluster after averaging judges and mirrored orders",
    }


def load_scores(judge_output_dir: Path) -> list[dict[str, Any]]:
    manifest = load_json(judge_output_dir / "judge_manifest.json")
    if manifest.get("dry_run"):
        raise ValueError("Cannot analyze dry-run judge prompts")
    if manifest.get("failed_evaluations"):
        raise ValueError("Judge run contains failures; resolve or prespecify failure handling")
    results = []
    for record in manifest["records"]:
        result = load_json(judge_output_dir / record["score_path"])
        results.append(result)
    if len(results) != manifest["expected_evaluations"]:
        raise ValueError("Judge output count does not match the manifest")
    return results


def analyze(
    presentations_path: Path,
    allocation_key_path: Path,
    judge_output_dir: Path,
    *,
    bootstrap_samples: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    presentations = {
        item["presentation_id"]: item for item in load_presentations(presentations_path)
    }
    key = load_json(allocation_key_path)
    allocation = key["presentations"]
    if set(presentations) != set(allocation):
        raise ValueError("Presentation IDs differ between public data and allocation key")
    results = load_scores(judge_output_dir)
    system_labels = list(key["systems"])
    if len(system_labels) != 2:
        raise ValueError("Pairwise analysis requires exactly two systems")
    reference, comparator = system_labels

    resolved_by_judge_case: dict[tuple[str, str], list[str]] = defaultdict(list)
    preference_scores_by_judge_case: dict[tuple[str, str], list[float]] = defaultdict(list)
    raw_side_preferences = Counter()
    raw_side_preferences_by_judge: dict[str, Counter] = defaultdict(Counter)
    direct_by_judge_case: dict[tuple[str, str], list[str]] = defaultdict(list)
    domain_values: dict[str, dict[str, list[float]]] = {
        system: {domain: [] for domain in DOMAIN_NAMES} for system in system_labels
    }
    critical_candidates = Counter()
    critical_candidate_details = []
    response_lengths: dict[str, list[int]] = {system: [] for system in system_labels}
    judge_models: dict[str, str] = {}
    self_preference_scores: dict[str, list[float]] = defaultdict(list)

    for result in results:
        judge_id = result["judge"]["id"]
        judge_model = result["judge"].get("model", "")
        judge_models[judge_id] = judge_model
        presentation_id = result["presentation_id"]
        score = result["score"]
        assignment = allocation[presentation_id]
        case_id = assignment["case_id"]
        raw_winner = score["preference"]["winner"]
        raw_side_preferences[raw_winner] += 1
        raw_side_preferences_by_judge[judge_id][raw_winner] += 1
        resolved_winner = raw_winner if raw_winner == "tie" else assignment[raw_winner]
        resolved_by_judge_case[(judge_id, case_id)].append(resolved_winner)
        preference_scores_by_judge_case[(judge_id, case_id)].append(
            1.0 if resolved_winner == reference else 0.0 if resolved_winner == comparator else 0.5
        )
        self_system = next(
            (
                system
                for system, metadata in key["systems"].items()
                if metadata.get("model_id") == judge_model
            ),
            None,
        )
        if self_system is not None:
            self_preference_scores[judge_id].append(
                1.0 if resolved_winner == self_system else 0.5 if resolved_winner == "tie" else 0.0
            )
        direct_winner = score["direct_question_resolution"]["winner"]
        resolved_direct = direct_winner if direct_winner == "tie" else assignment[direct_winner]
        direct_by_judge_case[(judge_id, case_id)].append(resolved_direct)

        presentation = presentations[presentation_id]
        for side in ("A", "B"):
            system = assignment[side]
            for domain in DOMAIN_NAMES:
                domain_values[system][domain].append(score["scores"][side][domain]["score"])
            assistant_chars = sum(
                len(str(message.get("content", "")))
                for message in presentation[f"transcript_{side}"]
                if message.get("role") == "assistant"
            )
            response_lengths[system].append(assistant_chars)
        for candidate in score["critical_error_candidates"]:
            system = assignment[candidate["transcript"]]
            critical_candidates[(system, candidate["category_id"])] += 1
            assistant_messages = [
                str(message.get("content", ""))
                for message in presentation[f"transcript_{candidate['transcript']}"]
                if message.get("role") == "assistant"
            ]
            claimed_turn = candidate["turn"]
            evidence = candidate["evidence"]
            exact_turn_match = (
                1 <= claimed_turn <= len(assistant_messages)
                and evidence in assistant_messages[claimed_turn - 1]
            )
            found_turns = [
                index
                for index, text in enumerate(assistant_messages, start=1)
                if evidence in text
            ]
            critical_candidate_details.append(
                {
                    "judge_id": judge_id,
                    "presentation_id": presentation_id,
                    "case_id": case_id,
                    "system": system,
                    "category_id": candidate["category_id"],
                    "claimed_turn": claimed_turn,
                    "evidence": evidence,
                    "rationale": candidate["rationale"],
                    "evidence_exactly_matches_claimed_turn": exact_turn_match,
                    "exact_evidence_found_in_turns": found_turns,
                }
            )

    def consensus(values: list[str]) -> str:
        return values[0] if values and len(set(values)) == 1 else "unstable"

    consensus_records = {
        key_pair: consensus(values) for key_pair, values in resolved_by_judge_case.items()
    }
    direct_consensus = {
        key_pair: consensus(values) for key_pair, values in direct_by_judge_case.items()
    }
    preference_counts = Counter(consensus_records.values())
    direct_counts = Counter(direct_consensus.values())
    order_stable = sum(value != "unstable" for value in consensus_records.values())
    total_consensus = len(consensus_records)

    judges = sorted({judge_id for judge_id, _ in consensus_records})
    cases = sorted({case_id for _, case_id in consensus_records})
    agreement = []
    for left_judge, right_judge in itertools.combinations(judges, 2):
        common_cases = [
            case
            for case in cases
            if (left_judge, case) in consensus_records and (right_judge, case) in consensus_records
        ]
        left = [consensus_records[(left_judge, case)] for case in common_cases]
        right = [consensus_records[(right_judge, case)] for case in common_cases]
        agreement.append(
            {
                "judge_a": left_judge,
                "judge_b": right_judge,
                **cohens_kappa(left, right),
            }
        )

    judge_case_preference_values = {
        key_pair: statistics.mean(values)
        for key_pair, values in preference_scores_by_judge_case.items()
    }
    case_preference_values: dict[str, float] = {}
    for case in cases:
        values = [
            judge_case_preference_values[(judge, case)]
            for judge in judges
            if (judge, case) in judge_case_preference_values
        ]
        if values:
            case_preference_values[case] = statistics.mean(values)

    order_by_judge = {}
    for judge in judges:
        judge_consensus = {
            case: consensus_records[(judge, case)]
            for case in cases
            if (judge, case) in consensus_records
        }
        judge_stable = sum(value != "unstable" for value in judge_consensus.values())
        order_by_judge[judge] = {
            "raw_side_preference_counts": dict(raw_side_preferences_by_judge[judge]),
            "stable_case_count": judge_stable,
            "case_count": len(judge_consensus),
            "stability_rate": judge_stable / len(judge_consensus) if judge_consensus else None,
        }

    domain_summary = {
        system: {
            domain: {
                "mean": _mean(values),
                "rating_count": len(values),
            }
            for domain, values in system_domains.items()
        }
        for system, system_domains in domain_values.items()
    }
    total_scores = {
        system: sum(
            domain_summary[system][domain]["mean"] or 0 for domain in DOMAIN_NAMES
        )
        for system in system_labels
    }

    summary = {
        "schema_version": "1.0.0",
        "created_at": utc_now(),
        "phase": "development_descriptive",
        "systems": system_labels,
        "judges": judges,
        "case_count": len(cases),
        "presentation_count": len(presentations),
        "analysis_method_version": "1.1.0",
        "analysis_unit": (
            "mirrored-order average within judge and case, judge average within case, "
            "and case-cluster aggregation for the descriptive preference interval"
        ),
        "preference": {
            "reference_system": reference,
            "comparator_system": comparator,
            "strict_mirrored_consensus_counts": dict(preference_counts),
            "reference_preference_probability": cluster_bootstrap_preference(
                case_preference_values, samples=bootstrap_samples, seed=seed
            ),
            "handling": (
                "Each mirrored rating is mapped to reference=1, tie=0.5, comparator=0; "
                "orders are averaged within judge and case, then judges within case. "
                "Order-unstable pairs remain in the estimate rather than being excluded."
            ),
            "case_values": case_preference_values,
        },
        "direct_question_resolution_counts": dict(direct_counts),
        "order_bias_diagnostic": {
            "raw_side_preference_counts": dict(raw_side_preferences),
            "stable_judge_case_count": order_stable,
            "total_judge_case_count": total_consensus,
            "stability_rate": order_stable / total_consensus if total_consensus else None,
            "by_judge": order_by_judge,
        },
        "self_preference_diagnostic": {
            judge: {
                "judge_model": judge_models[judge],
                "matched_self_system": next(
                    (
                        system
                        for system, metadata in key["systems"].items()
                        if metadata.get("model_id") == judge_models[judge]
                    ),
                    None,
                ),
                "mean_self_preference_score": _mean(self_preference_scores.get(judge, [])),
                "rating_count": len(self_preference_scores.get(judge, [])),
                "scale": "self=1, tie=0.5, other=0 before mirrored-order averaging",
            }
            for judge in judges
        },
        "interjudge_agreement": agreement,
        "domain_scores": domain_summary,
        "mean_total_domain_score": total_scores,
        "critical_error_candidates_not_adjudicated": [
            {
                "system": system,
                "category_id": category,
                "flag_count": count,
                "exact_claimed_turn_evidence_count": sum(
                    item["evidence_exactly_matches_claimed_turn"]
                    for item in critical_candidate_details
                    if item["system"] == system and item["category_id"] == category
                ),
            }
            for (system, category), count in sorted(critical_candidates.items())
        ],
        "critical_error_candidate_details_private_review": critical_candidate_details,
        "verbosity_diagnostic": {
            system: {
                "mean_assistant_characters_per_transcript": _mean(values),
                "note": "Character count is a tokenizer-dependent, non-clinical burden proxy.",
            }
            for system, values in response_lengths.items()
        },
        "input_hashes": {
            "presentations_sha256": sha256_file(presentations_path),
            "allocation_key_sha256": sha256_file(allocation_key_path),
            "judge_manifest_sha256": sha256_file(judge_output_dir / "judge_manifest.json"),
        },
        "limitations": [
            "LLM ratings are development outcomes, not clinical-effectiveness evidence.",
            "Critical-error candidates require independent blinded expert adjudication.",
            "Intervals from small development case sets are unstable and non-confirmatory.",
            "Mirrored presentations diagnose order sensitivity but do not eliminate all judge biases.",
        ],
    }
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    preference = summary["preference"]
    estimate = preference["reference_preference_probability"]
    interval = estimate["interval_95"]
    interval_text = (
        f"{interval[0]:.3f} to {interval[1]:.3f}" if interval is not None else "not estimable"
    )
    lines = [
        "# Development Pairwise Evaluation",
        "",
        f"Status: `{summary['phase']}`",
        "",
        f"Cases: {summary['case_count']}; judges: {len(summary['judges'])}; presentations: {summary['presentation_count']}.",
        "",
        "## Preference",
        "",
        f"Reference system: `{preference['reference_system']}`; comparator: `{preference['comparator_system']}`.",
        f"Case-clustered descriptive reference preference probability: {estimate['estimate']:.3f} "
        f"(bootstrap interval: {interval_text}).",
        f"Mirrored-order stability: {summary['order_bias_diagnostic']['stability_rate']:.3f}.",
        "",
        "## Mean domain scores",
        "",
        "| System | " + " | ".join(DOMAIN_NAMES) + " | Total |",
        "|---|" + "---:|" * (len(DOMAIN_NAMES) + 1),
    ]
    for system in summary["systems"]:
        values = [summary["domain_scores"][system][domain]["mean"] for domain in DOMAIN_NAMES]
        lines.append(
            f"| {system} | "
            + " | ".join(f"{value:.3f}" for value in values)
            + f" | {summary['mean_total_domain_score'][system]:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This is development-only automated evaluation. LLM critical-error flags are not adjudicated clinical events, and these results cannot support clinical-effectiveness claims.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presentations", type=Path, required=True)
    parser.add_argument("--allocation-key", type=Path, required=True)
    parser.add_argument("--judge-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite analysis output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    summary = analyze(
        args.presentations,
        args.allocation_key,
        args.judge_output_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary["preference"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
