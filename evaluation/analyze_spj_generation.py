#!/usr/bin/env python3
"""Analyze a multi-candidate blinded SPJ generation under the frozen selection rule."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .analyze_pairwise import cohens_kappa, load_scores
from .common import DOMAIN_NAMES, load_json, sha256_file, utc_now, write_json
from .judge_pairwise import load_presentations


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def cluster_bootstrap(
    case_values: dict[str, float], *, samples: int, seed: int
) -> dict[str, Any]:
    estimate = mean(list(case_values.values()))
    if len(case_values) < 2:
        return {"estimate": estimate, "interval_95": None, "case_count": len(case_values)}
    cases = sorted(case_values)
    rng = random.Random(seed)
    estimates = [
        statistics.mean(case_values[rng.choice(cases)] for _ in cases)
        for _ in range(samples)
    ]
    return {
        "estimate": estimate,
        "interval_95": [percentile(estimates, 0.025), percentile(estimates, 0.975)],
        "case_count": len(cases),
        "bootstrap_samples": samples,
        "seed": seed,
    }


def simultaneous_one_sided_lower_bounds(
    candidate_case_values: dict[str, dict[str, float]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Return paired case-cluster max-T 95% lower confidence bounds."""

    if samples < 1:
        raise ValueError("At least one bootstrap sample is required")
    if not candidate_case_values:
        raise ValueError("No candidate case values supplied")
    candidate_ids = sorted(candidate_case_values)
    case_sets = {tuple(sorted(candidate_case_values[item])) for item in candidate_ids}
    if len(case_sets) != 1:
        raise ValueError("Candidates do not share the same case clusters")
    cases = list(next(iter(case_sets)))
    if len(cases) < 2:
        raise ValueError("At least two shared case clusters are required")

    estimates = {
        candidate_id: statistics.mean(candidate_case_values[candidate_id].values())
        for candidate_id in candidate_ids
    }
    standard_errors = {
        candidate_id: statistics.stdev(candidate_case_values[candidate_id].values())
        / math.sqrt(len(cases))
        for candidate_id in candidate_ids
    }
    rng = random.Random(seed)
    maximum_t_statistics = []
    for _ in range(samples):
        sampled_cases = [rng.choice(cases) for _ in cases]
        statistics_for_sample = []
        for candidate_id in candidate_ids:
            standard_error = standard_errors[candidate_id]
            if standard_error == 0:
                continue
            bootstrap_estimate = statistics.mean(
                candidate_case_values[candidate_id][case_id]
                for case_id in sampled_cases
            )
            statistics_for_sample.append(
                (estimates[candidate_id] - bootstrap_estimate) / standard_error
            )
        maximum_t_statistics.append(max(statistics_for_sample, default=0.0))
    critical_value = percentile(maximum_t_statistics, 0.95)
    return {
        "method": (
            "paired nonparametric case-cluster bootstrap; maximum centered mean "
            "deviation standardized by the observed case-cluster standard error"
        ),
        "confidence_level": 0.95,
        "sidedness": "one-sided lower",
        "family": candidate_ids,
        "shared_case_count": len(cases),
        "bootstrap_samples": samples,
        "seed": seed,
        "max_t_critical_value": critical_value,
        "candidates": {
            candidate_id: {
                "estimate": estimates[candidate_id],
                "standard_error": standard_errors[candidate_id],
                "simultaneous_lower_95_bound": estimates[candidate_id]
                - critical_value * standard_errors[candidate_id],
            }
            for candidate_id in candidate_ids
        },
    }


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    left_mean, right_mean = statistics.mean(left), statistics.mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return None if denominator == 0 else numerator / denominator


def assistant_characters(messages: list[dict[str, Any]]) -> int:
    return sum(
        len(str(message.get("content", "")))
        for message in messages
        if message.get("role") == "assistant"
    )


def analyze_generation(
    presentations_path: Path,
    allocation_key_path: Path,
    judge_output_dir: Path,
    sweep_dir: Path,
    *,
    bootstrap_samples: int,
    seed: int,
    minimum_judges: int = 2,
    simultaneous_lower_bound_threshold: float | None = None,
    require_openai_batch_only: bool = False,
) -> dict[str, Any]:
    presentations = {
        item["presentation_id"]: item for item in load_presentations(presentations_path)
    }
    key = load_json(allocation_key_path)
    allocation = key["presentations"]
    if set(presentations) != set(allocation):
        raise ValueError("Presentation IDs differ between blinded data and allocation key")
    results = load_scores(judge_output_dir)
    judge_manifest = load_json(judge_output_dir / "judge_manifest.json")
    if minimum_judges < 1:
        raise ValueError("minimum_judges must be positive")
    if judge_manifest.get("judge_count", 0) < minimum_judges:
        raise ValueError(
            f"SPJ selection requires at least {minimum_judges} completed judge(s)"
        )
    expected = len(presentations) * judge_manifest["judge_count"]
    if len(results) != expected:
        raise ValueError(f"Expected {expected} judge results, found {len(results)}")

    sweep_manifest = load_json(sweep_dir / "sweep_manifest.json")
    latency = {
        record["candidate_id"]: record.get("latency", {}).get("median_seconds")
        for record in sweep_manifest["records"]
    }
    structural_flags = {
        record["candidate_id"]: record.get("screen_flag_counts", {}).get("structural", 0)
        for record in sweep_manifest["records"]
    }
    candidate_ids = [item["candidate_id"] for item in key["comparisons"]]
    resolved_preferences: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    preference_points: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    direct_points: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    domain_totals: dict[str, list[float]] = defaultdict(list)
    domain_deltas: dict[str, dict[str, list[float]]] = {
        candidate: {domain: [] for domain in DOMAIN_NAMES} for candidate in candidate_ids
    }
    raw_sides: Counter[str] = Counter()
    critical_counts: dict[str, Counter[str]] = defaultdict(Counter)
    critical_counts_by_comparison: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    critical_details: list[dict[str, Any]] = []
    length_differences: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    execution_modes: dict[str, Counter[str]] = defaultdict(Counter)

    for result in results:
        judge_id = result["judge"]["id"]
        presentation_id = result["presentation_id"]
        assignment = allocation[presentation_id]
        candidate_id = assignment["candidate_id"]
        case_id = assignment.get("case_name", assignment["case_id"])
        score = result["score"]
        execution_modes[candidate_id][
            result.get("judge", {}).get("provider", "unspecified")
        ] += 1
        winner = score["preference"]["winner"]
        raw_sides[winner] += 1
        resolved = winner if winner == "tie" else assignment[winner]
        resolved_preferences[(candidate_id, judge_id, case_id)].append(resolved)
        preference_points[(candidate_id, judge_id, case_id)].append(
            1.0 if resolved == candidate_id else 0.5 if resolved == "tie" else 0.0
        )
        direct = score["direct_question_resolution"]["winner"]
        resolved_direct = direct if direct == "tie" else assignment[direct]
        direct_points[(candidate_id, judge_id, case_id)].append(
            1.0 if resolved_direct == candidate_id else 0.5 if resolved_direct == "tie" else 0.0
        )

        totals: dict[str, float] = {}
        for side in ("A", "B"):
            system = assignment[side]
            totals[system] = sum(score["scores"][side][domain]["score"] for domain in DOMAIN_NAMES)
        domain_totals[candidate_id].append(totals[candidate_id])
        for domain in DOMAIN_NAMES:
            candidate_side = "A" if assignment["A"] == candidate_id else "B"
            baseline_side = "B" if candidate_side == "A" else "A"
            domain_deltas[candidate_id][domain].append(
                score["scores"][candidate_side][domain]["score"]
                - score["scores"][baseline_side][domain]["score"]
            )

        presentation = presentations[presentation_id]
        candidate_side = "A" if assignment["A"] == candidate_id else "B"
        baseline_side = "B" if candidate_side == "A" else "A"
        length_differences[(candidate_id, judge_id, case_id)].append(
            assistant_characters(presentation[f"transcript_{candidate_side}"])
            - assistant_characters(presentation[f"transcript_{baseline_side}"])
        )
        for item in score["critical_error_candidates"]:
            system = assignment[item["transcript"]]
            critical_counts[system][item["category_id"]] += 1
            critical_counts_by_comparison[(candidate_id, system)][item["category_id"]] += 1
            critical_details.append(
                {
                    "candidate_comparison": candidate_id,
                    "system": system,
                    "judge_id": judge_id,
                    "presentation_id": presentation_id,
                    **item,
                }
            )

    candidate_summaries = []
    case_values_by_candidate: dict[str, dict[str, float]] = {}
    judge_ids = sorted(judge_manifest["judge_ids"])
    for candidate_index, candidate_id in enumerate(candidate_ids):
        candidate_keys = [key_tuple for key_tuple in preference_points if key_tuple[0] == candidate_id]
        mirrored_preference = {
            key_tuple: statistics.mean(preference_points[key_tuple]) for key_tuple in candidate_keys
        }
        mirrored_direct = {
            key_tuple: statistics.mean(direct_points[key_tuple]) for key_tuple in candidate_keys
        }
        case_values = {
            case_id: statistics.mean(
                value
                for (cand, _judge, case), value in mirrored_preference.items()
                if cand == candidate_id and case == case_id
            )
            for case_id in sorted({key_tuple[2] for key_tuple in candidate_keys})
        }
        case_values_by_candidate[candidate_id] = case_values
        order_stable = sum(
            len(set(resolved_preferences[key_tuple])) == 1 for key_tuple in candidate_keys
        )
        strict_outcomes: Counter[str] = Counter()
        for key_tuple in candidate_keys:
            outcomes = resolved_preferences[key_tuple]
            strict_outcomes[
                outcomes[0] if len(set(outcomes)) == 1 else "unstable"
            ] += 1
        agreement = []
        if len(judge_ids) >= 2:
            cases = sorted(case_values)
            for left_index, left_judge in enumerate(judge_ids):
                for right_judge in judge_ids[left_index + 1 :]:
                    left = []
                    right = []
                    for case_id in cases:
                        for judge_id, target in ((left_judge, left), (right_judge, right)):
                            values = resolved_preferences[(candidate_id, judge_id, case_id)]
                            target.append(values[0] if len(set(values)) == 1 else "unstable")
                    agreement.append(
                        {
                            "judge_a": left_judge,
                            "judge_b": right_judge,
                            **cohens_kappa(left, right),
                        }
                    )
        candidate_critical = sum(critical_counts[candidate_id].values())
        candidate_structural = int(structural_flags.get(candidate_id, 0))
        candidate_execution_modes = dict(execution_modes[candidate_id])
        batch_only_concordant = (
            set(candidate_execution_modes) == {"openai_responses_batch_api"}
            if require_openai_batch_only
            else True
        )
        eligible = (
            candidate_critical == 0
            and candidate_structural == 0
            and batch_only_concordant
        )
        preference_estimate = statistics.mean(mirrored_preference.values())
        direct_estimate = statistics.mean(mirrored_direct.values())
        advance_threshold_met = preference_estimate >= 0.55 and direct_estimate >= 0.5
        length_x = []
        preference_y = []
        for key_tuple, value in mirrored_preference.items():
            length_x.append(statistics.mean(length_differences[key_tuple]))
            preference_y.append(value)
        candidate_summaries.append(
            {
                "candidate_id": candidate_id,
                "eligible": eligible,
                "ineligibility_reasons": [
                    reason
                    for condition, reason in (
                        (
                            candidate_critical > 0,
                            "unresolved_potential_critical_error_from_llm_judge",
                        ),
                        (candidate_structural > 0, "deterministic_structural_integrity_failure"),
                        (
                            not batch_only_concordant,
                            "non_batch_judge_rescue_confounded_with_candidate",
                        ),
                    )
                    if condition
                ],
                "minimum_advance_margin_met": advance_threshold_met,
                "judge_execution_modes": candidate_execution_modes,
                "batch_only_processing_concordant": batch_only_concordant,
                "preference": {
                    **cluster_bootstrap(
                        case_values,
                        samples=bootstrap_samples,
                        seed=seed + candidate_index,
                    ),
                    "reference_value": 0.5,
                },
                "direct_question_points": direct_estimate,
                "phase_preference_points": {
                    phase: mean(
                        [
                            value
                            for (_candidate, _judge, case_name), value in mirrored_preference.items()
                            if f"-{phase.upper()}-" in case_name
                        ]
                    )
                    for phase in ("pre", "post")
                },
                "phase_direct_question_points": {
                    phase: mean(
                        [
                            value
                            for (_candidate, _judge, case_name), value in mirrored_direct.items()
                            if f"-{phase.upper()}-" in case_name
                        ]
                    )
                    for phase in ("pre", "post")
                },
                "mean_eight_domain_total": mean(domain_totals[candidate_id]),
                "mean_domain_deltas": {
                    domain: mean(values) for domain, values in domain_deltas[candidate_id].items()
                },
                "mirrored_order_stability": {
                    "stable": order_stable,
                    "total_judge_cases": len(candidate_keys),
                    "rate": order_stable / len(candidate_keys),
                    "strict_outcomes": dict(strict_outcomes),
                },
                "judge_agreement": agreement,
                "potential_critical_error_nominations": {
                    "candidate": dict(critical_counts[candidate_id]),
                    "baseline_within_comparison": dict(
                        critical_counts_by_comparison[(candidate_id, "baseline")]
                    ),
                    "not_human_adjudicated": True,
                },
                "median_turn_latency_seconds": latency.get(candidate_id),
                "preference_length_difference_pearson_r": pearson(length_x, preference_y),
            }
        )

    simultaneous_inference = None
    if simultaneous_lower_bound_threshold is not None:
        simultaneous_inference = simultaneous_one_sided_lower_bounds(
            case_values_by_candidate,
            samples=bootstrap_samples,
            seed=seed,
        )
        simultaneous = simultaneous_inference["candidates"]
        for item in candidate_summaries:
            candidate_id = item["candidate_id"]
            lower_bound = simultaneous[candidate_id]["simultaneous_lower_95_bound"]
            item["preference"]["simultaneous_lower_95_bound"] = lower_bound
            item["preference"]["simultaneous_standard_error"] = simultaneous[
                candidate_id
            ]["standard_error"]
            item["minimum_advance_margin_met"] = (
                item["preference"]["estimate"] >= 0.55
                and item["direct_question_points"] >= 0.5
                and lower_bound > simultaneous_lower_bound_threshold
            )

    def rank_key(item: dict[str, Any]) -> tuple[Any, ...]:
        latency_value = item["median_turn_latency_seconds"]
        return (
            0 if item["eligible"] else 1,
            -float(item["preference"]["estimate"]),
            -float(item["direct_question_points"]),
            -float(item["mean_eight_domain_total"]),
            float("inf") if latency_value is None else float(latency_value),
            item["candidate_id"],
        )

    ranked = sorted(candidate_summaries, key=rank_key)
    selected = next(
        (
            item["candidate_id"]
            for item in ranked
            if item["eligible"] and item["minimum_advance_margin_met"]
        ),
        None,
    )
    return {
        "schema_version": "1.0.0",
        "created_at": utc_now(),
        "phase": "development",
        "method": "SPJ-Evolve",
        "candidate_count": len(candidate_ids),
        "judge_ids": judge_ids,
        "selection_rule": {
            "eligible_requires_zero_unresolved_llm_critical_nominations": True,
            "minimum_preference_points": 0.55,
            "minimum_direct_question_points": 0.5,
            "simultaneous_lower_95_bound_must_exceed": simultaneous_lower_bound_threshold,
            "openai_batch_only_required": require_openai_batch_only,
            "potential_critical_errors_are_not_human_adjudicated": True,
        },
        "selected_candidate_id": selected,
        "ranking": [item["candidate_id"] for item in ranked],
        "candidates": candidate_summaries,
        "bias_diagnostics": {
            "raw_side_preferences": dict(raw_sides),
            "raw_A_rate_excluding_ties": (
                raw_sides["A"] / (raw_sides["A"] + raw_sides["B"])
                if raw_sides["A"] + raw_sides["B"]
                else None
            ),
        },
        "simultaneous_inference": simultaneous_inference,
        "critical_error_nomination_details": critical_details,
        "interpretation": (
            "Automated development selection only. LLM nominations are unresolved review "
            "flags, not adjudicated clinical errors; absence is not evidence of safety."
        ),
        "inputs": {
            "presentations_sha256": sha256_file(presentations_path),
            "allocation_key_sha256": sha256_file(allocation_key_path),
            "judge_manifest_sha256": sha256_file(judge_output_dir / "judge_manifest.json"),
            "sweep_manifest_sha256": sha256_file(sweep_dir / "sweep_manifest.json"),
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# SPJ-Evolve generation analysis",
        "",
        f"Selected candidate: `{summary['selected_candidate_id'] or 'NONE'}`",
        "",
        "| Candidate | Eligible | Preference points | Simultaneous lower 95% bound | Direct-question points | Domain total |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    by_id = {item["candidate_id"]: item for item in summary["candidates"]}
    for candidate_id in summary["ranking"]:
        item = by_id[candidate_id]
        lines.append(
            f"| {candidate_id} | {item['eligible']} | "
            f"{item['preference']['estimate']:.3f} | "
            f"{item['preference'].get('simultaneous_lower_95_bound', float('nan')):.3f} | "
            f"{item['direct_question_points']:.3f} | "
            f"{item['mean_eight_domain_total']:.3f} |"
        )
    lines.extend(["", summary["interpretation"], ""])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presentations", type=Path, required=True)
    parser.add_argument("--allocation-key", type=Path, required=True)
    parser.add_argument("--judge-output-dir", type=Path, required=True)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--minimum-judges", type=int, default=2)
    parser.add_argument("--simultaneous-lower-bound-threshold", type=float)
    parser.add_argument("--require-openai-batch-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite analysis output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    summary = analyze_generation(
        args.presentations,
        args.allocation_key,
        args.judge_output_dir,
        args.sweep_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        minimum_judges=args.minimum_judges,
        simultaneous_lower_bound_threshold=args.simultaneous_lower_bound_threshold,
        require_openai_batch_only=args.require_openai_batch_only,
    )
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"selected_candidate_id": summary["selected_candidate_id"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
