#!/usr/bin/env python3
"""Run fixed synthetic cases and SPJ overlays through the configured agent API.

Produces the original SPJ sweep layout for the imported reflection, mirrored
judge, analysis, and lesson tools. It uses a running service, without scheduling
HPC jobs or selecting a model checkpoint implicitly.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import time
import uuid

import requests

from evaluation.common import canonical_sha256, sha256_file, utc_now, write_json
from evaluation.deterministic_screen import screen_paths
from evaluation.synthetic_case_bank import case_hash_payload, load_case_bank, select_case_ids
from experiments.spj_runtime_contract import validate_run_delivery
from model_config import load_environment
from prompt_candidate import baseline_candidate, load_prompt_candidate


ROOT = Path(__file__).resolve().parent


def build_parser():
    load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default=os.environ.get("MND_API_BASE", "http://127.0.0.1:8000"))
    parser.add_argument("--case-dir", type=Path, default=ROOT / "research/spj_v1/cases")
    parser.add_argument("--cases", default="all", help="Comma-separated synthetic case IDs, or all.")
    parser.add_argument("--candidate-dir", type=Path, help="Up to six original SPJ candidate JSON overlays; baseline is included automatically.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--plan-only", action="store_true", help="Validate inputs and print the plan without calling any service.")
    return parser


def load_candidates(directory=None):
    candidates = [baseline_candidate()]
    if directory is None:
        return candidates
    paths = sorted(directory.glob("*.json"))
    if not paths or len(paths) > 6:
        raise ValueError("A candidate directory must contain between one and six SPJ overlay JSON files.")
    candidates.extend(load_prompt_candidate(path) for path in paths)
    if len({candidate["candidate_id"] for candidate in candidates}) != len(candidates):
        raise ValueError("Candidate IDs must be unique; baseline is reserved.")
    generations = {candidate["generation"] for candidate in candidates[1:]}
    if len(generations) != 1 or 0 in generations:
        raise ValueError("All mutations must belong to the same positive generation.")
    return candidates


def request_json(method, url, *, timeout, **kwargs):
    # Writes are never retried automatically: a failed connection may already
    # have generated a paid upstream response.
    response = requests.request(method, url, timeout=timeout, **kwargs)
    response.raise_for_status()
    return response.json()


def run_sweep(args):
    if args.timeout <= 0:
        raise ValueError("Request timeout must be positive.")
    candidates = load_candidates(args.candidate_dir)
    cases = load_case_bank(args.case_dir)
    names = select_case_ids(cases, args.cases)
    selected = {name: cases[name] for name in names}
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new sweep directory: {args.output_dir}")
    plan = {
        "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
        "case_ids": names,
        "conversations": len(candidates) * len(names),
        "assistant_turns": len(candidates) * sum(len(cases[name]["turns"]) for name in names),
    }
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return plan
    api = args.api_base.rstrip("/").removesuffix("/v1")
    health = request_json("GET", api + "/health", timeout=args.timeout)
    if not health.get("prompt_candidates_enabled"):
        raise ValueError("Start the agent with MND_ALLOW_PROMPT_CANDIDATES=1 for an offline SPJ sweep.")
    if not health.get("effective_prompts_recorded"):
        raise ValueError("Start the agent with MND_RECORD_EFFECTIVE_PROMPTS=1 to authenticate delivered candidate instructions.")
    # Snapshot only stable model metadata for matched comparisons; session
    # counts and access times are not generation parameters.
    generation_parameters = {key: health.get(key) for key in (
        "backend", "agent_model", "deterministic", "max_new_tokens",
    )}
    provenance = {
        "prompt_bundle_sha256": canonical_sha256({
            filename: sha256_file(ROOT / filename)
            for filename in ("agent_base.py", "mnd_langgraph_agent.py", "prompt_candidate.py")
        }),
        "knowledge_sha256": sha256_file(ROOT / "knowledge_base/mnd_knowledge.json"),
        "case_set_sha256": canonical_sha256(case_hash_payload(cases, names)),
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    run_token = uuid.uuid4().hex
    write_json(output / "execution_inputs.json", {"created_at": utc_now(), "plan": plan, "provenance": provenance, "agent": generation_parameters})
    write_json(output / "candidate_set.json", {"candidates": candidates})
    records = []
    try:
        for candidate in candidates:
            cid = candidate["candidate_id"]
            run_dir = output / "candidates" / cid
            transcript_root = run_dir / "transcripts/scripted"
            write_json(run_dir / "prompt_candidate.json", candidate)
            elapsed_times = []
            for name in names:
                case = cases[name]
                case_dir = transcript_root / name / "run"
                case_dir.mkdir(parents=True)
                session_id = "spj-" + canonical_sha256([run_token, cid, name])[:32]
                headers = {"X-Session-ID": session_id}
                reset = request_json("POST", api + "/reset", timeout=args.timeout,
                    json={"session_id": session_id, "prompt_candidate": candidate}, headers=headers)
                if reset.get("candidate_sha256") != candidate["candidate_sha256"]:
                    raise ValueError("The service did not activate the requested candidate.")
                conversation, metrics = [], []
                for turn_index, patient_turn in enumerate(case["turns"], 1):
                    started = time.monotonic()
                    reply = request_json("POST", api + "/chat", timeout=args.timeout,
                        json={"message": patient_turn, "session_id": session_id}, headers=headers)
                    duration = time.monotonic() - started
                    text = reply.get("response")
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError(f"Empty agent reply in {cid}/{name}/{turn_index}")
                    conversation.extend([{"role": "user", "content": patient_turn}, {"role": "assistant", "content": text}])
                    metrics.append({"turn": turn_index, "elapsed_seconds": duration})
                    elapsed_times.append(duration)
                    write_json(case_dir / "conversation.json", conversation)
                state = request_json("GET", api + "/state", timeout=args.timeout, params={"session_id": session_id}, headers=headers)
                write_json(case_dir / "state.json", state)
                write_json(case_dir / "metrics.json", metrics)
                write_json(case_dir / "source_case.json", case["source_case"])
                (case_dir / "vignette.txt").write_text(case["vignette"] + "\n", encoding="utf-8")
            delivery = validate_run_delivery(run_dir, candidate, selected)
            screen = screen_paths([transcript_root])
            write_json(run_dir / "delivery_validation.json", delivery)
            write_json(run_dir / "deterministic_screen.json", screen)
            counts = {"structural": 0, "lexical_review": 0, "repetition_review": 0}
            for item in screen["results"]:
                counts["structural"] += len(item["result"]["structural_flags"]) + len(item["result"].get("output_integrity_flags", []))
                counts["lexical_review"] += len(item["result"]["lexical_review_flags"])
                counts["repetition_review"] += len(item["result"]["repetition_review_flags"])
            manifest = {
                "manifest_version": "1.0.0", "phase": "development", "created_at": utc_now(),
                "intervention": {"candidate_id": cid, "candidate_sha256": candidate["candidate_sha256"], **{key: value for key, value in provenance.items() if key != "case_set_sha256"}},
                "data": {"case_count": len(names), "case_set_sha256": provenance["case_set_sha256"]},
                "models": {"agent": {"model_id": health.get("agent_model"), "backend": health.get("backend")}},
                "generation": {"max_turns": max(len(cases[name]["turns"]) for name in names), "agent_parameters": generation_parameters, "simulator_parameters": {"type": "fixed_script", "case_order": names}},
                "outputs": {"transcript_root": str(transcript_root)},
                "validation_scope": "portable_spj_case_and_effective_prompt_delivery",
            }
            write_json(run_dir / "experiment_manifest.json", manifest)
            write_json(run_dir / "manifest_validation.json", {"ok": True, "errors": [], "scope": manifest["validation_scope"]})
            records.append({"candidate_id": cid, "candidate_sha256": candidate["candidate_sha256"],
                "screen_flag_counts": counts, "latency": {"turn_count": len(elapsed_times), "median_seconds": statistics.median(elapsed_times)}})
        summary = {"schema_version": "1.0.0", "method": "SPJ-Evolve", "created_at": utc_now(), "records": records, "failed_candidates": [], "plan": plan}
        write_json(output / "sweep_manifest.json", summary)
        write_json(output / "parent_inventory/summary.json", {"status": "UNSCORED_CANDIDATE_INVENTORY", "candidates": [{"candidate_id": record["candidate_id"]} for record in records]})
        print(f"Saved {plan['conversations']} SPJ conversations: {output}")
        return summary
    except Exception as exc:
        write_json(output / "failure.json", {"status": "FAILED_NOT_SEALED", "error_type": type(exc).__name__, "at": utc_now()})
        raise


def main(argv=None):
    run_sweep(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
