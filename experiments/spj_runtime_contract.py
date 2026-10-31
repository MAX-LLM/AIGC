"""Fail-closed, dependency-light contracts for fresh SPJ execution."""

from collections import Counter
from pathlib import Path
import json

from evaluation.common import canonical_sha256, write_json
from prompt_candidate import validate_prompt_candidate


def authenticate(path: Path, inputs: dict) -> str:
    """Bind resumable artifacts to exact input content, not just a path."""
    expected = {"inputs": inputs, "sha256": canonical_sha256(inputs)}
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError(f"Stale or changed SPJ inputs: {path}")
    else:
        write_json(path, expected)
    return expected["sha256"]


def assert_judge_requests(requests: list[dict], *, model=None, effort=None) -> None:
    """Check homogeneous provider requests against the selected judge settings."""
    from evaluation.provider_config import native_config

    if not requests:
        raise ValueError("No judge requests")
    config = native_config("judge", model=model, effort=effort)
    first = requests[0].get("body", requests[0].get("params", {}))
    expected_model = config.model or first.get("model")
    for request in requests:
        body = request.get("body", request.get("params", {}))
        request_effort = body.get("reasoning", body.get("output_config", {})).get("effort")
        if not expected_model or body.get("model") != expected_model or request_effort != config.effort:
            raise ValueError("SPJ judge requests differ from the configured model or effort")
        if config.provider == "openai" and request.get("url") != "/v1/responses":
            raise ValueError("OpenAI SPJ judge requests must use Responses")
        if config.provider == "anthropic" and "params" not in request:
            raise ValueError("Anthropic SPJ judge requests must use Messages parameters")


# Historical import name retained for source compatibility.
assert_luna_requests = assert_judge_requests


def parent_roster(ranking: list, candidates: list, count: int = 2) -> list[str]:
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    return [cid for cid in ranking if by_id[cid]["eligible"] and
            by_id[cid]["minimum_advance_margin_met"]][:count]


def repair_parent_roster(ranking: list, candidates: list, count: int = 2) -> list[str]:
    """Select development repair parents without changing final eligibility.

    Only unresolved LLM nominations may be bypassed for repair. Execution,
    structural and processing-mode failures remain disqualifying.
    """
    by_id = {candidate['candidate_id']: candidate for candidate in candidates}
    allowed = {'unresolved_potential_critical_error_from_llm_judge'}
    result = []
    for cid in ranking:
        item = by_id[cid]
        if not item['minimum_advance_margin_met']:
            continue
        reasons = item.get('ineligibility_reasons', [])
        if item['eligible'] or set(reasons) <= allowed:
            if item.get('batch_only_processing_concordant', True):
                result.append(cid)
    return result[:count]


def validate_run_delivery(run: Path, candidate: dict, cases: dict) -> dict:
    candidate = validate_prompt_candidate(candidate)
    paths = sorted((run / "transcripts/scripted").glob("*/*/state.json"))
    if len(paths) != len(cases) or {p.parent.parent.name for p in paths} != set(cases):
        raise ValueError("Run case roster differs from the locked case bank")
    applied = Counter()
    visits = Counter()
    replacements = Counter()
    for path in paths:
        state = json.loads(path.read_text())
        if state.get("candidate_sha256") != candidate["candidate_sha256"]:
            raise ValueError(f"Session candidate mismatch: {path}")
        conversation = json.loads((path.parent / "conversation.json").read_text())
        patient_turns = [turn["content"] for turn in conversation if turn["role"] == "user"]
        if patient_turns != cases[path.parent.parent.name]["turns"]:
            raise ValueError("Patient turns changed")
        records = state.get("prompt_audit")
        if not isinstance(records, list):
            raise ValueError("Missing effective-prompt audit")
        for record in records:
            if record["candidate_sha256"] != candidate["candidate_sha256"]:
                raise ValueError("Prompt audit candidate mismatch")
            messages = record.get("messages")
            if not messages or canonical_sha256(messages) != record["messages_sha256"]:
                raise ValueError("Missing or unauthenticated effective model messages")
            if canonical_sha256(messages[0]["content"]) != record["system_sha256"]:
                raise ValueError("System prompt hash mismatch")
            component = record["component"]
            visits[component] += 1
            expected_components = set(candidate["node_directives"]) & {"system", component}
            if set(record["applied_directives"]) != expected_components:
                raise ValueError("Prompt audit does not match intended component directives")
            for node in expected_components:
                directive = candidate["node_directives"][node]
                text = messages[0]["content"] if node == "system" else messages[-1]["content"]
                if directive not in text or canonical_sha256(directive) != record["applied_directives"][node]:
                    raise ValueError("Intended directive did not reach the archived model input")
                applied[node] += 1
            if record.get("deterministic_replacement"):
                replacements[record["deterministic_replacement"]] += 1
            if record.get("empty_reply_fallback"):
                replacements["empty_reply_fallback"] += 1
        for event in state.get("notes", {}).get("clinical_safety_events", []):
            if event.get("action") == "safe_fallback":
                replacements["clinical_safety_fallback"] += 1
    if any(applied[node] == 0 for node in candidate["node_directives"]):
        raise ValueError("Candidate contains an intervention not exercised by this case bank")
    return {"status": "VALIDATED", "case_count": len(paths),
            "component_visits": dict(visits), "applied_directives": dict(applied),
            "deterministic_replacements": dict(replacements)}
