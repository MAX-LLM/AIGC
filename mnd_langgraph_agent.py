import json
import os
import re
import uuid
from contextlib import nullcontext
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

from agent_base import (
    SUPPORT_SIGNPOST,
    CounsellingState,
    ExtractedTurnState,
    LLMBackend,
    Stage,
    build_system_message,
    find_gene_in_text,
)
from tools import format_mnd_knowledge, lookup_gene_info
from prompt_candidate import (
    apply_prompt_directive, baseline_candidate, canonical_sha256, validate_prompt_candidate,
)


class AgentGraphState(TypedDict, total=False):
    user_text: str
    reply: str
    route: str
    stage: str
    has_diagnosis_status: bool
    has_confirmed_diagnosis: bool
    has_family_structure: bool
    has_family_history: bool
    has_sharing_preferences: bool
    has_result_delivery_pref: bool
    education_given: bool
    support_given: bool
    essential_info: Dict[str, Any]
    notes: Dict[str, Any]
    summaries: List[str]
    history: List[Dict[str, str]]
    extracted: Dict[str, Any]
    gene_symbol: str
    distress_detected: bool
    summary_reason: str
    background_opt_out: bool


class TurnExtraction(BaseModel):
    diagnosis_status: Literal["unclear", "confirmed", "likely", "pending"] = "unclear"
    family_structure_answered: bool = False
    family_structure_note: str = ""
    family_history_answered: bool = False
    family_history_note: str = ""
    sharing_preferences_answered: bool = False
    sharing_preferences_note: str = ""
    result_delivery_answered: bool = False
    result_delivery_note: str = ""
    testing_context: Literal["pre_test", "post_test", "known_result"] = "pre_test"
    gene_symbol: str = ""
    direct_question: bool = False
    unresolved_question: str = ""
    concern_note: str = ""
    distress_detected: bool = False
    skip_current_question: bool = False
    skip_remaining_background: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


def _extract_turn_state(user_text: str) -> ExtractedTurnState:
    text = user_text.lower()

    diagnosis_status = "unclear"
    if re.search(
        r"\b(waiting|pending|not confirmed|not yet diagnosed|"
        r"tests? pending|awaiting (a )?diagnosis)\b",
        text,
    ):
        diagnosis_status = "pending"
    elif re.search(
        r"\b(i have|i've got|diagnosed with|diagnosis of|confirmed)\b"
        r".{0,35}\b(mnd|als)\b|\bmy (mnd|als)\b",
        text,
    ):
        diagnosis_status = "confirmed"
    elif re.search(r"\b(suspected|likely|probably|being treated as)\b.*\b(mnd|als)\b", text):
        diagnosis_status = "likely"
    has_family_structure = bool(
        re.search(
            r"\b(wife|husband|partner|partener|child|children|son|daughter|sibling|"
            r"brother|sister|mother|father|parent|close family|immediate family)\b",
            text,
        )
    )
    has_family_history = bool(
        re.search(
            r"\b(no family history|family history|runs in (my|the) family|"
            r"relative.*(mnd|als|dementia|mutation|variant)|"
            r"anyone else.*(mnd|als|dementia)|"
            r"no one else.*(mnd|als|dementia)|"
            r"(no one|nobody).*(family|relative).*(mnd|als|dementia)|"
            r"parkinson|frontotemporal|dementia|ftd|alzheimer|mutation|variant)\b",
            text,
        )
    )
    has_sharing_preferences = bool(
        re.search(
            r"\b(share|discuss)\b.{0,35}\b(result|testing|test|information)\b|"
            r"\b(tell|share with)\b.{0,20}\b(family|partner|partener|children|child|"
            r"son|daughter|brother|sister|relative)\b|"
            r"\b(prefer not to know|who should know|family should know)\b",
            text,
        )
    )
    has_result_delivery_pref = bool(
        re.search(
            r"\b(in person|phone|video|letter|email|receive.*result|"
            r"result.*by|how.*result)\b",
            text,
        )
    )

    testing_context = "pre_test"
    if re.search(
        r"\b(already tested|got.*result|result came back|positive|negative|"
        r"vus|variant of uncertain|mutation)\b",
        text,
    ):
        testing_context = "known_result"
    elif re.search(r"\b(test done|tested|sample taken)\b", text):
        testing_context = "post_test"

    gene_symbol = find_gene_in_text(user_text) or ""
    direct_question = "?" in user_text or bool(
        re.search(
            r"^\s*(what|why|how|when|who|should|could|can|does|do|"
            r"will|would|is|are)\b|\b(can|could|would) i\b",
            text,
        )
    )
    distress_detected = bool(
        re.search(
            r"\b(scared|worried|anxious|upset|overwhelmed|distressed|panic|"
            r"hopeless|suicidal|can't cope|cannot cope)\b",
            text,
        )
    )

    concern_note = ""
    if direct_question and re.search(
        r"\b(child|children|son|daughter|family|relative|brother|sister)\b",
        text,
    ):
        concern_note = "concern about family implications"
    elif re.search(r"\b(trial|treatment|therapy|research)\b", text):
        concern_note = "concern about treatment, trials, or research"
    elif distress_detected:
        concern_note = "emotional distress or worry"

    return ExtractedTurnState(
        diagnosis_status=diagnosis_status,
        has_family_structure=has_family_structure,
        family_structure_note=user_text if has_family_structure else "",
        has_family_history=has_family_history,
        family_history_note=user_text if has_family_history else "",
        has_sharing_preferences=has_sharing_preferences,
        sharing_preferences_note=user_text if has_sharing_preferences else "",
        has_result_delivery_pref=has_result_delivery_pref,
        result_delivery_pref_note=user_text if has_result_delivery_pref else "",
        testing_context=testing_context,
        known_gene_or_result=gene_symbol,
        user_is_asking_direct_question=direct_question,
        unresolved_question=user_text.strip()[:220] if direct_question else "",
        concern_note=concern_note,
        distress_detected=distress_detected,
    )


def _declines_background_questions(user_text: str) -> bool:
    text = user_text.lower()
    explicit_skip = bool(
        re.search(
            r"\b(skip (this|these|the questions?|that)|"
            r"rather not (answer|say|share)|"
            r"prefer not to (answer|say|share)|"
            r"do not want to (answer|say|share)|"
            r"don't want to (answer|say|share)|"
            r"no more (personal |background )?questions?)\b",
            text,
        )
    )
    return explicit_skip


def _requests_wrap_up(user_text: str) -> bool:
    text = user_text.lower()
    explicit_wrap = bool(
        re.search(
            r"\b("
            r"wrap up|finish (this|the) (conversation|chat|session)|"
            r"end (this|the) (conversation|chat|session)|"
            r"close (this|the) (conversation|chat|session)|"
            r"summarise and finish|summarize and finish|"
            r"summarise (our|this|the) (conversation|chat|session|discussion)|"
            r"summarize (our|this|the) (conversation|chat|session|discussion)|"
            r"summary of (our|this|the) (conversation|chat|session|discussion)"
            r")\b",
            text,
        )
    )
    return explicit_wrap


def _json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object in extraction response")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Extraction response was not a JSON object")
    return parsed


def _fallback_extraction(
    user_text: str,
    history: List[Dict[str, str]],
    pending_background_item: str,
) -> tuple[ExtractedTurnState, bool]:
    extracted = _apply_conversation_context(
        _extract_turn_state(user_text),
        user_text,
        history,
        pending_background_item,
    )
    return extracted, _declines_background_questions(user_text)


def _apply_conversation_context(
    extracted: ExtractedTurnState,
    user_text: str,
    history: List[Dict[str, str]],
    pending_background_item: str = "",
) -> ExtractedTurnState:
    if not history:
        return extracted

    last_assistant = next(
        (
            message.get("content", "").lower()
            for message in reversed(history)
            if message.get("role") == "assistant"
        ),
        "",
    )
    text = user_text.strip().lower()
    negative_answer = bool(
        re.search(
            r"^\s*(no|nope|none|not really|nobody|no one|i don't|i do not)\b",
            text,
        )
    )
    undecided_answer = bool(
        re.search(r"\b(not sure|haven't decided|have not decided|don't know yet)\b", text)
    )
    answering_diagnosis_question = pending_background_item == "diagnosis_status" or (
        "whether you have been diagnosed with mnd/als" in last_assistant
        or "whether diagnosis is still being confirmed" in last_assistant
    )
    answering_family_structure_question = (
        pending_background_item == "family_structure"
        or
        "tell me a little about your close family" in last_assistant
    )
    answering_family_history_question = pending_background_item == "family_history" or (
        "is there anyone else in the family with mnd/als" in last_assistant
        or "is there no known family history" in last_assistant
    )
    answering_sharing_question = pending_background_item == "sharing_preferences" or (
        "who in your family you might want to tell about testing" in last_assistant
        or "whether anyone might prefer not to know" in last_assistant
    )
    answering_delivery_question = (
        pending_background_item == "result_delivery_preference"
        or
        "how would you prefer to receive and discuss" in last_assistant
    )

    if answering_diagnosis_question and extracted.diagnosis_status == "unclear":
        if re.search(
            r"\b(yes|yeah|yep|have been diagnosed|diagnosis is confirmed|confirmed)\b",
            text,
        ):
            extracted.diagnosis_status = "confirmed"
        elif re.search(
            r"\b(no|not yet|still waiting|being confirmed|pending|suspected)\b",
            text,
        ):
            extracted.diagnosis_status = "pending"

    if answering_family_structure_question and not extracted.has_family_structure:
        if negative_answer:
            extracted.has_family_structure = True
            extracted.family_structure_note = "No close family members mentioned."
        elif undecided_answer:
            extracted.has_family_structure = True
            extracted.family_structure_note = "User preferred not to provide family details."

    if answering_family_history_question and not extracted.has_family_history:
        if negative_answer:
            extracted.has_family_history = True
            extracted.family_history_note = "No known family history of MND/ALS or FTD."
        elif undecided_answer:
            extracted.has_family_history = True
            extracted.family_history_note = "Family history is unknown."

    if answering_sharing_question and not extracted.has_sharing_preferences:
        if negative_answer or undecided_answer:
            extracted.has_sharing_preferences = True
            extracted.sharing_preferences_note = (
                "Has not decided who to tell about testing."
            )
        elif re.search(
            r"\b(my )?(partner|partener|wife|husband|spouse|child|children|son|daughter|"
            r"brother|sister|parent|parents|mother|father|family|relative|relatives)\b",
            text,
        ):
            extracted.has_sharing_preferences = True
            extracted.sharing_preferences_note = user_text.strip()

    if answering_delivery_question and not extracted.has_result_delivery_pref:
        if negative_answer or undecided_answer or "no preference" in text:
            extracted.has_result_delivery_pref = True
            extracted.result_delivery_pref_note = (
                "No result-delivery preference stated."
            )

    return extracted


def _state_from_graph(graph_state: AgentGraphState) -> CounsellingState:
    return CounsellingState(
        stage=Stage(graph_state.get("stage", Stage.BACKGROUND.value)),
        has_diagnosis_status=graph_state.get("has_diagnosis_status", False),
        has_confirmed_diagnosis=graph_state.get("has_confirmed_diagnosis", False),
        has_family_structure=graph_state.get("has_family_structure", False),
        has_family_history=graph_state.get("has_family_history", False),
        has_sharing_preferences=graph_state.get("has_sharing_preferences", False),
        has_result_delivery_pref=graph_state.get("has_result_delivery_pref", False),
        education_given=graph_state.get("education_given", False),
        support_given=graph_state.get("support_given", False),
        essential_info=dict(graph_state.get("essential_info", {})),
        notes=dict(graph_state.get("notes", {})),
        summaries=list(graph_state.get("summaries", [])),
    )


def _state_updates(state: CounsellingState) -> AgentGraphState:
    return {
        "stage": state.stage.value,
        "has_diagnosis_status": state.has_diagnosis_status,
        "has_confirmed_diagnosis": state.has_confirmed_diagnosis,
        "has_family_structure": state.has_family_structure,
        "has_family_history": state.has_family_history,
        "has_sharing_preferences": state.has_sharing_preferences,
        "has_result_delivery_pref": state.has_result_delivery_pref,
        "education_given": state.education_given,
        "support_given": state.support_given,
        "essential_info": state.essential_info,
        "notes": state.notes,
        "summaries": state.summaries,
    }


class LangGraphMNDAgent:
    """Stage-based MND pre-test counselling workflow implemented with LangGraph."""

    def __init__(self, backend: LLMBackend, prompt_candidate: Optional[Dict[str, Any]] = None):
        self.backend = backend
        self.prompt_candidate = validate_prompt_candidate(
            baseline_candidate() if prompt_candidate is None else prompt_candidate
        )
        self.prompt_audit: List[Dict[str, Any]] = []
        self.state = CounsellingState()
        self.history: List[Dict[str, str]] = []
        self.thread_id = uuid.uuid4().hex
        self.checkpointer = InMemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentGraphState)
        workflow.add_node("extract_information", self._extract_information_node)
        workflow.add_node("gene_information_tool", self._gene_tool_node)
        workflow.add_node("background_question", self._background_node)
        workflow.add_node("education", self._education_node)
        workflow.add_node("answer_questions", self._questions_node)
        workflow.add_node("wrap_up", self._wrap_up_node)
        workflow.add_node("support_signposting", self._support_node)
        workflow.add_node("summarize_context", self._summary_node)

        workflow.add_edge(START, "extract_information")
        workflow.add_conditional_edges(
            "extract_information",
            self._route_after_extraction,
            {
                "gene": "gene_information_tool",
                "background": "background_question",
                "education": "education",
                "questions": "answer_questions",
                "wrap_up": "wrap_up",
            },
        )
        for node in (
            "gene_information_tool",
            "background_question",
            "education",
            "answer_questions",
            "wrap_up",
        ):
            workflow.add_edge(node, "support_signposting")
        workflow.add_conditional_edges(
            "support_signposting",
            self._route_after_support,
            {"summarize": "summarize_context", "finish": END},
        )
        workflow.add_edge("summarize_context", END)
        return workflow.compile(checkpointer=self.checkpointer)

    def _initial_graph_state(self, user_text: str) -> AgentGraphState:
        return {
            "user_text": user_text,
            "reply": "",
            "route": "",
            "stage": self.state.stage.value,
            "has_diagnosis_status": self.state.has_diagnosis_status,
            "has_confirmed_diagnosis": self.state.has_confirmed_diagnosis,
            "has_family_structure": self.state.has_family_structure,
            "has_family_history": self.state.has_family_history,
            "has_sharing_preferences": self.state.has_sharing_preferences,
            "has_result_delivery_pref": self.state.has_result_delivery_pref,
            "education_given": self.state.education_given,
            "support_given": self.state.support_given,
            "essential_info": dict(self.state.essential_info),
            "notes": dict(self.state.notes),
            "summaries": list(self.state.summaries),
            "history": list(self.history),
            "extracted": {},
            "gene_symbol": "",
            "distress_detected": False,
            "summary_reason": "",
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "thread_id": self.thread_id,
            "state": _state_updates(self.state),
            "history": list(self.history),
            "prompt_candidate": self.prompt_candidate,
            "prompt_audit": list(self.prompt_audit),
        }

    def load_dict(self, payload: Dict[str, Any]) -> None:
        candidate = validate_prompt_candidate(payload.get("prompt_candidate", baseline_candidate()))
        if (self.prompt_candidate["node_directives"]
                and candidate["candidate_sha256"] != self.prompt_candidate["candidate_sha256"]):
            raise ValueError("Persisted session candidate differs from the configured candidate")
        self.prompt_candidate = candidate
        audit = payload.get("prompt_audit", [])
        if not isinstance(audit, list) or any(not isinstance(record, dict) for record in audit):
            raise ValueError("Persisted prompt audit must be a list of objects")
        self.prompt_audit = list(audit)
        state_payload = payload.get("state", {})
        if isinstance(state_payload, dict):
            self.state = _state_from_graph(state_payload)

        history = payload.get("history", [])
        if isinstance(history, list):
            self.history = [
                {
                    "role": str(message.get("role", "")),
                    "content": str(message.get("content", "")),
                }
                for message in history
                if isinstance(message, dict)
            ]

        thread_id = str(payload.get("thread_id", "")).strip()
        self.thread_id = thread_id or uuid.uuid4().hex

    def _extract_with_llm(
        self,
        user_text: str,
        history: List[Dict[str, str]],
        pending_background_item: str,
    ) -> tuple[ExtractedTurnState, bool, Dict[str, Any]]:
        last_assistant = next(
            (
                message.get("content", "")
                for message in reversed(history)
                if message.get("role") == "assistant"
            ),
            "",
        )
        prompt = f"""
Extract structured conversation state from one patient turn in an MND genetic
counselling conversation. Interpret ordinary spelling mistakes, missing first
letters, informal language, and short answers using the pending question.

Do not infer facts the patient did not state. "Yes" only answers the pending
question. If the patient says they have been diagnosed, "diagnosed with MND",
"confirmed diagnosis", or a clear misspelling of those phrases, set
diagnosis_status to "confirmed". If diagnosis is still being investigated, use
"pending".

For each background field, set *_answered true only when the patient supplied
an answer. A refusal or skip is not an answer. If the patient asks to skip the
current personal question, set skip_current_question true. If they say they are
undecided about testing and want to skip, decline the intake, or ask for general
information instead, set skip_remaining_background true so no more intake
questions are asked.

Return exactly one JSON object with these keys:
{{
  "diagnosis_status": "unclear|confirmed|likely|pending",
  "family_structure_answered": false,
  "family_structure_note": "",
  "family_history_answered": false,
  "family_history_note": "",
  "sharing_preferences_answered": false,
  "sharing_preferences_note": "",
  "result_delivery_answered": false,
  "result_delivery_note": "",
  "testing_context": "pre_test|post_test|known_result",
  "gene_symbol": "",
  "direct_question": false,
  "unresolved_question": "",
  "concern_note": "",
  "distress_detected": false,
  "skip_current_question": false,
  "skip_remaining_background": false,
  "confidence": 0.0
}}

Pending background item: {pending_background_item or "none"}
Previous assistant message: {last_assistant[-1200:]}
Patient message: {user_text}
""".strip()
        try:
            generation_context = (
                self.backend.generation_options(320)
                if hasattr(self.backend, "generation_options")
                else nullcontext()
            )
            with generation_context:
                raw = self.backend.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You are a careful structured-data extractor. "
                                "Output valid JSON only."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ]
                )
            parsed = TurnExtraction.model_validate(_json_object(raw))
            deterministic_extracted = _apply_conversation_context(
                _extract_turn_state(user_text),
                user_text,
                history,
                pending_background_item,
            )
            gene_symbol = find_gene_in_text(user_text) or ""
            direct_question = (
                parsed.direct_question
                or deterministic_extracted.user_is_asking_direct_question
            )
            extracted = ExtractedTurnState(
                diagnosis_status=parsed.diagnosis_status,
                has_family_structure=(
                    parsed.family_structure_answered
                    or deterministic_extracted.has_family_structure
                ),
                family_structure_note=(
                    parsed.family_structure_note.strip()
                    or deterministic_extracted.family_structure_note
                ),
                has_family_history=(
                    parsed.family_history_answered
                    or deterministic_extracted.has_family_history
                ),
                family_history_note=(
                    parsed.family_history_note.strip()
                    or deterministic_extracted.family_history_note
                ),
                has_sharing_preferences=(
                    parsed.sharing_preferences_answered
                    or deterministic_extracted.has_sharing_preferences
                ),
                sharing_preferences_note=(
                    parsed.sharing_preferences_note.strip()
                    or deterministic_extracted.sharing_preferences_note
                ),
                has_result_delivery_pref=(
                    parsed.result_delivery_answered
                    or deterministic_extracted.has_result_delivery_pref
                ),
                result_delivery_pref_note=(
                    parsed.result_delivery_note.strip()
                    or deterministic_extracted.result_delivery_pref_note
                ),
                testing_context=parsed.testing_context,
                known_gene_or_result=gene_symbol,
                user_is_asking_direct_question=direct_question,
                unresolved_question=(
                    parsed.unresolved_question.strip()[:220]
                    if parsed.direct_question
                    else deterministic_extracted.unresolved_question
                    if direct_question
                    else ""
                ),
                concern_note=(
                    parsed.concern_note.strip()
                    or deterministic_extracted.concern_note
                ),
                distress_detected=(
                    parsed.distress_detected
                    or deterministic_extracted.distress_detected
                ),
            )
            opt_out = (
                parsed.skip_remaining_background
                or parsed.skip_current_question
            )
            metadata = {
                "method": "llm",
                "confidence": parsed.confidence,
                "deterministic_direct_question": (
                    deterministic_extracted.user_is_asking_direct_question
                ),
                "skip_current_question": parsed.skip_current_question,
                "skip_remaining_background": parsed.skip_remaining_background,
            }
            return extracted, opt_out, metadata
        except Exception as exc:
            extracted, opt_out = _fallback_extraction(
                user_text,
                history,
                pending_background_item,
            )
            return extracted, opt_out, {
                "method": "deterministic_fallback",
                "error": type(exc).__name__,
                "skip_current_question": False,
                "skip_remaining_background": opt_out,
            }

    def _extract_information_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        user_text = graph_state["user_text"]
        counselling_state = _state_from_graph(graph_state)
        pending_background_item = str(
            counselling_state.notes.pop("pending_background_item", "")
        )
        extracted, background_opt_out, extraction_metadata = self._extract_with_llm(
            user_text,
            graph_state.get("history", []),
            pending_background_item,
        )
        background_opt_out = (
            counselling_state.stage == Stage.BACKGROUND and background_opt_out
        )
        counselling_state.notes["last_turn_extraction"] = extraction_metadata

        if extracted.diagnosis_status in {"confirmed", "likely"}:
            counselling_state.has_diagnosis_status = True
            counselling_state.has_confirmed_diagnosis = True
            counselling_state.record_note("diagnosis_status", extracted.diagnosis_status)
        elif extracted.diagnosis_status == "pending":
            counselling_state.has_diagnosis_status = True
            counselling_state.record_note("diagnosis_status", "pending confirmation")

        if extracted.has_family_structure:
            counselling_state.has_family_structure = True
            counselling_state.record_note(
                "family_structure",
                extracted.family_structure_note,
                overwrite=counselling_state.essential_info.get("family_structure")
                in {
                    "No close family members mentioned.",
                    "User preferred not to provide family details.",
                },
            )
        if extracted.has_family_history:
            counselling_state.has_family_history = True
            counselling_state.record_note(
                "family_history",
                extracted.family_history_note,
                overwrite=counselling_state.essential_info.get("family_history")
                in {
                    "No known family history of MND/ALS or FTD.",
                    "Family history is unknown.",
                }
                and not extracted.family_history_note.startswith("No known"),
            )
        if extracted.has_sharing_preferences:
            counselling_state.has_sharing_preferences = True
            counselling_state.record_note(
                "sharing_preferences",
                extracted.sharing_preferences_note,
                overwrite=counselling_state.essential_info.get("sharing_preferences")
                == "Has not decided who to tell about testing.",
            )
        if extracted.has_result_delivery_pref:
            counselling_state.has_result_delivery_pref = True
            counselling_state.record_note(
                "result_delivery_preference",
                extracted.result_delivery_pref_note,
                overwrite=counselling_state.essential_info.get(
                    "result_delivery_preference"
                )
                == "No result-delivery preference stated.",
            )

        counselling_state.record_note(
            "testing_context", extracted.testing_context, overwrite=True
        )
        if extracted.known_gene_or_result:
            counselling_state.record_note(
                "known_gene_or_result",
                extracted.known_gene_or_result,
                overwrite=True,
            )
        if extracted.unresolved_question:
            counselling_state.notes["last_user_question"] = extracted.unresolved_question
        if extracted.concern_note:
            concerns = counselling_state.essential_info.setdefault("main_concerns", [])
            if extracted.concern_note not in concerns:
                concerns.append(extracted.concern_note)

        if background_opt_out:
            missing_items = counselling_state.missing_background_items()
            deferred_items = missing_items
            existing_deferred = list(
                counselling_state.notes.get("deferred_background_items", [])
            )
            counselling_state.notes["deferred_background_items"] = list(
                dict.fromkeys(existing_deferred + deferred_items)
            )
            counselling_state.notes["background_questions_declined"] = True

        previous_stage = counselling_state.stage
        counselling_state.maybe_advance_stage()
        summary_reason = ""
        if previous_stage != counselling_state.stage:
            summary_reason = (
                f"transition from {previous_stage.value} "
                f"to {counselling_state.stage.value}"
            )

        if _requests_wrap_up(user_text):
            counselling_state.stage = Stage.WRAP_UP
            summary_reason = "user requested wrap-up"

        history = list(graph_state.get("history", []))
        history.append({"role": "user", "content": user_text})
        updates = _state_updates(counselling_state)
        updates.update(
            {
                "history": history,
                "extracted": vars(extracted),
                "gene_symbol": extracted.known_gene_or_result,
                "distress_detected": extracted.distress_detected,
                "summary_reason": summary_reason,
                "background_opt_out": background_opt_out,
            }
        )
        return updates

    @staticmethod
    def _route_after_extraction(graph_state: AgentGraphState) -> str:
        if graph_state.get("gene_symbol"):
            return "gene"
        extracted = graph_state.get("extracted", {})
        if extracted.get("user_is_asking_direct_question"):
            return "questions"
        stage = graph_state.get("stage", Stage.BACKGROUND.value)
        if stage == Stage.BACKGROUND.value:
            return "background"
        if stage == Stage.EDUCATION.value and not graph_state.get("education_given", False):
            return "education"
        if stage == Stage.WRAP_UP.value:
            return "wrap_up"
        return "questions"

    @staticmethod
    def _next_background_item(state: CounsellingState) -> str:
        missing = state.missing_background_items()
        return missing[0] if missing else ""

    @staticmethod
    def _next_background_question(state: CounsellingState) -> str:
        next_item = LangGraphMNDAgent._next_background_item(state)
        if next_item == "diagnosis_status":
            return (
                "To understand the context, could I check whether you have been diagnosed "
                "with MND/ALS, or whether diagnosis is still being confirmed?"
            )
        if next_item == "family_structure":
            return (
                "Could you tell me a little about your close family, for example whether "
                "you have a partner, children, siblings, or other relatives you are thinking about?"
            )
        if next_item == "family_history":
            return (
                "Is there anyone else in the family with MND/ALS, frontotemporal dementia, "
                "or a known genetic result, or is there no known family history?"
            )
        if next_item == "sharing_preferences":
            return (
                "Have you thought about who in your family you might want to tell about "
                "testing, and whether anyone might prefer not to know?"
            )
        if next_item == "result_delivery_preference":
            return (
                "If a result becomes available, how would you prefer to receive and discuss "
                "it, for example in person, by phone or video, or by letter?"
            )
        return "Thank you. I have the main background I need."

    def _messages_for_reply(
        self,
        graph_state: AgentGraphState,
        prompt: str = "",
        knowledge_query: str = "",
        prompt_node: str = "questions",
    ) -> List[Dict[str, str]]:
        state = _state_from_graph(graph_state)
        prompt = apply_prompt_directive(prompt, prompt_node, self.prompt_candidate)
        messages = [build_system_message(state, self.prompt_candidate)]
        if knowledge_query:
            knowledge, _ = format_mnd_knowledge(knowledge_query)
            if knowledge:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Use the curated references below to ground factual statements. "
                            "They are general information, not patient-specific interpretation. "
                            "Do not invent details beyond them. Mention a source and URL when "
                            "the user asks for references.\n\n"
                            f"{knowledge}"
                        ),
                    }
                )
        messages.extend(graph_state.get("history", [])[-8:])
        if prompt:
            messages.append({"role": "user", "content": prompt})
        applied = {}
        for component, content in (("system", messages[0]["content"]), (prompt_node, prompt)):
            directive = self.prompt_candidate["node_directives"].get(component)
            if directive:
                if directive not in content:
                    raise RuntimeError(f"Candidate directive {component} did not reach model input")
                applied[component] = canonical_sha256(directive)
        record = {
            "candidate_id": self.prompt_candidate["candidate_id"],
            "candidate_sha256": self.prompt_candidate["candidate_sha256"],
            "component": prompt_node,
            "turn": sum(message.get("role") == "user" for message in graph_state.get("history", [])),
            "messages_sha256": canonical_sha256(messages),
            "system_sha256": canonical_sha256(messages[0]["content"]),
            "stage_sha256": canonical_sha256(prompt),
            "applied_directives": applied,
            "safety_controls_sha256": canonical_sha256([]),
        }
        if (os.environ.get("MND_RECORD_EFFECTIVE_PROMPTS") == "1"
                and os.environ.get("MND_ALLOW_PROMPT_CANDIDATES") == "1"):
            record["messages"] = messages
        self.prompt_audit.append(record)
        return messages

    def _generate_reply(
        self,
        messages: List[Dict[str, str]],
        fallback: str,
    ) -> str:
        reply = self.backend.chat(messages).strip()
        if self.prompt_audit:
            self.prompt_audit[-1]["raw_reply_sha256"] = canonical_sha256(reply)
        if reply:
            return reply

        retry_messages = list(messages)
        retry_messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous generation was empty. Give a complete, concise, "
                    "patient-facing answer to the person's latest question. Do not mention "
                    "this retry or invent any personal facts."
                ),
            }
        )
        reply = self.backend.chat(retry_messages).strip()
        if self.prompt_audit:
            self.prompt_audit[-1]["retry_messages_sha256"] = canonical_sha256(retry_messages)
            self.prompt_audit[-1]["empty_reply_fallback"] = not bool(reply)
            if "messages" in self.prompt_audit[-1]:
                self.prompt_audit[-1]["retry_messages"] = retry_messages
        return reply or fallback

    @staticmethod
    def _knowledge_notes(
        graph_state: AgentGraphState, query: str
    ) -> Dict[str, Any]:
        _, references = format_mnd_knowledge(query)
        notes = dict(graph_state.get("notes", {}))
        notes["last_knowledge_sources"] = [
            {
                "id": reference["id"],
                "title": reference["title"],
                "url": reference["url"],
            }
            for reference in references
        ]
        return notes

    def _background_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        state = _state_from_graph(graph_state)
        extracted = graph_state.get("extracted", {})
        background_item = self._next_background_item(state)
        required_question = self._next_background_question(state)
        notes = dict(graph_state.get("notes", {}))
        notes["pending_background_item"] = background_item

        if extracted.get("user_is_asking_direct_question"):
            knowledge_query = graph_state["user_text"]
            notes.update(self._knowledge_notes(graph_state, knowledge_query))
            instruction = (
                "First answer the user's question directly in 1-2 short paragraphs. "
                "Then continue the conversation with the required follow-up question."
            )
        else:
            knowledge_query = ""
            instruction = (
                "Briefly acknowledge what the user said without repeating it mechanically. "
                "Then ask the required follow-up question."
            )

        prompt = (
            "Write the next patient-facing counselling reply in a warm, natural tone. "
            f"{instruction} The required information is '{background_item}'. "
            f"The intended follow-up is: {required_question}\n\n"
            "Ask exactly one question in the whole reply, covering the same information as "
            "the intended follow-up. You may phrase it naturally. Do not skip it, add another "
            "intake question, invent facts, or pressure the user. Keep the reply concise."
        )
        reply = self.backend.chat(
            self._messages_for_reply(
                graph_state,
                prompt,
                knowledge_query=knowledge_query,
                prompt_node="background",
            )
        ).strip()
        self.prompt_audit[-1]["raw_reply_sha256"] = canonical_sha256(reply)
        if not reply or reply.count("?") != 1:
            self.prompt_audit[-1]["deterministic_replacement"] = "background_question_count_or_empty_reply"
            reply = required_question
        return {"reply": reply, "notes": notes}

    def _education_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        knowledge_query = (
            "MND ALS genetic testing pre-test counselling positive negative uncertain "
            "results family implications"
        )
        prompt = (
            "Give the Stage 2 education explanation now. Start with one brief bridging "
            "sentence such as 'Thanks, that gives me the main context; I can now explain "
            "what testing can and cannot tell you.' Do not summarise the person's "
            "preferences as the opening. Explain, in plain language, what "
            "pre-test MND/ALS genetic testing can and cannot tell someone; what positive, "
            "negative, and uncertain results can mean; what it may mean for relatives; and "
            "that their clinical genetics/neurology team should guide final decisions. "
            "Use the recorded essential information. Follow the result interpretation "
            "guardrails in the system message. Keep it to 3-4 short paragraphs and end by "
            "inviting questions. If the person declined or deferred background questions, "
            "briefly confirm that this is fine and do not imply they have decided to test."
        )
        reply = self._generate_reply(
            self._messages_for_reply(
                graph_state,
                prompt,
                knowledge_query=knowledge_query,
                prompt_node="education",
            ),
            fallback=(
                "I could not produce a complete explanation just now. Your neurology or "
                "genetics team can explain what testing can and cannot show, the possible "
                "results, and what each result could mean for you and your relatives. "
                "What part would you most like to discuss first?"
            ),
        )
        return {
            "reply": reply,
            "education_given": True,
            "stage": Stage.QUESTIONS.value,
            "summary_reason": "transition from education to questions",
            "notes": self._knowledge_notes(graph_state, knowledge_query),
        }

    def _gene_tool_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        gene_symbol = graph_state["gene_symbol"]
        gene_info = lookup_gene_info(gene_symbol)
        notes = dict(graph_state.get("notes", {}))
        notes["last_gene_lookup"] = {
            "gene": gene_symbol,
            "tool_result": gene_info,
        }
        prompt = (
            f"The user asked about {gene_symbol} in the context of MND/ALS genetic testing.\n\n"
            f"Verified tool result:\n{gene_info}\n\n"
            "Use this result to answer in plain language. Explain that exact meaning depends "
            "on the specific variant and clinical genetics interpretation. Do not mention API "
            "calls or technical tool details. Keep it brief and supportive."
        )
        return {
            "reply": self._generate_reply(
                self._messages_for_reply(graph_state, prompt),
                fallback=(
                    f"I could not produce a complete answer about {gene_symbol} just now. "
                    "The meaning depends on the exact variant and should be interpreted by "
                    "your clinical genetics team."
                ),
            ),
            "notes": notes,
        }

    def _questions_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        knowledge_query = graph_state["user_text"]
        state = _state_from_graph(graph_state)
        background_context = ""
        if state.stage == Stage.BACKGROUND:
            background_context = (
                "The conversation is still in background collection, but the user has "
                "asked a direct question. Answer it now. Do not ask the next intake "
                "question in this reply. At most, end by saying you can return to the "
                "background questions when they are ready."
            )
        prompt = (
            f"Latest patient question: {graph_state['user_text']}\n\n"
            "Answer the person's latest question directly in 1-3 short paragraphs. "
            "Do not summarise the whole conversation, do not wrap up, and do not "
            "redirect to the clinical team unless it is needed for the specific "
            "answer. Use the recorded context only to tailor the answer.\n\n"
            "If asked about what testing involves, say it usually involves consent "
            "and a blood sample or sometimes saliva/buccal sample, with local practice "
            "varying. If asked about timing, say results often take several weeks to a "
            "few months depending on the laboratory, panel, and local pathway. If asked "
            "about likelihood of finding a pathogenic variant without family history, "
            "give a cautious approximate answer: a minority of apparently sporadic ALS "
            "cases have an identifiable pathogenic variant, often roughly around 5-10% "
            "depending on the tested genes, ancestry, and criteria; the person's own "
            "team should confirm local expectations. If asked about SOD1 treatment, "
            "say directly that SOD1-targeted therapy exists in some settings, but "
            "eligibility and access depend on the exact SOD1 variant/result, approval "
            "and funding in the country, and the neurology team.\n\n"
            "When explaining test outcomes, include positive/pathogenic or likely "
            "pathogenic, negative, and variant of uncertain significance (VUS). Explain "
            "that a VUS is not a confirmed cause and should not normally guide "
            "predictive testing in relatives.\n\n"
            "When relevant to the question, include these standard counselling points: "
            "apparently sporadic ALS/MND can still have a genetic contribution; if a "
            "pathogenic autosomal dominant ALS/FTD-related variant is found, each child "
            "may have a 1 in 2 chance of inheriting the variant, but this is not the "
            "same as a 1 in 2 chance of developing disease because penetrance varies; "
            "the affected person is usually tested first; predictive testing for "
            "unaffected adult relatives usually follows specialist counselling after a "
            "familial pathogenic variant is identified; predictive testing for "
            "adult-onset ALS/FTD conditions is generally not recommended for minors; "
            "partners are usually not biologically at risk from the person's result.\n\n"
            f"{background_context}"
        )
        messages = self._messages_for_reply(
            graph_state,
            prompt=prompt,
            knowledge_query=knowledge_query,
        )
        return {
            "reply": self._generate_reply(
                messages,
                fallback=(
                    "I could not produce a complete answer just now. Please try rephrasing "
                    "the question, or take it to your neurology or clinical genetics team "
                    "for advice specific to you."
                ),
            ),
            "notes": self._knowledge_notes(graph_state, knowledge_query),
        }

    def _wrap_up_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        prompt = (
            "Wrap up this pre-test counselling conversation. Summarise the person's main "
            "questions and preferences, state the important next steps, and encourage them "
            "to discuss testing with their neurology or clinical genetics team. Do not add "
            "facts. Keep it concise."
        )
        return {
            "reply": self._generate_reply(
                self._messages_for_reply(graph_state, prompt, prompt_node="wrap_up"),
                fallback=(
                    "The next step is to discuss your questions, preferences, and whether "
                    "testing feels right for you with your neurology or clinical genetics "
                    "team. They can guide consent, testing, and how results would be shared."
                ),
            )
        }

    def _support_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        reply = graph_state.get("reply", "").strip()
        support_given = graph_state.get("support_given", False)
        if graph_state.get("distress_detected") or not support_given:
            if SUPPORT_SIGNPOST not in reply:
                reply = f"{reply}\n\n{SUPPORT_SIGNPOST}".strip()
            support_given = True

        history = list(graph_state.get("history", []))
        history.append({"role": "assistant", "content": reply})
        summary_reason = graph_state.get("summary_reason", "")
        if len(history) > 14 and not summary_reason:
            summary_reason = "periodic context compaction"
        return {
            "reply": reply,
            "support_given": support_given,
            "history": history,
            "summary_reason": summary_reason,
        }

    @staticmethod
    def _route_after_support(graph_state: AgentGraphState) -> str:
        if graph_state.get("summary_reason") and len(graph_state.get("history", [])) >= 4:
            return "summarize"
        return "finish"

    def _summary_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        history = graph_state.get("history", [])
        transcript = "\n".join(
            f"{message.get('role', '').upper()}: {message.get('content', '')}"
            for message in history
        )
        reason = graph_state.get("summary_reason", "context compaction")
        prompt = (
            "Summarise this MND genetics counselling conversation for the next stage. "
            "Keep only clinically relevant facts the user gave, questions asked, emotional "
            "concerns, family context, gene/result mentions, and preferences. Do not add "
            "facts. Use 5-8 concise bullet points.\n\n"
            f"Reason: {reason}\n\nTranscript:\n{transcript}"
        )
        try:
            summary = self.backend.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You compact conversation history for a counselling agent. "
                            "Be factual and concise."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]
            ).strip()
        except Exception:
            summary = (
                f"Compact summary after {reason}: "
                f"{json.dumps(graph_state.get('essential_info', {}), ensure_ascii=False)}"
            )

        summaries = list(graph_state.get("summaries", []))
        if summary:
            summaries.append(summary)
        return {
            "summaries": summaries,
            "history": history[-4:],
            "summary_reason": "",
        }

    def ask(self, user_text: str) -> str:
        result = self.graph.invoke(
            self._initial_graph_state(user_text),
            config={"configurable": {"thread_id": self.thread_id}},
        )
        self.state = _state_from_graph(result)
        self.history = list(result.get("history", []))
        return str(result.get("reply", "")).strip()
