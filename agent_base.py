"""Shared model backend, counselling state, prompts, and legacy agent support."""

import json
import os
import queue
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from tools import lookup_gene_info, normalize_gene_symbol
from prompt_candidate import apply_prompt_directive


def resolve_model_path(model: str, base_path: str) -> Path:
    requested_path = Path(model).expanduser()
    if requested_path.is_absolute() or requested_path.exists():
        llm_path = requested_path
    else:
        llm_path = Path(base_path).expanduser() / "models" / model

    if not llm_path.exists():
        raise FileNotFoundError(f"Model directory does not exist: {llm_path}")
    return llm_path


KNOWN_MND_GENES = {
    "C9ORF72",
    "SOD1",
    "TARDBP",
    "FUS",
    "TBK1",
    "OPTN",
    "VCP",
}

GENE_PATTERN = re.compile(
    r"\b(?:gene\s+)?(C9orf72|SOD1|TARDBP|FUS|TBK1|OPTN|VCP|[A-Z]{2,6}[0-9]{0,3})\b",
    re.IGNORECASE,
)

SUPPORT_SIGNPOST = (
    "If this conversation brings up worries or feels upsetting, it is okay to pause. "
    "For support, you can contact your neurology/genetics team, the MND Association "
    "helpline on 0808 802 6262, or Samaritans on 116 123 at any time."
)


def find_gene_in_text(text: str) -> Optional[str]:
    """
    Detect a likely MND/ALS gene symbol in the latest user message.

    This intentionally favours common MND genes to avoid treating ordinary
    capitalised words as genes. Unknown gene-like tokens only match when the
    user writes "gene X".
    """
    for match in GENE_PATTERN.finditer(text):
        symbol = normalize_gene_symbol(match.group(1))
        if symbol in KNOWN_MND_GENES:
            return symbol
        if match.group(0).lower().startswith("gene "):
            return symbol
    return None


class LLMBackend:
    def __init__(self, args):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        self._torch = torch
        llm_path = resolve_model_path(args.model, args.base_path)

        try:
            self.processor = AutoProcessor.from_pretrained(
                str(llm_path),
                local_files_only=True,
            )
        except (KeyError, OSError, ValueError):
            self.processor = AutoTokenizer.from_pretrained(
                str(llm_path),
                local_files_only=True,
            )

        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(llm_path),
            device_map="auto",
            dtype="auto",
            local_files_only=True,
        )
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self.max_new_tokens = args.max_new_tokens
        self.model_path = str(llm_path)
        self.backend_name = "transformers"
        self.deterministic = True
        self.max_concurrent_generations = max(
            1,
            int(
                os.environ.get(
                    "MND_MAX_CONCURRENT_GENERATIONS",
                    "1",
                )
            ),
        )
        self._generation_slots = threading.BoundedSemaphore(
            self.max_concurrent_generations
        )
        self._request_options = threading.local()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    @contextmanager
    def generation_options(self, max_new_tokens: Optional[int] = None):
        previous = getattr(self._request_options, "max_new_tokens", None)
        self._request_options.max_new_tokens = max_new_tokens
        try:
            yield
        finally:
            self._request_options.max_new_tokens = previous

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        messages: list of {"role": "system"|"user"|"assistant", "content": "..."}
        Return only the newly generated assistant message.
        """
        torch = self._torch
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            prompt = self.processor.apply_chat_template(
                messages,
                enable_thinking=False,
                **template_kwargs,
            )
        except TypeError:
            prompt = self.processor.apply_chat_template(messages, **template_kwargs)

        model_inputs = self.processor(text=prompt, return_tensors="pt")
        model_inputs = {
            key: value.to(self.model.device)
            for key, value in model_inputs.items()
            if hasattr(value, "to")
        }

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", torch.ones_like(input_ids))
        input_len = input_ids.shape[-1]
        requested_max_tokens = getattr(
            self._request_options,
            "max_new_tokens",
            None,
        )
        max_new_tokens = requested_max_tokens or self.max_new_tokens

        with self._generation_slots:
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        generated_ids = output[0, input_len:]
        text = self.processor.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        for prefix in ["assistant", "Assistant"]:
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip(" :\n")

        return text


class VLLMBackend:
    def __init__(self, args):
        from transformers import AutoProcessor, AutoTokenizer

        os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
        os.environ.setdefault(
            "VLLM_CACHE_ROOT",
            str(Path(__file__).resolve().parent / ".runtime" / "vllm"),
        )

        from vllm import LLM, SamplingParams

        llm_path = resolve_model_path(args.model, args.base_path)

        try:
            self.processor = AutoProcessor.from_pretrained(
                str(llm_path),
                local_files_only=True,
            )
        except (KeyError, OSError, ValueError):
            self.processor = AutoTokenizer.from_pretrained(
                str(llm_path),
                local_files_only=True,
            )

        self.SamplingParams = SamplingParams
        self.vllm_max_num_seqs = getattr(args, "vllm_max_num_seqs", 8)
        self.model = LLM(
            model=str(llm_path),
            served_model_name=getattr(args, "served_model_name", None)
            or Path(llm_path).name,
            dtype=getattr(args, "vllm_dtype", "bfloat16"),
            max_model_len=getattr(args, "vllm_max_model_len", 4096),
            max_num_seqs=self.vllm_max_num_seqs,
            gpu_memory_utilization=getattr(args, "vllm_gpu_memory_utilization", 0.55),
            trust_remote_code=True,
        )
        self.max_new_tokens = args.max_new_tokens
        self.model_path = str(llm_path)
        self.backend_name = "vllm"
        self.deterministic = True
        self.max_concurrent_generations = max(
            1,
            int(
                os.environ.get(
                    "MND_MAX_CONCURRENT_GENERATIONS",
                    str(getattr(args, "vllm_max_num_seqs", 8)),
                )
            ),
        )
        self._generation_slots = threading.BoundedSemaphore(
            self.max_concurrent_generations
        )
        self._request_options = threading.local()
        self._batch_timeout_s = float(os.environ.get("MND_VLLM_BATCH_TIMEOUT", "0.03"))
        self._generation_queue: queue.Queue = queue.Queue()
        self._generation_worker = threading.Thread(
            target=self._run_generation_worker,
            name="mnd-vllm-generation-worker",
            daemon=True,
        )
        self._generation_worker.start()

    @contextmanager
    def generation_options(self, max_new_tokens: Optional[int] = None):
        previous = getattr(self._request_options, "max_new_tokens", None)
        self._request_options.max_new_tokens = max_new_tokens
        try:
            yield
        finally:
            self._request_options.max_new_tokens = previous

    def chat(self, messages: List[Dict[str, str]]) -> str:
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            prompt = self.processor.apply_chat_template(
                messages,
                enable_thinking=False,
                **template_kwargs,
            )
        except TypeError:
            prompt = self.processor.apply_chat_template(messages, **template_kwargs)

        requested_max_tokens = getattr(
            self._request_options,
            "max_new_tokens",
            None,
        )
        max_new_tokens = requested_max_tokens or self.max_new_tokens
        params = self.SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
        )
        request = {
            "prompt": prompt,
            "params": params,
            "event": threading.Event(),
            "text": "",
            "error": None,
        }

        with self._generation_slots:
            self._generation_queue.put(request)
            request["event"].wait()

        if request["error"] is not None:
            raise request["error"]

        text = str(request["text"]).strip()
        for prefix in ["assistant", "Assistant"]:
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip(" :\n")
        return text

    def _run_generation_worker(self) -> None:
        while True:
            first = self._generation_queue.get()
            batch = [first]
            deadline = time.monotonic() + self._batch_timeout_s

            while len(batch) < self.vllm_max_num_seqs:
                timeout = max(0.0, deadline - time.monotonic())
                if timeout <= 0:
                    break
                try:
                    batch.append(self._generation_queue.get(timeout=timeout))
                except queue.Empty:
                    break

            try:
                prompts = [item["prompt"] for item in batch]
                params = [item["params"] for item in batch]
                outputs = self.model.generate(prompts, params, use_tqdm=False)
                for item, output in zip(batch, outputs):
                    item["text"] = output.outputs[0].text
            except Exception as exc:  # pragma: no cover - defensive service path
                for item in batch:
                    item["error"] = exc
            finally:
                for item in batch:
                    item["event"].set()
                    self._generation_queue.task_done()


class Stage(str, Enum):
    BACKGROUND = "background"
    EDUCATION = "education"
    QUESTIONS = "questions"
    WRAP_UP = "wrap_up"


@dataclass
class CounsellingState:
    stage: Stage = Stage.BACKGROUND

    has_diagnosis_status: bool = False
    has_confirmed_diagnosis: bool = False
    has_family_structure: bool = False
    has_family_history: bool = False
    has_sharing_preferences: bool = False
    has_result_delivery_pref: bool = False
    education_given: bool = False
    support_given: bool = False

    essential_info: Dict[str, Any] = field(
        default_factory=lambda: {
            "diagnosis_status": "unclear",
            "testing_context": "pre_test",
            "family_structure": "",
            "family_history": "",
            "sharing_preferences": "",
            "result_delivery_preference": "",
            "known_gene_or_result": "",
            "main_concerns": [],
        }
    )
    notes: Dict[str, Any] = field(default_factory=dict)
    summaries: List[str] = field(default_factory=list)

    def missing_background_items(self) -> List[str]:
        missing = []
        deferred = set(self.notes.get("deferred_background_items", []))
        if not self.has_diagnosis_status:
            missing.append("diagnosis_status")
        if not self.has_family_structure:
            missing.append("family_structure")
        if not self.has_family_history:
            missing.append("family_history")
        if not self.has_sharing_preferences:
            missing.append("sharing_preferences")
        if not self.has_result_delivery_pref:
            missing.append("result_delivery_preference")
        return [item for item in missing if item not in deferred]

    def maybe_advance_stage(self) -> Optional[Tuple[Stage, Stage]]:
        previous = self.stage
        if self.stage == Stage.BACKGROUND and not self.missing_background_items():
            self.stage = Stage.EDUCATION
        if previous != self.stage:
            return previous, self.stage
        return None

    def record_note(self, key: str, value: str, overwrite: bool = False) -> None:
        value = value.strip()
        current = str(self.essential_info.get(key, "")).strip()
        if not value:
            return
        if overwrite or not current or current in {"unclear", "pre_test"}:
            self.essential_info[key] = value


@dataclass
class ExtractedTurnState:
    diagnosis_status: str = "unclear"
    has_family_structure: bool = False
    family_structure_note: str = ""
    has_family_history: bool = False
    family_history_note: str = ""
    has_sharing_preferences: bool = False
    sharing_preferences_note: str = ""
    has_result_delivery_pref: bool = False
    result_delivery_pref_note: str = ""
    testing_context: str = "pre_test"
    known_gene_or_result: str = ""
    user_is_asking_direct_question: bool = False
    unresolved_question: str = ""
    concern_note: str = ""
    distress_detected: bool = False


SYSTEM_PROMPT = """
You are an AI assistant supporting clinical genetics teams with pre-test counselling about MND/ALS genetic testing.

You give general counselling information in a warm, concise, clinically sensible way. You are not a clinician and do not replace the person's neurology or genetics team.

Core principles:
- Use only facts the person has given you or that are in the conversation summary.
- Do not invent diagnoses, family history, genetic results, or clinic arrangements.
- If something important is missing, ask one short clarifying question.
- Answer ordinary counselling questions directly in general terms.
- Keep replies short enough for someone who may be tired: usually 2-4 short paragraphs.
- Do not make personalised medical decisions.
- If a gene lookup is provided, use it to explain in plain language, but do not overstate certainty.
- Genetic testing is optional. Do not imply that the person has decided to test.
- If the person asks to skip personal or background questions, respect that choice,
  continue with general information, and do not ask the remaining intake questions.

Result interpretation guardrails:
- Do not say that genetic testing is not recommended solely because there is no
  known family history. Current ALS consensus guidance recommends offering
  genetic testing and counselling to all people with ALS; the final decision
  remains the person's choice.
- If the person already has an ALS/MND diagnosis, do not describe testing as
  estimating whether they will develop ALS/MND. Explain that testing may help
  identify a genetic contribution to the existing diagnosis.
- A negative result means no reportable disease-causing variant was found within the genes and methods used. It does not exclude every genetic contribution.
- Do not describe a negative result as reassuring without explaining those limits.
- A negative result usually leaves no identified familial variant for targeted predictive testing in relatives; relatives' risk still depends on the family history and specialist assessment.
- A variant of uncertain significance is not a confirmed explanation and should not normally be used for predictive testing in relatives.
- A positive result's implications depend on the exact variant, inheritance pattern, and specialist interpretation.
- Genetic testing may help explain an existing clinical diagnosis; do not say that it makes or rules out the clinical diagnosis of MND.
- Apparently sporadic ALS/MND can still have a genetic contribution. Reasons can
  include reduced penetrance, small family size, relatives dying before symptoms,
  unrecognised ALS/FTD, incomplete family information, or a new/de novo variant.
- For many ALS/FTD genes, a pathogenic variant may be inherited in an autosomal
  dominant pattern. If the affected person has such a variant, each child may
  have a 1 in 2 chance of inheriting the variant. Make clear this is not the
  same as a 1 in 2 chance of developing ALS/MND because penetrance and age of
  onset vary.
- Usually the affected person is tested first. Predictive testing for unaffected
  adult relatives is usually considered only if a pathogenic familial variant is
  found, and should involve specialist genetic counselling.
- Predictive testing for adult-onset ALS/FTD conditions is generally not
  recommended for minors; adult relatives should be able to choose for
  themselves when they are adults.
- A partner is usually not biologically at increased genetic risk because of the
  person's result, unless there are separate clinical or family-history reasons.

Stages:
1. BACKGROUND: establish diagnosis/testing context, close family, family history, sharing preferences, and result-delivery preference.
2. EDUCATION: give a short explanation of what pre-test genetic counselling and testing can mean for this person.
3. QUESTIONS: answer the person's questions directly and briefly.
4. WRAP_UP: summarise key points and encourage discussion with their clinical team.

Safety and support:
- Include support signposting when the person seems distressed, asks for help, or once early in the conversation.
- Support signposting may mention their neurology/genetics team, MND Association helpline 0808 802 6262, and Samaritans 116 123.
- Direct patient questions take priority over stage progression. Answer the
  latest question first; do not repeat an intake question if the person redirects,
  declines, is confused, or asks a factual counselling question.
"""


def build_system_message(
    state: CounsellingState, prompt_candidate: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    summary_lines = [
        f"Current stage: {state.stage.value}",
        "Stage goal: "
        + {
            Stage.BACKGROUND: "collect essential background one question at a time",
            Stage.EDUCATION: "give concise pre-test genetics explanation",
            Stage.QUESTIONS: "answer user questions directly",
            Stage.WRAP_UP: "summarise and signpost",
        }[state.stage],
        f"Missing background items: {state.missing_background_items()}",
        f"Essential information: {json.dumps(state.essential_info, ensure_ascii=False)}",
        f"Conversation summaries: {json.dumps(state.summaries[-3:], ensure_ascii=False)}",
        f"Notes: {json.dumps(state.notes, ensure_ascii=False)}",
    ]
    system_prompt = SYSTEM_PROMPT
    if prompt_candidate is not None:
        system_prompt = apply_prompt_directive(system_prompt, "system", prompt_candidate)
    return {
        "role": "system",
        "content": system_prompt + "\n\nAgent state:\n" + "\n".join(summary_lines),
    }


class MNDAgent:
    def __init__(self, backend: LLMBackend):
        self.backend = backend
        self.state = CounsellingState()
        self.history: List[Dict[str, str]] = []

    def _extract_turn_state(self, user_text: str) -> ExtractedTurnState:
        """
        Fast deterministic extraction of essential counselling information.

        This avoids spending an extra LLM call on every turn. The conversation
        still uses the LLM for patient-facing replies and stage summaries.
        """
        text = user_text.lower()

        diagnosis_status = "unclear"
        if re.search(r"\b(i have|i've got|diagnosed|diagnosis|confirmed|my mnd|my als)\b", text):
            diagnosis_status = "confirmed"
        elif re.search(r"\b(suspected|likely|probably|being treated as)\b.*\b(mnd|als)\b", text):
            diagnosis_status = "likely"
        elif re.search(r"\b(waiting|pending|not confirmed|tests? pending)\b", text):
            diagnosis_status = "pending"

        has_family_structure = bool(
            re.search(r"\b(wife|husband|partner|child|children|son|daughter|sibling|brother|sister|mother|father|parent|family)\b", text)
        )
        has_family_history = bool(
            re.search(
                r"\b(no family history|family history|runs in (my|the) family|"
                r"relative.*(mnd|als|dementia|mutation|variant)|"
                r"anyone else.*(mnd|als|dementia)|"
                r"no one else.*(mnd|als|dementia)|"
                r"parkinson|frontotemporal|dementia|ftd|alzheimer|mutation|variant)\b",
                text,
            )
        )
        has_sharing_preferences = bool(
            re.search(r"\b(tell|share|not know|prefer not to know|who should know|family should know|discuss with)\b", text)
        )
        has_result_delivery_pref = bool(
            re.search(r"\b(in person|phone|video|letter|email|receive.*result|result.*by|how.*result)\b", text)
        )

        testing_context = "pre_test"
        if re.search(r"\b(already tested|got.*result|result came back|positive|negative|vus|variant of uncertain|mutation)\b", text):
            testing_context = "known_result"
        elif re.search(r"\b(test done|tested|sample taken)\b", text):
            testing_context = "post_test"

        gene_symbol = find_gene_in_text(user_text) or ""
        user_is_asking_direct_question = "?" in user_text or bool(
            re.search(r"\b(what|why|how|when|who|should|could|can|does|do i|will|would)\b", text)
        )
        distress_detected = bool(
            re.search(r"\b(scared|worried|anxious|upset|overwhelmed|distressed|panic|hopeless|suicidal|can't cope|cannot cope)\b", text)
        )

        concern_note = ""
        if re.search(r"\b(child|children|son|daughter|family|relative|brother|sister)\b", text):
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
            user_is_asking_direct_question=user_is_asking_direct_question,
            unresolved_question=user_text.strip()[:220] if user_is_asking_direct_question else "",
            concern_note=concern_note,
            distress_detected=distress_detected,
        )

    def _apply_extracted_state(self, extracted: ExtractedTurnState) -> None:
        if extracted.diagnosis_status in {"confirmed", "likely"}:
            self.state.has_diagnosis_status = True
            self.state.has_confirmed_diagnosis = True
            self.state.record_note("diagnosis_status", extracted.diagnosis_status)
        elif extracted.diagnosis_status == "pending":
            self.state.has_diagnosis_status = True
            self.state.record_note("diagnosis_status", "pending confirmation")

        if extracted.has_family_structure:
            self.state.has_family_structure = True
            self.state.record_note("family_structure", extracted.family_structure_note)

        if extracted.has_family_history:
            self.state.has_family_history = True
            self.state.record_note("family_history", extracted.family_history_note)

        if extracted.has_sharing_preferences:
            self.state.has_sharing_preferences = True
            self.state.record_note("sharing_preferences", extracted.sharing_preferences_note)

        if extracted.has_result_delivery_pref:
            self.state.has_result_delivery_pref = True
            self.state.record_note("result_delivery_preference", extracted.result_delivery_pref_note)

        if extracted.testing_context:
            self.state.record_note("testing_context", extracted.testing_context, overwrite=True)

        if extracted.known_gene_or_result:
            self.state.record_note("known_gene_or_result", extracted.known_gene_or_result, overwrite=True)

        if extracted.unresolved_question:
            self.state.notes["last_user_question"] = extracted.unresolved_question

        if extracted.concern_note:
            concerns = self.state.essential_info.setdefault("main_concerns", [])
            if extracted.concern_note not in concerns:
                concerns.append(extracted.concern_note)

    def _next_background_question(self) -> str:
        if not self.state.has_confirmed_diagnosis:
            return (
                "To understand the context, could I check whether you have been diagnosed "
                "with MND/ALS, or whether diagnosis is still being confirmed?"
            )
        if not self.state.has_family_structure:
            return (
                "Could you tell me a little about your close family, for example whether "
                "you have a partner, children, siblings, or other relatives you are thinking about?"
            )
        if not self.state.has_family_history:
            return (
                "Is there anyone else in the family with MND/ALS, frontotemporal dementia, "
                "or a known genetic result, or is there no known family history?"
            )
        if not self.state.has_sharing_preferences:
            return (
                "Have you thought about who in your family you might want to tell about "
                "testing, and whether anyone might prefer not to know?"
            )
        if not self.state.has_result_delivery_pref:
            return (
                "If a result becomes available, how would you prefer to receive and discuss "
                "it, for example in person, by phone or video, or by letter?"
            )
        return "Thank you. I have the main background I need."

    def _summarize_and_compact_history(self, reason: str) -> None:
        if len(self.history) < 4:
            return

        transcript = "\n".join(
            f"{message.get('role', '').upper()}: {message.get('content', '')}"
            for message in self.history
        )
        prompt = (
            "Summarise this MND genetics counselling conversation for the next stage. "
            "Keep only clinically relevant facts the user gave, questions asked, emotional "
            "concerns, family context, gene/result mentions, and any preferences. "
            "Do not add facts not present. Use 5-8 concise bullet points.\n\n"
            f"Reason for summary: {reason}\n\nTranscript:\n{transcript}"
        )
        try:
            summary = self.backend.chat(
                [
                    {
                        "role": "system",
                        "content": "You compact conversation history for a counselling agent. Be factual and concise.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )
        except Exception:
            summary = (
                f"Compact summary after {reason}: "
                f"{json.dumps(self.state.essential_info, ensure_ascii=False)}"
            )

        if summary:
            self.state.summaries.append(summary.strip())

        # Keep the most recent exchange for immediate conversational continuity.
        self.history = self.history[-4:]

    def _messages_for_reply(self, user_text: str) -> List[Dict[str, str]]:
        messages = [build_system_message(self.state)]
        messages.extend(self.history[-8:])
        messages.append({"role": "user", "content": user_text})
        return messages

    def _append_support_if_needed(self, reply: str, distress_detected: bool = False) -> str:
        if distress_detected or not self.state.support_given:
            if SUPPORT_SIGNPOST not in reply:
                reply = reply.rstrip() + "\n\n" + SUPPORT_SIGNPOST
            self.state.support_given = True
        return reply

    def _generate_stage2_education(self) -> str:
        education_prompt = (
            "Give the Stage 2 education explanation now. Explain, in plain language, what "
            "pre-test MND/ALS genetic testing can and cannot tell someone; what positive, "
            "negative, and uncertain results can mean; what it may mean for relatives; and "
            "that their clinical genetics/neurology team should guide final decisions. "
            "Use the recorded essential information. Keep it concise and end by inviting questions."
        )
        reply = self.backend.chat(self._messages_for_reply(education_prompt))
        self.state.education_given = True
        previous = self.state.stage
        self.state.stage = Stage.QUESTIONS
        self._summarize_and_compact_history(f"transition from {previous.value} to questions")
        return self._append_support_if_needed(reply)

    def _answer_with_gene_tool(self, user_text: str, gene_symbol: str) -> str:
        gene_info = lookup_gene_info(gene_symbol)
        self.state.record_note("known_gene_or_result", gene_symbol, overwrite=True)
        self.state.notes["last_gene_lookup"] = {"gene": gene_symbol, "tool_result": gene_info}

        tool_prompt = (
            f"The user asked about {gene_symbol} in the context of MND/ALS genetic testing.\n\n"
            f"Tool result:\n{gene_info}\n\n"
            "Use this tool result to answer in plain language. Explain that exact meaning depends "
            "on the specific variant and clinical genetics interpretation. Do not mention API calls "
            "or technical tool details. Keep it brief and supportive."
        )
        reply = self.backend.chat(self._messages_for_reply(tool_prompt))
        return self._append_support_if_needed(reply)

    def _answer_in_current_stage(self, user_text: str) -> str:
        reply = self.backend.chat(self._messages_for_reply(user_text))
        return reply

    def update_state_from_user(self, user_text: str) -> ExtractedTurnState:
        extracted = self._extract_turn_state(user_text)
        self._apply_extracted_state(extracted)
        transition = self.state.maybe_advance_stage()
        if transition:
            previous, current = transition
            self._summarize_and_compact_history(f"transition from {previous.value} to {current.value}")
        return extracted

    def ask(self, user_text: str) -> str:
        extracted = self.update_state_from_user(user_text)
        self.history.append({"role": "user", "content": user_text})

        gene_symbol = extracted.known_gene_or_result or find_gene_in_text(user_text)
        if gene_symbol:
            reply = self._answer_with_gene_tool(user_text, gene_symbol)
            self.history.append({"role": "assistant", "content": reply})
            return reply

        if self.state.stage == Stage.BACKGROUND:
            if extracted.user_is_asking_direct_question:
                reply = (
                    "I can answer that, and I will keep it general for now. "
                    "First I need one bit of background so I do not make assumptions. "
                    + self._next_background_question()
                )
            else:
                reply = self._next_background_question()
            reply = self._append_support_if_needed(reply, distress_detected=extracted.distress_detected)
            self.history.append({"role": "assistant", "content": reply})
            return reply

        if self.state.stage == Stage.EDUCATION and not self.state.education_given:
            reply = self._generate_stage2_education()
            self.history.append({"role": "assistant", "content": reply})
            return reply

        if re.search(r"\b(summary|summarise|summarize|wrap up|finish|end)\b", user_text.lower()):
            self.state.stage = Stage.WRAP_UP

        reply = self._answer_in_current_stage(user_text)
        reply = self._append_support_if_needed(reply, distress_detected=extracted.distress_detected)
        self.history.append({"role": "assistant", "content": reply})

        if len(self.history) > 14:
            self._summarize_and_compact_history("periodic context compaction")

        return reply
