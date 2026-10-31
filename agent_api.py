import argparse
import json
import logging
import os
import re
import threading
import time
import uuid
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent_base import LLMBackend, VLLMBackend
from api_backend import APIBackend
from model_config import APP_DIR, ModelConfig, load_environment
from mnd_langgraph_agent import LangGraphMNDAgent
from tools import warm_knowledge_retrieval
from prompt_candidate import baseline_candidate, load_prompt_candidate, validate_prompt_candidate

load_environment()
SERVED_MODEL_NAME = os.environ.get("MND_MODEL", "mnd-local-agent")
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


class ResetRequest(BaseModel):
    session_id: Optional[str] = None
    prompt_candidate: Optional[dict] = None


class OpenAIMessage(BaseModel):
    role: str
    content: Union[str, list, dict, None] = ""


class OpenAIChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[OpenAIMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    user: Optional[str] = None


def build_args(argv=None):
    load_environment()
    config = ModelConfig.from_env("agent")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=config.model,
        help="API model name or local model directory (MND_AGENT_MODEL).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=config.max_tokens,
        help="Max new tokens for the local agent per reply.",
    )
    parser.add_argument(
        "--backend",
        choices=["api", "transformers", "vllm"],
        default=os.environ.get("MND_BACKEND", "api"),
        help="Inference backend to use.",
    )
    parser.add_argument(
        "--vllm_max_model_len",
        type=int,
        default=int(os.environ.get("VLLM_MAX_MODEL_LEN", "4096")),
        help="Maximum context length for the vLLM backend.",
    )
    parser.add_argument(
        "--vllm_max_num_seqs",
        type=int,
        default=int(os.environ.get("VLLM_MAX_NUM_SEQS", "10")),
        help="Maximum number of concurrent vLLM sequences.",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.95")),
        help="GPU memory fraction reserved by the vLLM backend.",
    )
    parser.add_argument(
        "--vllm_dtype",
        default=os.environ.get("VLLM_DTYPE", "bfloat16"),
        help="vLLM model dtype.",
    )
    parser.add_argument(
        "--served_model_name",
        default=os.environ.get("SERVED_MODEL_NAME"),
        help="Optional model name exposed inside the vLLM engine.",
    )
    parser.add_argument(
        "--save_state",
        action="store_true",
        help="If set, also save the final agent state as JSON per vignette.",
    )
    parser.add_argument(
        "--print_live",
        action="store_true",
        help="If set, print the simulated conversation to stdout in real time.",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default=os.environ.get("MND_BASE_PATH", str(APP_DIR)),
        help="Base directory used when --model is a model name.",
    )
    parser.add_argument("--api-base", "--api_base", default=config.api_base)
    parser.add_argument("--api-key", "--api_key", default=config.api_key)
    parser.add_argument("--timeout", type=float, default=config.timeout)
    parser.add_argument("--temperature", type=float, default=config.temperature)
    parser.add_argument("--token-parameter", choices=["max_tokens", "max_completion_tokens"], default=config.token_parameter)
    parser.add_argument("--prompt-candidate", default=os.environ.get("MND_PROMPT_CANDIDATE"),
                        help="Validated candidate JSON to use for new web/API conversations.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


def extract_text(content: Union[str, list, dict, None]) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        if content.get("type") == "text":
            return str(content.get("text", ""))
        return str(content)

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)

    return str(content)


# --- Open WebUI internal task detection and response ---
def detect_openwebui_task(messages: List[OpenAIMessage]) -> str:
    """
    Open WebUI may send extra internal requests for follow-up suggestions,
    titles, tags, or search queries. These should not be routed through the
    clinical counselling agent, otherwise the UI can appear to hang or display
    JSON-like helper outputs instead of the assistant answer.
    """
    for message in messages:
        text = extract_text(message.content).lstrip().lower()
        if not text.startswith("### task:"):
            continue
        # Only inspect the explicit task section; quoted chat history can contain
        # ordinary questions about titles, follow-up appointments, or searching.
        task = re.split(r"\n###\s+", text, maxsplit=1)[0]
        if any(term in task for term in ("follow_up", "follow-up", "follow ups", "followups")):
            return "followups"
        if "generate" in task and "title" in task:
            return "title"
        if "generate" in task and "tags" in task:
            return "tags"
        if any(term in task for term in ("search query", "search queries", "web search")):
            return "search"
    return "chat"


def answer_openwebui_internal_task(task: str, user_message: str) -> str:
    if task == "followups":
        return '{"follow_ups": []}'
    if task == "title":
        return "MND genetic counselling"
    if task == "tags":
        return '{"tags": ["MND", "genetic counselling"]}'
    if task == "search":
        return user_message or "MND genetic counselling"
    return ""


def build_backend(settings):
    if not settings.model:
        raise ValueError("Set MND_AGENT_MODEL or --model before starting the agent.")
    if settings.backend == "api":
        return APIBackend(ModelConfig.from_env(
            "agent", model=settings.model, api_base=settings.api_base,
            api_key=settings.api_key, timeout=settings.timeout,
            temperature=settings.temperature, max_tokens=settings.max_new_tokens,
            token_parameter=settings.token_parameter,
        ))
    return VLLMBackend(settings) if settings.backend == "vllm" else LLMBackend(settings)


# Importing this module never parses another program's arguments, loads weights,
# downloads retrieval models, or starts a background thread.
args = build_args([])
backend = None
configured_prompt_candidate = baseline_candidate()
cleanup_stop = threading.Event()


@asynccontextmanager
async def lifespan(application):
    global backend, configured_prompt_candidate
    candidate_path = args.prompt_candidate
    if candidate_path:
        candidate_path = Path(candidate_path).expanduser()
        if not candidate_path.is_absolute():
            candidate_path = APP_DIR / candidate_path
        configured_prompt_candidate = load_prompt_candidate(candidate_path)
    else:
        configured_prompt_candidate = baseline_candidate()
    warm_knowledge_retrieval()
    backend = build_backend(args)
    cleanup_stop.clear()
    cleanup_thread = threading.Thread(
        target=session_cleanup_loop, name="mnd-session-cleanup", daemon=True,
    )
    cleanup_thread.start()
    try:
        yield
    finally:
        cleanup_stop.set()
        cleanup_thread.join(timeout=2)
        with agents_lock:
            agents.clear()
            agent_locks.clear()
            agent_last_access.clear()
        backend = None
        configured_prompt_candidate = baseline_candidate()


app = FastAPI(title="MND Counselling Agent API", lifespan=lifespan)
agents: Dict[str, LangGraphMNDAgent] = {}
agent_locks: Dict[str, threading.Lock] = {}
agent_last_access: Dict[str, float] = {}
agents_lock = threading.Lock()
SESSION_TTL_SECONDS = int(os.environ.get("MND_SESSION_TTL_SECONDS", str(2 * 60 * 60)))
SESSION_CLEANUP_INTERVAL_SECONDS = int(
    os.environ.get("MND_SESSION_CLEANUP_INTERVAL_SECONDS", "300")
)
SESSION_MAX_COUNT = int(os.environ.get("MND_SESSION_MAX_COUNT", "200"))
SESSION_STATE_DIR = Path(
    os.environ.get(
        "MND_BACKEND_SESSION_DIR",
        str(APP_DIR / ".runtime" / "backend_sessions"),
    )
).expanduser()
SAFE_SESSION_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def session_state_path(session_id: str) -> Path:
    safe_id = SAFE_SESSION_RE.sub("_", session_id).strip("._") or "session"
    return SESSION_STATE_DIR / f"{safe_id}.json"


def load_agent_state(session_id: str, agent: LangGraphMNDAgent) -> None:
    path = session_state_path(session_id)
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Persisted agent state must be an object")
        stored_candidate = validate_prompt_candidate(payload.get("prompt_candidate", baseline_candidate()))
        same_candidate = stored_candidate["candidate_sha256"] == agent.prompt_candidate["candidate_sha256"]
        offline_restore = (os.environ.get("MND_ALLOW_PROMPT_CANDIDATES") == "1"
                           and not configured_prompt_candidate["node_directives"])
        if not same_candidate and not offline_restore:
            raise ValueError("Persisted session candidate differs from the configured candidate")
        agent.load_dict(payload)
    except Exception as exc:
        logger.warning("Could not restore persisted agent state (%s).", type(exc).__name__)
        raise HTTPException(
            status_code=409,
            detail="Stored session is incompatible or invalid. Reset this conversation before continuing.",
        ) from None


def save_agent_state(session_id: str, agent: LangGraphMNDAgent) -> None:
    try:
        SESSION_STATE_DIR.mkdir(parents=True, exist_ok=True)
        path = session_state_path(session_id)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(agent.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except Exception as exc:
        logger.warning("Could not save persisted agent state (%s).", type(exc).__name__)


def resolve_session_id(
    request: Request,
    explicit_session_id: Optional[str] = None,
) -> str:
    header_session_id = next(
        (
            request.headers.get(header)
            for header in (
                "x-session-id",
                "x-openwebui-chat-id",
                "x-conversation-id",
                "x-chat-id",
            )
            if request.headers.get(header)
        ),
        None,
    )
    if explicit_session_id or header_session_id:
        return str(explicit_session_id or header_session_id)
    if request.client and request.client.host:
        return f"client:{request.client.host}"
    return uuid.uuid4().hex


def get_agent_and_lock(session_id: str):
    with agents_lock:
        if session_id not in agents:
            agent = LangGraphMNDAgent(backend, prompt_candidate=configured_prompt_candidate)
            load_agent_state(session_id, agent)
            agents[session_id] = agent
            agent_locks[session_id] = threading.Lock()
        agent_last_access[session_id] = time.time()
        return agents[session_id], agent_locks[session_id]


def cleanup_idle_sessions(force: bool = False) -> Dict[str, Union[int, float]]:
    now = time.time()
    removed = 0
    with agents_lock:
        session_ids = list(agents)
        expired_session_ids = [
            session_id
            for session_id in session_ids
            if force
            or now - agent_last_access.get(session_id, now) > SESSION_TTL_SECONDS
        ]
        if (
            not force
            and SESSION_MAX_COUNT > 0
            and len(session_ids) - len(expired_session_ids) > SESSION_MAX_COUNT
        ):
            active_session_ids = [
                session_id
                for session_id in session_ids
                if session_id not in set(expired_session_ids)
            ]
            active_session_ids.sort(
                key=lambda session_id: agent_last_access.get(session_id, 0)
            )
            overflow = len(active_session_ids) - SESSION_MAX_COUNT
            expired_session_ids.extend(active_session_ids[:overflow])

        for session_id in dict.fromkeys(expired_session_ids):
            agents.pop(session_id, None)
            agent_locks.pop(session_id, None)
            agent_last_access.pop(session_id, None)
            removed += 1

        return {
            "removed": removed,
            "active_sessions": len(agents),
            "ttl_seconds": SESSION_TTL_SECONDS,
            "max_sessions": SESSION_MAX_COUNT,
        }


def session_cleanup_loop() -> None:
    while not cleanup_stop.wait(max(30, SESSION_CLEANUP_INTERVAL_SECONDS)):
        stats = cleanup_idle_sessions()
        if stats["removed"]:
            print(
                "Session cleanup removed",
                stats["removed"],
                "idle sessions; active_sessions=",
                stats["active_sessions"],
            )


def ask_agent(
    session_id: str,
    user_message: str,
    requested_tokens: Optional[int] = None,
) -> str:
    session_agent, session_lock = get_agent_and_lock(session_id)
    max_new_tokens = None
    if requested_tokens:
        max_new_tokens = max(128, min(int(requested_tokens), 1536))
    with session_lock:
        with backend.generation_options(max_new_tokens):
            response = session_agent.ask(user_message)
        save_agent_state(session_id, session_agent)
        return response


def cuda_available():
    torch = sys.modules.get("torch")
    return bool(torch is not None and torch.cuda.is_available())


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "MND agent API is running.",
        "model": SERVED_MODEL_NAME,
        "model_path": backend.model_path,
        "backend": backend.backend_name,
        "agent_model": backend.model_path,
        "candidate_id": configured_prompt_candidate["candidate_id"],
        "candidate_sha256": configured_prompt_candidate["candidate_sha256"],
        "deterministic": backend.deterministic,
        "max_concurrent_generations": backend.max_concurrent_generations,
        "max_new_tokens": backend.max_new_tokens,
        "prompt_candidates_enabled": os.environ.get("MND_ALLOW_PROMPT_CANDIDATES") == "1",
        "effective_prompts_recorded": (os.environ.get("MND_ALLOW_PROMPT_CANDIDATES") == "1"
                                       and os.environ.get("MND_RECORD_EFFECTIVE_PROMPTS") == "1"),
        "cuda_available": cuda_available(),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": backend.backend_name,
        "agent_model": backend.model_path,
        "candidate_id": configured_prompt_candidate["candidate_id"],
        "candidate_sha256": configured_prompt_candidate["candidate_sha256"],
        "deterministic": backend.deterministic,
        "cuda_available": cuda_available(),
        "max_concurrent_generations": backend.max_concurrent_generations,
        "max_new_tokens": backend.max_new_tokens,
        "prompt_candidates_enabled": os.environ.get("MND_ALLOW_PROMPT_CANDIDATES") == "1",
        "effective_prompts_recorded": (os.environ.get("MND_ALLOW_PROMPT_CANDIDATES") == "1"
                                       and os.environ.get("MND_RECORD_EFFECTIVE_PROMPTS") == "1"),
        "active_sessions": len(agents),
        "session_ttl_seconds": SESSION_TTL_SECONDS,
        "session_max_count": SESSION_MAX_COUNT,
    }


@app.get("/sessions")
def get_session_stats():
    with agents_lock:
        now = time.time()
        ages = [
            now - last_access
            for last_access in agent_last_access.values()
        ]
        return {
            "active_sessions": len(agents),
            "ttl_seconds": SESSION_TTL_SECONDS,
            "cleanup_interval_seconds": SESSION_CLEANUP_INTERVAL_SECONDS,
            "max_sessions": SESSION_MAX_COUNT,
            "oldest_idle_seconds": round(max(ages), 1) if ages else 0,
        }


@app.post("/sessions/cleanup")
def cleanup_sessions():
    return cleanup_idle_sessions(force=False)


@app.get("/state")
def get_agent_state(request: Request, session_id: Optional[str] = None):
    resolved_session_id = resolve_session_id(request, session_id)
    session_agent, session_lock = get_agent_and_lock(resolved_session_id)
    with session_lock:
        return {
            "session_id": resolved_session_id,
            "stage": session_agent.state.stage.value,
            "has_diagnosis_status": session_agent.state.has_diagnosis_status,
            "has_confirmed_diagnosis": session_agent.state.has_confirmed_diagnosis,
            "missing_background_items": session_agent.state.missing_background_items(),
            "essential_info": session_agent.state.essential_info,
            "notes": session_agent.state.notes,
            "summaries": session_agent.state.summaries,
            "education_given": session_agent.state.education_given,
            "support_given": session_agent.state.support_given,
            "candidate_id": session_agent.prompt_candidate["candidate_id"],
            "candidate_sha256": session_agent.prompt_candidate["candidate_sha256"],
            "prompt_audit": session_agent.prompt_audit,
        }


@app.post("/reset")
def reset_agent(request: Request, req: Optional[ResetRequest] = None):
    session_id = resolve_session_id(request, req.session_id if req is not None else None)
    candidate = req.prompt_candidate if req is not None else None
    if candidate is not None and os.environ.get("MND_ALLOW_PROMPT_CANDIDATES") != "1":
        raise HTTPException(status_code=403, detail=(
            "Prompt candidates are disabled; set MND_ALLOW_PROMPT_CANDIDATES=1 only for offline evaluation."
        ))
    try:
        selected = configured_prompt_candidate if candidate is None else validate_prompt_candidate(candidate)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    # Reset can recover an invalid or incompatible stored session without loading it.
    with agents_lock:
        session_lock = agent_locks.setdefault(session_id, threading.Lock())
    with session_lock:
        replacement = LangGraphMNDAgent(backend, prompt_candidate=selected)
        with agents_lock:
            agents[session_id] = replacement
            agent_last_access[session_id] = time.time()
        save_agent_state(session_id, replacement)
    return {
        "status": "ok",
        "session_id": session_id,
        "message": "Agent conversation state reset.",
        "candidate_id": replacement.prompt_candidate["candidate_id"],
        "candidate_sha256": replacement.prompt_candidate["candidate_sha256"],
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    user_message = req.message.strip()
    session_id = resolve_session_id(request, req.session_id)

    if not user_message:
        return ChatResponse(
            response="Please enter a message.",
            session_id=session_id,
        )

    response = ask_agent(session_id, user_message)

    return ChatResponse(response=response, session_id=session_id)


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": SERVED_MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


def build_chat_completion_response(model: str, response_text: str):
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def stream_chat_completion_response(model: str, response_text: str):
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    first_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

    # Stream in small chunks rather than one very large chunk. This works more
    # reliably with Open WebUI's frontend streaming state.
    words = response_text.split(" ")
    for word in words:
        if not word:
            continue
        content_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": word + " "},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(content_chunk, ensure_ascii=False)}\n\n"

    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
def openai_chat_completions(req: OpenAIChatRequest, request: Request):
    model = req.model or SERVED_MODEL_NAME
    session_id = resolve_session_id(request, req.user)

    user_messages = [extract_text(m.content) for m in req.messages if m.role == "user"]
    user_message = user_messages[-1].strip() if user_messages else ""
    task = detect_openwebui_task(req.messages)


    if not user_message:
        response_text = "Please enter a message."
    elif task != "chat":
        response_text = answer_openwebui_internal_task(task, user_message)
    else:
        requested_tokens = req.max_tokens or req.max_completion_tokens
        response_text = ask_agent(session_id, user_message, requested_tokens)

    response_text = str(response_text).strip()
    logger.info("Chat completion task=%s stream=%s characters=%s", task, bool(req.stream), len(response_text))

    if req.stream:
        return StreamingResponse(
            stream_chat_completion_response(model, response_text),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Session-ID": session_id,
            },
        )

    response = build_chat_completion_response(model, response_text)
    response["session_id"] = session_id
    return response


if __name__ == "__main__":
    import uvicorn

    args = build_args()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=False,
    )
