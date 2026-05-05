"""
Hardened vLLM serving bridge for EDISON.

Phase 1 upgrades over the original ``server.py``:

* Configuration via environment variables instead of hard-coded paths.
* Multiple named lanes (``fast`` / ``deep`` / ``vision``) with proportional
  GPU memory utilization and tensor-parallel size per lane.
* OpenAI-compatible ``/v1/chat/completions`` and ``/v1/completions`` shims so
  the rest of EDISON can talk to vLLM without bespoke clients.
* ``/healthz`` and ``/readyz`` endpoints that report engine load state and
  configured lanes.
* Optional attention/prefix-cache flags driven by env vars so the operator
  can enable them once their environment is known stable.

Run separately from ``edison_core`` (e.g. ``uvicorn services.edison_vllm.bridge:app
--port 8822``). The default ``server.py`` remains for backward compatibility.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# vLLM is optional; we tolerate its absence so the bridge can still expose a
# clear "not configured" response instead of crashing on import.
try:
    from vllm import SamplingParams  # type: ignore
    from vllm.engine.arg_utils import AsyncEngineArgs  # type: ignore
    from vllm.engine.async_llm_engine import AsyncLLMEngine  # type: ignore

    VLLM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    SamplingParams = None  # type: ignore
    AsyncEngineArgs = None  # type: ignore
    AsyncLLMEngine = None  # type: ignore
    VLLM_AVAILABLE = False


# ── Lane configuration ────────────────────────────────────────────────

#: Default lane definitions. Each lane has a model path env var and a sane
#: fallback. Use environment variables to override at runtime without
#: editing this file.
DEFAULT_LANES: Dict[str, Dict[str, Any]] = {
    "fast": {
        "model_env": "EDISON_VLLM_FAST_MODEL",
        "default_model": "/opt/edison/models/llm/qwen2.5-14b-instruct-q4_k_m.gguf",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.80,
        "max_model_len": 8192,
        "enable_prefix_caching": True,
    },
    "deep": {
        "model_env": "EDISON_VLLM_DEEP_MODEL",
        "default_model": "/opt/edison/models/llm/qwen2.5-32b-instruct-q4_k_m.gguf",
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.85,
        "max_model_len": 8192,
        "enable_prefix_caching": True,
    },
    "vision": {
        "model_env": "EDISON_VLLM_VISION_MODEL",
        "default_model": "",  # disabled unless explicitly configured
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.70,
        "max_model_len": 4096,
        "enable_prefix_caching": False,
    },
}


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_lane_config(lane: str, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the engine kwargs for ``lane`` or ``None`` if disabled."""
    model = os.environ.get(raw["model_env"], raw.get("default_model", ""))
    if not model or not isinstance(model, str):
        return None
    return {
        "model": model,
        "tensor_parallel_size": int(
            os.environ.get(
                f"EDISON_VLLM_{lane.upper()}_TP",
                raw.get("tensor_parallel_size", 1),
            )
        ),
        "gpu_memory_utilization": float(
            os.environ.get(
                f"EDISON_VLLM_{lane.upper()}_GPU_UTIL",
                raw.get("gpu_memory_utilization", 0.85),
            )
        ),
        "max_model_len": int(
            os.environ.get(
                f"EDISON_VLLM_{lane.upper()}_MAX_LEN",
                raw.get("max_model_len", 8192),
            )
        ),
        "enable_prefix_caching": _env_bool(
            f"EDISON_VLLM_{lane.upper()}_PREFIX_CACHE",
            raw.get("enable_prefix_caching", True),
        ),
    }


# ── Engine registry ───────────────────────────────────────────────────

class EngineRegistry:
    """Holds one vLLM engine per configured lane."""

    def __init__(self) -> None:
        self.engines: Dict[str, Any] = {}
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.errors: Dict[str, str] = {}
        self.ready: bool = False

    def configure(self) -> None:
        """Build engines for every lane that has a configured model.

        Any per-lane failure is captured but does not prevent other lanes
        from coming online.
        """
        if not VLLM_AVAILABLE:
            self.ready = False
            self.errors["__global__"] = "vllm not installed"
            return

        for lane, raw in DEFAULT_LANES.items():
            cfg = _resolve_lane_config(lane, raw)
            if not cfg:
                logger.info("vLLM lane %s disabled (no model configured)", lane)
                continue
            try:
                args = AsyncEngineArgs(**cfg)  # type: ignore[arg-type]
                self.engines[lane] = AsyncLLMEngine.from_engine_args(args)  # type: ignore[union-attr]
                self.configs[lane] = cfg
                logger.info("vLLM lane %s online: %s", lane, cfg["model"])
            except Exception as exc:  # noqa: BLE001
                self.errors[lane] = f"{type(exc).__name__}: {exc}"
                logger.exception("vLLM lane %s failed to start", lane)

        self.ready = bool(self.engines)

    def get(self, lane: str) -> Any:
        engine = self.engines.get(lane)
        if engine is None:
            raise HTTPException(
                status_code=503,
                detail=f"vLLM lane '{lane}' is not configured. "
                       f"Configured lanes: {list(self.engines.keys())}",
            )
        return engine

    def status(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "vllm_available": VLLM_AVAILABLE,
            "lanes": {
                lane: {
                    "model": cfg["model"],
                    "tensor_parallel_size": cfg["tensor_parallel_size"],
                    "gpu_memory_utilization": cfg["gpu_memory_utilization"],
                    "max_model_len": cfg["max_model_len"],
                    "enable_prefix_caching": cfg["enable_prefix_caching"],
                }
                for lane, cfg in self.configs.items()
            },
            "errors": dict(self.errors),
        }


registry = EngineRegistry()


# ── Pydantic models (OpenAI-compatible subset) ────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="fast", description="Lane name: fast|deep|vision")
    messages: List[ChatMessage]
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    stop: Optional[List[str]] = None


class CompletionRequest(BaseModel):
    model: str = "fast"
    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    stop: Optional[List[str]] = None


# ── App ───────────────────────────────────────────────────────────────

app = FastAPI(title="EDISON vLLM Bridge", version="1.0")


@app.on_event("startup")
async def _startup() -> None:
    registry.configure()
    if not registry.ready:
        logger.warning(
            "vLLM bridge started without any active lanes; "
            "set EDISON_VLLM_FAST_MODEL or EDISON_VLLM_DEEP_MODEL to enable."
        )


@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    return {"ok": registry.ready, "vllm_available": VLLM_AVAILABLE}


@app.get("/readyz")
async def readyz() -> JSONResponse:
    status = registry.status()
    code = 200 if status["ready"] else 503
    return JSONResponse(status, status_code=code)


@app.get("/v1/models")
async def list_models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": lane,
                "object": "model",
                "owned_by": "edison",
                "vllm_model": cfg["model"],
            }
            for lane, cfg in registry.configs.items()
        ],
    }


def _format_messages_as_prompt(messages: List[ChatMessage]) -> str:
    """Lightweight prompt builder. The real chat template is owned by the
    underlying tokenizer when vLLM is started with ``chat_template`` support;
    this is a best-effort fallback for raw-prompt mode.
    """
    parts: List[str] = []
    for msg in messages:
        parts.append(f"<|{msg.role}|>\n{msg.content}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


async def _run_engine(lane: str, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if SamplingParams is None:  # pragma: no cover - vllm missing
        raise HTTPException(status_code=503, detail="vllm not installed")

    engine = registry.get(lane)
    sampling = SamplingParams(  # type: ignore[call-arg]
        temperature=params.get("temperature", 0.7),
        top_p=params.get("top_p", 0.9),
        max_tokens=params.get("max_tokens", 1024),
        stop=params.get("stop"),
    )

    request_id = f"edison-{uuid.uuid4().hex[:12]}"
    final_text = ""
    async for output in engine.generate(prompt, sampling, request_id=request_id):
        # AsyncLLMEngine yields RequestOutput objects with cumulative text
        if output.outputs:
            final_text = output.outputs[0].text

    return {
        "id": request_id,
        "lane": lane,
        "text": final_text,
    }


async def _stream_engine(lane: str, prompt: str, params: Dict[str, Any]):
    """Yield Server-Sent Events for a streaming completion."""
    if SamplingParams is None:  # pragma: no cover
        raise HTTPException(status_code=503, detail="vllm not installed")

    engine = registry.get(lane)
    sampling = SamplingParams(  # type: ignore[call-arg]
        temperature=params.get("temperature", 0.7),
        top_p=params.get("top_p", 0.9),
        max_tokens=params.get("max_tokens", 1024),
        stop=params.get("stop"),
    )

    request_id = f"edison-{uuid.uuid4().hex[:12]}"
    last_len = 0
    created = int(time.time())
    async for output in engine.generate(prompt, sampling, request_id=request_id):
        if not output.outputs:
            continue
        full = output.outputs[0].text
        delta = full[last_len:]
        last_len = len(full)
        if not delta:
            continue
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": lane,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": delta},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # final terminator
    done = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": lane,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(done)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    prompt = _format_messages_as_prompt(req.messages)
    params = {
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "stop": req.stop,
    }
    if req.stream:
        return StreamingResponse(
            _stream_engine(req.model, prompt, params),
            media_type="text/event-stream",
        )
    result = await _run_engine(req.model, prompt, params)
    return {
        "id": result["id"],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": "stop",
            }
        ],
    }


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    params = {
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "stop": req.stop,
    }
    if req.stream:
        return StreamingResponse(
            _stream_engine(req.model, req.prompt, params),
            media_type="text/event-stream",
        )
    result = await _run_engine(req.model, req.prompt, params)
    return {
        "id": result["id"],
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "text": result["text"],
                "finish_reason": "stop",
            }
        ],
    }


# Backwards compatible alias
@app.post("/generate")
async def legacy_generate(req: CompletionRequest):
    return await completions(req)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    port = int(os.environ.get("EDISON_VLLM_PORT", "8822"))
    uvicorn.run(app, host="0.0.0.0", port=port)
