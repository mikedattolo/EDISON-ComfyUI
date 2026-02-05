"""
Optional vLLM inference server for faster batching.
Run separately from edison_core.
"""

import asyncio
import logging
from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

try:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
except Exception:  # pragma: no cover - optional dependency
    SamplingParams = None
    AsyncEngineArgs = None
    AsyncLLMEngine = None

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    mode: str = "fast"
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9

class VLLMInferenceServer:
    def __init__(self, model_map: Dict[str, str]):
        if AsyncLLMEngine is None:
            raise RuntimeError("vLLM not installed. Install vllm==0.6.3 to use this server.")
        self.engines = {
            name: AsyncLLMEngine.from_engine_args(
                AsyncEngineArgs(
                    model=path,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.85
                )
            )
            for name, path in model_map.items()
        }

    async def generate(self, prompt: str, mode: str, max_tokens: int, temperature: float, top_p: float):
        if mode not in self.engines:
            raise ValueError(f"Unknown mode '{mode}'")
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        results = await self.engines[mode].generate(prompt, sampling_params)
        return results

server = None

@app.post("/generate")
async def generate(req: GenerateRequest):
    global server
    if server is None:
        raise HTTPException(status_code=503, detail="vLLM server not initialized")
    try:
        results = await server.generate(
            req.prompt,
            req.mode,
            req.max_tokens,
            req.temperature,
            req.top_p
        )
        return {"results": [r.outputs[0].text for r in results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup():
    global server
    if AsyncLLMEngine is None:
        logger.warning("vLLM not installed; server will not start")
        return
    # Default models; override by setting environment variables or editing this file.
    model_map = {
        "fast": "/opt/edison/models/llm/qwen2.5-14b-instruct-q4_k_m.gguf",
        "deep": "/opt/edison/models/llm/qwen2.5-72b-instruct-q4_k_m.gguf"
    }
    server = VLLMInferenceServer(model_map)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8822)
