"""
EDISON Core Service - Main Application
FastAPI server with llama-cpp-python for local LLM inference
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Literal, Iterator
import logging
from pathlib import Path
import yaml
import sys
import requests
import json
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get repo root - works regardless of CWD
REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))

# Global state
llm_fast = None
llm_deep = None
rag_system = None
search_tool = None
config = None

# Request/Response models
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    mode: Literal["auto", "chat", "reasoning", "agent", "code"] = Field(
        default="auto", 
        description="Interaction mode"
    )
    remember: bool = Field(
        default=True, 
        description="Store conversation in memory"
    )

class ChatResponse(BaseModel):
    response: str
    mode_used: str
    model_used: str

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    num_results: int = Field(default=5, description="Number of results to return")

class SearchResponse(BaseModel):
    results: list
    query: str

class HealthResponse(BaseModel):
    status: str
    service: str
    models_loaded: dict
    qdrant_ready: bool
    repo_root: str

def load_config():
    """Load EDISON configuration with absolute paths"""
    global config
    try:
        config_path = REPO_ROOT / "config" / "edison.yaml"
        logger.info(f"Looking for config at: {config_path}")
        
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
                logger.info("Configuration loaded successfully")
        else:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            config = {
                "edison": {
                    "core": {
                        "host": "127.0.0.1",
                        "port": 8811,
                        "models_path": "models/llm",
                        "fast_model": "qwen2.5-14b-instruct-q4_k_m.gguf",
                        "deep_model": "qwen2.5-72b-instruct-q4_k_m.gguf"
                    },
                    "coral": {
                        "host": "127.0.0.1",
                        "port": 8808
                    }
                }
            }
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        config = {}

def load_llm_models():
    """Load GGUF models using llama-cpp-python with absolute paths"""
    global llm_fast, llm_deep
    
    try:
        from llama_cpp import Llama
    except ImportError:
        logger.error("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
        return
    
    # Get model paths relative to repo root
    models_rel_path = config.get("edison", {}).get("core", {}).get("models_path", "models/llm")
    models_path = REPO_ROOT / models_rel_path
    
    fast_model_name = config.get("edison", {}).get("core", {}).get("fast_model", "qwen2.5-14b-instruct-q4_k_m.gguf")
    deep_model_name = config.get("edison", {}).get("core", {}).get("deep_model", "qwen2.5-72b-instruct-q4_k_m.gguf")
    
    logger.info(f"Looking for models in: {models_path}")
    
    # Try to load fast model
    fast_model_path = models_path / fast_model_name
    if fast_model_path.exists():
        try:
            logger.info(f"Loading fast model: {fast_model_path}")
            llm_fast = Llama(
                model_path=str(fast_model_path),
                n_ctx=4096,
                n_gpu_layers=-1,  # Use GPU
                tensor_split=[0.5, 0.25, 0.25],  # Split across 3 GPUs: 3090 (50%), 5060ti (25%), 3060 (25%)
                verbose=False
            )
            logger.info("✓ Fast model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load fast model: {e}")
    else:
        logger.warning(f"Fast model not found at {fast_model_path}")
    
    # Try to load deep model
    deep_model_path = models_path / deep_model_name
    if deep_model_path.exists():
        try:
            logger.info(f"Loading deep model: {deep_model_path}")
            llm_deep = Llama(
                model_path=str(deep_model_path),
                n_ctx=8192,
                n_gpu_layers=-1,  # Use GPU
                tensor_split=[0.5, 0.25, 0.25],  # Split across 3 GPUs: 3090 (50%), 5060ti (25%), 3060 (25%)
                verbose=False
            )
            logger.info("✓ Deep model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load deep model: {e}")
    else:
        logger.warning(f"Deep model not found at {deep_model_path}")
    
    if not llm_fast and not llm_deep:
        logger.error("⚠ No models loaded. Please place GGUF models in the models/llm/ directory.")
        logger.error(f"Expected: {models_path / fast_model_name}")
        logger.error(f"      or: {models_path / deep_model_name}")

def init_rag_system():
    """Initialize RAG system with Qdrant and sentence-transformers"""
    global rag_system
    
    try:
        from services.edison_core.rag import RAGSystem
        # Use absolute path for qdrant storage
        qdrant_path = REPO_ROOT / "models" / "qdrant"
        rag_system = RAGSystem(storage_path=str(qdrant_path))
        logger.info(f"✓ RAG system initialized (storage: {qdrant_path})")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        rag_system = None

def init_search_tool():
    """Initialize web search tool"""
    global search_tool
    
    try:
        from services.edison_core.search import WebSearchTool
        search_tool = WebSearchTool()
        logger.info("✓ Web search tool initialized")
    except Exception as e:
        logger.error(f"Failed to initialize search tool: {e}")
        search_tool = None

def get_intent_from_coral(message: str) -> Optional[str]:
    """Try to get intent from coral service (with timeout fallback)"""
    try:
        coral_host = config.get("edison", {}).get("coral", {}).get("host", "127.0.0.1")
        coral_port = config.get("edison", {}).get("coral", {}).get("port", 8808)
        coral_url = f"http://{coral_host}:{coral_port}/intent"
        
        response = requests.post(
            coral_url,
            json={"text": message},
            timeout=2  # Short timeout for auto mode
        )
        
        if response.status_code == 200:
            data = response.json()
            intent = data.get("intent", None)
            confidence = data.get("confidence", 0.0)
            logger.info(f"Coral intent: {intent} (confidence: {confidence:.2f})")
            return intent
    except Exception as e:
        logger.debug(f"Coral service unavailable (using fallback): {e}")
    
    return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("=" * 50)
    logger.info("Starting EDISON Core Service...")
    logger.info(f"Repo root: {REPO_ROOT}")
    load_config()
    load_llm_models()
    init_rag_system()
    init_search_tool()
    logger.info("EDISON Core Service ready")
    logger.info("=" * 50)
    
    yield
    
    # Shutdown
    logger.info("Shutting down EDISON Core Service...")

# Initialize FastAPI app
app = FastAPI(
    title="EDISON Core Service",
    description="Local LLM service with RAG capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/rag/search")
async def rag_search(request: dict):
    """Search RAG memory for relevant context"""
    if not rag_system or not rag_system.is_ready():
        return {"error": "RAG system not ready", "results": []}
    
    query = request.get("query", "")
    top_k = request.get("top_k", 5)
    results = rag_system.get_context(query, n_results=top_k)
    
    # Format results for JSON response
    formatted_results = []
    for item in results:
        if isinstance(item, tuple):
            text, metadata = item
            formatted_results.append({"text": text, "metadata": metadata})
        else:
            formatted_results.append({"text": item, "metadata": {}})
    
    return {"results": formatted_results, "count": len(formatted_results)}

@app.get("/rag/stats")
async def rag_stats():
    """Get RAG system statistics"""
    if not rag_system or not rag_system.is_ready():
        return {"error": "RAG system not ready", "ready": False}
    
    try:
        collection_info = rag_system.client.get_collection(rag_system.collection_name)
        return {
            "ready": True,
            "collection": rag_system.collection_name,
            "points_count": collection_info.points_count
        }
    except Exception as e:
        logger.error(f"Error getting RAG stats: {e}")
        return {"error": str(e), "ready": False}

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "edison-core",
        "models_loaded": {
            "fast_model": llm_fast is not None,
            "deep_model": llm_deep is not None
        },
        "qdrant_ready": rag_system is not None,
        "repo_root": str(REPO_ROOT)
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with mode support"""
    
    # Check if any model is loaded
    if not llm_fast and not llm_deep:
        raise HTTPException(
            status_code=503,
            detail=f"No LLM models loaded. Please place GGUF models in {REPO_ROOT}/models/llm/ directory. See logs for details."
        )
    
    # Determine which mode to use
    mode = request.mode
    if mode == "auto":
        # Try to get intent from coral service
        intent = get_intent_from_coral(request.message)
        
        # Map intent to mode (simple heuristic)
        if intent in ["analyze_image", "system_status", "help"]:
            mode = "chat"
        elif intent in ["generate_image", "text_to_image", "modify_workflow"]:
            mode = "reasoning"
        elif intent in ["agent", "code"]:
            mode = intent
        else:
            # Fallback heuristic based on message length
            if len(request.message) > 200 or any(word in request.message.lower() for word in ["explain", "how", "why", "analyze"]):
                mode = "reasoning"
            else:
                mode = "chat"
        
        logger.info(f"Auto mode selected: {mode}")
    
    # Select model based on mode
    use_deep = mode in ["reasoning", "agent", "code"]
    llm = llm_deep if (use_deep and llm_deep) else llm_fast
    model_name = "deep" if (use_deep and llm_deep) else "fast"
    
    if not llm:
        raise HTTPException(
            status_code=503,
            detail=f"Required model not available. Mode '{mode}' needs {'deep' if use_deep else 'fast'} model."
        )
    
    # Retrieve context from RAG if remember is enabled
    context_chunks = []
    if request.remember and rag_system and rag_system.is_ready():
        try:
            context_chunks = rag_system.get_context(request.message, n_results=2)
            if context_chunks:
                logger.info(f"Retrieved {len(context_chunks)} context chunks from RAG")
            else:
                logger.info("No relevant context found in RAG")
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
    
    # Build prompt
    system_prompt = build_system_prompt(mode, has_context=len(context_chunks) > 0)
    full_prompt = build_full_prompt(system_prompt, request.message, context_chunks)
    
    # Debug: Log the prompt being sent
    logger.info(f"Prompt length: {len(full_prompt)} chars")
    if context_chunks:
        logger.info(f"Context in prompt: {[c[0][:50] if isinstance(c, tuple) else c[:50] for c in context_chunks]}")
    
    # Generate response
    try:
        logger.info(f"Generating response with {model_name} model in {mode} mode")
        
        response = llm(
            full_prompt,
            max_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            stop=["User:", "Human:", "\n\n\n"],
            echo=False
        )
        
        assistant_response = response["choices"][0]["text"].strip()
        
        # Store in memory if requested
        if request.remember and rag_system:
            try:
                rag_system.add_documents(
                    documents=[f"User: {request.message}\nAssistant: {assistant_response}"],
                    metadatas=[{"type": "conversation", "mode": mode}]
                )
                logger.info("Conversation stored in memory")
            except Exception as e:
                logger.warning(f"Failed to store conversation: {e}")
        
        return ChatResponse(
            response=assistant_response,
            mode_used=mode,
            model_used=model_name
        )
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def web_search(request: SearchRequest):
    """Web search endpoint"""
    if not search_tool:
        raise HTTPException(
            status_code=503,
            detail="Web search tool not available"
        )
    
    try:
        results = search_tool.search(request.query, num_results=request.num_results)
        return SearchResponse(
            results=results,
            query=request.query
        )
    except Exception as e:
        logger.error(f"Error performing search: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

def build_system_prompt(mode: str, has_context: bool = False) -> str:
    """Build system prompt based on mode"""
    base = "You are EDISON, a helpful AI assistant."
    
    # Add instruction to use retrieved context if available
    if has_context:
        base += " CHECK CONTEXT FROM MEMORY for answers about the user and previous conversations. Use that info directly."
    
    prompts = {
        "chat": base + " Respond conversationally.",
        "reasoning": base + " Think step-by-step and explain clearly.",
        "agent": base + " Plan tasks, execute code, provide results. Break down complex tasks.",
        "code": base + " Generate complete, working code solutions."
    }
    
    return prompts.get(mode, base)

def build_full_prompt(system_prompt: str, user_message: str, context_chunks: list) -> str:
    """Build the complete prompt with context"""
    parts = [system_prompt, ""]
    
    # Extract key facts from context if available
    if context_chunks:
        facts = []
        for item in context_chunks:
            if isinstance(item, tuple):
                text, metadata = item
                # Extract key information from conversation text
                if "my name is" in text.lower():
                    # Try to extract the name
                    import re
                    match = re.search(r'my name is (\w+)', text.lower())
                    if match:
                        facts.append(f"The user's name is {match.group(1).title()}")
                facts.append(text)
        
        if facts:
            parts.append("FACTS FROM PREVIOUS CONVERSATIONS:")
            for fact in facts[:2]:  # Limit to 2 facts
                parts.append(f"- {fact}")
            parts.append("")
    
    parts.append(f"User: {user_message}")
    parts.append("Assistant:")
    
    return "\n".join(parts)

if __name__ == "__main__":
    import uvicorn
    
    host = config.get("edison", {}).get("core", {}).get("host", "127.0.0.1") if config else "127.0.0.1"
    port = config.get("edison", {}).get("core", {}).get("port", 8811) if config else 8811
    
    logger.info(f"Starting EDISON Core on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
    
    # Shutdown
