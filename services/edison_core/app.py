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
llm_medium = None  # 32B model - fallback for deep mode
llm_deep = None
llm_vision = None  # VLM for image understanding
rag_system = None
search_tool = None
config = None

# Request/Response models
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    mode: Literal["auto", "chat", "reasoning", "agent", "code", "work"] = Field(
        default="auto", 
        description="Interaction mode"
    )
    remember: bool = Field(
        default=True, 
        description="Store conversation in memory"
    )
    images: Optional[list] = Field(
        default=None,
        description="Base64 encoded images for vision"
    )
    conversation_history: Optional[list] = Field(
        default=None,
        description="Recent conversation history for context (last 5 messages)"
    )

class ChatResponse(BaseModel):
    response: str
    mode_used: str
    model_used: str
    work_steps: Optional[list] = None
    context_used: Optional[int] = None
    search_results_count: Optional[int] = None

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
    vision_enabled: bool = False

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
    global llm_fast, llm_medium, llm_deep, llm_vision
    
    try:
        from llama_cpp import Llama
    except ImportError:
        logger.error("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
        return
    
    # Get model paths relative to repo root
    models_rel_path = config.get("edison", {}).get("core", {}).get("models_path", "models/llm")
    models_path = REPO_ROOT / models_rel_path
    
    fast_model_name = config.get("edison", {}).get("core", {}).get("fast_model", "qwen2.5-14b-instruct-q4_k_m.gguf")
    medium_model_name = config.get("edison", {}).get("core", {}).get("medium_model", "qwen2.5-32b-instruct-q4_k_m.gguf")
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
    
    # Try to load medium model (e.g., 32B - fallback for deep mode)
    medium_model_path = models_path / medium_model_name
    if medium_model_path.exists():
        try:
            logger.info(f"Loading medium model: {medium_model_path}")
            file_size_gb = medium_model_path.stat().st_size / (1024**3)
            logger.info(f"Medium model file size: {file_size_gb:.1f} GB")
            
            llm_medium = Llama(
                model_path=str(medium_model_path),
                n_ctx=4096,
                n_gpu_layers=-1,  # Use GPU
                tensor_split=[0.5, 0.25, 0.25],  # Split across 3 GPUs
                verbose=False
            )
            logger.info("✓ Medium model loaded successfully")
        except Exception as e:
            llm_medium = None
            logger.warning(f"Failed to load medium model: {e}")
    else:
        logger.info(f"Medium model not found at {medium_model_path} (optional - will use fast model as fallback)")
    
    # Try to load deep model (e.g., 72B)
    deep_model_path = models_path / deep_model_name
    if deep_model_path.exists():
        try:
            logger.info(f"Loading deep model: {deep_model_path}")
            # Check file size to warn about VRAM requirements
            file_size_gb = deep_model_path.stat().st_size / (1024**3)
            logger.info(f"Deep model file size: {file_size_gb:.1f} GB")
            
            llm_deep = Llama(
                model_path=str(deep_model_path),
                n_ctx=4096,  # Reduced from 8192 to save VRAM
                n_gpu_layers=-1,  # Use GPU
                tensor_split=[0.5, 0.25, 0.25],  # Split across 3 GPUs: 3090 (50%), 5060ti (25%), 3060 (25%)
                verbose=False
            )
            logger.info("✓ Deep model loaded successfully")
        except Exception as e:
            llm_deep = None  # Explicitly set to None to avoid cleanup errors
            logger.warning(f"Failed to load deep model (will fall back to medium or fast model): {e}")
            logger.info("💡 Tip: 72B models need ~42GB VRAM. Consider using 32B models or CPU offloading.")
    else:
        logger.warning(f"Deep model not found at {deep_model_path}")

    # Try to load vision model (VLM)
    vision_model_name = config.get("edison", {}).get("core", {}).get("vision_model")
    vision_clip_name = config.get("edison", {}).get("core", {}).get("vision_clip")
    
    if vision_model_name and vision_clip_name:
        vision_model_path = models_path / vision_model_name
        vision_clip_path = models_path / vision_clip_name
        
        if vision_model_path.exists() and vision_clip_path.exists():
            try:
                logger.info(f"Loading vision model: {vision_model_path}")
                logger.info(f"Loading CLIP projector: {vision_clip_path}")
                
                llm_vision = Llama(
                    model_path=str(vision_model_path),
                    clip_model_path=str(vision_clip_path),
                    n_ctx=4096,
                    n_gpu_layers=-1,
                    verbose=False
                )
                logger.info("✓ Vision model loaded successfully")
            except Exception as e:
                llm_vision = None
                logger.warning(f"Failed to load vision model: {e}")
        else:
            logger.info("Vision model or CLIP projector not found (optional - image understanding disabled)")
    else:
        logger.info("Vision model not configured (image understanding disabled)")
    
    if not llm_fast and not llm_medium and not llm_deep:
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
            "medium_model": llm_medium is not None,
            "deep_model": llm_deep is not None,
            "vision_model": llm_vision is not None
        },
        "vision_enabled": llm_vision is not None,
        "qdrant_ready": rag_system is not None,
        "repo_root": str(REPO_ROOT)
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with mode support"""
    
    # Check if any model is loaded
    if not llm_fast and not llm_medium and not llm_deep:
        raise HTTPException(
            status_code=503,
            detail=f"No LLM models loaded. Please place GGUF models in {REPO_ROOT}/models/llm/ directory. See logs for details."
        )
    
    # Auto-detect if this conversation should be remembered
    auto_remember = should_remember_conversation(request.message)
    remember = request.remember if request.remember is not None else auto_remember
    
    # Check if this is a recall request
    is_recall, recall_query = detect_recall_intent(request.message)
    
    logger.info(f"Auto-remember: {auto_remember}, Remember: {remember}, Recall request: {is_recall}")
    
    # Determine which mode to use
    mode = request.mode
    
    # Auto mode: Detect intent from message with enhanced patterns
    if mode == "auto":
        msg_lower = request.message.lower()
        
        # Try to get intent from coral service
        intent = get_intent_from_coral(request.message)
        
        # Enhanced fallback heuristics with better patterns
        if intent:
            logger.info(f"Intent from coral: {intent}")
            # Map coral intent to mode
            if intent in ["analyze_image", "system_status", "help"]:
                mode = "chat"
            elif intent in ["generate_image", "text_to_image", "modify_workflow"]:
                mode = "reasoning"
            elif intent in ["agent", "code"]:
                mode = intent
            else:
                mode = "chat"
        else:
            # Multi-step reasoning patterns
            reasoning_patterns = ["explain", "why", "how does", "what is", "analyze", "detail",
                                 "understand", "break down", "elaborate", "clarify", "reasoning",
                                 "think through", "step by step", "logic", "rationale"]
            
            # Code generation patterns
            code_patterns = ["code", "program", "function", "implement", "script", "write",
                            "create a", "build", "develop", "algorithm", "class", "method",
                            "debug", "fix this", "syntax", "refactor"]
            
            # Web search / agent patterns
            agent_patterns = ["search", "internet", "web", "find on", "lookup", "google",
                             "current", "latest", "news about", "information on",
                             "tell me about", "research", "browse"]
            
            # Work mode patterns (complex multi-step tasks)
            work_patterns = ["create a project", "build an app", "design a system", "plan",
                            "multi-step", "workflow", "organize", "manage",
                            "help me with", "work on", "collaborate", "break down this"]
            
            # Question patterns (recall from memory)
            recall_patterns = ["my name", "my favorite", "what did i", "do you remember",
                              "what's my", "tell me about myself", "who am i"]
            
            # Check patterns in priority order
            if any(pattern in msg_lower for pattern in work_patterns):
                mode = "work"
            elif any(pattern in msg_lower for pattern in recall_patterns):
                mode = "chat"  # Use chat for recall with memory
            elif any(pattern in msg_lower for pattern in agent_patterns):
                mode = "agent"
            elif any(pattern in msg_lower for pattern in code_patterns):
                mode = "code"
            elif any(pattern in msg_lower for pattern in reasoning_patterns):
                mode = "reasoning"
            else:
                # Check message length and complexity for reasoning
                words = msg_lower.split()
                if len(words) > 15 or '?' in request.message:
                    mode = "reasoning"
                else:
                    mode = "chat"
    
    # Keep work mode separate for special handling
    original_mode = mode
    if mode == "work":
        logger.info(f"Work mode activated for complex task")
        # Work mode will use reasoning capabilities but with special handling
        use_deep_mode = True
    else:
        use_deep_mode = mode in ["reasoning", "agent", "code"]
        
    logger.info(f"Mode selected: {mode} (from {request.mode})")
    
    if use_deep_mode:
        # Try deep first, then medium, then fast
        if llm_deep:
            llm = llm_deep
            model_name = "deep"
        elif llm_medium:
            llm = llm_medium
            model_name = "medium"
        else:
            llm = llm_fast
            model_name = "fast"
    else:
        # Chat mode: use fast model
        llm = llm_fast if llm_fast else (llm_medium if llm_medium else llm_deep)
        model_name = "fast" if llm_fast else ("medium" if llm_medium else "deep")
    
    if not llm:
        raise HTTPException(
            status_code=503,
            detail=f"No suitable model available for mode '{mode}'."
        )
    
    # Check if this is a vision request (has images)
    has_images = request.images and len(request.images) > 0
    if has_images:
        if not llm_vision:
            raise HTTPException(
                status_code=503,
                detail="Vision model not loaded. Please download LLaVA model to enable image understanding."
            )
        # Use vision model for image understanding
        llm = llm_vision
        model_name = "vision"
        logger.info("Using vision model for image understanding")
    
    # Retrieve context from RAG - always check for recall requests or follow-ups
    context_chunks = []
    if rag_system and rag_system.is_ready():
        try:
            # Handle explicit recall requests
            if is_recall:
                logger.info(f"Recall request detected, searching for: {recall_query}")
                # Do extensive search across all conversations
                recall_chunks = rag_system.get_context(recall_query, n_results=5)
                if recall_chunks:
                    context_chunks.extend(recall_chunks)
                    logger.info(f"Retrieved {len(recall_chunks)} chunks for recall request")
                
                # Also search with original message
                additional_chunks = rag_system.get_context(request.message, n_results=3)
                if additional_chunks:
                    context_chunks.extend(additional_chunks)
            
            # First, check if this is a follow-up question (uses pronouns or context words)
            msg_lower = request.message.lower()
            is_followup = any(word in msg_lower for word in [
                'that', 'it', 'this', 'the book', 'the page', 'her', 'his', 'their',
                'what page', 'which', 'where in', 'from that', 'about that'
            ])
            
            # If it's a follow-up and we have conversation history, use that context
            if is_followup and request.conversation_history and len(request.conversation_history) > 0:
                # Get the last few messages from conversation history
                recent_context = []
                for msg in request.conversation_history[-3:]:
                    if msg.get('role') == 'user':
                        recent_context.append(msg.get('content', ''))
                    elif msg.get('role') == 'assistant':
                        # Include last assistant response for context
                        recent_context.append(msg.get('content', ''))
                
                # Search RAG using recent conversation context as queries
                logger.info(f"Follow-up detected, searching with conversation context")
                for context_msg in recent_context:
                    if len(context_msg) > 10:  # Skip very short messages
                        chunks = rag_system.get_context(context_msg[:200], n_results=2)
                        if chunks:
                            context_chunks.extend(chunks[:1])  # Take top result from each
                
                # Also search with current message
                chunks = rag_system.get_context(request.message, n_results=2)
                if chunks:
                    context_chunks.extend(chunks)
            
            # Expand query to get better context - extract key terms from questions
            search_queries = [request.message]
            
            # Detect question patterns and extract key terms
            import re
            
            # Pattern: "what is my X" or "what's my X" -> extract X
            what_match = re.search(r"what(?:'s| is) (?:my|your) (\w+(?:\s+\w+)?)", msg_lower)
            if what_match:
                topic = what_match.group(1).strip()
                # Search for STATEMENTS about this topic, not questions
                if "name" in topic:
                    search_queries.extend(["my name is", "mike", "called", "name is"])
                elif "color" in topic or "colour" in topic:
                    search_queries.extend(["my favorite color", "color is", "blue", "red", "green"])
                elif "age" in topic:
                    search_queries.extend(["I am", "years old", "age is"])
                else:
                    # Generic: search for "my [topic] is" pattern
                    search_queries.append(f"my {topic} is")
                    search_queries.append(f"I {topic}")
            
            # Pattern: "tell me about X" -> extract X  
            tell_match = re.search(r"tell me about (.+?)(?:\?|$)", msg_lower)
            if tell_match:
                topic = tell_match.group(1).strip()
                # Search for statements about this topic
                if "myself" in topic or "me" == topic:
                    search_queries.extend(["my name is", "I am", "I like", "my favorite", "I enjoy"])
                else:
                    search_queries.append(f"about {topic}")
            
            # Pattern: "who am i" -> search for identity statements
            if "who am i" in msg_lower or "who i am" in msg_lower:
                search_queries.extend(["my name is", "I am", "my name"])
            
            # Remove duplicates while preserving order
            seen = set()
            search_queries = [q for q in search_queries if not (q in seen or seen.add(q))]
            
            logger.info(f"Expanded search queries: {search_queries}")
            
            # Get results from each query separately and take best from each
            all_chunks = []
            for query in search_queries[:5]:  # Limit to 5 queries max
                chunks = rag_system.get_context(query, n_results=2)
                if chunks:
                    # Log what we found
                    text_preview = chunks[0][0][:80] if isinstance(chunks[0], tuple) else chunks[0][:80]
                    logger.info(f"Query '{query}' top result: {text_preview}")
                    # Take top 1 from this query
                    all_chunks.append(chunks[0])
            
            # Deduplicate and prioritize informative chunks
            seen_texts = set()
            informative_chunks = []  # Contains statements like "my name is"
            question_chunks = []      # Contains questions
            
            for chunk in all_chunks:
                text = chunk[0] if isinstance(chunk, tuple) else chunk
                if text not in seen_texts:
                    seen_texts.add(text)
                    
                    # Check if this chunk has actual information vs just questions
                    text_lower = text.lower()
                    has_info = any(phrase in text_lower for phrase in [
                        "my name is", "i am", "i'm", "my favorite", "i like", 
                        "i enjoy", "i work", "i live", "called", "years old"
                    ])
                    
                    if has_info:
                        informative_chunks.append(chunk)
                    else:
                        question_chunks.append(chunk)
            
            # Prioritize informative chunks, fall back to questions if needed
            context_chunks = informative_chunks[:2]
            if len(context_chunks) < 2:
                context_chunks.extend(question_chunks[:2 - len(context_chunks)])
            
            logger.info(f"Found {len(informative_chunks)} informative chunks, {len(question_chunks)} question chunks")
            
            if context_chunks:
                logger.info(f"Retrieved {len(context_chunks)} context chunks from RAG")
            else:
                logger.info("No relevant context found in RAG")
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
    
    # Agent mode: Check if web search is requested
    search_results = []
    if mode in ["agent", "work"] and search_tool:
        msg_lower = request.message.lower()
        search_keywords = ["search", "internet", "web", "online", "news", "lookup", "find on", "google", "browse"]
        if any(keyword in msg_lower for keyword in search_keywords):
            try:
                # Extract search query from message
                import re
                # Try to extract what to search for
                search_query = request.message
                
                # Remove common prefixes
                for prefix in ["search the internet for", "search the internet about", "search for", "search about", "search", "look up", "find on the internet", "find", "google", "tell me about"]:
                    if prefix in msg_lower:
                        search_query = re.sub(rf"^.*?{prefix}\s+", "", request.message, flags=re.IGNORECASE)
                        break
                
                # Remove common suffixes like "and tell me about it"
                suffixes_to_remove = [
                    r"\s+and tell me.*",
                    r"\s+and give.*",
                    r"\s+and provide.*",
                    r"\s+and let me know.*"
                ]
                for suffix in suffixes_to_remove:
                    search_query = re.sub(suffix, "", search_query, flags=re.IGNORECASE)
                
                # Remove trailing punctuation
                search_query = search_query.strip().rstrip('?.!')
                
                # If query is too long, extract key terms
                if len(search_query.split()) > 8:
                    # Try to get the main topic (first few meaningful words)
                    words = search_query.split()[:5]
                    search_query = " ".join(words)
                
                logger.info(f"Performing web search for: {search_query}")
                results = search_tool.search(search_query, num_results=3)
                search_results = results
                logger.info(f"Found {len(results)} search results")
            except Exception as e:
                logger.error(f"Web search failed: {e}")
    
    # Build prompt
    system_prompt = build_system_prompt(mode, has_context=len(context_chunks) > 0, has_search=len(search_results) > 0)
    
    # Work mode: Break down task into actionable steps
    work_steps = []
    if original_mode == "work" and not has_images:
        try:
            task_analysis_prompt = f"""You are a task planning assistant. Break down this request into 3-7 clear, actionable steps.

Task: {request.message}

Provide a numbered list of specific steps. Be concise and action-oriented.

Steps:"""
            
            task_response = llm(
                task_analysis_prompt,
                max_tokens=400,
                temperature=0.3,
                stop=["Task:", "\\n\\n\\n"],
                echo=False
            )
            
            task_breakdown = task_response["choices"][0]["text"].strip()
            # Parse numbered steps
            import re
            work_steps = [s.strip() for s in re.findall(r'\\d+\\.\\s*(.+?)(?=\\n\\d+\\.|$)', task_breakdown, re.DOTALL)]
            work_steps = [s.replace('\\n', ' ').strip() for s in work_steps if s.strip()]
            
            if work_steps:
                logger.info(f"Task broken down into {len(work_steps)} steps")
                # Add steps to prompt context
                steps_text = "\\n".join([f"{i+1}. {step}" for i, step in enumerate(work_steps)])
                system_prompt += f"\\n\\nTask Plan:\\n{steps_text}\\n\\nFollow these steps to complete the task thoroughly."
            
        except Exception as e:
            logger.warning(f"Task breakdown failed: {e}")
    
    # For vision requests, handle images differently
    if has_images:
        # Vision models use different format with images
        full_prompt = request.message
        if context_chunks:
            context_text = "\n\n".join([chunk[0] if isinstance(chunk, tuple) else chunk for chunk in context_chunks])
            full_prompt = f"Context: {context_text}\n\n{full_prompt}"
    else:
        full_prompt = build_full_prompt(system_prompt, request.message, context_chunks, search_results, request.conversation_history)
    
    # Debug: Log the prompt being sent
    logger.info(f"Prompt length: {len(full_prompt)} chars")
    if context_chunks:
        logger.info(f"Context in prompt: {[c[0][:50] if isinstance(c, tuple) else c[:50] for c in context_chunks]}")
        # Log first 500 chars of actual prompt to see formatting
        logger.info(f"Prompt preview: {full_prompt[:500]}")
    
    # Generate response
    try:
        logger.info(f"Generating response with {model_name} model in {mode} mode")
        
        if has_images:
            # Vision model with images - llama-cpp-python LLaVA format
            logger.info(f"Processing {len(request.images)} images with vision model")
            
            # For llama-cpp-python with LLaVA, we need to use data_uri format directly
            # The API expects image data as base64 strings in the message content
            import base64
            import io
            from PIL import Image
            
            # Process each image
            image_data_list = []
            for img_b64 in request.images:
                if isinstance(img_b64, str):
                    # Remove data URL prefix if present (data:image/...;base64,)
                    if ',' in img_b64:
                        img_b64 = img_b64.split(',', 1)[1]
                    
                    # Add to data list with proper format
                    image_data_list.append(f"data:image/jpeg;base64,{img_b64}")
            
            logger.info(f"Vision request with {len(image_data_list)} images")
            logger.info(f"Prompt: {full_prompt}")
            logger.info(f"Image data length: {len(image_data_list[0][:100])}..." if image_data_list else "No images")
            
            # Try the multimodal content format
            content = [
                {"type": "text", "text": full_prompt}
            ]
            
            for img_data in image_data_list:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img_data}
                })
            
            logger.info(f"Content structure: {len(content)} parts (1 text + {len(image_data_list)} images)")
            
            try:
                response = llm.create_chat_completion(
                    messages=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    max_tokens=2048,
                    temperature=0.7
                )
                assistant_response = response["choices"][0]["message"]["content"]
                logger.info(f"Vision response generated: {assistant_response[:100]}...")
            except Exception as e:
                logger.error(f"Vision model error: {e}")
                logger.error(f"Trying fallback method...")
                
                # Fallback: try simple text prompt (vision model should still work as text model)
                response = llm(
                    f"[Image provided] {full_prompt}",
                    max_tokens=2048,
                    temperature=0.7
                )
                assistant_response = response["choices"][0]["text"].strip()
                logger.warning("Vision model used in text-only mode - images not processed")
        else:
            # Text-only model
            max_tokens = 3072 if original_mode == "work" else 2048  # More tokens for work mode
            response = llm(
                full_prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                frequency_penalty=0.3,  # Reduce repetition
                presence_penalty=0.2,   # Encourage new topics
                repeat_penalty=1.1,     # Penalize repeated tokens
                stop=["User:", "Human:", "\n\n\n", "Would you like to specify", "Please specify"],
                echo=False
            )
            
            assistant_response = response["choices"][0]["text"].strip()
        
        # Store in memory if auto-detected or requested
        if remember and rag_system:
            try:
                # Extract facts from conversation
                facts_extracted = extract_facts_from_conversation(request.message, assistant_response)
                
                # Store full conversation
                rag_system.add_documents(
                    documents=[f"User: {request.message}\nAssistant: {assistant_response}"],
                    metadatas=[{"type": "conversation", "mode": mode}]
                )
                
                # Store extracted facts separately for better retrieval
                if facts_extracted:
                    for fact in facts_extracted:
                        rag_system.add_documents(
                            documents=[fact],
                            metadatas=[{"type": "fact", "source": "conversation"}]
                        )
                    logger.info(f"Conversation + {len(facts_extracted)} facts stored in memory")
                else:
                    logger.info("Conversation stored in memory")
            except Exception as e:
                logger.warning(f"Failed to store conversation: {e}")
        
        # Build response with work mode metadata
        response_data = {
            "response": assistant_response,
            "mode_used": original_mode,
            "model_used": model_name
        }
        
        # Add work mode specific data
        if original_mode == "work" and work_steps:
            response_data["work_steps"] = work_steps
            response_data["context_used"] = len(context_chunks)
            response_data["search_results_count"] = len(search_results) if search_results else 0
        
        return ChatResponse(**response_data)
        
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

def build_system_prompt(mode: str, has_context: bool = False, has_search: bool = False) -> str:
    """Build system prompt based on mode"""
    base = "You are EDISON, a helpful AI assistant."
    
    # Add instruction to use retrieved context if available
    if has_context:
        base += " Use information from previous conversations to answer questions about the user."
    
    # Add instruction about web search results - make it stronger
    if has_search:
        base += " IMPORTANT: Web search results are provided below. You MUST use these search results to answer the user's question. Cite specific information from the search results and include the source titles. Do not make up information - only use what's in the search results."
    
    # Add conversation awareness instruction
    base += " Pay attention to the conversation history - if the user asks a follow-up question using pronouns like 'that', 'it', 'this', 'her', or refers to something previously discussed, use the conversation context to understand what they're referring to. Be conversationally aware and maintain context across messages."
    
    prompts = {
        "chat": base + " Respond conversationally.",
        "reasoning": base + " Think step-by-step and explain clearly.",
        "agent": base + " You can search the web for current information. Provide detailed, accurate answers based on search results.",
        "code": base + " Generate complete, working code solutions with explanations.",
        "work": base + " You are helping with a complex multi-step task. Follow the task plan provided, work through each step methodically, and provide comprehensive results. Be thorough and detail-oriented."
    }
    
    return prompts.get(mode, base)

def build_full_prompt(system_prompt: str, user_message: str, context_chunks: list, search_results: list = None, conversation_history: list = None) -> str:
    """Build the complete prompt with context, search results, and conversation history"""
    parts = [system_prompt, ""]
    
    # Add recent conversation history for context
    if conversation_history and len(conversation_history) > 0:
        parts.append("RECENT CONVERSATION:")
        for msg in conversation_history[-5:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("")
    
    # Add web search results if available
    if search_results:
        parts.append("WEB SEARCH RESULTS:")
        for i, result in enumerate(search_results[:3], 1):
            parts.append(f"{i}. {result.get('title', 'No title')}")
            parts.append(f"   {result.get('snippet', 'No description')}")
            parts.append(f"   Source: {result.get('url', 'No URL')}")
            parts.append("")
    
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
                        extracted_fact = f"The user's name is {match.group(1).title()}"
                        facts.append(extracted_fact)
                        logger.info(f"Extracted fact: {extracted_fact}")
                facts.append(text)
        
        if facts:
            parts.append("FACTS FROM PREVIOUS CONVERSATIONS:")
            for fact in facts[:2]:  # Limit to 2 facts
                parts.append(f"- {fact}")
            parts.append("")
    
    parts.append(f"User: {user_message}")
    parts.append("Assistant:")
    
    return "\n".join(parts)

# ============================================
# NEW FEATURES API ENDPOINTS
# ============================================

@app.post("/upload-document")
async def upload_document(request: dict):
    """Handle document upload and text extraction"""
    try:
        file_name = request.get('name', 'unknown')
        file_content = request.get('content', '')
        
        # Store in RAG if available
        if rag_system:
            rag_system.add_documents(
                documents=[file_content],
                metadatas=[{"type": "uploaded_document", "filename": file_name}]
            )
            logger.info(f"Document {file_name} stored in RAG")
        
        return {
            "status": "success",
            "filename": file_name,
            "characters": len(file_content)
        }
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-title")
async def generate_title(request: dict):
    """Generate a smart title for a chat based on first message"""
    try:
        message = request.get('message', '')
        
        if not llm_fast:
            # Fallback to simple title
            title = message[:40]
            return {"title": title}
        
        # Use LLM to generate concise title
        prompt = f"Generate a concise 3-5 word title for this message:\n\n{message}\n\nTitle:"
        
        response = llm_fast(
            prompt,
            max_tokens=20,
            temperature=0.5,
            stop=["\n"],
            echo=False
        )
        
        title = response["choices"][0]["text"].strip()
        return {"title": title}
        
    except Exception as e:
        logger.error(f"Error generating title: {e}")
        return {"title": message[:40]}

@app.get("/system/stats")
async def system_stats():
    """Get system hardware statistics"""
    import psutil
    try:
        # CPU stats
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory stats
        mem = psutil.virtual_memory()
        ram_used_gb = mem.used / (1024 ** 3)
        ram_total_gb = mem.total / (1024 ** 3)
        
        # Temperature (try to get CPU temp)
        temp_c = 0
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Try common sensor names
                for name in ['coretemp', 'cpu_thermal', 'k10temp']:
                    if name in temps:
                        temp_c = temps[name][0].current
                        break
        except:
            temp_c = 0
        
        # GPU stats (basic - could be enhanced with pynvml)
        gpu_percent = 0
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                    capture_output=True, text=True, timeout=1)
            if result.returncode == 0:
                gpu_percent = float(result.stdout.strip().split('\n')[0])
        except:
            gpu_percent = 0
        
        return {
            "cpu_percent": cpu_percent,
            "gpu_percent": gpu_percent,
            "ram_used_gb": ram_used_gb,
            "ram_total_gb": ram_total_gb,
            "temp_c": temp_c if temp_c > 0 else 50  # Default temp if unavailable
        }
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def should_remember_conversation(message: str) -> bool:
    """Determine if conversation should be stored based on content intent"""
    msg_lower = message.lower()
    
    # Personal information patterns (should remember)
    remember_patterns = [
        # Identity
        r"my name is", r"i'm called", r"call me", r"i am",
        # Preferences
        r"my favorite", r"i like", r"i love", r"i enjoy", r"i prefer",
        r"i hate", r"i dislike", r"i don't like",
        # Personal facts
        r"i live in", r"i'm from", r"i work", r"my job", r"my age",
        r"my birthday", r"i was born", r"my hobby", r"my hobbies",
        # Goals and plans
        r"i want to", r"i'm planning", r"i need to", r"my goal",
        r"i'm working on", r"i'm learning",
        # Relationships
        r"my wife", r"my husband", r"my partner", r"my friend",
        r"my family", r"my children", r"my parents",
        # Context-rich statements
        r"remind me", r"remember that", r"don't forget", r"keep in mind",
    ]
    
    # Check if message contains memorable content
    import re
    for pattern in remember_patterns:
        if re.search(pattern, msg_lower):
            return True
    
    # Don't remember simple queries or commands without personal context
    query_patterns = [
        r"^what is", r"^what's", r"^how do", r"^how to",
        r"^explain", r"^tell me about", r"^search for",
        r"^show me", r"^give me", r"^can you"
    ]
    
    for pattern in query_patterns:
        if re.match(pattern, msg_lower):
            # It's a query - only remember if it's about personal topics
            if any(word in msg_lower for word in ["my", "i", "myself", "me"]):
                return True
            return False
    
    # Default: remember substantive conversations (not too short)
    words = message.split()
    if len(words) > 8:
        return True
    
    return False

def detect_recall_intent(message: str) -> tuple[bool, str]:
    """Detect if user is asking to recall previous conversations
    Returns: (is_recall_request, search_query)
    """
    msg_lower = message.lower()
    
    # Explicit recall patterns
    recall_patterns = [
        (r"what did (we|i) (talk|discuss|say) about (.+)", 3),
        (r"recall (our|my) conversation about (.+)", 2),
        (r"remember when (we|i) (talked|discussed) about (.+)", 3),
        (r"search (my|our) (conversations|chats|history) for (.+)", 3),
        (r"find (the conversation|when we talked) about (.+)", 2),
        (r"what did you (say|tell me) about (.+)", 2),
        (r"did (we|i) (discuss|talk about|mention) (.+)", 3),
    ]
    
    import re
    for pattern, group_idx in recall_patterns:
        match = re.search(pattern, msg_lower)
        if match:
            search_query = match.group(group_idx) if match.lastindex >= group_idx else message
            return (True, search_query)
    
    # Simple recall keywords
    recall_keywords = [
        "recall", "remember when", "what did we", "our conversation",
        "search my chats", "find in history", "previous conversation",
        "earlier we talked", "you mentioned", "we discussed"
    ]
    
    if any(keyword in msg_lower for keyword in recall_keywords):
        # Extract the topic after the recall keyword
        for keyword in recall_keywords:
            if keyword in msg_lower:
                parts = msg_lower.split(keyword, 1)
                if len(parts) > 1:
                    search_query = parts[1].strip()
                    # Clean up common words
                    search_query = re.sub(r'^(about|that|the)\s+', '', search_query)
                    return (True, search_query if search_query else message)
        return (True, message)
    
    return (False, "")

def extract_facts_from_conversation(user_message: str, assistant_response: str) -> List[str]:
    """Extract factual statements from conversation for better memory"""
    facts = []
    
    # Extract personal information patterns
    import re
    
    # Name patterns
    name_patterns = [
        r"my name is (\w+)",
        r"i'm (\w+)",
        r"i am (\w+)",
        r"call me (\w+)"
    ]
    
    combined_text = f"{user_message.lower()} {assistant_response.lower()}"
    
    for pattern in name_patterns:
        matches = re.findall(pattern, combined_text)
        for match in matches:
            if len(match) > 1 and match.isalpha():  # Valid name
                facts.append(f"The user's name is {match.title()}.")
    
    # Favorite/preference patterns
    pref_patterns = [
        (r"my favorite (\w+) is ([^.!?]+)", "The user's favorite {} is {}."),
        (r"i like (\w+)", "The user likes {}."),
        (r"i love (\w+)", "The user loves {}."),
        (r"i enjoy (\w+)", "The user enjoys {}."),
    ]
    
    for pattern, template in pref_patterns:
        matches = re.findall(pattern, user_message.lower())
        for match in matches:
            if isinstance(match, tuple):
                fact = template.format(*match)
            else:
                fact = template.format(match)
            facts.append(fact.strip())
    
    # Age pattern
    age_match = re.search(r"i am (\d+) years old|i'm (\d+)", user_message.lower())
    if age_match:
        age = age_match.group(1) or age_match.group(2)
        facts.append(f"The user is {age} years old.")
    
    # Location patterns
    location_patterns = [
        r"i live in ([^.!?]+)",
        r"i'm from ([^.!?]+)",
        r"i am from ([^.!?]+)"
    ]
    
    for pattern in location_patterns:
        match = re.search(pattern, user_message.lower())
        if match:
            location = match.group(1).strip()
            facts.append(f"The user lives in {location.title()}.")
    
    # Remove duplicates
    facts = list(dict.fromkeys(facts))
    
    return facts[:5]  # Limit to 5 facts per conversation

if __name__ == "__main__":
    import uvicorn
    
    host = config.get("edison", {}).get("core", {}).get("host", "127.0.0.1") if config else "127.0.0.1"
    port = config.get("edison", {}).get("core", {}).get("port", 8811) if config else 8811
    
    logger.info(f"Starting EDISON Core on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
    
    # Shutdown
