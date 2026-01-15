"""
EDISON Coral Service - Intent Classification and Edge TPU Inference
FastAPI server with heuristic intent routing and optional TPU support
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import yaml
from pathlib import Path
import logging
import re
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
config = None
tpu_interpreter = None
tpu_available = False
intent_labels = []

# Request/Response models
class IntentRequest(BaseModel):
    text: str = Field(..., description="User input text for intent classification")

class IntentResponse(BaseModel):
    intent: str
    confidence: float
    method: str  # "heuristic" or "tpu"

class HealthResponse(BaseModel):
    status: str
    service: str
    tpu_available: bool
    tpu_model_loaded: bool
    intent_classifier_method: str

def load_config():
    """Load EDISON configuration"""
    global config
    try:
        config_path = Path(__file__).parent.parent.parent / "config" / "edison.yaml"
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
                logger.info("Configuration loaded successfully")
        else:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            config = {
                "edison": {
                    "coral": {
                        "enabled": True,
                        "host": "127.0.0.1",
                        "port": 8808,
                        "device": "/dev/apex_0"
                    }
                }
            }
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        config = {}

def load_intent_labels():
    """Load intent labels from config"""
    global intent_labels
    try:
        labels_path = Path(__file__).parent.parent.parent / "config" / "intent_labels.txt"
        if labels_path.exists():
            with open(labels_path) as f:
                intent_labels = [line.strip() for line in f.readlines() if line.strip()]
            logger.info(f"Loaded {len(intent_labels)} intent labels")
        else:
            logger.warning("Intent labels file not found, using defaults")
            intent_labels = [
                "generate_image", "modify_workflow", "analyze_image",
                "text_to_image", "image_to_image", "upscale",
                "style_transfer", "help", "system_status"
            ]
    except Exception as e:
        logger.error(f"Error loading intent labels: {e}")
        intent_labels = ["help"]

def check_tpu_availability():
    """Check if Coral TPU is available"""
    global tpu_available
    try:
        from pycoral.utils import edgetpu
        devices = edgetpu.list_edge_tpus()
        tpu_available = len(devices) > 0
        if tpu_available:
            logger.info(f"Coral TPU detected: {devices}")
        else:
            logger.info("No Coral TPU detected, using heuristic intent classification only")
    except ImportError:
        logger.warning("pycoral not installed, TPU support disabled")
        tpu_available = False
    except Exception as e:
        logger.warning(f"Error checking TPU availability: {e}")
        tpu_available = False

def initialize_tpu_model():
    """Initialize Edge TPU model if available (optional)"""
    global tpu_interpreter
    
    if not tpu_available:
        logger.info("Skipping TPU model initialization (TPU not available)")
        return
    
    try:
        from pycoral.utils import edgetpu
        
        # Look for intent classifier model (optional)
        intent_model_path = Path("models/coral/intent_classifier_edgetpu.tflite")
        
        if intent_model_path.exists():
            logger.info(f"Loading TPU intent classifier: {intent_model_path}")
            tpu_interpreter = edgetpu.make_interpreter(str(intent_model_path))
            tpu_interpreter.allocate_tensors()
            logger.info("TPU intent classifier loaded successfully")
        else:
            logger.info(f"TPU intent classifier not found at {intent_model_path}")
            logger.info("Using heuristic intent classification (V1)")
    
    except Exception as e:
        logger.warning(f"Failed to initialize TPU model: {e}")
        tpu_interpreter = None

def heuristic_intent_classification(text: str) -> tuple[str, float]:
    """
    V1: Heuristic intent classification using keyword matching
    Returns (intent, confidence)
    """
    text_lower = text.lower()
    
    # Define keyword patterns for each intent
    patterns = {
        "generate_image": [
            r"\b(generate|create|make|produce)\b.*\b(image|picture|photo|art)\b",
            r"\b(draw|paint|render)\b",
        ],
        "text_to_image": [
            r"\b(text to image|txt2img|t2i)\b",
            r"\bfrom.*text\b",
        ],
        "image_to_image": [
            r"\b(image to image|img2img|i2i)\b",
            r"\b(transform|convert|change).*image\b",
        ],
        "modify_workflow": [
            r"\b(modify|change|update|edit|adjust)\b.*\b(workflow|pipeline)\b",
            r"\b(add|remove|delete).*node\b",
        ],
        "analyze_image": [
            r"\b(analyze|examine|inspect|describe|explain)\b.*\b(image|picture)\b",
            r"\bwhat.*in.*image\b",
        ],
        "upscale": [
            r"\b(upscale|enlarge|increase.*resolution|scale.*up)\b",
            r"\bhigher.*resolution\b",
        ],
        "style_transfer": [
            r"\b(style.*transfer|apply.*style|stylize)\b",
            r"\bin.*style.*of\b",
        ],
        "control_net": [
            r"\bcontrol.*net\b",
            r"\b(pose|depth|canny).*control\b",
        ],
        "lora_apply": [
            r"\b(lora|apply.*lora|use.*lora)\b",
        ],
        "workflow_save": [
            r"\b(save|export).*workflow\b",
        ],
        "workflow_load": [
            r"\b(load|import|open).*workflow\b",
        ],
        "system_status": [
            r"\b(status|state|health|info|information)\b",
            r"\bhow.*doing\b",
        ],
        "help": [
            r"\b(help|assist|support|how.*do|what.*can)\b",
            r"\bexplain\b",
        ],
    }
    
    # Score each intent
    scores = {}
    for intent, pattern_list in patterns.items():
        score = 0
        for pattern in pattern_list:
            if re.search(pattern, text_lower):
                score += 1
        if score > 0:
            scores[intent] = score
    
    # If no matches, default to help
    if not scores:
        return "help", 0.3
    
    # Return highest scoring intent
    best_intent = max(scores, key=scores.get)
    max_score = scores[best_intent]
    
    # Convert score to confidence (normalize)
    confidence = min(0.5 + (max_score * 0.2), 0.95)
    
    return best_intent, confidence

def tpu_intent_classification(text: str) -> tuple[str, float]:
    """
    V2: TPU-based intent classification using EdgeTPU model
    Returns (intent, confidence)
    """
    if not tpu_interpreter:
        logger.warning("TPU classifier called but not available, falling back to heuristic")
        return heuristic_intent_classification(text)
    
    try:
        # TODO: Implement actual TPU inference when model is available
        # This would involve:
        # 1. Tokenize text
        # 2. Create input tensor
        # 3. Run inference
        # 4. Get output and map to intent label
        
        logger.info("TPU intent classification not yet implemented, using heuristic")
        return heuristic_intent_classification(text)
    
    except Exception as e:
        logger.error(f"TPU classification error: {e}, falling back to heuristic")
        return heuristic_intent_classification(text)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting EDISON Coral Service...")
    load_config()
    load_intent_labels()
    check_tpu_availability()
    initialize_tpu_model()
    logger.info("EDISON Coral Service ready")
    
    yield
    
    # Shutdown
    logger.info("Shutting down EDISON Coral Service...")

# Initialize FastAPI app
app = FastAPI(
    title="EDISON Coral Service",
    description="Intent classification with optional Edge TPU support",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint with TPU status"""
    method = "heuristic"
    if tpu_interpreter is not None:
        method = "tpu"
    elif tpu_available:
        method = "tpu_available_not_loaded"
    
    return {
        "status": "healthy",
        "service": "edison-coral",
        "tpu_available": tpu_available,
        "tpu_model_loaded": tpu_interpreter is not None,
        "intent_classifier_method": method
    }

@app.post("/intent", response_model=IntentResponse)
async def classify_intent(request: IntentRequest):
    """
    Classify user intent from text
    Uses TPU model if available, otherwise falls back to heuristic classification
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    text = request.text.strip()
    
    # Choose classification method
    if tpu_interpreter is not None:
        intent, confidence = tpu_intent_classification(text)
        method = "tpu"
    else:
        intent, confidence = heuristic_intent_classification(text)
        method = "heuristic"
    
    logger.info(f"Intent classified: '{intent}' (confidence: {confidence:.2f}, method: {method})")
    
    return IntentResponse(
        intent=intent,
        confidence=confidence,
        method=method
    )

@app.get("/intents")
async def list_intents():
    """List all available intent labels"""
    return {
        "intents": intent_labels,
        "count": len(intent_labels)
    }

if __name__ == "__main__":
    import uvicorn
    
    host = config.get("edison", {}).get("coral", {}).get("host", "127.0.0.1") if config else "127.0.0.1"
    port = config.get("edison", {}).get("coral", {}).get("port", 8808) if config else 8808
    
    logger.info(f"Starting EDISON Coral Service on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
