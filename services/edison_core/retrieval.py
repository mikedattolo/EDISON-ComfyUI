"""
Two-Stage Retrieval Pipeline for EDISON
Stage 1: Vector search (Qdrant)
Stage 2: Rerank by recency, relevance, confidence, type weighting, query intent
"""

import logging
import math
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    PROFILE = "profile"          # "what's my preferred language?"
    EPISODIC = "episodic"        # "what did we talk about yesterday?"
    SEMANTIC = "semantic"        # "how do I set up docker?"
    DEV_DOCS = "dev_docs"        # "FastAPI middleware docs"
    GENERAL = "general"


# Keywords for intent detection
_INTENT_KEYWORDS = {
    QueryIntent.PROFILE: [
        "my ", "i prefer", "i like", "i use", "my name", "my hardware",
        "my setup", "my project", "remember", "you know about me",
    ],
    QueryIntent.EPISODIC: [
        "yesterday", "last time", "earlier", "we discussed",
        "we talked about", "previous conversation", "before",
    ],
    QueryIntent.DEV_DOCS: [
        "docs", "documentation", "api reference", "how to use",
        "example of", "syntax for", "import", "function", "class",
        "method", "library", "package", "framework",
    ],
    QueryIntent.SEMANTIC: [
        "how do i", "how to", "steps to", "procedure", "workflow",
        "tutorial", "guide", "best practice",
    ],
}

# Type weights for reranking
TYPE_WEIGHTS = {
    "profile": 0.15,
    "episodic": 0.08,
    "semantic": 0.12,
    "fact": 0.12,
    "uploaded_document": 0.10,
    "message": 0.02,
}


def detect_query_intent(query: str) -> QueryIntent:
    """Detect the most likely intent of a retrieval query."""
    q = query.lower()
    scores = {}
    for intent, keywords in _INTENT_KEYWORDS.items():
        scores[intent] = sum(1 for kw in keywords if kw in q)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else QueryIntent.GENERAL


def rerank_results(
    results: List[Tuple[str, Dict[str, Any]]],
    query_intent: QueryIntent = QueryIntent.GENERAL,
    weights: Optional[Dict[str, float]] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Rerank retrieval results using multi-factor scoring.

    Args:
        results: List of (text, metadata) tuples from vector search
        query_intent: Detected query intent for type-boosting
        weights: Optional override for scoring weights

    Returns:
        Re-sorted list with rerank_score added to metadata
    """
    w = weights or {"relevance": 0.50, "recency": 0.20, "confidence": 0.15, "type": 0.15}
    current_time = time.time()
    scored = []

    for text, meta in results:
        base_score = meta.get("base_score", meta.get("score", 0.0))
        timestamp = meta.get("timestamp", 0)
        confidence = meta.get("confidence", 0.5)
        doc_type = meta.get("type", meta.get("memory_type", "message"))

        # 1. Relevance (from vector similarity)
        relevance = base_score

        # 2. Recency (exponential decay)
        if timestamp > 0:
            age_days = max(0, (current_time - timestamp) / 86400)
            half_life = 90 if doc_type in ("fact", "profile", "semantic") else 7
            recency = math.exp(-0.693 * age_days / half_life)
        else:
            recency = 0.0

        # 3. Confidence
        conf = min(1.0, max(0.0, confidence))

        # 4. Type weighting + intent boost
        type_w = TYPE_WEIGHTS.get(doc_type, 0.02)
        # Boost types that match query intent
        if query_intent == QueryIntent.PROFILE and doc_type == "profile":
            type_w += 0.10
        elif query_intent == QueryIntent.EPISODIC and doc_type == "episodic":
            type_w += 0.10
        elif query_intent == QueryIntent.SEMANTIC and doc_type == "semantic":
            type_w += 0.10
        elif query_intent == QueryIntent.DEV_DOCS and doc_type in ("dev_docs", "uploaded_document"):
            type_w += 0.10

        final = (
            w["relevance"] * relevance
            + w["recency"] * recency
            + w["confidence"] * conf
            + w["type"] * type_w
        )

        meta["rerank_score"] = round(final, 5)
        meta["rerank_components"] = {
            "relevance": round(relevance, 4),
            "recency": round(recency, 4),
            "confidence": round(conf, 4),
            "type_weight": round(type_w, 4),
            "query_intent": query_intent.value,
        }
        scored.append((text, meta))

    scored.sort(key=lambda x: x[1]["rerank_score"], reverse=True)

    if scored:
        logger.info(
            f"Reranked {len(scored)} results (intent={query_intent.value}, "
            f"top_score={scored[0][1]['rerank_score']:.4f})"
        )

    return scored
