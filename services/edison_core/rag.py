"""
Retrieval-Augmented Generation System for EDISON
Uses Qdrant for vector storage and sentence-transformers for embeddings
"""

from typing import List, Dict, Optional
from pathlib import Path
import logging
import uuid
import time
import threading

logger = logging.getLogger(__name__)

class RAGSystem:
    """RAG system for context retrieval using Qdrant and sentence-transformers"""
    
    def __init__(self, storage_path: str = "./qdrant_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.client = None
        self.encoder = None
        self.collection_name = "edison_memory"
        self.vector_size = 384  # all-MiniLM-L6-v2 dimension
        self._lock = threading.Lock()  # Protects encoder and client operations
        
        self._initialize_encoder()
        self._initialize_qdrant()
    
    def _initialize_encoder(self):
        """Initialize sentence transformer for embeddings"""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence transformer model: all-MiniLM-L6-v2")
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✓ Sentence transformer loaded successfully")
        except ImportError as e:
            logger.error(f"sentence-transformers package not installed: {e}")
            logger.error("Install with: pip install sentence-transformers")
            self.encoder = None
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {type(e).__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.encoder = None
    
    def _initialize_qdrant(self):
        """Initialize Qdrant client with local storage"""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
            
            logger.info(f"Initializing Qdrant at {self.storage_path}")
            self.client = QdrantClient(path=str(self.storage_path))
            
            # Create collection if it doesn't exist
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")
                
        except ImportError:
            logger.error("qdrant-client not installed")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            self.client = None
    
    def is_ready(self) -> bool:
        """Check if RAG system is fully initialized"""
        ready = self.client is not None and self.encoder is not None
        if not ready:
            if self.client is None:
                logger.warning("RAG system not ready: Qdrant client not initialized")
            if self.encoder is None:
                logger.warning("RAG system not ready: Sentence transformer not loaded")
        return ready
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None):
        """Add documents to the knowledge base"""
        if not self.is_ready():
            return
        
        if not documents:
            return
        
        try:
            from qdrant_client.models import PointStruct
            
            # Generate embeddings (thread-safe)
            with self._lock:
                embeddings = self.encoder.encode(documents, show_progress_bar=False)
            
            # Prepare points
            points = []
            for i, (doc, emb) in enumerate(zip(documents, embeddings)):
                point_id = str(uuid.uuid4())
                payload = {
                    "text": doc,
                    "timestamp": time.time(),
                    **(metadatas[i] if metadatas and i < len(metadatas) else {})
                }
                
                points.append(PointStruct(
                    id=point_id,
                    vector=emb.tolist(),
                    payload=payload
                ))
            
            # Upload to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Added {len(documents)} documents to RAG system")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
    
    def get_context(
        self,
        query: str,
        n_results: int = 3,
        chat_id: Optional[str] = None,
        scope_id: Optional[str] = None,
        global_search: bool = False,
        min_score: float = 0.0,
        fact_type_filter: Optional[str] = None
    ) -> List[tuple]:
        """
        Retrieve relevant context for a query - returns list of (text, metadata) tuples.
        
        Enhanced with:
        - Relevance threshold filtering (min_score)
        - Type-weighted scoring (facts > messages)
        - Improved recency with exponential decay
        - Optional fact_type filtering
        
        Args:
            query: Search query text
            n_results: Maximum number of results to return
            chat_id: Filter by specific chat ID (if provided and global_search=False)
            global_search: If True, search across all chats; if False, limit to chat_id
            min_score: Minimum relevance score to include (0.0 = no filter)
            fact_type_filter: Only return this fact type (e.g. "name", "preference")
        
        Backward compatible: Handles both old entries (only text field) and new entries
        (with role, chat_id, timestamp, tags, fact_type fields).
        """
        if not self.is_ready():
            return []
        
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Generate query embedding (thread-safe)
            with self._lock:
                query_vector = self.encoder.encode([query], show_progress_bar=False)[0]
            
            # Build filter for chat-scoped search
            filter_conditions = []
            if not global_search and scope_id:
                filter_conditions.append(
                    FieldCondition(key="scope_id", match=MatchValue(value=scope_id))
                )
                logger.info(f"Scoped search: filtering by scope_id={scope_id}")
            elif not global_search and chat_id:
                filter_conditions.append(
                    FieldCondition(key="chat_id", match=MatchValue(value=chat_id))
                )
                logger.info(f"Chat-scoped search: filtering by chat_id={chat_id}")
            elif global_search:
                logger.info("Global search: searching across all scopes")
            
            if fact_type_filter:
                filter_conditions.append(
                    FieldCondition(key="fact_type", match=MatchValue(value=fact_type_filter))
                )
            
            query_filter = Filter(must=filter_conditions) if filter_conditions else None
            
            # Fetch extra candidates for better filtering (2x requested)
            fetch_limit = max(n_results * 2, 8)
            
            # Search in Qdrant using query method
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist(),
                limit=fetch_limit,
                query_filter=query_filter,
                with_payload=True
            )
            
            # Extract text and metadata from results with enhanced scoring
            current_time = int(time.time())
            contexts = []
            points = search_results.points if hasattr(search_results, 'points') else search_results
            
            for result in points:
                if "text" in result.payload:
                    text = result.payload["text"]
                    metadata = {k: v for k, v in result.payload.items() if k != "text"}
                    
                    # ── Enhanced scoring pipeline ──
                    base_score = result.score  # Qdrant cosine similarity
                    timestamp = metadata.get("timestamp", 0)
                    doc_type = metadata.get("type", "message")
                    role = metadata.get("role", "")
                    
                    # 1. Recency boost with exponential decay (better than linear)
                    #    Half-life: 7 days for messages, 90 days for facts
                    if timestamp > 0:
                        age_seconds = max(0, current_time - timestamp)
                        age_days = age_seconds / 86400
                        if doc_type == "fact":
                            half_life = 90  # Facts decay slowly
                        else:
                            half_life = 7   # Messages decay faster
                        import math
                        recency_boost = math.exp(-0.693 * age_days / half_life)
                    else:
                        recency_boost = 0
                    
                    # 2. Type-weighted boost (facts are more valuable than raw messages)
                    type_boost = 0.0
                    if doc_type == "fact":
                        type_boost = 0.12  # Facts get priority
                        confidence = metadata.get("confidence", 0.5)
                        type_boost *= confidence  # Scale by extraction confidence
                    elif doc_type == "uploaded_document":
                        type_boost = 0.08  # User-uploaded docs are important
                    elif role == "assistant":
                        type_boost = 0.02  # Slight boost for assistant responses
                    
                    # 3. Composite score: 70% similarity + 15% recency + 15% type
                    final_score = 0.70 * base_score + 0.15 * recency_boost + 0.15 * type_boost
                    
                    # Add scoring metadata for debugging
                    metadata["base_score"] = round(base_score, 4)
                    metadata["recency_boost"] = round(recency_boost, 4)
                    metadata["type_boost"] = round(type_boost, 4)
                    metadata["final_score"] = round(final_score, 4)
                    metadata["score"] = round(final_score, 4)
                    
                    # Apply minimum score filter
                    if final_score >= min_score:
                        contexts.append((text, metadata))
            
            # Sort by final_score (descending) and limit
            contexts.sort(key=lambda x: x[1]["final_score"], reverse=True)
            contexts = contexts[:n_results]
            
            if contexts:
                logger.info(
                    f"Retrieved {len(contexts)} context chunks for query "
                    f"(scores: {[m['final_score'] for _, m in contexts]}, "
                    f"types: {[m.get('type', '?') for _, m in contexts]})"
                )
            else:
                logger.debug(f"No context found for query: {query[:50]}")
            return contexts
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        if not self.client:
            return
        
        try:
            with self._lock:
                self.client.delete_collection(self.collection_name)
                logger.info(f"Deleted collection: {self.collection_name}")
                self._initialize_qdrant()
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
    
    def get_stats(self) -> Dict:
        """Get statistics about the RAG system"""
        if not self.client:
            return {"status": "not_initialized"}
        
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "status": "ready",
                "points_count": collection_info.points_count,
                "collection_name": self.collection_name,
                "encoder_loaded": self.encoder is not None
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"status": "error", "error": str(e)}
