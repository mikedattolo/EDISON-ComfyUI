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
        global_search: bool = False
    ) -> List[tuple]:
        """
        Retrieve relevant context for a query - returns list of (text, metadata) tuples.
        
        Args:
            query: Search query text
            n_results: Maximum number of results to return
            chat_id: Filter by specific chat ID (if provided and global_search=False)
            global_search: If True, search across all chats; if False, limit to chat_id
        
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
            query_filter = None
            if not global_search and scope_id:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="scope_id",
                            match=MatchValue(value=scope_id)
                        )
                    ]
                )
                logger.info(f"Scoped search: filtering by scope_id={scope_id}")
            elif not global_search and chat_id:
                # Backward compatibility with chat_id-only filtering
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="chat_id",
                            match=MatchValue(value=chat_id)
                        )
                    ]
                )
                logger.info(f"Chat-scoped search: filtering by chat_id={chat_id}")
            elif global_search:
                logger.info("Global search: searching across all scopes")
            
            # Search in Qdrant using query method
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist(),
                limit=n_results,
                query_filter=query_filter,  # Apply filter if not global search
                with_payload=True  # Ensure we get full payload
            )
            
            # Extract text and metadata from results
            # Backward compatible: works with old entries that only have "text" field
            current_time = int(time.time())
            contexts = []
            points = search_results.points if hasattr(search_results, 'points') else search_results
            
            for result in points:
                if "text" in result.payload:
                    text = result.payload["text"]
                    metadata = {k: v for k, v in result.payload.items() if k != "text"}
                    
                    # Recency-aware scoring
                    base_score = result.score  # Qdrant similarity score
                    timestamp = metadata.get("timestamp", 0)  # If missing, treat as old
                    
                    # Compute recency boost
                    if timestamp > 0:
                        age_days = (current_time - timestamp) / 86400
                        recency_boost = max(0, min(1, 1 - (age_days / 30)))  # clamp(0, 1)
                    else:
                        recency_boost = 0  # Treat as old if no timestamp
                    
                    # Final score: 85% similarity, 15% recency
                    final_score = 0.85 * base_score + 0.15 * recency_boost
                    
                    # Add scoring metadata for debugging
                    metadata["base_score"] = base_score
                    metadata["recency_boost"] = recency_boost
                    metadata["final_score"] = final_score
                    metadata["score"] = final_score  # Primary score field
                    
                    contexts.append((text, metadata))
            
            # Sort by final_score (descending)
            contexts.sort(key=lambda x: x[1]["final_score"], reverse=True)
            
            logger.info(f"Retrieved {len(contexts)} context chunks for query (final_scores: {[m['final_score'] for _, m in contexts]})")
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
