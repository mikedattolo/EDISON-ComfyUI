"""
Retrieval-Augmented Generation System for EDISON
Uses Qdrant for vector storage and sentence-transformers for embeddings
"""

from typing import List, Dict, Optional
from pathlib import Path
import logging
import uuid
import time

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
            
            # Generate embeddings
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
    
    def get_context(self, query: str, n_results: int = 3) -> List[tuple]:
        """
        Retrieve relevant context for a query - returns list of (text, metadata) tuples.
        
        Backward compatible: Handles both old entries (only text field) and new entries
        (with role, chat_id, timestamp, tags, fact_type fields).
        """
        if not self.is_ready():
            return []
        
        try:
            # Generate query embedding
            query_vector = self.encoder.encode([query], show_progress_bar=False)[0]
            
            # Search in Qdrant using query method
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist(),
                limit=n_results,
                with_payload=True  # Ensure we get full payload
            )
            
            # Extract text and metadata from results
            # Backward compatible: works with old entries that only have "text" field
            contexts = []
            points = search_results.points if hasattr(search_results, 'points') else search_results
            for result in points:
                if "text" in result.payload:
                    text = result.payload["text"]
                    metadata = {k: v for k, v in result.payload.items() if k != "text"}
                    metadata["score"] = result.score  # Add relevance score
                    contexts.append((text, metadata))
            
            logger.info(f"Retrieved {len(contexts)} context chunks for query (scores: {[m['score'] for _, m in contexts]})")
            return contexts
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        if not self.client:
            return
        
        try:
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
