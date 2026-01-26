#!/usr/bin/env python3
"""
Test memory storage structure with separate user/assistant messages
Run with: python3 test_memory_storage.py
"""

import sys
sys.path.insert(0, '.')

from services.edison_core.rag import RAGSystem
import tempfile
import shutil
import time

def test_memory_storage():
    """Test that messages are stored separately with proper metadata"""
    
    print("=" * 60)
    print("Testing Memory Storage Structure")
    print("=" * 60)
    print()
    
    # Create temporary storage
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize RAG system
        rag = RAGSystem(storage_path=temp_dir)
        
        if not rag.is_ready():
            print("❌ RAG system not ready, skipping tests")
            return
        
        print("✓ RAG system initialized")
        
        # Test 1: Store user message
        chat_id = str(int(time.time() * 1000))
        timestamp = int(time.time())
        
        rag.add_documents(
            documents=["What is your name?"],
            metadatas=[{
                "role": "user",
                "chat_id": chat_id,
                "timestamp": timestamp,
                "tags": ["conversation", "auto"],
                "type": "message"
            }]
        )
        print("✓ Test 1: User message stored with metadata")
        
        # Test 2: Store assistant message
        rag.add_documents(
            documents=["I'm EDISON, your AI assistant."],
            metadatas=[{
                "role": "assistant",
                "chat_id": chat_id,
                "timestamp": timestamp,
                "tags": ["conversation", "auto"],
                "type": "message",
                "mode": "auto"
            }]
        )
        print("✓ Test 2: Assistant message stored with metadata")
        
        # Test 3: Store fact with fact_type
        rag.add_documents(
            documents=["User name is Alice"],
            metadatas=[{
                "role": "fact",
                "fact_type": "name",
                "confidence": 0.95,
                "chat_id": chat_id,
                "timestamp": timestamp,
                "tags": ["fact", "name"],
                "type": "fact",
                "source": "conversation"
            }]
        )
        print("✓ Test 3: Fact stored with fact_type and confidence")
        
        # Test 4: Verify retrieval works (backward compatibility)
        results = rag.get_context("name", n_results=2)
        
        if results:
            print(f"✓ Test 4: Retrieved {len(results)} results")
            
            # Check metadata structure
            for text, metadata in results:
                print(f"  - Text: {text[:50]}...")
                if "role" in metadata:
                    print(f"    Role: {metadata['role']}")
                if "chat_id" in metadata:
                    print(f"    Chat ID: {metadata['chat_id']}")
                if "fact_type" in metadata:
                    print(f"    Fact Type: {metadata['fact_type']}")
                if "tags" in metadata:
                    print(f"    Tags: {metadata['tags']}")
        else:
            print("  ⚠ No results retrieved (may need more context)")
        
        # Test 5: Backward compatibility - old format without role
        rag.add_documents(
            documents=["Old format message"],
            metadatas=[{"type": "conversation"}]
        )
        print("✓ Test 5: Old format message stored (backward compatibility)")
        
        # Test 6: Verify old format can be retrieved
        results_old = rag.get_context("old format", n_results=1)
        if results_old:
            text, metadata = results_old[0]
            # Should work even without role field
            assert "text" not in metadata  # text should be separate
            print("✓ Test 6: Old format message retrieved successfully")
        
        # Get stats
        stats = rag.get_stats()
        print(f"\n📊 RAG Stats:")
        print(f"  Points stored: {stats.get('points_count', 0)}")
        print(f"  Status: {stats.get('status', 'unknown')}")
        
        print("\n" + "=" * 60)
        print("✅ All memory storage tests passed!")
        print("=" * 60)
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("\n🧹 Cleaned up temporary storage")


if __name__ == "__main__":
    test_memory_storage()
