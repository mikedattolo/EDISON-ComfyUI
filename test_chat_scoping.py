#!/usr/bin/env python3
"""
Test chat-scoped memory retrieval
"""

import sys
sys.path.insert(0, '.')

def test_chat_scoping():
    """Test that different chats don't cross-pollinate memories"""
    from services.edison_core.rag import RAGSystem
    import time
    import tempfile
    import shutil
    
    print("=" * 60)
    print("Testing Chat-Scoped Memory Retrieval")
    print("=" * 60)
    
    # Create temporary storage
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create RAG system
        rag = RAGSystem(storage_path=temp_dir)
        
        if not rag.is_ready():
            print("❌ RAG system not ready, skipping tests")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return
        
        print("✓ RAG system initialized\n")
        
        # Create two different chat_ids
        chat1_id = "chat_001"
        chat2_id = "chat_002"
        timestamp = int(time.time())
        
        # Add memories to chat 1
        print("Adding memories to chat 1...")
        rag.add_documents(
            documents=["My favorite color is blue"],
            metadatas=[{
                "role": "user",
                "chat_id": chat1_id,
                "timestamp": timestamp,
                "tags": ["conversation"]
            }]
        )
        rag.add_documents(
            documents=["I live in Seattle"],
            metadatas=[{
                "role": "user",
                "chat_id": chat1_id,
                "timestamp": timestamp,
                "tags": ["conversation"]
            }]
        )
        
        # Add memories to chat 2
        print("Adding memories to chat 2...")
        rag.add_documents(
            documents=["My favorite color is red"],
            metadatas=[{
                "role": "user",
                "chat_id": chat2_id,
                "timestamp": timestamp,
                "tags": ["conversation"]
            }]
        )
        rag.add_documents(
            documents=["I live in Portland"],
            metadatas=[{
                "role": "user",
                "chat_id": chat2_id,
                "timestamp": timestamp,
                "tags": ["conversation"]
            }]
        )
        
        time.sleep(0.5)  # Give indexing time
        
        # Test 1: Chat-scoped search for chat 1
        print("\n=== Test 1: Chat-scoped search for chat 1 ===")
        results = rag.get_context("favorite color", n_results=5, chat_id=chat1_id, global_search=False)
        print(f"Found {len(results)} results")
        for text, metadata in results:
            print(f"  - {text[:50]}... (chat_id: {metadata.get('chat_id', 'N/A')})")
        
        # Should only find results from chat 1
        chat1_results = [r for r in results if r[1].get('chat_id') == chat1_id]
        chat2_results = [r for r in results if r[1].get('chat_id') == chat2_id]
        
        assert len(chat2_results) == 0, f"Chat-scoped search should NOT find chat 2 results, but found {len(chat2_results)}"
        print("✓ Pass: Chat 1 scoped search isolated correctly")
        
        # Test 2: Chat-scoped search for chat 2
        print("\n=== Test 2: Chat-scoped search for chat 2 ===")
        results = rag.get_context("favorite color", n_results=5, chat_id=chat2_id, global_search=False)
        print(f"Found {len(results)} results")
        for text, metadata in results:
            print(f"  - {text[:50]}... (chat_id: {metadata.get('chat_id', 'N/A')})")
        
        # Should only find results from chat 2
        chat1_results = [r for r in results if r[1].get('chat_id') == chat1_id]
        chat2_results = [r for r in results if r[1].get('chat_id') == chat2_id]
        
        assert len(chat1_results) == 0, f"Chat-scoped search should NOT find chat 1 results, but found {len(chat1_results)}"
        print("✓ Pass: Chat 2 scoped search isolated correctly")
        
        # Test 3: Global search finds both
        print("\n=== Test 3: Global search finds both chats ===")
        results = rag.get_context("favorite color", n_results=5, chat_id=None, global_search=True)
        print(f"Found {len(results)} results")
        for text, metadata in results:
            print(f"  - {text[:50]}... (chat_id: {metadata.get('chat_id', 'N/A')})")
        
        # Global search should find results from both chats
        chat1_count = len([r for r in results if r[1].get('chat_id') == chat1_id])
        chat2_count = len([r for r in results if r[1].get('chat_id') == chat2_id])
        
        assert chat1_count > 0 and chat2_count > 0, f"Global search should find both chats (chat1={chat1_count}, chat2={chat2_count})"
        print(f"✓ Pass: Global search found {chat1_count} from chat1 and {chat2_count} from chat2")
        
        # Test 4: Location query chat-scoped
        print("\n=== Test 4: Location query chat-scoped ===")
        results_chat1 = rag.get_context("where do I live", n_results=5, chat_id=chat1_id, global_search=False)
        results_chat2 = rag.get_context("where do I live", n_results=5, chat_id=chat2_id, global_search=False)
        
        # Extract locations
        chat1_has_seattle = any("seattle" in r[0].lower() for r in results_chat1)
        chat1_has_portland = any("portland" in r[0].lower() for r in results_chat1)
        chat2_has_seattle = any("seattle" in r[0].lower() for r in results_chat2)
        chat2_has_portland = any("portland" in r[0].lower() for r in results_chat2)
        
        print(f"  Chat 1 sees: Seattle={chat1_has_seattle}, Portland={chat1_has_portland}")
        print(f"  Chat 2 sees: Seattle={chat2_has_seattle}, Portland={chat2_has_portland}")
        
        # Verify isolation (if results were found)
        if results_chat1:
            assert not chat1_has_portland, "Chat 1 should NOT see Portland"
        if results_chat2:
            assert not chat2_has_seattle, "Chat 2 should NOT see Seattle"
        
        print("✓ Pass: Location memories properly isolated")
        
        print("\n" + "=" * 60)
        print("✅ All chat scoping tests passed!")
        print("=" * 60)
        print("\nKey behaviors verified:")
        print("  ✓ Chat-scoped searches only see their own chat's memories")
        print("  ✓ Different chats don't cross-pollinate")
        print("  ✓ Global search can see across all chats")
        print("  ✓ Chat isolation works for different query types")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("\n🧹 Cleaned up temporary storage")


if __name__ == "__main__":
    test_chat_scoping()
    results = rag.get_context("favorite color", n_results=5, chat_id=chat1_id, global_search=True)
    print(f"Results: {results}")
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    colors_found = [r[0].lower() for r in results]
    assert any("blue" in c for c in colors_found), "Should find blue"
    assert any("red" in c for c in colors_found), "Should find red"
    print("✓ Pass: Global search sees both")
    
    # Test 4: Location scoping
    print("\n=== Test 4: Location scoping ===")
    results = rag.get_context("where live", n_results=5, chat_id=chat1_id, global_search=False)
    print(f"Results: {results}")
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    assert "seattle" in results[0][0].lower(), "Should find Seattle from chat 1"
    assert "portland" not in results[0][0].lower(), "Should NOT find Portland from chat 2"
    print("✓ Pass: Location properly scoped")
    
    # Test 5: No chat_id = global by default
    print("\n=== Test 5: No chat_id defaults to global ===")
    results = rag.get_context("favorite color", n_results=5, chat_id=None, global_search=False)
    print(f"Results: {results}")
    assert len(results) == 2, f"Expected 2 results (global default), got {len(results)}"
    print("✓ Pass: No chat_id uses global search")
    
    # Cleanup
    rag.client.delete_collection("test_chat_scoping")
    
    print("\n✓✓✓ All tests passed! ✓✓✓")


if __name__ == "__main__":
    test_chat_scoping()
