#!/usr/bin/env python3
"""
Test conversation context and follow-up question handling
"""

import requests
import json
import time

API_URL = "http://127.0.0.1:8811"

def test_conversation_context():
    """Test that EDISON maintains context across follow-up questions"""
    print("=" * 60)
    print("Testing Conversation Context")
    print("=" * 60)
    
    # Test sequence 1: Ophelia's death follow-up
    print("\n📖 Test 1: Literature follow-up questions")
    print("-" * 60)
    
    conversation_history = []
    
    # First message: Ask about Ophelia
    print("\n👤 User: Tell me about Ophelia's death in Hamlet")
    response1 = requests.post(
        f"{API_URL}/chat",
        json={
            "message": "Tell me about Ophelia's death in Hamlet",
            "mode": "auto",
            "remember": True,
            "conversation_history": conversation_history
        },
        timeout=30
    )
    
    if response1.status_code == 200:
        data1 = response1.json()
        print(f"🤖 EDISON ({data1['mode_used']}): {data1['response'][:200]}...")
        
        # Add to conversation history
        conversation_history.append({"role": "user", "content": "Tell me about Ophelia's death in Hamlet"})
        conversation_history.append({"role": "assistant", "content": data1['response']})
    else:
        print(f"❌ Request failed: {response1.status_code}")
        return
    
    time.sleep(2)
    
    # Follow-up 1: What page is that on?
    print("\n👤 User: What page of the book would that be on?")
    response2 = requests.post(
        f"{API_URL}/chat",
        json={
            "message": "What page of the book would that be on?",
            "mode": "auto",
            "remember": True,
            "conversation_history": conversation_history
        },
        timeout=30
    )
    
    if response2.status_code == 200:
        data2 = response2.json()
        print(f"🤖 EDISON ({data2['mode_used']}): {data2['response'][:200]}...")
        
        conversation_history.append({"role": "user", "content": "What page of the book would that be on?"})
        conversation_history.append({"role": "assistant", "content": data2['response']})
        
        # Check if response mentions Hamlet or Act IV
        if "hamlet" in data2['response'].lower() or "act iv" in data2['response'].lower():
            print("✅ Context maintained - mentioned Hamlet/Act IV")
        else:
            print("⚠️ May have lost context - check response")
    else:
        print(f"❌ Request failed: {response2.status_code}")
        return
    
    time.sleep(2)
    
    # Follow-up 2: Just say "her death"
    print("\n👤 User: Her death")
    response3 = requests.post(
        f"{API_URL}/chat",
        json={
            "message": "Her death",
            "mode": "auto",
            "remember": True,
            "conversation_history": conversation_history
        },
        timeout=30
    )
    
    if response3.status_code == 200:
        data3 = response3.json()
        print(f"🤖 EDISON ({data3['mode_used']}): {data3['response'][:200]}...")
        
        if "ophelia" in data3['response'].lower():
            print("✅ Context maintained - understood 'her' refers to Ophelia")
        else:
            print("⚠️ May have lost context - didn't mention Ophelia")
    
    # Test sequence 2: Personal information recall
    print("\n\n👤 Test 2: Personal information with follow-ups")
    print("-" * 60)
    
    conversation_history2 = []
    
    # Store personal info
    print("\n👤 User: My name is Alice and I love Python programming")
    response4 = requests.post(
        f"{API_URL}/chat",
        json={
            "message": "My name is Alice and I love Python programming",
            "mode": "chat",
            "remember": True,
            "conversation_history": conversation_history2
        },
        timeout=30
    )
    
    if response4.status_code == 200:
        data4 = response4.json()
        print(f"🤖 EDISON: {data4['response'][:150]}...")
        conversation_history2.append({"role": "user", "content": "My name is Alice and I love Python programming"})
        conversation_history2.append({"role": "assistant", "content": data4['response']})
    
    time.sleep(2)
    
    # Follow-up question
    print("\n👤 User: What do I like?")
    response5 = requests.post(
        f"{API_URL}/chat",
        json={
            "message": "What do I like?",
            "mode": "auto",
            "remember": True,
            "conversation_history": conversation_history2
        },
        timeout=30
    )
    
    if response5.status_code == 200:
        data5 = response5.json()
        print(f"🤖 EDISON: {data5['response'][:150]}...")
        
        if "python" in data5['response'].lower():
            print("✅ Successfully recalled preference from conversation")
        else:
            print("⚠️ Did not recall Python preference")
    
    print("\n" + "=" * 60)
    print("✨ Conversation context testing complete!")
    print("=" * 60)

if __name__ == "__main__":
    # Check if service is running
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ EDISON service is running\n")
            test_conversation_context()
        else:
            print("❌ EDISON service not responding correctly")
    except Exception as e:
        print(f"❌ Cannot connect to EDISON service: {e}")
        print("   Make sure EDISON is running: python -m services.edison_core.app")
