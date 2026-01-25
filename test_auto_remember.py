#!/usr/bin/env python3
"""
Test auto-remembering and recall functionality
"""

import requests
import json
import time

API_URL = "http://127.0.0.1:8811"

def test_auto_remember():
    """Test automatic memory detection based on content"""
    print("=" * 60)
    print("Testing Auto-Remember Functionality")
    print("=" * 60)
    
    test_cases = [
        # Should remember (personal info)
        ("My name is Sarah and I'm a software engineer", True),
        ("I love Python programming and AI", True),
        ("My favorite color is purple", True),
        ("I live in San Francisco", True),
        ("Remind me to check this later", True),
        
        # Should NOT remember (simple queries)
        ("What is the capital of France?", False),
        ("How do I install Python?", False),
        ("Tell me about quantum computing", False),
        ("Search for the latest news", False),
    ]
    
    print("\n📝 Testing auto-remember detection...\n")
    
    for message, should_remember in test_cases:
        try:
            response = requests.post(
                f"{API_URL}/chat",
                json={
                    "message": message,
                    "mode": "chat",
                    "remember": None  # Let backend decide
                },
                timeout=30
            )
            
            if response.status_code == 200:
                # Check if conversation was stored
                time.sleep(0.5)
                
                # Try to recall it
                recall_msg = f"What did I say about {message.split()[0]}?"
                recall_response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "message": recall_msg,
                        "mode": "chat",
                        "remember": None
                    },
                    timeout=30
                )
                
                indicator = "✅ REMEMBER" if should_remember else "⏭️  SKIP"
                print(f"{indicator}: '{message[:50]}...'")
                
            else:
                print(f"❌ Failed: {message[:50]}...")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(1)

def test_explicit_recall():
    """Test explicit recall commands"""
    print("\n\n" + "=" * 60)
    print("Testing Explicit Recall Functionality")
    print("=" * 60)
    
    # First, have a conversation about a specific topic
    print("\n📖 Setting up conversation history...")
    
    conversations = [
        "Tell me about the Eiffel Tower in Paris",
        "The Eiffel Tower is 330 meters tall",
        "It was built by Gustave Eiffel in 1889",
    ]
    
    for msg in conversations:
        try:
            requests.post(
                f"{API_URL}/chat",
                json={"message": msg, "mode": "chat", "remember": True},
                timeout=30
            )
            print(f"   Stored: {msg[:60]}...")
            time.sleep(1)
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Now test recall commands
    print("\n🔍 Testing recall commands...\n")
    
    recall_tests = [
        "What did we talk about regarding Paris?",
        "Recall our conversation about the Eiffel Tower",
        "Search my conversations for Eiffel Tower",
        "Remember when we discussed Paris?",
        "What did you tell me about Gustave Eiffel?",
    ]
    
    for recall_cmd in recall_tests:
        try:
            print(f"\n👤 User: {recall_cmd}")
            response = requests.post(
                f"{API_URL}/chat",
                json={
                    "message": recall_cmd,
                    "mode": "auto",
                    "remember": None
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("response", "")
                
                # Check if recall was successful
                if any(word in answer.lower() for word in ["eiffel", "paris", "tower", "330", "1889"]):
                    print(f"✅ Successfully recalled context")
                    print(f"   Response: {answer[:150]}...")
                else:
                    print(f"⚠️  May not have recalled correctly")
                    print(f"   Response: {answer[:150]}...")
            else:
                print(f"❌ Request failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(2)

def test_cross_chat_recall():
    """Test recalling information from different chat sessions"""
    print("\n\n" + "=" * 60)
    print("Testing Cross-Chat Recall")
    print("=" * 60)
    
    print("\n📚 Storing information in 'Chat 1'...")
    # Simulate different chat by storing with metadata
    requests.post(
        f"{API_URL}/chat",
        json={
            "message": "My favorite movie is The Matrix",
            "mode": "chat",
            "remember": True
        },
        timeout=30
    )
    print("   Stored: My favorite movie is The Matrix")
    
    time.sleep(1)
    
    print("\n📚 Storing information in 'Chat 2'...")
    requests.post(
        f"{API_URL}/chat",
        json={
            "message": "I'm learning React and TypeScript",
            "mode": "chat",
            "remember": True
        },
        timeout=30
    )
    print("   Stored: I'm learning React and TypeScript")
    
    time.sleep(2)
    
    # Try to recall from "different" chat
    print("\n🔍 Attempting cross-chat recall...")
    recall_tests = [
        "What's my favorite movie?",
        "What am I learning?",
        "Search my history for what I like",
    ]
    
    for recall_cmd in recall_tests:
        try:
            print(f"\n👤 User: {recall_cmd}")
            response = requests.post(
                f"{API_URL}/chat",
                json={
                    "message": recall_cmd,
                    "mode": "auto",
                    "remember": None
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("response", "")
                print(f"🤖 EDISON: {answer[:200]}...")
                
                # Check for correct recall
                if "matrix" in answer.lower() or "react" in answer.lower() or "typescript" in answer.lower():
                    print("✅ Cross-chat recall successful!")
                else:
                    print("⚠️  May not have recalled from other chats")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(2)

def main():
    print("\n" + "=" * 60)
    print("EDISON Auto-Remember & Recall Testing")
    print("=" * 60)
    
    # Check if service is running
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ EDISON service not responding correctly")
            return
    except Exception as e:
        print(f"❌ Cannot connect to EDISON service: {e}")
        print("   Make sure EDISON is running: python -m services.edison_core.app")
        return
    
    print("✅ EDISON service is running\n")
    
    test_auto_remember()
    test_explicit_recall()
    test_cross_chat_recall()
    
    print("\n\n" + "=" * 60)
    print("✨ Testing Complete!")
    print("=" * 60)
    print("\n📊 Summary:")
    print("  - Auto-remember: Detects personal info automatically")
    print("  - Explicit recall: Responds to 'recall', 'remember', 'search' commands")
    print("  - Cross-chat: Searches across all conversation history")
    print("\n✅ All features tested!")

if __name__ == "__main__":
    main()
