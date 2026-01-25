#!/usr/bin/env python3
"""
Test script for EDISON enhancements
Tests: Intent detection, memory system, work mode
"""

import requests
import json
import time

API_URL = "http://127.0.0.1:8811"

def test_health():
    """Test if the service is running"""
    print("🔍 Testing service health...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Service is running")
            print(f"   Models loaded: {data['models_loaded']}")
            print(f"   RAG ready: {data['qdrant_ready']}")
            print(f"   Vision enabled: {data.get('vision_enabled', False)}")
            return True
        else:
            print(f"❌ Service returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Service not available: {e}")
        return False

def test_intent_detection():
    """Test enhanced intent detection"""
    print("\n🧠 Testing enhanced intent detection...")
    
    test_cases = [
        ("What is quantum computing?", "reasoning"),
        ("Write a function to sort a list", "code"),
        ("Search the internet for latest AI news", "agent"),
        ("Build an app that manages tasks", "work"),
        ("What's my name?", "chat"),
        ("Hello!", "chat"),
    ]
    
    for message, expected_mode in test_cases:
        try:
            response = requests.post(
                f"{API_URL}/chat",
                json={"message": message, "mode": "auto", "remember": False},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                detected_mode = data.get("mode_used", "unknown")
                match = "✅" if detected_mode == expected_mode else "⚠️"
                print(f"   {match} '{message[:40]}...' -> {detected_mode} (expected: {expected_mode})")
            else:
                print(f"   ❌ Failed for '{message[:40]}...'")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        time.sleep(1)  # Rate limiting

def test_memory_system():
    """Test memory system with fact extraction"""
    print("\n💾 Testing memory system...")
    
    # Store some facts
    print("   Storing facts...")
    facts = [
        "My name is John",
        "My favorite color is blue",
        "I am 25 years old",
    ]
    
    for fact in facts:
        try:
            response = requests.post(
                f"{API_URL}/chat",
                json={"message": fact, "mode": "chat", "remember": True},
                timeout=30
            )
            if response.status_code == 200:
                print(f"   ✅ Stored: {fact}")
            time.sleep(1)
        except Exception as e:
            print(f"   ❌ Error storing fact: {e}")
    
    # Test recall
    print("\n   Testing recall...")
    recall_tests = [
        "What is my name?",
        "What's my favorite color?",
        "How old am I?",
    ]
    
    for question in recall_tests:
        try:
            response = requests.post(
                f"{API_URL}/chat",
                json={"message": question, "mode": "auto", "remember": False},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("response", "")
                print(f"   Q: {question}")
                print(f"   A: {answer[:100]}...")
            time.sleep(1)
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_work_mode():
    """Test work mode with task breakdown"""
    print("\n🖥️ Testing work mode...")
    
    task = "Create a simple web application with a backend API"
    
    try:
        print(f"   Task: {task}")
        response = requests.post(
            f"{API_URL}/chat",
            json={"message": task, "mode": "work", "remember": False},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Mode used: {data.get('mode_used')}")
            
            if data.get('work_steps'):
                print(f"   📋 Task breakdown ({len(data['work_steps'])} steps):")
                for i, step in enumerate(data['work_steps'], 1):
                    print(f"      {i}. {step[:80]}...")
            
            print(f"\n   Response preview:")
            print(f"   {data.get('response', '')[:200]}...")
        else:
            print(f"   ❌ Failed with status {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

def test_rag_stats():
    """Test RAG system statistics"""
    print("\n📊 Testing RAG statistics...")
    try:
        response = requests.get(f"{API_URL}/rag/stats", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ RAG Status:")
            print(f"      Ready: {data.get('ready')}")
            print(f"      Collection: {data.get('collection')}")
            print(f"      Documents: {data.get('points_count', 0)}")
        else:
            print(f"   ⚠️ RAG stats unavailable")
    except Exception as e:
        print(f"   ❌ Error: {e}")

def main():
    print("=" * 60)
    print("EDISON Enhancement Testing")
    print("=" * 60)
    
    if not test_health():
        print("\n❌ Service not available. Please start EDISON first.")
        print("   Run: python -m services.edison_core.app")
        return
    
    test_rag_stats()
    test_intent_detection()
    test_memory_system()
    test_work_mode()
    
    print("\n" + "=" * 60)
    print("✨ Testing complete!")
    print("\n💡 For conversation context testing, run:")
    print("   python test_conversation_context.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
