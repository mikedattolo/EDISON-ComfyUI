#!/usr/bin/env python3
"""
Test script for OpenAI-compatible /v1/chat/completions endpoint.
Run with: python test_openai_compat.py
"""

import requests
import json
import sys

BASE_URL = "http://192.168.1.26:8811"
MODEL = "qwen2.5-14b"

def test_non_streaming():
    """Test non-streaming OpenAI-compatible response."""
    print("\n" + "="*60)
    print("TEST 1: Non-Streaming OpenAI-Compatible Response")
    print("="*60)
    
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": False
    }
    
    print(f"\nSending request to {url}")
    print(f"   Model: {MODEL}")
    print(f"   Stream: False")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        print(f"\nResponse received (status {response.status_code})")
        print(f"   ID: {data['id']}")
        print(f"   Model: {data['model']}")
        print(f"   Tokens: {data['usage']['prompt_tokens']} + {data['usage']['completion_tokens']} = {data['usage']['total_tokens']}")
        print(f"\nAssistant response:")
        print(f"   {data['choices'][0]['message']['content']}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return False

def test_streaming():
    """Test streaming OpenAI-compatible response."""
    print("\n" + "="*60)
    print("TEST 2: Streaming OpenAI-Compatible Response")
    print("="*60)
    
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a short story about a robot."}
        ],
        "temperature": 0.7,
        "max_tokens": 200,
        "stream": True
    }
    
    print(f"\nSending streaming request to {url}")
    print(f"   Model: {MODEL}")
    print(f"   Stream: True")
    
    try:
        response = requests.post(url, json=payload, stream=True, timeout=30)
        response.raise_for_status()
        
        print(f"\nStream started (status {response.status_code})")
        print(f"\nStreamed tokens:")
        
        full_response = ""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8') if isinstance(line, bytes) else line
                
                if line.startswith('data: '):
                    data_str = line[6:]
                    
                    if data_str == '[DONE]':
                        print("\n   [DONE]")
                        break
                    
                    try:
                        chunk = json.loads(data_str)
                        if chunk['choices'][0].get('delta', {}).get('content'):
                            token = chunk['choices'][0]['delta']['content']
                            full_response += token
                            print(token, end="", flush=True)
                    except json.JSONDecodeError:
                        pass
        
        print(f"\n\nTotal response length: {len(full_response)} characters")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return False

def test_model_routing():
    """Test model parameter routing."""
    print("\n" + "="*60)
    print("TEST 3: Model Parameter Routing")
    print("="*60)
    
    test_models = ["gpt-3.5-turbo", "gpt-4", "fast", "deep", "qwen2.5-14b"]
    
    for model in test_models:
        url = f"{BASE_URL}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 20,
            "stream": False
        }
        
        try:
            print(f"\nTesting model: {model}")
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            print(f"   Routed to: {data['model']}")
        except requests.exceptions.RequestException as e:
            print(f"   Error: {e}")

def test_with_openai_client():
    """Test with OpenAI Python client (if installed)."""
    print("\n" + "="*60)
    print("TEST 4: OpenAI Python Client Library")
    print("="*60)
    
    try:
        from openai import OpenAI
        
        client = OpenAI(
            api_key="not-needed",
            base_url=f"{BASE_URL}/v1"
        )
        
        print(f"\nTesting with OpenAI Python client")
        print(f"   Base URL: {BASE_URL}/v1")
        
        response = client.chat.completions.create(
            model="fast",
            messages=[
                {"role": "user", "content": "What is 2+2?"}
            ],
            max_tokens=50,
            temperature=0.5
        )
        
        print(f"\nOpenAI client worked!")
        print(f"   Response: {response.choices[0].message.content}")
        return True
    
    except ImportError:
        print("\nOpenAI Python client not installed")
        print("   Install with: pip install openai")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("EDISON OpenAI-Compatible Endpoint Tests")
    print(f"Base URL: {BASE_URL}")
    
    results = []
    results.append(("Non-streaming", test_non_streaming()))
    results.append(("Streaming", test_streaming()))
    results.append(("Model routing", test_model_routing()))
    results.append(("OpenAI client", test_with_openai_client()))
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {test_name}")
    
    passed_count = sum(1 for _, p in results if p)
    print(f"\n{passed_count}/{len(results)} tests passed")
    
    sys.exit(0 if passed_count == len(results) else 1)
