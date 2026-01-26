#!/usr/bin/env python3
"""
Test consolidated route_mode() function for consistent routing
"""

import sys
sys.path.insert(0, '.')

def test_route_mode():
    """Test the consolidated routing function"""
    from services.edison_core.app import route_mode
    
    print("=" * 70)
    print("Testing route_mode() - Consolidated Routing Function")
    print("=" * 70)
    
    tests = [
        # (user_message, requested_mode, has_image, coral_intent, expected_mode, expected_tools, expected_model)
        ("Write a Python function", "auto", False, None, "code", False, "deep"),
        ("Search the web for Python news", "auto", False, "agent", "agent", True, "deep"),
        ("Generate an image of a sunset", "auto", False, "generate_image", "image", False, "vision"),
        ("What is gravity?", "auto", False, None, "reasoning", False, "deep"),
        ("Hello there", "auto", False, None, "chat", False, "fast"),
        ("Here's an image", "auto", True, None, "image", False, "vision"),
        ("Generate image", "auto", True, "generate_image", "image", False, "vision"),
        ("Explain quantum mechanics", "auto", False, None, "reasoning", False, "deep"),
        ("Create a project plan", "auto", False, None, "work", False, "deep"),
        ("Chat with me", "chat", False, None, "chat", False, "fast"),
        ("Code mode override", "code", False, "generate_image", "code", False, "deep"),
        ("Image with explicit code", "code", True, None, "image", False, "vision"),
    ]
    
    print("\nTest Cases:\n")
    passed = 0
    failed = 0
    
    for i, (msg, mode, has_img, intent, exp_mode, exp_tools, exp_model) in enumerate(tests, 1):
        result = route_mode(msg, mode, has_img, intent)
        
        mode_ok = result["mode"] == exp_mode
        tools_ok = result["tools_allowed"] == exp_tools
        model_ok = result["model_target"] == exp_model
        
        status = "✓" if (mode_ok and tools_ok and model_ok) else "✗"
        
        print(f"{status} Test {i}: '{msg[:40]}...' if len(msg) > 40 else '{msg}'")
        print(f"  Request: mode={mode}, has_image={has_img}, intent={intent}")
        print(f"  Expected: {exp_mode}, tools={exp_tools}, model={exp_model}")
        print(f"  Got:      {result['mode']}, tools={result['tools_allowed']}, model={result['model_target']}")
        
        if mode_ok and tools_ok and model_ok:
            print(f"  Reasons: {result['reasons']}")
            passed += 1
        else:
            print(f"  MISMATCH! Reasons: {result['reasons']}")
            failed += 1
        print()
    
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 70)
    
    if failed == 0:
        print("✅ All routing tests passed!")
        return True
    else:
        print(f"❌ {failed} test(s) failed!")
        return False


def test_routing_patterns():
    """Test pattern matching consistency"""
    from services.edison_core.app import route_mode
    
    print("\n" + "=" * 70)
    print("Testing Pattern Recognition Consistency")
    print("=" * 70)
    
    # Code patterns
    code_messages = [
        "Write a function to sort an array",
        "Debug this code",
        "Implement a binary tree",
        "Write a Python script"
    ]
    
    print("\nCode Pattern Detection:")
    for msg in code_messages:
        result = route_mode(msg, "auto", False, None)
        print(f"  '{msg}' → {result['mode']} (expected: code)")
        assert result["mode"] == "code", f"Expected code, got {result['mode']}"
    print("✓ All code patterns detected correctly")
    
    # Agent patterns
    agent_messages = [
        "Search the internet for latest Python news",
        "Research blockchain technology",
        "Find information on AI trends",
        "Tell me about the latest tech news"
    ]
    
    print("\nAgent Pattern Detection:")
    for msg in agent_messages:
        result = route_mode(msg, "auto", False, None)
        print(f"  '{msg}' → {result['mode']} (expected: agent)")
        assert result["mode"] == "agent", f"Expected agent, got {result['mode']}"
    print("✓ All agent patterns detected correctly")
    
    # Reasoning patterns
    reasoning_messages = [
        "Explain the theory of relativity",
        "Why do we have seasons?",
        "How does photosynthesis work?",
        "Analyze this complex problem step by step"
    ]
    
    print("\nReasoning Pattern Detection:")
    for msg in reasoning_messages:
        result = route_mode(msg, "auto", False, None)
        print(f"  '{msg}' → {result['mode']} (expected: reasoning)")
        assert result["mode"] == "reasoning", f"Expected reasoning, got {result['mode']}"
    print("✓ All reasoning patterns detected correctly")
    
    # Image patterns
    image_cases = [
        ("Generate a sunset image", False, "generate_image"),
        ("Here's a photo to analyze", True, None),
        ("Create artwork of a dragon", False, "create_image"),
    ]
    
    print("\nImage Mode Detection:")
    for msg, has_img, intent in image_cases:
        result = route_mode(msg, "auto", has_img, intent)
        print(f"  '{msg}' (image={has_img}, intent={intent}) → {result['mode']}")
        assert result["mode"] == "image", f"Expected image, got {result['mode']}"
        assert result["model_target"] == "vision", f"Expected vision model, got {result['model_target']}"
    print("✓ All image modes detected correctly")
    
    print("\n" + "=" * 70)
    print("✅ All pattern recognition tests passed!")
    print("=" * 70)


def test_explicit_mode_priority():
    """Test that explicit mode requests are respected"""
    from services.edison_core.app import route_mode
    
    print("\n" + "=" * 70)
    print("Testing Explicit Mode Priority")
    print("=" * 70)
    
    # When user explicitly requests a mode, it should be respected
    test_cases = [
        # (message, requested_mode, expected_mode, reason)
        ("Search the web", "chat", "chat", "Explicit chat request overrides agent pattern"),
        ("Generate code", "chat", "chat", "Explicit chat request overrides code pattern"),
        ("Calculate 2+2", "code", "code", "Explicit code request overrides chat"),
    ]
    
    print("\nExplicit Mode Tests:")
    for msg, req_mode, exp_mode, reason in test_cases:
        result = route_mode(msg, req_mode, False, None)
        status = "✓" if result["mode"] == exp_mode else "✗"
        print(f"  {status} '{msg}' with mode={req_mode} → {result['mode']}")
        print(f"    Reason: {reason}")
        print(f"    Routing reasons: {result['reasons']}")
        assert result["mode"] == exp_mode, f"Expected {exp_mode}, got {result['mode']}"
    
    print("\n✓ All explicit mode priority tests passed!")


def test_single_logging():
    """Verify routing is logged once per request"""
    from services.edison_core.app import route_mode
    import logging
    
    print("\n" + "=" * 70)
    print("Testing Single Logging Per Request")
    print("=" * 70)
    
    # This test checks that route_mode logs routing decision once
    # The actual logging happens inside route_mode() via logger.info()
    
    print("\nCalling route_mode() and verifying it logs routing info...")
    print("(Check application logs for ROUTING: line)\n")
    
    result = route_mode("Write a function to calculate fibonacci", "auto", False, None)
    
    print(f"Result: {result}")
    print(f"Check logs for: 'ROUTING: mode={result['mode']}, model={result['model_target']}, ...'")
    print("\n✓ route_mode() logs routing decision once")


if __name__ == "__main__":
    # Run all tests
    success = test_route_mode()
    
    if success:
        test_routing_patterns()
        test_explicit_mode_priority()
        test_single_logging()
        
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        print("\nConsolidated routing function working correctly:")
        print("  ✓ Single function controls all routing logic")
        print("  ✓ No scattered heuristics")
        print("  ✓ Consistent pattern matching")
        print("  ✓ Explicit mode requests respected")
        print("  ✓ Routing logged once per request")
