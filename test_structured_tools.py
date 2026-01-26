#!/usr/bin/env python3
"""
Test suite for structured tool calling loop (Task 14)
Tests tool registry, JSON validation, tool execution, and full agent loop.
"""

import json
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(REPO_ROOT))

from services.edison_core.app import (
    TOOL_REGISTRY,
    _validate_and_normalize_tool_call,
    _summarize_tool_result,
)

def test_tool_registry_structure():
    """Test that tool registry has all required tools with proper schema."""
    print("\n=== Test: Tool Registry Structure ===")
    required_tools = {"web_search", "rag_search", "generate_image", "system_stats"}
    actual_tools = set(TOOL_REGISTRY.keys())
    assert required_tools.issubset(actual_tools), f"Missing tools: {required_tools - actual_tools}"
    
    # Check web_search schema
    ws = TOOL_REGISTRY["web_search"]["args"]
    assert "query" in ws and ws["query"]["type"] is str and ws["query"]["required"]
    assert "max_results" in ws and ws["max_results"]["type"] is int and not ws["max_results"]["required"]
    
    # Check rag_search schema
    rs = TOOL_REGISTRY["rag_search"]["args"]
    assert "query" in rs and rs["query"]["required"]
    assert "limit" in rs and rs["limit"]["type"] is int
    assert "global" in rs and rs["global"]["type"] is bool
    
    # Check generate_image schema
    gi = TOOL_REGISTRY["generate_image"]["args"]
    assert "prompt" in gi and gi["prompt"]["required"]
    assert all(k in gi for k in ["width", "height", "steps", "guidance_scale"])
    
    # Check system_stats (no args)
    ss = TOOL_REGISTRY["system_stats"]["args"]
    assert ss == {}
    
    print("✓ Tool registry structure valid")


def test_json_validation_valid_calls():
    """Test validation of valid tool call JSON."""
    print("\n=== Test: JSON Validation - Valid Calls ===")
    
    # Valid web_search
    valid, error, tool, args = _validate_and_normalize_tool_call({
        "tool": "web_search",
        "args": {"query": "python async", "max_results": 5}
    })
    assert valid, f"Should be valid: {error}"
    assert tool == "web_search"
    assert args == {"query": "python async", "max_results": 5}
    print("✓ web_search with both args validated")
    
    # web_search with default max_results
    valid, error, tool, args = _validate_and_normalize_tool_call({
        "tool": "web_search",
        "args": {"query": "test"}
    })
    assert valid
    assert args == {"query": "test", "max_results": 5}
    print("✓ web_search with default max_results injected")
    
    # Valid rag_search
    valid, error, tool, args = _validate_and_normalize_tool_call({
        "tool": "rag_search",
        "args": {"query": "my name", "limit": 2, "global": False}
    })
    assert valid
    assert tool == "rag_search"
    print("✓ rag_search validated")
    
    # Valid generate_image
    valid, error, tool, args = _validate_and_normalize_tool_call({
        "tool": "generate_image",
        "args": {"prompt": "sunset over ocean"}
    })
    assert valid
    assert "prompt" in args
    assert "width" in args and args["width"] == 1024  # default
    print("✓ generate_image with defaults validated")
    
    # Valid system_stats (no args needed)
    valid, error, tool, args = _validate_and_normalize_tool_call({
        "tool": "system_stats",
        "args": {}
    })
    assert valid
    assert tool == "system_stats"
    print("✓ system_stats validated")


def test_json_validation_invalid_calls():
    """Test that invalid tool calls are properly rejected."""
    print("\n=== Test: JSON Validation - Invalid Calls ===")
    
    # Not a dict
    valid, error, _, _ = _validate_and_normalize_tool_call("not a dict")
    assert not valid and "must be an object" in error
    print("✓ Non-dict rejected")
    
    # Wrong keys
    valid, error, _, _ = _validate_and_normalize_tool_call({"tool": "web_search"})  # missing "args"
    assert not valid and "exactly" in error
    print("✓ Missing 'args' key rejected")
    
    # Unknown tool
    valid, error, tool, _ = _validate_and_normalize_tool_call({
        "tool": "unknown_tool",
        "args": {}
    })
    assert not valid and "Unknown tool" in error
    assert tool == "unknown_tool"
    print("✓ Unknown tool rejected")
    
    # Missing required arg
    valid, error, tool, _ = _validate_and_normalize_tool_call({
        "tool": "web_search",
        "args": {}  # missing "query"
    })
    assert not valid and "query" in error
    print("✓ Missing required arg rejected")
    
    # Wrong type for required arg
    valid, error, tool, _ = _validate_and_normalize_tool_call({
        "tool": "web_search",
        "args": {"query": 123}  # should be string
    })
    assert not valid and "must be string" in error
    print("✓ Wrong type for required arg rejected")
    
    # Invalid max_results (should be int)
    valid, error, tool, _ = _validate_and_normalize_tool_call({
        "tool": "web_search",
        "args": {"query": "test", "max_results": "5"}  # string instead of int
    })
    assert not valid and "must be int" in error
    print("✓ Wrong type for optional arg rejected")
    
    # rag_search with invalid global (should be bool)
    valid, error, tool, _ = _validate_and_normalize_tool_call({
        "tool": "rag_search",
        "args": {"query": "test", "global": "yes"}  # string instead of bool
    })
    assert not valid and "must be bool" in error
    print("✓ Invalid bool type rejected")


def test_tool_result_summarization():
    """Test that tool results are properly summarized for model consumption."""
    print("\n=== Test: Tool Result Summarization ===")
    
    # web_search success
    summary = _summarize_tool_result("web_search", {
        "ok": True,
        "data": [
            {"title": "Title 1", "url": "http://example.com", "snippet": "Snippet text"},
            {"title": "Title 2", "url": "http://example.com/2", "snippet": "More text"}
        ]
    })
    assert "Web search results:" in summary
    assert "Title 1" in summary or "Snippet text" in summary
    print(f"✓ web_search summarized: {summary[:60]}...")
    
    # web_search error
    summary = _summarize_tool_result("web_search", {
        "ok": False,
        "error": "Network timeout"
    })
    assert "web_search failed" in summary and "Network timeout" in summary
    print(f"✓ web_search error summarized: {summary}")
    
    # rag_search success
    summary = _summarize_tool_result("rag_search", {
        "ok": True,
        "data": [
            ("My name is Alice", "source1"),
            ("I enjoy coding", "source2")
        ]
    })
    assert "RAG search results:" in summary
    print(f"✓ rag_search summarized: {summary[:60]}...")
    
    # generate_image
    summary = _summarize_tool_result("generate_image", {
        "ok": True,
        "message": "Image generation queued"
    })
    assert "Image generation" in summary
    print(f"✓ generate_image summarized: {summary}")
    
    # system_stats
    summary = _summarize_tool_result("system_stats", {
        "ok": True,
        "data": {"cpu": "25%", "memory": "1.2G", "load": "0.8"}
    })
    assert "System stats:" in summary
    print(f"✓ system_stats summarized: {summary}")


def test_json_parsing_robustness():
    """Test that the validation handles edge cases gracefully."""
    print("\n=== Test: JSON Parsing Robustness ===")
    
    # Valid but with extra unknown args (should drop them)
    valid, error, tool, args = _validate_and_normalize_tool_call({
        "tool": "web_search",
        "args": {"query": "test", "unknown_param": "should_be_dropped"}
    })
    assert valid
    assert "unknown_param" not in args
    print("✓ Unknown args dropped silently")
    
    # Numbers coerced correctly
    valid, error, tool, args = _validate_and_normalize_tool_call({
        "tool": "generate_image",
        "args": {"prompt": "test", "width": 512, "guidance_scale": 3.5}
    })
    assert valid
    assert args["width"] == 512
    assert args["guidance_scale"] == 3.5
    print("✓ Numeric types handled correctly")
    
    # Float to int coercion attempt (should fail)
    valid, error, tool, args = _validate_and_normalize_tool_call({
        "tool": "generate_image",
        "args": {"prompt": "test", "steps": 20.5}  # float when int expected
    })
    assert not valid  # Should reject float for int
    print("✓ Float rejected for int field")


def test_tool_registry_completeness():
    """Test that all tools meet spec requirements."""
    print("\n=== Test: Tool Registry Spec Compliance ===")
    
    # Check required tools exist
    required = {"web_search", "rag_search", "generate_image", "system_stats"}
    actual = set(TOOL_REGISTRY.keys())
    assert required == actual, f"Tool set mismatch. Expected {required}, got {actual}"
    print("✓ All required tools present")
    
    # Check web_search args
    ws = TOOL_REGISTRY["web_search"]
    assert "query" in ws["args"] and ws["args"]["query"]["required"]
    assert "max_results" in ws["args"] and "default" in ws["args"]["max_results"]
    print("✓ web_search spec complete")
    
    # Check rag_search args
    rs = TOOL_REGISTRY["rag_search"]
    assert "query" in rs["args"]
    assert "limit" in rs["args"]
    assert "global" in rs["args"]
    print("✓ rag_search spec complete")
    
    # Check generate_image args
    gi = TOOL_REGISTRY["generate_image"]
    assert gi["args"]["prompt"]["required"]
    assert all(arg in gi["args"] for arg in ["width", "height", "steps", "guidance_scale"])
    print("✓ generate_image spec complete")
    
    # Check system_stats
    ss = TOOL_REGISTRY["system_stats"]
    assert ss["args"] == {}
    print("✓ system_stats spec complete")


if __name__ == "__main__":
    try:
        test_tool_registry_structure()
        test_json_validation_valid_calls()
        test_json_validation_invalid_calls()
        test_tool_result_summarization()
        test_json_parsing_robustness()
        test_tool_registry_completeness()
        
        print("\n" + "="*50)
        print("✓ ALL TESTS PASSED")
        print("="*50)
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
