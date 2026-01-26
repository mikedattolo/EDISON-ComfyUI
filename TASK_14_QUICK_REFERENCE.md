# Task 14 Quick Reference

## What This Does

Converts EDISON's `agent` and `work` modes from **keyword-matching** to **agentic reasoning**. The LLM now decides what tools to use and iterates up to 5 times to refine answers.

## For Users

**Use agent mode** for research/discovery tasks:
```
POST /chat
{
  "message": "Find information about CRISPR gene therapy and summarize",
  "mode": "agent"  // Or "work"
}
```

**Response**:
```json
{
  "response": "CRISPR gene therapy is... [details from web search]. Based on search results [source:web_search]",
  "tools_used": ["web_search"],
  "mode_used": "agent"
}
```

## For Developers

### Adding a New Tool

1. **Add to registry** (app.py, ~line 410):
```python
TOOL_REGISTRY["my_tool"] = {
    "args": {
        "param1": {"type": str, "required": True},
        "param2": {"type": int, "required": False, "default": 10}
    }
}
```

2. **Add execution logic** (_execute_tool function, ~line 516):
```python
if tool_name == "my_tool":
    result = do_something(args.get("param1"), args.get("param2"))
    return {"ok": True, "data": result}
```

3. **Add summarization** (_summarize_tool_result function, ~line 486):
```python
if tool_name == "my_tool" and isinstance(data, list):
    return "My tool results: " + ", ".join([item[:50] for item in data])
```

4. **Test**: Add validation + execution test in test_structured_tools.py

### Key Functions

| Function | Purpose | Async? |
|----------|---------|--------|
| `_validate_and_normalize_tool_call(payload)` | Strict JSON validation | No |
| `_execute_tool(tool_name, args, chat_id)` | Run tool in thread pool | Yes |
| `_summarize_tool_result(tool_name, result)` | Convert result to text | No |
| `run_structured_tool_loop(llm, user_msg, context, ...)` | Orchestrate full loop | Yes |

### Testing

```bash
# Run all tests
python test_structured_tools.py

# Expected: "✓ ALL TESTS PASSED"
```

### Error Cases

**Invalid JSON** → LLM gets 1 correction attempt, then loop stops
```
User: agent mode
Step 1: LLM outputs malformed JSON
System: "Tool call was invalid (error), fix it"
Step 2: LLM outputs corrected JSON (or gives up)
If still invalid: Stop and return error message
```

**Unknown tool** → Immediate rejection
```json
{"tool":"not_a_tool","args":{}}
// Rejected: "Unknown tool 'not_a_tool'"
```

**Timeout** → Tool marked as failed, loop continues
```
Tool execution takes >12s
// Result: {"ok":false,"error":"timeout"}
// LLM continues with next step
```

### Config Constants

```python
TOOL_LOOP_MAX_STEPS = 5          # Max iterations
TOOL_CALL_TIMEOUT_SEC = 12       # Timeout per tool execution
TOOL_RESULT_CHAR_LIMIT = 900     # Max chars per tool result summary
```

## Architecture

```
User message (agent mode)
    ↓
route_mode() → tools_allowed = True
    ↓
run_structured_tool_loop()
    ├─ Step 1: LLM decides tool
    ├─ Step 2: Validate JSON schema
    ├─ Step 3: Execute tool (timeout)
    ├─ Step 4: Summarize + append history
    ├─ Step 5: Repeat (max 5 times)
    └─ Final: Add citations
    ↓
Return response with "tools_used" metadata
```

## Safety Summary

- ✅ **No injection attacks**: JSON strictly validated, unknown tools denied
- ✅ **No infinite loops**: 5 steps max, timeout per tool, cancellation support
- ✅ **No type confusion**: str/int/bool/float strictly distinguished
- ✅ **No silent failures**: Errors are summarized and fed back to LLM

## Files to Know

| File | Purpose |
|------|---------|
| `services/edison_core/app.py` | Main implementation (~600 new lines) |
| `test_structured_tools.py` | Validation test suite (~290 lines) |
| `TASK_14_STRUCTURED_TOOLS.md` | Detailed docs with examples |
| `TASK_14_COMPLETION.md` | Before/after comparison |

## Common Patterns

### Query a Tool
```python
# In run_structured_tool_loop, LLM generates:
{
  "tool": "web_search",
  "args": {
    "query": "python asyncio patterns",
    "max_results": 3
  }
}
```

### Validate and Execute
```python
valid, error, tool, args = _validate_and_normalize_tool_call(payload)
if not valid:
    return f"Invalid: {error}"
result = await _execute_tool(tool, args, chat_id)
```

### Build Citation
```python
if tool_events:
    sources = ", ".join([e["tool"] for e in tool_events])
    final_answer += f"\n\nSources: {sources}"
```

## Troubleshooting

**"Unknown tool 'foo'"**
→ Tool name not in `TOOL_REGISTRY`
→ Add it + define execution logic

**"max_results must be int"**
→ JSON has `"max_results": "5"` (string)
→ LLM should output `"max_results": 5` (number)

**"Tool timed out"**
→ Tool took >12s
→ Check network/system load
→ Increase `TOOL_CALL_TIMEOUT_SEC` if needed

**Loop ran 5 steps but no answer**
→ LLM kept requesting tools instead of concluding
→ Consider adding a "final answer" directive in loop prompt

---

**Version**: Task 14 Complete | **Date**: 2025-01-26
