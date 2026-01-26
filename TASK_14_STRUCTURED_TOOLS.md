# Task 14: Structured Tool Calling Loop

**Status**: ✅ **COMPLETE**

## Overview

Replaced keyword heuristics with structured tool calling. The system now:
- Maintains a strict **tool registry** with JSON schema validation
- Executes a **max 5-step tool loop** where the LLM reasons about tool use
- **Validates JSON strictly** against the registry schema before execution
- **Cites/attributes** tool results in the final answer
- Provides **timeout protection** and error recovery

## Implementation Details

### Tool Registry

Located in `services/edison_core/app.py` (constant `TOOL_REGISTRY`):

```python
{
  "web_search": {"args": {"query": str (required), "max_results": int (default=5)}},
  "rag_search": {"args": {"query": str (required), "limit": int (default=3), "global": bool (default=False)}},
  "generate_image": {"args": {"prompt": str (required), "width": int (default=1024), "height": int (default=1024), "steps": int (default=20), "guidance_scale": float (default=3.5)}},
  "system_stats": {"args": {}}
}
```

### Validation Function: `_validate_and_normalize_tool_call(payload: dict)`

**Returns**: `(valid: bool, error: str or None, tool_name: str or None, normalized_args: dict or None)`

**Validation rules**:
1. Payload must be a dict with exactly keys `{"tool", "args"}`
2. `tool` must be a string matching a registered tool name
3. `args` must be a dict
4. All required args must be present with correct types
5. Type checking: `str`, `int`, `bool`, `float` (no bool→int coercion)
6. Unknown args are silently dropped; defaults are injected

**Example**:
```python
valid, error, tool, args = _validate_and_normalize_tool_call({
    "tool": "web_search",
    "args": {"query": "python async"}
})
# Returns: (True, None, "web_search", {"query": "python async", "max_results": 5})
```

### Tool Execution: `async def _execute_tool(tool_name: str, args: dict, chat_id: Optional[str])`

Executes tool in thread pool with timeout (`TOOL_CALL_TIMEOUT_SEC = 12`).

**Tool behaviors**:
- `web_search`: Calls `search_tool.search()`, returns list of `{title, url, snippet}`
- `rag_search`: Calls `rag_system.get_context()`, returns text chunks
- `generate_image`: Returns instruction for frontend `/generate-image` endpoint
- `system_stats`: Returns CPU/memory/disk percentages

**Returns**: `{"ok": bool, "error": str or None, "data": any}`

### Result Summarization: `_summarize_tool_result(tool_name: str, result: dict) -> str`

Converts tool result into **max 900 character** summary for LLM consumption:
- Extracts top 3 results per tool
- Includes titles, URLs, and snippet text
- Truncates to 200 chars per item
- Formats as: `"[Tool] results: item1 | item2 | item3"`
- Error case: `"[Tool] failed: [reason]"`

### Main Loop: `async def run_structured_tool_loop(...)`

**Orchestration** (max 5 steps):

1. **Model decision**: LLM receives history + context + instruction and outputs either:
   - Plain text (final answer) → loop exits
   - JSON object `{"tool":"...", "args":{...}}` → execute tool

2. **Strict validation**: Check JSON against schema
   - If invalid: Ask LLM to correct JSON **once**
   - If still invalid: Stop and return error
   - If valid: Execute tool

3. **Tool execution**: Run with timeout; append result to history
   - Result included as: `"Tool N [tool_name]: [summary]"`

4. **Citation building**: After loop, if tools were used:
   - Suffix final answer with: `"\n\nSources: tool1, tool2, ..."`

5. **Loop exit conditions**:
   - Reached max 5 steps
   - LLM returns plain text (final answer)
   - JSON correction fails
   - Request cancelled

**Parameters**:
- `llm`: Model instance
- `user_message`: Original user query
- `context_note`: RAG context excerpt (up to 2000 chars)
- `model_name`: Model type ("fast", "deep", etc.)
- `chat_id`: For scoped RAG searches
- `request_id`: For cancellation checking

**Returns**: `(final_answer: str, tool_events: list)`
- Each event: `{"tool": name, "args": {}, "result": {}, "summary": ""}`

## Integration with `/chat` Endpoint

When `mode in ["agent", "work"]` AND `tools_allowed is True`:
1. Build context from RAG
2. Call `run_structured_tool_loop()` with async/await
3. Store response in memory
4. Return response with `"tools_used"` metadata

Fallback: If `tools_allowed is False`, use old heuristic search.

## Safety Features

- **Unknown tools denied**: Rejects `tool` values not in `TOOL_REGISTRY`
- **Type validation**: Strict type checking (no implicit coercion except int→float)
- **Timeout protection**: 12-second limit per tool call
- **JSON validation once**: Single correction opportunity for invalid JSON
- **Cancellation support**: Checks `active_requests[request_id]["cancelled"]` between steps
- **Argument sanitization**: Drops unknown args, injects defaults

## Testing

`test_structured_tools.py` validates:
- ✓ Tool registry structure (all 4 tools present with correct schema)
- ✓ Valid tool calls accepted and normalized
- ✓ Invalid calls rejected with clear error messages
- ✓ Type validation (int vs bool vs float vs str)
- ✓ Required vs optional args
- ✓ Default injection
- ✓ Unknown args dropped
- ✓ Result summarization for each tool
- ✓ Error cases handled gracefully

Run: `python test_structured_tools.py`

## Comparison: Before vs. After

### Before (Heuristic)
```
User: "search for python tutorials"
↓
Check: "search" in msg → yes
Check: "internet"/"web" in msg → no
Keyword heuristic decides tools_allowed
↓
Hardcoded search_query extraction with regex
↓
Results stuffed directly into system prompt
```

### After (Structured)
```
User: "search for python tutorials"
↓
tools_allowed=True → run_structured_tool_loop()
↓
LLM decides: output: {"tool":"web_search","args":{"query":"python tutorials"}}
↓
Strict schema validation: ✓ valid
↓
Execute with timeout
↓
Summarize result → "Web search results: [...3 items...]"
↓
LLM uses summary to refine answer
↓
Final answer with citation: "Based on search results [source:web_search], ..."
```

## Agent Mode Workflow

1. User sends message with `mode: "agent"` or `mode: "work"`
2. Routing detects agent intent → `tools_allowed = True`
3. `/chat` endpoint calls `run_structured_tool_loop()`
4. Tool loop:
   - Step 1: LLM sees user message + context → decides tool
   - Step 2: System validates JSON + executes tool (timeout 12s)
   - Step 3: LLM sees tool result → decides next action
   - Steps 4-5: Repeat as needed
   - Final: LLM returns answer with citations

## Limitations & Future Work

- **Tool loop timeout**: Fixed 12s per tool; could be configurable per tool
- **Summary truncation**: 900 chars limit may lose detail for large result sets
- **Correction window**: Only 1 correction attempt for invalid JSON
- **Tool expansion**: Adding new tools requires registry update + schema definition
- **No tool chaining**: Tools execute independently; no cross-tool results yet

## References

- `TOOL_REGISTRY` constant: Line ~405-440
- Validation function: `_validate_and_normalize_tool_call()` (~437)
- Execution function: `_execute_tool()` (~516)
- Summarization: `_summarize_tool_result()` (~486)
- Main loop: `run_structured_tool_loop()` (~577)
- `/chat` integration: (~1395)
