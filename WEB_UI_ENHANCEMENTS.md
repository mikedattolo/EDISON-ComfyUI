# EDISON Web UI Enhancement Plan

## Implemented Features (app_enhanced.js)

### 1. ✅ Scrollbar in Chat
- Messages container already has CSS for scrolling
- Added `scrollToBottom()` with requestAnimationFrame for smooth scrolling

### 2. ✅ Stop Button
- Added `stopBtn` that shows during generation
- Implements AbortController to cancel ongoing requests
- Shows "Response generation stopped by user" message

### 3. ✅ Regenerate/Edit Functionality
- **Edit**: Click edit button on user messages to load text back into input
- **Regenerate**: Click regenerate button on assistant messages to resend
- **Copy**: Copy assistant responses to clipboard

### 4. ✅ Smart Chat Titles
- Generates titles from first 6 words of conversation
- Max 40 characters
- Updates automatically after first exchange

### 5. ✅ Collapsible Sidebar
- Toggle button collapses/expands sidebar
- Persists state
- Smooth animations

### 6. ❌ RAG Memory Fix (Needs Backend Work)
**Problem**: RAG stores but doesn't retrieve context properly

**Solution Required in `services/edison_core/app.py`**:
```python
# Current code retrieves but doesn't include in prompt effectively
# Need to enhance build_full_prompt() to better integrate retrieved context
```

### 7. ❌ Web Search (Needs New Backend Endpoint)
**Required**: 
- New endpoint `/search` in edison_core
- Integration with DuckDuckGo or similar API
- Add search results to context before LLM call

### 8. ❌ Enhanced Agent Mode (Needs Tool Integration)
**Required**:
- Tool execution framework
- Python code execution sandbox
- File system operations (safe mode)

### 9. ✅ ChatGPT/Claude-like Features
- Message actions (copy, regenerate, edit)
- Collapsible sidebar
- Better chat management with delete
- Notification system
- Improved formatting (links, code blocks)

## Next Steps to Complete

### 1. Update HTML (web/index.html)
Add these elements:
```html
<!-- Add sidebar toggle button -->
<button id="sidebarToggle" class="sidebar-toggle">☰</button>

<!-- Add stop button next to send button -->
<button id="stopBtn" class="stop-btn" style="display: none;">
    <svg><!-- stop icon --></svg>
</button>

<!-- Update delete buttons in chat history items -->
```

### 2. Update CSS (web/styles.css)
Add styles for:
- `.sidebar.collapsed`
- `.stop-btn`
- `.message-actions`
- `.notification`
- `.delete-chat-btn`

### 3. Fix RAG Memory (backend)
Update `services/edison_core/rag.py`:
```python
class RAGSystem:
    def get_context(self, query: str, n_results: int = 3):
        # Current implementation
        results = self.collection.query(
            query_embeddings=[self.embed(query)],
            n_results=n_results
        )
        
        # Return WITH metadata for better context
        return [(doc, meta) for doc, meta in zip(
            results['documents'][0],
            results['metadatas'][0]
        )]
```

Update `services/edison_core/app.py`:
```python
def build_full_prompt(system_prompt: str, user_message: str, context_chunks: list) -> str:
    """Build the complete prompt with context"""
    parts = [system_prompt, ""]
    
    if context_chunks:
        parts.append("=== RELEVANT CONTEXT FROM MEMORY ===")
        for i, (chunk, metadata) in enumerate(context_chunks, 1):
            context_type = metadata.get('type', 'unknown')
            parts.append(f"[{context_type.upper()}] {chunk}")
        parts.append("=== END CONTEXT ===")
        parts.append("")
    
    parts.append(f"User: {user_message}")
    parts.append("Assistant:")
    
    return "\n".join(parts)
```

### 4. Add Web Search Capability
Create new file `services/edison_core/search.py`:
```python
import requests
from bs4 import BeautifulSoup

class WebSearchTool:
    def search(self, query: str, num_results: int = 3):
        """Search DuckDuckGo and return results"""
        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        # Implementation here
        pass
```

Add to `app.py`:
```python
@app.post("/search")
async def web_search(query: str):
    """Web search endpoint"""
    search_tool = WebSearchTool()
    results = search_tool.search(query)
    return {"results": results}
```

### 5. Deploy Changes
```bash
# Replace the current app.js with enhanced version
mv web/app_enhanced.js web/app.js

# Update HTML and CSS (see templates above)

# Commit and push
git add web/
git commit -m "feat: Add enhanced UI with stop button, regenerate, edit, collapsible sidebar"
git push origin main

# Deploy to AI PC
cd /opt/edison
sudo git pull origin main
sudo systemctl restart edison-web
```

## Priority Order

1. **HIGH**: Scrollbar fix (CSS only)
2. **HIGH**: Stop button (already implemented in enhanced JS)
3. **HIGH**: Collapsible sidebar (add HTML + CSS)
4. **MEDIUM**: Regenerate/edit (already implemented)
5. **MEDIUM**: Smart titles (already implemented)
6. **MEDIUM**: Fix RAG memory (backend changes)
7. **LOW**: Web search (new feature, significant work)
8. **LOW**: Enhanced agent mode (complex, needs sandboxing)

## Testing Checklist

- [ ] Chat scrolls properly when messages exceed container
- [ ] Stop button appears during generation
- [ ] Stop button cancels request
- [ ] Edit button loads message back into input
- [ ] Regenerate button resends previous message
- [ ] Copy button copies to clipboard
- [ ] Sidebar collapses/expands smoothly
- [ ] Chat titles update after first message
- [ ] Delete chat button works
- [ ] RAG retrieves and displays memory correctly
