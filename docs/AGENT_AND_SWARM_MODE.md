# Agent, Work Mode, and Swarm Notes

Agent Live View now has a small recoverable session model:

- `POST /agent/sessions`
- `GET /agent/sessions`
- `GET /agent/sessions/{session_id}`
- `POST /agent/sessions/{session_id}/state`
- `GET /agent/stream?session_id=...`
- `WS /ws/agent`

The session state tracks objective, running/paused/completed/failed state, recent actions, artifacts, and error state. Browser events, log events, and file diffs update the session store and continue to stream through SSE/WebSocket.

Swarm Mode already includes a Boss agent, shared notes, task/artifact scaffolding, memory-aware scheduling, and user feedback/direct-message paths. This pass hardens failure handling so one failed agent returns a structured failed contribution and the rest of the swarm can continue with partial results.

Unsafe autonomous external writes should still be gated by the existing Edison tool permissions and confirmation model.

