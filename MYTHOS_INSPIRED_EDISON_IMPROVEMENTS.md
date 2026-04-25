# Mythos-Inspired Improvements for Edison

This document captures practical, model-agnostic ideas inspired by public discussion around newer agentic models and applies them to Edison.

## 1) Add a Router + Specialist Pattern
Instead of one monolithic prompt, route each user request into one of a few specialist modes:
- **Chat mode**: direct answer, low latency.
- **Builder mode**: multi-step planning and file edits.
- **Research mode**: browsing + citations + source quality checks.
- **Safety mode**: strict refusal/deferral for risky requests.

Implementation notes:
- Use a lightweight intent classifier before main generation.
- Persist chosen mode in request metadata for analytics.
- Allow user override ("force research mode").

## 2) Deliberate Thinking Budget (Optional)
Introduce configurable depth levels:
- **Fast** (minimal reasoning, lower cost)
- **Balanced**
- **Deep** (better for coding and architecture)

Implementation notes:
- Map depth to max tokens, tool budget, and retry count.
- Auto-upgrade to Deep for high-complexity tasks (large diffs, vague requirements, failing tests).

## 3) Self-Critique Pass Before Final Output
For non-trivial tasks, run a short internal rubric pass before answering:
- Did we answer all user asks?
- Are instructions from AGENTS/README/tool constraints followed?
- Are claims backed by source/tests?

Implementation notes:
- Keep this pass hidden from users; only output polished result.
- Add structured checks to reduce partial responses.

## 4) Tool Reliability Layer
Improve tool use quality with defensive wrappers:
- Retry transient failures.
- Normalize and validate tool outputs.
- Add explicit "cannot verify" states instead of hallucinating.

Implementation notes:
- Add typed tool contracts (required fields, schema validation).
- Track per-tool error rate and latency.

## 5) Evidence-First Responses
When Edison gives factual guidance, prefer an evidence payload:
- Sources used
- Confidence score
- What could be stale

Implementation notes:
- Standardize citation format and attach to answers.
- Include exact dates for time-sensitive claims.

## 6) Memory with Expiration and User Control
Keep memory useful but predictable:
- Store user preferences, project context, and constraints.
- Apply TTL for volatile facts.
- Let users view/edit/delete memory entries.

Implementation notes:
- Separate "profile memory" from "task memory".
- Require explicit user confirmation before saving sensitive items.

## 7) Structured Output Contracts
For automations, always support JSON schema outputs in addition to prose.

Implementation notes:
- Define response schemas per feature (e.g., plan, patch summary, test results).
- Reject malformed outputs and trigger one auto-repair attempt.

## 8) Better Failure UX
When Edison cannot complete a task, it should still provide value:
- Explain blocker clearly.
- Provide the next best action.
- Offer a fallback path.

Implementation notes:
- Add reusable fallback templates in the response layer.

## 9) Evaluation Harness (Critical)
Add recurring evals for real Edison workflows:
- Coding tasks (patch correctness, tests pass).
- Retrieval tasks (citation precision/recall).
- Agentic tasks (multi-step success rate, tool efficiency).

Implementation notes:
- Store eval set in version control.
- Run nightly and before major prompt/model changes.

## 10) Rollout Plan
1. Implement router + depth controls + evidence format first.
2. Add self-critique and tool reliability wrappers.
3. Ship memory controls and structured output validation.
4. Stand up eval harness and gate releases by score.

## Suggested Priority for Edison (highest impact first)
1. Router + specialist modes
2. Evidence-first responses with citations and dates
3. Self-critique pass
4. Tool reliability wrappers
5. Eval harness

---

## Important Note
Do not copy proprietary model internals or leaked material. Focus on high-level, ethically-sourced patterns that are broadly known in modern agent systems.
