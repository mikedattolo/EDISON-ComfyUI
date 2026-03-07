"""
Edison Swarm Engine — Collaborative multi-agent orchestration with Boss agent,
direct @agent messaging, user-in-the-loop intervention, and session state.

Architecture:
  SwarmSession     — tracks all state for one swarm invocation (agents, rounds, scratchpad, votes)
  SwarmEngine      — stateless dispatcher: runs rounds, handles direct messages, synthesizes
  Boss agent       — rewrites user request into structured plan, delegates, judges final output
  @agent routing   — user can prefix messages with @Designer, @Coder etc. to talk directly to one agent
  User-in-the-loop — after each round the frontend can inject user feedback before continuing
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Agent Catalog
# ──────────────────────────────────────────────────────────────────────────────

AGENT_CATALOG_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "Boss",
        "icon": "👔",
        "role": "project lead who rewrites the user request into a clear plan, delegates tasks to specialists, "
                "resolves disagreements, and delivers the final verdict",
        "style": "authoritative but fair — restate the goal, assign sub-tasks, call out the best ideas, "
                 "merge conflicting opinions into a single actionable answer",
        "model_preference": ["deep", "medium", "fast"],
        "always_include": True,     # Boss is always in the swarm
        "is_boss": True,
    },
    {
        "name": "Researcher",
        "icon": "🔍",
        "role": "research specialist",
        "style": "cite sources and emphasize evidence",
        "model_preference": ["medium", "deep", "fast"],
        "keywords": r"research|market|trend|compare|benchmark|stats|insight",
    },
    {
        "name": "Searcher",
        "icon": "🌐",
        "role": "web search specialist who summarizes current information",
        "style": "summarize findings with links and dates",
        "model_preference": ["medium", "deep", "fast"],
        "keywords": r"search|web|internet|latest|current|news|today",
    },
    {
        "name": "Analyst",
        "icon": "🧠",
        "role": "strategic analyst",
        "style": "structured, decisive recommendations",
        "model_preference": ["deep", "medium", "fast"],
        "always_include": True,
    },
    {
        "name": "Implementer",
        "icon": "⚙️",
        "role": "implementation specialist",
        "style": "actionable steps and concrete details",
        "model_preference": ["fast", "medium", "deep"],
        "always_include": True,
    },
    {
        "name": "Coder",
        "icon": "💻",
        "role": "coding specialist focused on implementation details",
        "style": "code-first with minimal prose",
        "model_preference": ["medium", "fast", "deep"],
        "keywords": r"code|implement|build|script|program|debug|refactor",
    },
    {
        "name": "Critic",
        "icon": "🧯",
        "role": "critical reviewer who finds flaws, risks, and missing constraints",
        "style": "skeptical, highlight risks and gaps",
        "model_preference": ["deep", "medium", "fast"],
        "keywords": r"risk|critic|review|tradeoff|cons|pitfall",
    },
    {
        "name": "Planner",
        "icon": "🧭",
        "role": "project planner who breaks work into steps and milestones",
        "style": "sequenced plan with milestones",
        "model_preference": ["medium", "deep", "fast"],
        "keywords": r"plan|roadmap|milestone|phase|timeline|steps",
    },
    {
        "name": "Designer",
        "icon": "🎨",
        "role": "UX/UI designer focusing on layout, interaction, and aesthetics",
        "style": "visual-first, UX tradeoffs",
        "model_preference": ["fast", "medium", "deep"],
        "keywords": r"design|ui|ux|layout|branding|style|theme|visual",
    },
    {
        "name": "Marketer",
        "icon": "📣",
        "role": "growth marketer focusing on positioning, audience, and messaging",
        "style": "audience, positioning, CTA",
        "model_preference": ["fast", "medium", "deep"],
        "keywords": r"marketing|position|audience|persona|copy|seo|growth",
    },
    {
        "name": "Verifier",
        "icon": "✅",
        "role": "validator who checks constraints, requirements, and correctness",
        "style": "checklists, edge cases",
        "model_preference": ["medium", "deep", "fast"],
        "keywords": r"validate|verify|test|requirements|constraints|edge cases",
    },
    {
        "name": "ProjectManager",
        "icon": "🗂️",
        "role": "project manager who turns goals into tasks, tracks status, milestones, and ownership",
        "style": "structured task lists, clear ownership, status updates",
        "model_preference": ["medium", "deep", "fast"],
        "keywords": r"project|task|manage|milestone|track|assign|kanban|sprint",
        "can_use_tools": True,
    },
    {
        "name": "FileManager",
        "icon": "📁",
        "role": "file operations specialist who writes/edits files, organizes folders, and produces deliverables",
        "style": "precise file paths, structured output, clean organization",
        "model_preference": ["medium", "deep", "fast"],
        "keywords": r"file|write|create|folder|organize|template|document|deliverable",
        "can_use_tools": True,
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Swarm Session — mutable state for a single swarm invocation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SwarmSession:
    """Persistent state for a running swarm conversation."""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    user_request: str = ""
    boss_plan: str = ""                       # Boss's rewritten plan
    agents: List[Dict[str, Any]] = field(default_factory=list)
    conversation: List[Dict[str, Any]] = field(default_factory=list)
    shared_notes: List[str] = field(default_factory=list)
    _shared_note_set: Set[str] = field(default_factory=set)
    vote_counts: Dict[str, int] = field(default_factory=dict)
    vote_summary: str = ""
    rounds_completed: int = 0
    max_rounds: int = 3
    target_rounds: int = 2                    # Boss will override this (1-3)
    status: str = "initializing"              # initializing | running | paused | voting | synthesizing | done
    user_interventions: List[Dict[str, Any]] = field(default_factory=list)  # user messages injected mid-swarm
    direct_messages: List[Dict[str, Any]] = field(default_factory=list)     # @agent DMs from user
    created_at: float = field(default_factory=time.time)
    # ── Project task & artifact tracking ──
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    _task_counter: int = 0

    # ── helpers ───────────────────────────────────────────────────────────

    def add_note(self, text: str, max_items: int = 3):
        """Extract bullet points / sentences from text and add to shared scratchpad."""
        if not text:
            return
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        notes = [line.lstrip("-•*").strip() for line in lines if line[:1] in ["-", "•", "*"]]
        if not notes:
            sentences = re.split(r"(?<=[.!?])\s+", text.strip())
            notes = [s.strip() for s in sentences if s.strip()][:2]
        for note in notes[:max_items]:
            key = " ".join(note.split()).lower()
            if key and key not in self._shared_note_set:
                self._shared_note_set.add(key)
                self.shared_notes.append(note)

    def scratchpad_text(self) -> str:
        if self.shared_notes:
            return "\n".join(f"- {n}" for n in self.shared_notes)
        return "- (empty)"

    def add_task(self, title: str, owner: str = "") -> Dict[str, Any]:
        self._task_counter += 1
        task = {"id": f"T{self._task_counter}", "title": title, "status": "todo",
                "owner_agent": owner, "files": [], "notes": ""}
        self.tasks.append(task)
        return task

    def update_task(self, task_id: str, status: str = "", notes: str = ""):
        for t in self.tasks:
            if t["id"] == task_id:
                if status:
                    t["status"] = status
                if notes:
                    t["notes"] = notes
                return t
        return None

    def add_artifact(self, path: str, kind: str = "file", summary: str = "", created_by: str = ""):
        art = {"path": path, "kind": kind, "summary": summary, "created_by": created_by}
        self.artifacts.append(art)
        return art

    def tasks_text(self) -> str:
        if not self.tasks:
            return "- (no tasks)"
        return "\n".join(f"- [{t['status']}] {t['id']}: {t['title']} (owner: {t.get('owner_agent', '?')})" for t in self.tasks)

    def discussion_text(self, max_chars: int = 3000) -> str:
        """Return discussion history, truncated to fit in prompt context."""
        lines = [
            f"{c['icon']} {c['agent']}: {c['response']}" for c in self.conversation
        ]
        text = "\n".join(lines)
        if len(text) <= max_chars:
            return text
        # Keep most recent entries that fit
        result_lines = []
        total = 0
        for line in reversed(lines):
            if total + len(line) + 1 > max_chars:
                break
            result_lines.insert(0, line)
            total += len(line) + 1
        return "\n".join(result_lines)

    def agent_names(self) -> List[str]:
        return [a["name"] for a in self.agents]

    def get_agent(self, name: str) -> Optional[Dict[str, Any]]:
        name_lower = name.lower()
        for a in self.agents:
            if a["name"].lower() == name_lower:
                return a
        return None

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "status": self.status,
            "target_rounds": self.target_rounds,
            "rounds_completed": self.rounds_completed,
            "agents": [{"name": a["name"], "icon": a["icon"], "model_name": a.get("model_name", "")} for a in self.agents],
            "conversation_count": len(self.conversation),
            "shared_notes": self.shared_notes[:10],
            "vote_summary": self.vote_summary,
            "user_interventions": len(self.user_interventions),
            "tasks": self.tasks,
            "artifacts": self.artifacts,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Swarm Engine — stateless orchestration logic
# ──────────────────────────────────────────────────────────────────────────────

class SwarmEngine:
    """Orchestrates multi-agent swarm conversations with Boss oversight."""

    def __init__(
        self,
        available_models: Callable[[], List[Tuple[str, Any, str]]],
        get_lock_for_model: Callable,
        search_tool: Any = None,
        config: Dict[str, Any] = None,
        emit_fn: Callable = None,
        execute_tool: Callable = None,
    ):
        """
        Args:
            available_models: callable returning list of (tag, model_obj, display_name)
            get_lock_for_model: callable(model_obj) → threading.Lock
            search_tool: optional web search tool
            config: edison config dict
            emit_fn: optional callable(title, status) for agent live view events
            execute_tool: optional async callable(tool_name, args, chat_id) for tool execution
        """
        self._available_models = available_models
        self._get_lock = get_lock_for_model
        self._search_tool = search_tool
        self._config = config or {}
        self._emit = emit_fn or (lambda *a, **kw: None)
        self._execute_tool = execute_tool

    # ── Public API ────────────────────────────────────────────────────────

    async def run_swarm(
        self,
        user_request: str,
        has_images: bool = False,
        search_results: List[dict] = None,
        file_request: bool = False,
        parallel: bool = False,
        chat_id: str = None,
    ) -> Tuple[SwarmSession, str]:
        """
        Full swarm execution:  Boss plan → agent rounds → voting → Boss synthesis.

        Returns (session, synthesis_prompt) — the caller streams the synthesis_prompt.
        """
        session = SwarmSession(user_request=user_request)

        # 1) Select & assign models to agents
        self._select_agents(session, user_request)
        self._emit(
            title=f"Agents: {', '.join(a['icon'] + ' ' + a['name'] for a in session.agents)}",
            status="done",
        )

        # 2) Memory safety
        self._apply_memory_safety(session)

        # 3) Boss planning round
        session.status = "running"
        await self._boss_plan(session)

        # 4) Multi-round discussion (with optional tool execution)
        await self._run_rounds(session, parallel, search_results, chat_id=chat_id)

        # 5) Voting
        session.status = "voting"
        await self._voting_round(session)

        # 6) Boss final judgment
        session.status = "synthesizing"
        synthesis_prompt = self._build_synthesis_prompt(session, file_request)

        session.status = "done"
        return session, synthesis_prompt

    async def handle_direct_message(
        self,
        session: SwarmSession,
        target_agent_name: str,
        user_message: str,
        chat_id: str = None,
    ) -> Dict[str, Any]:
        """
        Handle @Agent direct message from the user.
        Returns the agent's response dict.
        """
        agent = session.get_agent(target_agent_name)
        if not agent:
            return {
                "agent": "System",
                "icon": "⚠️",
                "model": "N/A",
                "response": f"Agent '{target_agent_name}' not found. Available: {', '.join(session.agent_names())}",
                "is_direct": True,
            }

        # Build DM prompt — agent sees full discussion + user's direct question
        prompt = f"""You are {agent['name']}, a {agent['role']}.
Personality: {agent.get('style', 'concise and helpful')}.

The user is speaking directly to you (not the group). Respond personally.

Original Task: {session.user_request}

Discussion So Far:
{session.discussion_text() if session.conversation else '(none yet)'}

Shared Scratchpad:
{session.scratchpad_text()}

User says to you directly: {user_message}

Your response (be specific, personal, address their question):"""

        result = await self._run_agent_prompt(agent, prompt, 0.7)
        # If agent supports tools, also try tool-enabled path
        agent_defn = next((d for d in AGENT_CATALOG_DEFINITIONS if d["name"] == agent["name"]), {})
        if agent_defn.get("can_use_tools") and self._execute_tool:
            result = await self._run_agent_with_tools(agent, prompt, session, chat_id=chat_id)
        result["is_direct"] = True
        result["dm_from_user"] = user_message

        session.direct_messages.append(result)
        session.conversation.append(result)
        session.add_note(result["response"])

        return result

    async def handle_user_intervention(
        self,
        session: SwarmSession,
        user_feedback: str,
    ) -> List[Dict[str, Any]]:
        """
        User injects feedback mid-swarm. All agents see it and respond in one extra round.
        Returns list of agent responses.
        """
        session.user_interventions.append({
            "from": "user",
            "message": user_feedback,
            "timestamp": time.time(),
        })

        # All agents respond to user feedback in context of discussion
        responses = []
        for agent in session.agents:
            prompt = f"""You are {agent['name']}, a {agent['role']}.
Personality: {agent.get('style', 'concise and helpful')}.

Original Task: {session.user_request}

Discussion So Far:
{session.discussion_text()}

The user (your client/boss) just chimed in with this feedback:
"{user_feedback}"

Rules:
- Respond only in English.
- Acknowledge the user's input.
- Adjust your position if the feedback changes things.
- Keep to 2-3 sentences.

Your response:"""
            result = await self._run_agent_prompt(agent, prompt, 0.6)
            session.add_note(result["response"])
            session.conversation.append(result)
            responses.append(result)
            self._emit(
                title=f"{result['icon']} {result['agent']} (feedback): {result['response'][:100]}",
                status="done",
            )

        return responses

    @staticmethod
    def parse_direct_mention(message: str) -> Tuple[Optional[str], str]:
        """
        Parse @AgentName from user message.
        Returns (agent_name_or_None, remaining_message).
        """
        match = re.match(r"^@(\w+)\s*(.*)", message.strip(), re.DOTALL)
        if match:
            return match.group(1), match.group(2).strip()
        return None, message

    # ── Internal helpers ──────────────────────────────────────────────────

    def _select_agents(self, session: SwarmSession, user_text: str):
        """Pick agents from catalog based on user intent."""
        text_lower = user_text.lower()
        selected_names: Set[str] = set()

        for defn in AGENT_CATALOG_DEFINITIONS:
            if defn.get("always_include"):
                selected_names.add(defn["name"])
            elif defn.get("keywords") and re.search(defn["keywords"], text_lower):
                selected_names.add(defn["name"])

        # Always include Critic for complex tasks
        if len(text_lower) > 200:
            selected_names.add("Critic")

        # Cap at 6 agents (Boss + 5 specialists)
        max_agents = 6
        if len(selected_names) > max_agents:
            priority = [d["name"] for d in AGENT_CATALOG_DEFINITIONS]
            selected_names = set([n for n in priority if n in selected_names][:max_agents])

        # Build agent instances with model assignment
        used_models: Set[str] = set()
        catalog_lookup = {d["name"]: d for d in AGENT_CATALOG_DEFINITIONS}

        for name in [d["name"] for d in AGENT_CATALOG_DEFINITIONS]:  # preserve catalog order
            if name not in selected_names:
                continue
            defn = catalog_lookup[name]
            model, model_name = self._pick_model(defn.get("model_preference", ["fast"]), used_models)
            session.agents.append({
                "name": name,
                "icon": defn["icon"],
                "role": defn["role"],
                "style": defn["style"],
                "model": model,
                "model_name": model_name,
                "is_boss": defn.get("is_boss", False),
            })

        logger.info(f"🐝 Swarm agents: {', '.join(a['name'] for a in session.agents)}")

    def _pick_model(self, preference: List[str], used: Set[str]) -> Tuple[Any, str]:
        available = self._available_models()
        # Try unused first
        for pref in preference:
            for tag, model, name in available:
                if tag == pref and tag not in used:
                    used.add(tag)
                    return model, name
        # Fallback to any available
        for pref in preference:
            for tag, model, name in available:
                if tag == pref:
                    used.add(tag)
                    return model, name
        # Last resort: first available
        if available:
            tag, model, name = available[0]
            return model, name
        return None, "No Model"

    def _apply_memory_safety(self, session: SwarmSession):
        """Check VRAM and use shared single model for swarm to prevent OOM."""
        try:
            from services.edison_core.swarm_safety import (
                get_swarm_memory_policy, apply_degraded_mode, _flush_gpu,
            )
            # Always flush before swarm starts
            _flush_gpu()

            policy = get_swarm_memory_policy()
            available = self._available_models()
            loaded_map = {tag: True for tag, _, _ in available}
            decision = policy.assess(session.agents, loaded_map)
            logger.info(f"🐝 Memory safety: {decision}")

            # For swarm, always use a single shared model to avoid multi-model OOM.
            # Pick the fastest available model (lowest VRAM) for all agents.
            priority_order = ["fast", "medium", "deep"]
            chosen = available[0] if available else (None, None, "fallback")
            for pref in priority_order:
                for tag, model, name in available:
                    if tag == pref:
                        chosen = (tag, model, name)
                        break
                if chosen[0] == pref:
                    break
            apply_degraded_mode(session.agents, chosen[1], chosen[2])
            logger.info(f"🐝 Swarm using shared model: {chosen[2]} (prevents multi-model OOM)")
        except Exception as e:
            logger.debug(f"Swarm safety check skipped: {e}")

    async def _boss_plan(self, session: SwarmSession):
        """Boss agent rewrites user request into a structured plan."""
        boss = next((a for a in session.agents if a.get("is_boss")), None)
        if not boss:
            session.boss_plan = session.user_request
            return

        agent_roster = ", ".join(
            f"{a['icon']} {a['name']} ({a['role']})" for a in session.agents if not a.get("is_boss")
        )

        plan_prompt = f"""You are the Boss — the project lead. The user just submitted a request.

Your job:
1. Restate the goal clearly in 1-2 sentences.
2. Decide how many discussion rounds the team needs (1-3).
   - 1 round: simple task, factual question, or straightforward request
   - 2 rounds: moderate complexity, needs iteration
   - 3 rounds: complex multi-part task needing deep refinement
3. Break the task into 2-5 sub-tasks.
4. Assign each sub-task to one or more of your team members.
5. If the task requires creating files or code, include a TASKS_JSON block.

Your team:
{agent_roster}

User Request: {session.user_request}

Rules:
- Respond only in English.
- You MUST include **Rounds**: <number> (a single integer 1-3).
- Use this format:
  **Goal**: <restated goal>
  **Rounds**: <number>
  **Plan**:
  1. <task> → @AgentName
  2. <task> → @AgentName, @AgentName
  ...
- If file creation or code tasks are needed, also include:
  TASKS_JSON:
  {{"tasks":[{{"id":"T1","owner":"AgentName","action":"description"}}]}}
- Be concise. Maximum 15 lines total.

Your plan:"""

        self._emit(title="👔 Boss is creating a plan...", status="running")
        result = await self._run_agent_prompt(boss, plan_prompt, 0.5)
        session.boss_plan = result["response"]
        session.conversation.append(result)
        session.add_note(result["response"])

        # Parse Boss's round decision
        rounds_match = re.search(r"\*\*Rounds\*\*\s*[:：]\s*(\d+)", session.boss_plan)
        if not rounds_match:
            # Fallback: look for "Rounds: N" without bold
            rounds_match = re.search(r"[Rr]ounds\s*[:：]\s*(\d+)", session.boss_plan)
        if rounds_match:
            boss_rounds = int(rounds_match.group(1))
            boss_rounds = max(1, min(boss_rounds, session.max_rounds))  # clamp 1-3
            session.target_rounds = boss_rounds
            logger.info(f"👔 Boss decided {boss_rounds} round(s)")
        else:
            logger.info(f"👔 Boss didn't specify rounds, defaulting to {session.target_rounds}")

        self._emit(title=f"👔 Boss plan ready — {session.target_rounds} round(s)", status="done")
        logger.info(f"👔 Boss plan: {session.boss_plan[:120]}...")

        # Parse TASKS_JSON block if present
        tasks_match = re.search(r"TASKS_JSON\s*:\s*(\{.*\})", session.boss_plan, re.DOTALL)
        if tasks_match:
            try:
                tasks_data = json.loads(tasks_match.group(1))
                for t in tasks_data.get("tasks", []):
                    session.add_task(
                        title=t.get("action", t.get("title", "")),
                        owner=t.get("owner", ""),
                    )
                logger.info(f"👔 Boss created {len(session.tasks)} task(s)")
            except (json.JSONDecodeError, ValueError):
                pass

    async def _run_rounds(
        self,
        session: SwarmSession,
        parallel: bool,
        search_results: List[dict] = None,
        chat_id: str = None,
    ):
        """Execute multi-round agent discussion with VRAM-safe inter-round flushing."""
        from services.edison_core.swarm_safety import _flush_gpu, _get_free_vram_mb
        user_msg_truncated = _truncate(session.user_request, 2500)
        wants_search = bool(re.search(
            r"search|web|internet|latest|current|news|today|research|sources",
            session.user_request.lower(),
        ))

        # ── Round 1 ─────────────────────────────────────────────────────
        self._emit(title="🐝 Round 1 — initial perspectives", status="running")
        round1_prompts = []
        non_boss_agents = [a for a in session.agents if not a.get("is_boss")]

        for agent in non_boss_agents:
            shared_signals = self._relevant_shared_signals(session, agent["name"])
            search_block = ""
            if agent["name"] == "Searcher" and self._search_tool and wants_search:
                search_block = self._do_web_search(session.user_request)

            prompt = f"""You are {agent['name']}, a {agent['role']}. You're in a collaborative discussion led by the Boss.
Personality: {agent.get('style', 'concise and helpful')}.

User Request: {user_msg_truncated}

Boss's Plan:
{session.boss_plan}

Shared Scratchpad (read/write):
{session.scratchpad_text()}
Relevant Shared Signals:
{shared_signals}
{search_block}

Rules:
- Respond only in English.
- Be specific and concise (2-3 sentences).
- Provide unique insights from your role.
- Focus on the sub-tasks the Boss assigned to you (if any).

Your initial perspective:"""
            round1_prompts.append((agent, prompt))

        results = await self._execute_prompts(round1_prompts, parallel, 0.7, session=session, chat_id=chat_id)
        for result in results:
            session.add_note(result["response"])
            session.conversation.append(result)
            self._emit(title=f"{result['icon']} {result['agent']}: {result['response'][:100]}", status="done")

        session.rounds_completed = 1
        # Flush GPU after round 1
        _flush_gpu()

        # ── Auto-round adjustment (Jaccard similarity) ───────────────────
        # If round 1 responses are too similar and Boss chose few rounds, nudge up by 1
        round1_responses = [r["response"] for r in results]
        if _avg_jaccard(round1_responses) > 0.45 and session.target_rounds < session.max_rounds:
            session.target_rounds = min(session.target_rounds + 1, session.max_rounds)
            logger.info(f"🐝 Auto-round: bumped to {session.target_rounds} (responses too similar)")

        # ── Rounds 2+ ───────────────────────────────────────────────────
        for round_idx in range(2, session.target_rounds + 1):
            # Flush GPU cache between rounds to prevent OOM accumulation
            _flush_gpu()
            free_mb = _get_free_vram_mb()
            if free_mb > 0 and free_mb < 2000:
                logger.warning(f"🐝 Low VRAM ({free_mb:.0f}MB free) — stopping early at round {round_idx - 1}")
                break

            self._emit(title=f"🐝 Round {round_idx} — refining", status="running")
            round_prompts = []

            for agent in non_boss_agents:
                shared_signals = self._relevant_shared_signals(session, agent["name"])
                prompt = f"""You are {agent['name']}, continuing the discussion.
Personality: {agent.get('style', 'concise and helpful')}.

User Request: {user_msg_truncated}

Boss's Plan:
{session.boss_plan}

Other experts said:
{session.discussion_text()}

Shared Scratchpad (read/write):
{session.scratchpad_text()}
Relevant Shared Signals:
{shared_signals}

Rules:
- Respond only in English.
- Address at least one specific point from another agent by name.
- Add one new insight not previously mentioned.
- Keep it to 2-3 sentences.

Your refined contribution:"""
                round_prompts.append((agent, prompt))

            round_results = await self._execute_prompts(round_prompts, parallel, 0.6, session=session, chat_id=chat_id)
            for result in round_results:
                session.add_note(result["response"])
                session.conversation.append({
                    **result,
                    "agent": f"{result['agent']} (Round {round_idx})",
                })
                self._emit(
                    title=f"{result['icon']} {result['agent']} R{round_idx}: {result['response'][:100]}",
                    status="done",
                )

            session.rounds_completed = round_idx

            # Early stop if converged
            recent = [r["response"] for r in round_results]
            if _avg_jaccard(recent) > 0.6:
                logger.info("🐝 Auto-stop: responses converged")
                break

    async def _voting_round(self, session: SwarmSession):
        """Each agent votes for top 2 contributions."""
        # Flush GPU before voting round
        try:
            from services.edison_core.swarm_safety import _flush_gpu
            _flush_gpu()
        except Exception:
            pass
        self._emit(title="🗳️ Voting round", status="running")

        # Get latest response per agent
        latest: Dict[str, Dict] = {}
        for entry in session.conversation:
            base_name = entry["agent"].split(" (Round ")[0]
            if base_name != "Boss":
                latest[base_name] = entry

        candidates = list(latest.keys())
        if not candidates:
            session.vote_summary = "No votes (no contributions)"
            return

        candidate_summaries = "\n".join(
            f"{name}: {latest[name].get('response', '')[:160]}" for name in candidates
        )

        vote_counts: Dict[str, int] = {name: 0 for name in candidates}

        for agent in session.agents:
            vote_prompt = f"""You are {agent['name']}. Vote for the top 2 agent contributions (excluding yourself if possible).

Candidates:
{candidate_summaries}

Rules:
- Respond only in English.
- Reply with two agent names separated by a comma.
- Do not include any other text.

Your vote:"""
            result = await self._run_agent_prompt(agent, vote_prompt, 0.2, max_tokens=50)
            picks = [p.strip() for p in result["response"].split(",") if p.strip()]
            for pick in picks[:2]:
                for name in candidates:
                    if name.lower() in pick.lower():
                        vote_counts[name] += 1
                        break

        session.vote_counts = vote_counts
        sorted_votes = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)

        if sorted_votes:
            top_score = sorted_votes[0][1]
            tied = [name for name, score in sorted_votes if score == top_score and top_score > 0]
            if len(tied) > 1:
                self._emit(title=f"🗳️ Tie detected ({', '.join(tied)}). Running reasoned re-vote.", status="running")
                revote_counts = {name: 0 for name in tied}
                for agent in session.agents:
                    revote_prompt = f"""You are {agent['name']}. The vote is tied.

Tied candidates: {', '.join(tied)}

Provide:
1) One short reason (max 20 words) for your best candidate.
2) One final vote using this exact format:
   VOTE: <AgentName>

No other formatting.
"""
                    result = await self._run_agent_prompt(agent, revote_prompt, 0.2, max_tokens=80)
                    match = re.search(r"VOTE\s*:\s*(.+)", result.get("response", ""), re.IGNORECASE)
                    pick_raw = (match.group(1).strip() if match else "")
                    pick = next((name for name in tied if name.lower() in pick_raw.lower()), None)
                    if pick:
                        revote_counts[pick] += 1

                sorted_revote = sorted(revote_counts.items(), key=lambda x: x[1], reverse=True)
                if sorted_revote:
                    top_revote = sorted_revote[0][1]
                    finalists = [n for n, s in sorted_revote if s == top_revote]
                    if len(finalists) > 1:
                        # Boss adjudicates if tie persists after reasoned re-vote
                        boss = next((a for a in session.agents if a.get("is_boss")), None)
                        chosen = finalists[0]
                        if boss:
                            boss_prompt = f"""You are Boss. A final tie remains after a reasoned re-vote.
Choose exactly one winner from: {', '.join(finalists)}
Reply with only the agent name."""
                            boss_pick = await self._run_agent_prompt(boss, boss_prompt, 0.1, max_tokens=20)
                            picked = boss_pick.get("response", "").strip()
                            match_name = next((n for n in finalists if n.lower() in picked.lower()), None)
                            if match_name:
                                chosen = match_name
                        revote_counts[chosen] += 1  # force decisive outcome
                    vote_counts = {**vote_counts, **{k: vote_counts.get(k, 0) + v for k, v in revote_counts.items()}}
                    sorted_votes = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)

        winners = ", ".join(f"{name} ({count})" for name, count in sorted_votes if count > 0)
        session.vote_summary = f"Vote results (decisive): {winners or 'No clear consensus'}"
        self._emit(title=f"🗳️ {session.vote_summary}", status="done")

    def _relevant_shared_signals(self, session: SwarmSession, agent_name: str) -> str:
        """Return compact, relevant shared notes for an agent to improve collaboration."""
        notes = session.shared_notes[-10:]
        if not notes:
            return "- (none yet)"

        tokens = set(re.findall(r"[a-zA-Z]+", agent_name.lower()))
        scored = []
        for note in notes:
            note_tokens = set(re.findall(r"[a-zA-Z]+", note.lower()))
            overlap = len(tokens & note_tokens)
            scored.append((overlap, note))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [n for _, n in scored[:4]]
        return "\n".join(f"- {n}" for n in top)

    def _build_synthesis_prompt(self, session: SwarmSession, file_request: bool = False) -> str:
        """Build the final synthesis prompt — Boss delivers the verdict."""
        file_instruction = ""
        if file_request:
            file_instruction = (
                "\n\nIf the user asked for downloadable files, output a FILES block. "
                "Use .pptx for presentations, .docx for Word documents, .pdf for PDFs. "
                "Write FULL content. Do NOT repeat content."
            )

        # Include user interventions summary
        intervention_block = ""
        if session.user_interventions:
            lines = [f"- User said: \"{i['message']}\"" for i in session.user_interventions]
            intervention_block = f"\n\nUser Feedback During Discussion:\n" + "\n".join(lines)

        return f"""You are the Boss synthesizing a collaborative discussion between your expert team.

User Request: {_truncate(session.user_request, 2500)}

Your Original Plan:
{session.boss_plan}

Expert Discussion:
{session.discussion_text()}

Shared Scratchpad:
{session.scratchpad_text()}

Task Status:
{session.tasks_text()}

Artifacts Created:
{chr(10).join(f"- {a['path']} ({a['kind']}): {a['summary']}" for a in session.artifacts) if session.artifacts else '- (none)'}

Vote Summary:
{session.vote_summary}
{intervention_block}

Instructions:
- Provide a clear, actionable synthesis that integrates all perspectives.
- Call out which agent(s) had the strongest contributions (per the votes).
- If experts disagreed, explain your final decision.
- Do not repeat yourself. Keep it concise.
- Do not include multiple summaries or "Final Summary" variants.
{file_instruction}"""

    def _do_web_search(self, query: str) -> str:
        """Run web search and return formatted block."""
        try:
            if hasattr(self._search_tool, "deep_search"):
                results, _ = self._search_tool.deep_search(query.strip(), num_results=5)
            else:
                results = self._search_tool.search(query.strip(), num_results=3)
            lines = []
            for r in results or []:
                title = r.get("title", "")
                url = r.get("url", "")
                snippet = r.get("body") or r.get("snippet") or ""
                lines.append(f"- {title} ({url}) {snippet}".strip())
            if lines:
                return "\n\nWeb Search Results:\n" + "\n".join(lines)
        except Exception as e:
            logger.warning(f"Searcher web search failed: {e}")
        return ""

    async def _execute_prompts(
        self,
        prompts: List[Tuple[Dict, str]],
        parallel: bool,
        temperature: float,
        session: SwarmSession = None,
        chat_id: str = None,
    ) -> List[Dict[str, Any]]:
        """Run a batch of agent prompts (parallel or sequential), with optional tool use."""
        async def _run_one(agent, prompt):
            defn = next((d for d in AGENT_CATALOG_DEFINITIONS if d["name"] == agent["name"]), {})
            if defn.get("can_use_tools") and self._execute_tool and session:
                return await self._run_agent_with_tools(agent, prompt, session, chat_id=chat_id)
            return await self._run_agent_prompt(agent, prompt, temperature)

        if parallel:
            return list(await asyncio.gather(*[_run_one(a, p) for a, p in prompts]))
        results = []
        for agent, prompt in prompts:
            results.append(await _run_one(agent, prompt))
        return results

    async def _run_agent_prompt(
        self,
        agent: Dict[str, Any],
        prompt: str,
        temperature: float,
        max_tokens: int = 300,
    ) -> Dict[str, Any]:
        """Run a single agent prompt with CJK retry and OOM recovery."""
        def _invoke(p: str, t: float) -> str:
            model = agent["model"]
            if model is None:
                return "(Model not available)"
            lock = self._get_lock(model)
            with lock:
                stream = model.create_chat_completion(
                    messages=[{"role": "user", "content": p}],
                    max_tokens=max_tokens,
                    temperature=t,
                    repeat_penalty=1.3,
                    frequency_penalty=0.4,
                    presence_penalty=0.3,
                    stream=False,
                )
                return stream["choices"][0]["message"]["content"]

        try:
            response = await asyncio.to_thread(_invoke, prompt, temperature)
        except Exception as e:
            err_str = str(e).lower()
            if "out of memory" in err_str or "cuda" in err_str or "oom" in err_str:
                logger.warning(f"🐝 CUDA OOM in {agent['name']}, flushing and retrying with shorter prompt")
                try:
                    from services.edison_core.swarm_safety import _flush_gpu
                    _flush_gpu()
                except Exception:
                    pass
                # Retry with truncated prompt
                short_prompt = prompt[:2000] if len(prompt) > 2000 else prompt
                try:
                    response = await asyncio.to_thread(_invoke, short_prompt, temperature)
                except Exception:
                    response = f"(Agent {agent['name']} skipped due to memory constraints)"
            else:
                logger.error(f"🐝 Agent {agent['name']} error: {e}")
                response = f"(Agent {agent['name']} encountered an error)"

        # CJK retry
        if _contains_cjk(response):
            response = await asyncio.to_thread(_invoke, prompt + "\n\nStrictly respond in English only.", 0.5)
        if _contains_cjk(response):
            response = "(Response omitted: non-English output detected.)"

        return {
            "agent": agent["name"],
            "icon": agent["icon"],
            "model": agent.get("model_name", "Unknown"),
            "response": response,
        }

    async def _run_agent_with_tools(
        self,
        agent: Dict[str, Any],
        prompt: str,
        session: SwarmSession,
        chat_id: str = None,
        max_tool_iters: int = 3,
    ) -> Dict[str, Any]:
        """Run an agent prompt that can call tools (up to max_tool_iters rounds).

        The agent outputs either plain text or a JSON tool call
        ``{"tool":"name","args":{…}}``. Results are fed back until the agent
        produces a final text answer or exhausts iterations.
        """
        if not self._execute_tool:
            return await self._run_agent_prompt(agent, prompt, 0.5)

        # Available tool subset for swarm agents
        tool_hint = (
            "\nYou may call tools by outputting ONLY a JSON object: "
            '{"tool":"<name>","args":{…}}. Available: '
            "workspace.init, fs.read, fs.write, fs.list, fs.mkdir, fs.diff, "
            "pm.create_task, pm.list_tasks, pm.update_task, "
            "code.search, code.apply_unified_diff, codespace_exec, web_search. "
            "If done, reply with plain text (no JSON)."
        )
        messages = [prompt + tool_hint]
        tool_results = []

        for i in range(max_tool_iters):
            result = await self._run_agent_prompt(agent, "\n".join(messages), 0.5, max_tokens=600)
            text = result.get("response", "")

            # Try to parse tool call JSON
            parsed = None
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                # Try extracting embedded JSON
                m = re.search(r'\{[^{}]*"tool"\s*:', text)
                if m:
                    depth, start = 0, m.start()
                    for j in range(start, len(text)):
                        if text[j] == '{': depth += 1
                        elif text[j] == '}':
                            depth -= 1
                            if depth == 0:
                                try:
                                    parsed = json.loads(text[start:j + 1])
                                except (json.JSONDecodeError, ValueError):
                                    pass
                                break

            if parsed and isinstance(parsed, dict) and "tool" in parsed:
                tool_name = parsed["tool"]
                tool_args = parsed.get("args", {})
                self._emit(
                    title=f"{agent['icon']} {agent['name']} calling {tool_name}",
                    status="running",
                )
                try:
                    tool_result = await self._execute_tool(tool_name, tool_args, chat_id)
                    summary = json.dumps(tool_result, default=str)[:800]
                except Exception as exc:
                    summary = f"Tool error: {exc}"
                    tool_result = {"ok": False, "error": str(exc)}

                tool_results.append({"tool": tool_name, "args": tool_args, "result": tool_result})
                messages.append(f"\nTool result ({tool_name}): {summary}\nContinue or give your final answer:")
                self._emit(
                    title=f"{agent['icon']} {agent['name']}: {tool_name} → done",
                    status="done",
                )
                continue

            # Plain text → final answer
            result["tool_calls"] = tool_results
            return result

        # Exhausted iterations — return last result
        result["tool_calls"] = tool_results
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Active session registry (for @agent DMs and user interventions between rounds)
# ──────────────────────────────────────────────────────────────────────────────

_active_sessions: Dict[str, SwarmSession] = {}
_SESSION_TTL = 3600  # 1 hour

def get_session(session_id: str) -> Optional[SwarmSession]:
    s = _active_sessions.get(session_id)
    if s and (time.time() - s.created_at) > _SESSION_TTL:
        _active_sessions.pop(session_id, None)
        return None
    return s

def register_session(session: SwarmSession):
    # Prune old sessions
    now = time.time()
    stale = [k for k, v in _active_sessions.items() if (now - v.created_at) > _SESSION_TTL]
    for k in stale:
        _active_sessions.pop(k, None)
    _active_sessions[session.session_id] = session

def list_sessions() -> List[dict]:
    now = time.time()
    return [
        s.to_dict() for s in _active_sessions.values()
        if (now - s.created_at) <= _SESSION_TTL
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────

def _truncate(text: str, max_chars: int = 2500) -> str:
    if not text or len(text) <= max_chars:
        return text or ""
    return text[:max_chars] + f"\n\n[TRUNCATED: {len(text)} chars total]"


def _contains_cjk(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]", text))


def _normalize_text(text: str) -> set:
    return set(re.findall(r"[a-zA-Z]+", (text or "").lower()))


def _avg_jaccard(responses: list) -> float:
    if len(responses) < 2:
        return 0.0
    sets = [_normalize_text(r) for r in responses]
    scores = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            if not union:
                continue
            scores.append(len(sets[i] & sets[j]) / len(union))
    return sum(scores) / len(scores) if scores else 0.0
