"""
Knowledge Manager for EDISON
Unified intelligence layer that coordinates all knowledge sources.

Inspired by how ChatGPT/Claude/Gemini handle knowledge:
1. Multi-source retrieval: conversation memory + knowledge base + web search + Wikipedia
2. Automatic learning: extract and store facts from conversations AND search results
3. Smart routing: decide when to use cached knowledge vs live search
4. Fact deduplication and conflict resolution
5. Relevance-scored context assembly
6. Adaptive retrieval depth based on query complexity
"""

import logging
import re
import time
import hashlib
import threading
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    """A piece of context retrieved from any source"""
    text: str
    source: str  # "memory", "knowledge", "web_search", "wikipedia", "search_cache"
    score: float = 0.0
    title: str = ""
    url: str = ""
    metadata: Dict = field(default_factory=dict)
    is_fresh: bool = False  # True if from live search


class KnowledgeManager:
    """
    Unified knowledge orchestrator — the "brain" behind EDISON's intelligence.
    
    Like ChatGPT's architecture:
    - Retrieval pipeline with multiple stages
    - Automatic fact extraction and storage (learns from every interaction)
    - Smart caching of web search results
    - Conflict resolution for contradictory facts
    - Query understanding and expansion for better retrieval
    """

    # Query complexity thresholds
    SIMPLE_QUERY_WORDS = 5
    COMPLEX_QUERY_WORDS = 15

    def __init__(self, rag_system=None, knowledge_base=None, search_tool=None):
        self.rag = rag_system
        self.kb = knowledge_base
        self.search = search_tool
        self._lock = threading.Lock()

        # Learning statistics
        self.stats = {
            "queries_processed": 0,
            "facts_learned": 0,
            "search_results_cached": 0,
            "knowledge_hits": 0,
            "memory_hits": 0,
            "web_search_count": 0,
        }

        logger.info("✓ Knowledge Manager initialized")

    def update_components(self, rag_system=None, knowledge_base=None, search_tool=None):
        """Update component references (for lazy initialization)"""
        if rag_system:
            self.rag = rag_system
        if knowledge_base:
            self.kb = knowledge_base
        if search_tool:
            self.search = search_tool

    # ── Main Retrieval Pipeline ───────────────────────────────────────────

    def retrieve_context(
        self,
        query: str,
        chat_id: Optional[str] = None,
        max_results: int = 6,
        include_web_search: bool = False,
        search_if_needed: bool = True,
        min_relevance: float = 0.30
    ) -> List[RetrievedContext]:
        """
        Multi-source context retrieval pipeline.
        
        Order of operations (like GPT-4 retrieval):
        1. Check conversation memory (RAG) for personal/contextual info
        2. Check knowledge base for factual/encyclopedic info
        3. Check search cache for previously retrieved web info
        4. If insufficient, do live web search and cache results
        5. Merge, deduplicate, and rank all results
        
        Args:
            query: User's query
            chat_id: Current chat ID for scoped memory search
            max_results: Maximum context chunks to return
            include_web_search: Force web search even if we have cached results
            search_if_needed: Auto-trigger web search if knowledge is insufficient
            min_relevance: Minimum relevance score to include
            
        Returns:
            List of RetrievedContext objects, sorted by relevance
        """
        self.stats["queries_processed"] += 1
        all_contexts = []

        # Analyze query to determine retrieval strategy
        query_type = self._classify_query(query)
        logger.info(f"Query classification: {query_type} for: {query[:80]}")

        # 1. Conversation memory (RAG) — personal facts, prior conversations
        if self.rag and self.rag.is_ready():
            memory_contexts = self._retrieve_from_memory(query, chat_id, max_results=4)
            all_contexts.extend(memory_contexts)
            if memory_contexts:
                self.stats["memory_hits"] += 1

        # 2. Knowledge base — Wikipedia, uploaded docs, cached knowledge
        if self.kb and self.kb.is_ready():
            kb_contexts = self._retrieve_from_knowledge(query, max_results=4)
            all_contexts.extend(kb_contexts)
            if kb_contexts:
                self.stats["knowledge_hits"] += 1

        # 3. Determine if web search is needed
        needs_search = include_web_search or (
            search_if_needed and self._should_search_web(query, query_type, all_contexts)
        )

        if needs_search and self.search:
            web_contexts = self._retrieve_from_web(query, max_results=5)
            all_contexts.extend(web_contexts)
            self.stats["web_search_count"] += 1

        # 4. Merge, deduplicate, and rank
        merged = self._merge_and_rank(all_contexts, min_relevance)

        logger.info(
            f"Knowledge retrieval: {len(merged)} results "
            f"(memory={sum(1 for c in merged if c.source=='memory')}, "
            f"knowledge={sum(1 for c in merged if c.source in ('knowledge','wikipedia'))}, "
            f"web={sum(1 for c in merged if c.source in ('web_search','search_cache'))})"
        )

        return merged[:max_results]

    # ── Source-Specific Retrievers ────────────────────────────────────────

    def _retrieve_from_memory(self, query: str, chat_id: Optional[str], max_results: int = 4) -> List[RetrievedContext]:
        """Retrieve from conversation memory (existing RAG system)"""
        results = []
        try:
            # Chat-scoped search first
            if chat_id:
                scoped = self.rag.get_context(query, n_results=max_results, chat_id=chat_id)
                for text, meta in scoped:
                    results.append(RetrievedContext(
                        text=text,
                        source="memory",
                        score=meta.get("final_score", meta.get("score", 0)),
                        metadata=meta
                    ))

            # Global search for facts/preferences
            global_results = self.rag.get_context(query, n_results=max_results, global_search=True)
            for text, meta in global_results:
                results.append(RetrievedContext(
                    text=text,
                    source="memory",
                    score=meta.get("final_score", meta.get("score", 0)),
                    metadata=meta
                ))

        except Exception as e:
            logger.warning(f"Memory retrieval error: {e}")

        return results

    def _retrieve_from_knowledge(self, query: str, max_results: int = 4) -> List[RetrievedContext]:
        """Retrieve from knowledge base (Wikipedia, cached knowledge)"""
        results = []
        try:
            kb_results = self.kb.query(query, n_results=max_results)
            for text, meta in kb_results:
                source_type = meta.get("source_type", "knowledge")
                results.append(RetrievedContext(
                    text=text,
                    source=source_type,
                    score=meta.get("score", 0),
                    title=meta.get("title", ""),
                    url=meta.get("url", ""),
                    metadata=meta
                ))
        except Exception as e:
            logger.warning(f"Knowledge retrieval error: {e}")

        return results

    def _retrieve_from_web(self, query: str, max_results: int = 5) -> List[RetrievedContext]:
        """Do live web search, cache results, and return contexts"""
        results = []
        try:
            # Use deep_search for better coverage
            if hasattr(self.search, 'deep_search'):
                search_results, meta = self.search.deep_search(query, num_results=max_results)
            else:
                search_results = self.search.search(query, num_results=max_results)

            if not search_results:
                return results

            # Store search results in knowledge base for future retrieval
            if self.kb and self.kb.is_ready():
                stored = self.kb.add_search_results(query, search_results)
                self.stats["search_results_cached"] += stored
                logger.info(f"Cached {stored} search results for: {query[:50]}")

            # Convert to RetrievedContext
            for r in search_results:
                title = r.get('title', '')
                snippet = r.get('snippet', '')
                url = r.get('url', '')

                text = f"{title}. {snippet}" if title else snippet
                if url:
                    text += f" [Source: {url}]"

                results.append(RetrievedContext(
                    text=text,
                    source="web_search",
                    score=0.75,  # Default relevance for fresh search results
                    title=title,
                    url=url,
                    is_fresh=True,
                    metadata={"source_type": "web_search", "url": url}
                ))

        except Exception as e:
            logger.warning(f"Web search error: {e}")

        return results

    # ── Query Classification ──────────────────────────────────────────────

    def _classify_query(self, query: str) -> str:
        """
        Classify query type to determine retrieval strategy.
        
        Categories:
        - "personal": About the user (name, preferences, history)
        - "factual": Encyclopedic/knowledge base questions
        - "current": Needs fresh/real-time information
        - "technical": Code/technical questions
        - "creative": Creative tasks (less retrieval needed)
        - "general": General questions
        """
        q = query.lower()

        # Personal queries — prioritize memory
        personal_patterns = [
            r'\b(my|i|me|mine)\b.*\b(name|prefer|like|favorite|work|project|live|age|birthday)\b',
            r'\bremember\b.*\b(i|my|me)\b',
            r'\bwhat (do you|did i) (know|tell|say|mention)\b',
            r'\b(who am i|about me|my info|my profile)\b',
        ]
        if any(re.search(p, q) for p in personal_patterns):
            return "personal"

        # Current events — needs web search
        current_patterns = [
            r'\b(today|tonight|current|latest|recent|now|this week|this month|breaking)\b',
            r'\b(news|headlines|score|weather|stock|price|forecast)\b',
            r'\b(20[2-3]\d)\b',  # Year references
            r'\b(who (won|is winning)|what happened)\b',
        ]
        if any(re.search(p, q) for p in current_patterns):
            return "current"

        # Factual/encyclopedic — knowledge base
        factual_patterns = [
            r'\b(what is|who is|where is|when was|how does|explain|define|describe)\b',
            r'\b(history of|origin of|cause of|meaning of|purpose of)\b',
            r'\b(how (many|much|long|far|old|tall|big))\b',
            r'\b(capital|population|president|founder|inventor|discovery)\b',
            r'\b(science|physics|chemistry|biology|math|geography|astronomy)\b',
        ]
        if any(re.search(p, q) for p in factual_patterns):
            return "factual"

        # Technical
        technical_patterns = [
            r'\b(code|program|function|class|api|error|bug|debug|compile|import)\b',
            r'\b(python|javascript|java|rust|sql|html|css|docker|kubernetes)\b',
            r'\b(algorithm|data structure|design pattern|architecture)\b',
        ]
        if any(re.search(p, q) for p in technical_patterns):
            return "technical"

        # Creative — minimal retrieval
        creative_patterns = [
            r'\b(write|compose|create|design|imagine|story|poem|song|paint|draw)\b',
            r'\b(generate|make me|come up with)\b',
        ]
        if any(re.search(p, q) for p in creative_patterns):
            return "creative"

        return "general"

    def _should_search_web(
        self,
        query: str,
        query_type: str,
        existing_contexts: List[RetrievedContext]
    ) -> bool:
        """
        Decide if web search is needed based on query and existing results.
        
        Logic (inspired by ChatGPT's retrieval-augmented generation):
        - Always search for "current" queries (news, weather, real-time)
        - Search if factual query has no/low-quality knowledge base results
        - Don't search for personal queries (memory-only)
        - Don't search for creative queries (no external info needed)
        """
        if query_type == "current":
            return True

        if query_type in ("personal", "creative"):
            return False

        # Search if we don't have good enough existing context
        if not existing_contexts:
            return query_type in ("factual", "technical", "general")

        # Check quality of existing results
        max_score = max((c.score for c in existing_contexts), default=0)
        if max_score < 0.45:
            return True  # Low confidence — search for better info

        # Check if we have enough relevant results
        good_results = sum(1 for c in existing_contexts if c.score > 0.40)
        if good_results < 2 and query_type in ("factual", "technical"):
            return True

        return False

    # ── Merging & Ranking ─────────────────────────────────────────────────

    def _merge_and_rank(
        self,
        contexts: List[RetrievedContext],
        min_relevance: float = 0.30
    ) -> List[RetrievedContext]:
        """
        Merge results from multiple sources, deduplicate, and rank.
        
        Scoring approach (similar to RAG fusion / reciprocal rank fusion):
        - Base similarity score from vector search
        - Source priority boost (facts > knowledge > web)
        - Freshness boost for recent web results
        - Deduplication by text similarity
        """
        if not contexts:
            return []

        # Deduplicate by normalized text
        seen = set()
        unique = []
        for ctx in contexts:
            # Create a normalized key for dedup
            key = self._normalize_for_dedup(ctx.text)
            if key not in seen:
                seen.add(key)
                unique.append(ctx)

        # Apply source-based boosting
        for ctx in unique:
            boost = 0.0

            # Source priority (personal facts are most important)
            if ctx.source == "memory":
                fact_type = ctx.metadata.get("type", "")
                if fact_type == "fact":
                    boost += 0.15  # Strong boost for extracted facts
                else:
                    boost += 0.05  # Small boost for conversation memory

            elif ctx.source in ("knowledge", "wikipedia"):
                boost += 0.10  # Knowledge base is reliable

            elif ctx.source == "web_search" and ctx.is_fresh:
                boost += 0.08  # Fresh search results are valuable

            elif ctx.source == "search_cache":
                boost += 0.03  # Cached search results, lower priority

            ctx.score = min(1.0, ctx.score + boost)

        # Filter by minimum relevance
        filtered = [c for c in unique if c.score >= min_relevance]

        # Sort by score descending
        filtered.sort(key=lambda c: c.score, reverse=True)

        return filtered

    @staticmethod
    def _normalize_for_dedup(text: str) -> str:
        """Normalize text for deduplication"""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        # Use first 100 chars as dedup key
        return text[:100]

    # ── Learning from Conversations ───────────────────────────────────────

    def learn_from_exchange(
        self,
        user_message: str,
        assistant_response: str,
        search_results: Optional[List[Dict]] = None,
        chat_id: Optional[str] = None
    ):
        """
        Extract and store knowledge from a conversation exchange.
        
        This goes beyond the existing fact extraction by:
        1. Extracting facts from BOTH user message AND assistant response
        2. Storing search results that were used
        3. Learning topic associations for better future retrieval
        4. Detecting corrections/updates to existing facts
        """
        try:
            # Extract richer facts using enhanced extraction
            facts = self._extract_enhanced_facts(user_message, assistant_response)

            if facts and self.rag and self.rag.is_ready():
                for fact in facts:
                    # Check for conflicting existing facts
                    self._resolve_fact_conflict(fact, chat_id)

                    # Store the fact
                    self.rag.add_documents(
                        documents=[fact["text"]],
                        metadatas=[{
                            "role": "fact",
                            "fact_type": fact["type"],
                            "confidence": fact["confidence"],
                            "chat_id": chat_id or "",
                            "timestamp": int(time.time()),
                            "tags": ["fact", fact["type"], "enhanced"],
                            "type": "fact",
                            "source": fact.get("source", "conversation"),
                            "original_text": fact.get("original", "")[:200]
                        }]
                    )
                    self.stats["facts_learned"] += 1

                logger.info(f"Learned {len(facts)} enhanced facts from conversation")

            # Cache search results if provided
            if search_results and self.kb and self.kb.is_ready():
                # Build a composite query from the user message
                self.kb.add_search_results(user_message, search_results)

        except Exception as e:
            logger.warning(f"Learning from exchange failed: {e}")

    def _extract_enhanced_facts(self, user_msg: str, assistant_msg: str) -> List[Dict]:
        """
        Enhanced fact extraction that goes beyond regex.
        
        Extracts from both user and assistant messages:
        - Identity facts (name, age, location)
        - Preferences and opinions
        - Technical knowledge shared
        - Corrections and updates
        - Domain-specific info (projects, tools, skills)
        """
        facts = []
        combined = f"{user_msg}\n{assistant_msg}"

        # Skip very short or question-only messages
        if len(user_msg) < 15:
            return facts

        # ── Identity facts ──
        name_patterns = [
            (r"(?:my name is|i'm|i am|call me|they call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", 0.95),
            (r"(?:name'?s|name is)\s+([A-Z][a-z]+)", 0.90),
        ]
        for pattern, conf in name_patterns:
            m = re.search(pattern, user_msg, re.IGNORECASE)
            if m:
                name = m.group(1).strip()
                if name.lower() not in ('sorry', 'fine', 'ok', 'good', 'great', 'sure', 'yes', 'no',
                                         'hey', 'hi', 'hello', 'thanks', 'please', 'the', 'a', 'an'):
                    facts.append({
                        "type": "name",
                        "text": f"The user's name is {name}. My name is {name}. Call me {name}.",
                        "value": name,
                        "confidence": conf,
                        "source": "user_message",
                        "original": user_msg[:200]
                    })

        # ── Location facts ──
        location_patterns = [
            (r"(?:i live in|i'm from|i am from|based in|located in|i reside in)\s+([A-Z][a-zA-Z\s,]+?)(?:\.|,|!|\?|$)", 0.90),
            (r"(?:my (?:home|city|town|country|state) is)\s+(.+?)(?:\.|,|!|\?|$)", 0.85),
            (r"(?:i (?:just )?moved to)\s+(.+?)(?:\.|,|!|\?|$)", 0.85),
        ]
        for pattern, conf in location_patterns:
            m = re.search(pattern, user_msg, re.IGNORECASE)
            if m:
                location = m.group(1).strip().rstrip('.')
                if len(location) > 2 and len(location) < 100:
                    facts.append({
                        "type": "location",
                        "text": f"The user lives in {location}. My location is {location}.",
                        "value": location,
                        "confidence": conf,
                        "source": "user_message",
                        "original": user_msg[:200]
                    })

        # ── Preference facts ──
        pref_patterns = [
            (r"(?:i (?:really )?(?:love|like|enjoy|prefer|adore))\s+(.+?)(?:\.|,|!|\?|$)", 0.85),
            (r"(?:my favorite|my fave|my preferred)\s+(?:\w+\s+)?(?:is|are)\s+(.+?)(?:\.|,|!|\?|$)", 0.90),
            (r"(?:i'm (?:really )?(?:into|passionate about|interested in|a fan of))\s+(.+?)(?:\.|,|!|\?|$)", 0.85),
            (r"(?:i hate|i dislike|i can't stand|i don't like)\s+(.+?)(?:\.|,|!|\?|$)", 0.80),
        ]
        for pattern, conf in pref_patterns:
            m = re.search(pattern, user_msg, re.IGNORECASE)
            if m:
                pref = m.group(1).strip().rstrip('.')
                if len(pref) > 3 and len(pref) < 200:
                    facts.append({
                        "type": "preference",
                        "text": f"The user's preference: {pref}. I {m.group(0).strip().rstrip('.')}.",
                        "value": pref,
                        "confidence": conf,
                        "source": "user_message",
                        "original": user_msg[:200]
                    })

        # ── Work/Project facts ──
        work_patterns = [
            (r"(?:i (?:work|am working) (?:on|at|for|with|in))\s+(.+?)(?:\.|,|!|\?|$)", 0.85),
            (r"(?:my (?:job|work|profession|career|role) is)\s+(.+?)(?:\.|,|!|\?|$)", 0.90),
            (r"(?:i'm (?:a|an|the))\s+((?:software|web|data|ml|ai|full[ -]?stack|front[ -]?end|back[ -]?end|devops|cloud|mobile|systems?|network|security|database|game|embedded)\s*(?:engineer|developer|architect|scientist|analyst|Designer|administrator|admin|ops|specialist))", 0.90),
            (r"(?:my project|working on a project|building)\s+(?:called|named)?\s*(.+?)(?:\.|,|!|\?|$)", 0.85),
        ]
        for pattern, conf in work_patterns:
            m = re.search(pattern, user_msg, re.IGNORECASE)
            if m:
                work = m.group(1).strip().rstrip('.')
                if len(work) > 3 and len(work) < 200:
                    facts.append({
                        "type": "project",
                        "text": f"The user is working on: {work}.",
                        "value": work,
                        "confidence": conf,
                        "source": "user_message",
                        "original": user_msg[:200]
                    })

        # ── Technical skill/tool facts ──
        skill_patterns = [
            (r"(?:i (?:use|code in|program in|develop with|work with))\s+(.+?)(?:\.|,|!|\?|$)", 0.80),
            (r"(?:i know|i'm (?:familiar|experienced) with)\s+(.+?)(?:\.|,|!|\?|$)", 0.80),
        ]
        for pattern, conf in skill_patterns:
            m = re.search(pattern, user_msg, re.IGNORECASE)
            if m:
                skill = m.group(1).strip().rstrip('.')
                if len(skill) > 2 and len(skill) < 150:
                    facts.append({
                        "type": "skill",
                        "text": f"The user uses/knows: {skill}.",
                        "value": skill,
                        "confidence": conf,
                        "source": "user_message",
                        "original": user_msg[:200]
                    })

        # ── Facts from assistant response (learned knowledge) ──
        # Only store if the assistant provided detailed factual info
        if len(assistant_msg) > 200 and self._is_factual_response(assistant_msg):
            # Extract key sentences from assistant response
            key_sentences = self._extract_key_sentences(assistant_msg, max_sentences=3)
            if key_sentences:
                # Store as learned knowledge, not as user facts
                summary = " ".join(key_sentences)
                if len(summary) > 50:
                    facts.append({
                        "type": "learned_knowledge",
                        "text": f"Topic: {user_msg[:100]}. Key info: {summary[:500]}",
                        "value": summary[:500],
                        "confidence": 0.70,
                        "source": "assistant_response",
                        "original": user_msg[:200]
                    })

        # Limit to top 5 facts
        facts.sort(key=lambda f: f["confidence"], reverse=True)
        return facts[:5]

    @staticmethod
    def _is_factual_response(text: str) -> bool:
        """Check if an assistant response contains factual/educational content"""
        factual_indicators = [
            r'\b(is|are|was|were)\s+(a|an|the)\b',
            r'\b(defined as|refers to|means|known as|consists of)\b',
            r'\b(according to|research shows|studies show|data shows)\b',
            r'\b(first|second|third|fourth|fifth)\b.*\b(step|stage|phase|point)\b',
            r'\d+\s*(percent|%|million|billion|km|miles|years|centuries)',
        ]
        matches = sum(1 for p in factual_indicators if re.search(p, text, re.IGNORECASE))
        return matches >= 2

    @staticmethod
    def _extract_key_sentences(text: str, max_sentences: int = 3) -> List[str]:
        """Extract the most informative sentences from text"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if not sentences:
            return []

        # Score sentences by informativeness
        scored = []
        for s in sentences:
            if len(s) < 20 or len(s) > 500:
                continue
            score = 0
            # Prefer sentences with numbers, proper nouns, or factual language
            if re.search(r'\d', s):
                score += 2
            if re.search(r'[A-Z][a-z]+', s):
                score += 1
            if re.search(r'\b(is|are|was|were|means|defined|known|called|discovered|invented|founded)\b', s, re.IGNORECASE):
                score += 2
            if len(s) > 50:
                score += 1
            scored.append((score, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:max_sentences]]

    # ── Fact Conflict Resolution ──────────────────────────────────────────

    def _resolve_fact_conflict(self, new_fact: Dict, chat_id: Optional[str] = None):
        """
        Check for and resolve conflicting facts.
        
        E.g., if user said "my name is Alice" before, and now says "my name is Bob",
        we should update rather than accumulate conflicting entries.
        
        Approach:
        - For identity facts (name, location), search for existing facts of same type
        - If found with different value, log the update and let recency scoring handle priority
        - For preferences, accumulate (people can like multiple things)
        """
        if not self.rag or not self.rag.is_ready():
            return

        fact_type = new_fact.get("type", "")

        # Only resolve conflicts for identity-type facts
        if fact_type not in ("name", "location"):
            return

        try:
            # Search for existing facts of the same type
            search_query = f"user {fact_type}"
            existing = self.rag.get_context(
                search_query,
                n_results=5,
                global_search=True
            )

            for text, meta in existing:
                if meta.get("fact_type") == fact_type and meta.get("type") == "fact":
                    existing_value = text
                    new_value = new_fact["text"]

                    if existing_value != new_value:
                        logger.info(
                            f"Fact update detected [{fact_type}]: "
                            f"'{existing_value[:50]}' → '{new_value[:50]}'"
                        )
                        # The new fact will be stored with a newer timestamp,
                        # and recency scoring will prefer it over the old one.
                        # We don't delete the old fact — it serves as history.

        except Exception as e:
            logger.debug(f"Fact conflict check failed: {e}")

    # ── Context Formatting for LLM ────────────────────────────────────────

    def format_context_for_prompt(
        self,
        contexts: List[RetrievedContext],
        max_chars: int = 2000
    ) -> str:
        """
        Format retrieved contexts into a prompt-ready string.
        
        Groups by source type for clarity (like ChatGPT's citation approach):
        - Personal facts first (most relevant)
        - Knowledge base info
        - Web search results (with source URLs)
        """
        if not contexts:
            return ""

        sections = {
            "facts": [],      # Personal facts from memory
            "knowledge": [],  # Wikipedia / knowledge base
            "web": [],        # Web search results
            "memory": [],     # Conversation memory
        }

        for ctx in contexts:
            if ctx.source == "memory" and ctx.metadata.get("type") == "fact":
                sections["facts"].append(ctx)
            elif ctx.source in ("knowledge", "wikipedia"):
                sections["knowledge"].append(ctx)
            elif ctx.source in ("web_search", "search_cache"):
                sections["web"].append(ctx)
            elif ctx.source == "memory":
                sections["memory"].append(ctx)

        parts = []
        total_chars = 0

        # Facts first
        if sections["facts"]:
            facts_text = "KNOWN FACTS ABOUT THE USER:\n"
            for ctx in sections["facts"][:3]:
                line = f"• {ctx.text[:300]}\n"
                if total_chars + len(line) < max_chars:
                    facts_text += line
                    total_chars += len(line)
            parts.append(facts_text)

        # Knowledge base
        if sections["knowledge"]:
            kb_text = "RELEVANT KNOWLEDGE:\n"
            for ctx in sections["knowledge"][:3]:
                title_prefix = f"[{ctx.title}] " if ctx.title else ""
                line = f"• {title_prefix}{ctx.text[:400]}\n"
                if total_chars + len(line) < max_chars:
                    kb_text += line
                    total_chars += len(line)
            parts.append(kb_text)

        # Web search
        if sections["web"]:
            web_text = "WEB SEARCH RESULTS (may contain inaccurate info — verify):\n"
            for ctx in sections["web"][:3]:
                url_suffix = f" ({ctx.url})" if ctx.url else ""
                line = f"• {ctx.text[:350]}{url_suffix}\n"
                if total_chars + len(line) < max_chars:
                    web_text += line
                    total_chars += len(line)
            parts.append(web_text)

        # Conversation memory (lower priority)
        if sections["memory"] and total_chars < max_chars - 200:
            mem_text = "FROM PREVIOUS CONVERSATIONS:\n"
            for ctx in sections["memory"][:2]:
                line = f"• {ctx.text[:250]}\n"
                if total_chars + len(line) < max_chars:
                    mem_text += line
                    total_chars += len(line)
            parts.append(mem_text)

        return "\n".join(parts)

    # ── Stats ─────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Get knowledge manager statistics"""
        stats = dict(self.stats)

        if self.rag and self.rag.is_ready():
            stats["rag"] = self.rag.get_stats()

        if self.kb and self.kb.is_ready():
            stats["knowledge_base"] = self.kb.get_stats()

        return stats
