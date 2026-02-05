"""
Web Search Tool for EDISON
Provides web search capabilities using DuckDuckGo API
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class WebSearchTool:
    """Web search using DuckDuckGo API"""
    
    def __init__(self):
        """Initialize the web search tool."""
        try:
            # Try new package first
            from ddgs import DDGS
            logger.info("Using ddgs package")
        except ImportError:
            # Fall back to old package
            from duckduckgo_search import DDGS
            logger.info("Using duckduckgo_search package (deprecated)")
        
        self.ddgs = DDGS()
        logger.info("Web search tool initialized with DuckDuckGo API")
    
    def search(self, query: str, num_results: int = 5, time_range: str = None) -> List[Dict[str, str]]:
        """
        Search DuckDuckGo and return results using official API
        
        Args:
            query: Search query
            num_results: Number of results to return
            time_range: Time range filter - 'd' (day), 'w' (week), 'm' (month), 'y' (year)
            
        Returns:
            List of dicts with 'title', 'url', 'snippet' keys
        """
        try:
            # Detect if user wants recent/current/today news
            query_lower = query.lower()
            
            # Default to recent results for any news query
            if 'news' in query_lower and not time_range:
                time_range = 'd'  # Default news queries to last day
                logger.info(f"News query detected, defaulting to last day")
            
            # Override with more specific time requests
            if any(word in query_lower for word in ['today', 'latest', 'current']):
                time_range = 'd'  # Last day
                logger.info(f"Detected today/latest/current keywords, using time_range=d")
            elif 'this week' in query_lower or 'recent' in query_lower:
                time_range = 'w'  # Last week
                logger.info(f"Detected recent/this week keywords, using time_range=w")
            elif 'this month' in query_lower:
                time_range = 'm'  # Last month
                logger.info(f"Detected this month keyword, using time_range=m")
            
            # Use DuckDuckGo API with time range if specified
            logger.debug(f"Searching for: {query} (time_range={time_range})")
            
            # Call text() with timelimit parameter if time_range specified
            if time_range:
                raw_results = self.ddgs.text(query, max_results=num_results, timelimit=time_range)
            else:
                raw_results = self.ddgs.text(query, max_results=num_results)
            
            results = []
            result_count = 0
            
            # Convert generator to list to catch any issues
            try:
                for result in raw_results:
                    result_count += 1
                    logger.debug(f"Raw result #{result_count}: {result}")
                    
                    # DuckDuckGo API returns: title, href, body
                    title = result.get('title', '')
                    url = result.get('href', '')
                    snippet = result.get('body', '')
                    
                    if title or url:  # Only add if we have at least title or URL
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet
                        })
                        logger.debug(f"Added result: {title[:50]}...")
                    else:
                        logger.warning(f"Skipping result with no title/url: {result}")
                        
            except Exception as iter_error:
                logger.error(f"Error iterating search results: {iter_error}")
            
            logger.info(f"Found {len(results)} search results for: {query} (processed {result_count} raw results)")
            
            if result_count == 0:
                logger.warning(f"DuckDuckGo API returned no results for query: {query}")
            
            return results
            
        except Exception as e:
            logger.error(f"Web search error: {e}", exc_info=True)
            return []

    def deep_search(self, query: str, num_results: int = 5, variations: int = 3) -> tuple[List[Dict[str, str]], Dict[str, int]]:
        """
        Multi-query search expansion for deeper coverage.

        Returns tuple: (results, metadata)
        """
        queries = self._expand_queries(query, variations)
        all_results = []
        for q in queries:
            all_results.extend(self.search(q, num_results=num_results))

        deduped = []
        seen = set()
        for item in all_results:
            key = (item.get("url") or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append(item)

        meta = {
            "queries": len(queries),
            "results": len(deduped)
        }
        return deduped[: max(num_results, 5)], meta

    def _expand_queries(self, query: str, variations: int) -> List[str]:
        base = [query]
        templates = [
            f"{query} overview",
            f"{query} best practices",
            f"{query} recent findings",
            f"{query} pros and cons",
            f"{query} guidelines"
        ]
        for t in templates:
            if t not in base:
                base.append(t)
            if len(base) >= variations + 1:
                break
        return base
