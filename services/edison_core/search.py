"""
Web Search Tool for EDISON
Provides web search capabilities using DuckDuckGo API
"""

import logging
from typing import List, Dict
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

class WebSearchTool:
    """Web search using DuckDuckGo API"""
    
    def __init__(self):
        """Initialize the web search tool."""
        self.ddgs = DDGS()
        logger.info("Web search tool initialized with DuckDuckGo API")
    
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        Search DuckDuckGo and return results using official API
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of dicts with 'title', 'url', 'snippet' keys
        """
        try:
            # Use DuckDuckGo API
            logger.debug(f"Searching for: {query}")
            raw_results = self.ddgs.text(query, max_results=num_results)
            
            results = []
            for result in raw_results:
                # DuckDuckGo API returns: title, href, body
                results.append({
                    'title': result.get('title', ''),
                    'url': result.get('href', ''),
                    'snippet': result.get('body', '')
                })
                logger.debug(f"Found result: {result.get('title', '')[:50]}...")
            
            logger.info(f"Found {len(results)} search results for: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []
