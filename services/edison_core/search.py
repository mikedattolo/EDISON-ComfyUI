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
