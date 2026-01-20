"""
Web Search Tool for EDISON
Provides web search capabilities using DuckDuckGo
"""

import logging
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
import urllib.parse

logger = logging.getLogger(__name__)

class WebSearchTool:
    """Simple web search using DuckDuckGo HTML"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        Search DuckDuckGo and return results
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of dicts with 'title', 'url', 'snippet' keys
        """
        try:
            # Use DuckDuckGo HTML
            encoded_query = urllib.parse.quote(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Parse results - try multiple selectors in case HTML changed
            result_divs = soup.find_all('div', class_='result')
            if not result_divs:
                # Try alternative selector
                result_divs = soup.find_all('div', class_='results_links')
            
            logger.debug(f"Found {len(result_divs)} result divs")
            
            # Parse results
            for result_div in result_divs[:num_results]:
                try:
                    # Get title and URL - try multiple selectors
                    title_tag = result_div.find('a', class_='result__a')
                    if not title_tag:
                        title_tag = result_div.find('a', class_='large')
                    if not title_tag:
                        # Try any link in the result
                        title_tag = result_div.find('a', href=True)
                    
                    if not title_tag:
                        continue
                    
                    title = title_tag.get_text(strip=True)
                    url = title_tag.get('href', '')
                    
                    # Get snippet - try multiple selectors
                    snippet_tag = result_div.find('a', class_='result__snippet')
                    if not snippet_tag:
                        snippet_tag = result_div.find('div', class_='snippet')
                    if not snippet_tag:
                        snippet_tag = result_div.find('span', class_='result-snippet')
                    
                    snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                    
                    if title and url:
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet
                        })
                        logger.debug(f\"Found result: {title[:50]}...\")
                        
                except Exception as e:
                    logger.debug(f"Error parsing result: {e}")
                    continue
            
            if not results:
                logger.warning(f\"DuckDuckGo HTML returned no parseable results. HTML length: {len(response.text)}\")\n                logger.debug(f\"First 500 chars: {response.text[:500]}\")\n            \n            logger.info(f"Found {len(results)} search results for: {query}")
            return results
            
        except requests.RequestException as e:
            logger.error(f"Web search request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []
    
    def fetch_page_content(self, url: str, max_length: int = 2000) -> str:
        """
        Fetch and extract text content from a URL
        
        Args:
            url: URL to fetch
            max_length: Maximum content length to return
            
        Returns:
            Extracted text content
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style', 'nav', 'header', 'footer']):
                script.decompose()
            
            # Get text
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            text = '\n'.join(line for line in lines if line)
            
            # Truncate
            if len(text) > max_length:
                text = text[:max_length] + "..."
            
            logger.info(f"Fetched {len(text)} characters from {url}")
            return text
            
        except Exception as e:
            logger.error(f"Error fetching page content from {url}: {e}")
            return ""
