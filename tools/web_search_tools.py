"""
Web search tools using DuckDuckGo.
No API key required - uses the duckduckgo-search library.
"""

import requests
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import time
import re
from urllib.parse import quote_plus
from pathlib import Path

# Try to import duckduckgo_search, fall back to requests-based approach
try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False


class WebSearchTools:
    """
    Web search capabilities using DuckDuckGo.
    Provides search and basic webpage fetching.
    """
    
    def __init__(self, cache_dir: Path = None, rate_limit_seconds: float = 1.0):
        """
        Initialize web search tools.
        
        Args:
            cache_dir: Directory to cache search results
            rate_limit_seconds: Minimum time between requests
        """
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.rate_limit = rate_limit_seconds
        self.last_request_time = 0
        self.search_history = []
        
        # User agent for requests
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    
    def _rate_limit_wait(self):
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def search(
        self,
        query: str,
        max_results: int = 10,
        region: str = "us-en",
        time_range: str = None
    ) -> Dict[str, Any]:
        """
        Search the web using DuckDuckGo.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            region: Region for results (e.g., "us-en", "uk-en")
            time_range: Time filter ("d" = day, "w" = week, "m" = month, "y" = year)
            
        Returns:
            Dict with search results
        """
        self._rate_limit_wait()
        
        try:
            if HAS_DDGS:
                results = self._search_with_ddgs(query, max_results, region, time_range)
            else:
                results = self._search_with_requests(query, max_results)
            
            # Log search
            self._log_search(query, len(results))
            
            # Cache results if cache_dir is set
            if self.cache_dir:
                self._cache_results(query, results)
            
            return {
                "success": True,
                "query": query,
                "result_count": len(results),
                "results": results
            }
            
        except Exception as e:
            return {
                "success": False,
                "query": query,
                "error": str(e)
            }
    
    def _search_with_ddgs(
        self,
        query: str,
        max_results: int,
        region: str,
        time_range: str
    ) -> List[Dict]:
        """Search using duckduckgo-search library"""
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query,
                region=region,
                timelimit=time_range,
                max_results=max_results
            ))
        
        # Normalize result format
        normalized = []
        for r in results:
            normalized.append({
                "title": r.get("title", ""),
                "url": r.get("href", r.get("link", "")),
                "snippet": r.get("body", r.get("snippet", "")),
                "source": "duckduckgo"
            })
        
        return normalized
    
    def _search_with_requests(self, query: str, max_results: int) -> List[Dict]:
        """
        Fallback search using requests (HTML parsing).
        Less reliable but works without additional dependencies.
        """
        encoded_query = quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        
        response = requests.get(url, headers=self.headers, timeout=10)
        response.raise_for_status()
        
        # Simple HTML parsing (basic approach)
        results = []
        
        # Find result blocks
        result_pattern = r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>([^<]+)</a>'
        snippet_pattern = r'<a class="result__snippet"[^>]*>([^<]+)</a>'
        
        matches = re.findall(result_pattern, response.text)
        snippets = re.findall(snippet_pattern, response.text)
        
        for i, (url, title) in enumerate(matches[:max_results]):
            snippet = snippets[i] if i < len(snippets) else ""
            results.append({
                "title": title.strip(),
                "url": url,
                "snippet": snippet.strip(),
                "source": "duckduckgo_html"
            })
        
        return results
    
    def search_news(
        self,
        query: str,
        max_results: int = 10,
        region: str = "us-en"
    ) -> Dict[str, Any]:
        """
        Search for news articles.
        
        Args:
            query: Search query
            max_results: Maximum results
            region: Region for results
            
        Returns:
            Dict with news results
        """
        self._rate_limit_wait()
        
        try:
            if HAS_DDGS:
                with DDGS() as ddgs:
                    results = list(ddgs.news(
                        query,
                        region=region,
                        max_results=max_results
                    ))
                
                normalized = []
                for r in results:
                    normalized.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", r.get("link", "")),
                        "snippet": r.get("body", ""),
                        "date": r.get("date", ""),
                        "source": r.get("source", ""),
                        "type": "news"
                    })
                
                return {
                    "success": True,
                    "query": query,
                    "result_count": len(normalized),
                    "results": normalized
                }
            else:
                # Fall back to regular search with news keywords
                return self.search(f"{query} news", max_results)
                
        except Exception as e:
            return {
                "success": False,
                "query": query,
                "error": str(e)
            }
    
    def fetch_webpage(
        self,
        url: str,
        timeout: int = 10,
        extract_text: bool = True
    ) -> Dict[str, Any]:
        """
        Fetch and parse a webpage.
        
        Args:
            url: URL to fetch
            timeout: Request timeout
            extract_text: Whether to extract clean text
            
        Returns:
            Dict with page content
        """
        self._rate_limit_wait()
        
        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=timeout,
                allow_redirects=True
            )
            response.raise_for_status()
            
            result = {
                "success": True,
                "url": url,
                "final_url": response.url,
                "status_code": response.status_code,
                "content_type": response.headers.get("Content-Type", ""),
            }
            
            if extract_text:
                # Try to extract clean text using BeautifulSoup if available
                try:
                    from bs4 import BeautifulSoup
                    
                    soup = BeautifulSoup(response.text, "html.parser")
                    
                    # Remove script and style elements
                    for element in soup(["script", "style", "nav", "footer", "header"]):
                        element.decompose()
                    
                    # Get text
                    text = soup.get_text(separator="\n")
                    
                    # Clean up whitespace
                    lines = (line.strip() for line in text.splitlines())
                    text = "\n".join(line for line in lines if line)
                    
                    result["text"] = text[:10000]  # Limit size
                    result["title"] = soup.title.string if soup.title else ""
                    result["text_length"] = len(text)
                    
                except ImportError:
                    # Basic text extraction without BeautifulSoup
                    text = re.sub(r'<[^>]+>', ' ', response.text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    result["text"] = text[:10000]
                    result["text_length"] = len(text)
            else:
                result["html"] = response.text[:50000]  # Limit raw HTML size
            
            return result
            
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "url": url,
                "error": f"Request timed out after {timeout} seconds"
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "url": url,
                "error": str(e)
            }
    
    def search_and_fetch(
        self,
        query: str,
        max_results: int = 5,
        fetch_content: bool = True
    ) -> Dict[str, Any]:
        """
        Search and optionally fetch content from top results.
        
        Args:
            query: Search query
            max_results: Number of results to fetch
            fetch_content: Whether to fetch full page content
            
        Returns:
            Dict with search results and fetched content
        """
        # First search
        search_result = self.search(query, max_results=max_results)
        
        if not search_result["success"]:
            return search_result
        
        if not fetch_content:
            return search_result
        
        # Fetch content from each result
        enriched_results = []
        for result in search_result["results"]:
            url = result.get("url")
            if url:
                fetch_result = self.fetch_webpage(url)
                result["fetched_content"] = {
                    "success": fetch_result["success"],
                    "text": fetch_result.get("text", "")[:2000],  # Truncate
                    "error": fetch_result.get("error")
                }
            enriched_results.append(result)
        
        return {
            "success": True,
            "query": query,
            "result_count": len(enriched_results),
            "results": enriched_results
        }
    
    def _log_search(self, query: str, result_count: int):
        """Log search for tracking"""
        self.search_history.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "result_count": result_count
        })
    
    def _cache_results(self, query: str, results: List[Dict]):
        """Cache search results to file"""
        if not self.cache_dir:
            return
        
        # Create safe filename from query
        safe_query = re.sub(r'[^\w\s-]', '', query)[:50]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_file = self.cache_dir / f"search_{safe_query}_{timestamp}.json"
        
        with open(cache_file, 'w') as f:
            json.dump({
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "results": results
            }, f, indent=2)
    
    def get_search_history(self) -> List[Dict]:
        """Get search history for this session"""
        return self.search_history
    
    def get_cached_searches(self) -> List[str]:
        """Get list of cached search files"""
        if not self.cache_dir or not self.cache_dir.exists():
            return []
        
        return [f.name for f in self.cache_dir.glob("search_*.json")]


# Convenience function for simple searches
def quick_search(query: str, max_results: int = 5) -> List[Dict]:
    """
    Quick search without creating WebSearchTools instance.
    
    Args:
        query: Search query
        max_results: Maximum results
        
    Returns:
        List of search results
    """
    searcher = WebSearchTools()
    result = searcher.search(query, max_results=max_results)
    
    if result["success"]:
        return result["results"]
    else:
        return []
