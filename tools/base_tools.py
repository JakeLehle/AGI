"""
Base tools available to all agents.
These are the foundational capabilities defined at design time.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import requests
from bs4 import BeautifulSoup
import pandas as pd

class BaseTools:
    """Core tools that all agents can use"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        (self.data_dir / "inputs").mkdir(exist_ok=True)
        (self.data_dir / "outputs").mkdir(exist_ok=True)
    
    def read_file(self, filepath: str) -> str:
        """Read text file contents"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Failed to read file {filepath}: {str(e)}")
    
    def write_file(self, filepath: str, content: str) -> Dict[str, Any]:
        """Write content to file"""
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return {
                "success": True,
                "filepath": str(path),
                "size": len(content)
            }
        except Exception as e:
            raise Exception(f"Failed to write file {filepath}: {str(e)}")
    
    def list_files(self, directory: str, pattern: str = "*") -> List[str]:
        """List files in directory matching pattern"""
        try:
            dir_path = Path(directory)
            return [str(p) for p in dir_path.glob(pattern)]
        except Exception as e:
            raise Exception(f"Failed to list files in {directory}: {str(e)}")
    
    def web_search(self, query: str) -> List[Dict[str, str]]:
        """
        Simplified web search (you'd integrate with real search API)
        Returns list of {title, url, snippet}
        """
        # Placeholder - integrate with actual search API
        # For now, return mock data
        return [
            {
                "title": f"Search result for: {query}",
                "url": "https://example.com",
                "snippet": f"Mock search result for query: {query}"
            }
        ]
    
    def fetch_webpage(self, url: str) -> Dict[str, Any]:
        """Fetch and parse webpage content"""
        try:
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            text = '\n'.join(line for line in lines if line)
            
            return {
                "success": True,
                "url": url,
                "text": text[:5000],  # Limit size
                "title": soup.title.string if soup.title else ""
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def analyze_csv(self, filepath: str) -> Dict[str, Any]:
        """Analyze CSV file and return summary statistics"""
        try:
            df = pd.read_csv(filepath)
            
            return {
                "success": True,
                "rows": len(df),
                "columns": list(df.columns),
                "summary": df.describe().to_dict(),
                "sample": df.head(5).to_dict()
            }
        except Exception as e:
            raise Exception(f"Failed to analyze CSV {filepath}: {str(e)}")
    
    def save_json(self, data: Any, filepath: str) -> Dict[str, Any]:
        """Save data as JSON"""
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            return {
                "success": True,
                "filepath": str(path)
            }
        except Exception as e:
            raise Exception(f"Failed to save JSON to {filepath}: {str(e)}")
    
    def load_json(self, filepath: str) -> Any:
        """Load JSON file"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load JSON from {filepath}: {str(e)}")

# Global instance
base_tools = BaseTools()
