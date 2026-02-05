"""
Mem0 configuration loader for AGI pipeline.

Handles:
- Loading YAML configuration
- Environment variable substitution (${VAR_NAME} syntax)
- Default value fallbacks
- Path creation for data directories

Environment Variables:
    AGI_DATA_DIR: Base directory for data (default: ~/agi_data)
    OLLAMA_BASE_URL: Ollama server URL (default: http://localhost:11434)
    MEM0_LLM_MODEL: Override LLM model
    MEM0_EMBED_MODEL: Override embedding model
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import yaml, provide fallback if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not installed. Using fallback configuration.")


def get_default_config() -> Dict[str, Any]:
    """
    Return default Mem0 configuration when YAML file is not available.
    
    Provides a working embedded Qdrant + Ollama configuration.
    """
    agi_data_dir = os.environ.get("AGI_DATA_DIR", str(Path.home() / "agi_data"))
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    
    return {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": str(Path(agi_data_dir) / "qdrant_storage"),
                "collection_name": "agi_reflexion_memory",
                "embedding_model_dims": 768,
                "on_disk": True,
            }
        },
        "llm": {
            "provider": "ollama",
            "config": {
                "model": "llama3.1:70b",
                "temperature": 0,
                "max_tokens": 60000,
                "ollama_base_url": ollama_url,
            }
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": "nomic-embed-text",
                "ollama_base_url": ollama_url,
            }
        },
    }


def substitute_env_vars(config_str: str) -> str:
    """
    Substitute environment variables in config string.
    
    Supports:
        ${VAR_NAME} - simple substitution
        ${VAR_NAME:-default} - with default value
    """
    pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'
    
    def replacer(match):
        var_name = match.group(1)
        default = match.group(2) if match.group(2) else ""
        return os.environ.get(var_name, default)
    
    return re.sub(pattern, replacer, config_str)


def get_mem0_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load Mem0 configuration with environment variable substitution.
    
    Args:
        config_path: Path to config YAML file. Searches standard locations if None.
        
    Returns:
        Dictionary containing:
        - config: Mem0 configuration dict ready for Memory.from_config()
        - version: Config version string
        - custom_fact_extraction_prompt: Custom prompt if defined
        - custom_update_memory_prompt: Custom update prompt if defined
        - history_db_path: Path to history database
    """
    # Set up environment defaults
    agi_data_dir = os.environ.get("AGI_DATA_DIR", str(Path.home() / "agi_data"))
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Ensure environment variables are set for substitution
    os.environ.setdefault("AGI_DATA_DIR", agi_data_dir)
    os.environ.setdefault("OLLAMA_BASE_URL", ollama_url)
    
    # Create data directory
    Path(agi_data_dir).mkdir(parents=True, exist_ok=True)
    
    # Search for config file
    if config_path is None:
        search_paths = [
            Path(__file__).parent.parent / "config" / "mem0_config.yaml",
            Path.cwd() / "config" / "mem0_config.yaml",
            Path.cwd() / "mem0_config.yaml",
            Path(agi_data_dir) / "config" / "mem0_config.yaml",
        ]
        
        for path in search_paths:
            if path.exists():
                config_path = str(path)
                logger.info(f"Found config at: {config_path}")
                break
    
    # Load configuration
    if config_path and Path(config_path).exists() and YAML_AVAILABLE:
        logger.info(f"Loading Mem0 config from: {config_path}")
        
        with open(config_path) as f:
            config_str = f.read()
        
        # Substitute environment variables
        config_str = substitute_env_vars(config_str)
        config = yaml.safe_load(config_str)
    else:
        if config_path and not Path(config_path).exists():
            logger.warning(f"Config not found at {config_path}, using defaults")
        logger.info("Using default Mem0 configuration")
        config = get_default_config()
    
    # Extract metadata (not part of Memory.from_config)
    version = config.pop("version", "v1.1")
    custom_fact_prompt = config.pop("custom_fact_extraction_prompt", None)
    custom_update_prompt = config.pop("custom_update_memory_prompt", None)
    history_db = config.pop("history_db_path", None)
    
    # Apply environment variable overrides
    if llm_model := os.environ.get("MEM0_LLM_MODEL"):
        config["llm"]["config"]["model"] = llm_model
        logger.info(f"Overriding LLM model to: {llm_model}")
        
    if embed_model := os.environ.get("MEM0_EMBED_MODEL"):
        config["embedder"]["config"]["model"] = embed_model
        logger.info(f"Overriding embedding model to: {embed_model}")
    
    # Ensure storage directories exist
    if "vector_store" in config:
        vs_config = config["vector_store"].get("config", {})
        if "path" in vs_config:
            vs_path = Path(vs_config["path"])
            vs_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Qdrant storage: {vs_path}")
    
    return {
        "config": config,
        "version": version,
        "custom_fact_extraction_prompt": custom_fact_prompt,
        "custom_update_memory_prompt": custom_update_prompt,
        "history_db_path": history_db,
    }


def validate_ollama_connection(base_url: str = None) -> Dict[str, Any]:
    """
    Validate Ollama is running and required models are available.
    
    Args:
        base_url: Ollama server URL (uses env var if not provided)
        
    Returns:
        Dict with status and available models
    """
    import urllib.request
    import json
    
    base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    
    result = {
        "connected": False,
        "base_url": base_url,
        "models": [],
        "has_embed_model": False,
        "has_llm_model": False,
        "error": None,
    }
    
    try:
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            result["connected"] = True
            result["models"] = [m["name"] for m in data.get("models", [])]
            
            # Check for required models
            model_names = [m.split(":")[0] for m in result["models"]]
            result["has_embed_model"] = "nomic-embed-text" in model_names
            result["has_llm_model"] = any(
                m in model_names 
                for m in ["llama3.1", "llama3", "mistral", "qwen2"]
            )
            
    except Exception as e:
        result["error"] = str(e)
    
    return result


if __name__ == "__main__":
    # Test configuration loading
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    print("=" * 60)
    print("Mem0 Configuration Test")
    print("=" * 60)
    
    # Load config
    config_data = get_mem0_config()
    
    print(f"\nConfiguration loaded:")
    print(f"  Version: {config_data['version']}")
    print(f"  Vector Store: {config_data['config']['vector_store']['provider']}")
    print(f"  Storage Path: {config_data['config']['vector_store']['config'].get('path', 'N/A')}")
    print(f"  LLM: {config_data['config']['llm']['config']['model']}")
    print(f"  Embedder: {config_data['config']['embedder']['config']['model']}")
    
    # Validate Ollama
    print(f"\nValidating Ollama connection...")
    ollama_status = validate_ollama_connection()
    
    if ollama_status["connected"]:
        print(f"  ✅ Connected to {ollama_status['base_url']}")
        print(f"  Models available: {len(ollama_status['models'])}")
        print(f"  Has embedding model: {'✅' if ollama_status['has_embed_model'] else '❌'}")
        print(f"  Has LLM model: {'✅' if ollama_status['has_llm_model'] else '❌'}")
        
        if not ollama_status["has_embed_model"]:
            print("\n  ⚠️  Run: ollama pull nomic-embed-text")
    else:
        print(f"  ❌ Failed to connect: {ollama_status['error']}")
        print("  Make sure Ollama is running: ollama serve")
