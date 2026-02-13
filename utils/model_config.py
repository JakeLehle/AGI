"""
Centralized Ollama Model Configuration — Single Source of Truth

Every component that needs to know which model to use should call
resolve_model() instead of hardcoding a default. The resolution
priority is:

    1. Explicit parameter (passed directly to the function)
    2. Environment variable OLLAMA_MODEL (set by RUN scripts / sbatch)
    3. config.yaml ollama.model (loaded at startup)
    4. FALLBACK_MODEL constant (the only hardcoded model name in the codebase)

This means the model can be changed in exactly ONE place for each context:
    - Per-run:    sbatch --export=OLLAMA_MODEL=qwen3-coder:14b ...
    - Per-project: edit config/config.yaml → ollama.model
    - System-wide: change FALLBACK_MODEL below (but prefer config.yaml)

Similarly, resolve_base_url() centralizes the Ollama server URL.

Usage in agents:
    from utils.model_config import resolve_model, resolve_base_url

    class MasterAgent:
        def __init__(self, ollama_model=None, ollama_base_url=None, ...):
            model = resolve_model(ollama_model)
            base_url = resolve_base_url(ollama_base_url)
            self.llm = OllamaLLM(model=model, base_url=base_url)

Usage in main.py:
    from utils.model_config import resolve_model, resolve_base_url

    # After loading config but before passing to workflow:
    model = resolve_model(args.model, config)
    base_url = resolve_base_url(args.ollama_url, config)
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# =============================================================================
# THE ONLY HARDCODED MODEL NAME IN THE ENTIRE CODEBASE
# =============================================================================
# Change this when upgrading to a new default model. Everything else reads
# from config.yaml or the OLLAMA_MODEL environment variable.
#
# Selection criteria for V100S-32GB:
#   - Model weights (Q4_K_M) must fit mostly on GPU (<30 GiB)
#   - Must leave room for KV cache at full context length
#   - qwen3-coder:32b ≈ 20 GiB weights → fits with ~10 GiB for 32K KV cache
#   - qwen3-coder-next ≈ 48 GiB → requires CPU offload, causes 500 errors
# =============================================================================
FALLBACK_MODEL = "qwen3-coder:32b"
FALLBACK_BASE_URL = "http://127.0.0.1:11434"
FALLBACK_CONTEXT_LENGTH = 32768


def resolve_model(
    explicit: Optional[str] = None,
    config: Optional[dict] = None,
) -> str:
    """
    Resolve which Ollama model to use.

    Priority:
        1. explicit parameter (from CLI --model or direct kwarg)
        2. OLLAMA_MODEL environment variable (set by RUN scripts)
        3. config dict → ollama.model (from config.yaml)
        4. FALLBACK_MODEL constant

    Args:
        explicit: Model name passed directly (e.g. from CLI arg or constructor)
        config: Loaded config dict (e.g. from config.yaml)

    Returns:
        Resolved model name string
    """
    # 1. Explicit parameter
    if explicit:
        logger.debug(f"Model resolved from explicit parameter: {explicit}")
        return explicit

    # 2. Environment variable
    env_model = os.environ.get("OLLAMA_MODEL")
    if env_model:
        logger.debug(f"Model resolved from OLLAMA_MODEL env var: {env_model}")
        return env_model

    # 3. Config file
    if config:
        config_model = None
        if isinstance(config, dict):
            # Support both flat config and nested config
            config_model = config.get("ollama", {}).get("model") or config.get("model")
        if config_model:
            logger.debug(f"Model resolved from config: {config_model}")
            return config_model

    # 4. Fallback
    logger.debug(f"Model resolved from fallback: {FALLBACK_MODEL}")
    return FALLBACK_MODEL


def resolve_base_url(
    explicit: Optional[str] = None,
    config: Optional[dict] = None,
) -> str:
    """
    Resolve Ollama API base URL.

    Priority:
        1. explicit parameter
        2. OLLAMA_HOST environment variable (Ollama's native env var)
        3. config dict → ollama.base_url
        4. FALLBACK_BASE_URL constant

    Args:
        explicit: URL passed directly
        config: Loaded config dict

    Returns:
        Resolved base URL string
    """
    if explicit:
        return explicit

    env_url = os.environ.get("OLLAMA_HOST")
    if env_url:
        # OLLAMA_HOST may or may not include the scheme
        if not env_url.startswith("http"):
            env_url = f"http://{env_url}"
        return env_url

    if config:
        config_url = None
        if isinstance(config, dict):
            config_url = config.get("ollama", {}).get("base_url") or config.get("base_url")
        if config_url:
            return config_url

    return FALLBACK_BASE_URL


def resolve_context_length(
    explicit: Optional[int] = None,
    config: Optional[dict] = None,
) -> int:
    """
    Resolve model context length.

    Priority:
        1. explicit parameter
        2. OLLAMA_CONTEXT_LENGTH environment variable (set by RUN scripts)
        3. config dict → ollama.model_context_length
        4. FALLBACK_CONTEXT_LENGTH constant
    """
    if explicit:
        return explicit

    env_ctx = os.environ.get("OLLAMA_CONTEXT_LENGTH")
    if env_ctx:
        try:
            return int(env_ctx)
        except ValueError:
            pass

    if config:
        if isinstance(config, dict):
            ctx = config.get("ollama", {}).get("model_context_length")
            if ctx:
                return int(ctx)

    return FALLBACK_CONTEXT_LENGTH
