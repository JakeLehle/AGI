"""
Resilient LLM Invocation Utility

Replaces both:
  - master_agent.py's invoke_with_timeout() (SIGALRM-based, broken in threads)
  - sub_agent.py's bare self.llm.invoke() calls (no error handling at all)

Design philosophy (per Jake's requirements):
  - NO artificial timeouts that kill legitimate long-running generations
  - The only hard limit is the 3-day SLURM node window
  - Resilience comes from RETRY WITH BACKOFF on transient failures (HTTP 500s),
    not from killing slow completions
  - Ollama 500 errors are transient and recoverable — the model just needs
    a moment to recover (clear KV cache, free VRAM, etc.)

Usage:
    from utils.llm_invoke import invoke_resilient

    # In master_agent.py — replace invoke_with_timeout(self.llm, prompt)
    response = invoke_resilient(self.llm, prompt)

    # In sub_agent.py — replace self.llm.invoke(prompt)
    response = invoke_resilient(self.llm, prompt)
"""

import time
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)


class LLMInvocationError(Exception):
    """Raised when all retry attempts are exhausted."""
    pass


def invoke_resilient(
    llm,
    prompt: str,
    max_retries: int = 20,
    initial_backoff: float = 30.0,
    max_backoff: float = 300.0,
    backoff_multiplier: float = 1.5,
    ollama_base_url: str = "http://127.0.0.1:11434",
) -> str:
    """
    Invoke an LLM with exponential backoff retry on transient failures.

    This function is designed to survive the exact failure mode observed in
    production: Ollama returning HTTP 500 errors due to memory pressure from
    a partially-offloaded model (e.g. qwen3-coder-next with 28/49 layers on
    GPU). Rather than timing out and killing the request, it:

      1. Catches the exception from langchain-ollama
      2. Optionally checks if Ollama is still alive (health check)
      3. Waits with exponential backoff to let Ollama recover
      4. Retries up to max_retries times
      5. Only gives up after all retries are exhausted

    With default settings (20 retries, 30s initial, 1.5x multiplier, 300s max):
      Total worst-case wait: ~75 minutes of backoff time
      Then it raises LLMInvocationError so the caller can handle gracefully

    Args:
        llm: LangChain LLM instance (OllamaLLM)
        prompt: The prompt string
        max_retries: Maximum number of retry attempts (default 20)
        initial_backoff: Seconds to wait after first failure (default 30)
        max_backoff: Maximum backoff duration in seconds (default 300 = 5 min)
        backoff_multiplier: Multiply backoff by this each retry (default 1.5)
        ollama_base_url: Ollama API URL for health checks

    Returns:
        The LLM response string

    Raises:
        LLMInvocationError: If all retries exhausted (with details of failures)
    """
    backoff = initial_backoff
    last_error = None
    consecutive_failures = 0

    for attempt in range(1, max_retries + 1):
        try:
            response = llm.invoke(prompt)

            # Reset on success
            if consecutive_failures > 0:
                logger.info(
                    f"LLM call succeeded after {consecutive_failures} "
                    f"consecutive failures (attempt {attempt}/{max_retries})"
                )

            # Validate we got actual content back
            if response and isinstance(response, str) and len(response.strip()) > 0:
                return response

            # Empty response — treat as transient failure
            logger.warning(
                f"LLM returned empty response (attempt {attempt}/{max_retries})"
            )
            last_error = "Empty response from LLM"
            consecutive_failures += 1

        except Exception as e:
            last_error = str(e)
            consecutive_failures += 1
            error_type = type(e).__name__

            # Log the failure
            logger.warning(
                f"LLM call failed (attempt {attempt}/{max_retries}): "
                f"{error_type}: {last_error[:200]}"
            )

            # Check if Ollama is still alive before retrying
            if not _ollama_health_check(ollama_base_url):
                logger.error(
                    "Ollama server is not responding. Waiting for recovery..."
                )
                # Longer wait if server is down entirely
                time.sleep(min(backoff * 2, max_backoff))
            else:
                logger.info("Ollama server is alive, model may need recovery time")

        # Don't sleep after the last attempt
        if attempt < max_retries:
            sleep_time = min(backoff, max_backoff)
            logger.info(
                f"Backing off {sleep_time:.0f}s before retry "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(sleep_time)
            backoff = min(backoff * backoff_multiplier, max_backoff)

    # All retries exhausted
    raise LLMInvocationError(
        f"LLM invocation failed after {max_retries} attempts. "
        f"Last error: {last_error}"
    )


def _ollama_health_check(base_url: str, timeout: float = 10.0) -> bool:
    """Quick check if Ollama API is responding."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False
