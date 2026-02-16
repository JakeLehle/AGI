"""
Context window management for token-based limits.

v1.2.0 Updates:
- Added create_independent_context() for diagnostic agent's separate 25K budget
- Added reset_context_window() for reusable windows across invocations
- Added destroy_context_window() for cleanup
- Enhanced stats with per-window independent context tracking

Provides:
- Token estimation (~4 chars per token for LLaMA-style models)
- History truncation with summarization
- Tool output pagination (25K token pages)
- Context window tracking per agent
- Independent context windows for multi-agent budgets (v1.2.0)
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import re


@dataclass
class Message:
    """Represents a message in the context window"""
    role: str  # 'system', 'user', 'assistant', 'tool'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    token_estimate: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.token_estimate == 0:
            self.token_estimate = ContextManager.estimate_tokens(self.content)


@dataclass 
class ContextWindow:
    """Tracks a single context window state"""
    agent_id: str
    max_tokens: int
    messages: List[Message] = field(default_factory=list)
    total_tokens: int = 0
    summaries: List[str] = field(default_factory=list)
    independent: bool = False  # v1.2.0: True if not tied to parent budget
    
    def add_message(self, message: Message) -> bool:
        """Add message if it fits, return False if context is full"""
        if self.total_tokens + message.token_estimate > self.max_tokens:
            return False
        self.messages.append(message)
        self.total_tokens += message.token_estimate
        return True
    
    def get_remaining_tokens(self) -> int:
        """Get remaining token budget"""
        return self.max_tokens - self.total_tokens
    
    def reset(self):
        """Reset window to empty state, preserving max_tokens config.
        
        v1.2.0: Used when diagnostic agent gets a fresh budget per invocation.
        """
        self.messages.clear()
        self.total_tokens = 0
        self.summaries.clear()
    
    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "max_tokens": self.max_tokens,
            "total_tokens": self.total_tokens,
            "message_count": len(self.messages),
            "remaining_tokens": self.get_remaining_tokens(),
            "independent": self.independent,
        }


class ContextManager:
    """
    Manages context windows with token-based limits.
    
    Key features:
    - 70K token limit per task/sub-agent context (configurable)
    - 25K token limit for tool outputs (with pagination)
    - History truncation with summarization
    - Separate context tracking per agent
    - Independent context windows for diagnostic agent (v1.2.0)
    
    Settings loaded from config/config.yaml under 'context' section.
    
    v1.2.0 Multi-Agent Budget Pattern:
        The sub-agent and diagnostic agent each get their own token budget.
        The sub-agent creates its window via create_context_window() as before.
        The diagnostic agent gets a separate window via create_independent_context()
        that is NOT tied to the sub-agent's budget. Each diagnostic invocation
        can reset its window to get a fresh budget.
        
        Example:
            cm = ContextManager()
            # Sub-agent: 25K budget for the subtask
            cm.create_context_window("sub_agent_step01", max_tokens=25000)
            
            # Diagnostic agent: separate 25K per invocation
            diag_ctx = cm.create_independent_context(
                "diag_step01_inv1", max_tokens=25000
            )
            # ... diagnostic work ...
            
            # Next invocation gets fresh budget
            diag_ctx2 = cm.create_independent_context(
                "diag_step01_inv2", max_tokens=25000
            )
    """
    
    # Default limits (overridden by config if available)
    DEFAULT_MAX_CONTEXT_TOKENS = 70_000
    DEFAULT_MAX_TOOL_OUTPUT_TOKENS = 25_000
    DEFAULT_SUMMARY_TARGET_TOKENS = 2_000  # Target size for summaries
    DEFAULT_CHARS_PER_TOKEN = 4
    DEFAULT_RECENT_HISTORY_PERCENT = 30
    
    # Class-level config cache
    _config_loaded = False
    _config_settings = None
    
    @classmethod
    def _load_config(cls):
        """Load settings from config file"""
        if cls._config_loaded:
            return
        
        try:
            from utils.config_loader import get_context_settings
            cls._config_settings = get_context_settings()
            cls._config_loaded = True
        except ImportError:
            # Config loader not available, use defaults
            cls._config_settings = None
            cls._config_loaded = True
    
    # Token estimation: ~4 characters per token for LLaMA-style models
    # This is conservative; actual ratio varies by content
    CHARS_PER_TOKEN = 4
    
    def __init__(
        self,
        max_context_tokens: int = None,
        max_tool_output_tokens: int = None,
        llm_for_summarization = None
    ):
        # Load config if not already loaded
        self._load_config()
        
        # Use config values if available, then fall back to defaults
        if self._config_settings:
            default_context = self._config_settings.max_tokens_per_task
            default_tool = self._config_settings.max_tool_output_tokens
            self.CHARS_PER_TOKEN = self._config_settings.chars_per_token
            self._recent_history_percent = self._config_settings.recent_history_percent / 100.0
        else:
            default_context = self.DEFAULT_MAX_CONTEXT_TOKENS
            default_tool = self.DEFAULT_MAX_TOOL_OUTPUT_TOKENS
            self._recent_history_percent = self.DEFAULT_RECENT_HISTORY_PERCENT / 100.0
        
        # Use provided values, or config values, or defaults
        self.max_context_tokens = max_context_tokens or default_context
        self.max_tool_output_tokens = max_tool_output_tokens or default_tool
        self.llm = llm_for_summarization
        
        # Track context windows per agent
        self.context_windows: Dict[str, ContextWindow] = {}
        
        # Track token usage statistics
        self.stats = {
            "total_tokens_processed": 0,
            "truncations_performed": 0,
            "summaries_generated": 0,
            "paginations_performed": 0,
            "independent_contexts_created": 0,
        }
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate token count from text.
        
        Uses character count / 4 as approximation.
        This is conservative for English text with LLaMA-style tokenizers.
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        return len(text) // ContextManager.CHARS_PER_TOKEN
    
    @staticmethod
    def tokens_to_chars(tokens: int) -> int:
        """Convert token count to approximate character count"""
        return tokens * ContextManager.CHARS_PER_TOKEN
    
    def create_context_window(
        self, 
        agent_id: str, 
        max_tokens: int = None
    ) -> ContextWindow:
        """
        Create a new context window for an agent.
        
        Args:
            agent_id: Unique identifier for the agent/task
            max_tokens: Optional custom token limit
            
        Returns:
            New ContextWindow instance
        """
        window = ContextWindow(
            agent_id=agent_id,
            max_tokens=max_tokens or self.max_context_tokens
        )
        self.context_windows[agent_id] = window
        return window
    
    def create_independent_context(
        self,
        agent_id: str,
        max_tokens: int = None,
    ) -> ContextWindow:
        """
        Create an independent context window not tied to any parent budget.
        
        v1.2.0: Used by the diagnostic agent to get its own 25K token budget
        per invocation, completely separate from the sub-agent's context.
        
        The key difference from create_context_window() is semantic: independent
        contexts are marked as such for stats tracking, and callers are expected
        to use unique agent_ids per invocation (e.g. "diag_step01_inv1") so
        each invocation gets a fresh budget.
        
        If an independent context with this agent_id already exists, it is
        reset to provide a fresh budget rather than creating a duplicate.
        
        Args:
            agent_id: Unique identifier (should include invocation number)
            max_tokens: Token budget for this context (default: 25000)
            
        Returns:
            Fresh ContextWindow instance with independent budget
            
        Example:
            # Each diagnostic invocation gets fresh 25K
            ctx = cm.create_independent_context(
                f"diag_{step_id}_inv{n}", max_tokens=25000
            )
        """
        budget = max_tokens or self.DEFAULT_MAX_TOOL_OUTPUT_TOKENS
        
        # If this ID already exists, reset it for reuse
        existing = self.context_windows.get(agent_id)
        if existing and existing.independent:
            existing.reset()
            existing.max_tokens = budget
            return existing
        
        window = ContextWindow(
            agent_id=agent_id,
            max_tokens=budget,
            independent=True,
        )
        self.context_windows[agent_id] = window
        self.stats["independent_contexts_created"] += 1
        return window
    
    def reset_context_window(self, agent_id: str) -> bool:
        """
        Reset an existing context window to empty state.
        
        v1.2.0: Useful when the diagnostic agent needs a fresh budget
        for a new invocation but wants to reuse the same agent_id.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            True if window was found and reset, False if not found
        """
        window = self.context_windows.get(agent_id)
        if window:
            window.reset()
            return True
        return False
    
    def destroy_context_window(self, agent_id: str) -> bool:
        """
        Remove a context window entirely.
        
        v1.2.0: Used to clean up diagnostic agent windows after
        the sub-agent finishes a subtask.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            True if window was found and removed, False if not found
        """
        if agent_id in self.context_windows:
            del self.context_windows[agent_id]
            return True
        return False
    
    def get_context_window(self, agent_id: str) -> Optional[ContextWindow]:
        """Get existing context window for an agent"""
        return self.context_windows.get(agent_id)
    
    def add_to_context(
        self,
        agent_id: str,
        role: str,
        content: str,
        metadata: Dict = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Add content to an agent's context window.
        
        If the content would exceed limits, truncates history and summarizes.
        
        Args:
            agent_id: Agent identifier
            role: Message role ('user', 'assistant', 'tool', 'system')
            content: Message content
            metadata: Optional metadata
            
        Returns:
            Tuple of (success, warning_message)
        """
        window = self.context_windows.get(agent_id)
        if not window:
            window = self.create_context_window(agent_id)
        
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        # Check if it fits
        if window.add_message(message):
            self.stats["total_tokens_processed"] += message.token_estimate
            return True, None
        
        # Need to truncate history to make room
        warning = self._truncate_and_summarize(window, message.token_estimate)
        
        # Try again after truncation
        if window.add_message(message):
            self.stats["total_tokens_processed"] += message.token_estimate
            return True, warning
        
        # Still doesn't fit - content itself is too large
        return False, f"Content too large ({message.token_estimate} tokens) even after truncation"
    
    def _truncate_and_summarize(
        self, 
        window: ContextWindow, 
        needed_tokens: int
    ) -> str:
        """
        Truncate old messages and create summary to free up tokens.
        
        Strategy:
        1. Keep system messages
        2. Keep most recent N messages
        3. Summarize middle section
        
        Args:
            window: Context window to truncate
            needed_tokens: Tokens we need to free up
            
        Returns:
            Warning message about what was truncated
        """
        self.stats["truncations_performed"] += 1
        
        # Separate messages by type
        system_messages = [m for m in window.messages if m.role == 'system']
        other_messages = [m for m in window.messages if m.role != 'system']
        
        if len(other_messages) < 4:
            # Not enough to truncate, just remove oldest non-system
            if other_messages:
                removed = other_messages.pop(0)
                window.messages = system_messages + other_messages
                window.total_tokens -= removed.token_estimate
                return f"Removed oldest message ({removed.token_estimate} tokens)"
            return "No messages to truncate"
        
        # Calculate how much to keep at the end (recent context)
        # Keep roughly 30% of max context as recent history
        keep_recent_tokens = int(window.max_tokens * self._recent_history_percent)
        
        # Find split point - keep messages from end that fit in keep_recent_tokens
        recent_messages = []
        recent_tokens = 0
        for msg in reversed(other_messages):
            if recent_tokens + msg.token_estimate <= keep_recent_tokens:
                recent_messages.insert(0, msg)
                recent_tokens += msg.token_estimate
            else:
                break
        
        # Messages to summarize (everything not in recent)
        to_summarize = other_messages[:-len(recent_messages)] if recent_messages else other_messages
        
        if not to_summarize:
            return "No messages old enough to summarize"
        
        # Generate summary
        summary_text = self._generate_summary(to_summarize)
        summary_message = Message(
            role="system",
            content=f"[SUMMARY OF PREVIOUS CONTEXT]\n{summary_text}\n[END SUMMARY]",
            metadata={"is_summary": True, "messages_summarized": len(to_summarize)}
        )
        
        # Calculate tokens freed
        old_tokens = sum(m.token_estimate for m in to_summarize)
        
        # Rebuild message list
        window.messages = system_messages + [summary_message] + recent_messages
        window.total_tokens = sum(m.token_estimate for m in window.messages)
        window.summaries.append(summary_text)
        
        self.stats["summaries_generated"] += 1
        
        return f"Summarized {len(to_summarize)} messages ({old_tokens} tokens) → {summary_message.token_estimate} tokens"
    
    def _generate_summary(self, messages: List[Message]) -> str:
        """
        Generate a summary of messages.
        
        Uses LLM if available, otherwise creates structured summary.
        
        Args:
            messages: Messages to summarize
            
        Returns:
            Summary text
        """
        # Build content to summarize
        content_parts = []
        for msg in messages:
            prefix = msg.role.upper()
            content_parts.append(f"[{prefix}]: {msg.content[:500]}...")
        
        full_content = "\n".join(content_parts)
        
        if self.llm:
            # Use LLM for summarization
            prompt = f"""Summarize the following conversation history concisely.
Focus on:
1. Key decisions made
2. Files created/modified
3. Errors encountered
4. Current task status

Content to summarize:
{full_content[:8000]}

Provide a concise summary (max 500 words):"""
            
            try:
                summary = self.llm.invoke(prompt)
                return summary[:2000]  # Cap summary length
            except Exception as e:
                pass  # Fall through to manual summary
        
        # Manual structured summary
        summary_parts = [
            f"Previous conversation ({len(messages)} messages):",
        ]
        
        # Extract key information
        files_mentioned = set()
        errors = []
        decisions = []
        
        for msg in messages:
            content = msg.content
            
            # Find file paths
            file_patterns = [
                r'[\w/]+\.(?:py|yaml|yml|h5ad|csv|txt|sh|sbatch)',
            ]
            for pattern in file_patterns:
                files_mentioned.update(re.findall(pattern, content))
            
            # Find errors
            if 'error' in content.lower() or 'failed' in content.lower():
                # Extract error line
                for line in content.split('\n'):
                    if 'error' in line.lower() or 'failed' in line.lower():
                        errors.append(line.strip()[:100])
                        break
        
        if files_mentioned:
            summary_parts.append(f"Files referenced: {', '.join(list(files_mentioned)[:10])}")
        
        if errors:
            summary_parts.append(f"Errors encountered: {'; '.join(errors[:3])}")
        
        return "\n".join(summary_parts)
    
    def paginate_tool_output(
        self, 
        output: str, 
        page_size_tokens: int = None
    ) -> List[Dict[str, Any]]:
        """
        Split large tool output into pages.
        
        Args:
            output: Tool output to paginate
            page_size_tokens: Tokens per page (default: 25K)
            
        Returns:
            List of page dictionaries with content and metadata
        """
        page_size = page_size_tokens or self.max_tool_output_tokens
        page_size_chars = self.tokens_to_chars(page_size)
        
        if len(output) <= page_size_chars:
            return [{
                "page": 1,
                "total_pages": 1,
                "content": output,
                "tokens": self.estimate_tokens(output),
                "truncated": False
            }]
        
        self.stats["paginations_performed"] += 1
        
        # Split into pages
        pages = []
        current_pos = 0
        page_num = 1
        
        while current_pos < len(output):
            end_pos = min(current_pos + page_size_chars, len(output))
            
            # Try to break at a newline for cleaner splits
            if end_pos < len(output):
                newline_pos = output.rfind('\n', current_pos, end_pos)
                if newline_pos > current_pos + page_size_chars // 2:
                    end_pos = newline_pos + 1
            
            page_content = output[current_pos:end_pos]
            pages.append({
                "page": page_num,
                "content": page_content,
                "tokens": self.estimate_tokens(page_content),
                "start_char": current_pos,
                "end_char": end_pos
            })
            
            current_pos = end_pos
            page_num += 1
        
        # Add total_pages to each
        for page in pages:
            page["total_pages"] = len(pages)
            page["truncated"] = True
        
        return pages
    
    def truncate_for_prompt(
        self, 
        text: str, 
        max_tokens: int,
        keep_end: bool = True
    ) -> Tuple[str, bool]:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed
            keep_end: If True, keep end of text; if False, keep beginning
            
        Returns:
            Tuple of (truncated_text, was_truncated)
        """
        current_tokens = self.estimate_tokens(text)
        
        if current_tokens <= max_tokens:
            return text, False
        
        max_chars = self.tokens_to_chars(max_tokens)
        
        if keep_end:
            truncated = "...[truncated]...\n" + text[-max_chars:]
        else:
            truncated = text[:max_chars] + "\n...[truncated]..."
        
        return truncated, True
    
    def format_for_context(
        self,
        content: str,
        content_type: str,
        max_tokens: int = None
    ) -> str:
        """
        Format and potentially truncate content for context window.
        
        Args:
            content: Content to format
            content_type: Type of content ('code', 'log', 'output', 'error')
            max_tokens: Optional token limit
            
        Returns:
            Formatted content
        """
        max_tokens = max_tokens or self.max_tool_output_tokens
        
        # Apply type-specific formatting
        if content_type == 'code':
            # Keep code structure, truncate middle if needed
            lines = content.split('\n')
            if self.estimate_tokens(content) > max_tokens:
                # Keep first 40% and last 40%
                keep_lines = int(len(lines) * 0.4)
                content = '\n'.join(
                    lines[:keep_lines] + 
                    [f"\n... [{len(lines) - 2*keep_lines} lines truncated] ...\n"] +
                    lines[-keep_lines:]
                )
        
        elif content_type == 'log':
            # For logs, keep end (most recent)
            content, _ = self.truncate_for_prompt(content, max_tokens, keep_end=True)
        
        elif content_type == 'error':
            # For errors, keep beginning (traceback root)
            content, _ = self.truncate_for_prompt(content, max_tokens, keep_end=False)
        
        else:
            # Default truncation
            content, _ = self.truncate_for_prompt(content, max_tokens)
        
        return content
    
    def get_context_status(self, agent_id: str) -> Dict[str, Any]:
        """Get status of an agent's context window"""
        window = self.context_windows.get(agent_id)
        if not window:
            return {"exists": False}
        
        return {
            "exists": True,
            "agent_id": agent_id,
            "total_tokens": window.total_tokens,
            "max_tokens": window.max_tokens,
            "remaining_tokens": window.get_remaining_tokens(),
            "usage_percent": (window.total_tokens / window.max_tokens) * 100,
            "message_count": len(window.messages),
            "summaries_count": len(window.summaries),
            "independent": window.independent,
        }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get overall context management statistics"""
        independent_count = sum(
            1 for w in self.context_windows.values() if w.independent
        )
        return {
            **self.stats,
            "active_windows": len(self.context_windows),
            "independent_windows": independent_count,
            "windows": {
                agent_id: window.to_dict() 
                for agent_id, window in self.context_windows.items()
            }
        }
    
    def should_continue(self, agent_id: str, min_tokens_needed: int = 5000) -> Tuple[bool, str]:
        """
        Check if agent has enough context budget to continue.
        
        Args:
            agent_id: Agent identifier
            min_tokens_needed: Minimum tokens needed to continue
            
        Returns:
            Tuple of (can_continue, reason)
        """
        window = self.context_windows.get(agent_id)
        if not window:
            return True, "No context window (unlimited)"
        
        remaining = window.get_remaining_tokens()
        
        if remaining >= min_tokens_needed:
            return True, f"{remaining} tokens remaining"
        
        return False, f"Only {remaining} tokens remaining (need {min_tokens_needed})"
    
    def cleanup_independent_contexts(self, prefix: str = None) -> int:
        """
        Remove all independent context windows, optionally filtered by prefix.
        
        v1.2.0: Called by sub-agent after completing a subtask to free
        diagnostic agent context windows.
        
        Args:
            prefix: If provided, only remove windows whose agent_id
                    starts with this prefix (e.g. "diag_step01")
                    
        Returns:
            Number of windows removed
        """
        to_remove = []
        for agent_id, window in self.context_windows.items():
            if window.independent:
                if prefix is None or agent_id.startswith(prefix):
                    to_remove.append(agent_id)
        
        for agent_id in to_remove:
            del self.context_windows[agent_id]
        
        return len(to_remove)


# Global instance for convenience
context_manager = ContextManager()
