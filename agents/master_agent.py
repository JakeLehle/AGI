"""
Master Agent v3.2 - Comprehensive Task Decomposition with Validation

The master agent now follows a 3-phase approach:
1. EXTRACTION: Parse the detailed prompt and extract EVERY step exactly as written
2. EXPANSION: Expand each step into a comprehensive execution plan
3. VALIDATION: Verify each plan captures the original step's full intent

Key improvements over v3.1:
- Multi-format step detection (numbered, checkboxes, headers, bullet points)
- Preserves original step text verbatim alongside expanded plan
- Validates expanded plans against original steps
- Richer subtask format with original_text, expanded_plan, validation_status
- Better handling of code snippets and implementation hints in prompts

v3.2 Updates:
- Token budget sized for qwen3-coder-next (32K context → 25K working limit)
- Default model switched to qwen3-coder-next
- Multi-strategy JSON parsing with cleanup for malformed LLM output
- Per-step and total decomposition timeout to prevent pipeline freezes
- requires_gpu flag auto-detected from packages for dual cluster routing
- GPU package detection list aligned with sub_agent.py (18 packages)

The master prompt serves as a living document that:
1. Contains all pipeline steps with status
2. Gets updated when subtasks complete (adds script paths)
3. Gets updated when subtasks fail (adds error summaries)
4. Maintains a comprehensive view of pipeline state
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import re
import json
import signal
from datetime import datetime

try:
    from langchain_ollama import OllamaLLM
except ImportError:
    from langchain_community.llms import Ollama as OllamaLLM

from utils.logging_config import agent_logger
from utils.context_manager import ContextManager


# =============================================================================
# TIMEOUT HELPERS
# =============================================================================

class LLMTimeoutError(Exception):
    """Raised when an LLM call exceeds the allowed time."""
    pass


def _timeout_handler(signum, frame):
    raise LLMTimeoutError("LLM call timed out")


def invoke_with_timeout(llm, prompt: str, timeout_seconds: int = 300) -> str:
    """
    Invoke an LLM with a timeout guard.

    Uses SIGALRM on Unix. Falls back to no-timeout on platforms that lack it
    (Windows) or when called from a non-main thread (signal only works in main).

    Args:
        llm: LangChain LLM instance
        prompt: The prompt string
        timeout_seconds: Max seconds to wait (default 5 min)

    Returns:
        The LLM response string

    Raises:
        LLMTimeoutError: If the call exceeds timeout_seconds
    """
    use_signal = hasattr(signal, 'SIGALRM')
    if use_signal:
        try:
            # Only works in the main thread
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout_seconds)
        except ValueError:
            # "signal only works in main thread"
            use_signal = False

    try:
        response = llm.invoke(prompt)
        return response
    finally:
        if use_signal:
            signal.alarm(0)  # Cancel the alarm
            signal.signal(signal.SIGALRM, old_handler)


# =============================================================================
# JSON PARSING HELPERS
# =============================================================================

def parse_json_resilient(text: str) -> Optional[Dict]:
    """
    Multi-strategy JSON extraction from LLM output.

    Handles the common failure modes of smaller local models like
    qwen3-coder-next:
      1. Clean JSON inside a ```json ... ``` code fence
      2. Clean JSON inside a ``` ... ``` code fence (no language tag)
      3. First { ... } blob via greedy regex
      4. Truncated JSON (missing closing braces) — attempt brace balancing
      5. Trailing garbage after the closing } — strip and retry

    Returns the parsed dict, or None if all strategies fail.
    """
    if not text or not text.strip():
        return None

    # --- Strategy 1: ```json ... ``` code fence ----
    fence_match = re.search(r'```json\s*\n?([\s\S]*?)```', text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # --- Strategy 2: ``` ... ``` code fence (no language tag) ---
    fence_match2 = re.search(r'```\s*\n?([\s\S]*?)```', text)
    if fence_match2:
        candidate = fence_match2.group(1).strip()
        if candidate.startswith('{'):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    # --- Strategy 3: First { ... } via greedy regex ---
    brace_match = re.search(r'\{[\s\S]*\}', text)
    if brace_match:
        candidate = brace_match.group()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # --- Strategy 4: Strip trailing garbage after last valid } ---
            # Walk backwards to find the matching closing brace
            cleaned = _balance_braces(candidate)
            if cleaned:
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass

    # --- Strategy 5: Truncated JSON — force-close open braces ---
    first_brace = text.find('{')
    if first_brace >= 0:
        fragment = text[first_brace:]
        # Remove any trailing non-JSON text after last }
        last_brace = fragment.rfind('}')
        if last_brace >= 0:
            fragment = fragment[:last_brace + 1]
        # Count open vs close braces and force-close
        open_count = fragment.count('{') - fragment.count('}')
        if open_count > 0:
            fragment = fragment.rstrip().rstrip(',') + '}' * open_count
            try:
                return json.loads(fragment)
            except json.JSONDecodeError:
                pass

    return None


def _balance_braces(text: str) -> Optional[str]:
    """
    Walk the string tracking brace depth; return the substring from the first
    opening brace to the position where depth returns to 0.
    """
    depth = 0
    start = None
    in_string = False
    escape_next = False

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == '{':
            if start is None:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i + 1]

    return None


# =============================================================================
# GPU PACKAGE DETECTION
# =============================================================================

# Must stay in sync with sub_agent.py GPU_PACKAGES list
GPU_PACKAGES = {
    'torch', 'pytorch', 'tensorflow', 'keras', 'jax', 'cupy',
    'rapids', 'cuml', 'cudf', 'cugraph', 'dask-cuda',
    'nvidia', 'cuda', 'cudnn', 'tensorrt',
    'scvi', 'scvi-tools', 'cellbender', 'deepcell',
}


def detect_requires_gpu(packages: List[str], text: str = "") -> bool:
    """
    Determine whether a subtask requires GPU resources.

    Checks the explicit package list first (fast, reliable), then falls
    back to scanning free text for GPU keywords.

    Args:
        packages: List of package names from expansion
        text: Combined description + expanded_plan text for fallback scan

    Returns:
        True if the task should be routed to a GPU cluster
    """
    # Check explicit package list
    for pkg in packages:
        if pkg.lower() in GPU_PACKAGES:
            return True

    # Fallback: scan text for GPU-related keywords
    text_lower = text.lower()
    gpu_keywords = [
        'gpu', 'cuda', 'torch.', 'tensorflow', 'keras.',
        'nvidia', '.to(device)', 'scvi.model', 'cellbender',
    ]
    return any(kw in text_lower for kw in gpu_keywords)


# =============================================================================
# MASTER PROMPT DOCUMENT
# =============================================================================

class MasterPromptDocument:
    """
    Living document that tracks pipeline state.

    Structure:
    - Original prompt (preserved verbatim)
    - Extracted steps (parsed from original)
    - Expanded plans (detailed execution plans)
    - Validation status for each step
    - Completion status with script paths and outputs
    """

    def __init__(self, original_prompt: str, project_dir: Path):
        self.original_prompt = original_prompt
        self.project_dir = project_dir
        self.created_at = datetime.now()
        self.last_updated = datetime.now()

        # Step tracking - now includes more detail
        self.steps: Dict[str, Dict] = {}  # step_id -> step info
        self.step_order: List[str] = []   # Ordered list of step IDs

        # Extracted vs Expanded tracking
        self.extracted_steps: List[Dict] = []  # Raw extracted steps
        self.validation_results: Dict[str, Dict] = {}  # step_id -> validation

        # Document paths
        self.document_path = project_dir / "reports" / "master_prompt_state.json"
        self.markdown_path = project_dir / "reports" / "pipeline_status.md"

        # Ensure directories exist
        self.document_path.parent.mkdir(parents=True, exist_ok=True)

    def add_step(
        self,
        step_id: str,
        title: str,
        description: str,
        original_text: str = "",
        expanded_plan: str = "",
        validation_status: str = "pending",
        packages: List[str] = None,
        dependencies: List[str] = None,
        code_hints: List[str] = None,
    ):
        """Add a step with full context preservation"""
        self.steps[step_id] = {
            'id': step_id,
            'title': title,
            'description': description,
            'original_text': original_text,
            'expanded_plan': expanded_plan,
            'validation_status': validation_status,
            'status': 'pending',
            'attempts': 0,
            'script_path': None,
            'output_files': [],
            'error_summary': None,
            'packages': packages or [],
            'dependencies': dependencies or [],
            'code_hints': code_hints or [],
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
        if step_id not in self.step_order:
            self.step_order.append(step_id)
        self._save()

    def mark_running(self, step_id: str):
        """Mark step as currently running"""
        if step_id in self.steps:
            self.steps[step_id]['status'] = 'running'
            self.steps[step_id]['last_updated'] = datetime.now().isoformat()
            self._save()

    def mark_complete(self, step_id: str, script_path: str = None, output_files: List[str] = None):
        """Mark step as complete with results"""
        if step_id in self.steps:
            self.steps[step_id]['status'] = 'completed'
            self.steps[step_id]['script_path'] = script_path
            self.steps[step_id]['output_files'] = output_files or []
            self.steps[step_id]['last_updated'] = datetime.now().isoformat()
            self._save()

    def mark_failed(self, step_id: str, error_summary: str, attempts: int = 1, script_path: str = None):
        """Mark step as failed with error info"""
        if step_id in self.steps:
            self.steps[step_id]['status'] = 'failed'
            self.steps[step_id]['error_summary'] = error_summary[:1000]
            self.steps[step_id]['attempts'] = attempts
            self.steps[step_id]['script_path'] = script_path
            self.steps[step_id]['last_updated'] = datetime.now().isoformat()
            self._save()

    def update_validation(self, step_id: str, status: str, issues: List[str] = None):
        """Update validation status for a step"""
        self.validation_results[step_id] = {
            'status': status,
            'issues': issues or [],
            'validated_at': datetime.now().isoformat()
        }
        if step_id in self.steps:
            self.steps[step_id]['validation_status'] = status
        self._save()

    def _save(self):
        """Save state to JSON file"""
        self.last_updated = datetime.now()
        state = {
            'original_prompt': self.original_prompt,
            'steps': self.steps,
            'step_order': self.step_order,
            'extracted_steps': self.extracted_steps,
            'validation_results': self.validation_results,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }

        self.document_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.document_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)

        # Also save markdown version
        md = self.generate_status_markdown()
        self.markdown_path.write_text(md)

    def generate_status_markdown(self) -> str:
        """Generate markdown status document"""
        lines = [
            "# Pipeline Status",
            f"\nLast Updated: {self.last_updated.isoformat()}",
            f"\n## Summary",
            f"- Total Steps: {len(self.steps)}",
            f"- Completed: {len([s for s in self.steps.values() if s['status'] == 'completed'])}",
            f"- Failed: {len([s for s in self.steps.values() if s['status'] == 'failed'])}",
            f"- Pending: {len([s for s in self.steps.values() if s['status'] == 'pending'])}",
            f"- Running: {len([s for s in self.steps.values() if s['status'] == 'running'])}",
            "\n## Steps\n"
        ]

        for step_id in self.step_order:
            step = self.steps.get(step_id, {})
            status_icon = {
                'completed': '✅',
                'failed': '❌',
                'running': '🔄',
                'pending': '⏳'
            }.get(step.get('status', 'pending'), '⏳')

            validation = '✓' if step.get('validation_status') == 'validated' else '?'

            lines.append(f"### {status_icon} {step.get('title', step_id)} [{validation}]")
            lines.append(f"\n**Status**: {step.get('status', 'pending')}")

            if step.get('original_text'):
                lines.append(f"\n**Original Step**:\n> {step['original_text'][:500]}...")

            if step.get('expanded_plan'):
                lines.append(f"\n**Execution Plan**:\n{step['expanded_plan'][:1000]}...")

            if step.get('script_path'):
                lines.append(f"\n**Script**: `{step['script_path']}`")

            if step.get('error_summary'):
                lines.append(f"\n**Error**: {step['error_summary'][:300]}...")

            lines.append("")

        return '\n'.join(lines)

    @classmethod
    def load(cls, project_dir: Path) -> Optional['MasterPromptDocument']:
        """Load existing state from file"""
        state_path = project_dir / "reports" / "master_prompt_state.json"
        if not state_path.exists():
            return None

        try:
            with open(state_path) as f:
                state = json.load(f)

            doc = cls(state['original_prompt'], project_dir)
            doc.steps = state.get('steps', {})
            doc.step_order = state.get('step_order', [])
            doc.extracted_steps = state.get('extracted_steps', [])
            doc.validation_results = state.get('validation_results', {})
            doc.created_at = datetime.fromisoformat(state['created_at'])
            doc.last_updated = datetime.fromisoformat(state['last_updated'])
            return doc
        except Exception as e:
            print(f"Warning: Could not load master document: {e}")
            return None

    def to_context_string(self, max_tokens: int = 10000) -> str:
        """Generate string representation for LLM context"""
        completed = [s for s in self.steps.values() if s['status'] == 'completed']
        failed = [s for s in self.steps.values() if s['status'] == 'failed']
        pending = [s for s in self.steps.values() if s['status'] == 'pending']

        parts = [
            f"Pipeline Status: {len(completed)} completed, {len(failed)} failed, {len(pending)} pending",
            "",
            "Completed steps:"
        ]

        for s in completed[:10]:
            parts.append(f"  ✅ {s['title']}: {s.get('script_path', 'N/A')}")

        if failed:
            parts.append("\nFailed (need attention):")
            for s in failed[:5]:
                parts.append(f"  ❌ {s['title']}: {s.get('error_summary', 'Unknown')[:100]}")

        if pending:
            parts.append("\nPending:")
            for s in pending[:10]:
                parts.append(f"  ⏳ {s['title']}")

        return "\n".join(parts)


# =============================================================================
# MASTER AGENT
# =============================================================================

class MasterAgent:
    """
    Master agent v3.2 with comprehensive task decomposition.

    Key responsibilities:
    1. EXTRACT: Parse ALL steps from the detailed prompt
    2. EXPAND: Create detailed execution plans for each step
    3. VALIDATE: Verify plans match original intent
    4. ASSIGN: Hand off validated plans to sub-agents
    5. TRACK: Maintain the master prompt document

    v3.2 token budget sized for qwen3-coder-next (32K context):
      - MAX_CONTEXT_TOKENS:       25,000  (leaves ~7K for system prompt + response)
      - STEP_EXPAND_TIMEOUT:         300  (5 min per step expansion LLM call)
      - TOTAL_DECOMPOSITION_TIMEOUT: 1800 (30 min for entire decomposition)
    """

    # ── Token budget (must match sub_agent.py / config.yaml) ──────────────
    MAX_CONTEXT_TOKENS = 25_000

    # ── Timeout guards ────────────────────────────────────────────────────
    STEP_EXPAND_TIMEOUT = 300       # seconds per _expand_step LLM call
    TOTAL_DECOMPOSITION_TIMEOUT = 1800  # seconds for entire decompose_task
    REVIEW_TIMEOUT = 120            # seconds for review_failure LLM call

    def __init__(
        self,
        sandbox=None,
        ollama_model: str = "qwen3-coder-next",
        ollama_base_url: str = "http://127.0.0.1:11434",
        **kwargs
    ):
        self.llm = OllamaLLM(model=ollama_model, base_url=ollama_base_url)
        self.agent_id = "master"
        self.sandbox = sandbox

        # Project directory
        if sandbox:
            self.project_dir = Path(sandbox.project_dir)
        else:
            self.project_dir = Path(kwargs.get('project_dir', '.'))

        # Context management
        self.context_mgr = ContextManager(
            max_context_tokens=self.MAX_CONTEXT_TOKENS,
            llm_for_summarization=self.llm
        )

        # Master prompt document
        self.master_document: Optional[MasterPromptDocument] = None

        # Subtask tracking
        self.subtask_status: Dict[str, Dict] = {}

        # Store additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def initialize_document(self, main_task: str) -> MasterPromptDocument:
        """Initialize or load the master prompt document"""
        existing = MasterPromptDocument.load(self.project_dir)
        if existing:
            self.master_document = existing
            agent_logger.log_reflection(
                agent_name=self.agent_id,
                task_id="init",
                reflection=f"Loaded existing master document with {len(existing.steps)} steps"
            )
        else:
            self.master_document = MasterPromptDocument(main_task, self.project_dir)
            agent_logger.log_reflection(
                agent_name=self.agent_id,
                task_id="init",
                reflection="Created new master document"
            )

        return self.master_document

    # =========================================================================
    # PHASE 1: COMPREHENSIVE EXTRACTION
    # =========================================================================

    def _extract_steps_from_prompt(self, prompt: str) -> List[Dict[str, Any]]:
        """
        Extract ALL steps from the prompt using multiple detection strategies.

        Supports:
        - Numbered steps: "1. ", "1) ", "Step 1:"
        - Checkboxes: "- [ ]", "- [x]", "* [ ]"
        - Headers: "## Step 1", "### Task 1"
        - Bullet points with keywords: "- First,", "- Next,"
        - Code blocks associated with steps
        """
        extracted = []
        lines = prompt.split('\n')

        current_step = None
        current_content = []
        current_code_blocks = []
        in_code_block = False
        code_block_content = []
        step_number = 0

        for i, line in enumerate(lines):
            # Track code blocks
            if line.strip().startswith('```'):
                if in_code_block:
                    # End of code block
                    in_code_block = False
                    if current_step is not None:
                        current_code_blocks.append('\n'.join(code_block_content))
                    code_block_content = []
                else:
                    # Start of code block
                    in_code_block = True
                continue

            if in_code_block:
                code_block_content.append(line)
                continue

            # Detect step patterns
            step_match = None
            step_title = None

            # Pattern 1: Numbered steps "1. ", "1) ", "1:"
            numbered = re.match(r'^\s*(\d+)[.):]\s*(.+)', line)
            if numbered:
                step_match = numbered
                step_title = numbered.group(2).strip()

            # Pattern 2: "Step N:" or "Task N:"
            step_keyword = re.match(r'^\s*(?:Step|Task|Phase)\s*(\d+)[:\s]*(.+)?', line, re.IGNORECASE)
            if step_keyword and not step_match:
                step_match = step_keyword
                step_title = step_keyword.group(2).strip() if step_keyword.group(2) else f"Step {step_keyword.group(1)}"

            # Pattern 3: Markdown headers "## Step", "### 1."
            header = re.match(r'^#{1,4}\s*(?:Step\s*)?(\d+)?[.:\s]*(.+)?', line)
            if header and not step_match:
                # Only treat as step if it looks like a task header
                header_text = header.group(2) or ""
                if any(kw in header_text.lower() for kw in ['step', 'task', 'phase', 'stage', 'part']):
                    step_match = header
                    step_title = header_text.strip()

            # Pattern 4: Checkbox items "- [ ]", "- [x]"
            checkbox = re.match(r'^\s*[-*]\s*\[([x\s])\]\s*(.+)', line, re.IGNORECASE)
            if checkbox and not step_match:
                step_match = checkbox
                step_title = checkbox.group(2).strip()

            # Pattern 5: Bold/emphasized items that look like steps
            bold_step = re.match(r'^\s*[-*]?\s*\*\*(.+?)\*\*[:\s]*(.+)?', line)
            if bold_step and not step_match:
                title_text = bold_step.group(1).strip()
                # Check if it looks like a step
                if any(kw in title_text.lower() for kw in ['step', 'task', 'create', 'implement', 'setup', 'configure', 'run', 'analyze']):
                    step_match = bold_step
                    step_title = title_text + (f": {bold_step.group(2)}" if bold_step.group(2) else "")

            # If we found a new step
            if step_match:
                # Save previous step if exists
                if current_step is not None:
                    current_step['full_text'] = '\n'.join(current_content).strip()
                    current_step['code_hints'] = current_code_blocks.copy()
                    extracted.append(current_step)

                # Start new step
                step_number += 1
                current_step = {
                    'step_number': step_number,
                    'title': step_title[:200] if step_title else f"Step {step_number}",
                    'full_text': '',
                    'code_hints': [],
                    'line_start': i
                }
                current_content = [line]
                current_code_blocks = []

            elif current_step is not None:
                # Continue building current step content
                current_content.append(line)

        # Don't forget the last step
        if current_step is not None:
            current_step['full_text'] = '\n'.join(current_content).strip()
            current_step['code_hints'] = current_code_blocks.copy()
            extracted.append(current_step)

        # If no steps were found, treat the whole prompt as one step
        if not extracted:
            extracted.append({
                'step_number': 1,
                'title': 'Main Task',
                'full_text': prompt,
                'code_hints': [],
                'line_start': 0
            })

        return extracted

    def _extract_task_context(self, task: str) -> Dict[str, Any]:
        """Extract critical context from task description"""
        context = {
            "language": None,
            "packages": [],
            "reference_scripts": [],
            "input_files": [],
            "output_files": [],
            "completed_steps": [],
            "huggingface_repos": [],
            "environment_hints": []
        }

        # Detect language
        task_lower = task.lower()
        python_indicators = ['python', 'scanpy', 'squidpy', 'anndata', 'pandas', '.py', 'popv', 'h5ad', 'numpy', 'scipy']
        r_indicators = ['seurat', 'singlecell', 'bioconductor', '.R', 'r script']

        python_score = sum(1 for ind in python_indicators if ind in task_lower)
        r_score = sum(1 for ind in r_indicators if ind in task_lower)

        context["language"] = "python" if python_score >= r_score else "r"

        # Extract packages (expanded list)
        python_packages = [
            'scanpy', 'squidpy', 'anndata', 'pandas', 'numpy', 'scipy',
            'popv', 'popV', 'scvi', 'scvi-tools', 'cellxgene', 'leidenalg',
            'matplotlib', 'seaborn', 'celltypist', 'decoupler', 'sklearn',
            'scikit-learn', 'torch', 'pytorch', 'tensorflow', 'keras',
            'statsmodels', 'plotly', 'bokeh', 'networkx', 'igraph',
            'requests', 'aiohttp', 'fastapi', 'flask', 'django'
        ]
        for pkg in python_packages:
            if pkg.lower() in task_lower:
                pattern = re.compile(re.escape(pkg), re.IGNORECASE)
                matches = pattern.findall(task)
                if matches:
                    context["packages"].append(matches[0])

        # Extract file paths
        file_patterns = [
            (r'[`"\']?([^\s`"\']*\.h5ad)[`"\']?', 'h5ad'),
            (r'[`"\']?([^\s`"\']*\.csv)[`"\']?', 'csv'),
            (r'[`"\']?([^\s`"\']*\.tsv)[`"\']?', 'tsv'),
            (r'[`"\']?([^\s`"\']*\.parquet)[`"\']?', 'parquet'),
            (r'[`"\']?(data/[^\s`"\']+)[`"\']?', 'data'),
            (r'[`"\']?(output[s]?/[^\s`"\']+)[`"\']?', 'output'),
        ]
        for pattern, ftype in file_patterns:
            matches = re.findall(pattern, task)
            for match in matches:
                if 'input' in match.lower() or ftype in ['h5ad', 'csv', 'tsv', 'parquet']:
                    if match not in context["input_files"]:
                        context["input_files"].append(match)
                if 'output' in match.lower() or ftype == 'output':
                    if match not in context["output_files"]:
                        context["output_files"].append(match)

        # Extract reference scripts
        script_patterns = [
            r'scripts?/[^\s`"\']+\.py',
            r'[`"\']([^\s`"\']+\.py)[`"\']',
        ]
        for pattern in script_patterns:
            matches = re.findall(pattern, task)
            context["reference_scripts"].extend(matches)

        # Extract HuggingFace repos
        hf_pattern = r'huggingface_repo\s*=\s*["\']([^"\']+)["\']'
        context["huggingface_repos"] = re.findall(hf_pattern, task)

        # Extract completed steps
        completed_patterns = [
            r'✅\s*(?:COMPLETED:?)?\s*([^\n]+)',
            r'\[x\]\s*([^\n]+)',
            r'DONE:\s*([^\n]+)',
        ]
        for pattern in completed_patterns:
            matches = re.findall(pattern, task, re.IGNORECASE)
            context["completed_steps"].extend(matches)

        # Deduplicate
        for key in context:
            if isinstance(context[key], list):
                context[key] = list(dict.fromkeys(context[key]))

        return context

    # =========================================================================
    # PHASE 2: DETAILED EXPANSION
    # =========================================================================

    def _expand_step(self, step: Dict[str, Any], context: Dict[str, Any], all_steps: List[Dict]) -> Dict[str, Any]:
        """
        Expand a single extracted step into a detailed execution plan.

        This is where the magic happens - we take the user's step description
        and create a comprehensive plan that includes:
        - Detailed explanation of what needs to be done
        - Specific packages and imports
        - Input/output file specifications
        - Code structure suggestions
        - Success criteria

        v3.2: Uses invoke_with_timeout + parse_json_resilient for robustness.
        """
        step_text = step.get('full_text', step.get('title', ''))
        code_hints = step.get('code_hints', [])

        # Build context about other steps for dependency awareness
        other_steps_summary = "\n".join([
            f"  - Step {s['step_number']}: {s['title'][:100]}"
            for s in all_steps if s['step_number'] != step.get('step_number')
        ])

        prompt = f"""You are expanding a pipeline step into a detailed execution plan.

=== ORIGINAL STEP FROM USER'S PROMPT ===
Step {step.get('step_number', '?')}: {step.get('title', 'Unknown')}

Full Description:
{step_text}

{"Code Hints from User:" if code_hints else ""}
{chr(10).join(['```' + ch + '```' for ch in code_hints[:3]]) if code_hints else ""}

=== CONTEXT ===
Language: {context.get('language', 'python')}
Available Packages: {', '.join(context.get('packages', []))}
Input Files: {', '.join(context.get('input_files', []))}
Output Files: {', '.join(context.get('output_files', []))}

Other Steps in Pipeline:
{other_steps_summary}

=== YOUR TASK ===
Create a COMPREHENSIVE execution plan that:
1. Captures EVERYTHING the user described in this step
2. Expands on any code hints or suggestions they provided
3. Specifies exact packages and imports needed
4. Defines clear input and output files
5. Provides a step-by-step implementation approach
6. Includes success criteria (what files should exist, what results expected)

=== OUTPUT FORMAT (JSON) ===
```json
{{
    "expanded_title": "Descriptive title for this step",
    "expanded_plan": "A detailed paragraph explaining exactly what this step does and how to implement it. Include specific function calls, data transformations, and expected outputs. This should be comprehensive enough that a developer could implement it without referring back to the original prompt.",
    "packages": ["list", "of", "specific", "packages"],
    "imports_needed": ["import pandas as pd", "import scanpy as sc"],
    "input_files": ["path/to/input.h5ad"],
    "output_files": ["path/to/output.h5ad"],
    "key_operations": ["Load data", "Filter cells", "Normalize", "Save results"],
    "code_approach": "Brief description of the code structure - e.g., 'Use scanpy.pp.filter_cells() with min_genes=200, then normalize_total() with target_sum=1e4'",
    "success_criteria": "File output.h5ad exists and contains filtered data with >1000 cells",
    "dependencies": ["step_1", "step_2"],
    "estimated_complexity": "low|medium|high",
    "potential_issues": ["Memory usage if large dataset", "May need to adjust filter thresholds"]
}}
```

IMPORTANT: Your expanded_plan should be DETAILED and COMPREHENSIVE. Do not summarize - expand and clarify. Include ALL details from the user's original step description."""

        try:
            response = invoke_with_timeout(
                self.llm, prompt, timeout_seconds=self.STEP_EXPAND_TIMEOUT
            )

            # v3.2: Multi-strategy JSON parsing
            expanded = parse_json_resilient(response)
            if expanded:
                return {
                    'success': True,
                    'expanded': expanded,
                    'raw_response': response
                }
            else:
                agent_logger.log_reflection(
                    agent_name=self.agent_id,
                    task_id=f"expand_step_{step.get('step_number', '?')}",
                    reflection=f"All JSON parse strategies failed. Response length: {len(response)}"
                )

        except LLMTimeoutError:
            agent_logger.log_reflection(
                agent_name=self.agent_id,
                task_id=f"expand_step_{step.get('step_number', '?')}",
                reflection=f"LLM call timed out after {self.STEP_EXPAND_TIMEOUT}s"
            )
        except Exception as e:
            agent_logger.log_reflection(
                agent_name=self.agent_id,
                task_id=f"expand_step_{step.get('step_number', '?')}",
                reflection=f"Expansion error: {e}"
            )

        # Fallback: create basic expansion from the raw step text
        return {
            'success': False,
            'expanded': {
                'expanded_title': step.get('title', 'Unknown Step'),
                'expanded_plan': step_text,
                'packages': context.get('packages', []),
                'input_files': context.get('input_files', []),
                'output_files': context.get('output_files', []),
                'key_operations': [],
                'code_approach': '',
                'success_criteria': 'Step completes without error',
                'dependencies': [],
                'estimated_complexity': 'medium',
                'potential_issues': []
            }
        }

    # =========================================================================
    # PHASE 3: VALIDATION
    # =========================================================================

    def _validate_expansion(self, original_step: Dict, expanded: Dict) -> Dict[str, Any]:
        """
        Validate that the expanded plan captures the original step's intent.

        Checks:
        - Does the expanded plan mention key concepts from the original?
        - Are all code hints incorporated?
        - Are input/output files preserved?
        - Is the scope appropriate (not too narrow, not too broad)?
        """
        original_text = original_step.get('full_text', '').lower()
        expanded_plan = expanded.get('expanded_plan', '').lower()
        expanded_title = expanded.get('expanded_title', '').lower()

        issues = []

        # Check 1: Key terms from original should appear in expanded
        common_words = {'the', 'a', 'an', 'is', 'are', 'to', 'for', 'of', 'and', 'or', 'in', 'on', 'with', 'this', 'that'}
        original_words = set(re.findall(r'\b\w{4,}\b', original_text)) - common_words
        expanded_words = set(re.findall(r'\b\w{4,}\b', expanded_plan + ' ' + expanded_title))

        missing_key_terms = []
        for word in original_words:
            if len(word) > 5 and word not in expanded_words:
                if any(indicator in word for indicator in ['file', 'data', 'output', 'input', 'cell', 'gene', 'cluster']):
                    missing_key_terms.append(word)

        if len(missing_key_terms) > 3:
            issues.append(f"Missing key terms from original: {', '.join(missing_key_terms[:5])}")

        # Check 2: Code hints should be reflected
        code_hints = original_step.get('code_hints', [])
        if code_hints:
            code_hint_text = ' '.join(code_hints).lower()
            functions = re.findall(r'(\w+)\s*\(', code_hint_text)
            mentioned_functions = sum(1 for f in functions if f in expanded_plan)
            if functions and mentioned_functions < len(functions) / 2:
                issues.append(f"Code hints not fully incorporated (found {mentioned_functions}/{len(functions)} functions)")

        # Check 3: Expansion should be substantial
        if len(expanded.get('expanded_plan', '')) < len(original_text) * 0.5:
            issues.append("Expanded plan seems too brief compared to original")

        # Check 4: Should have concrete outputs
        if not expanded.get('output_files') and not expanded.get('success_criteria'):
            issues.append("Missing concrete output files or success criteria")

        # Determine validation status
        if not issues:
            status = 'validated'
        elif len(issues) <= 2:
            status = 'validated_with_warnings'
        else:
            status = 'needs_revision'

        return {
            'status': status,
            'issues': issues,
            'coverage_score': 1.0 - (len(issues) * 0.2),
        }

    # =========================================================================
    # MAIN DECOMPOSITION METHOD
    # =========================================================================

    def decompose_task(self, main_task: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Comprehensive task decomposition with extraction, expansion, and validation.

        This is the main entry point that orchestrates the 3-phase approach:
        1. Extract ALL steps from the detailed prompt
        2. Expand each step into a detailed execution plan
        3. Validate each plan against the original

        v3.2: Wrapped with total decomposition timeout.
        """
        decomp_start = datetime.now()

        # Initialize document
        self.initialize_document(main_task)

        # Check for already completed steps
        if self.master_document.steps:
            completed = [s for s in self.master_document.steps.values()
                        if s['status'] == 'completed']
            pending = [s for s in self.master_document.steps.values()
                      if s['status'] == 'pending']

            if completed:
                agent_logger.log_reflection(
                    agent_name=self.agent_id,
                    task_id="decompose",
                    reflection=f"Skipping {len(completed)} completed steps, {len(pending)} pending"
                )

                # Return pending steps for re-execution
                if pending:
                    return self._convert_steps_to_subtasks(pending)

        # Extract global context
        extracted_context = self._extract_task_context(main_task)
        if context:
            for key, value in context.items():
                if isinstance(value, list):
                    extracted_context[key] = list(set(extracted_context.get(key, []) + value))
                elif value:
                    extracted_context[key] = value

        agent_logger.log_reflection(
            agent_name=self.agent_id,
            task_id="decompose",
            reflection=f"Extracted context: {len(extracted_context.get('packages', []))} packages, {len(extracted_context.get('input_files', []))} inputs"
        )

        # =====================================================================
        # PHASE 1: EXTRACTION
        # =====================================================================
        print("\n  Phase 1: Extracting steps from prompt...")
        extracted_steps = self._extract_steps_from_prompt(main_task)
        self.master_document.extracted_steps = extracted_steps

        agent_logger.log_reflection(
            agent_name=self.agent_id,
            task_id="extraction",
            reflection=f"Extracted {len(extracted_steps)} steps from prompt"
        )
        print(f"    Found {len(extracted_steps)} steps")

        # =====================================================================
        # PHASE 2: EXPANSION
        # =====================================================================
        print("\n  Phase 2: Expanding each step into detailed plans...")
        subtasks = []

        for i, step in enumerate(extracted_steps):
            # v3.2: Check total decomposition timeout
            elapsed = (datetime.now() - decomp_start).total_seconds()
            if elapsed > self.TOTAL_DECOMPOSITION_TIMEOUT:
                agent_logger.log_reflection(
                    agent_name=self.agent_id,
                    task_id="decompose",
                    reflection=f"Total decomposition timeout ({self.TOTAL_DECOMPOSITION_TIMEOUT}s) reached at step {i+1}/{len(extracted_steps)}"
                )
                print(f"    ⚠️ Decomposition timeout reached, creating basic expansions for remaining {len(extracted_steps) - i} steps")
                # Create basic expansions for remaining steps
                for j in range(i, len(extracted_steps)):
                    remaining_step = extracted_steps[j]
                    step_id = f"step_{j+1}"
                    basic_subtask = self._create_basic_subtask(
                        remaining_step, step_id, j, extracted_context
                    )
                    subtasks.append(basic_subtask)
                    self._register_subtask_in_document(basic_subtask)
                break

            print(f"    Expanding step {i+1}/{len(extracted_steps)}: {step.get('title', 'Unknown')[:50]}...")

            # Expand the step
            expansion_result = self._expand_step(step, extracted_context, extracted_steps)
            expanded = expansion_result.get('expanded', {})

            # =================================================================
            # PHASE 3: VALIDATION
            # =================================================================
            validation = self._validate_expansion(step, expanded)

            if validation['status'] == 'needs_revision':
                print(f"      ⚠️ Validation issues: {', '.join(validation['issues'][:2])}")
                agent_logger.log_reflection(
                    agent_name=self.agent_id,
                    task_id=f"validate_step_{i+1}",
                    reflection=f"Validation issues: {validation['issues']}"
                )
            else:
                print(f"      ✓ Validated")

            # Create subtask with full context
            step_id = f"step_{i+1}"

            # v3.2: Merge packages from expansion with context packages
            step_packages = expanded.get('packages', extracted_context.get('packages', []))

            # v3.2: Auto-detect GPU requirement
            combined_text = (
                expanded.get('expanded_plan', '') + ' ' +
                step.get('full_text', '') + ' ' +
                expanded.get('code_approach', '')
            )
            requires_gpu = detect_requires_gpu(step_packages, combined_text)

            subtask = {
                'id': step_id,
                'title': expanded.get('expanded_title', step.get('title', f'Step {i+1}')),
                'description': expanded.get('expanded_plan', step.get('full_text', '')),
                'original_text': step.get('full_text', ''),
                'expanded_plan': expanded.get('expanded_plan', ''),
                'language': extracted_context.get('language', 'python'),
                'packages': step_packages,
                'imports_needed': expanded.get('imports_needed', []),
                'input_files': expanded.get('input_files', extracted_context.get('input_files', [])),
                'output_files': expanded.get('output_files', extracted_context.get('output_files', [])),
                'key_operations': expanded.get('key_operations', []),
                'code_approach': expanded.get('code_approach', ''),
                'code_hints': step.get('code_hints', []),
                'success_criteria': expanded.get('success_criteria', ''),
                'dependencies': expanded.get('dependencies', []),
                'validation_status': validation['status'],
                'validation_issues': validation.get('issues', []),
                'requires_gpu': requires_gpu,  # v3.2: GPU routing flag
                'status': 'pending',
                'attempts': 0
            }

            subtasks.append(subtask)
            self._register_subtask_in_document(subtask)

        # Save document
        self.master_document._save()

        print(f"\n  ✓ Decomposition complete: {len(subtasks)} subtasks created")
        gpu_count = sum(1 for s in subtasks if s.get('requires_gpu'))
        if gpu_count:
            print(f"    ({gpu_count} GPU, {len(subtasks) - gpu_count} CPU)")

        agent_logger.log_task_start(
            agent_name=self.agent_id,
            task_id="decomposition",
            description=f"Decomposed into {len(subtasks)} subtasks with validation ({gpu_count} GPU)",
            attempt=1
        )

        return subtasks

    def _register_subtask_in_document(self, subtask: Dict) -> None:
        """Register a subtask in the master document and status tracker."""
        self.master_document.add_step(
            step_id=subtask['id'],
            title=subtask['title'],
            description=subtask['description'],
            original_text=subtask['original_text'],
            expanded_plan=subtask['expanded_plan'],
            validation_status=subtask['validation_status'],
            packages=subtask['packages'],
            dependencies=subtask['dependencies'],
            code_hints=subtask.get('code_hints', [])
        )
        self.subtask_status[subtask['id']] = {'status': 'pending', 'attempts': 0}

    def _create_basic_subtask(
        self, step: Dict, step_id: str, index: int, context: Dict
    ) -> Dict[str, Any]:
        """Create a basic subtask without LLM expansion (used on timeout)."""
        step_text = step.get('full_text', step.get('title', ''))
        step_packages = context.get('packages', [])
        requires_gpu = detect_requires_gpu(step_packages, step_text)

        return {
            'id': step_id,
            'title': step.get('title', f'Step {index+1}'),
            'description': step_text,
            'original_text': step_text,
            'expanded_plan': step_text,
            'language': context.get('language', 'python'),
            'packages': step_packages,
            'imports_needed': [],
            'input_files': context.get('input_files', []),
            'output_files': context.get('output_files', []),
            'key_operations': [],
            'code_approach': '',
            'code_hints': step.get('code_hints', []),
            'success_criteria': 'Step completes without error',
            'dependencies': [],
            'validation_status': 'timeout_fallback',
            'validation_issues': ['Expansion skipped due to decomposition timeout'],
            'requires_gpu': requires_gpu,
            'status': 'pending',
            'attempts': 0
        }

    def _convert_steps_to_subtasks(self, steps: List[Dict]) -> List[Dict[str, Any]]:
        """Convert stored steps back to subtask format"""
        subtasks = []
        for step in steps:
            # v3.2: Detect GPU requirement from stored step data
            step_packages = step.get('packages', [])
            combined_text = (
                step.get('expanded_plan', '') + ' ' +
                step.get('original_text', '') + ' ' +
                step.get('description', '')
            )
            requires_gpu = detect_requires_gpu(step_packages, combined_text)

            subtasks.append({
                'id': step['id'],
                'title': step.get('title', step['id']),
                'description': step.get('expanded_plan', step.get('description', '')),
                'original_text': step.get('original_text', ''),
                'expanded_plan': step.get('expanded_plan', ''),
                'language': 'python',
                'packages': step_packages,
                'input_files': [],
                'output_files': step.get('output_files', []),
                'code_hints': step.get('code_hints', []),
                'success_criteria': step.get('success_criteria', ''),
                'dependencies': step.get('dependencies', []),
                'validation_status': step.get('validation_status', 'pending'),
                'requires_gpu': requires_gpu,
                'status': step.get('status', 'pending'),
                'attempts': step.get('attempts', 0)
            })
        return subtasks

    # =========================================================================
    # OTHER METHODS
    # =========================================================================

    def mark_subtask_complete(self, task_id: str, outputs: Dict = None, report: str = None):
        """Mark subtask as complete and update master document"""
        if task_id in self.subtask_status:
            self.subtask_status[task_id]['status'] = 'completed'
            self.subtask_status[task_id]['result'] = outputs

        if self.master_document:
            self.master_document.mark_complete(
                step_id=task_id,
                script_path=outputs.get('script_path') if outputs else None,
                output_files=outputs.get('output_files', []) if outputs else []
            )

    def mark_subtask_failed(self, task_id: str, error: str, details: str = None):
        """Mark subtask as failed and update master document"""
        if task_id in self.subtask_status:
            self.subtask_status[task_id]['status'] = 'failed'
            self.subtask_status[task_id]['error'] = error
            self.subtask_status[task_id]['attempts'] = self.subtask_status[task_id].get('attempts', 0) + 1

        if self.master_document:
            self.master_document.mark_failed(
                step_id=task_id,
                error_summary=error,
                attempts=self.subtask_status.get(task_id, {}).get('attempts', 1)
            )

    def review_failure(self, subtask: Dict, error: str = None, context: Dict = None) -> Dict[str, Any]:
        """
        Review failure and decide on action.

        v3.2: Uses invoke_with_timeout + parse_json_resilient.
        """
        failure_info = subtask.get('last_result', {})
        error = error or failure_info.get('error', 'Unknown error')

        context_summary = f"""
Subtask: {subtask.get('title', subtask.get('id', 'Unknown'))}
Original Intent: {subtask.get('original_text', subtask.get('description', ''))[:500]}
Expanded Plan: {subtask.get('expanded_plan', '')[:500]}
Attempts: {subtask.get('attempts', 1)}
Error: {error[:500]}
"""

        prompt = f"""A subtask has failed. Review and decide the next action.

{context_summary}

=== DECISION OPTIONS ===
1. RETRY - Try again with modifications (only if error seems recoverable)
2. SKIP - Mark as non-critical, continue pipeline
3. ESCALATE - Requires human intervention

Consider:
- Has context window been exhausted? -> SKIP or ESCALATE
- Is the error recoverable (missing package, wrong path)? -> RETRY
- Is it a fundamental issue (data doesn't exist, wrong approach)? -> SKIP or ESCALATE
- Has this been attempted multiple times already? -> SKIP or ESCALATE

Respond in JSON:
{{
    "decision": "RETRY|SKIP|ESCALATE",
    "reasoning": "Why this decision",
    "modification": "If RETRY, what specific changes to make"
}}"""

        try:
            response = invoke_with_timeout(
                self.llm, prompt, timeout_seconds=self.REVIEW_TIMEOUT
            )

            # v3.2: Multi-strategy JSON parsing
            decision = parse_json_resilient(response)
            if decision:
                agent_logger.log_reflection(
                    agent_name=self.agent_id,
                    task_id=subtask.get('id', 'unknown'),
                    reflection=f"Decision: {decision.get('decision')} - {decision.get('reasoning', '')[:100]}"
                )
                return decision

        except LLMTimeoutError:
            agent_logger.log_reflection(
                agent_name=self.agent_id,
                task_id=subtask.get('id', 'unknown'),
                reflection=f"Review LLM call timed out after {self.REVIEW_TIMEOUT}s"
            )
        except Exception as e:
            agent_logger.log_reflection(
                agent_name=self.agent_id,
                task_id=subtask.get('id', 'unknown'),
                reflection=f"Review error: {e}"
            )

        # Default based on attempts
        if subtask.get('attempts', 0) >= 3:
            return {"decision": "SKIP", "reasoning": "Max attempts reached"}

        return {"decision": "RETRY", "reasoning": "Default retry"}

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        if not self.master_document:
            return {'initialized': False}

        steps = self.master_document.steps
        return {
            'initialized': True,
            'total_steps': len(steps),
            'completed': len([s for s in steps.values() if s['status'] == 'completed']),
            'failed': len([s for s in steps.values() if s['status'] == 'failed']),
            'pending': len([s for s in steps.values() if s['status'] == 'pending']),
            'running': len([s for s in steps.values() if s['status'] == 'running']),
            'validated': len([s for s in steps.values() if s.get('validation_status') == 'validated']),
            'document_path': str(self.master_document.document_path)
        }
