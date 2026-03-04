"""
Master Agent v1.2.9 - Comprehensive Task Decomposition with Validation

The master agent follows a 3-phase approach:
1. EXTRACTION: Parse the detailed prompt and extract EVERY step exactly as written
2. EXPANSION: Expand each step into a comprehensive execution plan
3. VALIDATION: Verify each plan captures the original step's full intent

The master prompt serves as a living document that:
1. Contains all pipeline steps with status
2. Gets updated when subtasks complete (adds script paths)
3. Gets updated when subtasks fail (adds error summaries)
4. Maintains a comprehensive view of pipeline state

v1.2.9 Updates (Structured Prompt Parsing):
- NEW: _parse_structured_prompt() — deterministic parser for prompts using
  <<<STEP_N>>> / <<<END_STEP>>> delimiters and STEP_MANIFEST blocks.
  Replaces regex-based discovery with exact delimiter parsing. Step count
  is validated against the manifest. Dependencies are extracted directly
  from DEPENDS_ON fields — no LLM interpretation needed.
- NEW: _extract_header_blocks() — parses STEP_MANIFEST, GLOBAL_CONSTRAINTS,
  SHARED_CONSTANTS, and SHARED_ENV blocks from the prompt header.
- NEW: _parse_step_fields() — extracts labeled fields (GOAL, INPUT, OUTPUT,
  ENVIRONMENT, APPROACH, SUCCESS_CRITERIA, CONSTRAINTS, DEPENDS_ON) from
  within each <<<STEP_N>>> block.
- MODIFIED: decompose_task() — tries structured parsing first. Falls back to
  legacy extraction  only if <<<STEP_N>>> delimiters are not found.
- MODIFIED: _expand_step() — when structured fields are pre-parsed, the LLM
  prompt includes them directly instead of asking the LLM to infer packages,
  dependencies, and I/O files. Also adds Tier 2 complexity_warnings field.
- PRESERVED: All legacy methods (_extract_steps_from_prompt,
  _filter_executable_steps, _sanitize_dependencies) remain unchanged for
  backward compatibility with older prompts.

"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import re
import json
from datetime import datetime
from collections import deque

try:
    from langchain_ollama import OllamaLLM
except ImportError:
    from langchain_community.llms import Ollama as OllamaLLM

from utils.logging_config import agent_logger
from utils.context_manager import ContextManager
from utils.llm_invoke import invoke_resilient, LLMInvocationError
from utils.model_config import resolve_model, resolve_base_url


# =============================================================================
# JSON PARSING HELPERS
# =============================================================================

def parse_json_resilient(text: str) -> Optional[Dict]:
    """
    Multi-strategy JSON extraction from LLM output.

    Handles the common failure modes of smaller local models:
      1. Clean JSON inside a ```json ... ``` code fence
      2. Clean JSON inside a ``` ... ``` code fence (no language tag)
      3. First { ... } blob via greedy regex
      4. Truncated JSON (missing closing braces) — attempt brace balancing
      5. Trailing garbage after the closing } — strip and retry

    Returns the parsed dict, or None if all strategies fail.
    """
    if not text or not text.strip():
        return None

    # --- Strategy 1: ```json ... ``` code fence ---
    fence_match = re.search(r'```json\s*\n?([\s\S]*?)```', text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # --- Strategy 2: ``` ... ``` code fence (no language tag) ---
    fence_match2 = re.search(r'```\s*\n?([\s\S]*?)```', text)
    if fence_match2:
        try:
            return json.loads(fence_match2.group(1).strip())
        except json.JSONDecodeError:
            pass

    # --- Strategy 3: First { ... } blob ---
    brace_match = re.search(r'\{[\s\S]*\}', text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # --- Strategy 4: Truncated JSON — brace balancing ---
    first_brace = text.find('{')
    if first_brace >= 0:
        candidate = text[first_brace:]
        open_braces = 0
        for i, ch in enumerate(candidate):
            if ch == '{':
                open_braces += 1
            elif ch == '}':
                open_braces -= 1
            if open_braces == 0:
                try:
                    return json.loads(candidate[:i + 1])
                except json.JSONDecodeError:
                    break

        # Try adding missing closing braces
        if open_braces > 0:
            patched = candidate + ('}' * open_braces)
            try:
                return json.loads(patched)
            except json.JSONDecodeError:
                pass

    # --- Strategy 5: Strip trailing garbage ---
    if first_brace >= 0:
        candidate = text[first_brace:]
        # Find the last } and try parsing up to there
        last_brace = candidate.rfind('}')
        if last_brace >= 0:
            try:
                return json.loads(candidate[:last_brace + 1])
            except json.JSONDecodeError:
                pass

    return None


# =============================================================================
# GPU DETECTION HELPER
# =============================================================================

def detect_requires_gpu(packages: List[str], task_text: str) -> bool:
    """Detect if a task requires GPU based on packages and description.

    v3.2: Aligned with sub_agent.py GPU package detection (18 packages).
    """
    text_lower = (task_text + ' ' + ' '.join(packages)).lower()
    gpu_keywords = [
        'torch', 'pytorch', 'tensorflow', 'keras', 'jax',
        'rapids', 'cuml', 'cudf', 'cugraph', 'cupy',
        'gpu', 'cuda', 'scvi', 'scvi-tools', 'cellbender',
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
        """Save state to JSON and markdown"""
        self.last_updated = datetime.now()

        # JSON state
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

        # Markdown status
        markdown = self._generate_markdown()
        with open(self.markdown_path, 'w') as f:
            f.write(markdown)

    def _generate_markdown(self) -> str:
        """Generate human-readable markdown status report"""
        lines = [
            "# Pipeline Status Report",
            f"\nGenerated: {self.last_updated.strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n## Summary",
        ]

        total = len(self.steps)
        completed = sum(1 for s in self.steps.values() if s['status'] == 'completed')
        failed = sum(1 for s in self.steps.values() if s['status'] == 'failed')
        pending = sum(1 for s in self.steps.values() if s['status'] == 'pending')
        running = sum(1 for s in self.steps.values() if s['status'] == 'running')

        lines.append(f"\n| Status | Count |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total | {total} |")
        lines.append(f"| ✅ Completed | {completed} |")
        lines.append(f"| ❌ Failed | {failed} |")
        lines.append(f"| ⏳ Pending | {pending} |")
        lines.append(f"| 🔄 Running | {running} |")

        lines.append(f"\n## Steps\n")

        for step_id in self.step_order:
            step = self.steps.get(step_id, {})
            status_icon = {
                'completed': '✅', 'failed': '❌',
                'pending': '⏳', 'running': '🔄'
            }.get(step.get('status', 'pending'), '⏳')

            validation = step.get('validation_status', 'pending')

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
    Master agent v3.2.2 with comprehensive task decomposition.

    Key responsibilities:
    1. EXTRACT: Parse ALL steps from the detailed prompt
    2. FILTER: Remove non-executable documentation sections (v3.2.2)
    3. EXPAND: Create detailed execution plans for each step
    4. SANITIZE: Fix circular/invalid dependencies (v3.2.2)
    5. VALIDATE: Verify plans match original intent
    6. ASSIGN: Hand off validated plans to sub-agents
    7. TRACK: Maintain the master prompt document

    v3.2.1 — No artificial timeouts. The 3-day SLURM node window is the
    only hard limit. LLM calls use invoke_resilient() with exponential
    backoff retry to survive transient Ollama 500 errors.

    v3.2.2 — Fixes A+B: extraction filter + dependency sanitizer.
    Prevents the "0 completed, 0 failed, all blocked" deadlock caused by
    documentation sections creating circular dependency chains.

    Model resolution (no hardcoded model names):
      Constructor accepts ollama_model=None by default. Actual model is
      resolved via utils.model_config.resolve_model() with priority:
        1. Explicit parameter (from workflow/CLI)
        2. OLLAMA_MODEL environment variable (from RUN script)
        3. config.yaml → ollama.model
        4. Fallback constant in utils/model_config.py

    Token budget (must match sub_agent.py / config.yaml):
      - MAX_CONTEXT_TOKENS: 25,000  (leaves ~7K for system prompt + response)
    """

    # ── Token budget (must match sub_agent.py / config.yaml) ──────────────
    MAX_CONTEXT_TOKENS = 25_000

    def __init__(
        self,
        sandbox=None,
        ollama_model: str = None,
        ollama_base_url: str = None,
        **kwargs
    ):
        resolved_model = resolve_model(ollama_model)
        resolved_url = resolve_base_url(ollama_base_url)

        self.llm = OllamaLLM(model=resolved_model, base_url=resolved_url)
        self.ollama_base_url = resolved_url
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
    # PHASE 1: LEGACY EXTRACTION (preserved for backward compatibility)
    # =========================================================================

    def _extract_steps_from_prompt(self, prompt: str) -> List[Dict[str, Any]]:
        """
        Extract ALL steps from the prompt using multiple detection strategies.

        Legacy method preserved for prompts that do not use <<<STEP_N>>>
        delimiters. For new prompts, use the structured format instead.

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

            # Pattern 1: Numbered steps "1. ", "1) ", "Step 1:"
            numbered = re.match(
                r'^\s*(?:(?:Step\s+)?(\d+)[\.\):\-]\s+(.+)|(\d+)\.\s+(.+))',
                line, re.IGNORECASE
            )
            if numbered:
                step_match = True
                num = numbered.group(1) or numbered.group(3)
                step_title = (numbered.group(2) or numbered.group(4)).strip()

            # Pattern 2: Checkboxes "- [ ] ", "- [x] "
            if not step_match:
                checkbox = re.match(r'^\s*[\-\*]\s*\[[ xX]\]\s+(.+)', line)
                if checkbox:
                    step_match = True
                    step_title = checkbox.group(1).strip()

            # Pattern 3: Headers "## Step 1", "### Task:"
            if not step_match:
                header = re.match(r'^\s*#{2,4}\s+(?:Step\s+\d+[:\-\s]*)?(.+)', line)
                if header:
                    title = header.group(1).strip()
                    # Only treat as step if it looks like a task
                    if len(title) > 10 or any(kw in title.lower() for kw in
                        ['create', 'build', 'run', 'analyze', 'process', 'generate',
                         'setup', 'install', 'download', 'prepare', 'filter', 'merge',
                         'cluster', 'annotate', 'align', 'quality', 'normalize']):
                        step_match = True
                        step_title = title

            # Pattern 4: Keyword bullets "- First,", "- Next,", "- Finally,"
            if not step_match:
                keyword = re.match(
                    r'^\s*[\-\*]\s+((?:First|Next|Then|After|Finally|Subsequently|Lastly)[,:\s]+.+)',
                    line, re.IGNORECASE
                )
                if keyword:
                    step_match = True
                    step_title = keyword.group(1).strip()

            if step_match and step_title:
                # Save previous step
                if current_step is not None:
                    current_step['full_text'] = '\n'.join(current_content).strip()
                    current_step['code_hints'] = current_code_blocks.copy()
                    extracted.append(current_step)

                step_number += 1
                current_step = {
                    'step_number': step_number,
                    'title': step_title[:200],
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

    # =========================================================================
    # PHASE 1b: FILTER NON-EXECUTABLE STEPS (legacy)
    # =========================================================================

    def _filter_executable_steps(
        self, extracted_steps: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Classify extracted steps as executable or documentation.

        Legacy filter preserved for prompts that do not use <<<STEP_N>>>
        delimiters. For new prompts, the structured parser handles this
        by only extracting content within step delimiters.

        Classification criteria (step is DOCUMENTATION if ANY match):
          1. Has code_hints → always executable (strongest signal)
          2. Title matches known documentation patterns → filtered
          3. Title starts with bold markers → filtered
          4. Short content with no procedural title verbs → filtered
          5. Content is primarily a markdown table → filtered
        """
        # ── Documentation title patterns ──────────────────────────────────
        DOC_TITLE_PATTERNS = [
            r'(?i)^expected\s+output',
            r'(?i)^primary\s+data\s+file',
            r'(?i)^anndata\s+structure',
            r'(?i)^figures?\s*\(',
            r'(?i)^reports?$',
            r'(?i)^checkpoints?$',
            r'(?i)^input\s+files?$',
            r'(?i)^output\s+files?$',
            r'(?i)^dependenc',
            r'(?i)^success\s+criteria',
            r'(?i)^notes?$',
            r'(?i)^environment$',
            r'(?i)^prerequisites?$',
            r'(?i)^requirements?$',
            r'(?i)^overview$',
            r'(?i)^context$',
            r'(?i)^summary$',
        ]

        BOLD_NOTE_PATTERN = re.compile(r'^\*\*[A-Z]')

        PROCEDURAL_VERBS = {
            'run', 'execute', 'create', 'generate', 'compute', 'calculate',
            'load', 'save', 'write', 'read', 'filter', 'normalize',
            'cluster', 'annotate', 'align', 'process', 'validate',
            'download', 'install', 'build', 'train', 'predict',
            'merge', 'combine', 'split', 'transform', 'plot',
            'visualize', 'export', 'import', 'analyze', 'check',
            'verify', 'submit', 'setup', 'configure', 'prepare',
        }

        executable = []
        filtered = []

        for step in extracted_steps:
            title = step.get('title', '').strip()
            full_text = step.get('full_text', '')
            code_hints = step.get('code_hints', [])
            content_len = len(full_text)

            reason = None

            # ── Rule 1: Has code_hints → always executable ────────────────
            if code_hints:
                executable.append(step)
                continue

            # ── Rule 2: Title matches documentation patterns ──────────────
            for pattern in DOC_TITLE_PATTERNS:
                if re.search(pattern, title):
                    reason = f"title matches doc pattern: {pattern}"
                    break

            # ── Rule 3: Bold-prefixed notes ───────────────────────────────
            if not reason and BOLD_NOTE_PATTERN.match(title):
                reason = f"bold-prefixed note: {title[:50]}"

            # ── Rule 4: Very short content with no procedural title ───────
            if not reason and content_len < 200:
                title_words = set(re.findall(r'\b\w+\b', title.lower()))
                has_procedural = bool(title_words & PROCEDURAL_VERBS)
                if not has_procedural:
                    reason = f"short content ({content_len} chars) with no procedural title"

            # ── Rule 5: Content is primarily a markdown table ─────────────
            if not reason and content_len > 0:
                lines = [l.strip() for l in full_text.split('\n') if l.strip()]
                table_lines = sum(1 for l in lines if l.startswith('|'))
                if len(lines) > 0 and table_lines / len(lines) > 0.5:
                    text_lower = full_text.lower()
                    has_procedural = any(
                        re.search(rf'\b{v}\b', text_lower) for v in
                        ['run', 'execute', 'create', 'generate', 'compute',
                         'load', 'save', 'filter', 'normalize', 'cluster']
                    )
                    if not has_procedural:
                        reason = f"primarily table content ({table_lines}/{len(lines)} lines)"

            # ── Classify ──────────────────────────────────────────────────
            if reason:
                step['filter_reason'] = reason
                filtered.append(step)
            else:
                executable.append(step)

        # ── Log results ───────────────────────────────────────────────────
        if filtered:
            agent_logger.log_reflection(
                agent_name=self.agent_id,
                task_id="filter_steps",
                reflection=(
                    f"Filtered {len(filtered)}/{len(extracted_steps)} "
                    f"non-executable sections: "
                    f"{', '.join(s['title'][:40] for s in filtered[:5])}"
                    f"{'...' if len(filtered) > 5 else ''}"
                )
            )
            print(
                f"    Filtered {len(filtered)} documentation sections "
                f"(keeping {len(executable)} executable steps)"
            )

        # Re-number the executable steps sequentially
        for i, step in enumerate(executable):
            step['step_number'] = i + 1

        return executable, filtered

    def _extract_task_context(self, task: str) -> Dict[str, Any]:
        """Extract critical context from task description."""
        context = {
            "language": None,
            "packages": [],
            "reference_scripts": [],
            "input_files": [],
            "output_files": [],
            "completed_steps": [],
            "huggingface_repos": [],
        }

        # Detect language
        if 'python' in task.lower() or 'import ' in task or '.py' in task:
            context["language"] = "python"
        elif 'library(' in task or '.R' in task:
            context["language"] = "r"
        elif 'bash' in task.lower() or '#!/bin' in task:
            context["language"] = "bash"

        # Extract packages
        package_patterns = [
            r'(?:import|from)\s+(\w+)',
            r'library\((\w+)\)',
            r'pip install\s+([\w\-]+)',
            r'conda install\s+([\w\-]+)',
        ]
        for pattern in package_patterns:
            matches = re.findall(pattern, task)
            context["packages"].extend(matches)

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
    # STRUCTURED PROMPT PARSING (v1.2.9)
    # =========================================================================

    def _detect_structured_format(self, prompt: str) -> bool:
        """Check if the prompt uses <<<STEP_N>>> delimiters."""
        return bool(re.search(r'<<<STEP_\d+>>>', prompt))

    def _extract_header_blocks(self, prompt: str) -> Dict[str, Any]:
        """
        Extract labeled header blocks from the prompt.

        Parses:
          - STEP_MANIFEST    → list of {number, title}
          - GLOBAL_CONSTRAINTS → list of constraint strings
          - SHARED_CONSTANTS   → raw code string
          - SHARED_ENV:name    → dict of {name: yaml_string}

        All content before the first <<<STEP_N>>> delimiter is Header.
        """
        # Find where step definitions begin
        first_step = re.search(r'<<<STEP_\d+>>>', prompt)
        header_text = prompt[:first_step.start()] if first_step else prompt

        header = {
            'step_manifest': [],
            'global_constraints': [],
            'shared_constants': '',
            'shared_environments': {},
            'raw_header': header_text,
        }

        # ── STEP_MANIFEST ─────────────────────────────────────────────────
        manifest_match = re.search(
            r'```STEP_MANIFEST\s*\n(.*?)```',
            header_text, re.DOTALL
        )
        if manifest_match:
            manifest_text = manifest_match.group(1).strip()
            for line in manifest_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                m = re.match(r'STEP\s+(\d+)\s*:\s*(.+)', line, re.IGNORECASE)
                if m:
                    header['step_manifest'].append({
                        'number': int(m.group(1)),
                        'title': m.group(2).strip(),
                    })

        # ── GLOBAL_CONSTRAINTS ────────────────────────────────────────────
        constraints_match = re.search(
            r'```GLOBAL_CONSTRAINTS\s*\n(.*?)```',
            header_text, re.DOTALL
        )
        if constraints_match:
            for line in constraints_match.group(1).strip().split('\n'):
                line = line.strip()
                if line.startswith('- '):
                    line = line[2:].strip()
                if line:
                    header['global_constraints'].append(line)

        # ── SHARED_CONSTANTS ──────────────────────────────────────────────
        constants_match = re.search(
            r'```SHARED_CONSTANTS\s*\n(.*?)```',
            header_text, re.DOTALL
        )
        if constants_match:
            header['shared_constants'] = constants_match.group(1).strip()

        # ── SHARED_ENV:name ───────────────────────────────────────────────
        for env_match in re.finditer(
            r'```SHARED_ENV:(\S+)\s*\n(.*?)```',
            header_text, re.DOTALL
        ):
            env_name = env_match.group(1).strip()
            env_yaml = env_match.group(2).strip()
            header['shared_environments'][env_name] = env_yaml

        return header

    def _parse_step_fields(self, step_content: str) -> Dict[str, Any]:
        """
        Extract labeled fields from a single step block's content.

        Recognized fields: GOAL, DEPENDS_ON, INPUT, OUTPUT, ENVIRONMENT,
        APPROACH, SUCCESS_CRITERIA, CONSTRAINTS, REFERENCE_SCRIPTS, NOTES.

        Returns a dict with each field's content as a string. Missing fields
        get empty string/list defaults.
        """
        fields = {
            'goal': '',
            'depends_on': [],
            'input': '',
            'output': '',
            'environment': '',
            'approach': '',
            'success_criteria': '',
            'constraints': [],
            'reference_scripts': '',
            'notes': '',
        }

        # Field labels we recognize (order matters — longer matches first)
        FIELD_LABELS = [
            'SUCCESS_CRITERIA', 'REFERENCE_SCRIPTS', 'DEPENDS_ON',
            'ENVIRONMENT', 'CONSTRAINTS', 'APPROACH', 'OUTPUT',
            'INPUT', 'NOTES', 'GOAL',
        ]

        # Build regex that splits on field labels at the start of a line
        # Match: "FIELD_NAME:" at start of line (with optional whitespace)
        label_pattern = re.compile(
            r'^(' + '|'.join(FIELD_LABELS) + r')\s*:',
            re.MULTILINE | re.IGNORECASE
        )

        # Find all label positions
        label_positions = []
        for m in label_pattern.finditer(step_content):
            label_positions.append((m.start(), m.end(), m.group(1).upper()))

        # Extract content between labels
        for i, (start, end, label) in enumerate(label_positions):
            # Content runs from after the colon to the next label (or end)
            if i + 1 < len(label_positions):
                content = step_content[end:label_positions[i + 1][0]]
            else:
                content = step_content[end:]

            content = content.strip()
            field_key = label.lower()

            if field_key == 'depends_on':
                # Parse as list: [step_1, step_2] or []
                dep_match = re.search(r'\[(.*?)\]', content)
                if dep_match:
                    dep_str = dep_match.group(1).strip()
                    if dep_str:
                        fields['depends_on'] = [
                            d.strip() for d in dep_str.split(',') if d.strip()
                        ]
                    else:
                        fields['depends_on'] = []
                else:
                    fields['depends_on'] = []

            elif field_key == 'constraints':
                # Parse as list of constraint strings
                constraints = []
                for line in content.split('\n'):
                    line = line.strip()
                    if line.startswith('- '):
                        line = line[2:].strip()
                    if line:
                        constraints.append(line)
                fields['constraints'] = constraints

            elif field_key == 'environment':
                # Could be inline YAML or a USE_SHARED pointer
                fields['environment'] = content

            else:
                fields[field_key] = content

        # ── Extract code hints from APPROACH ──────────────────────────────
        # Pull out fenced code blocks from the approach field
        code_hints = []
        if fields['approach']:
            for code_match in re.finditer(
                r'```(?:\w+)?\s*\n(.*?)```',
                fields['approach'], re.DOTALL
            ):
                code_hints.append(code_match.group(1).strip())
        fields['code_hints'] = code_hints

        return fields

    def _parse_structured_prompt(
        self, prompt: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Parse a structured prompt with <<<STEP_N>>> delimiters.

        Returns:
            (steps, header) where steps is a list of step dicts ready for
            expansion, and header contains global context blocks.

        Raises ValueError if the prompt structure is invalid (step count
        mismatch, missing delimiters, etc.).
        """
        # ── Parse header ──────────────────────────────────────────────────
        header = self._extract_header_blocks(prompt)
        manifest = header['step_manifest']

        if not manifest:
            raise ValueError(
                "Structured prompt detected (has <<<STEP_N>>> delimiters) "
                "but no STEP_MANIFEST block found. Add a ```STEP_MANIFEST block "
                "listing all steps."
            )

        # ── Extract step blocks ───────────────────────────────────────────
        step_pattern = re.compile(
            r'<<<STEP_(\d+)>>>\s*\n(.*?)<<<END_STEP>>>',
            re.DOTALL
        )
        step_blocks = []
        for m in step_pattern.finditer(prompt):
            step_blocks.append({
                'number': int(m.group(1)),
                'raw_content': m.group(2),
            })

        # ── Validate step count ───────────────────────────────────────────
        manifest_numbers = {entry['number'] for entry in manifest}
        block_numbers = {block['number'] for block in step_blocks}

        if manifest_numbers != block_numbers:
            missing_blocks = manifest_numbers - block_numbers
            extra_blocks = block_numbers - manifest_numbers
            error_parts = []
            if missing_blocks:
                error_parts.append(
                    f"Manifest declares steps {sorted(missing_blocks)} "
                    f"but no <<<STEP_N>>> blocks found for them"
                )
            if extra_blocks:
                error_parts.append(
                    f"Found <<<STEP_N>>> blocks for {sorted(extra_blocks)} "
                    f"but they are not in the STEP_MANIFEST"
                )
            raise ValueError(
                f"Step count mismatch: {'; '.join(error_parts)}"
            )

        # ── Parse each step block ─────────────────────────────────────────
        # Sort by step number
        step_blocks.sort(key=lambda b: b['number'])
        manifest_by_num = {e['number']: e for e in manifest}

        extracted_steps = []
        for block in step_blocks:
            num = block['number']
            manifest_entry = manifest_by_num[num]
            fields = self._parse_step_fields(block['raw_content'])

            step = {
                'step_number': num,
                'title': manifest_entry['title'],
                'full_text': block['raw_content'].strip(),
                'code_hints': fields['code_hints'],
                'line_start': 0,
                # ── Structured fields (v3.3.0) ────────────────────────────
                'structured': True,
                'goal': fields['goal'],
                'depends_on': fields['depends_on'],
                'input_spec': fields['input'],
                'output_spec': fields['output'],
                'environment_spec': fields['environment'],
                'approach': fields['approach'],
                'success_criteria': fields['success_criteria'],
                'constraints': fields['constraints'],
                'reference_scripts': fields['reference_scripts'],
                'notes': fields['notes'],
            }

            extracted_steps.append(step)

        print(f"    ✓ Structured parsing: {len(extracted_steps)} steps from "
              f"manifest (validated against <<<STEP_N>>> blocks)")

        agent_logger.log_reflection(
            agent_name=self.agent_id,
            task_id="structured_parse",
            reflection=(
                f"Structured prompt parsed: {len(extracted_steps)} steps, "
                f"{len(header['global_constraints'])} global constraints, "
                f"{len(header['shared_environments'])} shared environments"
            )
        )

        return extracted_steps, header

    # =========================================================================
    # PHASE 2: DETAILED EXPANSION
    # =========================================================================

    def _expand_step(
        self, step: Dict[str, Any], context: Dict[str, Any], all_steps: List[Dict]
    ) -> Dict[str, Any]:
        """
        Expand a single extracted step into a detailed execution plan.

        v3.3.0: Two paths depending on prompt format:
        - Structured (step['structured'] == True): Pre-parsed fields are included
          directly. The LLM fleshes out the plan but does NOT discover deps/packages/IO.
          Also evaluates Tier 2 complexity warnings.
        - Legacy: Original behavior — LLM discovers everything from step text.

        v3.2.1: Uses invoke_resilient with exponential backoff retry.
        No artificial timeouts — will retry through transient Ollama 500 errors.
        """
        step_text = step.get('full_text', step.get('title', ''))
        code_hints = step.get('code_hints', [])
        is_structured = step.get('structured', False)

        # Build context about other steps for dependency awareness
        other_steps_summary = "\n".join([
            f"  - Step {s['step_number']}: {s['title'][:100]}"
            for s in all_steps if s['step_number'] != step.get('step_number')
        ])

        # v3.2.2: Include valid step IDs so LLM generates correct dep format
        valid_step_ids = [f"step_{s['step_number']}" for s in all_steps]

        # ── Build global context block ────────────────────────────────────
        global_context_block = ""
        if context.get('global_constraints'):
            constraints_str = '\n'.join(
                f"  - {c}" for c in context['global_constraints']
            )
            global_context_block += f"\nGlobal Constraints:\n{constraints_str}\n"
        if context.get('shared_constants'):
            global_context_block += (
                f"\nShared Constants (include at top of script):\n"
                f"```python\n{context['shared_constants']}\n```\n"
            )

        if is_structured:
            # ══════════════════════════════════════════════════════════════
            # STRUCTURED PATH (v3.3.0)
            # ══════════════════════════════════════════════════════════════
            # Pre-parsed fields are available — LLM expands the plan but
            # does not need to discover packages, deps, or I/O files.

            constraints_block = ""
            if step.get('constraints'):
                constraints_str = '\n'.join(
                    f"  - {c}" for c in step['constraints']
                )
                constraints_block = (
                    f"\nCONSTRAINTS (DO NOT violate these):\n{constraints_str}"
                )

            env_block = ""
            if step.get('environment_spec'):
                env_block = f"\nEnvironment Specification:\n{step['environment_spec']}"

            prompt = f"""You are expanding a pipeline step into a detailed execution plan.

=== STEP {step.get('step_number', '?')}: {step.get('title', 'Unknown')} ===

GOAL: {step.get('goal', 'See full description below')}

Full Description:
{step_text}
{constraints_block}

{"Code Hints from User:" if code_hints else ""}
{chr(10).join(['```' + ch + '```' for ch in code_hints[:3]]) if code_hints else ""}
{global_context_block}

Input Files: {step.get('input_spec', 'See description')}
Output Files: {step.get('output_spec', 'See description')}
{env_block}

Other Steps in Pipeline:
{other_steps_summary}

=== YOUR TASK ===
Create a COMPREHENSIVE execution plan that a developer could implement as
a SINGLE Python script without referring back to the original prompt.

The user has already specified dependencies, packages, and I/O files.
Your job is to:
1. Flesh out the implementation approach into a detailed plan
2. Identify any ADDITIONAL packages beyond what the user listed
3. Specify exact imports needed
4. Describe the code structure (functions, control flow)
5. Note any potential issues

ALSO evaluate whether this step might be too complex for a single script.
Set "complexity_warnings" to a list of concerns if ANY of these apply:
- The step requires more than one programming language
- Part of the step needs GPU/high-memory and part is lightweight
- The step produces multiple independent output artifacts
- The implementation would exceed ~500 lines of complex logic
If none apply, set "complexity_warnings" to an empty list.

=== OUTPUT FORMAT (JSON) ===
```json
{{
    "expanded_title": "Descriptive title for this step",
    "expanded_plan": "A detailed paragraph explaining exactly what this step does and how. Include specific function calls, data transformations, and expected outputs.",
    "packages": ["list", "of", "ALL", "packages", "needed"],
    "imports_needed": ["import pandas as pd", "import scanpy as sc"],
    "key_operations": ["Load data", "Filter cells", "Save results"],
    "code_approach": "Description of code structure — functions, loops, error handling",
    "estimated_complexity": "low|medium|high",
    "potential_issues": ["Issue 1", "Issue 2"],
    "complexity_warnings": []
}}
```

IMPORTANT:
- Your expanded_plan should be DETAILED and COMPREHENSIVE. Include ALL details
  from the user's original step description plus implementation specifics.
- Do NOT include "dependencies" in your output — they are already defined by the user.
- Do NOT include "input_files" or "output_files" — they are already defined.
- Do NOT include "success_criteria" — it is already defined.
- Focus on the HOW: what functions to call, what transformations to apply,
  what the script's control flow should look like."""

        else:
            # ══════════════════════════════════════════════════════════════
            # LEGACY PATH (v3.2.2)
            # ══════════════════════════════════════════════════════════════

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
{global_context_block}

Other Steps in Pipeline:
{other_steps_summary}

Valid step IDs for dependencies: {', '.join(valid_step_ids)}

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
    "code_approach": "Brief description of the code structure",
    "success_criteria": "File output.h5ad exists and contains filtered data with >1000 cells",
    "dependencies": ["step_1", "step_2"],
    "estimated_complexity": "low|medium|high",
    "potential_issues": ["Issue 1", "Issue 2"],
    "complexity_warnings": []
}}
```

IMPORTANT: Your expanded_plan should be DETAILED and COMPREHENSIVE. Do not summarize - expand and clarify. Include ALL details from the user's original step description.
- For "dependencies", ONLY use IDs from this list: {', '.join(valid_step_ids)}
- A step that has no prerequisites should have "dependencies": []
- Do NOT create circular dependencies
- Set "complexity_warnings" to a list of concerns if the step seems too complex
  for a single script (multiple languages, mixed compute profiles, multiple
  independent artifacts, >500 lines). Otherwise set to empty list."""

        # ── Common: invoke LLM and parse response ─────────────────────────
        try:
            response = invoke_resilient(
                self.llm,
                prompt,
                ollama_base_url=self.ollama_base_url,
                max_retries=20,
                initial_backoff=30.0,
            )

            # Multi-strategy JSON parsing
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

        except LLMInvocationError as e:
            agent_logger.log_reflection(
                agent_name=self.agent_id,
                task_id=f"expand_step_{step.get('step_number', '?')}",
                reflection=f"LLM invocation failed after all retries: {e}"
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
        common_words = {
            'the', 'a', 'an', 'is', 'are', 'to', 'for', 'of',
            'and', 'or', 'in', 'on', 'with', 'this', 'that'
        }
        original_words = set(re.findall(r'\b\w{4,}\b', original_text)) - common_words
        expanded_words = set(re.findall(r'\b\w{4,}\b', expanded_plan + ' ' + expanded_title))

        missing_key_terms = []
        for word in original_words:
            if len(word) > 5 and word not in expanded_words:
                if any(indicator in word for indicator in
                       ['file', 'data', 'output', 'input', 'cell', 'gene', 'cluster']):
                    missing_key_terms.append(word)

        if len(missing_key_terms) > 3:
            issues.append(
                f"Missing key terms from original: {', '.join(missing_key_terms[:5])}"
            )

        # Check 2: Code hints should be reflected
        code_hints = original_step.get('code_hints', [])
        if code_hints:
            code_hint_text = ' '.join(code_hints).lower()
            functions = re.findall(r'(\w+)\s*\(', code_hint_text)
            mentioned_functions = sum(1 for f in functions if f in expanded_plan)
            if functions and mentioned_functions < len(functions) / 2:
                issues.append(
                    f"Code hints not fully incorporated "
                    f"(found {mentioned_functions}/{len(functions)} functions)"
                )

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
    # PHASE 2b: DEPENDENCY SANITIZATION
    # =========================================================================

    def _sanitize_dependencies(
        self, subtasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Post-process LLM-generated dependencies to guarantee a valid DAG.

        v3.2.2 Fix B: Local LLMs frequently generate dependency values that
        are free-form text ("Validate Input Data"), non-existent IDs ("step_0"),
        self-references, or circular chains. This method enforces structural
        correctness AFTER the LLM expansion phase.

        Sanitization passes (applied in order):
          1. Normalize: convert any dep value to a valid step_N ID format
          2. Prune: remove refs to non-existent IDs and self-references
          3. Root guarantee: first step always gets deps=[]
          4. Cycle detection + breaking via Kahn's algorithm
          5. Fallback: if graph is still cyclic, impose sequential ordering

        Returns the subtask list with sanitized dependencies.
        """
        if not subtasks:
            return subtasks

        valid_ids = {s['id'] for s in subtasks}
        id_to_idx = {s['id']: i for i, s in enumerate(subtasks)}

        # ── Pass 1: Normalize dep values to step_N format ─────────────────
        title_to_id = {}
        for s in subtasks:
            title_lower = s.get('title', '').lower().strip()
            title_to_id[title_lower] = s['id']
            idx = id_to_idx[s['id']]
            title_to_id[f"step {idx + 1}"] = s['id']
            title_to_id[f"step_{idx + 1}"] = s['id']
            title_to_id[str(idx + 1)] = s['id']

        for s in subtasks:
            raw_deps = s.get('dependencies', [])
            normalized = []
            for dep in raw_deps:
                dep_str = str(dep).strip().lower()
                if dep_str in valid_ids:
                    normalized.append(dep_str)
                elif dep_str in title_to_id:
                    normalized.append(title_to_id[dep_str])
                else:
                    for title, sid in title_to_id.items():
                        if dep_str in title or title in dep_str:
                            normalized.append(sid)
                            break
            s['dependencies'] = normalized

        # ── Pass 2: Prune invalid refs and self-references ────────────────
        for s in subtasks:
            s['dependencies'] = [
                d for d in s['dependencies']
                if d in valid_ids and d != s['id']
            ]
            seen = set()
            deduped = []
            for d in s['dependencies']:
                if d not in seen:
                    seen.add(d)
                    deduped.append(d)
            s['dependencies'] = deduped

        # ── Pass 3: First step always has no dependencies ─────────────────
        subtasks[0]['dependencies'] = []

        # ── Pass 4+5: Cycle detection and breaking ────────────────────────
        subtasks = self._break_dependency_cycles(subtasks, valid_ids, id_to_idx)

        # ── Log results ───────────────────────────────────────────────────
        dep_summary = {s['id']: s['dependencies'] for s in subtasks}
        agent_logger.log_reflection(
            agent_name=self.agent_id,
            task_id="sanitize_deps",
            reflection=(
                f"Sanitized dependencies for {len(subtasks)} subtasks. "
                f"Dep graph: {json.dumps(dep_summary)}"
            )
        )

        return subtasks

    def _break_dependency_cycles(
        self,
        subtasks: List[Dict[str, Any]],
        valid_ids: set,
        id_to_idx: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """
        Detect and break cycles in the dependency graph using Kahn's algorithm.

        Iteratively removes back-edges (highest-numbered source node in cycle)
        until the graph is acyclic. Falls back to sequential ordering if
        cycles cannot be resolved within 50 iterations.
        """
        MAX_CYCLE_BREAK_ATTEMPTS = 50

        for attempt in range(MAX_CYCLE_BREAK_ATTEMPTS):
            # Build in-degree and adjacency
            in_degree = {s['id']: 0 for s in subtasks}
            dependents = {s['id']: [] for s in subtasks}

            for s in subtasks:
                for dep in s['dependencies']:
                    if dep in in_degree:
                        in_degree[s['id']] += 1
                        dependents[dep].append(s['id'])

            # Kahn's algorithm
            queue = deque([sid for sid, deg in in_degree.items() if deg == 0])
            sorted_order = []

            while queue:
                node = queue.popleft()
                sorted_order.append(node)
                for dependent in dependents[node]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

            if len(sorted_order) == len(subtasks):
                return subtasks  # Valid DAG

            # ── Cycle detected — find and break it ────────────────────────
            cycle_nodes = [
                sid for sid in in_degree if sid not in set(sorted_order)
            ]
            if not cycle_nodes:
                break

            cycle_nodes_sorted = sorted(
                cycle_nodes, key=lambda sid: id_to_idx.get(sid, 0), reverse=True
            )

            broken = False
            for node_id in cycle_nodes_sorted:
                node = next(s for s in subtasks if s['id'] == node_id)
                cycle_set = set(cycle_nodes)
                back_edges = [d for d in node['dependencies'] if d in cycle_set]
                if back_edges:
                    remove_dep = max(back_edges, key=lambda d: id_to_idx.get(d, 0))
                    node['dependencies'].remove(remove_dep)
                    agent_logger.log_reflection(
                        agent_name=self.agent_id,
                        task_id="break_cycle",
                        reflection=(
                            f"Broke cycle: removed {node_id} → {remove_dep} "
                            f"(attempt {attempt + 1})"
                        )
                    )
                    broken = True
                    break

            if not broken:
                node_id = cycle_nodes_sorted[0]
                node = next(s for s in subtasks if s['id'] == node_id)
                node['dependencies'] = []
                agent_logger.log_reflection(
                    agent_name=self.agent_id,
                    task_id="break_cycle",
                    reflection=f"Force-cleared all deps for {node_id} (attempt {attempt + 1})"
                )

        # ── Fallback: sequential ordering ─────────────────────────────────
        print("    ⚠ Could not resolve cycles — falling back to sequential ordering")
        agent_logger.log_reflection(
            agent_name=self.agent_id,
            task_id="break_cycle",
            reflection="Exhausted cycle-break attempts, falling back to sequential deps"
        )
        for i, s in enumerate(subtasks):
            if i == 0:
                s['dependencies'] = []
            else:
                s['dependencies'] = [subtasks[i - 1]['id']]

        return subtasks

    # =========================================================================
    # MAIN DECOMPOSITION METHOD
    # =========================================================================

    def decompose_task(
        self, main_task: str, context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Comprehensive task decomposition with extraction, expansion, and validation.

        This is the main entry point that orchestrates the multi-phase approach:
        1. Extract ALL steps from the detailed prompt
        1b. Filter non-executable documentation sections (v3.2.2 Fix A)
        2. Expand each step into a detailed execution plan
        2b. Sanitize dependencies to guarantee valid DAG (v3.2.2 Fix B)
        3. Validate each plan against the original

        v3.2.1: No artificial timeouts. Each LLM call uses invoke_resilient()
        which retries with exponential backoff through transient failures.
        The only hard limit is the 3-day SLURM node window.

        v3.2.2: Fixes A+B prevent the "all tasks blocked" deadlock by
        filtering documentation sections before expansion and sanitizing
        dependencies after expansion.
        """
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
                    reflection=(
                        f"Skipping {len(completed)} completed steps, "
                        f"{len(pending)} pending"
                    )
                )

                # Return pending steps for re-execution
                if pending:
                    return self._convert_steps_to_subtasks(pending)

        # Extract global context
        extracted_context = self._extract_task_context(main_task)
        if context:
            for key, value in context.items():
                if isinstance(value, list):
                    extracted_context[key] = list(
                        set(extracted_context.get(key, []) + value)
                    )
                elif value:
                    extracted_context[key] = value

        agent_logger.log_reflection(
            agent_name=self.agent_id,
            task_id="decompose",
            reflection=(
                f"Extracted context: {len(extracted_context.get('packages', []))} packages, "
                f"{len(extracted_context.get('input_files', []))} inputs"
            )
        )

        # =====================================================================
        # PHASE 1: EXTRACTION (v1.2.9: structured-first, legacy fallback)
        # =====================================================================
        print("\n  Phase 1: Extracting steps from prompt...")

        structured_header = None  # Will hold header blocks if structured

        if self._detect_structured_format(main_task):
            # ── v1.2.9: Structured prompt with <<<STEP_N>>> delimiters ────
            print("    Detected structured prompt format (<<<STEP_N>>> delimiters)")
            try:
                extracted_steps, structured_header = self._parse_structured_prompt(main_task)

                # Inject global constraints and shared constants into context
                if structured_header.get('global_constraints'):
                    extracted_context['global_constraints'] = structured_header['global_constraints']
                if structured_header.get('shared_constants'):
                    extracted_context['shared_constants'] = structured_header['shared_constants']
                if structured_header.get('shared_environments'):
                    extracted_context['shared_environments'] = structured_header['shared_environments']

                # Store for auditability
                self.master_document.extracted_steps = extracted_steps

            except ValueError as e:
                print(f"    ⚠ Structured parsing failed: {e}")
                print("    Falling back to legacy extraction...")
                agent_logger.log_reflection(
                    agent_name=self.agent_id,
                    task_id="extraction",
                    reflection=f"Structured parsing failed ({e}), using legacy"
                )
                # Fall through to legacy path below
                structured_header = None
                extracted_steps = None

        if structured_header is None:
            # ── Legacy extraction ─────────────────────────────────────────
            raw_extracted_steps = self._extract_steps_from_prompt(main_task)
            print(f"    Found {len(raw_extracted_steps)} raw steps (legacy extraction)")

            # Phase 1b: Filter non-executable sections
            extracted_steps, filtered_steps = self._filter_executable_steps(raw_extracted_steps)

            # Store for auditability
            self.master_document.extracted_steps = raw_extracted_steps

            agent_logger.log_reflection(
                agent_name=self.agent_id,
                task_id="extraction",
                reflection=(
                    f"Legacy extraction: {len(raw_extracted_steps)} raw steps, "
                    f"filtered to {len(extracted_steps)} executable "
                    f"({len(filtered_steps)} documentation sections removed)"
                )
            )
            print(f"    → {len(extracted_steps)} executable steps for expansion")

        # =====================================================================
        # PHASE 2: EXPANSION (no artificial timeout — retries handle failures)
        # =====================================================================
        print("\n  Phase 2: Expanding each step into detailed plans...")
        subtasks = []

        for i, step in enumerate(extracted_steps):
            print(
                f"    Expanding step {i+1}/{len(extracted_steps)}: "
                f"{step.get('title', 'Unknown')[:50]}..."
            )

            # Expand the step (invoke_resilient handles retries internally)
            expansion_result = self._expand_step(step, extracted_context, extracted_steps)
            expanded = expansion_result.get('expanded', {})

            # =================================================================
            # PHASE 3: VALIDATION
            # =================================================================
            validation = self._validate_expansion(step, expanded)

            if validation['status'] == 'needs_revision':
                print(
                    f"      ⚠️ Validation issues: "
                    f"{', '.join(validation['issues'][:2])}"
                )
                agent_logger.log_reflection(
                    agent_name=self.agent_id,
                    task_id=f"validate_step_{i+1}",
                    reflection=f"Validation issues: {validation['issues']}"
                )
            else:
                print(f"      ✓ Validated")

            # Create subtask with full context
            step_id = f"step_{i+1}"
            is_structured = step.get('structured', False)

            # Merge packages from expansion with context packages
            step_packages = expanded.get(
                'packages', extracted_context.get('packages', [])
            )

            # Auto-detect GPU requirement
            combined_text = (
                expanded.get('expanded_plan', '') + ' ' +
                step.get('full_text', '') + ' ' +
                expanded.get('code_approach', '')
            )
            requires_gpu = detect_requires_gpu(step_packages, combined_text)

            # ── v1.2.9: Structured fields override LLM-generated fields ──
            if is_structured:
                # Dependencies come from parsed DEPENDS_ON, not LLM
                step_deps = step.get('depends_on', [])
                # Input/output from parsed fields
                step_input = step.get('input_spec', '')
                step_output = step.get('output_spec', '')
                # Success criteria from parsed field
                step_success = step.get('success_criteria', '')
                # Constraints passed through for sub-agent
                step_constraints = step.get('constraints', [])
                # Environment spec passed through
                step_env_spec = step.get('environment_spec', '')
            else:
                # Legacy: LLM provides everything
                step_deps = expanded.get('dependencies', [])
                step_input = expanded.get('input_files', extracted_context.get('input_files', []))
                step_output = expanded.get('output_files', extracted_context.get('output_files', []))
                step_success = expanded.get('success_criteria', '')
                step_constraints = []
                step_env_spec = ''

            # ── Tier 2: Surface complexity warnings ───────────────────────
            complexity_warnings = expanded.get('complexity_warnings', [])
            if complexity_warnings:
                warnings_str = '; '.join(complexity_warnings)
                print(f"      ⚠ COMPLEXITY WARNING: {warnings_str}")
                print(f"        Consider splitting this step in the master prompt.")
                agent_logger.log_reflection(
                    agent_name=self.agent_id,
                    task_id=f"complexity_{step_id}",
                    reflection=(
                        f"Tier 2 complexity warning for {step_id}: {warnings_str}"
                    )
                )

            subtask = {
                'id': step_id,
                'title': expanded.get(
                    'expanded_title', step.get('title', f'Step {i+1}')
                ),
                'description': expanded.get(
                    'expanded_plan', step.get('full_text', '')
                ),
                'original_text': step.get('full_text', ''),
                'expanded_plan': expanded.get('expanded_plan', ''),
                'language': extracted_context.get('language', 'python'),
                'packages': step_packages,
                'imports_needed': expanded.get('imports_needed', []),
                'input_files': step_input,
                'output_files': step_output,
                'key_operations': expanded.get('key_operations', []),
                'code_approach': expanded.get('code_approach', ''),
                'code_hints': step.get('code_hints', []),
                'success_criteria': step_success,
                'dependencies': step_deps,
                'validation_status': validation['status'],
                'validation_issues': validation.get('issues', []),
                'requires_gpu': requires_gpu,
                'status': 'pending',
                'attempts': 0,
                # v1.2.9: New fields for structured prompts
                'constraints': step_constraints,
                'complexity_warnings': complexity_warnings,
                'environment_spec': step_env_spec,
            }

            subtasks.append(subtask)

        # =====================================================================
        # PHASE 2b: DEPENDENCY SANITIZATION
        # =====================================================================
        if structured_header is not None:
            # v3.3.0: Structured prompts — only validate, don't rewrite deps
            print("\n  Phase 2b: Validating declared dependencies...")
            valid_ids = {s['id'] for s in subtasks}
            for s in subtasks:
                invalid_deps = [d for d in s['dependencies'] if d not in valid_ids]
                if invalid_deps:
                    print(f"    ⚠ {s['id']}: removing invalid deps {invalid_deps}")
                    agent_logger.log_reflection(
                        agent_name=self.agent_id,
                        task_id="dep_validate",
                        reflection=(
                            f"Removed invalid deps from {s['id']}: {invalid_deps}"
                        )
                    )
                    s['dependencies'] = [d for d in s['dependencies'] if d in valid_ids]
        else:
            # Legacy: full sanitization with cycle breaking
            print("\n  Phase 2b: Sanitizing dependencies...")
            subtasks = self._sanitize_dependencies(subtasks)

        # =====================================================================
        # Register in document and save (AFTER sanitization)
        # =====================================================================
        for subtask in subtasks:
            self._register_subtask_in_document(subtask)

        # Save document
        self.master_document._save()

        print(f"\n  ✓ Decomposition complete: {len(subtasks)} subtasks created")
        gpu_count = sum(1 for s in subtasks if s.get('requires_gpu'))
        if gpu_count:
            print(f"    ({gpu_count} GPU, {len(subtasks) - gpu_count} CPU)")

        # Print dependency chain for visibility
        for s in subtasks:
            deps_str = ', '.join(s['dependencies']) if s['dependencies'] else '(none)'
            print(f"    {s['id']}: {s['title'][:40]}  deps=[{deps_str}]")

        agent_logger.log_task_start(
            agent_name=self.agent_id,
            task_id="decomposition",
            description=(
                f"Decomposed into {len(subtasks)} subtasks with validation "
                f"({gpu_count} GPU)"
            ),
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
        """Create a basic subtask without LLM expansion (used as fallback)."""
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
            'validation_status': 'basic_fallback',
            'validation_issues': ['LLM expansion failed, using original text'],
            'requires_gpu': requires_gpu,
            'status': 'pending',
            'attempts': 0
        }

    def _convert_steps_to_subtasks(self, steps: List[Dict]) -> List[Dict[str, Any]]:
        """Convert stored steps back to subtask format"""
        subtasks = []
        for step in steps:
            # Detect GPU requirement from stored step data
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
                'description': step.get(
                    'expanded_plan', step.get('description', '')
                ),
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

    def mark_subtask_complete(
        self, task_id: str, outputs: Dict = None, report: str = None
    ):
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

    def mark_subtask_failed(
        self, task_id: str, error: str, details: str = None
    ):
        """Mark subtask as failed and update master document"""
        if task_id in self.subtask_status:
            self.subtask_status[task_id]['status'] = 'failed'
            self.subtask_status[task_id]['error'] = error
            self.subtask_status[task_id]['attempts'] = (
                self.subtask_status[task_id].get('attempts', 0) + 1
            )

        if self.master_document:
            self.master_document.mark_failed(
                step_id=task_id,
                error_summary=error,
                attempts=self.subtask_status.get(task_id, {}).get('attempts', 1)
            )

    def review_failure(
        self, subtask: Dict, error: str = None, context: Dict = None
    ) -> Dict[str, Any]:
        """
        Review failure and decide on action.

        v3.2.1: Uses invoke_resilient with exponential backoff retry.
        No artificial timeouts.
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
            response = invoke_resilient(
                self.llm,
                prompt,
                ollama_base_url=self.ollama_base_url,
                max_retries=10,           # Fewer retries for review (less critical)
                initial_backoff=15.0,
            )

            # Multi-strategy JSON parsing
            decision = parse_json_resilient(response)
            if decision:
                agent_logger.log_reflection(
                    agent_name=self.agent_id,
                    task_id=subtask.get('id', 'unknown'),
                    reflection=(
                        f"Decision: {decision.get('decision')} - "
                        f"{decision.get('reasoning', '')[:100]}"
                    )
                )
                return decision

        except LLMInvocationError as e:
            agent_logger.log_reflection(
                agent_name=self.agent_id,
                task_id=subtask.get('id', 'unknown'),
                reflection=f"Review LLM call failed after all retries: {e}"
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
            'completed': len([
                s for s in steps.values() if s['status'] == 'completed'
            ]),
            'failed': len([
                s for s in steps.values() if s['status'] == 'failed'
            ]),
            'pending': len([
                s for s in steps.values() if s['status'] == 'pending'
            ]),
            'running': len([
                s for s in steps.values() if s['status'] == 'running'
            ]),
            'validated': len([
                s for s in steps.values()
                if s.get('validation_status') == 'validated'
            ]),
            'document_path': str(self.master_document.document_path)
        }
