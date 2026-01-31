"""
Master Agent with Living Document Management

The master prompt serves as a living document that:
1. Contains all pipeline steps with status
2. Gets updated when subtasks complete (adds script paths)
3. Gets updated when subtasks fail (adds error summaries)
4. Maintains a comprehensive view of pipeline state

Token-based context management for the master's coordination window.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import re
import json
from datetime import datetime

try:
    from langchain_ollama import OllamaLLM
except ImportError:
    from langchain_community.llms import Ollama as OllamaLLM

from utils.logging_config import agent_logger
from utils.context_manager import ContextManager


class MasterPromptDocument:
    """
    Living document that tracks pipeline state.
    
    Structure:
    - Pipeline overview
    - Step definitions with status
    - Completed steps with script paths and outputs
    - Failed steps with error summaries
    - Dependencies and relationships
    """
    
    def __init__(self, original_prompt: str, project_dir: Path):
        self.original_prompt = original_prompt
        self.project_dir = project_dir
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        
        # Step tracking
        self.steps: Dict[str, Dict] = {}  # step_id -> step info
        self.step_order: List[str] = []   # Ordered list of step IDs
        
        # Parse original prompt to extract steps
        self._parse_original_prompt()
        
        # Document path
        self.document_path = project_dir / "reports" / "master_prompt_state.json"
        self.markdown_path = project_dir / "reports" / "pipeline_status.md"
    
    def _parse_original_prompt(self):
        """Extract steps from original prompt"""
        # This is a simplified parser - could be more sophisticated
        lines = self.original_prompt.split('\n')
        step_count = 0
        
        for line in lines:
            # Look for numbered steps or task markers
            step_match = re.match(r'^\s*(\d+)\.\s*(.+)', line)
            if step_match:
                step_count += 1
                step_id = f"step_{step_count}"
                self.steps[step_id] = {
                    'id': step_id,
                    'title': step_match.group(2).strip()[:100],
                    'description': step_match.group(2).strip(),
                    'status': 'pending',
                    'attempts': 0,
                    'script_path': None,
                    'output_files': [],
                    'error_summary': None,
                    'created_at': datetime.now().isoformat()
                }
                self.step_order.append(step_id)
            
            # Look for checkbox items
            checkbox_match = re.match(r'^\s*[-*]\s*\[([x\s])\]\s*(.+)', line, re.IGNORECASE)
            if checkbox_match:
                step_count += 1
                step_id = f"step_{step_count}"
                is_complete = checkbox_match.group(1).lower() == 'x'
                self.steps[step_id] = {
                    'id': step_id,
                    'title': checkbox_match.group(2).strip()[:100],
                    'description': checkbox_match.group(2).strip(),
                    'status': 'completed' if is_complete else 'pending',
                    'attempts': 0,
                    'script_path': None,
                    'output_files': [],
                    'error_summary': None,
                    'created_at': datetime.now().isoformat()
                }
                self.step_order.append(step_id)
    
    def add_step(self, step_id: str, title: str, description: str, **kwargs):
        """Add or update a step"""
        if step_id not in self.steps:
            self.step_order.append(step_id)
        
        self.steps[step_id] = {
            'id': step_id,
            'title': title,
            'description': description,
            'status': kwargs.get('status', 'pending'),
            'attempts': kwargs.get('attempts', 0),
            'script_path': kwargs.get('script_path'),
            'output_files': kwargs.get('output_files', []),
            'conda_env_yaml': kwargs.get('conda_env_yaml'),
            'error_summary': kwargs.get('error_summary'),
            'dependencies': kwargs.get('dependencies', []),
            'packages': kwargs.get('packages', []),
            'created_at': datetime.now().isoformat()
        }
        self.last_updated = datetime.now()
    
    def mark_complete(self, step_id: str, script_path: str, output_files: List[str], **kwargs):
        """Mark a step as complete with its outputs"""
        if step_id in self.steps:
            self.steps[step_id]['status'] = 'completed'
            self.steps[step_id]['script_path'] = script_path
            self.steps[step_id]['output_files'] = output_files
            self.steps[step_id]['completed_at'] = datetime.now().isoformat()
            self.steps[step_id].update(kwargs)
            self.last_updated = datetime.now()
            self._save()
    
    def mark_failed(self, step_id: str, error_summary: str, attempts: int, **kwargs):
        """Mark a step as failed with error information"""
        if step_id in self.steps:
            self.steps[step_id]['status'] = 'failed'
            self.steps[step_id]['error_summary'] = error_summary
            self.steps[step_id]['attempts'] = attempts
            self.steps[step_id]['failed_at'] = datetime.now().isoformat()
            self.steps[step_id].update(kwargs)
            self.last_updated = datetime.now()
            self._save()
    
    def mark_running(self, step_id: str):
        """Mark a step as currently running"""
        if step_id in self.steps:
            self.steps[step_id]['status'] = 'running'
            self.steps[step_id]['started_at'] = datetime.now().isoformat()
            self.last_updated = datetime.now()
    
    def get_pending_steps(self) -> List[Dict]:
        """Get all pending steps"""
        return [self.steps[sid] for sid in self.step_order 
                if self.steps[sid]['status'] == 'pending']
    
    def get_ready_steps(self) -> List[Dict]:
        """Get steps that are ready to run (dependencies satisfied)"""
        ready = []
        completed_ids = {sid for sid in self.step_order 
                        if self.steps[sid]['status'] == 'completed'}
        
        for step_id in self.step_order:
            step = self.steps[step_id]
            if step['status'] != 'pending':
                continue
            
            deps = set(step.get('dependencies', []))
            if deps.issubset(completed_ids):
                ready.append(step)
        
        return ready
    
    def generate_status_markdown(self) -> str:
        """Generate markdown summary of pipeline status"""
        completed = [s for s in self.steps.values() if s['status'] == 'completed']
        failed = [s for s in self.steps.values() if s['status'] == 'failed']
        pending = [s for s in self.steps.values() if s['status'] == 'pending']
        running = [s for s in self.steps.values() if s['status'] == 'running']
        
        md = f"""# Pipeline Status

**Last Updated**: {self.last_updated.isoformat()}
**Total Steps**: {len(self.steps)}

## Summary
- ✅ Completed: {len(completed)}
- ❌ Failed: {len(failed)}
- ⏳ Pending: {len(pending)}
- 🔄 Running: {len(running)}

---

## Completed Steps

"""
        for step in completed:
            md += f"""### ✅ {step['title']}
- **Script**: `{step.get('script_path', 'N/A')}`
- **Outputs**: {', '.join(step.get('output_files', [])) or 'None recorded'}
- **Completed**: {step.get('completed_at', 'Unknown')}

"""
        
        if failed:
            md += """---

## Failed Steps

"""
            for step in failed:
                md += f"""### ❌ {step['title']}
- **Attempts**: {step.get('attempts', 0)}
- **Error**: {step.get('error_summary', 'No details')[:200]}
- **Last Script**: `{step.get('script_path', 'N/A')}`

"""
        
        if pending:
            md += """---

## Pending Steps

"""
            for step in pending:
                deps = step.get('dependencies', [])
                md += f"- [ ] {step['title']}"
                if deps:
                    md += f" (depends on: {', '.join(deps)})"
                md += "\n"
        
        return md
    
    def _save(self):
        """Save state to files"""
        self.document_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save JSON state
        state = {
            'original_prompt': self.original_prompt[:5000],
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'steps': self.steps,
            'step_order': self.step_order
        }
        with open(self.document_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        # Save markdown
        md = self.generate_status_markdown()
        self.markdown_path.write_text(md)
    
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
            doc.steps = state['steps']
            doc.step_order = state['step_order']
            doc.created_at = datetime.fromisoformat(state['created_at'])
            doc.last_updated = datetime.fromisoformat(state['last_updated'])
            return doc
        except Exception:
            return None
    
    def to_context_string(self, max_tokens: int = 10000) -> str:
        """Generate string representation for LLM context"""
        completed = [s for s in self.steps.values() if s['status'] == 'completed']
        failed = [s for s in self.steps.values() if s['status'] == 'failed']
        pending = [s for s in self.steps.values() if s['status'] == 'pending']
        
        parts = [
            f"Pipeline Status: {len(completed)} completed, {len(failed)} failed, {len(pending)} pending",
            "",
            "Completed with scripts:"
        ]
        
        for s in completed[:10]:  # Limit for context
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


class MasterAgent:
    """
    Master agent that coordinates task decomposition and manages the living document.
    
    Key responsibilities:
    1. Decompose tasks into subtasks with full context preservation
    2. Maintain the master prompt document
    3. Review failures and decide next actions
    4. Generate final reports
    """
    
    # Context limits for master coordination
    MAX_CONTEXT_TOKENS = 70_000
    
    def __init__(
        self,
        sandbox=None,
        ollama_model: str = "llama3.1:70b",
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
        # Try to load existing
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
    
    def decompose_task(self, main_task: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Break down task into subtasks while preserving all context.
        Updates the master document with decomposed steps.
        """
        # Initialize document
        self.initialize_document(main_task)
        
        # Extract context from the task
        extracted_context = self._extract_task_context(main_task)
        
        # Check for already completed steps
        if self.master_document.steps:
            completed = [s for s in self.master_document.steps.values() 
                        if s['status'] == 'completed']
            if completed:
                agent_logger.log_reflection(
                    agent_name=self.agent_id,
                    task_id="decompose",
                    reflection=f"Skipping {len(completed)} already completed steps"
                )
        
        context_str = ""
        if context:
            context_str = f"\nAdditional Context:\n{json.dumps(context, indent=2, default=str)}"
        
        # Build decomposition prompt
        prompt = f"""You are a master coordinator for a computational biology pipeline.
Break down this task into specific, executable subtasks.

=== MAIN TASK ===
{main_task}
{context_str}

=== CURRENT STATUS ===
{self.master_document.to_context_string(5000) if self.master_document else 'No existing steps'}

=== EXTRACTED CONTEXT (PRESERVE EXACTLY) ===
- Language: {extracted_context.get('language', 'python')}
- Packages: {', '.join(extracted_context.get('packages', []))}
- Reference Scripts: {', '.join(extracted_context.get('reference_scripts', []))}
- Input Files: {', '.join(extracted_context.get('input_files', []))}
- Output Files: {', '.join(extracted_context.get('output_files', []))}

=== SUBTASK REQUIREMENTS ===
Each subtask should:
1. Be independently executable as a script
2. Have clear input and output files
3. Use the EXACT packages specified (no substitutions)
4. Include success criteria (output files that should exist)
5. Note dependencies on other subtasks

=== OUTPUT FORMAT (JSON) ===
Return a JSON array:
```json
[
  {{
    "id": "subtask_1",
    "title": "Brief title",
    "description": "What this subtask does",
    "language": "python",
    "packages": ["scanpy", "pandas"],
    "input_files": ["data/input.h5ad"],
    "output_files": ["data/outputs/result.h5ad"],
    "reference_scripts": [],
    "dependencies": [],
    "success_criteria": "File data/outputs/result.h5ad exists"
  }}
]
```

Return ONLY the JSON array."""

        response = self.llm.invoke(prompt)
        
        # Parse response
        subtasks = self._parse_subtasks_json(response, extracted_context)
        
        # Update master document with new subtasks
        for subtask in subtasks:
            self.master_document.add_step(
                step_id=subtask['id'],
                title=subtask.get('title', subtask['id']),
                description=subtask.get('description', ''),
                packages=subtask.get('packages', []),
                dependencies=subtask.get('dependencies', [])
            )
            self.subtask_status[subtask['id']] = {'status': 'pending', 'attempts': 0}
        
        # Save document
        self.master_document._save()
        
        agent_logger.log_task_start(
            agent_name=self.agent_id,
            task_id="decomposition",
            description=f"Decomposed into {len(subtasks)} subtasks",
            attempt=1
        )
        
        return subtasks
    
    def _extract_task_context(self, task: str) -> Dict[str, Any]:
        """Extract critical context from task description"""
        context = {
            "language": None,
            "packages": [],
            "reference_scripts": [],
            "input_files": [],
            "output_files": [],
            "completed_steps": [],
            "huggingface_repos": []
        }
        
        # Detect language
        task_lower = task.lower()
        python_indicators = ['python', 'scanpy', 'squidpy', 'anndata', 'pandas', '.py', 'popv', 'h5ad']
        r_indicators = ['seurat', 'singlecell', 'bioconductor', '.R']
        
        python_score = sum(1 for ind in python_indicators if ind in task_lower)
        r_score = sum(1 for ind in r_indicators if ind in task_lower)
        
        context["language"] = "python" if python_score >= r_score else "r"
        
        # Extract packages
        python_packages = [
            'scanpy', 'squidpy', 'anndata', 'pandas', 'numpy', 'scipy',
            'popv', 'popV', 'scvi', 'scvi-tools', 'cellxgene', 'leidenalg',
            'matplotlib', 'seaborn', 'celltypist', 'decoupler'
        ]
        for pkg in python_packages:
            if pkg.lower() in task_lower:
                pattern = re.compile(re.escape(pkg), re.IGNORECASE)
                matches = pattern.findall(task)
                if matches:
                    context["packages"].append(matches[0])
        
        # Extract file paths
        file_patterns = [
            (r'[`"]?([^\s`"]*\.h5ad)[`"]?', 'h5ad'),
            (r'[`"]?([^\s`"]*\.csv)[`"]?', 'csv'),
            (r'[`"]?(data/[^\s`"]+)[`"]?', 'data'),
        ]
        for pattern, _ in file_patterns:
            matches = re.findall(pattern, task)
            for match in matches:
                if 'input' in match.lower() or 'processed' in match.lower():
                    context["input_files"].append(match)
                elif 'output' in match.lower():
                    context["output_files"].append(match)
        
        # Extract reference scripts
        script_pattern = r'scripts?/[^\s`"]+\.py'
        context["reference_scripts"] = re.findall(script_pattern, task)
        
        # Extract HuggingFace repos
        hf_pattern = r'huggingface_repo\s*=\s*["\']([^"\']+)["\']'
        context["huggingface_repos"] = re.findall(hf_pattern, task)
        
        # Extract completed steps
        completed_pattern = r'✅\s*(?:COMPLETED:?)?\s*([^\n]+)'
        context["completed_steps"] = re.findall(completed_pattern, task)
        
        # Deduplicate
        for key in context:
            if isinstance(context[key], list):
                context[key] = list(dict.fromkeys(context[key]))
        
        return context
    
    def _parse_subtasks_json(self, response: str, extracted_context: Dict) -> List[Dict[str, Any]]:
        """Parse JSON response into subtasks"""
        try:
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                subtasks = json.loads(json_match.group())
                
                for i, subtask in enumerate(subtasks):
                    subtask.setdefault('id', f'subtask_{i+1}')
                    subtask.setdefault('status', 'pending')
                    subtask.setdefault('attempts', 0)
                    subtask.setdefault('dependencies', [])
                    
                    # Inherit context if not specified
                    if not subtask.get('language'):
                        subtask['language'] = extracted_context.get('language', 'python')
                    if not subtask.get('packages'):
                        subtask['packages'] = extracted_context.get('packages', [])
                    if not subtask.get('reference_scripts'):
                        subtask['reference_scripts'] = extracted_context.get('reference_scripts', [])
                    if extracted_context.get('huggingface_repos'):
                        subtask['huggingface_repos'] = extracted_context['huggingface_repos']
                    
                    # Sync packages
                    subtask['required_packages'] = subtask.get('packages', [])
                
                return subtasks
        except json.JSONDecodeError:
            pass
        
        # Fallback: create single subtask from response
        return [{
            'id': 'subtask_1',
            'title': 'Main task',
            'description': response[:500],
            'language': extracted_context.get('language', 'python'),
            'packages': extracted_context.get('packages', []),
            'reference_scripts': extracted_context.get('reference_scripts', []),
            'input_files': extracted_context.get('input_files', []),
            'output_files': extracted_context.get('output_files', []),
            'status': 'pending',
            'attempts': 0,
            'dependencies': []
        }]
    
    def mark_subtask_complete(self, task_id: str, result: Dict = None):
        """Mark subtask as complete and update master document"""
        if task_id in self.subtask_status:
            self.subtask_status[task_id]['status'] = 'completed'
            self.subtask_status[task_id]['result'] = result
        
        if self.master_document:
            self.master_document.mark_complete(
                step_id=task_id,
                script_path=result.get('script_path') if result else None,
                output_files=result.get('output_files', []) if result else []
            )
    
    def mark_subtask_failed(self, task_id: str, result: Dict = None):
        """Mark subtask as failed and update master document"""
        if task_id in self.subtask_status:
            self.subtask_status[task_id]['status'] = 'failed'
            self.subtask_status[task_id]['result'] = result
            self.subtask_status[task_id]['attempts'] = self.subtask_status[task_id].get('attempts', 0) + 1
        
        if self.master_document:
            error_summary = result.get('error', 'Unknown error') if result else 'Unknown error'
            attempts = result.get('iterations', 1) if result else 1
            self.master_document.mark_failed(
                step_id=task_id,
                error_summary=error_summary,
                attempts=attempts,
                script_path=result.get('script_path') if result else None
            )
    
    def review_failure(self, subtask: Dict, failure_info: Dict) -> Dict[str, Any]:
        """Review failure and decide on action"""
        # Build context
        context_summary = f"""
Subtask: {subtask.get('description', subtask.get('title', 'Unknown'))}
Language: {subtask.get('language', 'python')}
Packages: {', '.join(subtask.get('packages', []))}
Attempts: {failure_info.get('iterations', 1)}
Error: {failure_info.get('error', 'Unknown')}
"""
        
        prompt = f"""A subtask has failed. Decide the next action.

{context_summary}

Previous script (if any): {failure_info.get('script_path', 'None')}

Options:
1. RETRY - Try again with modified approach (same tools)
2. SKIP - Mark as non-critical, continue pipeline
3. ESCALATE - Requires human intervention

Consider:
- Has context window been exhausted? -> SKIP or ESCALATE
- Is the error recoverable (missing package, wrong path)? -> RETRY
- Is it a fundamental issue (data doesn't exist)? -> SKIP or ESCALATE

Respond in JSON:
{{
    "decision": "RETRY|SKIP|ESCALATE",
    "reasoning": "Why this decision",
    "modification": "If RETRY, what to change"
}}"""

        response = self.llm.invoke(prompt)
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
                agent_logger.log_reflection(
                    agent_name=self.agent_id,
                    task_id=subtask.get('id', 'unknown'),
                    reflection=f"Decision: {decision.get('decision')} - {decision.get('reasoning')}"
                )
                return decision
        except:
            pass
        
        # Default to SKIP if context exhausted
        if failure_info.get('context_status', {}).get('remaining_tokens', 10000) < 5000:
            return {
                "decision": "SKIP",
                "reasoning": "Context window exhausted"
            }
        
        return {
            "decision": "RETRY",
            "reasoning": "Default retry"
        }
    
    def generate_final_report(self, main_task: str, subtask_results: List[Dict]) -> str:
        """Generate final report using master document"""
        if self.master_document:
            self.master_document._save()
            md_report = self.master_document.generate_status_markdown()
        else:
            md_report = "No master document available"
        
        # Add summary
        completed = [r for r in subtask_results if r.get('success')]
        failed = [r for r in subtask_results if not r.get('success')]
        
        summary = f"""# Pipeline Execution Report

**Task**: {main_task[:200]}...
**Completed**: {len(completed)} subtasks
**Failed**: {len(failed)} subtasks
**Timestamp**: {datetime.now().isoformat()}

## Scripts Generated

"""
        for r in completed:
            if r.get('script_path'):
                summary += f"- `{r['script_path']}`\n"
        
        summary += f"\n{md_report}"
        
        # Save report
        report_path = self.project_dir / "reports" / "final_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(summary)
        
        return summary
    
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
            'document_path': str(self.master_document.document_path)
        }
