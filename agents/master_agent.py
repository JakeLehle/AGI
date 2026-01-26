"""
Master agent that decomposes high-level tasks into subtasks,
coordinates sub-agents, and synthesizes final reports.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import json
import re

# Use updated import to avoid deprecation warning
try:
    from langchain_ollama import OllamaLLM
except ImportError:
    from langchain_community.llms import Ollama as OllamaLLM

from utils.logging_config import agent_logger
from tools.sandbox import Sandbox


class MasterAgent:
    """
    Coordinates task decomposition, sub-agent assignment, and report synthesis.
    Handles structured prompts with input files and expected outputs.
    """
    
    def __init__(
        self,
        sandbox: Sandbox,
        ollama_model: str = "llama3.1:70b",
        ollama_base_url: str = "http://127.0.0.1:11434"
    ):
        """
        Initialize master agent.
        
        Args:
            sandbox: Sandbox instance for file operations
            ollama_model: Ollama model to use
            ollama_base_url: Ollama server URL
        """
        self.sandbox = sandbox
        self.agent_id = "master"
        self.llm = OllamaLLM(
            model=ollama_model,
            base_url=ollama_base_url
        )
        
        # Track overall progress
        self.subtasks = []
        self.completed_reports = []
        self.task_dependencies = {}
    
    def decompose_task(
        self,
        main_task: str,
        context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Break down a high-level task into specific subtasks.
        
        Args:
            main_task: The main objective
            context: Additional context including:
                - input_files: List of input file paths
                - expected_outputs: List of expected output file paths
                - notes: Additional constraints or requirements
        
        Returns:
            List of subtask dictionaries with dependencies
        """
        context = context or {}
        
        # Get input files and expected outputs
        input_files = context.get("input_files", [])
        expected_outputs = context.get("expected_outputs", [])
        
        # Build context string
        context_str = ""
        
        if input_files:
            context_str += "\nInput files available:\n"
            for f in input_files:
                # Check if file exists and get basic info
                file_path = self.sandbox.project_dir / f
                if file_path.exists():
                    size = file_path.stat().st_size
                    context_str += f"  - {f} ({size} bytes)\n"
                else:
                    context_str += f"  - {f} (not found - may need to be created or fetched)\n"
        
        if expected_outputs:
            context_str += "\nExpected outputs to produce:\n"
            for f in expected_outputs:
                context_str += f"  - {f}\n"
        
        if context.get("notes"):
            context_str += f"\nAdditional context:\n{context.get('notes')}\n"
        
        # Current project structure
        file_tree = self.sandbox.get_directory_tree(max_depth=2)
        
        prompt = f"""You are a master coordinator breaking down a complex task into subtasks.

Main Task: {main_task}
{context_str}

Current project structure:
```
{file_tree}
```

Requirements for subtasks:
1. Each subtask should be specific and actionable
2. Order subtasks by dependency (what needs to happen first)
3. Each subtask should have clear success criteria
4. Consider what environment/tools each subtask needs
5. Final subtask(s) should produce the expected outputs

Available capabilities for subtasks:
- Python, R, bash, Perl, Java script execution
- Web searching for information
- File reading/writing
- Conda package installation (no sudo)
- Data analysis and transformation

Create 3-7 subtasks. Respond in JSON format:
{{
    "subtasks": [
        {{
            "id": "subtask_1",
            "title": "Brief title",
            "description": "Detailed description of what to do",
            "success_criteria": "How to know it's complete",
            "dependencies": [],
            "required_packages": ["package1", "package2"],
            "language": "python|r|bash|mixed",
            "expected_files": ["files this subtask should create"]
        }},
        {{
            "id": "subtask_2",
            "title": "Brief title",
            "description": "Detailed description",
            "success_criteria": "Success criteria",
            "dependencies": ["subtask_1"],
            "required_packages": [],
            "language": "python",
            "expected_files": []
        }}
    ],
    "environment_name": "suggested conda environment name",
    "reasoning": "Brief explanation of task breakdown"
}}
"""
        
        response = self.llm.invoke(prompt)
        
        # Parse the response
        subtasks = self._parse_subtasks(response, context)
        
        # Store for tracking
        self.subtasks = subtasks
        self._build_dependency_graph(subtasks)
        
        agent_logger.log_task_start(
            agent_name=self.agent_id,
            task_id="decomposition",
            description=f"Decomposed '{main_task[:50]}...' into {len(subtasks)} subtasks",
            attempt=1
        )
        
        # Save decomposition to file
        decomp_path = self.sandbox.get_reports_dir() / "task_decomposition.json"
        with open(decomp_path, 'w') as f:
            json.dump({
                "main_task": main_task,
                "context": context,
                "subtasks": subtasks,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        return subtasks
    
    def _parse_subtasks(self, response: str, context: Dict) -> List[Dict[str, Any]]:
        """Parse LLM response into structured subtasks"""
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                subtasks = parsed.get("subtasks", [])
                
                # Normalize and enhance subtasks
                normalized = []
                for i, st in enumerate(subtasks):
                    normalized.append({
                        "id": st.get("id", f"subtask_{i+1}"),
                        "title": st.get("title", f"Subtask {i+1}"),
                        "description": st.get("description", ""),
                        "success_criteria": st.get("success_criteria", ""),
                        "dependencies": st.get("dependencies", []),
                        "required_packages": st.get("required_packages", []),
                        "language": st.get("language", "python"),
                        "expected_files": st.get("expected_files", []),
                        "status": "pending",
                        "attempts": 0,
                        "context": {
                            "input_files": context.get("input_files", []),
                            "expected_outputs": context.get("expected_outputs", []),
                            "project_dir": str(self.sandbox.project_dir)
                        }
                    })
                
                return normalized
                
        except json.JSONDecodeError:
            pass
        
        # Fallback: try to parse numbered list
        return self._parse_subtasks_from_text(response, context)
    
    def _parse_subtasks_from_text(self, response: str, context: Dict) -> List[Dict[str, Any]]:
        """Fallback parser for non-JSON responses"""
        
        subtasks = []
        lines = response.strip().split('\n')
        current_subtask = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for numbered items
            if line[0].isdigit() and '.' in line[:3]:
                if current_subtask:
                    subtasks.append(current_subtask)
                
                # Extract title and description
                content = line.split('.', 1)[1].strip()
                if ':' in content:
                    title, desc = content.split(':', 1)
                else:
                    title = content[:50]
                    desc = content
                
                current_subtask = {
                    "id": f"subtask_{len(subtasks)+1}",
                    "title": title.strip(),
                    "description": desc.strip(),
                    "success_criteria": "",
                    "dependencies": [f"subtask_{len(subtasks)}"] if subtasks else [],
                    "required_packages": [],
                    "language": "python",
                    "expected_files": [],
                    "status": "pending",
                    "attempts": 0,
                    "context": context
                }
            elif current_subtask and ('success' in line.lower() or 'criteria' in line.lower()):
                current_subtask["success_criteria"] = line.split(':', 1)[-1].strip()
        
        if current_subtask:
            subtasks.append(current_subtask)
        
        return subtasks
    
    def _build_dependency_graph(self, subtasks: List[Dict]):
        """Build dependency graph for task ordering"""
        self.task_dependencies = {}
        
        for st in subtasks:
            task_id = st["id"]
            deps = st.get("dependencies", [])
            self.task_dependencies[task_id] = {
                "dependencies": deps,
                "dependents": [],
                "status": "pending"
            }
        
        # Build reverse dependencies (dependents)
        for st in subtasks:
            task_id = st["id"]
            for dep in st.get("dependencies", []):
                if dep in self.task_dependencies:
                    self.task_dependencies[dep]["dependents"].append(task_id)
    
    def get_next_subtask(self) -> Optional[Dict[str, Any]]:
        """Get next subtask that has all dependencies satisfied"""
        
        for st in self.subtasks:
            if st["status"] != "pending":
                continue
            
            # Check if all dependencies are complete
            deps_satisfied = True
            for dep_id in st.get("dependencies", []):
                dep_info = self.task_dependencies.get(dep_id, {})
                if dep_info.get("status") != "completed":
                    deps_satisfied = False
                    break
            
            if deps_satisfied:
                return st
        
        return None
    
    def mark_subtask_complete(self, task_id: str, report: Dict):
        """Mark a subtask as complete and record its report"""
        
        for st in self.subtasks:
            if st["id"] == task_id:
                st["status"] = "completed"
                st["report"] = report
                break
        
        if task_id in self.task_dependencies:
            self.task_dependencies[task_id]["status"] = "completed"
        
        self.completed_reports.append(report)
    
    def mark_subtask_failed(self, task_id: str, report: Dict):
        """Mark a subtask as failed"""
        
        for st in self.subtasks:
            if st["id"] == task_id:
                st["status"] = "failed"
                st["report"] = report
                break
        
        if task_id in self.task_dependencies:
            self.task_dependencies[task_id]["status"] = "failed"
        
        self.completed_reports.append(report)
    
    def review_failure(
        self,
        subtask: Dict,
        failure_info: Dict
    ) -> Dict[str, Any]:
        """
        Review a failed subtask and decide next steps.
        
        Args:
            subtask: The failed subtask
            failure_info: Information about the failure
        
        Returns:
            Decision on how to proceed
        """
        
        # Get attempt count from failure info
        total_attempts = failure_info.get('total_attempts', failure_info.get('iterations', 1))
        
        prompt = f"""A subtask has failed. Review and decide what to do.

Subtask: {subtask['description']}
Success Criteria: {subtask.get('success_criteria', 'Not specified')}

Failure Information:
- Total attempts so far: {total_attempts}
- Errors: {failure_info.get('errors', [])}
- Summary: {failure_info.get('report', {}).get('summary', 'No summary')}
- File exploration results: {failure_info.get('file_exploration', {})}

Options:
1. REFORMULATE: Rewrite the subtask with a different approach
2. SPLIT: Break this subtask into smaller pieces
3. SKIP: Mark as non-critical and continue (if other tasks don't depend on it)
4. ESCALATE: This is blocking and needs human intervention

IMPORTANT: If total_attempts >= 10, strongly prefer SKIP or ESCALATE to prevent infinite loops.

Dependents that need this task: {self.task_dependencies.get(subtask['id'], {}).get('dependents', [])}

Respond in JSON:
{{
    "decision": "REFORMULATE|SPLIT|SKIP|ESCALATE",
    "reasoning": "explain your decision",
    "new_approach": "if REFORMULATE, describe new approach",
    "sub_subtasks": ["if SPLIT, list smaller tasks"],
    "blocking": true/false
}}
"""
        
        response = self.llm.invoke(prompt)
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
                
                agent_logger.log_reflection(
                    agent_name=self.agent_id,
                    task_id=subtask['id'],
                    reflection=f"Master decision: {decision.get('decision')} - {decision.get('reasoning')}"
                )
                
                return decision
        except:
            pass
        
        # Fallback - if many attempts, force SKIP
        if total_attempts >= 10:
            return {
                "decision": "SKIP",
                "reasoning": f"Too many attempts ({total_attempts}), skipping to prevent infinite loop",
                "blocking": False
            }
        
        return {
            "decision": "ESCALATE",
            "reasoning": "Could not determine best course of action",
            "blocking": True
        }
    
    def generate_final_report(
        self,
        main_task: str,
        subtask_results: List[Dict]
    ) -> str:
        """
        Generate comprehensive final report from all subtask results.
        
        Args:
            main_task: Original task description
            subtask_results: Results from all subtasks
        
        Returns:
            Final report as markdown string
        """
        
        # Gather statistics
        total_subtasks = len(self.subtasks)
        completed = len([s for s in self.subtasks if s["status"] == "completed"])
        failed = len([s for s in self.subtasks if s["status"] in ["failed", "skipped", "max_attempts_exceeded"]])
        
        # Gather all files created
        all_files = []
        all_tools = set()
        total_iterations = 0
        
        for report in self.completed_reports:
            all_files.extend(report.get("files_created", []))
            all_tools.update(report.get("tools_used", []))
            total_iterations += report.get("iterations_used", 0)
        
        # Current file structure
        file_tree = self.sandbox.get_directory_tree(max_depth=3)
        
        # Get summaries from each subtask
        summaries = []
        for report in self.completed_reports:
            summaries.append(f"- {report.get('task_description', 'Unknown')}: {report.get('summary', 'No summary')[:200]}")
        
        prompt = f"""Generate a comprehensive final report for this completed project.

Original Task: {main_task}

Statistics:
- Total subtasks: {total_subtasks}
- Completed: {completed}
- Failed/Skipped: {failed}
- Total iterations: {total_iterations}

Subtask summaries:
{chr(10).join(summaries)}

Files created: {all_files}
Tools used: {list(all_tools)}

Final project structure:
```
{file_tree}
```

Create a professional markdown report that includes:
1. Executive Summary (2-3 sentences)
2. What Was Accomplished (bullet points)
3. Files Generated (with descriptions)
4. Issues Encountered (if any)
5. Recommendations for Next Steps
6. How to Use the Outputs

Keep it concise but informative.
"""
        
        report = self.llm.invoke(prompt)
        
        # Add header and metadata
        full_report = f"""# Project Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Task**: {main_task[:100]}...
**Status**: {'Completed' if failed == 0 else 'Partially Completed'}

---

{report}

---

## Project Statistics

| Metric | Value |
|--------|-------|
| Total Subtasks | {total_subtasks} |
| Completed | {completed} |
| Failed/Skipped | {failed} |
| Total Iterations | {total_iterations} |
| Files Created | {len(all_files)} |

## File Structure

```
{file_tree}
```

---

*Report generated by Multi-Agent System*
"""
        
        # Save report to file
        report_path = self.sandbox.get_reports_dir() / "final_report.md"
        report_path.write_text(full_report)
        
        return full_report
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get current progress summary"""
        
        completed = len([s for s in self.subtasks if s["status"] == "completed"])
        failed = len([s for s in self.subtasks if s["status"] in ["failed", "skipped", "max_attempts_exceeded"]])
        pending = len([s for s in self.subtasks if s["status"] == "pending"])
        
        return {
            "total_subtasks": len(self.subtasks),
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "progress_percent": (completed / len(self.subtasks) * 100) if self.subtasks else 0,
            "subtasks": [
                {"id": s["id"], "title": s["title"], "status": s["status"]}
                for s in self.subtasks
            ]
        }
