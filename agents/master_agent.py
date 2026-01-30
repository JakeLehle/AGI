"""
Improved Master agent that decomposes high-level tasks into subtasks
while preserving critical context: tools, file paths, reference scripts, and language requirements.

Integrated with sandbox, SLURM, and conda infrastructure.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import re
import json

# Use non-deprecated import
try:
    from langchain_ollama import OllamaLLM
except ImportError:
    from langchain_community.llms import Ollama as OllamaLLM

from utils.logging_config import agent_logger


class MasterAgent:
    """Coordinates task decomposition and sub-agent assignment with full context preservation"""
    
    def __init__(
        self,
        sandbox=None,
        ollama_model: str = "llama3.1:70b",
        ollama_base_url: str = "http://127.0.0.1:11434",
        **kwargs
    ):
        """
        Initialize MasterAgent.
        
        Args:
            sandbox: Sandbox instance for file operations
            ollama_model: Ollama model to use for LLM calls
            ollama_base_url: Ollama server URL
            **kwargs: Additional arguments for forward compatibility
        """
        self.llm = OllamaLLM(model=ollama_model, base_url=ollama_base_url)
        self.agent_id = "master"
        self.sandbox = sandbox
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        
        # Track subtask status for reporting
        self.subtask_status = {}
        
        # Store any additional kwargs for subclass compatibility
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def decompose_task(self, main_task: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Break down a high-level task into specific subtasks while preserving ALL context.
        
        Args:
            main_task: The main objective (full prompt with all specifications)
            context: Additional context (user requirements, constraints, etc.)
        
        Returns:
            List of subtask dictionaries with full context preserved
        """
        
        # First, extract critical context from the main task
        extracted_context = self._extract_task_context(main_task)
        
        context_str = ""
        if context:
            context_str = f"\nAdditional Context:\n{json.dumps(context, indent=2, default=str)}"
        
        # Build a detailed prompt that enforces context preservation
        prompt = f"""You are a master coordinator for a computational pipeline. Break down this high-level task into specific, actionable subtasks.

CRITICAL: You MUST preserve ALL specific details from the original task. Do NOT generalize or substitute tools/packages.

=== MAIN TASK ===
{main_task}
{context_str}

=== EXTRACTED CONTEXT (MUST BE PRESERVED) ===
- Language/Framework: {extracted_context.get('language', 'Not specified')}
- Specified Packages: {', '.join(extracted_context.get('packages', [])) or 'None specified'}
- Reference Scripts: {', '.join(extracted_context.get('reference_scripts', [])) or 'None specified'}
- Input Files: {', '.join(extracted_context.get('input_files', [])) or 'None specified'}
- Output Files: {', '.join(extracted_context.get('output_files', [])) or 'None specified'}
- Completed Steps: {', '.join(extracted_context.get('completed_steps', [])) or 'None'}

=== SUBTASK REQUIREMENTS ===
1. Each subtask MUST include the EXACT tools/packages specified in the original task
2. Do NOT substitute packages (e.g., don't suggest R/Seurat if Python/Scanpy was specified)
3. Include specific file paths mentioned in the task
4. Reference any example scripts mentioned
5. Mark dependencies on previous subtasks
6. Skip any steps marked as COMPLETED (✅)
7. Each subtask should have clear inputs and outputs

=== OUTPUT FORMAT (JSON) ===
Return a JSON array of subtasks. Each subtask must have this structure:
```json
[
  {{
    "id": "subtask_1",
    "title": "Brief title",
    "description": "Detailed description preserving ALL specifics from original task",
    "language": "python|r|bash|other",
    "packages": ["package1", "package2"],
    "required_packages": ["package1", "package2"],
    "reference_scripts": ["path/to/script.py"],
    "input_files": ["path/to/input.h5ad"],
    "output_files": ["path/to/output.h5ad"],
    "success_criteria": "How to verify completion",
    "dependencies": ["subtask_id of dependencies"],
    "skip_if_exists": ["files that indicate this is already done"],
    "code_hints": "Any code snippets or specific function calls mentioned"
  }}
]
```

IMPORTANT: 
- If the task mentions specific packages like "popV", "Scanpy", "Squidpy" - use THOSE EXACT packages
- If the task mentions a reference script, include it so the sub-agent can examine it
- If files are marked as existing or steps as completed, skip them

Return ONLY the JSON array, no other text.
"""
        
        response = self.llm.invoke(prompt)
        
        # Parse the response
        subtasks = self._parse_subtasks_json(response, extracted_context)
        
        # Filter out completed subtasks
        subtasks = self._filter_completed_subtasks(subtasks, extracted_context)
        
        # Initialize tracking
        for st in subtasks:
            self.subtask_status[st['id']] = {'status': 'pending', 'attempts': 0}
        
        agent_logger.log_task_start(
            agent_name=self.agent_id,
            task_id="decomposition",
            description=f"Decomposed task into {len(subtasks)} subtasks (preserved context: {len(extracted_context.get('packages', []))} packages, {len(extracted_context.get('reference_scripts', []))} reference scripts)",
            attempt=1
        )
        
        return subtasks
    
    def _extract_task_context(self, task: str) -> Dict[str, Any]:
        """Extract critical context from the task description"""
        
        context = {
            "language": None,
            "packages": [],
            "reference_scripts": [],
            "input_files": [],
            "output_files": [],
            "completed_steps": [],
            "code_blocks": [],
            "huggingface_repos": [],
            "specific_functions": []
        }
        
        # Detect language
        python_indicators = ['python', 'scanpy', 'squidpy', 'anndata', 'pandas', 'numpy', '.py', 'popv', 'h5ad']
        r_indicators = ['seurat', 'singlecell', 'bioconductor', '.R', 'library(']
        
        task_lower = task.lower()
        python_score = sum(1 for ind in python_indicators if ind in task_lower)
        r_score = sum(1 for ind in r_indicators if ind in task_lower)
        
        if python_score > r_score:
            context["language"] = "python"
        elif r_score > python_score:
            context["language"] = "r"
        
        # Extract Python packages (common bioinformatics packages)
        python_packages = [
            'scanpy', 'squidpy', 'anndata', 'pandas', 'numpy', 'scipy',
            'popv', 'popV', 'scvi', 'scvi-tools', 'cellxgene', 'leidenalg',
            'matplotlib', 'seaborn', 'plotly', 'umap', 'scikit-learn',
            'harmony', 'bbknn', 'scanorama', 'mnnpy', 'celltypist',
            'decoupler', 'omnipath', 'pydeseq2', 'gseapy', 'pyscenic'
        ]
        for pkg in python_packages:
            # Case-insensitive but preserve original case if found
            if pkg.lower() in task_lower:
                # Find the actual case used in the task
                pattern = re.compile(re.escape(pkg), re.IGNORECASE)
                matches = pattern.findall(task)
                if matches:
                    context["packages"].append(matches[0])
        
        # Extract reference scripts
        script_patterns = [
            r'[Rr]eference\s+[Ss]cript[s]?:\s*[`"]?([^\s`"]+\.py)[`"]?',
            r'[Ee]xample\s+[Ss]cript[s]?:\s*[`"]?([^\s`"]+\.py)[`"]?',
            r'scripts?/[^\s`"]+\.py',
            r'[`"]([^`"]+\.py)[`"]'
        ]
        for pattern in script_patterns:
            matches = re.findall(pattern, task)
            context["reference_scripts"].extend(matches)
        
        # Extract file paths
        file_patterns = [
            r'[`"]?([^\s`"]*\.h5ad)[`"]?',  # AnnData files
            r'[`"]?([^\s`"]*\.csv)[`"]?',    # CSV files
            r'[`"]?(data/[^\s`"]+)[`"]?',    # Data directory paths
            r'[`"]?(outputs?/[^\s`"]+)[`"]?' # Output directory paths
        ]
        for pattern in file_patterns:
            matches = re.findall(pattern, task)
            for match in matches:
                if 'input' in match.lower() or 'processed' in match.lower():
                    context["input_files"].append(match)
                elif 'output' in match.lower() or 'result' in match.lower():
                    context["output_files"].append(match)
        
        # Extract completed steps (marked with ✅ or "COMPLETED")
        completed_patterns = [
            r'✅\s*(?:COMPLETED:?)?\s*([^\n]+)',
            r'\[x\]\s*([^\n]+)',
            r'COMPLETED:\s*([^\n]+)'
        ]
        for pattern in completed_patterns:
            matches = re.findall(pattern, task)
            context["completed_steps"].extend(matches)
        
        # Extract code blocks
        code_block_pattern = r'```(?:python|py)?\n?(.*?)```'
        code_blocks = re.findall(code_block_pattern, task, re.DOTALL)
        context["code_blocks"] = code_blocks
        
        # Extract HuggingFace repos
        hf_pattern = r'huggingface_repo\s*=\s*["\']([^"\']+)["\']'
        hf_matches = re.findall(hf_pattern, task)
        context["huggingface_repos"] = hf_matches
        
        # Extract specific function calls mentioned
        func_pattern = r'(?:use|call|run)\s+[`"]?(\w+\.\w+|\w+)\([^)]*\)[`"]?'
        func_matches = re.findall(func_pattern, task, re.IGNORECASE)
        context["specific_functions"] = func_matches
        
        # Deduplicate
        for key in context:
            if isinstance(context[key], list):
                context[key] = list(dict.fromkeys(context[key]))
        
        return context
    
    def _parse_subtasks_json(self, response: str, extracted_context: Dict) -> List[Dict[str, Any]]:
        """Parse JSON response into structured subtasks"""
        
        # Try to extract JSON from response
        try:
            # Look for JSON array
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                subtasks = json.loads(json_match.group())
                
                # Ensure all subtasks have required fields and inherit context
                for i, subtask in enumerate(subtasks):
                    subtask.setdefault("id", f"subtask_{i+1}")
                    subtask.setdefault("status", "pending")
                    subtask.setdefault("attempts", 0)
                    subtask.setdefault("dependencies", [])
                    
                    # Inherit context if not specified
                    if not subtask.get("language") and extracted_context.get("language"):
                        subtask["language"] = extracted_context["language"]
                    
                    if not subtask.get("packages") and extracted_context.get("packages"):
                        subtask["packages"] = extracted_context["packages"]
                    
                    # Sync packages and required_packages
                    if subtask.get("packages") and not subtask.get("required_packages"):
                        subtask["required_packages"] = subtask["packages"]
                    
                    if not subtask.get("reference_scripts") and extracted_context.get("reference_scripts"):
                        subtask["reference_scripts"] = extracted_context["reference_scripts"]
                    
                    # Add code hints from extracted context
                    if extracted_context.get("code_blocks") and not subtask.get("code_hints"):
                        subtask["code_hints"] = extracted_context["code_blocks"]
                    
                    if extracted_context.get("huggingface_repos"):
                        subtask["huggingface_repos"] = extracted_context["huggingface_repos"]
                
                return subtasks
        except json.JSONDecodeError:
            pass
        
        # Fallback: parse as text (old format)
        return self._parse_subtasks_text(response, extracted_context)
    
    def _parse_subtasks_text(self, response: str, extracted_context: Dict) -> List[Dict[str, Any]]:
        """Fallback parser for text-based responses"""
        
        subtasks = []
        lines = response.strip().split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            
            parts = line.split('.', 1)
            if len(parts) < 2:
                continue
            
            content = parts[1].strip()
            
            if ':' in content:
                title, rest = content.split(':', 1)
                title = title.strip()
                
                success_criteria = ""
                if '|' in rest:
                    description, criteria_part = rest.split('|', 1)
                    description = description.strip()
                    if 'Success criteria:' in criteria_part:
                        success_criteria = criteria_part.split('Success criteria:')[1].strip()
                else:
                    description = rest.strip()
            else:
                title = content[:50]
                description = content
                success_criteria = ""
            
            subtasks.append({
                "id": f"subtask_{i+1}",
                "title": title,
                "description": description,
                "success_criteria": success_criteria,
                "status": "pending",
                "attempts": 0,
                "dependencies": [],
                # Inherit from extracted context
                "language": extracted_context.get("language"),
                "packages": extracted_context.get("packages", []),
                "required_packages": extracted_context.get("packages", []),
                "reference_scripts": extracted_context.get("reference_scripts", []),
                "input_files": extracted_context.get("input_files", []),
                "output_files": extracted_context.get("output_files", []),
                "code_hints": extracted_context.get("code_blocks", []),
                "huggingface_repos": extracted_context.get("huggingface_repos", [])
            })
        
        return subtasks
    
    def _filter_completed_subtasks(self, subtasks: List[Dict], extracted_context: Dict) -> List[Dict]:
        """Filter out subtasks that are already completed"""
        
        completed_steps = extracted_context.get("completed_steps", [])
        if not completed_steps:
            return subtasks
        
        filtered = []
        for subtask in subtasks:
            # Check if this subtask matches any completed step
            is_completed = False
            for completed in completed_steps:
                # Simple matching - could be more sophisticated
                if (completed.lower() in subtask.get("title", "").lower() or
                    completed.lower() in subtask.get("description", "").lower()):
                    is_completed = True
                    agent_logger.log_reflection(
                        agent_name=self.agent_id,
                        task_id=subtask["id"],
                        reflection=f"Skipping subtask '{subtask['title']}' - matches completed step: {completed}"
                    )
                    break
            
            if not is_completed:
                filtered.append(subtask)
        
        return filtered
    
    def mark_subtask_complete(self, task_id: str, result: Dict = None):
        """Mark a subtask as complete"""
        if task_id in self.subtask_status:
            self.subtask_status[task_id]['status'] = 'completed'
            self.subtask_status[task_id]['result'] = result
    
    def mark_subtask_failed(self, task_id: str, result: Dict = None):
        """Mark a subtask as failed"""
        if task_id in self.subtask_status:
            self.subtask_status[task_id]['status'] = 'failed'
            self.subtask_status[task_id]['result'] = result
            self.subtask_status[task_id]['attempts'] += 1
    
    def review_failure(self, subtask: Dict, failure_info: Dict) -> Dict[str, Any]:
        """
        Review a failed subtask and decide next steps, preserving context.
        """
        
        # Include full context in the review
        context_summary = f"""
Subtask Context:
- Language: {subtask.get('language', 'Not specified')}
- Packages: {', '.join(subtask.get('packages', []))}
- Reference Scripts: {', '.join(subtask.get('reference_scripts', []))}
- Input Files: {', '.join(subtask.get('input_files', []))}
"""
        
        prompt = f"""
A subtask has failed after {failure_info.get('attempts', 1)} attempt(s).

Subtask: {subtask.get('description', subtask.get('title', 'Unknown'))}
Success Criteria: {subtask.get('success_criteria', 'Not specified')}

{context_summary}

Failure Information:
{failure_info.get('reflection', {}).get('analysis', 'No analysis available')}

Error: {failure_info.get('error', 'No error message')}

As the master coordinator, what should we do?

IMPORTANT: When reformulating, you MUST preserve:
- The same language/framework ({subtask.get('language', 'as specified')})
- The same packages ({', '.join(subtask.get('packages', []))})
- The same reference scripts
- The same input/output files

Options:
1. REFORMULATE: Rewrite the subtask with different approach (but SAME tools)
2. SPLIT: Break this subtask into smaller pieces
3. SKIP: Mark as non-critical and continue
4. ESCALATE: This is blocking and needs human intervention

Respond in JSON:
{{
    "decision": "REFORMULATE/SPLIT/SKIP/ESCALATE",
    "reasoning": "explain your decision",
    "new_approach": "if REFORMULATE, describe new approach using the SAME specified tools",
    "sub_subtasks": ["if SPLIT, list smaller tasks"],
    "preserved_context": {{
        "language": "{subtask.get('language')}",
        "packages": {json.dumps(subtask.get('packages', []))},
        "reference_scripts": {json.dumps(subtask.get('reference_scripts', []))}
    }}
}}
"""
        
        response = self.llm.invoke(prompt)
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
                
                agent_logger.log_reflection(
                    agent_name=self.agent_id,
                    task_id=subtask.get('id', 'unknown'),
                    reflection=f"Master decision: {decision.get('decision')} - {decision.get('reasoning')}"
                )
                
                return decision
        except:
            pass
        
        return {
            "decision": "REFORMULATE",
            "reasoning": "Failed to parse master decision, defaulting to reformulation",
            "new_approach": "Try alternative method using the same specified tools"
        }
    
    def generate_final_report(self, main_task: str, subtask_results: List[Dict]) -> str:
        """Generate comprehensive report from all subtask results"""
        
        # Build results summary
        results_summary = []
        for r in subtask_results:
            if r.get('success') or r.get('status') == 'completed':
                task_id = r.get('task_id', r.get('id', 'unknown'))
                output = r.get('result', {}).get('output', r.get('report', {}).get('summary', 'No output'))
                if isinstance(output, dict):
                    output = str(output)
                results_summary.append(f"- {task_id}: {output[:200]}")
        
        results_str = "\n".join(results_summary) if results_summary else "No completed subtasks"
        
        # Extract tools used
        all_packages = set()
        for r in subtask_results:
            packages = r.get('packages', [])
            if packages:
                all_packages.update(packages)
        
        prompt = f"""
Generate a comprehensive final report for this completed project.

Original Task: {main_task}

Completed Subtasks:
{results_str}

Tools/Packages Used: {', '.join(all_packages) if all_packages else 'Not tracked'}

Create a professional report that:
1. Summarizes the overall accomplishment
2. Highlights key findings from each subtask
3. Lists the specific tools and packages used
4. Provides actionable recommendations
5. Notes any limitations or areas for future work

Format as a well-structured document with sections.
"""
        
        report = self.llm.invoke(prompt)
        
        return report
