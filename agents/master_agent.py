"""
Master agent that decomposes high-level tasks into subtasks
and coordinates sub-agents.
"""

from typing import Dict, Any, List
from langchain_community.llms import Ollama
from utils.logging_config import agent_logger

class MasterAgent:
    """Coordinates task decomposition and sub-agent assignment"""
    
    def __init__(self, ollama_model: str = "llama3.1:70b"):
        self.llm = Ollama(model=ollama_model)
        self.agent_id = "master"
    
    def decompose_task(self, main_task: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Break down a high-level task into specific subtasks
        
        Args:
            main_task: The main objective
            context: Additional context (user requirements, constraints, etc.)
        
        Returns:
            List of subtask dictionaries
        """
        
        context_str = ""
        if context:
            context_str = f"\nContext:\n{context}"
        
        prompt = f"""
You are a master coordinator. Break down this high-level task into 3-7 specific, actionable subtasks.

Main Task: {main_task}
{context_str}

Requirements for subtasks:
1. Each subtask should be specific and measurable
2. Subtasks should build on each other logically
3. Each subtask should have clear success criteria
4. Order subtasks by dependency (what needs to happen first)

Format your response as a numbered list with this structure:
1. [Subtask title]: [Detailed description] | Success criteria: [How to know it's done]
2. [Subtask title]: [Detailed description] | Success criteria: [How to know it's done]
...

Example:
1. Data Collection: Gather all research papers from PubMed about biotech in San Antonio | Success criteria: At least 20 relevant papers collected and saved as JSON
2. Analysis: Extract key themes and capabilities from papers | Success criteria: Structured summary of top 5 research areas with supporting evidence
"""
        
        response = self.llm.invoke(prompt)
        
        # Parse the response into structured subtasks
        subtasks = self._parse_subtasks(response)
        
        agent_logger.log_task_start(
            agent_name=self.agent_id,
            task_id="decomposition",
            description=f"Decomposed '{main_task}' into {len(subtasks)} subtasks",
            attempt=1
        )
        
        return subtasks
    
    def _parse_subtasks(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured subtasks"""
        
        subtasks = []
        lines = response.strip().split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            
            # Remove numbering
            parts = line.split('.', 1)
            if len(parts) < 2:
                continue
            
            content = parts[1].strip()
            
            # Split into title and description
            if ':' in content:
                title, rest = content.split(':', 1)
                title = title.strip()
                
                # Extract success criteria if present
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
                "dependencies": []  # Could be enhanced to parse dependencies
            })
        
        return subtasks
    
    def review_failure(self, subtask: Dict, failure_info: Dict) -> Dict[str, Any]:
        """
        Review a failed subtask and decide next steps
        
        Args:
            subtask: The failed subtask
            failure_info: Information about the failure
        
        Returns:
            Decision on how to proceed
        """
        
        prompt = f"""
A subtask has failed after {failure_info.get('attempts', 1)} attempt(s).

Subtask: {subtask['description']}
Success Criteria: {subtask.get('success_criteria', 'Not specified')}

Failure Information:
{failure_info.get('reflection', {}).get('analysis', 'No analysis available')}

Error: {failure_info.get('error', 'No error message')}

As the master coordinator, what should we do?

Options:
1. REFORMULATE: Rewrite the subtask with different approach
2. SPLIT: Break this subtask into smaller pieces
3. SKIP: Mark as non-critical and continue
4. ESCALATE: This is blocking and needs human intervention

Respond in JSON:
{{
    "decision": "REFORMULATE/SPLIT/SKIP/ESCALATE",
    "reasoning": "explain your decision",
    "new_approach": "if REFORMULATE, describe new approach",
    "sub_subtasks": ["if SPLIT, list smaller tasks"]
}}
"""
        
        response = self.llm.invoke(prompt)
        
        # Parse decision
        try:
            import json
            import re
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
        
        # Fallback
        return {
            "decision": "REFORMULATE",
            "reasoning": "Failed to parse master decision, defaulting to reformulation",
            "new_approach": "Try alternative method based on available tools"
        }
    
    def generate_final_report(self, main_task: str, subtask_results: List[Dict]) -> str:
        """Generate comprehensive report from all subtask results"""
        
        results_summary = "\n".join([
            f"- {r['task_id']}: {r.get('result', {}).get('output', 'No output')[:200]}"
            for r in subtask_results
            if r.get('success')
        ])
        
        prompt = f"""
Generate a comprehensive final report for this completed project.

Original Task: {main_task}

Completed Subtasks:
{results_summary}

Create a professional report that:
1. Summarizes the overall accomplishment
2. Highlights key findings from each subtask
3. Provides actionable recommendations
4. Notes any limitations or areas for future work

Format as a well-structured document with sections.
"""
        
        report = self.llm.invoke(prompt)
        
        return report
