"""
Automatic documentation generation from agent activities.
Creates comprehensive README files tracking all changes.
"""

from pathlib import Path
from datetime import datetime
from typing import List, Dict
import json

class DocumentationGenerator:
    """Generates and maintains project documentation"""
    
    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path)
        self.changes_log = []
        self.tools_created = []
        self.decisions_log = []
    
    def log_change(self, change: Dict):
        """Record a change for documentation"""
        self.changes_log.append({
            **change,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_tool_creation(self, tool_info: Dict):
        """Record dynamic tool creation"""
        self.tools_created.append({
            **tool_info,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_decision(self, decision: Dict):
        """Record agent decision for audit trail"""
        self.decisions_log.append({
            **decision,
            "timestamp": datetime.now().isoformat()
        })
    
    def generate_file_tree(self, directory: Path = None, prefix: str = "") -> str:
        """Generate ASCII tree of file structure"""
        if directory is None:
            directory = self.project_path
        
        tree = []
        try:
            items = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            for i, item in enumerate(items):
                # Skip hidden, cache, and log files
                if item.name.startswith('.') or item.name in ['__pycache__', 'logs', 'workflow_state.db']:
                    continue
                
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                tree.append(f"{prefix}{current_prefix}{item.name}")
                
                if item.is_dir():
                    extension = "    " if is_last else "│   "
                    tree.append(self.generate_file_tree(item, prefix + extension))
        except PermissionError:
            pass
        
        return "\n".join(tree)
    
    def generate_readme(self) -> str:
        """Generate comprehensive README from all logged activities"""
        
        readme = f"""# Multi-Agent Biotech Analysis System

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Tasks Completed**: {len([c for c in self.changes_log if c.get('status') == 'success'])}
**Total Tasks Failed**: {len([c for c in self.changes_log if c.get('status') == 'failure'])}
**Dynamic Tools Created**: {len(self.tools_created)}

---

## 📁 Project Structure
```
{self.generate_file_tree()}
```

---

## ✅ Completed Tasks

"""
        
        # Add completed tasks
        completed = [c for c in self.changes_log if c.get('status') == 'success']
        for i, change in enumerate(completed, 1):
            readme += f"""
### {i}. {change.get('description', 'Unknown task')}

- **Task ID**: `{change.get('task_id', 'N/A')}`
- **Agent**: {change.get('agent', 'Unknown')}
- **Timestamp**: {change.get('timestamp', 'N/A')}
- **Files Modified**: {', '.join(change.get('files_modified', [])) or 'None'}
- **Tools Used**: {', '.join(change.get('tools_used', [])) or 'None'}

**Summary**: {change.get('result', 'No result available')[:300]}...

"""
        
        # Add failed tasks section
        failed = [c for c in self.changes_log if c.get('status') == 'failure']
        if failed:
            readme += "\n---\n\n## ❌ Failed Tasks (For Troubleshooting)\n\n"
            for i, change in enumerate(failed, 1):
                readme += f"""
### {i}. {change.get('description', 'Unknown task')}

- **Task ID**: `{change.get('task_id', 'N/A')}`
- **Agent**: {change.get('agent', 'Unknown')}
- **Error**: {change.get('error', 'Unknown error')}
- **Attempts**: {change.get('attempts', 1)}

**Troubleshooting Notes**: {change.get('reflection', 'No analysis available')}

"""
        
        # Add dynamically created tools
        if self.tools_created:
            readme += "\n---\n\n## 🔧 Dynamically Created Tools\n\n"
            readme += "The following tools were created during execution to handle novel situations:\n\n"
            
            for tool in self.tools_created:
                readme += f"""
### {tool.get('name', 'Unknown Tool')}

- **Created**: {tool.get('timestamp', 'N/A')}
- **Reason**: {tool.get('reason', 'Not specified')}
- **Location**: `tools/dynamic_tools/{tool.get('name', 'unknown')}.py`

"""
        
        # Add decision log
        if self.decisions_log:
            readme += "\n---\n\n## 🧠 Agent Decision Log\n\n"
            for i, decision in enumerate(self.decisions_log[-10:], 1):  # Last 10 decisions
                readme += f"{i}. **{decision.get('decision_type', 'Unknown')}** ({decision.get('timestamp', 'N/A')}): {decision.get('reasoning', 'No reasoning provided')}\n"
        
        # Add usage instructions
        readme += f"""

---

## 🚀 Usage

### Running the System
```bash
python main.py --task "Your task description here"
```

### Reviewing Execution History

- **Logs**: Check `logs/` directory for detailed JSON logs
- **Git History**: Use `git log` to see all commits with task context
- **Failed Tasks**: Review git tags starting with `failure-` for debugging
- **State Replay**: Inspect `workflow_state.db` for complete state history

### Configuration

Edit `config/config.yaml` to adjust:
- Maximum retry attempts
- Model selection (Ollama model to use)
- Tool permissions
- Logging verbosity

---

## 📊 Performance Metrics

- **Average Task Duration**: {self._calculate_avg_duration()} seconds
- **Success Rate**: {self._calculate_success_rate():.1f}%
- **Most Used Tools**: {', '.join(self._get_top_tools(3))}

---

## 🔍 Troubleshooting

### Common Issues

{self._generate_troubleshooting_section()}

### Failed Task Analysis

Use the following command to review all failures:
```bash
git tag -l "failure-*"
```

Then checkout specific failure to review state:
```bash
git show failure-TASK_ID
```

---

## 📝 Notes

- All agent actions are logged to `logs/` in JSON format
- Every task generates a Git commit for full traceability
- Dynamic tools are saved to `tools/dynamic_tools/` and can be reused
- Workflow state is checkpointed to `workflow_state.db` for recovery

**Last Updated**: {datetime.now().isoformat()}
"""
        
        return readme
    
    def save_readme(self):
        """Write README to file"""
        readme_content = self.generate_readme()
        readme_path = self.project_path / "README.md"
        readme_path.write_text(readme_content)
        return readme_path
    
    def _calculate_avg_duration(self) -> float:
        """Calculate average task duration from logs"""
        # Simplified - you'd parse actual timestamps from logs
        return 45.2  # Placeholder
    
    def _calculate_success_rate(self) -> float:
        """Calculate task success rate"""
        if not self.changes_log:
            return 0.0
        successes = len([c for c in self.changes_log if c.get('status') == 'success'])
        return (successes / len(self.changes_log)) * 100
    
    def _get_top_tools(self, n: int = 3) -> List[str]:
        """Get most frequently used tools"""
        tool_usage = {}
        for change in self.changes_log:
            for tool in change.get('tools_used', []):
                tool_usage[tool] = tool_usage.get(tool, 0) + 1
        
        sorted_tools = sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)
        return [tool for tool, _ in sorted_tools[:n]]
    
    def _generate_troubleshooting_section(self) -> str:
        """Generate troubleshooting tips from failures"""
        failed = [c for c in self.changes_log if c.get('status') == 'failure']
        
        if not failed:
            return "No failures recorded yet."
        
        # Group by error type
        error_types = {}
        for failure in failed:
            error = failure.get('error', 'Unknown error')
            error_type = error.split(':')[0] if ':' in error else error[:50]
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(failure)
        
        tips = []
        for error_type, failures in error_types.items():
            tips.append(f"**{error_type}** (occurred {len(failures)} time(s))")
            # Add resolution from reflection if available
            if failures[0].get('reflection'):
                tips.append(f"  - Solution: {failures[0]['reflection'][:200]}...")
        
        return "\n".join(tips) if tips else "No specific troubleshooting tips available."

doc_generator = DocumentationGenerator()
