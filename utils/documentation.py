"""
Automatic documentation generation from agent activities.
Creates comprehensive README files tracking all changes.

Supports project-scoped documentation where each project has its own docs.
"""

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import json


class DocumentationGenerator:
    """
    Generates and maintains project documentation.
    
    Can be initialized with a specific project path to ensure all documentation
    goes to the project directory rather than the AGI root.
    """
    
    def __init__(self, project_path: str = None):
        """
        Initialize documentation generator for a specific project.
        
        Args:
            project_path: Path to the project directory. If None, documentation
                         is deferred until configure() is called.
        """
        self.project_path = Path(project_path) if project_path else None
        self.changes_log = []
        self.tools_created = []
        self.decisions_log = []
        self._configured = False
        
        if self.project_path:
            self._configured = True
    
    def configure(self, project_path: str):
        """
        Configure or reconfigure documentation for a specific project directory.
        
        This should be called early in main.py after project_dir is known.
        
        Args:
            project_path: Absolute path to the project directory
        """
        self.project_path = Path(project_path).resolve()
        self._configured = True
        
        # Ensure reports directory exists
        reports_dir = self.project_path / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
    
    def _ensure_configured(self):
        """Ensure generator is configured before use"""
        if not self._configured or not self.project_path:
            raise RuntimeError(
                "DocumentationGenerator not configured. "
                "Call configure(project_path) before use."
            )
    
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
    
    def generate_file_tree(self, directory: Path = None, prefix: str = "", max_depth: int = 3, current_depth: int = 0) -> str:
        """Generate ASCII tree of file structure"""
        self._ensure_configured()
        
        if directory is None:
            directory = self.project_path
        
        if current_depth >= max_depth:
            return ""
        
        tree = []
        try:
            items = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            
            # Filter out items to skip
            skip_patterns = {'.git', '__pycache__', 'logs', 'workflow_state.db', 
                          '.pyc', 'temp', 'node_modules', '.pytest_cache'}
            items = [i for i in items if i.name not in skip_patterns and not i.name.startswith('.')]
            
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                tree.append(f"{prefix}{current_prefix}{item.name}")
                
                if item.is_dir():
                    extension = "    " if is_last else "│   "
                    subtree = self.generate_file_tree(item, prefix + extension, max_depth, current_depth + 1)
                    if subtree:
                        tree.append(subtree)
        except PermissionError:
            pass
        
        return "\n".join(filter(None, tree))
    
    def generate_readme(self, project_name: str = None) -> str:
        """Generate comprehensive README from all logged activities"""
        self._ensure_configured()
        
        if project_name is None:
            project_name = self.project_path.name
        
        completed_count = len([c for c in self.changes_log if c.get('status') == 'success'])
        failed_count = len([c for c in self.changes_log if c.get('status') == 'failure'])
        
        readme = f"""# {project_name}

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Tasks Completed**: {completed_count}
**Total Tasks Failed**: {failed_count}
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
            files_modified = change.get('files_modified', [])
            tools_used = change.get('tools_used', [])
            
            readme += f"""
### {i}. {change.get('description', 'Unknown task')}

- **Task ID**: `{change.get('task_id', 'N/A')}`
- **Agent**: {change.get('agent', 'Unknown')}
- **Timestamp**: {change.get('timestamp', 'N/A')}
- **Files Modified**: {', '.join(files_modified) if files_modified else 'None'}
- **Tools Used**: {', '.join(tools_used) if tools_used else 'None'}

**Summary**: {str(change.get('result', 'No result available'))[:300]}...

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
            for i, decision in enumerate(self.decisions_log[-10:], 1):
                readme += f"{i}. **{decision.get('decision_type', 'Unknown')}** ({decision.get('timestamp', 'N/A')}): {decision.get('reasoning', 'No reasoning provided')}\n"
        
        # Add usage instructions
        readme += f"""

---

## 🚀 Usage

### Re-running the Pipeline

```bash
# Activate environment
conda activate AGI

# Re-run with same prompt
python main.py --prompt-file prompts/[original_prompt].txt --project-dir {self.project_path}
```

### Reviewing Execution History

- **Logs**: Check `logs/` directory for detailed JSON logs
- **Git History**: Use `git log` to see all commits with task context
- **Reports**: Check `reports/` for generated analysis

---

## 📊 Performance Metrics

- **Average Task Duration**: {self._calculate_avg_duration()} seconds
- **Success Rate**: {self._calculate_success_rate():.1f}%
- **Most Used Tools**: {', '.join(self._get_top_tools(3)) or 'None'}

---

## 🔍 Troubleshooting

{self._generate_troubleshooting_section()}

---

## 📝 Notes

- All agent actions are logged to `logs/` in JSON format
- Every task generates a Git commit for full traceability  
- Dynamic tools are saved to `tools/dynamic_tools/` and can be reused
- Workflow state is checkpointed to `workflow_state.db` for recovery

**Last Updated**: {datetime.now().isoformat()}
"""
        
        return readme
    
    def save_readme(self, filename: str = "README.md") -> Path:
        """Write README to file in project directory"""
        self._ensure_configured()
        
        readme_content = self.generate_readme()
        readme_path = self.project_path / filename
        readme_path.write_text(readme_content)
        return readme_path
    
    def save_execution_report(self, report_name: str = None) -> Path:
        """Save detailed execution report to reports/ directory"""
        self._ensure_configured()
        
        reports_dir = self.project_path / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        if report_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"execution_report_{timestamp}.json"
        
        report_path = reports_dir / report_name
        
        report_data = {
            "generated_at": datetime.now().isoformat(),
            "project_path": str(self.project_path),
            "summary": {
                "total_tasks": len(self.changes_log),
                "completed": len([c for c in self.changes_log if c.get('status') == 'success']),
                "failed": len([c for c in self.changes_log if c.get('status') == 'failure']),
                "tools_created": len(self.tools_created),
                "decisions_made": len(self.decisions_log)
            },
            "changes_log": self.changes_log,
            "tools_created": self.tools_created,
            "decisions_log": self.decisions_log,
            "metrics": {
                "success_rate": self._calculate_success_rate(),
                "top_tools": self._get_top_tools(5)
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        return report_path
    
    def _calculate_avg_duration(self) -> float:
        """Calculate average task duration from logs"""
        # TODO: Parse actual timestamps from logs for real duration
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
            return "No failures recorded - all tasks completed successfully! 🎉"
        
        # Group by error type
        error_types = {}
        for failure in failed:
            error = failure.get('error', 'Unknown error')
            error_type = error.split(':')[0] if ':' in error else error[:50]
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(failure)
        
        tips = ["### Common Issues\n"]
        for error_type, failures in error_types.items():
            tips.append(f"**{error_type}** (occurred {len(failures)} time(s))")
            if failures[0].get('reflection'):
                tips.append(f"  - Suggestion: {failures[0]['reflection'][:200]}...")
            tips.append("")
        
        return "\n".join(tips) if len(tips) > 1 else "No specific troubleshooting tips available."


# Global instance - will be configured by main.py
doc_generator = DocumentationGenerator()


def configure_documentation(project_path: str):
    """
    Convenience function to configure the global documentation generator.
    
    Call this from main.py after project_dir is known:
        from utils.documentation import configure_documentation
        configure_documentation(project_dir)
    """
    doc_generator.configure(project_path)
