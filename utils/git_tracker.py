"""
Git integration for automatic change tracking.
Every agent action is committed with detailed context.
"""

import git
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import json

class GitTracker:
    """Tracks all agent changes via Git commits"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        
        # Initialize or open existing repo
        try:
            self.repo = git.Repo(repo_path)
        except git.InvalidGitRepositoryError:
            self.repo = git.Repo.init(repo_path)
            # Initial commit
            self._initial_setup()
    
    def _initial_setup(self):
        """Create initial commit with .gitignore"""
        gitignore_content = """
# Agent system files
*.pyc
__pycache__/
workflow_state.db
logs/*.jsonl
.env

# Keep structure
!logs/.gitkeep
!data/inputs/.gitkeep
!data/outputs/.gitkeep
"""
        gitignore_path = self.repo_path / ".gitignore"
        gitignore_path.write_text(gitignore_content)
        
        self.repo.index.add([".gitignore"])
        self.repo.index.commit("Initial commit: Agent system setup")
    
    def commit_task_attempt(
        self, 
        task_id: str,
        agent_name: str,
        description: str,
        status: str,  # "success", "failure", "in_progress"
        files_modified: List[str],
        tools_used: List[str],
        result: Optional[str] = None,
        error: Optional[str] = None,
        reasoning: Optional[str] = None
    ):
        """Commit with detailed task context"""
        
        # Create comprehensive commit message
        message = f"""[{status.upper()}] Task {task_id}: {description}

Agent: {agent_name}
Status: {status}
Timestamp: {datetime.now().isoformat()}

Files Modified:
{chr(10).join(f'  - {f}' for f in files_modified) if files_modified else '  None'}

Tools Used:
{chr(10).join(f'  - {t}' for t in tools_used) if tools_used else '  None'}
"""
        
        if reasoning:
            message += f"\nReasoning:\n{reasoning}\n"
        
        if result:
            message += f"\nResult:\n{result[:500]}...\n"
        
        if error:
            message += f"\nError:\n{error}\n"
        
        # Stage and commit
        try:
            # Add all changes
            self.repo.index.add('*')
            
            # Commit
            commit = self.repo.index.commit(message)
            
            # Tag failures for easy identification
            if status == "failure":
                tag_name = f"failure-{task_id}-{datetime.now():%Y%m%d_%H%M%S}"
                self.repo.create_tag(tag_name, message=f"Failed attempt - Task {task_id}")
            
            return commit.hexsha
            
        except Exception as e:
            print(f"Git commit failed: {e}")
            return None
    
    def get_task_history(self, task_id: str) -> List[dict]:
        """Retrieve all commits related to a task"""
        commits = []
        for commit in self.repo.iter_commits():
            if task_id in commit.message:
                commits.append({
                    "sha": commit.hexsha,
                    "message": commit.message,
                    "timestamp": datetime.fromtimestamp(commit.committed_date),
                    "author": str(commit.author)
                })
        return commits
    
    def get_all_failures(self) -> List[dict]:
        """Get all failure tags for troubleshooting"""
        failures = []
        for tag in self.repo.tags:
            if tag.name.startswith("failure-"):
                failures.append({
                    "tag": tag.name,
                    "message": tag.tag.message if tag.tag else "",
                    "commit": tag.commit.hexsha,
                    "timestamp": datetime.fromtimestamp(tag.commit.committed_date)
                })
        return failures

git_tracker = GitTracker()
