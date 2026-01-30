"""
Git integration for automatic change tracking.
Every agent action is committed with detailed context.

Supports project-scoped git operations where each project has its own repository.

NOTE: Git tracking is OPTIONAL. If git is not available, the pipeline will
continue to function - you just won't have automatic git commits.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional, List
import json
import os

# Handle missing git gracefully - don't crash on import
GIT_AVAILABLE = False
git = None

try:
    # Suppress the GitPython warning if git isn't found
    os.environ.setdefault('GIT_PYTHON_REFRESH', 'quiet')
    import git as git_module
    git = git_module
    GIT_AVAILABLE = True
except ImportError as e:
    # Git module not installed or git executable not found
    GIT_AVAILABLE = False
    print(f"Note: Git tracking disabled - {e}")


class GitTracker:
    """
    Tracks all agent changes via Git commits.
    
    Can be initialized with a specific project path to ensure all git operations
    happen in the project's repository, not the AGI root.
    
    If git is not available, all operations become no-ops and the pipeline
    continues without git tracking.
    """
    
    def __init__(self, repo_path: str = None):
        """
        Initialize git tracker for a specific project.
        
        Args:
            repo_path: Path to the project directory (git repo root).
                      If None, git operations are deferred until configure() is called.
        """
        self.repo_path = Path(repo_path) if repo_path else None
        self.repo = None
        self._configured = False
        self._git_available = GIT_AVAILABLE
        
        if not self._git_available:
            return
            
        if self.repo_path:
            self._init_repo()
    
    def configure(self, repo_path: str, auto_init: bool = True):
        """
        Configure or reconfigure git tracking for a specific project directory.
        
        This should be called early in main.py after project_dir is known.
        
        Args:
            repo_path: Absolute path to the project directory
            auto_init: If True, initialize git repo if it doesn't exist
        """
        if not self._git_available:
            print(f"Git tracking: Disabled (git not available)")
            return
            
        self.repo_path = Path(repo_path).resolve()
        
        if auto_init:
            self._init_repo()
        else:
            # Just try to open existing repo
            try:
                self.repo = git.Repo(self.repo_path)
                self._configured = True
            except git.InvalidGitRepositoryError:
                self.repo = None
                self._configured = False
    
    def _init_repo(self):
        """Initialize or open the git repository"""
        if not self._git_available or not self.repo_path:
            return
            
        try:
            self.repo = git.Repo(self.repo_path)
            self._configured = True
        except git.InvalidGitRepositoryError:
            # Initialize new repo
            try:
                self.repo = git.Repo.init(self.repo_path)
                self._initial_setup()
                self._configured = True
            except Exception as e:
                print(f"Git tracking: Could not initialize repo - {e}")
                self._configured = False
    
    def _initial_setup(self):
        """Create initial commit with .gitignore if this is a new repo"""
        if not self.repo:
            return
            
        gitignore_path = self.repo_path / ".gitignore"
        
        # Only create .gitignore if it doesn't exist
        if not gitignore_path.exists():
            gitignore_content = """# =============================================================================
# AGI Pipeline Project .gitignore
# =============================================================================

# Data - never track
data/

# Logs - never track  
logs/
*.log
*.jsonl

# Temporary files
temp/
*.tmp

# Reports (generated)
reports/

# SLURM job output
slurm/logs/
*.out
*.err

# Python
__pycache__/
*.py[cod]
*.so
.Python
*.egg-info/

# Database/state files
*.db
workflow_state.db

# IDE
.idea/
.vscode/

# OS
.DS_Store
Thumbs.db

# Keep tracked items
!.gitkeep
!config/
!prompts/
!scripts/
!slurm/scripts/
!envs/*.yml
"""
            gitignore_path.write_text(gitignore_content)
        
        # Try to make initial commit
        try:
            self.repo.index.add([".gitignore"])
            
            # Add any other tracked files that exist
            for pattern in ["*.py", "*.yaml", "*.yml", "*.md", "*.txt", "*.sh"]:
                try:
                    self.repo.index.add(list(self.repo_path.glob(pattern)))
                except:
                    pass
            
            # Check if there's anything to commit
            if self.repo.index.diff("HEAD") or not list(self.repo.iter_commits()):
                self.repo.index.commit(f"Initial commit: {self.repo_path.name} project setup")
        except Exception as e:
            # Repo might already have commits, that's fine
            pass
    
    def _ensure_configured(self):
        """Ensure tracker is configured before use"""
        if not self._git_available:
            return False
        if not self._configured or not self.repo:
            return False
        return True
    
    def is_available(self) -> bool:
        """Check if git tracking is available and configured"""
        return self._git_available and self._configured and self.repo is not None
    
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
    ) -> Optional[str]:
        """
        Commit with detailed task context.
        
        Returns:
            Commit SHA if successful, None otherwise
        """
        if not self._ensure_configured():
            return None
        
        # Create comprehensive commit message
        files_section = '\n'.join(f'  - {f}' for f in files_modified) if files_modified else '  None'
        tools_section = '\n'.join(f'  - {t}' for t in tools_used) if tools_used else '  None'
        
        message = f"""[{status.upper()}] Task {task_id}: {description[:50]}

Agent: {agent_name}
Status: {status}
Timestamp: {datetime.now().isoformat()}
Project: {self.repo_path.name}

Files Modified:
{files_section}

Tools Used:
{tools_section}
"""
        
        if reasoning:
            message += f"\nReasoning:\n{reasoning[:500]}\n"
        
        if result:
            message += f"\nResult:\n{result[:500]}...\n"
        
        if error:
            message += f"\nError:\n{error}\n"
        
        # Stage and commit
        try:
            # Add all changes (respecting .gitignore)
            self.repo.git.add(A=True)
            
            # Check if there's anything to commit
            if not self.repo.index.diff("HEAD") and not self.repo.untracked_files:
                # Nothing to commit
                return None
            
            # Commit
            commit = self.repo.index.commit(message)
            
            # Tag failures for easy identification
            if status == "failure":
                tag_name = f"failure-{task_id}-{datetime.now():%Y%m%d_%H%M%S}"
                try:
                    self.repo.create_tag(tag_name, message=f"Failed attempt - Task {task_id}")
                except:
                    pass  # Tag might already exist
            
            return commit.hexsha
            
        except Exception as e:
            # Git operations failing shouldn't break the pipeline
            print(f"Git commit warning: {e}")
            return None
    
    def get_task_history(self, task_id: str) -> List[dict]:
        """Retrieve all commits related to a task"""
        if not self._ensure_configured():
            return []
            
        commits = []
        try:
            for commit in self.repo.iter_commits():
                if task_id in commit.message:
                    commits.append({
                        "sha": commit.hexsha,
                        "message": commit.message,
                        "timestamp": datetime.fromtimestamp(commit.committed_date),
                        "author": str(commit.author)
                    })
        except Exception:
            pass
        return commits
    
    def get_all_failures(self) -> List[dict]:
        """Get all failure tags for troubleshooting"""
        if not self._ensure_configured():
            return []
            
        failures = []
        try:
            for tag in self.repo.tags:
                if tag.name.startswith("failure-"):
                    failures.append({
                        "tag": tag.name,
                        "message": tag.tag.message if tag.tag else "",
                        "commit": tag.commit.hexsha,
                        "timestamp": datetime.fromtimestamp(tag.commit.committed_date)
                    })
        except Exception:
            pass
        return failures
    
    def get_recent_commits(self, n: int = 10) -> List[dict]:
        """Get recent commits for this project"""
        if not self._ensure_configured():
            return []
            
        commits = []
        try:
            for commit in list(self.repo.iter_commits())[:n]:
                commits.append({
                    "sha": commit.hexsha[:8],
                    "message": commit.message.split('\n')[0],
                    "timestamp": datetime.fromtimestamp(commit.committed_date).isoformat(),
                    "author": str(commit.author)
                })
        except Exception:
            pass
        return commits


# Global instance - will be configured by main.py
git_tracker = GitTracker()


def configure_git_tracking(project_path: str, auto_init: bool = True):
    """
    Convenience function to configure the global git tracker.
    
    Call this from main.py after project_dir is known:
        from utils.git_tracker import configure_git_tracking
        configure_git_tracking(project_dir)
    """
    git_tracker.configure(project_path, auto_init=auto_init)
