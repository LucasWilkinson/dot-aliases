"""Git operations for branch comparison."""

import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from .utils import note


class GitError(Exception):
    """Exception raised for git operation errors."""
    pass


class GitManager:
    """Manages git operations for vLLM testing."""
    
    def __init__(self, repo_dir: str, venv_path: Optional[str] = None):
        """Initialize GitManager.
        
        Args:
            repo_dir: Path to git repository.
            venv_path: Path to virtual environment for building.
        """
        self.repo_dir = Path(repo_dir).resolve()
        self.venv_path = venv_path
        
        if not (self.repo_dir / ".git").exists():
            raise GitError(f"Not a git repository: {self.repo_dir}")
    
    def _run_git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run a git command in the repo directory.
        
        Args:
            *args: Git command arguments.
            check: Whether to raise exception on non-zero exit.
        
        Returns:
            CompletedProcess instance.
        """
        cmd = ["git", "-C", str(self.repo_dir)] + list(args)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if check and result.returncode != 0:
            raise GitError(
                f"Git command failed: {' '.join(cmd)}\n"
                f"Error: {result.stderr}"
            )
        
        return result
    
    def fetch_tags(self) -> None:
        """Fetch tags from origin."""
        note("Fetching tags from origin...")
        self._run_git("fetch", "origin", "--tags")
    
    def checkout(self, ref: str) -> None:
        """Checkout a branch, tag, or commit.
        
        Args:
            ref: Git reference (branch, tag, or commit hash).
        
        Raises:
            GitError: If checkout fails.
        """
        note(f"Checking out '{ref}'...")
        self._run_git("checkout", ref)
        
        # Show current commit for verification
        result = self._run_git("rev-parse", "--short", "HEAD")
        commit_hash = result.stdout.strip()
        note(f"Now at commit {commit_hash}")
    
    def pull(self, ref: str, rebase: bool = True) -> bool:
        """Pull latest changes for a branch.
        
        Only pulls if ref looks like a branch name (not a commit hash).
        
        Args:
            ref: Git reference.
            rebase: Whether to use --rebase flag.
        
        Returns:
            True if pull was performed, False if skipped.
        """
        # Check if ref looks like a commit hash (7-40 hex characters)
        if re.match(r'^[0-9a-f]{7,40}$', ref):
            note(f"Ref '{ref}' looks like a commit hash, skipping pull")
            return False
        
        note(f"Pulling latest changes for '{ref}'...")
        try:
            if rebase:
                self._run_git("pull", "--rebase")
            else:
                self._run_git("pull")
            return True
        except GitError as e:
            note(f"Warning: Pull failed: {e}")
            return False
    
    def build(self, build_type: str = "build_ext") -> None:
        """Build vLLM extensions.
        
        Args:
            build_type: Build command type (e.g., 'build_ext').
        
        Raises:
            GitError: If build fails.
        """
        note(f"Building vLLM with: python setup.py {build_type} --inplace")
        
        # Get python from venv if available
        if self.venv_path:
            python_cmd = os.path.join(self.venv_path, "bin", "python")
        else:
            python_cmd = "python"
        
        # Run build in repo directory
        cmd = [python_cmd, "setup.py", build_type, "--inplace"]
        result = subprocess.run(
            cmd,
            cwd=self.repo_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise GitError(
                f"Build failed:\n"
                f"Command: {' '.join(cmd)}\n"
                f"Error: {result.stderr}"
            )
        
        note("Build completed successfully")
    
    def get_current_branch(self) -> str:
        """Get name of current branch.
        
        Returns:
            Current branch name or 'HEAD' if detached.
        """
        result = self._run_git("rev-parse", "--abbrev-ref", "HEAD")
        return result.stdout.strip()
    
    def get_current_commit(self) -> str:
        """Get current commit hash.
        
        Returns:
            Short commit hash.
        """
        result = self._run_git("rev-parse", "--short", "HEAD")
        return result.stdout.strip()
    
    def is_dirty(self) -> bool:
        """Check if working directory has uncommitted changes.
        
        Returns:
            True if there are uncommitted changes.
        """
        result = self._run_git("status", "--porcelain")
        return bool(result.stdout.strip())

