"""Configuration and environment management."""

import os
from pathlib import Path
from typing import Dict, Optional


class Config:
    """Manages configuration and environment variables."""
    
    def __init__(self, venv_path: Optional[str] = None):
        """Initialize Config.
        
        Args:
            venv_path: Path to virtual environment.
        """
        self.venv_path = Path(venv_path) if venv_path else None
        self._env_overrides: Dict[str, str] = {}
    
    def set_env(self, key: str, value: str) -> None:
        """Set an environment variable override.
        
        Args:
            key: Environment variable name.
            value: Environment variable value.
        """
        self._env_overrides[key] = value
    
    def get_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable value.
        
        Args:
            key: Environment variable name.
            default: Default value if not found.
        
        Returns:
            Environment variable value or default.
        """
        return self._env_overrides.get(key, os.environ.get(key, default))
    
    def get_full_env(self) -> Dict[str, str]:
        """Get full environment with overrides applied.
        
        Returns:
            Dictionary of environment variables.
        """
        env = os.environ.copy()
        env.update(self._env_overrides)
        return env
    
    def activate_venv(self) -> None:
        """Activate virtual environment by updating PATH and env vars."""
        if not self.venv_path:
            return
        
        if not self.venv_path.exists():
            raise ValueError(f"Virtual environment not found: {self.venv_path}")
        
        # Update PATH to include venv bin directory
        venv_bin = self.venv_path / "bin"
        current_path = os.environ.get("PATH", "")
        
        # Add venv bin to front of PATH if not already there
        venv_bin_str = str(venv_bin)
        if venv_bin_str not in current_path:
            os.environ["PATH"] = f"{venv_bin_str}:{current_path}"
            self._env_overrides["PATH"] = os.environ["PATH"]
        
        # Set VIRTUAL_ENV
        os.environ["VIRTUAL_ENV"] = str(self.venv_path)
        self._env_overrides["VIRTUAL_ENV"] = str(self.venv_path)
    
    @staticmethod
    def normalize_path(path: str) -> Path:
        """Normalize and resolve a path to absolute form.
        
        Args:
            path: Path string.
        
        Returns:
            Resolved absolute Path.
        """
        return Path(path).expanduser().resolve()
    
    @staticmethod
    def ensure_dir(path: str) -> Path:
        """Ensure directory exists, creating if necessary.
        
        Args:
            path: Directory path.
        
        Returns:
            Path object for the directory.
        """
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

