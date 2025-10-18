"""Logging management for vLLM test infrastructure."""

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, TextIO

from .utils import timestamp


class LogManager:
    """Manages log files and logging configuration."""
    
    def __init__(self, log_dir: str):
        """Initialize LogManager.
        
        Args:
            log_dir: Directory to store log files.
        """
        self.log_dir = Path(log_dir)
        self.log_files = {}
        self._loggers = {}
    
    def setup(self) -> None:
        """Create log directories and initialize log files."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def get_log_path(self, name: str) -> Path:
        """Get path for a named log file.
        
        Args:
            name: Log file name (without .log extension).
        
        Returns:
            Path to log file.
        """
        return self.log_dir / f"{name}.log"
    
    def init_log_file(self, name: str) -> Path:
        """Initialize a log file (truncate if exists).
        
        Args:
            name: Log file name.
        
        Returns:
            Path to initialized log file.
        """
        log_path = self.get_log_path(name)
        log_path.write_text("")  # Truncate
        self.log_files[name] = log_path
        return log_path
    
    def get_logger(self, name: str, log_to_file: Optional[str] = None) -> logging.Logger:
        """Get or create a logger.
        
        Args:
            name: Logger name.
            log_to_file: Optional log file name to write to.
        
        Returns:
            Configured logger instance.
        """
        if name in self._loggers:
            return self._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()  # Remove any existing handlers
        
        # Console handler with timestamp
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            fmt='[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # Optional file handler
        if log_to_file:
            log_path = self.get_log_path(log_to_file)
            file_handler = logging.FileHandler(log_path, mode='a')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)
        
        logger.propagate = False
        self._loggers[name] = logger
        return logger
    
    @contextmanager
    def redirect_to_file(self, name: str, mode: str = 'a'):
        """Context manager to redirect stdout/stderr to a log file.
        
        Args:
            name: Log file name.
            mode: File open mode ('a' for append, 'w' for write).
        
        Yields:
            File handle for the log file.
        """
        log_path = self.get_log_path(name)
        
        # Save original stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        try:
            with open(log_path, mode) as f:
                sys.stdout = f
                sys.stderr = f
                yield f
        finally:
            # Restore original stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    def tail_file(self, name: str, num_lines: int = 50) -> str:
        """Get the last N lines of a log file.
        
        Args:
            name: Log file name.
            num_lines: Number of lines to return.
        
        Returns:
            Last N lines of the file as a string.
        """
        log_path = self.get_log_path(name)
        
        if not log_path.exists():
            return ""
        
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
                return ''.join(lines[-num_lines:])
        except Exception:
            return ""
    
    def follow_file(self, name: str, start_pos: int = 0):
        """Generator that yields new lines from a log file as they're written.
        
        Args:
            name: Log file name.
            start_pos: Starting file position.
        
        Yields:
            New lines as they appear in the file.
        """
        log_path = self.get_log_path(name)
        
        # Wait for file to exist
        while not log_path.exists():
            import time
            time.sleep(0.1)
        
        with open(log_path, 'r') as f:
            f.seek(start_pos)
            while True:
                line = f.readline()
                if line:
                    yield line
                else:
                    # No new data, sleep briefly
                    import time
                    time.sleep(0.1)
    
    def search_file_for_patterns(self, name: str, patterns: list[str], 
                                  start_pos: int = 0) -> Optional[tuple[str, str]]:
        """Search log file for error patterns.
        
        Args:
            name: Log file name.
            patterns: List of regex patterns to search for.
            start_pos: Starting file position.
        
        Returns:
            Tuple of (pattern, matching_line) if found, None otherwise.
        """
        import re
        
        log_path = self.get_log_path(name)
        
        if not log_path.exists():
            return None
        
        compiled_patterns = [(p, re.compile(p, re.IGNORECASE)) for p in patterns]
        
        try:
            with open(log_path, 'r') as f:
                f.seek(start_pos)
                for line in f:
                    for pattern_str, pattern_re in compiled_patterns:
                        if pattern_re.search(line):
                            return (pattern_str, line.strip())
        except Exception:
            pass
        
        return None

