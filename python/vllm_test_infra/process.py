"""Process management for vLLM test infrastructure."""

import atexit
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

from .logging import LogManager
from .utils import cleanup_zombie_processes, note


class ProcessManager:
    """Manages subprocess execution and monitoring."""
    
    def __init__(self, log_manager: Optional[LogManager] = None):
        """Initialize ProcessManager.
        
        Args:
            log_manager: Optional LogManager for process output.
        """
        self.log_manager = log_manager
        self.processes: Dict[str, subprocess.Popen] = {}
        self._cleanup_registered = False
        
        # Register cleanup on exit
        if not self._cleanup_registered:
            atexit.register(self.cleanup_on_exit)
            self._cleanup_registered = True
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals."""
        note(f"Received signal {signum}, cleaning up...")
        self.terminate_all()
        # Re-raise to allow normal signal handling
        if signum == signal.SIGINT:
            raise KeyboardInterrupt()
    
    def run(self, 
            name: str,
            command: List[str],
            env: Optional[Dict[str, str]] = None,
            cwd: Optional[str] = None,
            timeout: Optional[int] = None,
            log_file: Optional[str] = None,
            capture_output: bool = False) -> subprocess.CompletedProcess:
        """Run a command and wait for completion.
        
        Args:
            name: Process name for tracking.
            command: Command and arguments to execute.
            env: Optional environment variables (merged with current env).
            cwd: Optional working directory.
            timeout: Optional timeout in seconds.
            log_file: Optional log file name to write output to.
            capture_output: If True, capture stdout/stderr to return object.
        
        Returns:
            CompletedProcess instance with results.
        
        Raises:
            subprocess.TimeoutExpired: If timeout is exceeded.
            subprocess.CalledProcessError: If command returns non-zero exit code.
        """
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        
        note(f"Running {name}: {' '.join(command)}")
        
        # Determine output handling
        stdout_dest = subprocess.PIPE if capture_output else None
        stderr_dest = subprocess.PIPE if capture_output else None
        
        if log_file and self.log_manager:
            log_path = self.log_manager.get_log_path(log_file)
            stdout_dest = open(log_path, 'a')
            stderr_dest = subprocess.STDOUT
        
        try:
            result = subprocess.run(
                command,
                env=full_env,
                cwd=cwd,
                timeout=timeout,
                stdout=stdout_dest,
                stderr=stderr_dest,
                text=True
            )
            
            if result.returncode != 0:
                note(f"{name} exited with code {result.returncode}")
            else:
                note(f"{name} completed successfully")
            
            return result
        
        finally:
            # Close log file if we opened it
            if log_file and self.log_manager and not capture_output:
                if stdout_dest and hasattr(stdout_dest, 'close'):
                    stdout_dest.close()
    
    def run_background(self,
                       name: str,
                       command: List[str],
                       env: Optional[Dict[str, str]] = None,
                       cwd: Optional[str] = None,
                       log_file: Optional[str] = None) -> subprocess.Popen:
        """Start a background process.
        
        Args:
            name: Process name for tracking.
            command: Command and arguments to execute.
            env: Optional environment variables (merged with current env).
            cwd: Optional working directory.
            log_file: Optional log file name to write output to.
        
        Returns:
            Popen instance for the running process.
        """
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        
        note(f"Starting background process {name}: {' '.join(command)}")
        
        # Determine output handling
        stdout_dest = subprocess.DEVNULL
        stderr_dest = subprocess.DEVNULL
        
        if log_file and self.log_manager:
            log_path = self.log_manager.get_log_path(log_file)
            stdout_dest = open(log_path, 'a')
            stderr_dest = subprocess.STDOUT
        
        # Use setsid to create new process group for better cleanup
        process = subprocess.Popen(
            command,
            env=full_env,
            cwd=cwd,
            stdout=stdout_dest,
            stderr=stderr_dest,
            start_new_session=True,  # Create new process group
            text=True
        )
        
        self.processes[name] = process
        note(f"Started {name} with PID {process.pid}")
        
        return process
    
    def is_running(self, name: str) -> bool:
        """Check if a named process is still running.
        
        Args:
            name: Process name.
        
        Returns:
            True if process exists and is running.
        """
        if name not in self.processes:
            return False
        
        process = self.processes[name]
        return process.poll() is None
    
    def terminate(self, name: str, timeout: int = 10) -> bool:
        """Terminate a named process gracefully.
        
        Args:
            name: Process name.
            timeout: Timeout in seconds for graceful shutdown.
        
        Returns:
            True if process was terminated, False if not found or already dead.
        """
        if name not in self.processes:
            return False
        
        process = self.processes[name]
        
        if process.poll() is not None:
            # Already dead
            del self.processes[name]
            return False
        
        note(f"Terminating {name} (PID {process.pid})...")
        
        # Try graceful shutdown first (SIGINT)
        try:
            # Send to process group
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            pass
        
        # Wait briefly for graceful exit
        try:
            process.wait(timeout=5)
            note(f"{name} terminated gracefully")
            del self.processes[name]
            return True
        except subprocess.TimeoutExpired:
            pass
        
        # Try SIGTERM
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=3)
            note(f"{name} terminated with SIGTERM")
            del self.processes[name]
            return True
        except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError):
            pass
        
        # Force kill with SIGKILL
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait(timeout=2)
            note(f"{name} killed with SIGKILL")
        except (ProcessLookupError, PermissionError):
            pass
        
        del self.processes[name]
        return True
    
    def terminate_all(self) -> None:
        """Terminate all managed processes."""
        if not self.processes:
            return
        
        note("Terminating all processes...")
        
        # Make a copy of process names to avoid dict modification during iteration
        process_names = list(self.processes.keys())
        
        for name in process_names:
            self.terminate(name)
        
        # Give processes time to die
        time.sleep(1)
    
    def cleanup_on_exit(self) -> None:
        """Cleanup function registered with atexit."""
        note("Running process cleanup...")
        
        # Terminate all managed processes
        self.terminate_all()
        
        # Clean up zombie processes
        cleanup_zombie_processes()
    
    def get_process(self, name: str) -> Optional[subprocess.Popen]:
        """Get process handle by name.
        
        Args:
            name: Process name.
        
        Returns:
            Popen instance if found, None otherwise.
        """
        return self.processes.get(name)

