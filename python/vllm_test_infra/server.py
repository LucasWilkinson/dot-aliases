"""vLLM server management."""

import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from .logging import LogManager
from .process import ProcessManager
from .utils import (
    compute_gpu_count,
    extract_tp_dp_from_args,
    is_chg_available,
    note,
    split_args_string,
)


class VLLMServerError(Exception):
    """Exception raised for vLLM server errors."""
    pass


class VLLMServer:
    """Manages vLLM server lifecycle."""
    
    # Error patterns to detect in logs for fast-fail
    ERROR_PATTERNS = [
        r'AssertionError',
        r'OutOfMemoryError',
        r'CUDA out of memory',
        r'RuntimeError.*CUDA',
        r'Failed to allocate',
        r'torch\.cuda\.OutOfMemoryError',
        r'Cannot allocate memory',
        r'killed by signal',
        r'Segmentation fault',
        r'chg:.*No GPUs available',
        r'chg:.*GPU allocation failed',
    ]
    
    def __init__(self,
                 model: str,
                 host: str = "localhost",
                 port: int = 8000,
                 venv_path: Optional[str] = None,
                 log_manager: Optional[LogManager] = None,
                 process_manager: Optional[ProcessManager] = None):
        """Initialize VLLMServer.
        
        Args:
            model: Model name or path.
            host: Server host.
            port: Server port.
            venv_path: Path to virtual environment.
            log_manager: LogManager instance for logging.
            process_manager: ProcessManager instance for process control.
        """
        self.model = model
        self.host = host
        self.port = port
        self.venv_path = venv_path
        self.log_manager = log_manager
        self.process_manager = process_manager or ProcessManager(log_manager)
        
        self._server_process = None
        self._server_args = []
        self._extra_env = {}
    
    def _get_vllm_command(self) -> str:
        """Get path to vllm command."""
        if self.venv_path:
            vllm_cmd = os.path.join(self.venv_path, "bin", "vllm")
            if os.path.exists(vllm_cmd):
                return vllm_cmd
        
        # Fallback to system vllm
        return "vllm"
    
    def _build_command(self, args: str, env_csv: str = "") -> Tuple[List[str], Dict[str, str]]:
        """Build server command with GPU allocation.
        
        Args:
            args: Space-separated server arguments.
            env_csv: Comma-separated environment variables (K=V,K2=V2).
        
        Returns:
            Tuple of (command_list, env_dict).
        """
        # Parse environment variables
        env_dict = {}
        if env_csv:
            for kv in env_csv.split(','):
                kv = kv.strip()
                if '=' in kv:
                    key, value = kv.split('=', 1)
                    env_dict[key.strip()] = value.strip()
        
        # Extract GPU requirements from args
        tp_size, dp_size = extract_tp_dp_from_args(args)
        gpu_count = compute_gpu_count(tp_size, dp_size)
        
        # Build base command
        vllm_cmd = self._get_vllm_command()
        arg_list = split_args_string(args) if args else []
        
        command = [vllm_cmd, "serve", self.model, "--host", self.host, "--port", str(self.port)]
        command.extend(arg_list)
        
        # Add GPU launcher prefix if chg is available and GPUs are needed
        if gpu_count > 1 or (gpu_count == 1 and is_chg_available()):
            if is_chg_available():
                note(f"Using chg to reserve {gpu_count} GPU(s)")
                command = ["chg", "run", "-g", str(gpu_count), "--"] + command
            else:
                note(f"chg not available, launching without GPU reservation (need {gpu_count} GPUs)")
        
        return command, env_dict
    
    def start(self, 
              args: str = "",
              env_csv: str = "",
              log_file: str = "server") -> None:
        """Start the vLLM server.
        
        Args:
            args: Space-separated server arguments.
            env_csv: Comma-separated environment variables.
            log_file: Log file name for server output.
        
        Raises:
            VLLMServerError: If server is already running or fails to start.
        """
        if self._server_process and self.process_manager.is_running("vllm_server"):
            raise VLLMServerError("Server is already running")
        
        # Build command
        command, env_dict = self._build_command(args, env_csv)
        self._server_args = args
        self._extra_env = env_dict
        
        note(f"Starting vLLM server: {' '.join(command)}")
        if env_dict:
            note(f"Environment: {env_dict}")
        
        # Start server process
        self._server_process = self.process_manager.run_background(
            name="vllm_server",
            command=command,
            env=env_dict,
            log_file=log_file
        )
    
    def wait_for_ready(self, timeout: int = 600, check_interval: float = 1.0) -> bool:
        """Wait for server to be ready, monitoring logs for errors.
        
        Args:
            timeout: Maximum time to wait in seconds.
            check_interval: Time between checks in seconds.
        
        Returns:
            True if server is ready, False on timeout.
        
        Raises:
            VLLMServerError: If fatal error detected in logs or process died.
        """
        health_url = f"http://{self.host}:{self.port}/health"
        models_url = f"http://{self.host}:{self.port}/v1/models"
        start_time = time.time()
        last_log_check = 0
        
        note(f"Waiting for server at {health_url} (timeout: {timeout}s)...")
        
        while time.time() - start_time < timeout:
            # Check if process is still alive
            if not self.process_manager.is_running("vllm_server"):
                error_msg = "Server process died unexpectedly"
                if self.log_manager:
                    log_tail = self.log_manager.tail_file("server", num_lines=100)
                    error_msg += f"\n\nLast 100 lines of server log:\n{log_tail}"
                raise VLLMServerError(error_msg)
            
            # Check logs for errors periodically (every 2 seconds)
            current_time = time.time()
            if self.log_manager and current_time - last_log_check >= 2.0:
                error = self.log_manager.search_file_for_patterns(
                    "server",
                    self.ERROR_PATTERNS,
                    start_pos=0
                )
                if error:
                    pattern, line = error
                    log_tail = self.log_manager.tail_file("server", num_lines=100)
                    raise VLLMServerError(
                        f"Fatal error detected in server logs:\n"
                        f"  Pattern: {pattern}\n"
                        f"  Line: {line}\n\n"
                        f"Last 100 lines of server log:\n{log_tail}"
                    )
                last_log_check = current_time
            
            # Check health endpoint
            try:
                response = requests.get(health_url, timeout=2)
                if response.status_code == 200:
                    note("Server is ready!")
                    return True
            except requests.RequestException:
                pass
            
            # Also try models endpoint as fallback
            try:
                response = requests.get(models_url, timeout=2)
                if response.status_code == 200:
                    note("Server is ready!")
                    return True
            except requests.RequestException:
                pass
            
            time.sleep(check_interval)
        
        note(f"WARNING: Server did not become ready after {timeout}s")
        return False
    
    def stop(self) -> None:
        """Stop the vLLM server gracefully."""
        if not self._server_process:
            return
        
        self.process_manager.terminate("vllm_server")
        self._server_process = None
    
    def is_running(self) -> bool:
        """Check if server is running.
        
        Returns:
            True if server process is alive.
        """
        return self.process_manager.is_running("vllm_server")
    
    def restart(self, args: Optional[str] = None, env_csv: Optional[str] = None) -> None:
        """Restart the server with same or new configuration.
        
        Args:
            args: Optional new server arguments (uses previous if None).
            env_csv: Optional new environment variables (uses previous if None).
        """
        self.stop()
        time.sleep(2)  # Give port time to be released
        
        if args is None:
            args = self._server_args
        if env_csv is None:
            env_csv = ','.join(f"{k}={v}" for k, v in self._extra_env.items())
        
        self.start(args=args, env_csv=env_csv)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure server is stopped."""
        self.stop()
        return False

