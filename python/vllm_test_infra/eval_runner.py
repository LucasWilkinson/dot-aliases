"""Shared evaluation runner utilities for vLLM evaluations."""

import glob
from pathlib import Path
from typing import List, Optional

from .logging import LogManager
from .process import ProcessManager
from .server import VLLMServer


class EvalRunner:
    """Base class for running evaluations with vLLM server."""
    
    def __init__(
        self,
        model: str,
        task_name: str,
        host: str,
        port: int,
        venv_path: str,
        log_manager: LogManager,
        process_manager: ProcessManager,
        out_base: Path,
    ):
        """Initialize EvalRunner.
        
        Args:
            model: Model to evaluate.
            task_name: Task name for lm_eval (e.g., "gsm8k", "gpqa_diamond").
            host: Server host.
            port: Server port.
            venv_path: Path to virtual environment.
            log_manager: LogManager instance.
            process_manager: ProcessManager instance.
            out_base: Base output directory.
        """
        self.model = model
        self.task_name = task_name
        self.host = host
        self.port = port
        self.venv_path = venv_path
        self.log_manager = log_manager
        self.process_manager = process_manager
        self.out_base = out_base
        self.logger = log_manager.get_logger(f"{task_name}_eval", log_to_file="script")
    
    def run_evaluation(
        self,
        server_args: str = "",
        limit: Optional[int] = None,
        num_concurrent: int = 256,
        batch_size: str = "auto",
        timeout: int = 3600,
    ) -> int:
        """Run evaluation with vLLM server.
        
        Args:
            server_args: Additional server arguments.
            limit: Limit number of test cases.
            num_concurrent: Number of concurrent requests.
            batch_size: Batch size for lm_eval.
            timeout: Evaluation timeout in seconds.
        
        Returns:
            Exit code (0 for success, non-zero for failure).
        """
        try:
            # Log server command
            server_cmd = f"vllm serve {self.model} --host {self.host} --port {self.port}"
            if server_args:
                server_cmd += f" {server_args}"
            self.logger.info("=" * 60)
            self.logger.info("Server Command (for manual reproduction):")
            self.logger.info(f"  {server_cmd}")
            self.logger.info("=" * 60)
            
            # Create and start server
            self.logger.info("Starting vLLM server...")
            server = VLLMServer(
                model=self.model,
                host=self.host,
                port=self.port,
                venv_path=self.venv_path,
                log_manager=self.log_manager,
                process_manager=self.process_manager
            )
            
            with server:
                server.start(args=server_args, log_file="server")
                
                # Wait for server to be ready
                if not server.wait_for_ready(timeout=600):
                    self.logger.error("Server failed to start within timeout")
                    self.logger.error(f"Check server log: {self.log_manager.get_log_path('server')}")
                    self.logger.error("")
                    self.logger.error("=" * 60)
                    self.logger.error("Last 200 lines of server log:")
                    self.logger.error("=" * 60)
                    server_log_tail = self.log_manager.tail_file('server', num_lines=200)
                    self.logger.error(server_log_tail)
                    self.logger.error("=" * 60)
                    return 1
                
                # Run evaluation
                self.logger.info("")
                self.logger.info("=" * 50)
                self.logger.info(f"Running {self.task_name} evaluation...")
                self.logger.info("=" * 50)
                self.logger.info("")
                
                # Build lm_eval command
                eval_cmd = self._build_eval_command(
                    limit=limit,
                    num_concurrent=num_concurrent,
                    batch_size=batch_size
                )
                
                self.logger.info("=" * 60)
                self.logger.info("Evaluation Command (for manual reproduction):")
                self.logger.info(f"  {' '.join(eval_cmd)}")
                self.logger.info("=" * 60)
                self.logger.info("")
                
                # Run evaluation
                try:
                    result = self.process_manager.run(
                        name="lm_eval",
                        command=eval_cmd,
                        log_file="eval",
                        timeout=timeout
                    )
                    
                    if result.returncode == 0:
                        self.logger.info("")
                        self.logger.info("=" * 50)
                        self.logger.info("Evaluation complete!")
                        self.logger.info("=" * 50)
                        self.logger.info(f"Results: {self.out_base}")
                        self.logger.info(f"Server log: {self.log_manager.get_log_path('server')}")
                        self.logger.info(f"Eval log: {self.log_manager.get_log_path('eval')}")
                        return 0
                    else:
                        self.logger.error("")
                        self.logger.error("=" * 50)
                        self.logger.error(f"Evaluation failed with status: {result.returncode}")
                        self.logger.error("=" * 50)
                        return result.returncode
                
                except Exception as e:
                    self.logger.error(f"Evaluation error: {e}")
                    return 1
        
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
            return 130
        
        except Exception as e:
            self.logger.error(f"Error: {e}", exc_info=True)
            return 1
    
    def _build_eval_command(
        self,
        limit: Optional[int],
        num_concurrent: int,
        batch_size: str
    ) -> List[str]:
        """Build lm_eval command.
        
        Args:
            limit: Limit number of test cases.
            num_concurrent: Number of concurrent requests.
            batch_size: Batch size for lm_eval.
        
        Returns:
            Command list.
        """
        eval_cmd = [
            "lm_eval",
            "--model", "local-completions",
            "--model_args", (
                f"model={self.model},"
                f"base_url=http://{self.host}:{self.port}/v1/completions,"
                f"num_concurrent={num_concurrent}"
            ),
            "--tasks", self.task_name,
            "--output_path", str(self.out_base),
            "--log_samples",
            "--batch_size", batch_size,
        ]
        
        if limit:
            eval_cmd.extend(["--limit", str(limit)])
        
        return eval_cmd
    
    def find_results(self) -> Optional[Path]:
        """Find most recent results file.
        
        Returns:
            Path to results file, or None if not found.
        """
        # Try finding results files in subdirectories first
        result_files = sorted(glob.glob(str(self.out_base / "*" / "results_*.json")), reverse=True)
        
        # Also try legacy format
        result_files.extend(sorted(glob.glob(str(self.out_base / "results_*" / "results.json")), reverse=True))
        
        if result_files:
            return Path(result_files[0])
        
        return None

