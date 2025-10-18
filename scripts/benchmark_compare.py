#!/usr/bin/env python3
"""Benchmark comparison script for vLLM across branches."""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.vllm_test_infra import (
    Config,
    GitManager,
    LogManager,
    ProcessManager,
    VLLMServer,
    UIManager,
    note,
    parse_variants,
    split_args_string,
)
from python.vllm_test_infra.ui import run_with_ui


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run benchmark comparison across vLLM branches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model meta-llama/Meta-Llama-3-8B-Instruct --rates 1,5,10
  %(prog)s --variants 'base::;fullcg::-O {"full_cuda_graph":true}'
  %(prog)s --which pr --build-pr 1 --pr-ref my-feature-branch
        """
    )
    
    # General options
    parser.add_argument("--venv", default=".venv", help="Virtualenv path")
    parser.add_argument("--host", default="localhost", help="Host")
    parser.add_argument("--port", type=int, default=3333, help="Port")
    
    # Model / Data
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model (HF id or local path)"
    )
    parser.add_argument(
        "--terse-name",
        help="Short name for filenames (default: derived from model)"
    )
    parser.add_argument("--dataset", default="random", help="Dataset name")
    parser.add_argument("--random-in", type=int, default=1000, help="Random input len")
    parser.add_argument("--random-out", type=int, default=100, help="Random output len")
    parser.add_argument("--random-range-ratio", type=float, default=0, help="Random range ratio")
    
    # Parallelism
    parser.add_argument("-tp", "--tensor-parallel-size", type=int, default=1, help="Tensor-parallel size")
    
    # Rates / Durations
    parser.add_argument("--rates", default="1,5,10,25,100", help="Request rates CSV")
    parser.add_argument("--run-seconds", type=int, default=120, help="Seconds per rate")
    
    # Git / Build
    parser.add_argument("--main-ref", default="main", help="Checkout this ref for MAIN")
    parser.add_argument("--pr-ref", help="Checkout this ref for PR")
    parser.add_argument("--pr-branch", default="full_cudagraph_FA2_FlashInfer", help="PR branch fallback")
    parser.add_argument("--build-main", type=int, default=0, help="Run setup.py build on MAIN")
    parser.add_argument("--build-pr", type=int, default=0, help="Run setup.py build on PR")
    parser.add_argument("--repo-dir", default=".", help="Git repository directory")
    parser.add_argument("--pull-latest", action="store_true", help="Pull latest for MAIN and PR")
    parser.add_argument("--pull-latest-main", action="store_true", help="Pull latest for MAIN")
    parser.add_argument("--pull-latest-pr", action="store_true", help="Pull latest for PR")
    parser.add_argument("--resume", action="store_true", help="Resume and only run missing combos")
    parser.add_argument("--re-run-pr", action="store_true", help="With --resume, re-run all PR variants")
    parser.add_argument("--re-run-main", action="store_true", help="With --resume, re-run all MAIN variants")
    parser.add_argument(
        "--which",
        choices=["both", "main", "pr"],
        default="both",
        help="Which side(s) to run"
    )
    
    # Output / Paths
    parser.add_argument("--out-base", default="./results", help="Base output directory")
    parser.add_argument("--results-main", help="Results dir for MAIN")
    parser.add_argument("--results-pr", help="Results dir for PR")
    parser.add_argument("--log-dir", help="Log directory")
    parser.add_argument("--label-suffix", default="", help="Suffix for result filenames")
    
    # Server Args
    parser.add_argument("--server-args-base", default="", help="Args added to EVERY server run")
    parser.add_argument("--server-args-main", default="", help="Extra args for MAIN server")
    parser.add_argument("--server-args-pr", default="", help="Extra args for PR server")
    
    # Variants
    parser.add_argument("--variants", help="Variant spec for both MAIN and PR")
    parser.add_argument("--variants-main", help="Variant spec for MAIN only")
    parser.add_argument("--variants-pr", help="Variant spec for PR only")
    
    # UI
    parser.add_argument(
        "--ui-mode",
        choices=["tui", "simple", "auto"],
        default="auto",
        help="UI mode"
    )
    
    return parser.parse_args()


class BenchmarkRunner:
    """Manages benchmark execution across branches and variants."""
    
    # Always-on server args
    ALWAYS_SERVER_ARGS = [
        "--no-enable-prefix-caching",
        "--disable-log-stats",
        "--trust-remote-code",
    ]
    
    def __init__(self, args):
        """Initialize BenchmarkRunner.
        
        Args:
            args: Parsed command line arguments.
        """
        self.args = args
        
        # Setup configuration
        self.config = Config(venv_path=args.venv)
        self.config.activate_venv()
        
        # Derive terse model name
        if args.terse_name:
            self.terse_model_name = args.terse_name
        else:
            self.terse_model_name = Path(args.model).name.replace("/", "_")
        
        # Setup output directories
        self.out_base = Path(args.out_base).resolve()
        self.results_dir_main = Path(
            args.results_main or self.out_base / f"bench_main_{self.terse_model_name}_{args.dataset}"
        ).resolve()
        self.results_dir_pr = Path(
            args.results_pr or self.out_base / f"bench_pr_{self.terse_model_name}_{args.dataset}"
        ).resolve()
        self.log_dir = Path(
            args.log_dir or self.out_base / "logs"
        ).resolve()
        
        # Create directories
        self.out_base.mkdir(parents=True, exist_ok=True)
        self.results_dir_main.mkdir(parents=True, exist_ok=True)
        self.results_dir_pr.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse rates
        self.rates = [float(r.strip()) for r in args.rates.split(',')]
        
        # Setup logging
        self.log_manager = LogManager(str(self.log_dir))
        self.log_manager.setup()
        self.log_manager.init_log_file("server")
        self.log_manager.init_log_file("bench")
        self.log_manager.init_log_file("script")
        self.log_manager.init_log_file("summary_current")
        
        self.logger = self.log_manager.get_logger("benchmark", log_to_file="script")
        
        # Setup process manager
        self.process_manager = ProcessManager(self.log_manager)
        
        # Setup git manager
        self.git_manager = GitManager(args.repo_dir, venv_path=args.venv)
        
        # Parse variants
        self.variants_main = self._parse_variant_spec(
            args.variants_main or args.variants,
            args.server_args_main
        )
        self.variants_pr = self._parse_variant_spec(
            args.variants_pr or args.variants,
            args.server_args_pr
        )
        
        # Build base server args
        self.server_args_base = self._build_base_server_args()
    
    def _parse_variant_spec(self, spec: Optional[str], default_args: str) -> List[Tuple[str, str, str]]:
        """Parse variant specification.
        
        Args:
            spec: Variant specification string.
            default_args: Default args if no spec provided.
        
        Returns:
            List of (label, args, env_csv) tuples.
        """
        if spec:
            return parse_variants(spec)
        else:
            # No variants specified, use single "base" variant with default args
            return [("base", default_args, "")]
    
    def _build_base_server_args(self) -> str:
        """Build base server arguments."""
        base_args = split_args_string(self.args.server_args_base)
        
        # Add -tp if specified and not already in args
        if self.args.tensor_parallel_size > 1:
            if not any('-tp' in arg or '--tensor-parallel-size' in arg for arg in base_args):
                base_args.extend(["-tp", str(self.args.tensor_parallel_size)])
        
        # Add always-on args
        base_args.extend(self.ALWAYS_SERVER_ARGS)
        
        return ' '.join(base_args)
    
    def _get_result_filename(self, branch: str, variant: str, rate: float) -> str:
        """Generate result filename.
        
        Args:
            branch: Branch label (MAIN or PR).
            variant: Variant label.
            rate: Request rate.
        
        Returns:
            Result filename.
        """
        num_prompts = int(rate * self.args.run_seconds)
        ds_safe = re.sub(r'[^A-Za-z0-9._-]+', '-', self.args.dataset)
        
        suffix = f"_{self.args.label_suffix}" if self.args.label_suffix else ""
        
        return (
            f"bench_model-{self.terse_model_name}_rate-{rate}_v-{variant}_"
            f"np-{num_prompts}_in-{self.args.random_in}_out-{self.args.random_out}_"
            f"ds-{ds_safe}{suffix}.json"
        )
    
    def _result_exists(self, results_dir: Path, branch: str, variant: str, rate: float) -> bool:
        """Check if result file exists and is non-empty.
        
        Args:
            results_dir: Results directory.
            branch: Branch label.
            variant: Variant label.
            rate: Request rate.
        
        Returns:
            True if result exists.
        """
        filename = self._get_result_filename(branch, variant, rate)
        filepath = results_dir / filename
        return filepath.exists() and filepath.stat().st_size > 0
    
    def run_client_for_rate(
        self,
        results_dir: Path,
        branch: str,
        variant: str,
        rate: float
    ) -> bool:
        """Run benchmark client for a specific rate.
        
        Args:
            results_dir: Directory to save results.
            branch: Branch label.
            variant: Variant label.
            rate: Request rate.
        
        Returns:
            True if successful.
        """
        num_prompts = int(rate * self.args.run_seconds)
        filename = self._get_result_filename(branch, variant, rate)
        filepath = results_dir / filename
        
        # Check if we should skip
        if self._result_exists(results_dir, branch, variant, rate):
            should_rerun = (
                (branch == "PR" and self.args.re_run_pr) or
                (branch == "MAIN" and self.args.re_run_main)
            )
            if not should_rerun:
                self.logger.info(f"Reusing existing result for {branch}/{variant} rate={rate}")
                return True
        
        self.logger.info(f"Running client: {branch}/{variant} rate={rate} prompts={num_prompts}")
        
        # Build vllm bench command
        vllm_cmd = os.path.join(self.args.venv, "bin", "vllm")
        
        command = [
            vllm_cmd, "bench", "serve",
            "--model", self.args.model,
            "--host", self.args.host,
            "--port", str(self.args.port),
            "--dataset-name", self.args.dataset,
            "--random-input-len", str(self.args.random_in),
            "--random-output-len", str(self.args.random_out),
            "--random-range-ratio", str(self.args.random_range_ratio),
            "--num-prompts", str(num_prompts),
            "--request-rate", str(rate),
            "--save-result",
            "--result-dir", str(results_dir),
            "--result-filename", filename,
            "--seed", "42",
            "--ignore-eos",
            "--trust-remote-code",
        ]
        
        try:
            result = self.process_manager.run(
                name=f"bench_{branch}_{variant}_{rate}",
                command=command,
                log_file="bench",
                timeout=self.args.run_seconds * 3  # 3x safety margin
            )
            
            if result.returncode == 0:
                self.logger.info(f"Completed {branch}/{variant} rate={rate}")
                return True
            else:
                self.logger.error(f"Benchmark failed for {branch}/{variant} rate={rate}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error running benchmark: {e}")
            return False
    
    def run_variant(
        self,
        results_dir: Path,
        branch: str,
        variant_label: str,
        variant_args: str,
        variant_env: str
    ) -> None:
        """Run benchmarks for a single variant.
        
        Args:
            results_dir: Results directory.
            branch: Branch label.
            variant_label: Variant label.
            variant_args: Variant-specific server args.
            variant_env: Variant-specific environment variables.
        """
        self.logger.info(f"Starting variant: {branch}/{variant_label}")
        
        # Check if all results already exist and we're not re-running
        all_exist = all(
            self._result_exists(results_dir, branch, variant_label, rate)
            for rate in self.rates
        )
        
        should_rerun = (
            (branch == "PR" and self.args.re_run_pr) or
            (branch == "MAIN" and self.args.re_run_main)
        )
        
        if all_exist and self.args.resume and not should_rerun:
            self.logger.info(f"Skipping {branch}/{variant_label} (all results present)")
            return
        
        # Build full server args
        full_args = f"{self.server_args_base} {variant_args}".strip()
        
        # Start server
        server = VLLMServer(
            model=self.args.model,
            host=self.args.host,
            port=self.args.port,
            venv_path=self.args.venv,
            log_manager=self.log_manager,
            process_manager=self.process_manager
        )
        
        with server:
            self.logger.info(f"Starting server for {branch}/{variant_label}")
            server.start(args=full_args, env_csv=variant_env, log_file="server")
            
            if not server.wait_for_ready(timeout=600):
                self.logger.error(f"Server failed to start for {branch}/{variant_label}")
                return
            
            # Run benchmarks for each rate
            for rate in self.rates:
                self.run_client_for_rate(results_dir, branch, variant_label, rate)
        
        self.logger.info(f"Completed variant: {branch}/{variant_label}")
    
    def run_branch(self, branch: str, ref: str, variants: List[Tuple[str, str, str]], 
                   results_dir: Path, build: bool, pull: bool) -> None:
        """Run benchmarks for a branch.
        
        Args:
            branch: Branch label (MAIN or PR).
            ref: Git reference.
            variants: List of (label, args, env) tuples.
            results_dir: Results directory.
            build: Whether to rebuild after checkout.
            pull: Whether to pull latest changes.
        """
        self.logger.info(f"=" * 60)
        self.logger.info(f"Running {branch} branch: {ref}")
        self.logger.info(f"=" * 60)
        
        # Checkout branch
        self.git_manager.checkout(ref)
        
        # Pull if requested
        if pull:
            self.git_manager.pull(ref)
        
        # Build if requested
        if build:
            self.git_manager.build()
        
        # Run each variant
        for variant_label, variant_args, variant_env in variants:
            self.run_variant(results_dir, branch, variant_label, variant_args, variant_env)
    
    def run(self) -> int:
        """Run the benchmark comparison.
        
        Returns:
            Exit code.
        """
        # Log configuration info (will appear in TUI script pane or simple mode stdout)
        self.logger.info("=" * 60)
        self.logger.info("Benchmark Comparison")
        self.logger.info("=" * 60)
        self.logger.info(f"Model: {self.args.model}")
        self.logger.info(f"Dataset: {self.args.dataset}")
        self.logger.info(f"Rates: {self.args.rates}")
        self.logger.info(f"Repository: {self.args.repo_dir}")
        self.logger.info(f"Results MAIN: {self.results_dir_main}")
        self.logger.info(f"Results PR: {self.results_dir_pr}")
        self.logger.info(f"Logs: {self.log_dir}")
        self.logger.info("=" * 60)
        
        try:
            # Run MAIN branch
            if self.args.which in ["both", "main"]:
                self.run_branch(
                    "MAIN",
                    self.args.main_ref,
                    self.variants_main,
                    self.results_dir_main,
                    bool(self.args.build_main),
                    self.args.pull_latest or self.args.pull_latest_main
                )
            
            # Run PR branch
            if self.args.which in ["both", "pr"]:
                pr_ref = self.args.pr_ref or self.args.pr_branch
                self.run_branch(
                    "PR",
                    pr_ref,
                    self.variants_pr,
                    self.results_dir_pr,
                    bool(self.args.build_pr),
                    self.args.pull_latest or self.args.pull_latest_pr
                )
            
            self.logger.info("=" * 60)
            self.logger.info("Benchmark comparison completed!")
            self.logger.info(f"Results MAIN: {self.results_dir_main}")
            self.logger.info(f"Results PR: {self.results_dir_pr}")
            self.logger.info(f"Logs: {self.log_dir}")
            self.logger.info("=" * 60)
            
            return 0
        
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
            return 130
        
        except Exception as e:
            self.logger.error(f"Error: {e}", exc_info=True)
            return 1


def main():
    """Main entry point."""
    args = parse_args()
    runner = BenchmarkRunner(args)
    
    # Setup UI manager
    ui_manager = UIManager(runner.log_manager, mode=args.ui_mode)
    
    # Create work function
    def work_func():
        return runner.run()
    
    # Run with appropriate UI mode
    panes = {
        "server-pane": "server",
        "bench-pane": "bench",
        "script-pane": "script",
    }
    
    exit_code = run_with_ui(ui_manager, work_func, panes)
    
    # Print summary after UI closes
    print()
    print("=" * 60)
    if exit_code == 0:
        print("✅ Benchmark comparison completed!")
        print(f"📁 Results MAIN: {runner.results_dir_main}")
        print(f"📁 Results PR: {runner.results_dir_pr}")
    else:
        print("❌ Benchmark comparison failed!")
    
    print(f"📋 Logs: {runner.log_dir}")
    print("=" * 60)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

