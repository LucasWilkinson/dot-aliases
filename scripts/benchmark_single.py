#!/usr/bin/env python3
"""Standalone benchmark script for current branch."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.vllm_test_infra import BaseBenchmarkRunner, UIManager
from python.vllm_test_infra.ui import run_with_ui


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark the current branch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model deepseek-ai/DeepSeek-V3.2-Exp -tp 8
  %(prog)s --model meta-llama/Meta-Llama-3-8B-Instruct --rates 1,5,10
  %(prog)s --variants 'base::;full::-O.cudagraph_mode=FULL'
  %(prog)s --model deepseek-ai/DeepSeek-V2-Lite --build --max-model-len 8192
        """
    )
    
    # General options
    parser.add_argument("--venv", default=".venv", help="Virtualenv path")
    parser.add_argument("--host", default="localhost", help="Host")
    parser.add_argument("--port", type=int, default=3333, help="Port")
    
    # Model / Data
    parser.add_argument(
        "--model",
        required=True,
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
    parser.add_argument("--rates", default="1,5,10", help="Request rates CSV")
    parser.add_argument("--run-seconds", type=int, default=120, help="Seconds per rate")
    
    # Git / Build
    parser.add_argument("--branch", help="Checkout this branch/ref (default: current branch)")
    parser.add_argument("--build", action="store_true", help="Run setup.py build")
    parser.add_argument("--repo-dir", default=".", help="Git repository directory")
    parser.add_argument("--pull-latest", action="store_true", help="Pull latest changes")
    parser.add_argument("--resume", action="store_true", help="Resume and only run missing combos")
    
    # Output / Paths
    parser.add_argument("--out-base", default="./results", help="Base output directory")
    parser.add_argument("--results-dir", help="Results directory (default: out-base/bench_<model>_<dataset>)")
    parser.add_argument("--log-dir", help="Log directory (default: out-base/logs)")
    parser.add_argument("--label-suffix", default="", help="Suffix for result filenames")
    
    # Server Args
    parser.add_argument("--server-args-base", default="", help="Args added to server")
    parser.add_argument("--server-args", default="", help="Alias for --server-args-base")
    
    # Variants
    parser.add_argument("--variants", help="Variant spec (e.g., 'base::;full::-O.cudagraph_mode=FULL')")
    
    # UI
    parser.add_argument(
        "--ui-mode",
        choices=["tui", "simple", "auto"],
        default="simple",
        help="UI mode (default: simple for agents)"
    )
    
    return parser.parse_args()


class SingleBenchmarkRunner(BaseBenchmarkRunner):
    """Manages benchmark execution for a single branch."""
    
    def __init__(self, args):
        """Initialize SingleBenchmarkRunner.
        
        Args:
            args: Parsed command line arguments.
        """
        # Handle server-args alias
        if args.server_args and not args.server_args_base:
            args.server_args_base = args.server_args
        
        # Call parent constructor
        super().__init__(args)
        
        # Parse variants
        self.variants = self._parse_variant_spec(
            args.variants,
            ""  # No default args, all in server_args_base
        )
        
        # Determine branch
        self.branch = args.branch or self._get_current_branch()
    
    def _get_current_branch(self) -> str:
        """Get current git branch."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.args.repo_dir,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            self.logger.warning("Could not determine current git branch, using 'current'")
            return "current"
    
    def _setup_directories(self):
        """Setup output directories for single branch mode."""
        self.out_base = Path(self.args.out_base).resolve()
        self.results_dir = Path(
            self.args.results_dir or self.out_base / f"bench_{self.terse_model_name}_{self.args.dataset}"
        ).resolve()
        self.log_dir = Path(
            self.args.log_dir or self.out_base / "logs"
        ).resolve()
        
        # Create directories
        self.out_base.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self) -> int:
        """Run the benchmark.
        
        Returns:
            Exit code.
        """
        # Log configuration info
        self.logger.info("=" * 60)
        self.logger.info("Benchmark (Single Branch)")
        self.logger.info("=" * 60)
        self.logger.info(f"Model: {self.args.model}")
        self.logger.info(f"Branch: {self.branch}")
        self.logger.info(f"Dataset: {self.args.dataset}")
        self.logger.info(f"Rates: {self.args.rates}")
        self.logger.info(f"Repository: {self.args.repo_dir}")
        self.logger.info(f"Results: {self.results_dir}")
        self.logger.info(f"Logs: {self.log_dir}")
        self.logger.info("=" * 60)
        
        try:
            # Run the branch
            self.run_branch(
                "BENCH",
                self.branch,
                self.variants,
                self.results_dir,
                self.args.build,
                self.args.pull_latest,
                force_rerun=False
            )
            
            self.logger.info("=" * 60)
            self.logger.info("Benchmark completed!")
            self.logger.info(f"Results: {self.results_dir}")
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
    runner = SingleBenchmarkRunner(args)
    
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
        print("✅ Benchmark completed!")
        print(f"📁 Results: {runner.results_dir}")
    else:
        print("❌ Benchmark failed!")
    
    print(f"📋 Logs: {runner.log_dir}")
    print("=" * 60)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
